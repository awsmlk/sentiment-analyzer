// server.cpp — HTTP bridge: web frontend ↔ sentiment engine + Ollama proxy
//
// Compile:  g++ -std=c++17 -O2 -Iinclude -o server src/server.cpp src/sentiment.cpp -pthread
// Run:      ./server
//
// Endpoints:
//   GET  /ping              health check
//   GET  /accuracy          model accuracy on dataset.csv
//   GET  /dataset-info      number of rows in dataset.csv
//   GET  /model-info        accuracy + dataset rows + vocab size
//   POST /predict           {"text":"..."} → {label, confidence, keywords, isUnknown, oovRatio}
//   GET  /history           all predictions made so far
//   POST /teach             {"text":"...","label":"..."} → appends to dataset.csv
//   POST /retrain           re-trains model, returns new accuracy
//   POST /ollama-proxy      {"model":"...","prompt":"...","ollamaUrl":"..."} → streams Ollama response
//
// All responses include CORS headers for localhost frontend access.

#include "sentiment.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cstring>
#include <thread>
#include <mutex>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>

using namespace std;

static mutex trainMutex;

// ─────────────────────────────────────────────────────────────────────────────
// JSON helpers
// ─────────────────────────────────────────────────────────────────────────────
static string jStr(const string &s)
{
    string o = "\"";
    for (char c : s)
    {
        if      (c == '"')  o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else                o += c;
    }
    return o + "\"";
}

static string jBool(bool b) { return b ? "true" : "false"; }

static string jDouble(double d, int prec = 2)
{
    ostringstream ss;
    ss.precision(prec);
    ss << fixed << d;
    return ss.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP response
// ─────────────────────────────────────────────────────────────────────────────
static string httpResp(int code, const string &body,
                        const string &ct = "application/json")
{
    string status = (code == 200) ? "200 OK"
                  : (code == 400) ? "400 Bad Request"
                  : "500 Internal Server Error";
    ostringstream r;
    r << "HTTP/1.1 " << status << "\r\n"
      << "Content-Type: " << ct << "; charset=utf-8\r\n"
      << "Access-Control-Allow-Origin: *\r\n"
      << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
      << "Access-Control-Allow-Headers: Content-Type\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "Connection: close\r\n\r\n"
      << body;
    return r.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON field extraction
// ─────────────────────────────────────────────────────────────────────────────
static string jsonField(const string &json, const string &key)
{
    string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == string::npos) return "";
    pos = json.find(':', pos + needle.size());
    if (pos == string::npos) return "";
    pos++;
    while (pos < json.size() && json[pos] == ' ') pos++;
    if (pos >= json.size()) return "";

    if (json[pos] == '"')
    {
        pos++;
        string val;
        while (pos < json.size() && json[pos] != '"')
        {
            if (json[pos] == '\\' && pos + 1 < json.size()) { pos++; }
            val += json[pos++];
        }
        return val;
    }
    auto end = json.find_first_of(",}\n", pos);
    string val = json.substr(pos, end == string::npos ? string::npos : end - pos);
    while (!val.empty() && (val.back() == ' ' || val.back() == '\r')) val.pop_back();
    return val;
}

// ─────────────────────────────────────────────────────────────────────────────
// CSV utilities
// ─────────────────────────────────────────────────────────────────────────────
static int csvRowCount(const string &path)
{
    ifstream f(path);
    if (!f.is_open()) return 0;
    string line;
    int count = -1; // skip header
    while (getline(f, line)) count++;
    return max(0, count);
}

static bool appendToDataset(const string &path,
                             const string &text,
                             const string &label)
{
    string safe = text;
    for (char &c : safe) if (c == '\n' || c == '\r') c = ' ';
    bool needsQuote = safe.find(',') != string::npos ||
                      safe.find('"') != string::npos;
    // Escape inner quotes
    string escaped;
    for (char c : safe) { if (c == '"') escaped += "\"\""; else escaped += c; }
    string row = (needsQuote ? "\"" + escaped + "\"" : escaped) + "," + label + "\n";

    ofstream f(path, ios::app);
    if (!f.is_open()) return false;
    f << row;
    return true;
}

static bool hasChunkedEncoding(const string &headers)
{
    string lower = headers;
    for (char &c : lower)
        c = static_cast<char>(tolower(static_cast<unsigned char>(c)));

    return lower.find("transfer-encoding: chunked") != string::npos;
}

static string decodeChunkedBody(const string &body)
{
    string decoded;
    size_t pos = 0;

    while (pos < body.size())
    {
        size_t lineEnd = body.find("\r\n", pos);
        if (lineEnd == string::npos) break;

        string sizeText = body.substr(pos, lineEnd - pos);
        size_t semicolon = sizeText.find(';');
        if (semicolon != string::npos)
            sizeText = sizeText.substr(0, semicolon);

        size_t chunkSize = 0;
        stringstream ss;
        ss << hex << sizeText;
        ss >> chunkSize;

        pos = lineEnd + 2;
        if (chunkSize == 0) break;
        if (pos + chunkSize > body.size()) break;

        decoded.append(body, pos, chunkSize);
        pos += chunkSize + 2; // skip chunk data plus trailing CRLF
    }

    return decoded;
}

// ─────────────────────────────────────────────────────────────────────────────
// Ollama proxy — forwards POST to local Ollama, returns full body
// Reads ollamaUrl from request body field "ollamaUrl" (default localhost:11434)
// ─────────────────────────────────────────────────────────────────────────────
static string ollamaRequest(const string &ollamaHost, int ollamaPort,
                              const string &path, const string &body)
{
    // TCP connect
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    string portStr = to_string(ollamaPort);
    if (getaddrinfo(ollamaHost.c_str(), portStr.c_str(), &hints, &res) != 0)
        return "";

    int sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sock < 0) { freeaddrinfo(res); return ""; }

    if (connect(sock, res->ai_addr, res->ai_addrlen) < 0)
    {
        freeaddrinfo(res); close(sock); return "";
    }
    freeaddrinfo(res);

    // Send HTTP request
    ostringstream req;
    req << "POST " << path << " HTTP/1.1\r\n"
        << "Host: " << ollamaHost << ":" << ollamaPort << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    string reqStr = req.str();
    send(sock, reqStr.c_str(), reqStr.size(), 0);

    // Read response
    string response;
    char buf[4096];
    ssize_t n;
    while ((n = read(sock, buf, sizeof(buf))) > 0)
        response.append(buf, n);
    close(sock);

    // Strip HTTP headers. Ollama may use normal Content-Length responses or
    // chunked responses, so handle both before returning JSON to the browser.
    auto sep = response.find("\r\n\r\n");
    if (sep != string::npos)
    {
        string headers = response.substr(0, sep);
        string bodyOnly = response.substr(sep + 4);
        if (hasChunkedEncoding(headers))
            return decodeChunkedBody(bodyOnly);
        return bodyOnly;
    }
    return response;
}

// Parse "http://host:port" into host + port
static pair<string,int> parseUrl(const string &url)
{
    string h = url;
    if (h.substr(0, 7) == "http://")  h = h.substr(7);
    if (h.substr(0, 8) == "https://") h = h.substr(8);
    // strip trailing slash
    if (!h.empty() && h.back() == '/') h.pop_back();

    auto colon = h.rfind(':');
    if (colon != string::npos)
    {
        string portStr = h.substr(colon + 1);
        try { return {h.substr(0, colon), stoi(portStr)}; }
        catch(...) {}
    }
    return {h, 11434};
}

// ─────────────────────────────────────────────────────────────────────────────
// Build keywords JSON array from PredictResult
// ─────────────────────────────────────────────────────────────────────────────
static string keywordsJson(const vector<KeywordScore> &kws)
{
    ostringstream o;
    o << "[";
    for (size_t i = 0; i < kws.size(); i++)
    {
        if (i) o << ",";
        o << "{"
          << "\"word\":"     << jStr(kws[i].word)     << ","
          << "\"polarity\":" << jStr(kws[i].polarity) << ","
          << "\"score\":"    << jDouble(kws[i].score)
          << "}";
    }
    o << "]";
    return o.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// Handle one client
// ─────────────────────────────────────────────────────────────────────────────
static void handleClient(int fd)
{
    // Read full request (allow up to 256 KB for large batch bodies)
    string req;
    {
        char buf[4096];
        ssize_t n;
        while ((n = read(fd, buf, sizeof(buf))) > 0)
        {
            req.append(buf, n);
            // Stop reading when we have the full body (Content-Length met)
            auto headerEnd = req.find("\r\n\r\n");
            if (headerEnd != string::npos)
            {
                auto clPos = req.find("Content-Length:");
                if (clPos == string::npos) break;
                clPos += 15;
                while (clPos < req.size() && req[clPos] == ' ') clPos++;
                auto clEnd = req.find("\r\n", clPos);
                int cl = 0;
                try { cl = stoi(req.substr(clPos, clEnd - clPos)); } catch(...) {}
                int bodyLen = (int)req.size() - (int)(headerEnd + 4);
                if (bodyLen >= cl) break;
            }
        }
    }
    if (req.empty()) { close(fd); return; }

    istringstream hs(req);
    string method, path;
    hs >> method >> path;

    // ── CORS pre-flight ───────────────────────────────────────────────────
    if (method == "OPTIONS")
    {
        string r = httpResp(200, "");
        write(fd, r.c_str(), r.size());
        close(fd); return;
    }

    // ── GET /ping ─────────────────────────────────────────────────────────
    if (method == "GET" && path == "/ping")
    {
        string r = httpResp(200, R"({"status":"ok"})");
        write(fd, r.c_str(), r.size());
        close(fd); return;
    }

    // ── GET /accuracy ─────────────────────────────────────────────────────
    if (method == "GET" && path == "/accuracy")
    {
        double acc = testAccuracy("data/dataset.csv");
        string body = "{\"accuracy\":" + jDouble(acc) + "}";
        string r = httpResp(200, body);
        write(fd, r.c_str(), r.size());
        close(fd); return;
    }

    // ── GET /dataset-info ─────────────────────────────────────────────────
    if (method == "GET" && path == "/dataset-info")
    {
        int rows = csvRowCount("data/dataset.csv");
        string body = "{\"rows\":" + to_string(rows) + "}";
        string r = httpResp(200, body);
        write(fd, r.c_str(), r.size());
        close(fd); return;
    }

    // ── GET /model-info ───────────────────────────────────────────────────
    if (method == "GET" && path == "/model-info")
    {
        double acc  = testAccuracy("data/dataset.csv");
        int    rows = csvRowCount("data/dataset.csv");
        size_t voc  = getVocabSize();
        ostringstream body;
        body << "{"
             << "\"accuracy\":"   << jDouble(acc)        << ","
             << "\"datasetRows\":" << rows               << ","
             << "\"vocabSize\":"   << voc
             << "}";
        string r = httpResp(200, body.str());
        write(fd, r.c_str(), r.size());
        close(fd); return;
    }

    // ── POST /predict ─────────────────────────────────────────────────────
    if (method == "POST" && path == "/predict")
    {
        auto sep = req.find("\r\n\r\n");
        string body = (sep != string::npos) ? req.substr(sep + 4) : "";

        string text = jsonField(body, "text");
        if (text.empty())
        {
            string r = httpResp(400, R"({"error":"missing text field"})");
            write(fd, r.c_str(), r.size());
            close(fd); return;
        }

        PredictResult res = predictFull(text);

        ostringstream out;
        out << "{"
            << "\"label\":"      << jStr(res.label)             << ","
            << "\"confidence\":" << jDouble(res.confidence)     << ","
            << "\"isUnknown\":"  << jBool(res.isUnknown)        << ","
            << "\"oovRatio\":"   << jDouble(res.oovRatio, 3)    << ","
            << "\"keywords\":"   << keywordsJson(res.keywords)
            << "}";

        string r = httpResp(200, out.str());
        write(fd, r.c_str(), r.size());
        close(fd); return;
    }

    // ── GET /history ──────────────────────────────────────────────────────
    if (method == "GET" && path == "/history")
    {
        auto &hist = getHistory();
        ostringstream out;
        out << "[";
        for (size_t i = 0; i < hist.size(); i++)
        {
            if (i) out << ",";
            out << "{"
                << "\"time\":"       << jStr(hist[i].time)    << ","
                << "\"label\":"      << jStr(hist[i].label)   << ","
                << "\"confidence\":" << hist[i].score         << ","
                << "\"text\":"       << jStr(hist[i].text)
                << "}";
        }
        out << "]";
        string r = httpResp(200, out.str());
        write(fd, r.c_str(), r.size());
        close(fd); return;
    }

    // ── POST /teach ───────────────────────────────────────────────────────
    if (method == "POST" && path == "/teach")
    {
        auto sep = req.find("\r\n\r\n");
        string body = (sep != string::npos) ? req.substr(sep + 4) : "";

        string text  = jsonField(body, "text");
        string label = jsonField(body, "label");

        if (text.empty() || label.empty())
        {
            string r = httpResp(400, R"({"error":"missing text or label"})");
            write(fd, r.c_str(), r.size());
            close(fd); return;
        }

        if (label != "positive" && label != "neutral" && label != "negative")
        {
            string r = httpResp(400, R"({"error":"label must be positive, neutral, or negative"})");
            write(fd, r.c_str(), r.size());
            close(fd); return;
        }

        bool ok = appendToDataset("data/dataset.csv", text, label);
        if (!ok)
        {
            cerr << "[teach] Failed to write to dataset.csv\n";
            string r = httpResp(500, R"({"error":"could not write to dataset.csv"})");
            write(fd, r.c_str(), r.size());
            close(fd); return;
        }

        cout << "[teach] Added: \"" << text.substr(0, 60) << "\" → " << label << "\n";

        string r = httpResp(200, R"({"status":"saved"})");
        write(fd, r.c_str(), r.size());
        close(fd); return;
    }

    // ── POST /retrain ─────────────────────────────────────────────────────
    if (method == "POST" && path == "/retrain")
    {
        lock_guard<mutex> lock(trainMutex);
        cout << "[retrain] Retraining model from dataset.csv…\n";
        trainModel("data/dataset.csv");
        double acc = testAccuracy("data/dataset.csv");
        cout << "[retrain] Done — accuracy: " << acc << "%\n";
        ostringstream body;
        body << "{\"status\":\"ok\",\"accuracy\":" << jDouble(acc) << ","
             << "\"vocabSize\":" << getVocabSize() << "}";
        string r = httpResp(200, body.str());
        write(fd, r.c_str(), r.size());
        close(fd); return;
    }

    // ── POST /ollama-proxy ────────────────────────────────────────────────
    // Body: {"model":"llama3","prompt":"...","ollamaUrl":"http://localhost:11434"}
    // Forwards to Ollama /api/generate and returns raw response JSON.
    // This avoids CORS issues when the browser can't reach Ollama directly.
    if (method == "POST" && path == "/ollama-proxy")
    {
        auto sep = req.find("\r\n\r\n");
        string body = (sep != string::npos) ? req.substr(sep + 4) : "";

        string model     = jsonField(body, "model");
        string prompt    = jsonField(body, "prompt");
        string ollamaUrl = jsonField(body, "ollamaUrl");

        if (model.empty())  model     = "llama3";
        if (ollamaUrl.empty()) ollamaUrl = "http://localhost:11434";

        if (prompt.empty())
        {
            string r = httpResp(400, R"({"error":"missing prompt field"})");
            write(fd, r.c_str(), r.size());
            close(fd); return;
        }

        auto [ollamaHost, ollamaPort] = parseUrl(ollamaUrl);

        // Build forwarded body for Ollama
        // We set stream:false so we get a single JSON response
        ostringstream fwdBody;
        fwdBody << "{"
                << "\"model\":" << jStr(model) << ","
                << "\"prompt\":" << jStr(prompt) << ","
                << "\"stream\":false,"
                << "\"options\":{\"temperature\":0.1}"
                << "}";

        cout << "[ollama-proxy] → " << ollamaHost << ":" << ollamaPort
             << " model=" << model << "\n";

        string ollamaResp = ollamaRequest(ollamaHost, ollamaPort,
                                           "/api/generate", fwdBody.str());

        if (ollamaResp.empty())
        {
            string r = httpResp(500, R"({"error":"Could not reach Ollama — is it running?"})");
            write(fd, r.c_str(), r.size());
            close(fd); return;
        }

        // Pass Ollama's JSON response directly back to the browser
        string r = httpResp(200, ollamaResp);
        write(fd, r.c_str(), r.size());
        close(fd); return;
    }

    // ── 404 ───────────────────────────────────────────────────────────────
    string r = httpResp(400, R"({"error":"endpoint not found"})");
    write(fd, r.c_str(), r.size());
    close(fd);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    const string dataset = "data/dataset.csv";
    const int    port    = 8080;

    cout << "╔══════════════════════════════════════╗\n";
    cout << "║   Sentiment Analyzer Server  v2      ║\n";
    cout << "╚══════════════════════════════════════╝\n\n";

    cout << "Training on " << dataset << "…\n";
    trainModel(dataset);

    double acc = testAccuracy(dataset);
    cout << "Model ready | Accuracy: " << acc << "% | Dataset rows: "
         << csvRowCount(dataset) << " | Vocab: " << getVocabSize() << "\n\n";

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return 1; }

    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);

    if (::bind(srv, (sockaddr*)&addr, sizeof(addr)) < 0)
    {
        perror("bind"); return 1;
    }
    if (listen(srv, 64) < 0) { perror("listen"); return 1; }

    cout << "Server → http://localhost:" << port << "\n\n";
    cout << "Endpoints:\n";
    cout << "  GET  /ping           health check\n";
    cout << "  GET  /accuracy       model accuracy\n";
    cout << "  GET  /dataset-info   row count\n";
    cout << "  GET  /model-info     accuracy + rows + vocab size\n";
    cout << "  POST /predict        {text} → {label, confidence, isUnknown, oovRatio, keywords[]}\n";
    cout << "  POST /teach          {text, label} → appends to dataset.csv\n";
    cout << "  POST /retrain        retrain from updated dataset\n";
    cout << "  GET  /history        all predictions\n";
    cout << "  POST /ollama-proxy   {model, prompt, ollamaUrl} → Ollama response\n\n";

    while (true)
    {
        int client = accept(srv, nullptr, nullptr);
        if (client < 0) continue;
        thread(handleClient, client).detach();
    }
}
