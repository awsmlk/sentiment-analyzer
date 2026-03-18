#include "sentiment.h"
#include "ui.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <cctype>
#include <unordered_set>

using namespace std;

map<string, map<string, int>> wordCounts;
map<string, int> labelCounts;
map<string, int> totalWords;
map<string, double> priors;

unordered_set<string> vocabulary;
vector<HistoryRow> history;

// ----------------- Helpers -----------------
string lower(string s)
{
    transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

string removePunctuation(string s)
{
    s.erase(remove_if(s.begin(), s.end(),
                      [](unsigned char c) { return ispunct(c); }),
            s.end());
    return s;
}

string normalize(string s)
{
    if (s == "im") return "i";
    if (s == "ive") return "i";
    if (s == "dont") return "not";
    if (s == "cant") return "not";
    return s;
}

// ----------------- Tokenizer with Negation + Bigrams -----------------
vector<string> tokenize(string text)
{
    vector<string> words;
    string prev = "";
    string word;
    stringstream ss(text);

    while(ss >> word)
    {
        word = lower(word);
        word = normalize(word);
        word = removePunctuation(word);
        if(word.empty()) continue;

        // Handle negation "not X"
        if(prev == "not")
        {
            words.pop_back();          // remove the "not" token
            word = "not_" + word;      // combine
        }

        // Add bigram with previous word if exists
        if(!prev.empty())
            words.push_back(prev + "_" + word);

        words.push_back(word);

        prev = word; // Update prev AFTER processing
    }

    return words;
}
// ----------------- Training -----------------
void trainModel(const string &filename)
{
    wordCounts.clear();
    labelCounts.clear();
    totalWords.clear();
    priors.clear();
    vocabulary.clear();

    ifstream file(filename);
    if (!file.is_open())
    {
        cout << "Error: Could not open dataset file\n";
        return;
    }

    string line;
    getline(file, line); // skip header

    while (getline(file, line))
    {
        string text, label;
        stringstream ss(line);
        getline(ss, text, ',');
        getline(ss, label, ',');
        label = lower(label);

        labelCounts[label]++;
        auto tokens = tokenize(text);

        for (auto &w : tokens)
        {
            wordCounts[label][w]++;
            totalWords[label]++;
            vocabulary.insert(w);
        }
    }

    int docs = 0;
    for (auto &p : labelCounts) docs += p.second;
    for (auto &p : labelCounts) priors[p.first] = (double)p.second / docs;
}

// ----------------- Internal Prediction -----------------
pair<string,double> predictInternal(const string &text, bool logHistory)
{
    if (priors.empty()) return {"Model not trained", 0};

    auto words = tokenize(text);

    string bestLabel;
    double bestScore = -1e9;
    map<string,double> logScores;

    for(auto &p : priors)
    {
        string label = p.first;
        double score = log(p.second);

        for(auto &w : words)
        {
            int count = wordCounts[label][w];
            double prob = (count + 1.0) / (totalWords[label] + max((size_t)1, vocabulary.size()));
            score += log(prob);
        }

        logScores[label] = score;
        if(score > bestScore)
        {
            bestScore = score;
            bestLabel = label;
        }
    }

    // Log-sum-exp trick
    double maxScore = bestScore;
    double sumExp = 0;
    for(auto &p : logScores) sumExp += exp(p.second - maxScore);

    double confidence = 0;
    if(sumExp > 0) confidence = (1.0 / sumExp) * 100;

    // -------- Neutral fallback --------
    const double NEUTRAL_THRESHOLD = 40.0; // confidence below this → neutral
    if(confidence < NEUTRAL_THRESHOLD)
    {
        bestLabel = "neutral";
        confidence = 100.0; // or leave confidence as-is
    }

    if(logHistory)
    {
        time_t now = time(0);
        string t = ctime(&now);
        t.pop_back();

        HistoryRow row;
        row.time = t;
        row.label = bestLabel;
        row.score = confidence;
        row.text = text;

        history.push_back(row);
    }

    return {bestLabel, confidence};
}

// ----------------- Public Prediction -----------------
pair<string,double> predict(const string &text)
{
    return predictInternal(text, true);
}

// ----------------- History -----------------
vector<HistoryRow> &getHistory() { return history; }

void exportHistory(const string &path)
{
    ofstream file(path);
    file << "Time,Label,Confidence,Text\n";
    for (auto &h : history)
    {
        file << h.time << "," << h.label << "," << h.score << "," << h.text << "\n";
    }
}

// ----------------- Accuracy Testing -----------------
double testAccuracy(const string &filename)
{
    ifstream file(filename);
    if (!file.is_open()) return 0;

    string line;
    getline(file, line); // skip header
    int correct = 0, total = 0;

    while (getline(file, line))
    {
        string text, label;
        stringstream ss(line);
        getline(ss, text, ',');
        getline(ss, label, ',');
        label = lower(label);

        auto result = predictInternal(text, false);
        if(result.first == label) correct++;
        total++;
    }

    if(total == 0) return 0;
    return (double)correct / total * 100;
}