#include "sentiment_system.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

/*
    The class declarations live in separate .h files so each part of the
    sentiment engine is easier to find. The implementations stay here so the
    build command can remain simple:

        g++ -std=c++17 -O2 -o server server.cpp sentiment.cpp -pthread
*/

TrainingExample::TrainingExample() {}

TrainingExample::TrainingExample(const string &textValue, const string &labelValue)
{
    text = textValue;
    label = labelValue;
}

TrainingExample CsvLineReader::parseLine(const string &line) const
{
    string text;
    string label;

    if (line.empty()) return TrainingExample();

    // If the text starts with a quote, it may contain commas.
    if (line[0] == '"')
    {
        size_t closeQuote = line.find('"', 1);
        if (closeQuote != string::npos)
        {
            text = line.substr(1, closeQuote - 1);
            size_t labelStart = closeQuote + 2;
            if (labelStart < line.size())
                label = line.substr(labelStart);
        }
    }
    else
    {
        stringstream ss(line);
        getline(ss, text, ',');
        getline(ss, label, ',');
    }

    return TrainingExample(text, label);
}

string LabelHelper::clean(string label) const
{
    transform(label.begin(), label.end(), label.begin(),
              [](unsigned char c) { return static_cast<char>(tolower(c)); });

    while (!label.empty() && (label.back() == '\r' || label.back() == ' '))
        label.pop_back();

    while (!label.empty() && label.front() == ' ')
        label.erase(label.begin());

    return label;
}

bool LabelHelper::isValid(const string &label) const
{
    return label == "positive" || label == "neutral" || label == "negative";
}

string WordCleaner::toLower(string word) const
{
    transform(word.begin(), word.end(), word.begin(),
              [](unsigned char c) { return static_cast<char>(tolower(c)); });
    return word;
}

string WordCleaner::removePunctuation(string word) const
{
    word.erase(remove_if(word.begin(), word.end(),
                         [](unsigned char c) { return ispunct(c); }),
               word.end());
    return word;
}

string WordCleaner::normalize(const string &word) const
{
    if (word == "im") return "i";
    if (word == "ive") return "i";
    if (word == "dont") return "not";
    if (word == "cant") return "not";
    if (word == "isnt") return "not";
    if (word == "wasnt") return "not";
    if (word == "wont") return "not";
    return word;
}

string WordCleaner::clean(string word) const
{
    word = toLower(word);
    word = removePunctuation(word);
    word = normalize(word);
    return word;
}

vector<string> Tokenizer::tokenize(const string &text) const
{
    vector<string> words;
    string previousWord;
    string currentWord;
    stringstream ss(text);

    while (ss >> currentWord)
    {
        currentWord = cleaner.clean(currentWord);
        if (currentWord.empty()) continue;

        // Convert "not good" into one stronger token: "not_good".
        if (previousWord == "not")
        {
            words.pop_back();
            currentWord = "not_" + currentWord;
        }

        // Also store simple word pairs, like "very_good".
        if (!previousWord.empty() && previousWord != "not")
            words.push_back(previousWord + "_" + currentWord);

        words.push_back(currentWord);
        previousWord = currentWord;
    }

    return words;
}

void SentimentModel::clear()
{
    wordCounts.clear();
    labelCounts.clear();
    totalWords.clear();
    priors.clear();
    vocabulary.clear();
}

void SentimentModel::learnFromTokens(const string &label, const vector<string> &tokens)
{
    labelCounts[label]++;

    for (const string &word : tokens)
    {
        wordCounts[label][word]++;
        totalWords[label]++;
        vocabulary.insert(word);
    }
}

void SentimentModel::calculatePriors()
{
    int totalDocuments = 0;
    for (const auto &item : labelCounts)
        totalDocuments += item.second;

    if (totalDocuments == 0) return;

    for (const auto &item : labelCounts)
        priors[item.first] = static_cast<double>(item.second) / totalDocuments;
}

bool SentimentModel::isTrained() const
{
    return !priors.empty();
}

size_t SentimentModel::vocabSize() const
{
    return vocabulary.size();
}

int SentimentModel::countWord(const string &label, const string &word) const
{
    auto labelIt = wordCounts.find(label);
    if (labelIt == wordCounts.end()) return 0;

    auto wordIt = labelIt->second.find(word);
    if (wordIt == labelIt->second.end()) return 0;

    return wordIt->second;
}

int SentimentModel::totalForLabel(const string &label) const
{
    auto it = totalWords.find(label);
    return (it == totalWords.end()) ? 0 : it->second;
}

TrainingReport ModelTrainer::train(SentimentModel &model, const string &filename) const
{
    TrainingReport report;
    model.clear();

    ifstream file(filename);
    if (!file.is_open())
    {
        cout << "Error: Could not open dataset file: " << filename << "\n";
        return report;
    }

    string line;
    getline(file, line); // skip CSV header

    while (getline(file, line))
    {
        TrainingExample example = csvReader.parseLine(line);
        example.label = labelHelper.clean(example.label);

        if (!labelHelper.isValid(example.label)) continue;

        vector<string> tokens = tokenizer.tokenize(example.text);
        model.learnFromTokens(example.label, tokens);
        report.rowsUsed++;
    }

    model.calculatePriors();
    return report;
}

double OovChecker::calculateRatio(const SentimentModel &model, const vector<string> &words) const
{
    int unknownWords = 0;
    int normalWords = 0;

    for (const string &word : words)
    {
        // Words with "_" are generated by our tokenizer, so skip them here.
        if (word.find('_') != string::npos) continue;

        normalWords++;
        if (model.vocabulary.find(word) == model.vocabulary.end())
            unknownWords++;
    }

    if (normalWords == 0) return 0.0;
    return static_cast<double>(unknownWords) / normalWords;
}

void ScoreBoard::addScore(const string &label, double score)
{
    logScores[label] = score;
}

string ScoreBoard::bestLabel() const
{
    string best = "neutral";
    double bestScore = -1e18;

    for (const auto &item : logScores)
    {
        if (item.second > bestScore)
        {
            bestScore = item.second;
            best = item.first;
        }
    }

    return best;
}

double ScoreBoard::bestScore() const
{
    double best = -1e18;
    for (const auto &item : logScores)
        best = max(best, item.second);
    return best;
}

double ScoreBoard::confidence() const
{
    double best = bestScore();
    double sum = 0.0;

    for (const auto &item : logScores)
        sum += exp(item.second - best);

    if (sum <= 0.0) return 50.0;
    return (1.0 / sum) * 100.0;
}

ScoreBoard BayesPredictor::scoreText(const SentimentModel &model, const vector<string> &words) const
{
    ScoreBoard scores;
    size_t vocabSize = max(static_cast<size_t>(1), model.vocabSize());

    for (const auto &prior : model.priors)
    {
        string label = prior.first;
        double score = log(prior.second);

        for (const string &word : words)
        {
            int count = model.countWord(label, word);
            double total = static_cast<double>(model.totalForLabel(label)) + vocabSize;
            double probability = (count + 1.0) / total;
            score += log(probability);
        }

        scores.addScore(label, score);
    }

    return scores;
}

vector<KeywordScore> KeywordFinder::findKeywords(const SentimentModel &model,
                                                 const vector<string> &words) const
{
    vector<KeywordScore> keywords;
    map<string, double> polarityScores;
    size_t vocabSize = max(static_cast<size_t>(1), model.vocabSize());

    double positiveTotal = static_cast<double>(model.totalForLabel("positive")) + vocabSize;
    double negativeTotal = static_cast<double>(model.totalForLabel("negative")) + vocabSize;

    for (const string &word : words)
    {
        if (word.find('_') != string::npos) continue;

        double positiveScore = log((model.countWord("positive", word) + 1.0) / positiveTotal);
        double negativeScore = log((model.countWord("negative", word) + 1.0) / negativeTotal);
        polarityScores[word] = positiveScore - negativeScore;
    }

    vector<pair<string, double>> sorted(polarityScores.begin(), polarityScores.end());
    sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
        return abs(a.second) > abs(b.second);
    });

    for (size_t i = 0; i < min(static_cast<size_t>(6), sorted.size()); i++)
    {
        KeywordScore keyword;
        keyword.word = sorted[i].first;
        keyword.score = sorted[i].second;
        keyword.polarity = (keyword.score > 0.5) ? "pos"
                         : (keyword.score < -0.5) ? "neg"
                         : "neu";
        keywords.push_back(keyword);
    }

    return keywords;
}

void HistoryManager::add(const string &text, const string &label, double confidence)
{
    time_t now = time(nullptr);
    string timestamp = ctime(&now);
    if (!timestamp.empty()) timestamp.pop_back();

    HistoryRow row;
    row.time = timestamp;
    row.label = label;
    row.score = confidence;
    row.text = text;
    rows.push_back(row);
}

vector<HistoryRow>& HistoryManager::getRows()
{
    return rows;
}

double AccuracyTester::test(const string &filename, SentimentSystem &system) const
{
    ifstream file(filename);
    if (!file.is_open()) return 0.0;

    string line;
    getline(file, line); // skip CSV header

    int correct = 0;
    int total = 0;

    while (getline(file, line))
    {
        TrainingExample example = csvReader.parseLine(line);
        example.label = labelHelper.clean(example.label);

        if (!labelHelper.isValid(example.label)) continue;

        PredictResult result = system.predictFull(example.text, false);
        if (result.label == example.label)
            correct++;

        total++;
    }

    if (total == 0) return 0.0;
    return static_cast<double>(correct) / total * 100.0;
}

void HistoryExporter::exportToCsv(const string &path, const vector<HistoryRow> &rows) const
{
    ofstream file(path);
    file << "Time,Label,Confidence,Text\n";

    for (const HistoryRow &row : rows)
        file << row.time << "," << row.label << "," << row.score << "," << row.text << "\n";
}

void SentimentSystem::train(const string &filename)
{
    TrainingReport report = trainer.train(model, filename);
    cout << "[sentiment] Trained on " << report.rowsUsed
         << " rows | vocab=" << model.vocabSize() << "\n";
}

PredictResult SentimentSystem::predictFull(const string &text, bool saveToHistory)
{
    PredictResult result;
    result.label = "neutral";
    result.confidence = 50.0;
    result.isUnknown = false;
    result.oovRatio = 0.0;

    if (!model.isTrained())
    {
        result.label = "Model not trained";
        return result;
    }

    vector<string> words = tokenizer.tokenize(text);
    if (words.empty()) return result;

    result.oovRatio = oovChecker.calculateRatio(model, words);

    ScoreBoard scores = predictor.scoreText(model, words);
    result.label = scores.bestLabel();
    result.confidence = scores.confidence();

    if (result.confidence < 40.0)
        result.label = "neutral";

    result.keywords = keywordFinder.findKeywords(model, words);
    result.isUnknown = (result.oovRatio > 0.6 && result.confidence < 70.0);

    if (saveToHistory)
        history.add(text, result.label, result.confidence);

    return result;
}

double SentimentSystem::testAccuracy(const string &filename)
{
    return accuracyTester.test(filename, *this);
}

vector<HistoryRow>& SentimentSystem::getHistory()
{
    return history.getRows();
}

size_t SentimentSystem::getVocabSize() const
{
    return model.vocabSize();
}

void SentimentSystem::exportHistory(const string &path)
{
    historyExporter.exportToCsv(path, history.getRows());
}

static SentimentSystem sentimentSystem;

void trainModel(const string &filename)
{
    sentimentSystem.train(filename);
}

double testAccuracy(const string &filename)
{
    return sentimentSystem.testAccuracy(filename);
}

pair<string, double> predict(const string &text)
{
    PredictResult result = sentimentSystem.predictFull(text, true);
    return {result.label, result.confidence};
}

PredictResult predictFull(const string &text)
{
    return sentimentSystem.predictFull(text, true);
}

vector<HistoryRow>& getHistory()
{
    return sentimentSystem.getHistory();
}

size_t getVocabSize()
{
    return sentimentSystem.getVocabSize();
}

void exportHistory(const string &path)
{
    sentimentSystem.exportHistory(path);
}
