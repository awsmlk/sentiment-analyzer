#ifndef SENTIMENT_H
#define SENTIMENT_H

#include <string>
#include <vector>
#include <utility>
#include <map>

using namespace std;

// ─────────────────────────────────────────────────────────────
// Structs
// ─────────────────────────────────────────────────────────────

struct KeywordScore {
    string word;
    double score;
    string polarity; // "pos", "neg", "neu"
};

struct PredictResult {
    string label;
    double confidence;
    bool isUnknown;
    double oovRatio;
    vector<KeywordScore> keywords;
};

struct HistoryRow {
    string time;
    string label;
    double score;
    string text;
};

// ─────────────────────────────────────────────────────────────
// Core API
// ─────────────────────────────────────────────────────────────

void trainModel(const string &filename);
double testAccuracy(const string &filename);

pair<string,double> predict(const string &text);
PredictResult predictFull(const string &text);

vector<HistoryRow>& getHistory();

size_t getVocabSize();

void exportHistory(const string &path);

#endif