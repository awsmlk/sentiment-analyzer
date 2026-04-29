#ifndef SENTIMENT_MODEL_H
#define SENTIMENT_MODEL_H

#include <map>
#include <string>
#include <unordered_set>
#include <vector>

class SentimentModel {
public:
    std::map<std::string, std::map<std::string, int>> wordCounts;
    std::map<std::string, int> labelCounts;
    std::map<std::string, int> totalWords;
    std::map<std::string, double> priors;
    std::unordered_set<std::string> vocabulary;

    void clear();
    void learnFromTokens(const std::string &label, const std::vector<std::string> &tokens);
    void calculatePriors();
    bool isTrained() const;
    size_t vocabSize() const;
    int countWord(const std::string &label, const std::string &word) const;
    int totalForLabel(const std::string &label) const;
};

#endif
