#ifndef SENTIMENT_H
#define SENTIMENT_H

#include <string>
#include <vector>
#include <map>

struct HistoryRow {
    std::string time;
    std::string label;
    double score;
    std::string text;
};

void trainModel(const std::string& filename);

std::pair<std::string,double> predict(const std::string& text);

std::vector<HistoryRow>& getHistory();

void exportHistory(const std::string& path);

double testAccuracy(const std::string& filename);

#endif