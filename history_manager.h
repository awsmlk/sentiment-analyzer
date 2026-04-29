#ifndef HISTORY_MANAGER_H
#define HISTORY_MANAGER_H

#include "sentiment.h"

#include <string>
#include <vector>

class HistoryManager {
    std::vector<HistoryRow> rows;

public:
    void add(const std::string &text, const std::string &label, double confidence);
    std::vector<HistoryRow>& getRows();
};

#endif
