#ifndef HISTORY_EXPORTER_H
#define HISTORY_EXPORTER_H

#include "sentiment.h"

#include <string>
#include <vector>

class HistoryExporter {
public:
    void exportToCsv(const std::string &path, const std::vector<HistoryRow> &rows) const;
};

#endif
