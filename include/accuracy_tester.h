#ifndef ACCURACY_TESTER_H
#define ACCURACY_TESTER_H

#include "csv_line_reader.h"
#include "label_helper.h"

#include <string>

class SentimentSystem;

class AccuracyTester {
    CsvLineReader csvReader;
    LabelHelper labelHelper;

public:
    double test(const std::string &filename, SentimentSystem &system) const;
};

#endif
