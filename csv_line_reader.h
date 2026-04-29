#ifndef CSV_LINE_READER_H
#define CSV_LINE_READER_H

#include "training_example.h"

#include <string>

class CsvLineReader {
public:
    TrainingExample parseLine(const std::string &line) const;
};

#endif
