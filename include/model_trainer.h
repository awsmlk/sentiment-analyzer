#ifndef MODEL_TRAINER_H
#define MODEL_TRAINER_H

#include "csv_line_reader.h"
#include "label_helper.h"
#include "sentiment_model.h"
#include "tokenizer.h"
#include "training_report.h"

#include <string>

class ModelTrainer {
    CsvLineReader csvReader;
    LabelHelper labelHelper;
    Tokenizer tokenizer;

public:
    TrainingReport train(SentimentModel &model, const std::string &filename) const;
};

#endif
