#ifndef SENTIMENT_SYSTEM_H
#define SENTIMENT_SYSTEM_H

#include "accuracy_tester.h"
#include "bayes_predictor.h"
#include "history_exporter.h"
#include "history_manager.h"
#include "keyword_finder.h"
#include "model_trainer.h"
#include "oov_checker.h"
#include "sentiment.h"
#include "sentiment_model.h"
#include "tokenizer.h"

#include <string>
#include <vector>

class SentimentSystem {
    SentimentModel model;
    ModelTrainer trainer;
    Tokenizer tokenizer;
    OovChecker oovChecker;
    BayesPredictor predictor;
    KeywordFinder keywordFinder;
    HistoryManager history;
    AccuracyTester accuracyTester;
    HistoryExporter historyExporter;

public:
    void train(const std::string &filename);
    PredictResult predictFull(const std::string &text, bool saveToHistory);
    double testAccuracy(const std::string &filename);
    std::vector<HistoryRow>& getHistory();
    size_t getVocabSize() const;
    void exportHistory(const std::string &path);
};

#endif
