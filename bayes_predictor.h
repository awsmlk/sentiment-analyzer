#ifndef BAYES_PREDICTOR_H
#define BAYES_PREDICTOR_H

#include "score_board.h"
#include "sentiment_model.h"

#include <string>
#include <vector>

class BayesPredictor {
public:
    ScoreBoard scoreText(const SentimentModel &model, const std::vector<std::string> &words) const;
};

#endif
