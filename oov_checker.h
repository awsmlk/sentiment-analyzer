#ifndef OOV_CHECKER_H
#define OOV_CHECKER_H

#include "sentiment_model.h"

#include <string>
#include <vector>

class OovChecker {
public:
    double calculateRatio(const SentimentModel &model, const std::vector<std::string> &words) const;
};

#endif
