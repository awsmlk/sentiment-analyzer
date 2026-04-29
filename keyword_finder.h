#ifndef KEYWORD_FINDER_H
#define KEYWORD_FINDER_H

#include "sentiment.h"
#include "sentiment_model.h"

#include <string>
#include <vector>

class KeywordFinder {
public:
    std::vector<KeywordScore> findKeywords(const SentimentModel &model,
                                           const std::vector<std::string> &words) const;
};

#endif
