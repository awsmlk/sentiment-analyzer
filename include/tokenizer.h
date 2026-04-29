#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "word_cleaner.h"

#include <string>
#include <vector>

class Tokenizer {
    WordCleaner cleaner;

public:
    std::vector<std::string> tokenize(const std::string &text) const;
};

#endif
