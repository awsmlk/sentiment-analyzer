#ifndef WORD_CLEANER_H
#define WORD_CLEANER_H

#include <string>

class WordCleaner {
public:
    std::string toLower(std::string word) const;
    std::string removePunctuation(std::string word) const;
    std::string normalize(const std::string &word) const;
    std::string clean(std::string word) const;
};

#endif
