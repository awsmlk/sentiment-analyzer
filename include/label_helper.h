#ifndef LABEL_HELPER_H
#define LABEL_HELPER_H

#include <string>

class LabelHelper {
public:
    std::string clean(std::string label) const;
    bool isValid(const std::string &label) const;
};

#endif
