#ifndef TRAINING_EXAMPLE_H
#define TRAINING_EXAMPLE_H

#include <string>

class TrainingExample {
public:
    std::string text;
    std::string label;

    TrainingExample();
    TrainingExample(const std::string &textValue, const std::string &labelValue);
};

#endif
