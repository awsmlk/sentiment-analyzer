#ifndef SCORE_BOARD_H
#define SCORE_BOARD_H

#include <map>
#include <string>

class ScoreBoard {
public:
    std::map<std::string, double> logScores;

    void addScore(const std::string &label, double score);
    std::string bestLabel() const;
    double bestScore() const;
    double confidence() const;
};

#endif
