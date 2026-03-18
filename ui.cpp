#include "ui.h"
#include "sentiment.h"

#include <iostream>
#include <limits>
#include <sstream>

using namespace std;

// ANSI color codes
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define RESET   "\033[0m"

void menu() {
    cout << "\n==============================\n";
    cout << " Sentiment Analyzer\n";
    cout << "==============================\n";
    cout << "1 Analyze Text\n";
    cout << "2 Show History\n";
    cout << "3 Export CSV\n";
    cout << "4 Exit\n";
    cout << "Select: ";
}

void startUI() {
    int choice;

    // Automatically train the model on program start
    cout << "Training model automatically...\n";
    trainModel("data/dataset.csv");
    double acc = testAccuracy("data/dataset.csv");
    cout << "Model trained | Accuracy: " << acc << "%\n";

    while (true) {
        menu();

        string line;
        getline(cin, line);
        stringstream ss(line);

        if (!(ss >> choice)) {
            cout << "Invalid input, please enter a number.\n";
            continue;
        }

        if (choice == 1) {  // Analyze Text
            string continueTesting;
            do {
                cout << "\nEnter text to analyze:\n";
                string text;
                getline(cin, text);

                auto result = predict(text);

                string color = RESET;
                if(result.first == "positive") color = GREEN;
                else if(result.first == "neutral") color = YELLOW;
                else if(result.first == "negative") color = RED;

                cout << "Prediction: " << color << result.first << RESET
                     << " | Confidence: " << result.second << "%\n";

                cout << "Test another sentence? (y/n): ";
                getline(cin, continueTesting);

            } while (!continueTesting.empty() &&
                     (continueTesting[0] == 'y' || continueTesting[0] == 'Y'));
        }
        else if (choice == 2) { // Show History
            auto &hist = getHistory();
            cout << "\n--- Prediction History ---\n";
            for(auto &h : hist) {
                string color = RESET;
                if(h.label == "positive") color = GREEN;
                else if(h.label == "neutral") color = YELLOW;
                else if(h.label == "negative") color = RED;

                cout << h.time << " | "
                     << color << h.label << RESET
                     << " | " << h.score << "% | "
                     << h.text << "\n";
            }
        }
        else if (choice == 3) { // Export CSV
            exportHistory("history/history.csv");
            cout << "History exported to history/history.csv\n";
        }
        else if (choice == 4) { // Exit
            cout << "Exiting program. Goodbye!\n";
            break;
        }
        else {
            cout << "Invalid choice. Try again.\n";
        }
    }
}