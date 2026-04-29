#include "ui.h"
#include "sentiment.h"

#include <iostream>
#include <limits>
#include <sstream>
#include <iomanip>
#include <fstream>

using namespace std;

#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define CYAN    "\033[36m"
#define RESET   "\033[0m"
#define BOLD    "\033[1m"

void menu()
{
    cout << "\n==============================\n";
    cout << BOLD " Sentiment Analyzer  v2\n" RESET;
    cout << "==============================\n";
    cout << "1  Analyze Text\n";
    cout << "2  Show History\n";
    cout << "3  Export CSV\n";
    cout << "4  Exit\n";
    cout << "Select: ";
}

void startUI()
{
    cout << "Training model automatically…\n";
    trainModel("data/dataset.csv");
    double acc = testAccuracy("data/dataset.csv");
    cout << "Model trained | Accuracy: " << fixed << setprecision(1)
         << acc << "% | Vocab: " << getVocabSize() << " tokens\n";

    int choice;
    while (true)
    {
        menu();

        string line;
        getline(cin, line);
        stringstream ss(line);

        if (!(ss >> choice))
        {
            cout << "Invalid input, please enter a number.\n";
            continue;
        }

        // ── 1. Analyze Text ───────────────────────────────────────────────
        if (choice == 1)
        {
            string cont;
            do
            {
                cout << "\nEnter text to analyze:\n> ";
                string text;
                getline(cin, text);

                PredictResult res = predictFull(text);

                string color = RESET;
                if      (res.label == "positive") color = GREEN;
                else if (res.label == "neutral")  color = YELLOW;
                else if (res.label == "negative") color = RED;

                cout << "Prediction : " << color << BOLD << res.label << RESET
                     << "  |  Confidence: " << fixed << setprecision(1)
                     << res.confidence << "%\n";

                // OOV / unknown hint
                if (res.isUnknown)
                {
                    cout << YELLOW << "⚠ Warning: " << (int)(res.oovRatio * 100)
                         << "% of words are unknown to the model.\n"
                         << "  Consider teaching this example via the teach endpoint\n"
                         << "  or adding it to data/dataset.csv.\n" << RESET;
                }

                // Keywords
                if (!res.keywords.empty())
                {
                    cout << "Key signals: ";
                    for (size_t i = 0; i < res.keywords.size(); i++)
                    {
                        if (i) cout << "  ";
                        const auto &k = res.keywords[i];
                        string kc = (k.polarity == "pos") ? GREEN
                                  : (k.polarity == "neg") ? RED
                                  :                         YELLOW;
                        cout << kc << k.word << RESET;
                    }
                    cout << "\n";
                }

                // Prompt to teach if unknown
                if (res.isUnknown)
                {
                    cout << "\nWould you like to teach this example? (y/n): ";
                    string answer;
                    getline(cin, answer);
                    if (!answer.empty() && (answer[0] == 'y' || answer[0] == 'Y'))
                    {
                        cout << "Label this text [positive/neutral/negative]: ";
                        string label;
                        getline(cin, label);
                        // normalise
                        if (label == "p" || label == "pos") label = "positive";
                        if (label == "n" || label == "neg") label = "negative";
                        if (label == "neu" || label == "0") label = "neutral";

                        if (label == "positive" || label == "neutral" || label == "negative")
                        {
                            // Append to dataset and retrain
                            ofstream f("data/dataset.csv", ios::app);
                            if (f.is_open())
                            {
                                bool needsQuote = text.find(',') != string::npos;
                                string safe = text;
                                for (char &c : safe) if (c == '\n' || c == '\r') c = ' ';
                                f << (needsQuote ? "\"" + safe + "\"" : safe)
                                  << "," << label << "\n";
                                cout << CYAN << "✓ Saved to dataset.csv as \"" << label << "\"\n" << RESET;
                                cout << "  Retraining model… ";
                                trainModel("data/dataset.csv");
                                double newAcc = testAccuracy("data/dataset.csv");
                                cout << "done | New accuracy: " << fixed << setprecision(1)
                                     << newAcc << "%\n";
                            }
                            else
                            {
                                cout << RED << "Error: could not open data/dataset.csv for writing.\n" << RESET;
                            }
                        }
                        else
                        {
                            cout << "Unknown label, skipping.\n";
                        }
                    }
                }

                cout << "\nTest another sentence? (y/n): ";
                getline(cin, cont);

            } while (!cont.empty() && (cont[0] == 'y' || cont[0] == 'Y'));
        }

        // ── 2. Show History ───────────────────────────────────────────────
        else if (choice == 2)
        {
            auto &hist = getHistory();
            cout << "\n--- Prediction History (" << hist.size() << " entries) ---\n";
            for (auto &h : hist)
            {
                string color = RESET;
                if      (h.label == "positive") color = GREEN;
                else if (h.label == "neutral")  color = YELLOW;
                else if (h.label == "negative") color = RED;

                cout << h.time << " | "
                     << color << h.label << RESET
                     << " | " << fixed << setprecision(1) << h.score << "% | "
                     << h.text << "\n";
            }
        }

        // ── 3. Export CSV ─────────────────────────────────────────────────
        else if (choice == 3)
        {
            exportHistory("history/history.csv");
            cout << "History exported to history/history.csv\n";
        }

        // ── 4. Exit ───────────────────────────────────────────────────────
        else if (choice == 4)
        {
            cout << "Exiting. Goodbye!\n";
            break;
        }
        else
        {
            cout << "Invalid choice. Try again.\n";
        }
    }
}