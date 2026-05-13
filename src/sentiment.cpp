#include "sentiment_system.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;

/*
      to turn on backend:
      cd ..
      g++ -std=c++17 -O2 -Iinclude -o server src/server.cpp src/sentiment.cpp -pthread
      ./server
*/

TrainingExample::TrainingExample() {} // Default constructor for empty examples

TrainingExample::TrainingExample(const string &textValue, const string &labelValue) // Constructor to initialize text and label
{
    text = textValue;
    label = labelValue;
}

TrainingExample CsvLineReader::parseLine(const string &line) const // take a CSV line and convert it to text using TrainingExample, handling quoted text with commas
{
    string text;
    string label;

    if (line.empty()) return TrainingExample(); // Return an empty example if the line is empty

    // If the text starts with a quote, it may contain commas.
    if (line[0] == '"') 
    {
        size_t closeQuote = line.find('"', 1); // Find the closing quote
        if (closeQuote != string::npos) // If we found a closing quote, extract the text and label
        {
            text = line.substr(1, closeQuote - 1); // Extract text between the quotes
            size_t labelStart = closeQuote + 2; // The label should start after the closing quote and a comma
            if (labelStart < line.size()) 
                label = line.substr(labelStart); // Extract the label from the remaining part of the line
        }
    }
    else
    {
        stringstream ss(line); // If there are no quotes, we can simply split on the first comma
        getline(ss, text, ','); // Extract text up to the first comma
        getline(ss, label, ','); // Extract label from the remaining part of the line   
    }

    return TrainingExample(text, label); // Returns the example with text and label
}

string LabelHelper::clean(string label) const // Clean the label by converting to lowercase and removing whitespace and cursor returns
{
    transform(label.begin(), label.end(), label.begin(), // Convert label to lowercase
              [](unsigned char c) { return static_cast<char>(tolower(c)); }); //conv operator algo lib

    while (!label.empty() && (label.back() == '\r' || label.back() == ' ')) // Remove trailing carriage returns and spaces
        label.pop_back();

    while (!label.empty() && label.front() == ' ') // Remove leading spaces
        label.erase(label.begin());

    return label;
}

bool LabelHelper::isValid(const string &label) const // Check if the label is one of the valid sentiment labels: positive, neutral, or negative
{
    return label == "positive" || label == "neutral" || label == "negative"; // Only these three labels are considered valid for training and prediction
}

string WordCleaner::toLower(string word) const // Convert a word to lowercase using the standard library transform function
{
    transform(word.begin(), word.end(), word.begin(), // Convert each character to lowercase using the tolower function from the C library
              [](unsigned char c) { return static_cast<char>(tolower(c)); }); // This ensures that the word is in a consistent format for further processing
    return word;
}

string WordCleaner::removePunctuation(string word) const // Remove punctuation from the word using the remove_if algorithm and ispunct function from the C library
{
    word.erase(remove_if(word.begin(), word.end(), // Remove punctuation characters from the word
                         [](unsigned char c) { return ispunct(c); }), // The remove_if algorithm moves non-punctuation characters to the front of the string and returns an iterator to the new end of the string
               word.end()); // The erase function then removes the "removed" characters from the string, leaving only the non-punctuation characters
    return word;
}

string WordCleaner::normalize(const string &word) const // Normalize common contractions like (isnt to is not) and variations to their base forms to improve model consistency
{
    if (word == "im") return "i";
    if (word == "ive") return "i";
    if (word == "dont") return "not";
    if (word == "cant") return "not";
    if (word == "isnt") return "not";
    if (word == "wasnt") return "not";
    if (word == "wont") return "not";
    return word;
}

string WordCleaner::clean(string word) const // Clean the word by applying lowercase conversion, punctuation removal, and normalization in sequence to prepare it for tokenization and model training/prediction
{
    word = toLower(word);
    word = normalize(word);
    return word;
}

vector<string> Tokenizer::tokenize(const string &text) const // Tokenize the input text into a vector of cleaned words, while also creating combined tokens for negations (like, "not good" becomes "not_good") and word pairs (like, "very good" becomes "very_good") to capture more complex sentiment patterns in the text
{
    vector<string> words; // This will hold the final list of tokens extracted from the input text after cleaning and combining certain words based on their context (negations and pairs).
    string previousWord; // This variable keeps track of the previous word processed, which is necessary for creating combined tokens for negations and word pairs. It starts empty and gets updated as we iterate through the words in the input text.
    string currentWord; // This variable holds the current word being processed in the loop. It is cleaned and potentially combined with the previous word before being added to the list of tokens.
    stringstream ss(text); // A stringstream is used to read the input text word by word. It allows us to easily extract words from the text while handling whitespace and other delimiters.

    while (ss >> currentWord) // Read each word from the input text into currentWord. The loop continues until there are no more words to read from the stringstream.
    {
        currentWord = cleaner.clean(currentWord); // Clean the current word using the WordCleaner, which applies lowercase conversion, punctuation removal, and normalization to prepare the word for tokenization and model training/prediction.
        if (currentWord.empty()) continue; // If the cleaned word is empty (which can happen if the original word was just punctuation), we skip it and move on to the next word in the input text.

        // If the previous word was "not", we want to combine it with the current word to create a stronger token that captures the negation. For example, if the input text contains "not good", we want to create a token "not_good" that represents the negated sentiment more effectively than treating "not" and "good" as separate tokens.
        if (previousWord == "not") {
            words.pop_back();
            currentWord = "not_" + currentWord;
        }

        // Also store simple word pairs, like "very_good".
        if (!previousWord.empty() && previousWord != "not") 
            words.push_back(previousWord + "_" + currentWord);

        words.push_back(currentWord); // Add the current word (which may have been modified to include "not_" or combined with the previous word) to the list of tokens. This ensures that we still include the individual words in the token list, in addition to any combined tokens we may have created.
        previousWord = currentWord; // Update the previousWord variable to the current word for the next iteration of the loop. This allows us to check for negations and create combined tokens in the next iteration when we process the next word in the input text.
    }

    return words;
}

void SentimentModel::clear() // Clear the model's data structures to reset the model to an untrained state, allowing it to be retrained from scratch with new data. This is useful for scenarios where we want to discard the current model and start fresh without any of the previously learned information.
{
    wordCounts.clear();
    labelCounts.clear();
    totalWords.clear();
    priors.clear();
    vocabulary.clear();
}

void SentimentModel::learnFromTokens(const string &label, const vector<string> &tokens) // Update the model's word counts, label counts, total word counts, and vocabulary based on the provided label and tokens. This function is called during training to allow the model to learn from the training examples by counting how many times each word appears in examples of each label, as well as keeping track of the total number of words for each label and the overall vocabulary of unique words seen during training.
{
    labelCounts[label]++;

    for (const string &word : tokens) // For each token in the input vector of tokens, we update the model's data structures to reflect the occurrence of that word in an example with the given label. This involves incrementing the count of that word for the specific label, updating the total word count for that label, and adding the word to the overall vocabulary set.
    {
        wordCounts[label][word]++;
        totalWords[label]++; 
        vocabulary.insert(word);
    }
}

void SentimentModel::calculatePriors() // Calculate the prior probabilities for each label based on the label counts. This function is called after training to compute the priors, which represent the overall probability of each label in the training data. The priors are used during prediction to help determine the most likely label for a given input text based on the learned word counts and the overall distribution of labels in the training data.
{
    int totalDocuments = 0;
    for (const auto &item : labelCounts) // Iterate through the labelCounts map, where each item is a pair of label and its count. We add up the counts of all labels to get the total number of documents in the training data.
        totalDocuments += item.second; // The second element of the pair (item.second) is the count of examples for that label, and we add it to the totalDocuments variable.

    if (totalDocuments == 0) return; // If there are no documents, we cannot calculate priors, so we return early.

    for (const auto &item : labelCounts) // Now that we have the total number of documents, we can calculate the prior probability for each label by dividing the count of examples for that label by the total number of documents. We store these priors in the priors map, which will be used during prediction to help determine the most likely label for a given input text based on the learned word counts and the overall distribution of labels in the training data.
        priors[item.first] = static_cast<double>(item.second) / totalDocuments; // The first element of the pair (item.first) is the label, and the second element (item.second) is the count of examples for that label. We calculate the prior probability by dividing the count for that label by the total number of documents, and we store it in the priors map with the label as the key.
}

bool SentimentModel::isTrained() const // Check if the model has been trained by verifying that the priors map is not empty. If the priors map contains entries, it indicates that the model has been trained on some data and has calculated the prior probabilities for each label. If the priors map is empty, it means that the model has not been trained yet and does not have any learned information to make predictions.
{
    return !priors.empty();
}

size_t SentimentModel::vocabSize() const // Return the size of the vocabulary, which is the number of unique words that the model has seen during training. This is calculated by returning the size of the vocabulary set, which contains all the unique words that were encountered in the training examples. The vocabulary size can be useful for understanding the complexity of the model and for debugging purposes to see how many unique words it has learned from the training data.
{
    return vocabulary.size();
}

int SentimentModel::countWord(const string &label, const string &word) const // Return the count of how many times a specific word has been seen in examples with a given label. This function looks up the word count for the specified label and word in the wordCounts map. If the label or word is not found, it returns 0, indicating that the word has not been seen in examples with that label during training. This function is used during prediction to calculate the likelihood of a given input text belonging to a specific label based on the learned word counts from the training data.
{
    auto labelIt = wordCounts.find(label); // First, we look for the specified label in the wordCounts map. If the label is not found, it means that we have not seen any examples with that label during training, so we return 0.
    if (labelIt == wordCounts.end()) return 0; // If the label is found, we then look for the specified word in the inner map that corresponds to that label. If the word is not found, it means that we have not seen that word in examples with that label during training, so we return 0.

    auto wordIt = labelIt->second.find(word); // If both the label and word are found, we return the count of how many times that word has been seen in examples with that label, which is stored in the inner map of wordCounts.
    if (wordIt == labelIt->second.end()) return 0; // If the word is not found in the inner map for that label, we return 0, indicating that the word has not been seen in examples with that label during training.

    return wordIt->second; // If both the label and word are found, we return the count of how many times that word has been seen in examples with that label, which is stored in the inner map of wordCounts.
}

int SentimentModel::totalForLabel(const string &label) const // Return the total number of words seen in examples with a given label. This function looks up the total word count for the specified label in the totalWords map. If the label is not found, it returns 0, indicating that no words have been seen in examples with that label during training. This function is used during prediction to calculate the likelihood of a given input text belonging to a specific label based on the learned word counts from the training data.
{
    auto it = totalWords.find(label); // We look for the specified label in the totalWords map. If the label is not found, it means that we have not seen any examples with that label during training, so we return 0.
    return (it == totalWords.end()) ? 0 : it->second; // If the label is found, we return the total number of words seen in examples with that label, which is stored in the totalWords map. If the label is not found, we return 0, indicating that no words have been seen in examples with that label during training.
}

TrainingReport ModelTrainer::train(SentimentModel &model, const string &filename) const // Train the provided SentimentModel using the training data from the specified CSV file. This function reads the training data from the file, processes each line to extract the text and label, cleans and tokenizes the text, and updates the model with the learned information. It also keeps track of how many rows were used for training and returns a TrainingReport with that information.
{
    TrainingReport report; // Initialize a TrainingReport to keep track of how many rows were used for training. This report will be returned at the end of the function to provide feedback on the training process.
    model.clear(); // Clear the model's data structures to reset it to an untrained state before we start training with new data. This ensures that any previous training information is discarded and the model starts fresh with the new training data from the specified CSV file.

    ifstream file(filename); // Open the specified CSV file for reading. This file is expected to contain the training data, where each line represents a training example with text and its corresponding label. If the file cannot be opened, we print an error message and return the report with zero rows used for training.
    if (!file.is_open()) // Check if the file was successfully opened. If not, we print an error message and return the report with zero rows used for training, since we cannot proceed without access to the training data.
    {
        cout << "Error: Could not open dataset file: " << filename << endl ; 
        return report;
    }

    string line; // Read the first line of the CSV file, which is expected to be the header, and skip it. This allows us to start processing the actual training examples from the second line onward. The header typically contains column names and is not part of the training data, so we skip it to avoid including it in the model training.
    getline(file, line); // skip CSV header

    while (getline(file, line)) // Read each subsequent line from the CSV file, which represents a training example. For each line, we parse it to extract the text and label, clean the label, check if it's valid, tokenize the text, and then update the model with the learned information from that example. We also increment the count of rows used for training in the report.
    {
        TrainingExample example = csvReader.parseLine(line); // Parse the line from the CSV file to extract the text and label, creating a TrainingExample object that holds this information. The parseLine function handles the parsing of the CSV line, including dealing with quoted text that may contain commas, to ensure that we correctly extract the text and label for each training example.
        example.label = labelHelper.clean(example.label); // Clean the label using the LabelHelper, which converts it to lowercase and removes any leading or trailing whitespace and carriage returns. This ensures that the label is in a consistent format for training and prediction, allowing the model to learn from the training examples effectively.

        if (!labelHelper.isValid(example.label)) continue; // Check if the cleaned label is valid (like, it is one of "positive", "neutral", or "negative"). If the label is not valid, we skip this training example and move on to the next line in the CSV file. This ensures that we only train the model on examples with valid sentiment labels, which helps improve the quality of the model's predictions.

        vector<string> tokens = tokenizer.tokenize(example.text); // Tokenize the text of the training example using the Tokenizer, which cleans the text and creates tokens while also handling negations and word pairs. This process prepares the text for training by converting it into a format that the model can learn from, allowing it to capture the sentiment patterns in the text effectively.
        model.learnFromTokens(example.label, tokens); //
        report.rowsUsed++; // Update the model with the learned information from this training example by calling the learnFromTokens function, which updates the word counts, label counts, total word counts, and vocabulary based on the provided label and tokens. This allows the model to learn from the training examples and improve its ability to make accurate predictions in the future. We also increment the count of rows used for training in the report to keep track of how many examples were processed during training.
    }

    model.calculatePriors(); // After processing all the training examples, we calculate the prior probabilities for each label based on the label counts. This is an important step in preparing the model for making predictions, as the priors represent the overall probability of each label in the training data and are used during prediction to help determine the most likely label for a given input text based on the learned word counts and the overall distribution of labels in the training data.
    return report; // After processing all the training examples, we calculate the prior probabilities for each label based on the label counts. This is an important step in preparing the model for making predictions, as the priors represent the overall probability of each label in the training data and are used during prediction to help determine the most likely label for a given input text based on the learned word counts and the overall distribution of labels in the training data. Finally, we return the TrainingReport with information about how many rows were used for training.
}

double OovChecker::calculateRatio(const SentimentModel &model, const vector<string> &words) const // Calculate the ratio of out-of-vocabulary (OOV) words in the input text based on the model's vocabulary. This function counts how many words in the input vector of words are not present in the model's vocabulary and divides that count by the total number of normal words (excluding any special tokens with underscores) to get the OOV ratio. This ratio can be used to determine how much of the input text contains words that the model has not seen during training, which can affect the confidence of predictions and help identify cases where the model may struggle to make accurate predictions due to unfamiliar vocabulary.
{
    int unknownWords = 0; // This variable counts how many words in the input vector of words are not present in the model's vocabulary. We will increment this count for each word that is not found in the model's vocabulary, which will help us calculate the OOV ratio later on.
    int normalWords = 0; // This variable counts the total number of normal words in the input vector of words, excluding any special tokens that contain underscores (which are generated by our tokenizer for negations and word pairs). We will increment this count for each normal word we encounter, which will be used as the denominator when calculating the OOV ratio to determine the proportion of words in the input text that are out-of-vocabulary for the model.

    for (const string &word : words) // Iterate through each word in the input vector of words. For each word, we will check if it is a normal word (not containing an underscore) and if it is present in the model's vocabulary. This allows us to count how many normal words are in the input and how many of those are unknown to the model, which will be used to calculate the OOV ratio.
    {
        // Words with "_" are generated by our tokenizer, so skip them here.
        if (word.find('_') != string::npos) continue; 

        normalWords++;
        if (model.vocabulary.find(word) == model.vocabulary.end()) // If the word is not found in the model's vocabulary, we increment the count of unknown words. This indicates that the model has not seen this word during training, which can affect the confidence of predictions and help identify cases where the model may struggle to make accurate predictions due to unfamiliar vocabulary.
            unknownWords++;
    }

    if (normalWords == 0) return 0.0; // If there are no normal words in the input, we return an OOV ratio of 0.0 to avoid division by zero. This means that if the input text does not contain any normal words (only special tokens with underscores), we consider the OOV ratio to be 0, as there are no normal words to compare against the model's vocabulary.
    return static_cast<double>(unknownWords) / normalWords; // Finally, we calculate the OOV ratio by dividing the count of unknown words by the total number of normal words. This ratio represents the proportion of words in the input text that are out-of-vocabulary for the model, which can be used to assess how much of the input text contains unfamiliar vocabulary that may affect the confidence of predictions.
}

void ScoreBoard::addScore(const string &label, double score) // Add a score for a specific label to the ScoreBoard. This function takes a label and its corresponding score (which is typically a log probability) and stores it in the logScores map. The logScores map holds the scores for each label, which will be used to determine the best label and calculate confidence during prediction.
{
    logScores[label] = score; // We store the score for the given label in the logScores map, where the key is the label and the value is the score. This allows us to keep track of the scores for each label, which will be used later to determine the best label and calculate confidence during prediction.
}

string ScoreBoard::bestLabel() const // Determine the best label based on the scores in the ScoreBoard. This function iterates through the logScores map to find the label with the highest score (which is typically a log probability). It returns the label that has the highest score, which is considered the best label for the given input text based on the model's predictions.
{
    string best = "neutral"; // Initialize the best label to "neutral" as a default. This means that if all scores are very low or if there are no scores, we will return "neutral" as the best label. However, we will update this variable as we iterate through the logScores to find the label with the highest score.
    double bestScore = -1e18; // Initialize the best score to a very low value (negative infinity) to ensure that any valid score will be higher than this initial value. This allows us to correctly identify the label with the highest score as we iterate through the logScores map.

    for (const auto &item : logScores) // Iterate through each item in the logScores map, where each item is a pair of label and its corresponding score. We will compare the score of each label to the current best score to determine if we have found a new best label.
    {
        if (item.second > bestScore) // If the score for the current label (item.second) is greater than the current best score, we update the best score and the best label to reflect this new best label. This allows us to find the label with the highest score as we iterate through the logScores map.
        {
            bestScore = item.second; // Update the best score to the score of the current label, since it is higher than the previous best score.
            best = item.first; // Update the best label to the current label (item.first) since it has the highest score so far.
        }
    }

    return best;
}

double ScoreBoard::bestScore() const // Return the best score from the ScoreBoard, which is the highest score among all the labels. This function iterates through the logScores map to find and return the highest score, which can be used to assess the confidence of the prediction for the best label.
{
    double best = -1e18; // Initialize the best score to a very low value (negative infinity) to ensure that any valid score will be higher than this initial value. This allows us to correctly identify the highest score as we iterate through the logScores map.
    for (const auto &item : logScores) // Iterate through each item in the logScores map, where each item is a pair of label and its corresponding score. We will compare the score of each label to the current best score to determine if we have found a new best score.
        best = max(best, item.second); // Update the best score to be the maximum of the current best score and the score of the current label (item.second). This allows us to find the highest score among all the labels as we iterate through the logScores map.
    return best;
}

double ScoreBoard::confidence() const // Calculate the confidence of the prediction based on the scores in the ScoreBoard. This function uses the log scores to calculate a confidence value between 0 and 100, which represents how confident the model is in its prediction for the best label. The confidence is calculated using a softmax-like approach, where we exponentiate the log scores and normalize them to get a probability distribution over the labels, and then convert that to a percentage.
{
    double best = bestScore(); // First, we get the best score from the ScoreBoard, which is the highest log score among all the labels. This will be used as a reference point to calculate the confidence of the prediction.  
    double sum = 0.0; // We will calculate the sum of the exponentiated log scores, normalized by the best score, to get a value that we can use to calculate the confidence. This is similar to the softmax function, where we exponentiate the log scores and normalize them to get a probability distribution over the labels.

    for (const auto &item : logScores) // Iterate through each item in the logScores map, where each item is a pair of label and its corresponding score. We will exponentiate the log score of each label, normalized by the best score, and add it to the sum. This allows us to calculate a value that we can use to determine the confidence of the prediction for the best label.
        sum += exp(item.second - best); // We exponentiate the log score of each label (item.second) normalized by the best score (best) to get a value that represents the relative likelihood of that label compared to the best label. We add this value to the sum, which will be used to calculate the confidence of the prediction for the best label.

    if (sum <= 0.0) return 50.0; // If the sum is zero or negative (which can happen if all scores are very low), we return a default confidence of 50.0, indicating that the model is uncertain about the prediction. This is a safeguard against cases where the scores do not provide a clear indication of confidence.
    return (1.0 / sum) * 100.0; // Finally, we calculate the confidence of the prediction for the best label by taking the reciprocal of the sum (which gives us a value between 0 and 1) and multiplying it by 100 to convert it to a percentage. This confidence value represents how confident the model is in its prediction for the best label based on the scores in the ScoreBoard.
}

ScoreBoard BayesPredictor::scoreText(const SentimentModel &model, const vector<string> &words) const // Calculate the scores for each label based on the input words and the trained SentimentModel. This function uses the Naive Bayes approach to calculate the log probabilities for each label by combining the prior probabilities of the labels with the likelihood of the input words given each label. It returns a ScoreBoard containing the scores for each label, which can be used to determine the best label and calculate confidence during prediction.
{
    ScoreBoard scores; // Initialize a ScoreBoard to hold the scores for each label. This will be used to store the calculated log probabilities for each label based on the input words and the trained SentimentModel. The scores in the ScoreBoard will be used later to determine the best label and calculate confidence during prediction.
    size_t vocabSize = max(static_cast<size_t>(1), model.vocabSize()); // Get the size of the vocabulary from the model, which is the number of unique words that the model has seen during training. We use max with 1 to ensure that we do not have a vocabulary size of zero, which would cause issues when calculating probabilities with Laplace smoothing.

    for (const auto &prior : model.priors) // Iterate through each prior probability in the model's priors map, where each prior is a pair of label and its corresponding prior probability. For each label, we will calculate the log probability score based on the prior and the likelihood of the input words given that label, and we will add that score to the ScoreBoard.
    {
        string label = prior.first; // Get the label from the prior, which is the first element of the pair (prior.first). This label will be used to calculate the score for that specific label based on the input words and the trained SentimentModel.
        double score = log(prior.second); // Start the score with the log of the prior probability for that label, which is the second element of the pair (prior.second). This represents the initial log probability of the label before considering the input words, and we will add to this score based on the likelihood of the input words given that label.

        for (const string &word : words) // Iterate through each word in the input vector of words. For each word, we will calculate the likelihood of that word given the current label using the model's word counts and total word counts, applying Laplace smoothing to handle cases where the word may not have been seen in examples with that label during training. We will add the log of this likelihood to the score for the current label.
        {
            int count = model.countWord(label, word); // Get the count of how many times the current word has been seen in examples with the current label from the model. This count will be used to calculate the likelihood of the word given the label, which will be added to the score for that label.
            double total = static_cast<double>(model.totalForLabel(label)) + vocabSize; // Get the total number of words seen in examples with the current label from the model, and add the vocabulary size to it for Laplace smoothing. This total will be used as the denominator when calculating the likelihood of the word given the label, which helps to handle cases where the word may not have been seen in examples with that label during training by giving it a small non-zero probability.
            double probability = (count + 1.0) / total; // Calculate the likelihood of the current word given the current label using Laplace smoothing. We add 1 to the count to ensure that we do not have a zero probability for words that were not seen in examples with that label during training, and we divide by the total (which includes the vocabulary size) to get a valid probability value.
            score += log(probability); // Add the log of the likelihood of the current word given the current label to the score for that label. This allows us to combine the prior probability with the likelihood of the input words to calculate the overall log probability score for that label based on the input words and the trained SentimentModel.
        }

        scores.addScore(label, score); // After calculating the score for the current label based on the prior and the likelihood of the input words, we add that score to the ScoreBoard using the addScore function, which stores the score for that label in the logScores map. This allows us to keep track of the scores for each label, which will be used later to determine the best label and calculate confidence during prediction.
    }

    return scores;
}

vector<KeywordScore> KeywordFinder::findKeywords(const SentimentModel &model,
                                                 const vector<string> &words) const // Identify the most influential keywords in the input text based on their contribution to the sentiment prediction. This function calculates a polarity score for each word in the input vector of words by comparing its likelihood under the "positive" and "negative" labels in the model. It then sorts the words by their absolute polarity scores and returns a list of the top keywords along with their scores and assigned polarities (positive, negative, or neutral) based on predefined thresholds.
{
    vector<KeywordScore> keywords; // Initialize a vector to hold the identified keywords along with their scores and polarities. This vector will be populated with the most influential keywords from the input text based on their contribution to the sentiment prediction, and it will be returned at the end of the function.
    map<string, double> polarityScores; // This map will hold the calculated polarity scores for each word in the input vector of words. The key is the word, and the value is the polarity score, which represents how strongly that word contributes to a positive or negative sentiment based on the model's learned information from the training data.
    size_t vocabSize = max(static_cast<size_t>(1), model.vocabSize()); // Get the size of the vocabulary from the model, which is the number of unique words that the model has seen during training. We use max with 1 to ensure that we do not have a vocabulary size of zero, which would cause issues when calculating probabilities with Laplace smoothing.

    double positiveTotal = static_cast<double>(model.totalForLabel("positive")) + vocabSize; // Get the total number of words seen in examples with the "positive" label from the model, and add the vocabulary size to it for Laplace smoothing. This total will be used as the denominator when calculating the likelihood of each word given the "positive" label, which helps to handle cases where a word may not have been seen in examples with that label during training by giving it a small non-zero probability. This is necessary for calculating the polarity scores for each word based on their contribution to the positive sentiment.
    double negativeTotal = static_cast<double>(model.totalForLabel("negative")) + vocabSize; // Get the total number of words seen in examples with the "negative" label from the model, and add the vocabulary size to it for Laplace smoothing. This total will be used as the denominator when calculating the likelihood of each word given the "negative" label, which helps to handle cases where a word may not have been seen in examples with that label during training by giving it a small non-zero probability. This is necessary for calculating the polarity scores for each word based on their contribution to the negative sentiment.

    for (const string &word : words) // Iterate through each word in the input vector of words. For each word, we will calculate a polarity score by comparing its likelihood under the "positive" and "negative" labels in the model. We will skip any words that contain underscores, as those are special tokens generated by our tokenizer for negations and word pairs, and we want to focus on individual words for identifying influential keywords.
    {
        if (word.find('_') != string::npos) continue; // Skip words that contain underscores, as those are special tokens generated by our tokenizer for negations and word pairs. We want to focus on individual words for identifying influential keywords, so we will not calculate polarity scores for these special tokens.

        double positiveScore = log((model.countWord("positive", word) + 1.0) / positiveTotal); // Calculate the log likelihood of the current word given the "positive" label using Laplace smoothing. We add 1 to the count of the word for the "positive" label to ensure that we do not have a zero probability for words that were not seen in examples with that label during training, and we divide by the positiveTotal (which includes the vocabulary size) to get a valid probability value. We then take the log of this probability to get the positive score for that word, which represents how strongly that word contributes to a positive sentiment based on the model's learned information from the training data.
        double negativeScore = log((model.countWord("negative", word) + 1.0) / negativeTotal); // same but for negative label
        polarityScores[word] = positiveScore - negativeScore; // Calculate the polarity score for the current word by taking the difference between its positive score and negative score. A positive polarity score indicates that the word contributes more to a positive sentiment, while a negative polarity score indicates that the word contributes more to a negative sentiment. This polarity score will be used to identify the most influential keywords in the input text based on their contribution to the sentiment prediction.
    }

    vector<pair<string, double>> sorted(polarityScores.begin(), polarityScores.end()); // Create a vector of pairs from the polarityScores map to allow for sorting. Each pair consists of a word and its corresponding polarity score. We will sort this vector by the absolute value of the polarity scores to identify the most influential keywords, regardless of whether they contribute to a positive or negative sentiment.
    sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) { // Sort the vector of pairs by the absolute value of the polarity scores in descending order. This allows us to identify the most influential keywords based on their contribution to the sentiment prediction, regardless of whether they contribute to a positive or negative sentiment. The lambda function compares the absolute values of the polarity scores for two pairs and returns true if the first pair has a higher absolute score than the second pair, which results in sorting the vector in descending order of absolute polarity scores.
        return abs(a.second) > abs(b.second); // Compare the absolute values of the polarity scores for two pairs (a and b) and return true if the absolute score of pair a is greater than the absolute score of pair b. This will sort the vector in descending order of absolute polarity scores, allowing us to identify the most influential keywords based on their contribution to the sentiment prediction, regardless of whether they contribute to a positive or negative sentiment.
    });

    for (size_t i = 0; i < min(static_cast<size_t>(6), sorted.size()); i++) // Iterate through the sorted vector of pairs to identify the top keywords based on their absolute polarity scores. We will take the top 6 keywords (or fewer if there are not enough words) and create KeywordScore objects for each of them, which will include the word, its polarity score, and its assigned polarity (positive, negative, or neutral) based on predefined thresholds. These KeywordScore objects will be added to the keywords vector, which will be returned at the end of the function.
    {
        KeywordScore keyword; // Create a KeywordScore object to hold the information for the current keyword, including the word itself, its polarity score, and its assigned polarity (positive, negative, or neutral) based on predefined thresholds. We will populate this object with the relevant information for the current keyword and add it to the keywords vector.
        keyword.word = sorted[i].first; // Set the word for the current keyword to be the first element of the pair in the sorted vector, which is the word itself. This allows us to keep track of which word is associated with the calculated polarity score and assigned polarity for that keyword.
        keyword.score = sorted[i].second; // Set the score for the current keyword to be the second element of the pair in the sorted vector, which is the polarity score that we calculated for that word. This score represents how strongly that word contributes to a positive or negative sentiment based on the model's learned information from the training data.
        keyword.polarity = (keyword.score > 0.5) ? "pos" // Assign the polarity for the current keyword based on its polarity score using predefined thresholds. If the polarity score is greater than 0.5, we assign it a polarity of "pos" (positive). If the polarity score is less than -0.5, we assign it a polarity of "neg" (negative). If the polarity score is between -0.5 and 0.5, we assign it a polarity of "neu" (neutral). This allows us to categorize the keywords based on their contribution to the sentiment prediction, which can provide insights into which words are driving the positive or negative sentiment in the input text.
                         : (keyword.score < -0.5) ? "neg" // If the polarity score is less than -0.5, we assign it a polarity of "neg" (negative).
                         : "neu"; // If the polarity score is between -0.5 and 0.5, we assign it a polarity of "neu" (neutral).
        keywords.push_back(keyword); // Add the current KeywordScore object to the keywords vector, which will hold the identified keywords along with their scores and assigned polarities. This allows us to keep track of the most influential keywords in the input text based on their contribution to the sentiment prediction, and we will return this vector at the end of the function.
    }

    return keywords;
}

void HistoryManager::add(const string &text, const string &label, double confidence) // Add a new entry to the history of predictions. This function takes the input text, the predicted label, and the confidence of the prediction, and it creates a HistoryRow object that includes this information along with a timestamp of when the prediction was made. The HistoryRow is then added to the rows vector, which maintains a history of all predictions made by the SentimentSystem.
{
    time_t now = time(nullptr); // Get the current time as a time_t object, which represents the number of seconds since the Unix epoch (January 1, 1970). This timestamp will be used to record when the prediction was made and will be included in the HistoryRow for this entry in the history.
    string timestamp = ctime(&now); // Convert the time_t object to a human-readable string format using ctime, which returns a string representation of the time. This timestamp will be included in the HistoryRow for this entry in the history to indicate when the prediction was made. The ctime function adds a newline character at the end of the string, so we will remove that before storing it in the HistoryRow.
    if (!timestamp.empty()) timestamp.pop_back(); // Remove the trailing newline character from the timestamp string if it is not empty. This ensures that the timestamp is stored in a clean format without any extra newline characters, which can be important for displaying the history or exporting it to a CSV file.

    HistoryRow row; // Create a new HistoryRow object to hold the information for this entry in the history, including the timestamp, predicted label, confidence, and input text. We will populate this object with the relevant information for this prediction and add it to the rows vector, which maintains a history of all predictions made by the SentimentSystem.
    row.time = timestamp;
    row.label = label;
    row.score = confidence;
    row.text = text;
    rows.push_back(row);
}

vector<HistoryRow>& HistoryManager::getRows() // Return a reference to the vector of HistoryRow objects that represents the history of predictions. This allows other parts of the program, such as the SentimentSystem or the HistoryExporter, to access and manipulate the history of predictions as needed, such as displaying it to the user or exporting it to a CSV file.
{
    return rows;
}

double AccuracyTester::test(const string &filename, SentimentSystem &system) const // Test the accuracy of the SentimentSystem using a dataset from a specified CSV file. This function reads the test data from the file, processes each line to extract the text and label, cleans the label, and then uses the SentimentSystem to predict the sentiment for each text. It compares the predicted label with the actual label and keeps track of how many predictions were correct out of the total number of valid examples. Finally, it calculates and returns the accuracy as a percentage.
{
    ifstream file(filename); // Open the specified CSV file for reading, which is expected to contain the test data with text and corresponding labels. If the file cannot be opened, we return an accuracy of 0.0 since we cannot perform the test without access to the test data.
    if (!file.is_open()) return 0.0; // Check if the file was successfully opened. If not, we return an accuracy of 0.0 since we cannot perform the test without access to the test data.

    string line; // Read the first line of the CSV file, which is expected to be the header, and skip it. This allows us to start processing the actual test examples from the second line onward. The header typically contains column names and is not part of the test data, so we skip it to avoid including it in the accuracy calculation.
    getline(file, line); // skip CSV header

    int correct = 0; // This variable will count the number of correct predictions made by the SentimentSystem. We will increment this count each time the predicted label matches the actual label from the test data, which will allow us to calculate the accuracy of the SentimentSystem at the end of the function.
    int total = 0;

    while (getline(file, line)) // Read each line from the CSV file, which represents a test example. For each line, we parse it to extract the text and label, clean the label, check if it's valid, and then use the SentimentSystem to predict the sentiment for the text. We compare the predicted label with the actual label from the test data, and if they match, we increment the count of correct predictions. We also increment the total count of valid examples processed. This allows us to calculate the accuracy of the SentimentSystem based on how many predictions were correct out of the total number of valid examples.
    {
        TrainingExample example = csvReader.parseLine(line); // take the line from the CSV file to extract the text and label, creating a TrainingExample object that holds this information. The parseLine function handles the parsing of the CSV line, including dealing with quoted text that may contain commas, to ensure that we correctly extract the text and label for each test example.
        example.label = labelHelper.clean(example.label); // Clean the label from the test example using the LabelHelper, which converts it to lowercase and removes any leading or trailing whitespace and carriage returns. This ensures that the label is in a consistent format for prediction, allowing the SentimentSystem to make accurate predictions based on the cleaned labels from the test data.

        if (!labelHelper.isValid(example.label)) continue; // Check if the cleaned label is valid using the LabelHelper. If the label is not valid (i.e., it is not "positive", "negative", or "neutral"), we skip this test example and move on to the next line in the CSV file. This ensures that we only test the model on examples with valid sentiment labels, which allows us to calculate an accurate measure of the model's performance based on valid test cases.

        PredictResult result = system.predictFull(example.text, false); // Use the SentimentSystem to predict the sentiment for the text from the test example. We call the predictFull function with the text and set saveToHistory to false, since we do not want to save these predictions to the history as they are part of the accuracy testing process. The predictFull function will return a PredictResult that includes the predicted label, confidence, and other information about the prediction.
        if (result.label == example.label) // Compare the predicted label from the PredictResult with the actual label from the test example. If they match, it means that the SentimentSystem made a correct prediction for this test example, and we increment the count of correct predictions. This allows us to keep track of how many predictions were correct out of the total number of valid examples processed, which will be used to calculate the accuracy of the SentimentSystem at the end of the function.
            correct++;

        total++;
    }

    if (total == 0) return 0.0; // If there were no valid examples processed (like, total is zero), we return an accuracy of 0.0 to avoid division by zero. This means that if the test data did not contain any valid examples with proper labels, we consider the accuracy to be 0 since we cannot evaluate the model's performance without valid test cases.
    return static_cast<double>(correct) / total * 100.0; // Finally, we calculate the accuracy of the SentimentSystem by dividing the count of correct predictions by the total number of valid examples processed, and then multiplying by 100 to convert it to a percentage. This accuracy value represents how well the SentimentSystem is performing on the test data, indicating the percentage of test examples for which the model's predicted label matches the actual label from the test data.
}

void HistoryExporter::exportToCsv(const string &path, const vector<HistoryRow> &rows) const // Export the history of predictions to a CSV file at the specified path. This function takes a vector of HistoryRow objects, which represent the history of predictions made by the SentimentSystem, and writes them to a CSV file with columns for time, label, confidence, and text. This allows users to save and review the history of predictions in a structured format that can be easily opened in spreadsheet software or used for further analysis.
{
    ofstream file(path); // Open the specified file for writing. If the file cannot be opened, we simply return without doing anything, as we cannot export the history without access to the file.
    if (!file.is_open()) return; // Check if the file was successfully opened. If
    file << "Time,Label,Confidence,Text\n"; // Write the header row to the CSV file, which includes the column names "Time", "Label", "Confidence", and "Text". This header will help to identify the contents of each column when the CSV file is opened in spreadsheet software or used for analysis.

    for (const HistoryRow &row : rows) // Iterate through each HistoryRow in the rows vector, which represents the history of predictions. For each row, we write its contents to the CSV file in a structured format, with the time, label, confidence, and text separated by commas. This allows us to export the history of predictions in a format that can be easily reviewed and analyzed.
        file << row.time << "," << row.label << "," << row.score << "," << row.text << "\n"; // Write the contents of the current HistoryRow to the CSV file, with the time, label, confidence (score), and text separated by commas. We also add a newline character at the end of each row to ensure that each entry is on a separate line in the CSV file. This structured format allows us to easily review and analyze the history of predictions when the CSV file is opened in spreadsheet software or used for further analysis.
}

void SentimentSystem::train(const string &filename) // Train the SentimentSystem using a dataset from a specified CSV file. This function uses the ModelTrainer to train the SentimentModel based on the data in the CSV file, which typically contains text and corresponding sentiment labels. After training, it outputs a report indicating how many rows were used for training and the size of the vocabulary learned by the model. This allows users to understand the extent of the training process and the richness of the model's vocabulary after training.
{
    TrainingReport report = trainer.train(model, filename); // Use the ModelTrainer to train the SentimentModel based on the data in the specified CSV file. The train function will read the training data from the file, process it, and update the SentimentModel with the learned information. It returns a TrainingReport that includes details about how many rows were used for training, which can provide insights into the extent of the training process.
    cout << "[sentiment] Trained on " << report.rowsUsed // Output a report indicating how many rows were used for training and the size of the vocabulary learned by the model. This information can help users understand the extent of the training process and the richness of the model's vocabulary after training, which can be important for assessing the model's performance and generalization capabilities.
         << " rows | vocab=" << model.vocabSize() << endl ; // Output the size of the vocabulary learned by the model after training, which is obtained by calling the vocabSize function on the SentimentModel. The vocabulary size indicates how many unique words the model has seen during training, which can be an important factor in the model's ability to make accurate predictions on new input text. A larger vocabulary typically allows the model to capture more nuances in language and improve its performance on a wider range of inputs.
}

PredictResult SentimentSystem::predictFull(const string &text, bool saveToHistory) // Predict the sentiment of the given input text and return a PredictResult that includes the predicted label, confidence, OOV ratio, and identified keywords. This function first checks if the model is trained, then tokenizes the input text, calculates the OOV ratio, scores the text using the BayesPredictor, determines the best label and confidence, identifies influential keywords using the KeywordFinder, and finally returns a PredictResult with all this information. If saveToHistory is true, it also saves the prediction to the history for later review.
{
    PredictResult result;
    result.label = "neutral";
    result.confidence = 50.0;
    result.isUnknown = false;
    result.oovRatio = 0.0;

    if (!model.isTrained()) // Check if the SentimentModel is trained before attempting to make a prediction. If the model is not trained, we set the label in the PredictResult to "Model not trained" and return the result immediately, since we cannot make a valid prediction without a trained model. This is a safeguard to ensure that we do not attempt to predict sentiment using an untrained model, which would likely yield unreliable results.
    {
        result.label = "Model not trained";
        return result;
    }

    vector<string> words = tokenizer.tokenize(text); // Tokenize the input text using the Tokenizer, which will break the text into individual words (tokens) that can be processed by the model. The tokenize function will also apply any necessary cleaning and preprocessing to the text, such as handling negations and word pairs, to ensure that the input is in a suitable format for prediction. The resulting vector of words will be used for calculating the OOV ratio, scoring the text, and identifying influential keywords.
    if (words.empty()) return result; // If the tokenization results in an empty vector of words, we return the default PredictResult with a label of "neutral" and a confidence of 50.0. This is a safeguard to handle cases where the input text may not contain any valid tokens after tokenization, which would make it impossible to make a meaningful prediction. By returning a default result in this case, we can avoid errors and provide a reasonable fallback for empty or invalid input.

    result.oovRatio = oovChecker.calculateRatio(model, words); // Calculate the OutOfVocabulary (OOV) ratio for the input words using the OovChecker. The OOV ratio represents the proportion of words in the input text that were not seen during training and are not present in the model's vocabulary. This ratio can provide insights into how much of the input text is unfamiliar to the model, which can affect the confidence and reliability of the sentiment prediction. A high OOV ratio may indicate that the model is less confident in its prediction due to a lack of familiarity with the input words.

    ScoreBoard scores = predictor.scoreText(model, words); // Use the BayesPredictor to calculate the scores for each label based on the input words and the trained SentimentModel. The scoreText function will return a ScoreBoard containing the log probabilities for each label, which can be used to determine the best label and calculate confidence during prediction. The scores in the ScoreBoard will be used to set the predicted label and confidence in the PredictResult.
    result.label = scores.bestLabel();
    result.confidence = scores.confidence();
    if (result.confidence < 40.0) // If the confidence of the prediction is below a certain threshold (40.0), we set the predicted label to "neutral" to indicate that the model is not confident enough in its prediction to assign a specific sentiment label. This is a way to handle cases where the model's prediction is uncertain, and it allows us to provide a more cautious result by categorizing it as neutral rather than potentially making an incorrect positive or negative prediction.
        result.label = "neutral";

    result.keywords = keywordFinder.findKeywords(model, words); // Use the KeywordFinder to identify the most influential keywords in the input text based on their contribution to the sentiment prediction. The findKeywords function will return a vector of KeywordScore objects that include the word, its polarity score, and its assigned polarity (positive, negative, or neutral) based on predefined thresholds. These keywords can provide insights into which words in the input text are driving the positive or negative sentiment in the prediction.
    result.isUnknown = (result.oovRatio > 0.6 && result.confidence < 70.0); // This indicates that the model considers this prediction to be uncertain or potentially unreliable due to a high proportion of unfamiliar words and a relatively low confidence score. This flag can be used by the caller to handle such cases differently, such as by providing a warning to the user or treating the result with caution.

    if (saveToHistory) // If the saveToHistory flag is true, we add this prediction to the history using the HistoryManager. We call the add function with the input text, the predicted label, and the confidence of the prediction, which will create a new entry in the history of predictions. This allows us to keep a record of all predictions made by the SentimentSystem, which can be useful for later review or analysis.
        history.add(text, result.label, result.confidence);
    return result;
}

double SentimentSystem::testAccuracy(const string &filename) // Test the accuracy of the SentimentSystem using a dataset from a specified CSV file. This function delegates the testing process to the AccuracyTester, which will read the test data from the file, make predictions using the SentimentSystem, and calculate the accuracy based on how many predictions were correct compared to the actual labels in the test data. The resulting accuracy is returned as a percentage.
{
    return accuracyTester.test(filename, *this);
}

vector<HistoryRow>& SentimentSystem::getHistory() // Return a reference to the vector of HistoryRow objects that represents the history of predictions made by the SentimentSystem. This allows other parts of the program, such as the SentimentSystem itself or the HistoryExporter, to access and manipulate the history of predictions as needed, such as displaying it to the user or exporting it to a CSV file.
{
    return history.getRows();
}

size_t SentimentSystem::getVocabSize() const // Return the size of the vocabulary learned by the SentimentModel. This function calls the vocabSize function on the SentimentModel, which returns the number of unique words that the model has seen during training. The vocabulary size can provide insights into the richness of the model's learned information and its ability to make accurate predictions on a wide range of input text.
{
    return model.vocabSize();
}

void SentimentSystem::exportHistory(const string &path) // Export the history of predictions to a CSV file at the specified path. This function delegates the export process to the HistoryExporter, which will take the vector of HistoryRow objects from the HistoryManager and write them to a CSV file with columns for time, label, confidence, and text. This allows users to save and review the history of predictions in a structured format that can be easily opened in spreadsheet software or used for further analysis.
{
    historyExporter.exportToCsv(path, history.getRows());
}

static SentimentSystem sentimentSystem; // Create a static instance of the SentimentSystem that will be used by the functions defined below to perform training, prediction, accuracy testing, and history management. This allows us to have a single instance of the SentimentSystem that can be accessed and manipulated through these functions, providing a convenient interface for interacting with the sentiment analysis system.

void trainModel(const string &filename) // Train the SentimentSystem using a dataset from a specified CSV file. This function calls the train method of the SentimentSystem with the given filename, which will read the training data from the file and update the SentimentModel with the learned information. After training, it will output a report indicating how many rows were used for training and the size of the vocabulary learned by the model. This allows users to understand the extent of the training process and the richness of the model's vocabulary after training.
{
    sentimentSystem.train(filename);
}

double testAccuracy(const string &filename) // Test the accuracy of the SentimentSystem using a dataset from a specified CSV file. This function calls the testAccuracy method of the SentimentSystem with the given filename, which will read the test data from the file, make predictions using the SentimentSystem, and calculate the accuracy based on how many predictions were correct compared to the actual labels in the test data. The resulting accuracy is returned as a percentage, allowing users to evaluate the performance of the SentimentSystem on the test dataset.
{
    return sentimentSystem.testAccuracy(filename);
}

pair<string, double> predict(const string &text) // Predict the sentiment of the given input text and return a pair containing the predicted label and confidence. This function calls the predictFull method of the SentimentSystem with the input text and saveToHistory set to true, which will perform the prediction and also save it to the history. The predictFull method returns a PredictResult, from which we extract the predicted label and confidence to return as a pair. This provides a simple interface for making predictions while also keeping a record of them in the history.
{
    PredictResult result = sentimentSystem.predictFull(text, true);
    return {result.label, result.confidence};
}

PredictResult predictFull(const string &text) // Predict the sentiment of the given input text and return a PredictResult that includes the predicted label, confidence, OOV ratio, and identified keywords. This function calls the predictFull method of the SentimentSystem with the input text and saveToHistory set to true, which will perform the prediction and also save it to the history. The predictFull method will return a PredictResult with all the relevant information about the prediction, which we then return from this function. This allows users to get a detailed result from the prediction while also keeping a record of it in the history.
{
    return sentimentSystem.predictFull(text, true);
}

vector<HistoryRow>& getHistory() // Return a reference to the vector of HistoryRow objects that represents the history of predictions made by the SentimentSystem. This function calls the getHistory method of the SentimentSystem, which returns a reference to the vector of HistoryRow objects from the HistoryManager. This allows users to access and manipulate the history of predictions as needed, such as displaying it to the user or exporting it to a CSV file.
{
    return sentimentSystem.getHistory();
}

size_t getVocabSize() // Return the size of the vocabulary learned by the SentimentModel. This function calls the getVocabSize method of the SentimentSystem, which in turn calls the vocabSize function on the SentimentModel to return the number of unique words that the model has seen during training. The vocabulary size can provide insights into the richness of the model's learned information and its ability to make accurate predictions on a wide range of input text.
{
    return sentimentSystem.getVocabSize();
}

void exportHistory(const string &path) // Export the history of predictions to a CSV file at the specified path. This function calls the exportHistory method of the SentimentSystem with the given path, which will delegate the export process to the HistoryExporter. The HistoryExporter will take the vector of HistoryRow objects from the HistoryManager and write them to a CSV file with columns for time, label, confidence, and text. This allows users to save and review the history of predictions in a structured format that can be easily opened in spreadsheet software or used for further analysis.
{
    sentimentSystem.exportHistory(path);
}