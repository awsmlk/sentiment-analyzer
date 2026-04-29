# Sentiment Analyzer (C++ / OOP Project)

This project implements a sentiment analysis system in C++ using object-oriented programming and a custom Naive Bayes model. It can classify text as positive, negative, or neutral from a CSV dataset.

## Features

- Sentiment classification with confidence scores
- Naive Bayes model built from scratch
- Tokenization, negation handling, and simple bigrams
- CSV-based training data
- Terminal UI
- Local HTTP server for the web interface
- Browser UI in `web/index.html`
- History export to CSV

## Project Structure

```text
sentiment-analyzer/
├── include/          Header files for all C++ classes
├── src/              C++ implementation files
├── web/              Browser frontend
├── data/             Training dataset
├── history/          Exported prediction history
├── .vscode/          VS Code build/debug config
├── .gitignore
└── README.md
```

Important files:

```text
include/sentiment.h         Public sentiment API
include/sentiment_system.h  Main class that connects the engine pieces
src/sentiment.cpp           Sentiment engine implementations
src/ui.cpp                  Terminal UI
src/server.cpp              HTTP backend for the web app
web/index.html              Web interface
data/dataset.csv            Training data
```

## Build

Terminal app:

```bash
g++ -std=c++17 -O2 -Iinclude -o sentiment src/main.cpp src/ui.cpp src/sentiment.cpp
```

Web backend server:

```bash
g++ -std=c++17 -O2 -Iinclude -o server src/server.cpp src/sentiment.cpp -pthread
```

## Run

Terminal app:

```bash
./sentiment
```

Web app:

```bash
./server
```

Then open:

```text
web/index.html
```

The web UI expects the backend at `http://localhost:8080`.

## Dataset Format

```csv
text,label
I love this product,positive
This is terrible,negative
It is okay,neutral
```

## Notes

The project is intentionally lightweight and educational. It demonstrates OOP design, file handling, tokenization, model training, prediction, and a simple local server without external C++ libraries.
