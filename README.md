# News Content Analysis Project

## Overview

This project analyzes news articles and user interactions, creating various features for applications like classification, recommendation, or network analysis. Using tools from natural language processing (NLP), graph analysis, and sentiment analysis, it extracts features from news articles and builds relationships among user interactions to create comprehensive insights.

---

## Features

- **Sentiment Analysis**: Extracts sentiment metrics for news articles, summaries, and titles.
- **Emotion Detection**: Uses the NRC Emotion Lexicon to identify emotional tones within articles.
- **Readability & Structure**: Measures readability scores, punctuation counts, and part-of-speech distribution.
- **Graph Analysis**: Builds user relationships using Jaccard similarity and network features.
- **User Metrics**: Analyzes unique user interactions, total shares, and average follower count per article.

---

## Prerequisites

- Python 3.x
- Install dependencies with:

    ```bash
    pip install -r requirements.txt
    ```

- Download required NLTK data:

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('vader_lexicon')
    ```

---

## Data Files

Place the following data files in the `data/` folder:

- **NewsContent.csv**: Contains main news article content.
- **News_User.csv**: Links users to articles with share counts.
- **User_user.csv**: Lists user-to-user relationships (followers).
- **NRC-Emotion-Lexicon-Wordlevel-v0.92.txt**: Emotion lexicon file for emotion mapping.

---

## Project Structure

```plaintext
project_root/
│
├── data/
│   ├── NewsContent.csv
│   ├── News_User.csv
│   └── User_user.csv
│
├── NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
│
├── src/
│   ├── feature_extraction.py    # Main feature extraction functions
│   ├── sentiment_analysis.py    # Sentiment and emotion analysis functions
│   ├── graph_analysis.py        # User relationship and graph analysis
│   └── main.py                  # Main script to execute feature extraction
│
└── README.md
