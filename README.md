# Fake News Detection Project

## Overview

## Overview
The goal of this project is to build a machine learning classifier that can predict whether a news article is fake or not based on its features. The project involves preprocessing data, training multiple classification models, and evaluating their performance using 5-fold cross-validation.

### Trained Models:
- **XGBoost**
- **Gradient Boosting (SGBoost)**
- **Logistic Regression**
- **Random Forest**
- **Decision Tree**

### Evaluation Metrics:
The models are compared based on the following evaluation metrics:
- **Precision**
- **Recall**
- **F1 Score**
- **Accuracy**
- **AUROC (Area Under the Receiver Operating Characteristic Curve)**

5-fold cross-validation is used to ensure the robustness and generalizability of the models.

The models are trained on the dataset, and the evaluation metrics for each classifier are recorded for comparison. The results will help identify the most effective model for detecting fake news articles.

---
## Main Functions

### Feature Extraction Functions
- **`calculate_total_shares`**: Aggregates total shares per news article.
- **`mark_top_img_presence`**: Marks if the main image (top_img) is present for each news article.
- **`count_images`**: Counts the number of images in each news article.
- **`count_unique_user_shares`**: Calculates the number of unique users who shared each news article.

### Text Analysis Functions
- **`extract_pos_features`**: Extracts Part-of-Speech (POS) features from the text, such as nouns, verbs, adjectives, etc.
- **`extract_named_entities`**: Uses TextBlob to extract named entities from the text.
- **`count_punctuations`**: Counts the number of punctuation marks in the text.

### Sentiment Analysis Functions
- **`analyze_sentiment_vader`**: Uses NLTK’s VADER sentiment analyzer to calculate sentiment scores for the text.
- **`get_sentiment_for_titles`**: Calculates sentiment scores specifically for news article titles.
- **`get_sentiment_for_text`**: Calculates sentiment scores for the main body of text.

### Emotion Analysis Functions
- **`emotion_NRC`**: Computes emotion vectors for each news article text or title using the NRC Emotion Lexicon.
- **`NRC_dict`**: Reads the emotion lexicon file to create a dictionary mapping words to emotions.
- **`getEmotionVector`**: Given a text, this function returns a vector of emotions based on words present in the text and their associations in the NRC lexicon.

### Graph Analysis Functions
- **`jaccard_similarity`**: Computes Jaccard similarity between two nodes in a graph.
- **`average_node_similarity`**: Calculates the average Jaccard similarity for users who engaged with each news article.

### User Analysis Functions
- **`get_follower_counts`**: Calculates follower counts for each user from the user-to-user relationship data.
- **`get_unique_users_per_news`**: Identifies unique users who interacted with each news article.
- **`get_avg_followers_per_news`**: Computes the average number of followers for users interacting with each news article.


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



 


