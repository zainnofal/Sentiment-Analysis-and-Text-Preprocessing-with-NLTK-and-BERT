Sentiment Analysis and Text Preprocessing with NLTK, BERT, and Flask
This project performs sentiment analysis on a dataset of tweets (tweet_emotions.csv) using a mix of NLP techniques and machine learning models, and includes deployment with Flask.

Key Components:

Text Preprocessing: Utilizes NLTK for tokenization, stopword removal, and stemming to clean and prepare tweet text.

Data Visualization: Creates word clouds and bar plots to visualize frequent words and insights from the preprocessed text.

Logistic Regression Model: Builds and trains a logistic regression model with TF-IDF vectorization for sentiment classification.

BERT-based Model: Employs a fine-tuned DistilBERT model for advanced sentiment prediction.

Model Persistence: Saves trained models using joblib for future use.

Flask Deployment: Deploys the sentiment analysis functionality as a web service using Flask, enabling real-time predictions via a web interface.

This project integrates text processing, visualization, advanced sentiment analysis, and web deployment, providing a complete solution for analyzing and predicting tweet sentiments.






