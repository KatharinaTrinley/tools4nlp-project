import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, classification_report

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentAnalyzer:
    """
    A class for sentiment analysis using VADER and SpacyTextBlob, with visualization of confusion matrices.
    """

    def __init__(self, pre_df):
        """
        Initialize the SentimentAnalyzer object.

        Parameters:
        - pre_df (pd.DataFrame): DataFrame containing the 'tweet' and 'sentiment' columns.
        """
        self.pre_df = pre_df
        self.sid = SentimentIntensityAnalyzer()
        self.nlp = spacy.load("en_core_web_lg")
        self.nlp.add_pipe("spacytextblob")

    def vader_sentiment_analysis(self):
        """
        Perform sentiment analysis using VADER on the 'tweet' column and store the results in the DataFrame.
        """
        self.pre_df["vader_compound"] = self.pre_df["tweet"].apply(
            lambda tweet: self.sid.polarity_scores(str(tweet))
        )

        self.pre_df["positive"] = self.pre_df["vader_compound"].apply(
            lambda scores: scores["pos"]
        )
        self.pre_df["neutral"] = self.pre_df["vader_compound"].apply(
            lambda scores: scores["neu"]
        )
        self.pre_df["negative"] = self.pre_df["vader_compound"].apply(
            lambda scores: scores["neg"]
        )

        self.pre_df["vader_prediction"] = self.pre_df[
            ["positive", "neutral", "negative"]
        ].idxmax(axis=1)

        self.pre_df.drop(columns=["positive", "neutral", "negative"], inplace=True)

    def spacy_sentiment_analysis(self):
        """
        Perform sentiment analysis using SpacyTextBlob on the 'tweet' column and store the results in the DataFrame.
        """
        self.pre_df["spacy_compound"] = self.pre_df["tweet"].apply(
            lambda tweet: self.nlp(tweet)._.polarity
        )

        self.pre_df["spacy_prediction"] = self.pre_df["spacy_compound"].apply(
            lambda score: (
                "negative" if score < 0 else ("neutral" if score == 0 else "positive")
            )
        )

    def hf_sentiment_analysis(self):
        """
        Perform sentiment analysis using Hugging Face on the 'tweet' column and store the results in the DataFrame.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

        def classify_tweet(tweet):
            inputs = tokenizer(tweet, return_tensors="pt")
            logits = model(**inputs).logits
            label = logits.argmax().item()
            
            label_words = {0.0: "negative", 1.0: "neutral", 2.0: "positive"}
            label_word = label_words.get(float(label), "Unknown")
            return label_word

        self.pre_df["hf_sentiment"] = self.pre_df["tweet"].apply(classify_tweet)

    def visualize_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Visualize the confusion matrix using seaborn.

        Parameters:
        - y_true (pd.Series): True sentiment values.
        - y_pred (pd.Series): Predicted sentiment values.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["negative", "neutral", "positive"],
            yticklabels=["negative", "neutral", "positive"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        if save_path:
            plt.savefig(save_path, format="png")
        else:
            plt.show()

    def analyze_and_visualize(self):
        """
        Perform sentiment analysis using VADER and SpacyTextBlob, and visualize confusion matrices.
        """
        self.vader_sentiment_analysis()
        y_true_vader = self.pre_df["sentiment"]
        y_pred_vader = self.pre_df["vader_prediction"]

        cm_vader = confusion_matrix(y_true_vader, y_pred_vader)
        report_vader = classification_report(y_true_vader, y_pred_vader)

        print("\nVADER Sentiment Analysis:")
        print("\nConfusion Matrix:")
        print(cm_vader)
        print("\nClassification Report:")
        print(report_vader)

        self.visualize_confusion_matrix(
            y_true_vader, y_pred_vader, save_path="vader_cm.png"
        )

        self.spacy_sentiment_analysis()
        y_true_spacy = self.pre_df["sentiment"]
        y_pred_spacy = self.pre_df["spacy_prediction"]

        cm_spacy = confusion_matrix(y_true_spacy, y_pred_spacy)
        report_spacy = classification_report(y_true_spacy, y_pred_spacy)

        print("\nSpacyTextBlob Sentiment Analysis:")
        print("\nConfusion Matrix:")
        print(cm_spacy)
        print("\nClassification Report:")
        print(report_spacy)

        self.visualize_confusion_matrix(
            y_true_spacy, y_pred_spacy, save_path="spacy_cm.png"
        )

        self.hf_sentiment_analysis()
        y_true_hf = self.pre_df["sentiment"]
        y_pred_hf = self.pre_df["hf_sentiment"]

        cm_hf = confusion_matrix(y_true_hf, y_pred_hf)
        report_hf = classification_report(y_true_hf, y_pred_hf)

        print("\nHugging Face Sentiment Analysis:")
        print("\nConfusion Matrix:")
        print(cm_hf)
        print("\nClassification Report:")
        print(report_hf)

        self.visualize_confusion_matrix(
            y_true_hf, y_pred_hf, save_path="roberta_cm.png"
        )
