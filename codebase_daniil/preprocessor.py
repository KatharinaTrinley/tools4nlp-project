import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from textblob import TextBlob
import matplotlib.pyplot as plt

class TextDataProcessor:
    def __init__(self, data):
        """
        Initialize the TextDataProcessor with input data.

        Parameters:
        - data: str or pd.DataFrame
            Input data, either a string or a DataFrame.
        """
        if isinstance(data, str):
            self.data = pd.DataFrame({3: [data]})
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise ValueError("Unsupported data type. Use a string or a DataFrame.")

    def check_missing_values(self):
        """
        Check and print missing values in the data.
        """
        print("Missing Values:")
        print(self.data.isnull().sum())

    def fill_missing_values(self, strategy='mean'):
        """
        Fill missing values in the data using a chosen strategy.

        Parameters:
        - strategy: str, default='mean'
            Strategy to fill missing values.
        """
        self.data.dropna(inplace=True)
        print(f"Missing Values after filling with {strategy}:")
        print(self.data.isnull().sum())

    def modify_data_types(self):
        """
        Modify data types in the DataFrame.
        """
        self.data[0] = pd.to_numeric(self.data[0], errors='coerce')
        self.data[1] = self.data[1].astype(str)
        self.data[2] = self.data[2].astype(str)
        self.data[3] = self.data[3].astype(str)

    def lowercase_text(self):
        """
        Lowercase all text in a specific column (Column 3).
        """
        self.data[3] = self.data[3].str.lower()

    def remove_ascii_characters(self):
        """
        Remove ASCII characters from a specific column (Column 3).
        """
        self.data[3] = self.data[3].apply(lambda x: ''.join([i if ord(i) < 128 else ' ' for i in x]))

    def remove_additional_white_spaces(self):
        """
        Remove additional white spaces from a specific column (Column 3).
        """
        self.data[3] = self.data[3].str.replace('\s+', ' ', regex=True)

    def count_vectorize_text(self):
        """
        Perform Count Vectorization on the text in Column 3.
        """
        vectorizer = CountVectorizer()
        text_vectorized = vectorizer.fit_transform(self.data[3])
        text_vectorized_df = pd.DataFrame(text_vectorized.toarray(), columns=vectorizer.get_feature_names_out())
        print("CountVectorized Text:")
        print(text_vectorized_df.shape)

    def group_and_aggregate_data(self):
        """
        Group and aggregate data based on Column 2.
        """
        grouped_data = self.data.groupby(2)[0].mean()
        print("Grouped and Aggregated Data:")
        print(grouped_data)

    def plot_histogram(self):
        """
        Plot a histogram of numeric values in Column 0.
        """
        plt.hist(self.data[0], bins=20, color='blue', alpha=0.7)
        plt.title('Histogram of Numeric Column')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.show()

    def extract_information_using_regex(self):
        """
        Extract information using regex and create a new column 'Extracted'.
        """
        self.data['Extracted'] = self.data[3].apply(lambda x: re.findall(r'\b\w{1,2}\b', str(x)))
        print("Extracted Information using Regex:")
        print(self.data[['Extracted', 3]].head())

    def nltk_frequency_distribution(self):
        """
        Perform NLTK frequency distribution on the text in Column 3.
        """
        nltk.download('punkt')
        self.data['NLTK_Tokens'] = self.data[3].apply(word_tokenize)
        nltk_freq_dist = FreqDist([word for tokens in self.data['NLTK_Tokens'] for word in tokens])
        print("NLTK Frequency Distribution:")
        print(nltk_freq_dist.most_common(10))

    def gensim_word_embedding_similarity(self, word1='is', word2='am'):
        """
        Perform Gensim Word Embedding Similarity between two words.

        Parameters:
        - word1: str, default='is'
            First word for similarity comparison.
        - word2: str, default='am'
            Second word for similarity comparison.
        """
        sentences = [text.split() for text in self.data[3]]
        gensim_model = Word2Vec(sentences, 100, window=5, min_count=1, workers=4)
        similarity = gensim_model.wv.similarity(word1, word2)
        print(f"Gensim Word Embedding Similarity between '{word1}' and '{word2}': {similarity}")

    def textblob_sentiment_analysis(self):
        """
        Perform TextBlob Sentiment Analysis and add a new column 'TextBlob_Sentiment'.
        """
        self.data['TextBlob_Sentiment'] = self.data[3].apply(lambda x: TextBlob(x).sentiment.polarity)
        print("TextBlob Sentiment Analysis:")
        print(self.data[['TextBlob_Sentiment', 3]])