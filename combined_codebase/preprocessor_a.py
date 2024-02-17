import pandas as pd
import re
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("vader_lexicon")
nltk.download("stopwords")

x


class TextPreprocessor:
    def __init__(self, file_path):
        """
        Initialize the TextPreprocessor with a CSV file path.

        Parameters:
        - file_path (str): Path to the CSV file containing the data.
        """

        try:
            self.df = pd.read_csv(
                file_path,
                header=None,
                names=["number", "account", "sentiment", "tweet"],
            )
            self.pre_df = self.df.copy()
        except pd.errors.ParserError:
            raise ValueError(
                "Error parsing the CSV file. Make sure it is a CSV format."
            )

    def display_head(self, n=5):
        """
        Display the first few rows of the DataFrame.
        """
        print("Head of the DataFrame:")
        print(self.df.head(n))

    def display_summary_statistics(self):
        """
        Display summary/descriptive statistics of the DataFrame.
        """
        print("\nSummary Statistics:")
        print(self.df.describe())

    def handle_missing_values(self):
        """
        Handle missing values in the 'tweet' column by dropping rows with missing values.
        """
        missing_values = self.pre_df["tweet"].isnull().sum()
        print(f"Number of missing values in 'tweet' column: {missing_values}")
        percent_missing = (missing_values / len(self.pre_df)) * 100
        print(f"Percentage of missing values in 'tweet' column: {percent_missing:.2f}%")

        self.pre_df.dropna(inplace=True)
        missing_values_after_drop = self.pre_df.isnull().sum()
        print(
            f"Number of missing values in 'tweet' column after drop: {missing_values_after_drop['tweet']}"
        )

    def convert_data_types(self):
        """
        Convert data types of columns in the DataFrame.
        """
        self.pre_df["number"] = self.pre_df["number"].astype(int)
        self.pre_df["account"] = self.pre_df["account"].astype(str)
        self.pre_df["sentiment"] = self.pre_df["sentiment"].astype(str)
        self.pre_df["tweet"] = self.pre_df["tweet"].astype(str)
        print("Data types converted successfully.")

    def display_data_types(self):
        """
        Display the current data types of the columns in the DataFrame.
        """
        print("Current Data Types:")
        print(self.pre_df.dtypes)

    def lowercase_text(self):
        """
        Preprocess the 'tweet' column by lowercasing, removing non-ASCII characters,
        emojis, stopwords, stemming, numbers, and punctuation.
        """
        self.pre_df["tweet"] = self.pre_df["tweet"].str.lower()
        self.pre_df["account"] = self.pre_df["account"].str.lower()
        self.pre_df["sentiment"] = self.pre_df["sentiment"].str.lower()
        print("Text lowercased successfully.")

    def remove_non_ascii(self):
        """
        Preprocess the 'tweet' column by lowercasing, removing extra white spaces and \\n markers.
        """
        self.pre_df["tweet"] = self.pre_df["tweet"].str.replace(r"[^\x00-\x7F]+", " ")
        self.pre_df["tweet"] = self.pre_df["tweet"].str.replace(r"\s+", " ")
        self.pre_df["tweet"] = self.pre_df["tweet"].apply(
            lambda x: re.sub(r"\\n\w*", "", str(x))
        )
        print("Non-ASCII characters removed successfully.")

    def remove_emojis(self):
        """
        Preprocess the 'tweet' column by removing emojis.
        """
        self.pre_df["tweet"] = self.pre_df["tweet"].apply(
            lambda x: self._remove_emojis(x)
        )
        print("Emojis removed successfully.")

    def remove_stopwords(self):
        """
        Preprocess the 'tweet' column by removing stop words.
        """
        stop_words_nltk = set(stopwords.words("english"))
        self.pre_df["tweet"] = self.pre_df["tweet"].apply(
            lambda x: " ".join(
                [word for word in x.split() if word not in stop_words_nltk]
            )
        )
        print("Stop words removed successfully.")

    def stem_words(self):
        """
        Preprocess the 'tweet' column by stemming the words.
        """
        porter_stemmer = PorterStemmer()
        self.pre_df["tweet"] = self.pre_df["tweet"].apply(
            lambda x: " ".join([porter_stemmer.stem(word) for word in x.split()])
        )
        print("Words stemmed successfully.")

    def remove_numbers(self):
        """
        Preprocess the 'tweet' column by removing the numbers.
        """
        self.pre_df["tweet"] = self.pre_df["tweet"].str.replace("\d+", "", regex=True)
        print("Numbers removed successfully.")

    def remove_punct(self):
        """
        Preprocess the 'tweet' column by removing the punctuation from the text.
        """

        def remove_punctuation(text):
            translator = str.maketrans("", "", string.punctuation)
            return text.translate(translator)

        self.pre_df["tweet"] = self.pre_df["tweet"].apply(remove_punctuation)
        print("Punctuation removed successfully.")

    @staticmethod
    def _remove_emojis(text):
        """
        Remove emojis from text using a regular expression.

        Parameters:
        - text (str): Input text

        Returns:
        - str: Text with emojis removed.
        """
        emoji_pattern = re.compile(
            "[" "\U0001F600-\U0001F64F"
            # emoticons
            "\U0001F300-\U0001F5FF"
            # symbols & pictographs
            "\U0001F680-\U0001F6FF"
            # transport & map symbols
            "\U0001F700-\U0001F77F"
            # alchemical symbols
            "\U0001F780-\U0001F7FF"
            # geometric shapes extended
            "\U0001F800-\U0001F8FF"
            # supplemental arrows-c
            "\U0001F900-\U0001F9FF"
            # supplemental symbols and pictographs
            "\U0001FA00-\U0001FA6F"
            # chess symbols
            "\U0001FA70-\U0001FAFF"
            # symbols and pictographs extended-a
            "\U00002702-\U000027B0"
            # dingbats
            "\U000024C2-\U0001F251" "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r"", text)

    def remove_non_english(self):
        """
        Remove all non-English words from the 'tweet' column.
        """

        def non_english(text):
            modified_text = re.sub(r"[^\u0000-\u007F]+", " ", text)
            return modified_text

        self.pre_df["tweet"] = self.pre_df["tweet"].apply(non_english)
        print("Non-English words removed successfully.")

    def non_alphabetic_words(self):
        """
        Print out non-alphabetic words in the 'tweet' column.
        """
        non_alphabetic_words = []

        for text in self.pre_df["tweet"]:
            non_alpha_matches = re.findall(r"\b[^a-zA-Z\s]+\b", text)
            non_alphabetic_words.extend(non_alpha_matches)

        non_alphabetic_words = list(set(non_alphabetic_words))

        print("Non-Alphabetic Words:")
        print(non_alphabetic_words)

    def fix_labels(self):
        """
        Assimilating the labels.
        """
        # self.pre_df['sentiment'] = self.pre_df['sentiment'].replace('irrelevant', 'neutral')
        self.pre_df = self.pre_df[self.pre_df["sentiment"] != "irrelevant"]
        print("Labels fixed successfully.")

    def remove_empty_tweets(self):
        """
        Remove rows with empty tweets.
        """
        self.pre_df = self.pre_df[self.pre_df["tweet"] != ""]
        print("Empty tweets removed successfully.")
