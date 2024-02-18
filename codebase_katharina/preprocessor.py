import pandas as pd
import re
import spacy
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords

class DataPreprocessor:
    def __init__(self):
        self.data = None

    def read_csv(self, file_path):
        self.data = pd.read_csv(file_path, header=None)

    def display_head(self):
        if self.data is not None:
            print(self.data.head())
        else:
            print("No data loaded. Please load data using read_csv method.")

    def descriptive_stats(self):
        if self.data is not None:
            print("Descriptive Statistics:")
            print(self.data.describe())
            print("Info:")
            print(self.data.info())
        else:
            print("No data loaded. Please load data using read_csv method.")

    def rename_dataframe(self, new_name):
        if self.data is not None:
            self.data = self.data.rename(columns=new_name)
        else:
            print("No data loaded. Please load data using read_csv method.")

    def find_missing_values(self, column_index):
        if self.data is not None:
            missing_values = self.data[column_index].isnull().sum()
            print(f"Number of missing values in column {column_index}: {missing_values}")
            return missing_values
        else:
            print("No data loaded. Please load data using read_csv method.")

    def fill_missing_values(self, column_index, fill_value=''):
        if self.data is not None:
            self.data[column_index].fillna(fill_value, inplace=True)
            missing_values_after_fill = self.data[column_index].isnull().sum()
            print(f"Number of missing values in column {column_index} after filling: {missing_values_after_fill}")
            return missing_values_after_fill
        else:
            print("No data loaded. Please load data using read_csv method.")

    def lowercase_text(self, column_index):
        if self.data is not None:
            self.data[column_index] = self.data[column_index].str.lower()
            return self.data
        else:
            print("No data loaded. Please load data using read_csv method.")

    def remove_nonascii_add_whitespace(self, column_index):
        if self.data is not None:
            import re

            def remove_nonascii_and_add_whitespace(text):
                if isinstance(text, str):
                    return re.sub(r'[^\x00-\x7F]+', ' ', text)
                else:
                    return ''

            self.data[column_index] = self.data[column_index].apply(remove_nonascii_and_add_whitespace)

            self.data[column_index] = self.data[column_index].str.replace(r'\s+', ' ', regex=True)

            return self.data
        else:
            print("No data loaded. Please load data using read_csv method.")

    def split_text(self, column_index):
        if self.data is not None:
            self.data[column_index] = self.data[column_index].str.split()

            def clean_text(words_list):
                return [word.strip() for word in words_list]

            self.data[column_index] = self.data[column_index].apply(clean_text)

            return self.data
        else:
            print("No data loaded. Please load data using read_csv method.")

    def remove_backslash_n(self, column_index):
        if self.data is not None:
        
            def remove_backslash_n(words_list):
                return [word.replace('\n', '') for word in words_list]

            self.data[column_index] = self.data[column_index].apply(remove_backslash_n)
            return self.data
        else:
            print("No data loaded. Please load data using read_csv method.")

    def remove_emojis(self, column_index):
        if self.data is not None:
        
            def remove_emojis(data):
                if isinstance(data, str):
                    # Source: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
                    emoji_pattern = re.compile("["
                                                u"\U0001F600-\U0001F64F"  # emoticons
                                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                                u"\U00002500-\U00002BEF"  # chinese char
                                                u"\U00002702-\U000027B0"
                                                u"\U000024C2-\U0001F251"
                                                u"\U0001f926-\U0001f937"
                                                u"\U00010000-\U0010ffff"
                                                u"\u2640-\u2642"
                                                u"\u2600-\u2B55"
                                                u"\u200d"
                                                u"\u23cf"
                                                u"\u23e9"
                                                u"\u231a"
                                                u"\ufe0f"  # dingbats
                                                u"\u3030"
                                                "]+", re.UNICODE)
                    return re.sub(emoji_pattern, '', data)
                else:
                    return data

            self.data[column_index] = self.data[column_index].apply(remove_emojis)
            return self.data
        else:
            print("No data loaded. Please load data using read_csv method.")

    def nltk_stopwords_(self):
        return set(nltk.corpus.stopwords.words('english'))

    def spacy_stopwords_(self):
        nlp = spacy.load("en_core_web_sm")
        return nlp.Defaults.stop_words

    def textblob_stopwords_(self):
        return set(nltk.corpus.stopwords.words('english'))

    def sklearn_stopwords_(self):
        return sklearn_stopwords

    def remove_non_alphabetic(self, column_index):
        if self.data is not None:
            def remove_non_alphabetic(text):
                if isinstance(text, str):
                    return re.sub(r'\b[^a-zA-Z\s]+\b', '', text)
                else:
                    return ''

            self.data['cleaned_text'] = self.data[column_index].apply(remove_non_alphabetic)
            all_words = ' '.join(self.data['cleaned_text']).split()
            non_alphabetic_words = [word for word in set(all_words) if not word.isalpha()]

            print("Non-alphabetic words remaining:")
            print(non_alphabetic_words)
            print("\nNumber of non-alphabetic words:", len(non_alphabetic_words))

            return self.data
        else:
            print("No data loaded. Please load data using read_csv method.")
