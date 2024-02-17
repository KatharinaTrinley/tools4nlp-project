import argparse

from preprocessor_a import TextPreprocessor
from sentiment_analyser_a import SentimentAnalyzer


def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Text Preprocessing and Sentiment Analysis"
    )
    parser.add_argument(
        "--file_path",
        default="twitter_training.csv",
        type=str,
        help="Path to the CSV file",
        required=True,
    )
    parser.add_argument(
        "--sentiment_analysis", action="store_true", help="Perform sentiment analysis"
    )

    args = parser.parse_args()

    # STEP 1: Preprocessing the data
    file_path = args.file_path
    preprocessor = TextPreprocessor(file_path)

    # displaying the head of the DataFrame
    preprocessor.display_head()

    # displaying summary statistics
    preprocessor.display_summary_statistics()

    # handling missing values
    preprocessor.handle_missing_values()

    # converting data types
    preprocessor.convert_data_types()

    # displaying current data types
    preprocessor.display_data_types()

    # preprocessing text
    preprocessor.lowercase_text()
    preprocessor.remove_non_ascii()
    preprocessor.remove_emojis()
    preprocessor.remove_stopwords()
    preprocessor.stem_words()
    preprocessor.remove_numbers()
    preprocessor.remove_punct()
    preprocessor.remove_non_english()

    # displaying non-alphabetic words
    preprocessor.non_alphabetic_words()

    # fixing labels
    preprocessor.fix_labels()

    # removing empty tweets
    preprocessor.remove_empty_tweets()

    # displaying the head of the pre-processed DataFrame
    preprocessor.display_head()

    # STEP 2: Sentiment Analysis
    if args.sentiment_analysis:
        sentiment_analyzer = SentimentAnalyzer(preprocessor.pre_df)
        sentiment_analyzer.analyze_and_visualize()


if __name__ == "__main__":
    main()
