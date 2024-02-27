# Sentiment Analysis Project

## Overview

This repository contains the code and findings from an exploration of Twitter data preprocessing and sentiment analysis. The analysis compares sentiment analysis models VADER, SpacyTextBlob, and Hugging Face's RoBERTa. The focus is on understanding their performance characteristics and drawing insights from the results.

## Project Structure

    ├── LICENSE
    ├── README.md
    ├── bonus (screenshots for bonus tasks)
    ├── codebase_daniil
    │   └── preprocessor.py
    ├── combined_codebase
    │   ├── combine_versions.sh
    │   ├── main.py
    │   ├── preprocessor_a.py
    │   ├── sentiment_analyser_a.py
    ├── data
    ├── project_requirements.txt (specifies what requirements we attempted to fulfill)
    └── requirements.txt


## Usage 

To run the preprocessing and sentiment analysis, execute the provided bash script:

`bash combined_codebase/combine_versions.sh`

In the bash script, the file path should be provided through the corresponding flag. Also, the other flag can be used for choosing if the sentiment analysis should be performed or not. The flags are showcased below:

`--file_path data/data.csv`
`--sentiment_analysis`

## Data Preprocessing
The data preprocessing phase covers various steps, including:

- Handling missing values
- Converting data types
- Lowercasing text
- Removing non-ASCII characters, emojis, stopwords
- Stemming words
- Removing numbers, punctuation, non-English words
- Fixing labels and removing empty tweets
- These steps collectively create a clean and standardized dataset for effective sentiment analysis.

## Sentiment Analysis
The sentiment analysis compares [VADER](https://ojs.aaai.org/index.php/ICWSM/article/view/14550), [SpacyTextBlob](https://spacytextblob.netlify.app/), and [RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest?library=transformers) models. The findings indicate that VADER and TextBlob, being lexicon and rule-based models, struggle with nuanced sentiment expressions. RoBERTa, a deep learning model, outperforms with superior precision, recall, and overall F1-score across all sentiment classes.

## Discussion
The discussion delves into the factors contributing to RoBERTa's superior performance, emphasizing its deep learning architecture, pre-training on a large dataset, and fine-tuning for sentiment analysis tasks.

## Conclusion
This report highlights the importance of robust data preprocessing in preparing Twitter data for sentiment analysis. It emphasizes the limitations of traditional lexicon-based and rule-based models and showcases the advancements achieved with state-of-the-art deep learning models like RoBERTa.

## Dependencies
- Python 3.x
- Required Python packages (install using pip install -r requirements.txt) - included in the bash script
