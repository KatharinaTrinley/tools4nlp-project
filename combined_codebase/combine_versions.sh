#!/bin/bash

script_path="/path/to/your/script.py"

csv_path="/path/to/your/csv/file.csv"

pip install -r requirements.txt

python -m spacy download en_core_web_lg

python ${script_path} --file_path ${csv_path} --sentiment_analysis