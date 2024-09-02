# ultimate version
import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
from nltk.tokenize import TweetTokenizer

# pip install pandas lxml nltk

# Set the folder path
source_folder = ""  # training dataset path
label_file = ""  # label file path
output_csv = ""  # output file path

# Reading the label file
labels = {}
with open(label_file, 'r') as f:
    for line in f:
        user_id, gender = line.strip().split(':::')
        labels[user_id] = gender

# initialize TweetTokenizer
tokenizer = TweetTokenizer()

# Defining text cleaning functions
def clean_text(text):
    text = re.sub(r'\n', '<LineFeed>', text)  # Replace line feeds with <LineFeed>
    text = re.sub(r'http\S+|www\S+|https\S+', '<URLURL>', text, flags=re.MULTILINE)  # Replace URL with <URLURL>
    text = re.sub(r'@\w+', '<UsernameMention>', text)  # Replace @UsernameMention with <UsernameMention>
    text = text.lower()  # Convert all characters to lowercase
    text = re.sub(r'(.)\1{2,}', r'\1\1\1', text)  # Trim repeated characters
    text = re.sub(r'[^A-Za-z\s<EndOfTweet><LineFeed><URLURL><UsernameMention>]', '', text)  # Remove punctuation, but keep the specified tags
    return text.strip()

# Parse the XML file and extract tweets
documents = []
user_ids = []
y = []
for filename in os.listdir(source_folder):
    if filename.endswith(".xml"):
        user_id = filename.split('.')[0]
        if user_id in labels:  # Only process tagged files
            tree = ET.parse(os.path.join(source_folder, filename))
            root = tree.getroot()
            tweets = [clean_text(doc.text) + ' <EndOfTweet>' for doc in root.findall(".//documents/document")]
            document = " ".join(tweets)
            documents.append(document)
            user_ids.append(user_id)
            y.append(labels[user_id])

# Convert the data into a DataFrame
df = pd.DataFrame({'user_id': user_ids, 'document': documents, 'gender': y})

# Save as CSV file
df.to_csv(output_csv, index=False, encoding='utf-8')
print(f"Data has been written to {output_csv}")
