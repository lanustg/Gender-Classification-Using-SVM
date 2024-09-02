# ultimate version
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# pip install pandas scikit-learn

# Set the folder path
cleaned_csv = ""  # cleaned file path
output_csv = ""  # output file path

# Reading the cleaned CSV file
df = pd.read_csv(cleaned_csv)

# Defining the TF-IDF vectorizer
di = TfidfVectorizer(
    ngram_range=(1, 1),  # Use 1-gram, 2-gram, 3-gram
    sublinear_tf=True,   # Sublinear word frequency scaling
    min_df=2,            # Minimum document frequency
    max_df=0.99,         # The maximum document frequency is set to 0.99 to ignore words that appear in all documents.
    max_features=4000
)

# Process data in batches
batch_size = 1501  # The number of documents to process in each batch
num_batches = (len(df) // batch_size) + 1

for i in range(num_batches):
    start = i * batch_size
    end = (i + 1) * batch_size
    batch_df = df[start:end]

    # Vectorize the document using TF-IDF
    X = vectorizer.fit_transform(batch_df['document'])

    # Convert TF-IDF features to DataFrame
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    # Add user_id and gender columns
    tfidf_df.insert(0, 'user_id', batch_df['user_id'].reset_index(drop=True))
    tfidf_df['gender'] = batch_df['gender'].reset_index(drop=True)

    # Move the gender column to the second column
    cols = list(tfidf_df.columns)
    cols.insert(1, cols.pop(cols.index('gender')))
    tfidf_df = tfidf_df[cols]

    # Save as CSV file, append mode
    if i == 0:
        tfidf_df.to_csv(output_csv, index=False, encoding='utf-8')
    else:
        tfidf_df.to_csv(output_csv, index=False, encoding='utf-8', mode='a', header=False)
    print(f"Batch {i + 1}/{num_batches} has been processed and written to {output_csv}")

print(f"All data has been written to {output_csv}")
