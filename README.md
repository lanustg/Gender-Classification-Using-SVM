# Gender-Classification-Using-SVM
Weka is a necessary software tool required for this project. I used Weka version 3.8.6.
Python version 3.11.5
Data Cleaning and CSV Generation (generate_cleaned_csv.py):
This script is used to match the IDs in the TrainingText folder with their corresponding gender labels found in Training.txt, and to integrate this information into a single CSV file. It then cleans the text data using the clean_text(text) function.
Required Inputs:
source_folder: The absolute path to the TrainingText folder.
label_file: The absolute path to Training.txt.
output_csv: The absolute path where the cleaned CSV file will be saved.
The output CSV will contain the cleaned and preprocessed text data, ready for further analysis.
TF-IDF Vectorization (generate_tfidf_csv2.py):
This script vectorizes the cleaned CSV file generated in the previous step using a TF-IDF vectorizer. The vectorizer can be customized by adjusting the ngram_range for the number of n-grams and max_features for the maximum number of features to retain.
Required Inputs:
cleaned_csv: The absolute path to the cleaned CSV file generated by generate_cleaned_csv.py.
output_csv: The absolute path where the TF-IDF vectorized CSV file will be saved.
After this step, the resulting file can be imported into Weka for anonymization, classification category setup, and exported as an ARFF file for model building and evaluation.
Dimensionality Reduction with LSA (generate_tfidf_LSA_csv2.py):
This script further reduces the dimensionality of the TF-IDF vectorized CSV file using Latent Semantic Analysis (LSA). You can adjust the number of LSA dimensions to keep by modifying the n_components parameter.
Required Inputs:
input_tfidf_csv: The absolute path to the TF-IDF vectorized CSV file generated by generate_tfidf_csv2.py.
output_lsa_csv: The absolute path where the LSA-reduced CSV file will be saved.
Similar to the previous step, the output can be imported into Weka for subsequent operations.

![image](https://github.com/user-attachments/assets/4621c97c-02ab-45c1-b1a2-a2af172d2df3)
