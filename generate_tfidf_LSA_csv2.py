import pandas as pd
from sklearn.decomposition import TruncatedSVD

# Set file paths
input_tfidf_csv = ""  # Path to input TF-IDF matrix file
output_lsa_csv = ""  # Path to save the output LSA matrix file

# Load the TF-IDF matrix (assuming each row is a document, and columns are TF-IDF features)
df_tfidf = pd.read_csv(input_tfidf_csv)

# Separate metadata (such as 'user_id' and 'gender'), assuming these columns exist
meta_data = df_tfidf[['user_id', 'gender']]  # Keep user_id and gender columns
tfidf_matrix = df_tfidf.drop(columns=['user_id', 'gender'])  # Keep only the TF-IDF matrix data

# Perform Latent Semantic Analysis (LSA) using TruncatedSVD
n_components = 500  # Number of LSA dimensions to keep, adjust as needed
svd = TruncatedSVD(n_components=n_components)

# Apply dimensionality reduction
lsa_matrix = svd.fit_transform(tfidf_matrix)

# Convert the LSA matrix to a DataFrame
lsa_df = pd.DataFrame(lsa_matrix, columns=[f'lsa_component_{i+1}' for i in range(n_components)])

# Merge metadata and LSA results
result_df = pd.concat([meta_data.reset_index(drop=True), lsa_df], axis=1)

# Save the LSA results to a CSV file
result_df.to_csv(output_lsa_csv, index=False, encoding='utf-8')

print(f"The LSA reduced matrix has been saved to {output_lsa_csv}")
