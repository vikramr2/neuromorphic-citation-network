import json
import pandas as pd

DATA_DIR = "../data/"

with open(DATA_DIR + "mongo_data/mongo_data_with_dois.json", "r") as f:
    mongo_data_with_dois = json.load(f)

print(mongo_data_with_dois[0].keys())

mongo_titles = [entry['title'] for entry in mongo_data_with_dois if 'title' in entry]
mongo_dois = [entry['doi'] for entry in mongo_data_with_dois if 'doi' in entry]
mongo_data_df = pd.DataFrame({
    'pdf_filename': 'mongoDB',
    'title': mongo_titles,
    'doi': mongo_dois
})

print(mongo_data_df.head())

fixed_titles_with_dois = pd.read_csv(DATA_DIR + "fixed_titles_with_dois.csv")
papers_with_dois = pd.read_csv(DATA_DIR + "papers_with_dois.csv")

# Drop old_title column from fixed_titles_with_dois
fixed_titles_with_dois = fixed_titles_with_dois.drop(columns=["old_title"])

# Rename new)_title to title for clarity
fixed_titles_with_dois = fixed_titles_with_dois.rename(columns={"new_title": "title"})

# Entries from the fixed titles such that the doi is empty
fixed_titles_with_dois = fixed_titles_with_dois[fixed_titles_with_dois['doi'].notna()]

print(fixed_titles_with_dois.head())
print(papers_with_dois.head())

# Remove entries in papers_with_dois where the pdf_filename exists in fixed_titles_with_dois
print(f"Number of entries in papers_with_dois before filtering: {len(papers_with_dois)}")
filtered_papers_with_dois = papers_with_dois[~papers_with_dois['pdf_filename'].isin(fixed_titles_with_dois['pdf_filename'])]
print(f"Number of entries in papers_with_dois after filtering: {len(filtered_papers_with_dois)}")

# Combine the two dataframes
combined = pd.concat([fixed_titles_with_dois, filtered_papers_with_dois], ignore_index=True)
print(f"Combined dataset has {len(combined)} entries.")

# Remove entries in mongo_data_df where the doi exists in combined
print(f"Number of entries in mongo_data_df before filtering: {len(mongo_data_df)}")
filtered_mongo_data_df = mongo_data_df[~mongo_data_df['doi'].isin(combined['doi'])]
print(f"Number of entries in mongo_data_df after filtering: {len(filtered_mongo_data_df)}")

# Combine with mongo_data_df
final_combined = pd.concat([combined, filtered_mongo_data_df], ignore_index=True)
print(f"Final combined dataset has {len(final_combined)} entries.")

# Remove entries where title or doi is missing
final_combined = final_combined[final_combined['title'].notna() & final_combined['doi'].notna()]
print(f"Final combined dataset after removing missing titles or DOIs has {len(final_combined)} entries.")

# Save the final combined dataframe
output_path = DATA_DIR + "combined_neuromorphic_title_dois.csv"
final_combined.to_csv(output_path, index=False)
