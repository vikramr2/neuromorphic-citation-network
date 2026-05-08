import pandas as pd

NEURO_AI_CITATION_CSV = "../../data/aiml/neuro_aiml_citation_events.csv"
PAPERS_CSV = "../../data/aiml/combined_ml_papers_dois.csv"

citation_df = pd.read_csv(NEURO_AI_CITATION_CSV)
papers_df = pd.read_csv(PAPERS_CSV)

print(citation_df.shape)

citation_dois_set = set(citation_df['citing_doi'].tolist() + citation_df['cited_doi'].tolist())

filtered_papers_df = papers_df[papers_df['doi'].isin(citation_dois_set)]

print(f"Total papers before filtering: {len(papers_df)}")
print(f"Total papers after filtering: {len(filtered_papers_df)}")

print(filtered_papers_df.head())

# Convert year to int
filtered_papers_df['year'] = pd.to_numeric(filtered_papers_df['year'], errors='coerce').fillna(0).astype(int)

print(filtered_papers_df.head())

# Save the filtered papers to a new CSV
filtered_papers_df.to_csv("../../data/aiml/filtered_neuro_aiml_papers.csv", index=False)
