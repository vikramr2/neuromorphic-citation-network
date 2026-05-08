import pandas as pd
import json
from tqdm import tqdm

COMMON_PAPER_METADATA = "../../data/neuroscience/cleaned_common_citations.json"
NEURO_PAPERS_CSV = "../../data/neuroscience/expanded_neuroscience_papers.csv"

def get_corresponding_value(doi, common_metadata, key):
    for entry in common_metadata:
        if entry['doi'] == doi:
            return entry['metadata'].get(key, None)
    return None

def set_row_data(doi, common_metadata, neuro_papers_df):
    # First get the corresponding fields from the common metadata
    title = get_corresponding_value(doi, common_metadata, 'title')
    abstract = get_corresponding_value(doi, common_metadata, 'abstract')
    authors = get_corresponding_value(doi, common_metadata, 'authors')
    date = get_corresponding_value(doi, common_metadata, 'publication_date')

    # All papers in this dataset are neuroscience papers
    category = "Neuroscience"

    # Process the authors list into a string separated by semicolons
    if authors is not None:
        authors_str = "; ".join([author for author in authors])
    else:
        authors_str = None

    # Update the DataFrame row
    neuro_papers_df.loc[neuro_papers_df['doi'] == doi, 'title'] = title
    neuro_papers_df.loc[neuro_papers_df['doi'] == doi, 'abstract'] = abstract
    neuro_papers_df.loc[neuro_papers_df['doi'] == doi, 'authors'] = authors_str
    neuro_papers_df.loc[neuro_papers_df['doi'] == doi, 'date'] = date
    neuro_papers_df.loc[neuro_papers_df['doi'] == doi, 'category'] = category


with open(COMMON_PAPER_METADATA, "r") as f:
    common_metadata = json.load(f)

neuro_papers_df = pd.read_csv(NEURO_PAPERS_CSV)

# Remove the published_date column if it exists
if 'published_date' in neuro_papers_df.columns:
    neuro_papers_df = neuro_papers_df.drop(columns=['published_date'])

# Drop the version column: we probably won't use it
if 'version' in neuro_papers_df.columns:
    neuro_papers_df = neuro_papers_df.drop(columns=['version'])

for entry in tqdm(common_metadata):
    doi = entry['doi']
    set_row_data(doi, common_metadata, neuro_papers_df)

neuro_papers_df.to_csv(NEURO_PAPERS_CSV, index=False)
