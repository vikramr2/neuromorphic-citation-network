import pandas as pd
import requests
import time
from urllib.parse import quote
from tqdm import tqdm

ARXIV_CSV = "../../data/aiml/arxiv_ml_papers_detailed_info.csv"
ICML_CSV = "../../data/aiml/icml_2021_2024_papers_titles_abstracts.csv"

def get_doi_from_row(row):
    """Fetch DOI for a paper using CrossRef API based on title."""
    title = row['title']
    encoded_title = quote(title)
    url = f"https://api.crossref.org/works?query.title={encoded_title}&rows=1"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            items = data.get('message', {}).get('items', [])
            if items:
                return items[0].get('DOI')
    except requests.RequestException:
        pass

    return None

arxiv_df = pd.read_csv(ARXIV_CSV)

print(f"Total arXiv ML papers loaded: {len(arxiv_df)}")

dois = arxiv_df['doi'].dropna().unique().tolist()

# Get corresponding titles, abstracts, urls, years to the DOIs
titles = []
abstracts = []
urls = []
years = []

for doi in tqdm(dois, desc="Processing arXiv DOIs"):
    row = arxiv_df[arxiv_df['doi'] == doi].iloc[0]
    titles.append(row['title'])
    abstracts.append(row['abstract'])
    urls.append(row.get('pdf_url', None))
    years.append(str(row.get('year', '')))

print(f"Total unique DOIs extracted: {len(dois)}")

icml_df = pd.read_csv(ICML_CSV)

print(f"Total ICML papers loaded: {len(icml_df)}")

print(icml_df.head())

# Fetch DOIs for ICML papers
for idx, row in tqdm(icml_df.iterrows(), total=len(icml_df), desc="Fetching DOIs"):
    doi = get_doi_from_row(row)
    tag = row.get('tags', '----')

    dois.append(doi)
    titles.append(row['title'])
    abstracts.append(row['abstract'])
    urls.append(row.get('Download PDF', None))
    years.append(tag[-4:])  # Extract last 4 characters (year)
    
    time.sleep(0.1)  # Rate limiting to avoid hitting API limits

full_df = pd.DataFrame({
    'doi': dois,
    'title': titles,
    'abstract': abstracts,
    'pdf_url': urls,
    'year': years
})

# Save the dois to a csv
full_df.to_csv("../../data/aiml/combined_ml_papers_dois.csv", index=False)

# with open("../../data/aiml/combined_ml_papers_dois.txt", 'w') as f:
#     for doi in dois:
#         f.write(f"{doi}\n")

print(f"Total unique DOIs combined: {len(dois)}")
print("DOIs saved to '../../data/aiml/combined_ml_papers_dois.txt'")
