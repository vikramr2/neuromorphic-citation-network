import pandas as pd
import requests
import time
from typing import Optional
from tqdm import tqdm

DATA_DIR = "../data/"

def get_doi_from_crossref(title: str) -> Optional[str]:
    """
    Query Crossref API to get DOI from paper title.

    Args:
        title: Paper title to search for

    Returns:
        DOI string if found, None otherwise
    """
    if pd.isna(title) or not title.strip():
        return None

    try:
        # Crossref API endpoint
        url = "https://api.crossref.org/works"
        params = {
            'query.title': title,
            'rows': 1,
            'select': 'DOI,title,score'
        }

        # Add polite pool parameters (recommended by Crossref)
        headers = {
            'User-Agent': 'Mozilla/5.0 (mailto:your-email@example.com)'
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data.get('message', {}).get('items'):
            item = data['message']['items'][0]
            # Check if score is reasonable (> 80 is usually a good match)
            score = item.get('score', 0)
            if score > 50:  # Lower threshold to catch more matches
                doi = item.get('DOI')
                return doi
            else:
                return None
        else:
            return None

    except Exception as e:
        return None


malformed_titles = pd.read_csv(DATA_DIR + "malformed_titles_with_dois.csv")
missing_titles = pd.read_csv(DATA_DIR + "missing_papers.csv")

# Remove doi from malformed titles (old DOI is incorrect)
malformed_titles = malformed_titles.drop(columns=["doi"])

# Rename title in malformed titles to old_title for clarity
malformed_titles = malformed_titles.rename(columns={"title": "old_title"})

print(f"Malformed titles: \n{malformed_titles.head()}\n")
print(f"Missing titles: \n{missing_titles.head()}\n")

# Append missing titles to malformed titles
combined = pd.concat([malformed_titles, missing_titles], ignore_index=True)

print(f"Combined dataset has {len(combined)} entries.")
print(f"Sample entries:\n{combined.head()}\n")

# Fetch DOIs from Crossref based on new_title
print("\nFetching DOIs from Crossref API...")
print("This may take a while...\n")

dois = []
for idx, row in tqdm(combined.iterrows(), total=len(combined), desc="Fetching DOIs"):
    new_title = row['new_title']  # row is already the Series
    doi = get_doi_from_crossref(new_title)
    dois.append(doi)

    # Be respectful to the API - rate limit to ~1 request per second
    time.sleep(1)

combined['doi'] = dois

print(f"\nDOI fetching complete!")
print(f"Successfully found {combined['doi'].notna().sum()} DOIs out of {len(combined)} papers.")
print(f"\nSample entries with DOIs:\n{combined.head(10)}\n")

# Save the updated dataframe
output_path = DATA_DIR + "fixed_titles_with_dois.csv"
combined.to_csv(output_path, index=False)
print(f"Saved combined dataset with DOIs to {output_path}")

