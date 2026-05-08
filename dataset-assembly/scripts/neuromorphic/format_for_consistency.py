import pandas as pd
import json
import requests
import time
from tqdm import tqdm

def fetch_abstract(doi):
    """Fetch abstract from CrossRef API for a given DOI."""
    if not doi:
        return ""
    try:
        crossref_url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(crossref_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            time.sleep(1)  # To respect API rate limits
            return data["message"].get("abstract", "")
    except requests.exceptions.RequestException:
        pass
    return ""

def fetch_title(doi):
    """Fetch title from CrossRef API for a given DOI."""
    if not doi:
        return ""
    try:
        crossref_url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(crossref_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            time.sleep(1)  # To respect API rate limits
            title_list = data["message"].get("title", [])
            return title_list[0] if title_list else ""
    except requests.exceptions.RequestException:
        pass
    return ""

def process_json_entry(entry):
    doi = entry.get("doi", "")
    title = fetch_title(doi) if doi else entry.get("title", "")
    pdf_url = f"papers/{entry.get('pdf_filename', '')}"

    authors = ""
    date = ""
    crossref_metadata = entry.get("crossref_metadata", {})
    if crossref_metadata:
        author_list = crossref_metadata.get("authors", [])
        authors = "; ".join(
            a.get("full_name", "") or a.get("name", "") for a in author_list
        )
        publication_dates = crossref_metadata.get("publication_dates", {})
        if publication_dates and "issued" in publication_dates:
            date = publication_dates["issued"].get("formatted", "")

    return {
        "doi": doi,
        "title": title,
        "pdf_url": pdf_url,
        "authors": authors,
        "date": date,
        "category": "neuromorphic",
    }

NEUROMORPHIC_JSON_PATH = "../../data/neuromorphic/neuromorphic_papers_cleaned_enhanced.json"
NEUROMORPHIC_CSV_PATH = "../../data/neuromorphic/neuromorphic_nodelist_final.csv"
OUTPUT_CSV_PATH = "../../data/neuromorphic/neuromorphic_nodelist_with_metadata.csv"

neuromorphic_df = pd.read_csv(NEUROMORPHIC_CSV_PATH)

with open(NEUROMORPHIC_JSON_PATH, "r") as f:
    neuromorphic_json = json.load(f)

print(f"Nodelist: {len(neuromorphic_df)} rows")
print(f"JSON entries: {len(neuromorphic_json)}")

# Process all JSON entries for metadata (authors, date, pdf_url)
rows = []
for entry in tqdm(neuromorphic_json, desc="Processing JSON entries"):
    rows.append(process_json_entry(entry))

extra_info_df = pd.DataFrame(rows)

# Fetch abstracts from CrossRef for entries that have a DOI
print("Fetching abstracts from CrossRef...")
abstracts = []
for doi in tqdm(extra_info_df["doi"], desc="Fetching abstracts"):
    abstracts.append(fetch_abstract(doi))
extra_info_df["abstract"] = abstracts

# Join with nodelist on doi
result_df = neuromorphic_df.merge(extra_info_df, on="doi", how="left")

print(f"Result: {len(result_df)} rows, columns: {list(result_df.columns)}")
print(result_df.head())

result_df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Saved to {OUTPUT_CSV_PATH}")
