import pandas as pd
import requests
import time
import json
from tqdm import tqdm

common_citations_df = pd.read_csv("../../data/neuroscience/common_cited_dois.csv")

def process_cited_doi(full_doi):
    dois = str(full_doi).split(' ')

    # The second one is the actual DOI
    if len(dois) > 1:
        return dois[1].strip()
    else:
        return None

full_dois = common_citations_df['cited_doi'].tolist()
cleaned_dois = [process_cited_doi(doi) for doi in full_dois]

neuroscience_citing_dois = common_citations_df['neuroscience_citing_dois'].tolist()
neuromorphic_citing_dois = common_citations_df['neuromorphic_citing_dois'].tolist()

neuroscience_citing_dois = [str(doi).split(';') for doi in neuroscience_citing_dois]
neuromorphic_citing_dois = [str(doi).split(';') for doi in neuromorphic_citing_dois]


def fetch_crossref_metadata(doi):
    """Fetch metadata from Crossref API for a given DOI."""
    if not doi:
        return None

    url = f"https://api.crossref.org/works/{doi}"
    headers = {
        "User-Agent": "DatasetAssembly/1.0 (mailto:research@example.com)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()["message"]

            # Extract authors
            authors = []
            if "author" in data:
                for author in data["author"]:
                    name_parts = []
                    if "given" in author:
                        name_parts.append(author["given"])
                    if "family" in author:
                        name_parts.append(author["family"])
                    if name_parts:
                        authors.append(" ".join(name_parts))

            # Extract publication date
            pub_date = None
            pub_year = None
            if "published-print" in data:
                date_parts = data["published-print"].get("date-parts", [[]])[0]
            elif "published-online" in data:
                date_parts = data["published-online"].get("date-parts", [[]])[0]
            elif "created" in data:
                date_parts = data["created"].get("date-parts", [[]])[0]
            else:
                date_parts = []

            if date_parts:
                pub_year = date_parts[0] if len(date_parts) > 0 else None
                if len(date_parts) >= 3:
                    pub_date = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
                elif len(date_parts) >= 2:
                    pub_date = f"{date_parts[0]}-{date_parts[1]:02d}"
                elif len(date_parts) >= 1:
                    pub_date = str(date_parts[0])

            # Extract title
            title = data.get("title", [None])[0] if data.get("title") else None

            # Extract abstract
            abstract = data.get("abstract", None)

            # Extract journal name
            journal = None
            if "container-title" in data and data["container-title"]:
                journal = data["container-title"][0]

            return {
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "journal": journal,
                "publication_year": pub_year,
                "publication_date": pub_date
            }
        else:
            print(f"Failed to fetch {doi}: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching {doi}: {e}")
        return None


# Fetch metadata for all cleaned DOIs
metadata_array = []
for doi in tqdm(cleaned_dois, desc="Fetching Crossref metadata"):
    metadata = fetch_crossref_metadata(doi)
    metadata_array.append(metadata)
    time.sleep(0.1)  # Rate limiting to be respectful to the API

# Filter out None values for summary
valid_metadata = [m for m in metadata_array if m is not None]
print(f"Successfully fetched metadata for {len(valid_metadata)}/{len(cleaned_dois)} DOIs")

# Assemble the full json
detailed_metadata = []
for doi, meta, neuro_citers, nm_citers in zip(cleaned_dois, metadata_array, neuroscience_citing_dois, neuromorphic_citing_dois):
    entry = {
        "doi": doi,
        "metadata": meta,
        "neuroscience_citing_dois": neuro_citers,
        "neuromorphic_citing_dois": nm_citers
    }
    detailed_metadata.append(entry)

# Save to JSON
with open("../../data/neuroscience/cleaned_common_citations.json", "w") as f:
    json.dump(detailed_metadata, f, indent=2)
    