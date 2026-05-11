import pandas as pd
import requests
from tqdm import tqdm

def abstract_from_doi(doi):
    url = f"https://api.crossref.org/works/{doi}"
    try:
        r = requests.get(url, timeout=10,
                         headers={"User-Agent": "neuromorphic-citation-network/1.0 (https://github.com/neuromorphic-citation-network)"})
        if r.status_code == 200:
            data = r.json()
            return data.get("message", {}).get("abstract", "")
        else:
            return ""
    except requests.RequestException as e:
        print(f"Error fetching abstract for DOI {doi}: {e}")
        return ""

neuroscience_curr = pd.read_csv("data/neuroscience/neuroscience_nodes.csv")
neuroscience_new = pd.read_csv("new_neuroscience_data/references_with_doi.csv")

# Check for DOI intersections
curr_dois = set(neuroscience_curr["doi"].dropna().str.lower())
new_dois = set(neuroscience_new["DOI"].dropna().str.lower())
intersection = curr_dois.intersection(new_dois)

print(f"Current dataset has {len(curr_dois)} unique DOIs.")
print(f"New dataset has {len(new_dois)} unique DOIs.")
print(f"Intersection has {len(intersection)} DOIs.")

# Filter out rows from the new dataset that are already in the current dataset
filtered_new = neuroscience_new[~neuroscience_new["DOI"].str.lower().isin(intersection)]

# Filter to only include rows with a non-empty DOI
filtered_new = filtered_new[filtered_new["DOI"].notna() & (filtered_new["DOI"].str.strip() != "")]

print(f"After filtering, {len(filtered_new)} new entries remain.")

processed_rows = []

dayan_abbot_row = {
    "id": 750,
    "doi": "",
    "title": "Theoretical Neuroscience: Computational and Mathematical Modeling of Neural Systems",
    "abstract": "",
    "pdf_url": "https://boulderschool.yale.edu/sites/default/files/files/DayanAbbott.pdf",
    "authors": "Peter Dayan; L. F. Abbott",
    "date": "2001",
    "category": "neuroscience"
}
processed_rows.append(dayan_abbot_row)

for i, row in enumerate(tqdm(filtered_new.itertuples(index=False), total=len(filtered_new))):
    doi = row.DOI

    new_row = {
        "id": 750 + i + 1,  # start after Dayan & Abbott
        "doi": row.DOI,
        "title": row.Title,
        "abstract": abstract_from_doi(doi),
        "pdf_url": "",
        "authors": row.Authors,
        "date": row.Year,
        "category": "neuroscience"
    }
    processed_rows.append(new_row)

# Create a DataFrame and save to CSV
processed_df = pd.DataFrame(processed_rows)
neuroscience_curr = pd.concat([neuroscience_curr, processed_df], ignore_index=True)
print(neuroscience_curr.tail())

# Remove all <jats:*> tags from the abstract column
neuroscience_curr["abstract"] = neuroscience_curr["abstract"].str.replace(r"<jats:[^>]+>", "", regex=True)

out_path = "data/neuroscience/neuroscience_nodes_updated.csv"
neuroscience_curr.to_csv(out_path, index=False)
print(f"\nUpdated dataset saved to {out_path}")
