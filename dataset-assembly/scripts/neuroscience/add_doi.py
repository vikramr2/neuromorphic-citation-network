import time
import pandas as pd
import requests

refs = pd.read_csv("new_neuroscience_data/raw_references.csv")


def fetch_doi(title, authors, year):
    url = "https://api.crossref.org/works"
    params = {
        "query.title": title,
        "query.author": authors,
        "filter": f"from-pub-date:{year},until-pub-date:{year}",
        "rows": 1,
        "select": "DOI,title,author,published",
    }
    try:
        r = requests.get(url, params=params, timeout=10,
                         headers={"User-Agent": "neuromorphic-citation-network/1.0 (vikramr2@illinois.edu)"})
        r.raise_for_status()
        items = r.json()["message"]["items"]
        if items:
            return items[0].get("DOI", "")
    except Exception:
        pass
    return ""


dois = []
for _, row in refs.iterrows():
    doi = fetch_doi(row["Title"], row["Authors"], row["Year"])
    dois.append(doi)
    print(f"{'OK' if doi else '--'} | {row['Title'][:60]}")
    time.sleep(0.1)  # be polite to the API

refs["DOI"] = dois
out_path = "new_neuroscience_data/references_with_doi.csv"
refs.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
