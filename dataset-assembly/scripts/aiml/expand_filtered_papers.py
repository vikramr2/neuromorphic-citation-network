import pandas as pd
import requests
import time
from tqdm import tqdm

FILTERED_PAPERS_CSV = "../../data/aiml/filtered_neuro_aiml_papers.csv"
COMBINED_PAPERS_CSV = "../../data/aiml/combined_ml_papers_dois.csv"

S2_API = "https://api.semanticscholar.org/graph/v1"
NUM_HOPS = 2
MAX_PAGES = 5  # Cap pagination at 5 pages (5000 results per endpoint call)
MAX_RETRIES = 5  # Max 429 retries before giving up on a request
REQUEST_DELAY = 3  # Seconds between requests (unauthenticated limit ~100 req/5 min)

ARXIV_DOI_PREFIX = "10.48550/arXiv."

def doi_to_s2_id(doi):
    """Convert a DOI to a Semantic Scholar paper identifier."""
    if doi.startswith(ARXIV_DOI_PREFIX):
        return f"ARXIV:{doi[len(ARXIV_DOI_PREFIX):]}"
    return f"DOI:{doi}"

def s2_paper_to_doi(external_ids):
    """Convert S2 externalIds to the DOI format used in our combined table."""
    if not external_ids:
        return None
    # Prefer arXiv-derived DOI since that's what most of our table uses
    arxiv_id = external_ids.get("ArXiv")
    if arxiv_id:
        return f"{ARXIV_DOI_PREFIX}{arxiv_id}"
    doi = external_ids.get("DOI")
    if doi:
        return doi
    return None

def get_references(doi):
    """Fetch DOIs of papers that a given paper cites."""
    s2_id = doi_to_s2_id(doi)
    results = []
    offset = 0
    pages = 0
    retries = 0
    while pages < MAX_PAGES:
        url = f"{S2_API}/paper/{s2_id}/references?fields=externalIds&limit=1000&offset={offset}"
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 429:
                retries += 1
                if retries > MAX_RETRIES:
                    print(f"  [WARN] references {doi}: rate limited, skipping")
                    break
                time.sleep(5)
                continue
            retries = 0
            if response.status_code != 200:
                print(f"  [WARN] references {doi}: HTTP {response.status_code}")
                break
            data = response.json()
            for item in (data.get("data") or []):
                ref_doi = s2_paper_to_doi(item.get("citedPaper", {}).get("externalIds"))
                if ref_doi:
                    results.append(ref_doi)
            pages += 1
            if "next" not in data:
                break
            offset = data["next"]
        except requests.RequestException as e:
            print(f"  [ERR] references {doi}: {e}")
            break
    time.sleep(REQUEST_DELAY)
    return results

def get_citations(doi):
    """Fetch DOIs of papers that cite a given paper."""
    s2_id = doi_to_s2_id(doi)
    results = []
    offset = 0
    pages = 0
    retries = 0
    while pages < MAX_PAGES:
        url = f"{S2_API}/paper/{s2_id}/citations?fields=externalIds&limit=1000&offset={offset}"
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 429:
                retries += 1
                if retries > MAX_RETRIES:
                    print(f"  [WARN] citations {doi}: rate limited, skipping")
                    break
                time.sleep(5)
                continue
            retries = 0
            if response.status_code != 200:
                print(f"  [WARN] citations {doi}: HTTP {response.status_code}")
                break
            data = response.json()
            for item in (data.get("data") or []):
                ref_doi = s2_paper_to_doi(item.get("citingPaper", {}).get("externalIds"))
                if ref_doi:
                    results.append(ref_doi)
            pages += 1
            if "next" not in data:
                break
            offset = data["next"]
        except requests.RequestException as e:
            print(f"  [ERR] citations {doi}: {e}")
            break
    time.sleep(REQUEST_DELAY)
    return results

def expand_one_hop(frontier_dois, rest_doi_set):
    """Given a set of frontier DOIs, find all citation events to/from rest_doi_set.
    Returns new citation events and the set of newly discovered DOIs."""
    citation_events = []

    # frontier -> rest: what do frontier papers cite?
    for doi in tqdm(frontier_dois, desc="  references (frontier -> rest)"):
        for cited_doi in get_references(doi):
            if cited_doi in rest_doi_set:
                citation_events.append((doi, cited_doi))

    # rest -> frontier: what papers cite the frontier papers?
    for doi in tqdm(frontier_dois, desc="  citations  (rest -> frontier)"):
        for citing_doi in get_citations(doi):
            if citing_doi in rest_doi_set:
                citation_events.append((citing_doi, doi))

    new_dois = set()
    for citing, cited in citation_events:
        new_dois.add(citing)
        new_dois.add(cited)

    return citation_events, new_dois

def main():
    filtered_papers_df = pd.read_csv(FILTERED_PAPERS_CSV)
    combined_papers_df = pd.read_csv(COMBINED_PAPERS_CSV)

    filtered_doi_set = set(filtered_papers_df['doi'].dropna().unique())
    all_aiml_doi_set = set(combined_papers_df['doi'].dropna().unique())
    rest_doi_set = all_aiml_doi_set - filtered_doi_set

    print(f"Filtered DOIs: {len(filtered_doi_set)}")
    print(f"Rest of AIML DOIs: {len(rest_doi_set)}")
    print(f"Number of hops: {NUM_HOPS}\n")

    all_citation_events = []
    expanded_dois = set(filtered_doi_set)
    frontier_dois = set(filtered_doi_set)

    for hop in range(1, NUM_HOPS + 1):
        print(f"--- Hop {hop} (frontier size: {len(frontier_dois)}) ---")
        events, new_dois = expand_one_hop(frontier_dois, rest_doi_set)
        all_citation_events.extend(events)

        # The next frontier is only the newly discovered DOIs
        newly_found = new_dois - expanded_dois
        print(f"  {len(events)} citation events, {len(newly_found)} new DOIs\n")

        expanded_dois.update(newly_found)
        rest_doi_set -= newly_found
        frontier_dois = newly_found

        if not frontier_dois:
            print("No new DOIs found, stopping early.")
            break

    print(f"Total citation events: {len(all_citation_events)}")
    print(f"Total expanded DOIs: {len(expanded_dois)}")

    if all_citation_events:
        events_df = pd.DataFrame(all_citation_events, columns=['citing_doi', 'cited_doi'])
        events_df.to_csv("../../data/aiml/aiml_cross_citation_events.csv", index=False)
        print("Saved citation events to aiml_cross_citation_events.csv")

    # Expanded dataset = filtered papers + all newly discovered papers
    expanded_papers_df = combined_papers_df[combined_papers_df['doi'].isin(expanded_dois)]
    print(f"Total papers in expanded dataset: {len(expanded_papers_df)}")

    expanded_papers_df.to_csv("../../data/aiml/expanded_neuro_aiml_papers.csv", index=False)
    print("Saved to expanded_neuro_aiml_papers.csv")

if __name__ == "__main__":
    main()
