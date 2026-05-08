import pandas as pd
import requests
import time
from tqdm import tqdm

MARKER_TEXT_FILE = "../../data/neuroscience/marker_papers.txt"
NODELIST = "../../data/neuroscience/final_neuroscience_papers_with_ids.csv"
EDGE_LIST = "../../data/neuroscience/final_neuroscience_edges.csv"

S2_API = "https://api.semanticscholar.org/graph/v1"
REQUEST_DELAY = 3
MAX_RETRIES = 5

def load_marker_papers(marker_file):
    with open(marker_file, "r") as f:
        markers = [line.strip() for line in f if line.strip()]
    return markers

def fetch_metadata_from_crossref(doi):
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url)
    if response.status_code != 200:
        return "", "", "", ""

    data = response.json().get("message", {})
    title = data.get("title", [""])[0]
    abstract = data.get("abstract", "")
    authors_list = data.get("author", [])
    authors = "; ".join(
        f"{a.get('given', '')} {a.get('family', '')}".strip() for a in authors_list
    )
    publication_date = data.get("published-print", data.get("published-online", {}))
    date_parts = publication_date.get("date-parts", [[]])
    date = ""
    if date_parts and date_parts[0]:
        date = "-".join(str(part) for part in date_parts[0])
    return title, abstract, authors, date

def s2_get_related_dois(doi, endpoint):
    """
    Fetch DOIs from a Semantic Scholar references or citations endpoint.
    endpoint: "references" or "citations"
    """
    s2_id = f"DOI:{doi}"
    results = []
    offset = 0
    retries = 0
    paper_key = "citedPaper" if endpoint == "references" else "citingPaper"
    while True:
        url = f"{S2_API}/paper/{s2_id}/{endpoint}?fields=externalIds&limit=1000&offset={offset}"
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 429:
                retries += 1
                if retries > MAX_RETRIES:
                    print(f"  [WARN] {endpoint} {doi}: rate limited, skipping")
                    break
                time.sleep(5)
                continue
            retries = 0
            if response.status_code != 200:
                print(f"  [WARN] {endpoint} {doi}: HTTP {response.status_code}")
                break
            data = response.json()
            for item in (data.get("data") or []):
                ext_ids = item.get(paper_key, {}).get("externalIds") or {}
                ref_doi = ext_ids.get("DOI")
                if ref_doi:
                    results.append(ref_doi)
            if "next" not in data:
                break
            offset = data["next"]
        except requests.RequestException as e:
            print(f"  [ERR] {endpoint} {doi}: {e}")
            break
    time.sleep(REQUEST_DELAY)
    return results

def identify_citations_to_list(new_id, new_doi, nodelist_df):
    """
    Find citation edges between a new paper and the existing nodelist
    using the Semantic Scholar API.

    Returns a list of (source, target) tuples where source cites target.
    """
    existing_dois = set(nodelist_df['doi'].dropna().unique())
    doi_to_id = dict(zip(nodelist_df['doi'], nodelist_df['id']))
    edges = []

    # Papers the new paper cites (references): new_id -> existing_id
    for ref_doi in s2_get_related_dois(new_doi, "references"):
        if ref_doi in existing_dois:
            edges.append((new_id, doi_to_id[ref_doi]))

    # Papers that cite the new paper (citations): existing_id -> new_id
    for citing_doi in s2_get_related_dois(new_doi, "citations"):
        if citing_doi in existing_dois:
            edges.append((doi_to_id[citing_doi], new_id))

    return edges

marker_papers = load_marker_papers(MARKER_TEXT_FILE)

print(f"Loaded {len(marker_papers)} marker papers.")

# Load nodelist and edge list
nodelist_df = pd.read_csv(NODELIST)
edgelist_df = pd.read_csv(EDGE_LIST)

# Filter out marker papers that are already in the nodelist
existing_dois = set(nodelist_df['doi'].dropna().unique())
new_marker_papers = [doi for doi in marker_papers if doi not in existing_dois]

print(f"Found {len(new_marker_papers)} new marker papers to add.")
print(new_marker_papers)

final_node_id = nodelist_df['id'].max() + 1 if not nodelist_df.empty else 1
new_nodes = []
for doi in tqdm(new_marker_papers, desc="Fetching metadata"):
    title, abstract, authors, date = fetch_metadata_from_crossref(doi)

    new_nodes.append({
        'id': final_node_id,
        'doi': doi,
        'title': title,
        'abstract': abstract,
        'pdf_url': 'dropbox',
        'authors': authors,
        'date': date,
        'category': 'neuroscience',
    })
    final_node_id += 1

new_nodes_df = pd.DataFrame(new_nodes)

print(f"Fetched metadata for {len(new_nodes_df)} new nodes.")

# Add new nodes to the nodelist so citation lookups can see earlier marker papers too
updated_nodelist_df = pd.concat([nodelist_df, new_nodes_df], ignore_index=True)

# Find citation edges between each new paper and the full nodelist
new_edges = []
for _, row in tqdm(new_nodes_df.iterrows(), total=len(new_nodes_df), desc="Finding citations"):
    edges = identify_citations_to_list(row['id'], row['doi'], updated_nodelist_df)
    new_edges.extend(edges)
    tqdm.write(f"  {row['title'][:60]}... → {len(edges)} edges")

new_edges_df = pd.DataFrame(new_edges, columns=['source', 'target'])

print(f"Identified {len(new_edges_df)} new edges between marker papers and existing nodelist.")

# Merge and save
final_nodelist = updated_nodelist_df
final_edgelist = pd.concat([edgelist_df, new_edges_df], ignore_index=True)

final_nodelist.to_csv(NODELIST, index=False)
final_edgelist.to_csv(EDGE_LIST, index=False)

print(f"\nAdded {len(new_nodes_df)} nodes and {len(new_edges_df)} edges")
print(f"Nodelist: {len(final_nodelist)} total nodes")
print(f"Edgelist: {len(final_edgelist)} total edges")
