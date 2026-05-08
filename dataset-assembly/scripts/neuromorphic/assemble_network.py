import pandas as pd
import requests
import time
from typing import List, Set
from tqdm import tqdm

DATA_DIR = "../data/"

# Load neuromorphic title and doi data
neuromorphic_title_dois = pd.read_csv(DATA_DIR + "combined_neuromorphic_title_dois.csv")

# Remove duplicate DOIs
print(f"Neuromorphic dataset has {len(neuromorphic_title_dois)} entries before removing duplicates.")
neuromorphic_title_dois = neuromorphic_title_dois.drop_duplicates(subset=['doi'])
print(f"After removing duplicate DOIs: {len(neuromorphic_title_dois)} entries.")

print(f"Neuromorphic dataset has {len(neuromorphic_title_dois)} entries.")

# Filter out rows with missing DOIs
neuromorphic_title_dois = neuromorphic_title_dois.dropna(subset=['doi'])
print(f"After filtering missing DOIs: {len(neuromorphic_title_dois)} entries.")

dois_with_int_ids = list(enumerate(neuromorphic_title_dois['doi']))

nodelist_df = pd.DataFrame(dois_with_int_ids, columns=['id', 'doi'])

# Save the nodelist dataframe
nodelist_output_path = DATA_DIR + "neuromorphic_nodelist.csv"
nodelist_df.to_csv(nodelist_output_path, index=False)
print(f"Saved nodelist to {nodelist_output_path}")

# Create DOI to ID mapping for fast lookup
doi_to_id = {doi: idx for idx, doi in dois_with_int_ids}

print(f"\n=== Building Citation Network ===")
print(f"Total nodes: {len(doi_to_id)}\n")


def get_references_from_crossref(doi: str) -> List[str]:
    """
    Get list of DOIs that a paper references using Crossref API.

    Args:
        doi: DOI of the paper

    Returns:
        List of referenced DOIs
    """
    try:
        # Normalize DOI
        doi = doi.strip()

        url = f"https://api.crossref.org/works/{doi}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (mailto:your-email@example.com)'
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        data = response.json()

        # Extract references
        references = data.get('message', {}).get('reference', [])

        # Extract DOIs from references
        ref_dois = []
        for ref in references:
            ref_doi = ref.get('DOI')
            if ref_doi:
                ref_dois.append(ref_doi)

        return ref_dois

    except Exception as e:
        # Silently fail for individual papers
        return []


def get_citations_from_opencitations(doi: str) -> List[str]:
    """
    Get list of DOIs that cite a paper using OpenCitations API.

    Args:
        doi: DOI of the paper

    Returns:
        List of citing DOIs
    """
    try:
        # Normalize DOI
        doi = doi.strip()

        url = f"https://opencitations.net/index/coci/api/v1/citations/{doi}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (mailto:your-email@example.com)'
        }

        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code == 404:
            # No citations found
            return []

        response.raise_for_status()

        data = response.json()

        # Extract citing DOIs
        citing_dois = [item.get('citing') for item in data if item.get('citing')]

        return citing_dois

    except Exception as e:
        # Silently fail for individual papers
        return []


# Build edgelist by fetching references for each paper
edges = []
citation_stats = {
    'internal_references': 0,  # References within our dataset
    'external_references': 0,   # References outside our dataset
    'total_references_found': 0
}

print("Fetching references from Crossref API...")
for idx, doi in tqdm(dois_with_int_ids, desc="Processing papers"):
    source_id = idx

    # Get papers this paper references
    referenced_dois = get_references_from_crossref(doi)
    citation_stats['total_references_found'] += len(referenced_dois)

    for ref_doi in referenced_dois:
        # Check if the referenced paper is in our dataset
        if ref_doi in doi_to_id:
            target_id = doi_to_id[ref_doi]
            # Add edge: source cites target (source -> target)
            edges.append((source_id, target_id))
            citation_stats['internal_references'] += 1
        else:
            citation_stats['external_references'] += 1

    # Rate limiting - be respectful to APIs
    time.sleep(0.5)

# Create edgelist dataframe
edgelist_df = pd.DataFrame(edges, columns=['source', 'target'])

# Remove self-loops if any
edgelist_df = edgelist_df[edgelist_df['source'] != edgelist_df['target']]

# Remove duplicate edges
edgelist_df = edgelist_df.drop_duplicates()

print(f"\n=== Citation Network Statistics ===")
print(f"Total references found: {citation_stats['total_references_found']}")
print(f"Internal references (within dataset): {citation_stats['internal_references']}")
print(f"External references (outside dataset): {citation_stats['external_references']}")
print(f"Total edges in network: {len(edgelist_df)}")
print(f"Total nodes: {len(doi_to_id)}")
print(f"Average degree: {2 * len(edgelist_df) / len(doi_to_id):.2f}")

# Save the edgelist
edgelist_output_path = DATA_DIR + "neuromorphic_edgelist.csv"
edgelist_df.to_csv(edgelist_output_path, index=False)
print(f"\nSaved edgelist to {edgelist_output_path}")

# Optional: Create a summary of the network
print(f"\nSample edges:")
print(edgelist_df.head(10))

# Save network statistics
stats_df = pd.DataFrame([{
    'num_nodes': len(doi_to_id),
    'num_edges': len(edgelist_df),
    'total_references_found': citation_stats['total_references_found'],
    'internal_references': citation_stats['internal_references'],
    'external_references': citation_stats['external_references'],
    'avg_degree': 2 * len(edgelist_df) / len(doi_to_id) if len(doi_to_id) > 0 else 0
}])

stats_output_path = DATA_DIR + "neuromorphic_network_statistics.csv"
stats_df.to_csv(stats_output_path, index=False)
print(f"Saved network statistics to {stats_output_path}")

