import pandas as pd
from typing import List

DATA_DIR = "../data/"

CLUSTERING_PREFIX = DATA_DIR + "neuromorphic_communities_"

SUFFIXES = [
    "0p01.tsv",
    "0p001.tsv",
    "0p0001.tsv",
    "mod.tsv"
]

DOI_MAPPING = DATA_DIR + "neuromorphic_nodelist.csv"

def get_largest_community(clustering_path: str) -> List[int]:
    """
    Given a clustering file path, return the largest community as a dataframe.

    Args:
        clustering_path: Path to the clustering TSV file
    Returns:
        List of node IDs in the largest community
    """
    clustering_df = pd.read_csv(clustering_path, sep="\t")
    largest_community_id = clustering_df['community'].value_counts().idxmax()
    largest_community_nodes = clustering_df[clustering_df['community'] == largest_community_id]['node'].tolist()
    return largest_community_nodes

def get_doi_from_ids(node_ids: List[int], doi_mapping_df: pd.DataFrame) -> List[str]:
    """
    Given a list of node IDs, return the corresponding DOIs.

    Args:
        node_ids: List of node IDs
        doi_mapping_df: DataFrame mapping node IDs to DOIs
    Returns:
        List of DOIs corresponding to the node IDs
    """
    dois = doi_mapping_df[doi_mapping_df['id'].isin(node_ids)]['doi'].tolist()
    return dois

if __name__ == "__main__":
    # Print size of largest community for each clustering
    doi_mapping_df = pd.read_csv(DOI_MAPPING)
    for suffix in SUFFIXES:
        clustering_path = CLUSTERING_PREFIX + suffix
        largest_community_nodes = get_largest_community(clustering_path)
        largest_community_dois = get_doi_from_ids(largest_community_nodes, doi_mapping_df)
        print(f"Largest community in {suffix} has {len(largest_community_nodes)} nodes")
        