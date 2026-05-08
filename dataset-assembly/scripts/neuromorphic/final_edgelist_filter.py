import pandas as pd
# import json

# NEUROMORPHIC_PAPERS_JSON = "../../data/neuromorphic/neuromorphic_papers_cleaned_enhanced.json"

# with open(NEUROMORPHIC_PAPERS_JSON, 'r') as f:
#     neuromorphic_papers = json.load(f)

# print(f"Total neuromorphic papers loaded: {len(neuromorphic_papers)}")

EDGELIST_CSV = "../../data/neuromorphic/neuromorphic_edgelist.csv"
NODELIST_CSV = "../../data/neuromorphic/neuromorphic_nodelist.csv"
CLUSTERS_CSV = "../../data/neuromorphic/neuromorphic_communities_0p0001.tsv"

edgelist_df = pd.read_csv(EDGELIST_CSV)
nodelist_df = pd.read_csv(NODELIST_CSV)
clusters_df = pd.read_csv(CLUSTERS_CSV, sep='\t')

print(f"Original edgelist size: {len(edgelist_df)}")
print(f"Original nodelist size: {len(nodelist_df)}")
print(f"Original clusters size: {len(clusters_df)}")

# Get the set of node IDs that are in the largest cluster
largest_cluster_id = clusters_df['community'].value_counts().idxmax()
nodes_in_largest_cluster = set(clusters_df[clusters_df['community'] == largest_cluster_id]['node'])
print(f"Largest cluster ID: {largest_cluster_id} with {len(nodes_in_largest_cluster)} nodes")

# Filter edgelist to only include edges where both source and target are in the largest cluster
filtered_edgelist_df = edgelist_df[
    (edgelist_df['source'].isin(nodes_in_largest_cluster)) &
    (edgelist_df['target'].isin(nodes_in_largest_cluster))
]
print(f"Filtered edgelist size: {len(filtered_edgelist_df)}")

# Filter nodelist to only include nodes in the largest cluster
filtered_nodelist_df = nodelist_df[nodelist_df['id'].isin(nodes_in_largest_cluster)]
print(f"Filtered nodelist size: {len(filtered_nodelist_df)}")   

# Save as the same nale but _final suffix
filtered_edgelist_df.to_csv(EDGELIST_CSV.replace('.csv', '_final.csv'), index=False)
filtered_nodelist_df.to_csv(NODELIST_CSV.replace('.csv', '_final.csv'), index=False)
