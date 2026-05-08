import pandas as pd
import sys

# Usage:
#   python reindex_network.py <nodes_csv> <edges_csv> [metadata_csv]
#
# Examples:
#   python reindex_network.py data/aiml/expanded_aiml_nodes.csv data/aiml/expanded_aiml_edges.csv data/aiml/expanded_aiml_metadata.csv
#   python reindex_network.py data/neuromorphic/neuromorphic_nodelist_with_metadata.csv data/neuromorphic/neuromorphic_edgelist_final.csv

if len(sys.argv) < 3:
    print("Usage: python reindex_network.py <nodes_csv> <edges_csv> [metadata_csv]")
    sys.exit(1)

NODES_CSV = sys.argv[1]
EDGES_CSV = sys.argv[2]
METADATA_CSV = sys.argv[3] if len(sys.argv) > 3 else None

nodes_df = pd.read_csv(NODES_CSV)
edges_df = pd.read_csv(EDGES_CSV)

# Build old_id -> new_id mapping (continuous from 0)
old_ids = sorted(nodes_df['id'].unique())
id_map = {old: new for new, old in enumerate(old_ids)}

# Reindex nodes
nodes_df['id'] = nodes_df['id'].map(id_map)

# Reindex edges
edges_df['source'] = edges_df['source'].map(id_map)
edges_df['target'] = edges_df['target'].map(id_map)

# Save to new files with _reindexed suffix
def reindexed_path(path):
    from pathlib import Path
    p = Path(path)
    return str(p.with_stem(p.stem + "_reindexed"))

nodes_df.to_csv(reindexed_path(NODES_CSV), index=False)
edges_df.to_csv(reindexed_path(EDGES_CSV), index=False)

print(f"Nodes: {len(nodes_df)} (IDs 0 to {nodes_df['id'].max()})")
print(f"Edges: {len(edges_df)}")

# Optional: join node IDs into a separate metadata CSV via DOI
if METADATA_CSV:
    metadata_df = pd.read_csv(METADATA_CSV)
    metadata_df = metadata_df.merge(nodes_df[['doi', 'id']], on='doi', how='left')
    metadata_df.to_csv(reindexed_path(METADATA_CSV), index=False)
    print(f"Metadata: {len(metadata_df)} rows, {metadata_df['id'].notna().sum()} with node IDs")
