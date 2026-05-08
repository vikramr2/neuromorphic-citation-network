import pandas as pd

NEUROSCIENCE_PAPERS_CSV = "../../data/neuroscience/expanded_neuroscience_papers_with_ids.csv"
NEUROSCIENCE_PAPERS_EDGELIST_CSV = "../../data/neuroscience/expanded_neuroscience_edges.csv"

FINAL_NEUROSCIENCE_PAPERS_CSV = "../../data/neuroscience/final_neuroscience_papers_with_ids.csv"
FINAL_NEUROSCIENCE_PAPERS_EDGELIST_CSV = "../../data/neuroscience/final_neuroscience_edges.csv"

neuroscience_papers_df = pd.read_csv(NEUROSCIENCE_PAPERS_CSV)
neuroscience_papers_edgelist_df = pd.read_csv(NEUROSCIENCE_PAPERS_EDGELIST_CSV)

print(f"Initial number of neuroscience papers: {len(neuroscience_papers_df)}")
print(f"Initial number of edges in edgelist: {len(neuroscience_papers_edgelist_df)}")

neuroscience_papers_df = neuroscience_papers_df[neuroscience_papers_df["pdf_url"] != "unable"]
remaining_ids = set(neuroscience_papers_df["id"].tolist())

neuroscience_papers_edgelist_df = neuroscience_papers_edgelist_df[
    neuroscience_papers_edgelist_df["source"].isin(remaining_ids) &
    neuroscience_papers_edgelist_df["target"].isin(remaining_ids)
]

print(f"Number of neuroscience papers after filtering: {len(neuroscience_papers_df)}")
print(f"Number of edges in edgelist after filtering: {len(neuroscience_papers_edgelist_df)}")

# Get the largest connected component
import networkx as nx

G = nx.from_pandas_edgelist(neuroscience_papers_edgelist_df, source="source", target="target")
largest_cc = max(nx.connected_components(G), key=len)
largest_cc_ids = set(largest_cc)

neuroscience_papers_df = neuroscience_papers_df[neuroscience_papers_df["id"].isin(largest_cc_ids)]
neuroscience_papers_edgelist_df = neuroscience_papers_edgelist_df[
    neuroscience_papers_edgelist_df["source"].isin(largest_cc_ids) &
    neuroscience_papers_edgelist_df["target"].isin(largest_cc_ids)
]

print(f"Number of neuroscience papers in largest connected component: {len(neuroscience_papers_df)}")
print(f"Number of edges in edgelist in largest connected component: {len(neuroscience_papers_edgelist_df)}")

# Save the filtered dataframes back to CSV
neuroscience_papers_df.to_csv(FINAL_NEUROSCIENCE_PAPERS_CSV, index=False)
neuroscience_papers_edgelist_df.to_csv(FINAL_NEUROSCIENCE_PAPERS_EDGELIST_CSV, index=False)
