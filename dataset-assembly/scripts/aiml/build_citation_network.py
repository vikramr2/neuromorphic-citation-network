import pandas as pd

EXPANDED_PAPERS_CSV = "../../data/aiml/expanded_neuro_aiml_papers.csv"
CITATION_EVENTS_CSV = "../../data/aiml/aiml_cross_citation_events.csv"

OUTPUT_NODES_CSV = "../../data/aiml/citation_network_nodes.csv"
OUTPUT_EDGES_CSV = "../../data/aiml/citation_network_edges.csv"

papers_df = pd.read_csv(EXPANDED_PAPERS_CSV)
events_df = pd.read_csv(CITATION_EVENTS_CSV)

# Assign integer IDs to each paper
doi_to_id = {doi: i for i, doi in enumerate(papers_df['doi'].dropna().unique())}

# Map citation events to integer IDs, dropping any that aren't in the expanded set
edges = []
skipped = 0
for _, row in events_df.iterrows():
    citing_id = doi_to_id.get(row['citing_doi'])
    cited_id = doi_to_id.get(row['cited_doi'])
    if citing_id is not None and cited_id is not None:
        edges.append((citing_id, cited_id))
    else:
        skipped += 1

# Save node mapping
nodes_df = pd.DataFrame(list(doi_to_id.items()), columns=['doi', 'id'])
nodes_df.to_csv(OUTPUT_NODES_CSV, index=False)

# Save edge list
edges_df = pd.DataFrame(edges, columns=['source', 'target'])
edges_df.to_csv(OUTPUT_EDGES_CSV, index=False)

print(f"Nodes: {len(nodes_df)}")
print(f"Edges: {len(edges_df)} ({skipped} skipped — DOI not in expanded set)")
print(f"Saved to {OUTPUT_NODES_CSV} and {OUTPUT_EDGES_CSV}")
