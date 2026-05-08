import pandas as pd
import networkx as nx

ress = [
    '0p1',
    '0p01',
    '0p001',
    '0p0001',
    '0p5'
]

CLUSTER_FILENAME_TEMPLATE = "../../data/aiml/citation_network_clusters_leiden_{}.tsv"
EDGELIST_FILENAME = "../../data/aiml/citation_network_edges.csv"
NODELIST_FILENAME = "../../data/aiml/citation_network_nodes.csv"
METADATA_FILENAME = "../../data/aiml/expanded_neuro_aiml_papers.csv"
NEURO_CITATION_EVENTS_FILENAME = "../../data/aiml/neuro_aiml_citation_events.csv"

edgelist_df = pd.read_csv(EDGELIST_FILENAME)
nodelist_df = pd.read_csv(NODELIST_FILENAME)
metadata_df = pd.read_csv(METADATA_FILENAME)
neuro_citation_events_df = pd.read_csv(NEURO_CITATION_EVENTS_FILENAME)

# Get the set of DOIs in the neuro AIML citation events that are also in the metadata
neuro_dois = set(
    pd.concat([neuro_citation_events_df['citing_doi'],
               neuro_citation_events_df['cited_doi']])
    .unique()
).intersection(
    set(metadata_df['doi'].dropna().unique())
)

# Get most cited papers
most_cited = (
    edgelist_df['target']
    .value_counts()
    .head(100)
    .index
    .tolist()
)

# Get the DOI from the node list
most_cited_dois = (
    nodelist_df[nodelist_df['id'].isin(most_cited)]['doi']
    .dropna()
    .unique()
    .tolist()
)

# Get metadata for most cited papers
most_cited_metadata = (
    metadata_df[metadata_df['doi'].isin(most_cited_dois)]
    .copy()
)

neuro_node_ids = nodelist_df[nodelist_df['doi'].isin(neuro_dois)]['id'].tolist()

marker_paper_titles = [
    "Attention Is All You Need",
    "Wide Residual Networks",
    "Conditional Generative Adversarial Nets"
]

# Fetch marker paper node IDs
marker_paper_ids = []
for title in marker_paper_titles:
    row = metadata_df[metadata_df['title'] == title]
    if not row.empty:
        doi = row.iloc[0]['doi']
        node_id = nodelist_df[nodelist_df['doi'] == doi]['id'].values
        if len(node_id) > 0:
            marker_paper_ids.append(int(node_id[0]))

for res in ress:
    cluster_filename = CLUSTER_FILENAME_TEMPLATE.format(res)
    cluster_df = pd.read_csv(cluster_filename, sep='\t')

    # Get the largest clusters
    largest_clusters = (
        cluster_df['community']
        .value_counts()
        .head(5)
        .index
        .tolist()
    )

    print(f"\nTop 5 Largest Clusters at Resolution {res}:")
    for cluster in largest_clusters:
        size = cluster_df[cluster_df['community'] == cluster].shape[0]
        numnber_of_highly_cited = cluster_df[
            (cluster_df['community'] == cluster) &
            (cluster_df['node'].isin(most_cited))
        ].shape[0]
        number_of_marker_papers = cluster_df[
            (cluster_df['community'] == cluster) &
            (cluster_df['node'].isin(marker_paper_ids))
        ].shape[0]
        print(f"Cluster {cluster}, Size: {size}, Highly Cited Papers: {numnber_of_highly_cited}, Marker Papers: {number_of_marker_papers}")

    # Find the clusters that contain marker papers
    print(f"\nClusters Containing Marker Papers at Resolution {res}:")
    for marker_id in marker_paper_ids:
        cluster_info = cluster_df[cluster_df['node'] == marker_id]
        if not cluster_info.empty:
            cluster_id = cluster_info.iloc[0]['community']
            size = cluster_df[cluster_df['community'] == cluster_id].shape[0]
            print(f"Marker Paper ID {marker_id} is in Cluster {cluster_id} of Size {size}")
        else:
            print(f"Marker Paper ID {marker_id} not found in clustering data.")

# Minimum network (Steiner tree) connecting neuro, marker, and most-cited nodes
G = nx.from_pandas_edgelist(edgelist_df, source='source', target='target')

terminal_nodes = set(neuro_node_ids + marker_paper_ids + most_cited)
terminal_nodes = [n for n in terminal_nodes if G.has_node(n)]

st = nx.approximation.steiner_tree(G, terminal_nodes)
print(f"\nSteiner tree: {st.number_of_nodes()} nodes, {st.number_of_edges()} edges")

st_nodes = set(st.nodes())

print("\nExpanded network size per resolution:")
for res in ress:
    cluster_filename = CLUSTER_FILENAME_TEMPLATE.format(res)
    cluster_df = pd.read_csv(cluster_filename, sep='\t')

    # Find communities that contain at least one Steiner tree node
    communities = cluster_df[cluster_df['node'].isin(st_nodes)]['community'].unique()

    # Get all nodes in those communities
    expanded_nodes = set(cluster_df[cluster_df['community'].isin(communities)]['node'].tolist())

    expanded_subgraph = G.subgraph(expanded_nodes)
    print(f"  Resolution {res}: {len(communities)} communities, "
          f"{expanded_subgraph.number_of_nodes()} nodes, "
          f"{expanded_subgraph.number_of_edges()} edges")

print("Choosing Resolution 0.1 -> 1163 papers in expanded network.")

# Export expanded network at resolution 0.1
cluster_df_0p1 = pd.read_csv(CLUSTER_FILENAME_TEMPLATE.format('0p1'), sep='\t')
communities_0p1 = cluster_df_0p1[cluster_df_0p1['node'].isin(st_nodes)]['community'].unique()
expanded_node_ids = set(cluster_df_0p1[cluster_df_0p1['community'].isin(communities_0p1)]['node'].tolist())

expanded_nodelist = nodelist_df[nodelist_df['id'].isin(expanded_node_ids)]
expanded_edgelist = edgelist_df[
    edgelist_df['source'].isin(expanded_node_ids) &
    edgelist_df['target'].isin(expanded_node_ids)
]
expanded_dois = set(expanded_nodelist['doi'].dropna().unique())
expanded_metadata = metadata_df[metadata_df['doi'].isin(expanded_dois)]

expanded_nodelist.to_csv("../../data/aiml/expanded_aiml_nodes.csv", index=False)
expanded_edgelist.to_csv("../../data/aiml/expanded_aiml_edges.csv", index=False)
expanded_metadata.to_csv("../../data/aiml/expanded_aiml_metadata.csv", index=False)

print(f"\nSaved expanded network (res 0.1):")
print(f"  Nodes: {len(expanded_nodelist)}")
print(f"  Edges: {len(expanded_edgelist)}")
print(f"  Metadata rows: {len(expanded_metadata)}")
