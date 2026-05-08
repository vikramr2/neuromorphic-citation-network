"""Visualize the unified cross-disciplinary citation network, colored by field."""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

nodes_df = pd.read_csv(DATA_DIR / "unified_nodes.csv")
edges_df = pd.read_csv(DATA_DIR / "unified_edges.csv")

G = nx.DiGraph()
field_colors = {
    "neuroscience": "#4C72B0",
    "neuromorphic": "#DD8452",
    "aiml": "#55A868",
}

for _, row in nodes_df.iterrows():
    G.add_node(row["global_id"], field=row["field"])

for _, row in edges_df.iterrows():
    G.add_edge(row["source"], row["target"])

# Remove isolated nodes for a cleaner layout
isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)
print(f"Plotting {G.number_of_nodes()} connected nodes, {G.number_of_edges()} edges ({len(isolates)} isolates hidden)")

node_colors = [field_colors[G.nodes[n]["field"]] for n in G.nodes]
degrees = dict(G.degree())
node_sizes = [max(degrees[n] * 3, 5) for n in G.nodes]

# Identify cross-field edges
cross_edges = [(u, v) for u, v in G.edges()
               if G.nodes[u]["field"] != G.nodes[v]["field"]]
intra_edges = [(u, v) for u, v in G.edges()
               if G.nodes[u]["field"] == G.nodes[v]["field"]]

print(f"Cross-field edges visible: {len(cross_edges)}")

fig, ax = plt.subplots(figsize=(20, 16))

print("Computing layout (this may take a moment)...")
pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

# Draw intra-field edges faintly
nx.draw_networkx_edges(G, pos, edgelist=intra_edges,
                       alpha=0.15, width=0.3, edge_color="#999999",
                       arrows=False, ax=ax)

# Draw cross-field edges prominently
nx.draw_networkx_edges(G, pos, edgelist=cross_edges,
                       alpha=0.5, width=0.8, edge_color="#E05555",
                       arrows=False, ax=ax)

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                       alpha=0.7, linewidths=0, ax=ax)

# Legend
for field, color in field_colors.items():
    ax.scatter([], [], c=color, s=80, label=field)
ax.scatter([], [], c="#E05555", s=40, marker="_", linewidths=2, label="cross-field citation")
ax.legend(fontsize=14, loc="upper left", framealpha=0.9)

ax.set_title("Unified Citation Network (colored by field)", fontsize=18)
ax.axis("off")
plt.tight_layout()

out_path = DATA_DIR / "unified_network.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {out_path}")
plt.close()
