#!/usr/bin/env python3
"""
Knowledge Graph Visualization Script

This script extracts a subgraph from the merged predications and creates a visualization.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_predications(file_path: Path) -> pd.DataFrame:
    """Load predications from the merged file."""
    # Assuming pipe-delimited format
    df = pd.read_csv(file_path, sep='|', dtype=str)
    
    # Filter out rows with empty or whitespace-only entity names
    df = df.dropna(subset=['subject_entity_name', 'object_entity_name'])
    df = df[
        (df['subject_entity_name'].str.strip() != '') &
        (df['object_entity_name'].str.strip() != '')
    ]
    
    # Filter out entities that don't contain human-readable information
    # Keep only entities with at least 2 alphanumeric characters
    import re
    def is_human_readable(text):
        if not text or not isinstance(text, str):
            return False
        # Check if it has at least 2 alphanumeric characters
        alphanumeric_count = len(re.findall(r'[a-zA-Z0-9]', text))
        return alphanumeric_count >= 2
    
    df = df[
        df['subject_entity_name'].apply(is_human_readable) &
        df['object_entity_name'].apply(is_human_readable)
    ]
    
    return df

def extract_subgraph(df: pd.DataFrame, max_nodes: int = 100) -> nx.DiGraph:
    """Extract a subgraph with top entities by frequency."""
    import re
    
    def is_human_readable(text):
        if not text or not isinstance(text, str):
            return False
        # Check if it has at least 2 alphanumeric characters
        alphanumeric_count = len(re.findall(r'[a-zA-Z0-9]', text))
        return alphanumeric_count >= 2
    
    # Filter out empty entity names from counting
    valid_subjects = df['subject_entity_name'].apply(is_human_readable)
    valid_objects = df['object_entity_name'].apply(is_human_readable)
    valid_df = df[valid_subjects & valid_objects]
    
    # Count entity frequencies
    subject_counts = valid_df['subject_entity_name'].value_counts()
    object_counts = valid_df['object_entity_name'].value_counts()
    entity_counts = subject_counts.add(object_counts, fill_value=0).sort_values(ascending=False)

    # Select top entities (excluding non-human-readable strings)
    top_entities = [entity for entity in entity_counts.head(max_nodes).index.tolist() 
                   if is_human_readable(entity)]

    # Filter predications to only include top entities
    subgraph_df = valid_df[
        valid_df['subject_entity_name'].isin(top_entities) &
        valid_df['object_entity_name'].isin(top_entities)
    ]

    # Build NetworkX graph
    G = nx.DiGraph()
    for _, row in subgraph_df.iterrows():
        subject = row['subject_entity_name'].strip()
        obj = row['object_entity_name'].strip()
        if is_human_readable(subject) and is_human_readable(obj):  # Double-check
            G.add_edge(subject, obj, relation=row['predicate'])

    return G

def visualize_graph(G: nx.DiGraph, output_path: Path):
    """Create and save graph visualization."""
    # Remove isolated nodes (degree 0) as they appear as empty nodes
    G.remove_nodes_from(list(nx.isolates(G)))
    
    if len(G.nodes) == 0:
        print("Warning: No connected nodes remaining after filtering isolates")
        return
    
    plt.figure(figsize=(12, 12))

    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', alpha=0.7)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, arrows=True, arrowsize=10)

    # Draw labels for all nodes (not just high-degree ones)
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

    # Draw edge labels (relations) - only for edges between nodes
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        edge_labels[(u, v)] = data.get('relation', '')

    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)

    plt.title('Knowledge Graph Subgraph Visualization\n(Top 50 Entities by Frequency)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize Knowledge Graph Subgraph')
    parser.add_argument('--input', type=Path, default=Path('/scratch/ramki/knight/approach1/kg-builder/output_triple/merged/merged_predications.txt'),
                        help='Path to merged predications file')
    parser.add_argument('--output', type=Path, default=Path('/scratch/ramki/knight/approach1/paper/figs/kg-visualization.png'),
                        help='Output path for visualization')
    parser.add_argument('--max-nodes', type=int, default=50,
                        help='Maximum number of nodes in subgraph')

    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading predications...")
    df = load_predications(args.input)
    print(f"Loaded {len(df)} predications")

    print("Extracting subgraph...")
    G = extract_subgraph(df, args.max_nodes)
    print(f"Subgraph has {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Debug: show some sample nodes
    sample_nodes = list(G.nodes())[:10]
    print(f"Sample nodes: {sample_nodes}")

    print("Creating visualization...")
    visualize_graph(G, args.output)

if __name__ == '__main__':
    main()