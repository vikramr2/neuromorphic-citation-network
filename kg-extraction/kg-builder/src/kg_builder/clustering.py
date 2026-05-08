import pandas as pd
from pathlib import Path
import networkx as nx
from community import community_louvain
from collections import defaultdict
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def build_graph(triples: list) -> nx.DiGraph:
    G = nx.DiGraph()
    for triple in triples:
        # Handle both old format (h,r,t) and new format (subject_entity_name, predicate, object_entity_name)
        if 'h' in triple and 't' in triple:
            h = triple['h']
            t = triple['t']
            r = triple.get('r', 'related_to')
        elif 'subject_entity_name' in triple and 'object_entity_name' in triple:
            h = triple['subject_entity_name']
            t = triple['object_entity_name']
            r = triple.get('predicate', 'related_to')
        else:
            logging.warning(f"Skipping triple with unknown format: {triple}")
            continue

        # Skip triples with None or empty head/tail
        if not h or not t:
            logging.warning(f"Skipping invalid triple: {triple}")
            continue
            
        # Prepare edge attributes
        edge_attrs = {'relation': r}
        
        # Add document_id if present
        if 'document_id' in triple:
            edge_attrs['document_id'] = triple['document_id']
            
        # Add other potentially useful attributes
        for key in ['predication_id', 'model_name', 'confidence', 'polarity']:
            if key in triple and pd.notna(triple[key]):
                edge_attrs[key] = triple[key]

        G.add_edge(h, t, **edge_attrs)
    return G


def render_graph_image(G: nx.DiGraph, image_path: Path, title: str = "Knowledge Graph"):
    """
    Render the graph as a PNG image using matplotlib.

    Args:
        G: NetworkX graph to render
        image_path: Path where to save the image
        title: Title for the graph visualization
    """
    plt.figure(figsize=(16, 12))

    # Calculate node positions using spring layout
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)

    # Get node degrees for sizing
    degrees = dict(G.degree())
    node_sizes = [max(300, degrees[node] * 100) for node in G.nodes()]

    # Color nodes by degree
    node_colors = [degrees[node] for node in G.nodes()]

    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos,
                                   node_size=node_sizes,
                                   node_color=node_colors,
                                   cmap=plt.cm.viridis,
                                   alpha=0.7)

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', arrows=True, arrowsize=10)

    # Draw labels for important nodes (high degree)
    important_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
    labels = {node: node[:30] + '...' if len(node) > 30 else node
              for node, degree in important_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

    # Add colorbar
    plt.colorbar(nodes, label='Node Degree')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')

    # Save the image
    plt.savefig(image_path, dpi=300, bbox_inches='tight', format=image_path.suffix[1:])
    plt.close()

    logging.info(f"Saved graph visualization to {image_path}")


def render_graphml_to_image(graphml_path: Path, image_path: Path, title: str = "Knowledge Graph"):
    """
    Load a GraphML file and render it as an image.

    Args:
        graphml_path: Path to the GraphML file
        image_path: Path where to save the image
        title: Title for the graph visualization
    """
    if not graphml_path.exists():
        logging.error(f"GraphML file not found: {graphml_path}")
        return

    # Load the graph from GraphML
    G = nx.read_graphml(graphml_path)

    # Render the image
    render_graph_image(G, image_path, title)

    logging.info(f"Rendered {graphml_path} to {image_path}")


def cluster_graph(G: nx.DiGraph, min_cluster_size: int) -> nx.DiGraph:
    # Convert to undirected for community detection
    G_undir = G.to_undirected()
    partition = community_louvain.best_partition(G_undir)

    # Group nodes by community
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)

    # Retain only large communities
    large_nodes = set()
    for comm, nodes in communities.items():
        if len(nodes) >= min_cluster_size:
            large_nodes.update(nodes)

    G_refined = G.subgraph(large_nodes).copy()
    logging.info(f"Refined graph from {len(G)} to {len(G_refined)} nodes")
    return G_refined


def clustering(merged_dir: Path, refined_path: Path, min_cluster_size: int, enable_visualization: bool = False, track_deleted_entities: bool = True):
    deduped_txt  = merged_dir / "deduped_predications.txt"
    deduped_jsonl = merged_dir / "deduped.jsonl"

    if deduped_txt.exists():
        df = pd.read_csv(deduped_txt, sep='|', encoding='utf-8', engine='python')
        triples = df.to_dict('records')
    elif deduped_jsonl.exists():
        import json
        with open(deduped_jsonl, encoding='utf-8') as f:
            triples = [json.loads(l) for l in f if l.strip()]
        logging.info(f"Loaded {len(triples)} triples from {deduped_jsonl}")
    else:
        logging.error(f"No deduped file found: tried {deduped_txt} and {deduped_jsonl}")
        return
    G = build_graph(triples)
    
    # Track original nodes before clustering
    original_nodes = set(G.nodes())
    
    G_refined = cluster_graph(G, min_cluster_size)
    
    # Track deleted entities if enabled
    if track_deleted_entities:
        deleted_nodes = original_nodes - set(G_refined.nodes())
        if deleted_nodes:
            deleted_entities_path = merged_dir / "deleted_entities.txt"
            with open(deleted_entities_path, 'w', encoding='utf-8') as f:
                f.write("entity_name\n")  # Header
                for node in sorted(deleted_nodes):
                    f.write(f"{node}\n")
            logging.info(f"Saved {len(deleted_nodes)} deleted entities to {deleted_entities_path}")
    
    nx.write_graphml(G_refined, refined_path)
    logging.info(f"Saved refined graph to {refined_path}")

    # Generate image visualization only if enabled
    if enable_visualization:
        image_path = refined_path.with_suffix('.png')
        render_graph_image(G_refined, image_path, f"Knowledge Graph (Refined, {len(G_refined)} nodes)")

        # Also generate JPG version
        jpg_path = refined_path.with_suffix('.jpg')
        render_graph_image(G_refined, jpg_path, f"Knowledge Graph (Refined, {len(G_refined)} nodes)")
    else:
        logging.info("Graph visualization disabled - skipping image generation")
