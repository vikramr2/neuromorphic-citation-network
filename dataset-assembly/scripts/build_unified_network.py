"""
Build a unified cross-disciplinary citation network from three field-specific networks.

Outputs:
  data/unified_nodes.csv  — global_id, field, local_id
  data/unified_edges.csv  — source, target  (global IDs, includes cross-field edges)

Cross-field citations are discovered via the OpenCitations COCI API.
"""

import pandas as pd
import asyncio
import aiohttp
import json
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FIELDS = {
    "neuroscience": {
        "nodes": DATA_DIR / "neuroscience" / "neuroscience_nodes.csv",
        "edges": DATA_DIR / "neuroscience" / "neuroscience_edges.csv",
    },
    "neuromorphic": {
        "nodes": DATA_DIR / "neuromorphic" / "neuromorphic_nodes.csv",
        "edges": DATA_DIR / "neuromorphic" / "neuromorphic_edges.csv",
    },
    "aiml": {
        "nodes": DATA_DIR / "aiml" / "aiml_nodes.csv",
        "edges": DATA_DIR / "aiml" / "aiml_edges.csv",
    },
}

OPENCITATIONS_API = "https://opencitations.net/index/coci/api/v1"
CONCURRENCY_LIMIT = 10
CHECKPOINT_FILE = DATA_DIR / "unified_cross_field_checkpoint.json"


# ── Phase A: Build unified nodelist & remap intra-field edges ────────────────

def build_nodelist_and_intra_edges():
    """Load per-field CSVs, assign global IDs, remap edges."""
    all_nodes = []
    all_edges = []
    doi_to_global = {}
    global_to_field = {}
    offset = 0

    for field, paths in FIELDS.items():
        nodes_df = pd.read_csv(paths["nodes"])
        edges_df = pd.read_csv(paths["edges"])

        n = len(nodes_df)
        print(f"{field}: {n} nodes, {len(edges_df)} intra-field edges")

        for _, row in nodes_df.iterrows():
            local_id = int(row["id"])
            global_id = offset + local_id
            doi = row.get("doi", "")
            if pd.notna(doi) and doi:
                doi_to_global[str(doi).strip()] = global_id
                global_to_field[global_id] = field
            all_nodes.append({"global_id": global_id, "field": field, "local_id": local_id})

        for _, row in edges_df.iterrows():
            src = offset + int(row["source"])
            tgt = offset + int(row["target"])
            all_edges.append((src, tgt))

        offset += n

    nodes_df = pd.DataFrame(all_nodes)
    print(f"\nUnified nodelist: {len(nodes_df)} papers, {len(all_edges)} intra-field edges")
    return nodes_df, all_edges, doi_to_global, global_to_field


# ── Phase B: Discover cross-field citations via OpenCitations ────────────────

async def fetch_references(session, semaphore, doi):
    """Fetch the reference list for a single DOI."""
    url = f"{OPENCITATIONS_API}/references/{doi}"
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return doi, [ref.get("cited") for ref in data if ref.get("cited")]
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
    return doi, []


async def discover_cross_field_edges(doi_to_global, global_to_field):
    """Query OpenCitations for every DOI; keep edges that cross field boundaries."""
    all_dois = list(doi_to_global.keys())
    print(f"\nQuerying OpenCitations for {len(all_dois)} DOIs...")

    # Load checkpoint if it exists (resume interrupted runs)
    already_done = {}
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            already_done = json.load(f)
        print(f"  Resuming from checkpoint: {len(already_done)} DOIs already fetched")

    remaining = [d for d in all_dois if d not in already_done]
    print(f"  {len(remaining)} DOIs remaining to fetch")

    cross_edges = []
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # Process cross-edges from checkpoint data
    for citing_doi, cited_dois in already_done.items():
        citing_gid = doi_to_global.get(citing_doi)
        if citing_gid is None:
            continue
        citing_field = global_to_field.get(citing_gid)
        for cited_doi in cited_dois:
            cited_gid = doi_to_global.get(cited_doi)
            if cited_gid is not None:
                cited_field = global_to_field.get(cited_gid)
                if citing_field != cited_field:
                    cross_edges.append((citing_gid, cited_gid))

    # Fetch remaining DOIs in batches to checkpoint periodically
    BATCH_SIZE = 200
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start : batch_start + BATCH_SIZE]

        async with aiohttp.ClientSession() as session:
            tasks = [fetch_references(session, semaphore, doi) for doi in batch]
            results = await tqdm_asyncio.gather(
                *tasks, desc=f"Batch {batch_start // BATCH_SIZE + 1}"
            )

        for citing_doi, cited_dois in results:
            already_done[citing_doi] = cited_dois
            citing_gid = doi_to_global.get(citing_doi)
            if citing_gid is None:
                continue
            citing_field = global_to_field.get(citing_gid)
            for cited_doi in cited_dois:
                cited_gid = doi_to_global.get(cited_doi)
                if cited_gid is not None:
                    cited_field = global_to_field.get(cited_gid)
                    if citing_field != cited_field:
                        cross_edges.append((citing_gid, cited_gid))

        # Save checkpoint
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(already_done, f)

    print(f"\nCross-field edges discovered: {len(cross_edges)}")
    return cross_edges


# ── Phase C: Merge and write outputs ─────────────────────────────────────────

def write_outputs(nodes_df, intra_edges, cross_edges):
    """Deduplicate all edges and write final CSVs."""
    all_edges = set(intra_edges) | set(cross_edges)

    edges_df = pd.DataFrame(sorted(all_edges), columns=["source", "target"])

    nodes_path = DATA_DIR / "unified_nodes.csv"
    edges_path = DATA_DIR / "unified_edges.csv"

    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)

    # Summary stats
    intra_count = len(set(intra_edges))
    cross_count = len(set(cross_edges))
    print(f"\n{'='*50}")
    print(f"Unified nodes:  {len(nodes_df)}")
    print(f"Intra-field edges: {intra_count}")
    print(f"Cross-field edges: {cross_count}")
    print(f"Total edges:       {len(edges_df)}")

    # Cross-field breakdown
    if cross_edges:
        from collections import Counter
        pair_counts = Counter()
        nodes_lookup = nodes_df.set_index("global_id")["field"]
        for src, tgt in set(cross_edges):
            pair = tuple(sorted([nodes_lookup[src], nodes_lookup[tgt]]))
            pair_counts[pair] += 1
        print("\nCross-field edge breakdown:")
        for (f1, f2), count in pair_counts.most_common():
            print(f"  {f1} <-> {f2}: {count}")

    print(f"\nSaved to:\n  {nodes_path}\n  {edges_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    nodes_df, intra_edges, doi_to_global, global_to_field = build_nodelist_and_intra_edges()
    cross_edges = asyncio.run(discover_cross_field_edges(doi_to_global, global_to_field))
    write_outputs(nodes_df, intra_edges, cross_edges)


if __name__ == "__main__":
    main()
