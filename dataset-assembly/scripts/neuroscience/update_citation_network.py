import asyncio
import json
import pandas as pd
import aiohttp
from tqdm.asyncio import tqdm_asyncio
from pathlib import Path

EDGES_PATH      = "../../../data/neuroscience/neuroscience_edges.csv"
NODES_PATH      = "../../../data/neuroscience/neuroscience_nodes_updated.csv"
OUTPUT_EDGES    = "../../../data/neuroscience/neuroscience_edges_updated.csv"
CHECKPOINT_PATH = "../../../data/neuroscience/oc_checkpoint.json"

OC_API_BASE        = "https://api.opencitations.net/index/v2"
CONCURRENT_REQUESTS = 10

current_edgelist = pd.read_csv(EDGES_PATH)

max_node_id = max(current_edgelist["source"].max(), current_edgelist["target"].max())
print(f"Current max node ID: {max_node_id}")

updated_nodes = pd.read_csv(NODES_PATH)
print(updated_nodes.tail())

new_nodes = updated_nodes[updated_nodes["id"] > max_node_id]
print(f"New nodes to add: {len(new_nodes)}")

new_node_doi_tuples = set(zip(new_nodes["id"], new_nodes["doi"]))
all_node_doi_tuples = set(zip(updated_nodes["id"], updated_nodes["doi"]))

doi_to_id = {
    doi.lower().strip(): node_id
    for node_id, doi in all_node_doi_tuples
    if isinstance(doi, str) and doi.strip()
}

new_dois = [
    doi.lower().strip()
    for _, doi in new_node_doi_tuples
    if isinstance(doi, str) and doi.strip()
]
print(f"New DOIs to query: {len(new_dois)}")


def _extract_doi(field: str) -> str | None:
    for token in field.split():
        if token.startswith("doi:"):
            return token[4:].lower().strip()
    return None


def load_checkpoint():
    cp = Path(CHECKPOINT_PATH)
    if cp.exists():
        with open(cp) as f:
            data = json.load(f)
        print(f"Loaded checkpoint: {len(data.get('references', {}))} references, "
              f"{len(data.get('citations', {}))} citations already fetched")
        return data
    return {"references": {}, "citations": {}}


def save_checkpoint(checkpoint):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f)


async def fetch_oc(session, endpoint: str, doi: str, semaphore):
    async with semaphore:
        url = f"{OC_API_BASE}/{endpoint}/doi:{doi}"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    key = "cited" if endpoint == "references" else "citing"
                    dois = [d for item in data if (d := _extract_doi(item[key])) is not None]
                    return doi, dois
                return doi, []
        except Exception as exc:
            print(f"  Warning: {endpoint} fetch failed for {doi}: {exc}")
            return doi, []


async def fetch_missing(dois, endpoint, already_done, semaphore, session):
    remaining = [d for d in dois if d not in already_done]
    if not remaining:
        print(f"  All {len(dois)} DOIs already in checkpoint for {endpoint}, skipping.")
        return {}
    print(f"  Fetching {len(remaining)} remaining DOIs (skipping {len(already_done)} cached)...")
    tasks = [fetch_oc(session, endpoint, doi, semaphore) for doi in remaining]
    results = await tqdm_asyncio.gather(*tasks, desc=f"OpenCitations {endpoint}")
    return dict(results)


async def main():
    checkpoint = load_checkpoint()

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=20)

    async with aiohttp.ClientSession(connector=connector) as session:
        print("\nFetching references for new nodes (outgoing edges)...")
        new_refs = await fetch_missing(
            new_dois, "references", checkpoint["references"], semaphore, session
        )
        checkpoint["references"].update(new_refs)
        save_checkpoint(checkpoint)

        print("\nFetching citations for new nodes (incoming edges)...")
        new_cits = await fetch_missing(
            new_dois, "citations", checkpoint["citations"], semaphore, session
        )
        checkpoint["citations"].update(new_cits)
        save_checkpoint(checkpoint)

    references = checkpoint["references"]
    citations  = checkpoint["citations"]

    new_edges = []

    # Manual addition: Dayan & Abbott (750) → all nodes 751-1103
    for i in range(751, 1104):
        new_edges.append((750, i))

    for source_doi, cited_dois in references.items():
        source_id = doi_to_id.get(source_doi)
        if source_id is None:
            continue
        for cited_doi in cited_dois:
            target_id = doi_to_id.get(cited_doi)
            if target_id is not None:
                new_edges.append((source_id, target_id))

    for target_doi, citing_dois in citations.items():
        target_id = doi_to_id.get(target_doi)
        if target_id is None:
            continue
        for citing_doi in citing_dois:
            source_id = doi_to_id.get(citing_doi)
            if source_id is not None:
                new_edges.append((source_id, target_id))

    print(f"\nNew citation events found: {len(new_edges)}")

    existing_pairs = set(zip(current_edgelist["source"], current_edgelist["target"]))
    deduped_new = [(s, t) for s, t in new_edges if (s, t) not in existing_pairs]
    print(f"New edges after deduplication: {len(deduped_new)}")

    new_edges_df = pd.DataFrame(deduped_new, columns=["source", "target"])
    combined = pd.concat([current_edgelist, new_edges_df], ignore_index=True)
    combined.to_csv(OUTPUT_EDGES, index=False)
    print(f"Combined edge list saved to {OUTPUT_EDGES} ({len(combined)} total edges)")

    # Clean up checkpoint on successful completion
    Path(CHECKPOINT_PATH).unlink(missing_ok=True)
    print("Checkpoint cleaned up.")


asyncio.run(main())
