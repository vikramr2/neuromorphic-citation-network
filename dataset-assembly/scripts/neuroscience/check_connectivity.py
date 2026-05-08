"""
Find common DOIs cited by both neuroscience and neuromorphic papers.
Uses async requests for speed.
"""

import json
import pandas as pd
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio
from collections import defaultdict

NEUROSCIENCE_PAPERS = "../../data/biorxiv_papers/neuroscience_papers_complete.csv"
NEUROMORPHIC_PAPERS = "../../data/neuromorphic/neuromorphic_papers_cleaned_enhanced.json"

OC_API_BASE = "https://opencitations.net/index/api/v2"
CONCURRENT_REQUESTS = 10  # OpenCitations rate limit


async def get_references(session, doi, semaphore):
    """Get all DOIs that the given DOI references using OpenCitations API."""
    async with semaphore:
        url = f"{OC_API_BASE}/references/doi:{doi}"
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return doi, [item["cited"].replace("doi:", "") for item in data]
                return doi, []
        except Exception as e:
            print(f"Error fetching references for {doi}: {e}")
            return doi, []


async def fetch_all_references(dois, desc):
    """Fetch references for all DOIs concurrently."""
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=20)
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [get_references(session, doi, semaphore) for doi in dois]
        results = await tqdm_asyncio.gather(*tasks, desc=desc)
        return dict(results)


if __name__ == "__main__":
    # Load data
    neuroscience_df = pd.read_csv(NEUROSCIENCE_PAPERS)
    neuroscience_dois = list(set(neuroscience_df["doi"].dropna().tolist()))

    with open(NEUROMORPHIC_PAPERS, "r") as f:
        neuromorphic_papers = json.load(f)
    neuromorphic_dois = list(set(paper["doi"] for paper in neuromorphic_papers if paper.get("doi")))

    print(f"Neuroscience DOIs: {len(neuroscience_dois)}")
    print(f"Neuromorphic DOIs: {len(neuromorphic_dois)}")

    # Fetch references for both groups
    print("\nFetching references...")
    neuro_refs = asyncio.run(fetch_all_references(neuroscience_dois, "Neuroscience refs"))
    nm_refs = asyncio.run(fetch_all_references(neuromorphic_dois, "Neuromorphic refs"))

    # Build sets of all cited DOIs per group
    neuro_cited = set()
    for refs in neuro_refs.values():
        neuro_cited.update(refs)

    nm_cited = set()
    for refs in nm_refs.values():
        nm_cited.update(refs)

    # Find common citations
    common_dois = neuro_cited & nm_cited

    print(f"\nNeuroscience papers cite {len(neuro_cited)} unique DOIs")
    print(f"Neuromorphic papers cite {len(nm_cited)} unique DOIs")
    print(f"Common DOIs cited by both: {len(common_dois)}")

    # Build detailed info: which papers from each group cite each common DOI
    common_citations = []
    for common_doi in common_dois:
        neuro_citers = [doi for doi, refs in neuro_refs.items() if common_doi in refs]
        nm_citers = [doi for doi, refs in nm_refs.items() if common_doi in refs]
        common_citations.append({
            "cited_doi": common_doi,
            "neuroscience_citing_count": len(neuro_citers),
            "neuromorphic_citing_count": len(nm_citers),
            "neuroscience_citing_dois": ";".join(neuro_citers),
            "neuromorphic_citing_dois": ";".join(nm_citers),
        })

    # Sort by total citations
    common_citations.sort(key=lambda x: x["neuroscience_citing_count"] + x["neuromorphic_citing_count"], reverse=True)

    # Save to CSV
    common_df = pd.DataFrame(common_citations)
    common_df.to_csv("common_cited_dois.csv", index=False)
    print(f"\nSaved {len(common_citations)} common cited DOIs to 'common_cited_dois.csv'")
