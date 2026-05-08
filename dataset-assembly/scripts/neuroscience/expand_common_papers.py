import json
import asyncio
import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

COMMON_CITATIONS_PATH = "../../data/neuroscience/cleaned_common_citations.json"
NEUROSCIENCE_PAPERS_PATH = "../../data/neuroscience/biorxiv_papers/neuroscience_papers_complete.csv"

with open(COMMON_CITATIONS_PATH, "r") as f:
    common_citations = json.load(f)

seed_neuro_dois = set()
common_citation_dois = {entry["doi"] for entry in common_citations}

print(f"Common citation DOIs count: {len(common_citation_dois)}")

for entry in common_citations:
    seed_neuro_dois = seed_neuro_dois.union(set(entry["neuroscience_citing_dois"]))

print(f"Seed neuroscience DOIs count: {len(seed_neuro_dois)}")

neuroscience_df = pd.read_csv(NEUROSCIENCE_PAPERS_PATH)

# Find the rows corresponding to the seed DOIs
expanded_neuro_df = neuroscience_df[neuroscience_df["doi"].isin(seed_neuro_dois)]

# Expand within the biorxiv dataset by doing one-hop citations from the seed papers
async def fetch_citations(session, doi, semaphore):
    """Fetch papers that cite this DOI from Crossref API."""
    async with semaphore:
        url = f"https://api.crossref.org/works/{doi}"
        headers = {"User-Agent": "DatasetAssembly/1.0 (mailto:research@example.com)"}
        try:
            async with session.get(url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    references = data.get("message", {}).get("reference", [])
                    cited_dois = []
                    for ref in references:
                        if "DOI" in ref:
                            cited_dois.append(ref["DOI"].lower())
                    return doi, cited_dois
                return doi, []
        except Exception as e:
            print(f"Error fetching {doi}: {e}")
            return doi, []


async def expand_one_hop(seed_dois, biorxiv_dois):
    """Get all papers cited by seed papers that are in the biorxiv dataset."""
    semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_citations(session, doi, semaphore) for doi in seed_dois]
        results = await tqdm_asyncio.gather(*tasks, desc="Fetching citations from Crossref")

    # Collect all cited DOIs that are in the biorxiv dataset
    one_hop_dois = set()
    for doi, cited_dois in results:
        for cited_doi in cited_dois:
            if cited_doi in biorxiv_dois:
                one_hop_dois.add(cited_doi)

    return one_hop_dois


# Get all biorxiv DOIs for filtering
biorxiv_dois = set(neuroscience_df["doi"].str.lower().tolist())

# Run the async expansion
one_hop_dois = asyncio.run(expand_one_hop(seed_neuro_dois, biorxiv_dois))
print(f"Found {len(one_hop_dois)} one-hop papers in BioRxiv dataset")

# Add one-hop papers to the expanded dataframe
one_hop_df = neuroscience_df[neuroscience_df["doi"].str.lower().isin(one_hop_dois)]
expanded_neuro_df = pd.concat([expanded_neuro_df, one_hop_df]).drop_duplicates(subset=["doi"])

print(f"Expanded neuroscience DOIs count after one hop: {len(expanded_neuro_df)}")

# Run another hop (two-hop expansion)
two_hop_dois = asyncio.run(expand_one_hop(one_hop_dois, biorxiv_dois))
print(f"Found {len(two_hop_dois)} two-hop papers in BioRxiv dataset")

# Add two-hop papers to the expanded dataframe
two_hop_df = neuroscience_df[neuroscience_df["doi"].str.lower().isin(two_hop_dois)]
expanded_neuro_df = pd.concat([expanded_neuro_df, two_hop_df]).drop_duplicates(subset=["doi"])

print(f"Expanded neuroscience DOIs count after two hops: {len(expanded_neuro_df)}")

# Append the common citation DOIs but with None for other fields
# I'll enter the values manually later
for doi in common_citation_dois:
    if doi not in expanded_neuro_df["doi"].values:
        expanded_neuro_df = expanded_neuro_df._append({
            "doi": doi,
            "title": None,
            "abstract": None,
            "authors": None,
            "pdf_url": None,
            "published_date": None
        }, ignore_index=True)

# Save the expanded dataframe
expanded_neuro_df.to_csv("../../data/neuroscience/expanded_neuroscience_papers.csv", index=False)
print("Saved expanded neuroscience papers to 'expanded_neuroscience_papers.csv'")

# Save the citation network as two csvs: a id doi node list and an edges list
node_list = []
doi_to_id = {}
for idx, doi in enumerate(expanded_neuro_df["doi"].tolist()):
    node_list.append({"id": idx, "doi": doi})
    doi_to_id[doi] = idx

node_df = pd.DataFrame(node_list)
node_df.to_csv("../../data/neuroscience/expanded_neuroscience_nodes.csv", index=False)
print("Saved node list to 'expanded_neuroscience_nodes.csv'")

# Build edges
edge_list = []
for entry in common_citations:
    cited_doi = entry["doi"]
    for citing_doi in entry["neuroscience_citing_dois"]:
        if citing_doi in doi_to_id and cited_doi in doi_to_id:
            edge_list.append({
                "source": doi_to_id[citing_doi],
                "target": doi_to_id[cited_doi]
            })
edge_df = pd.DataFrame(edge_list)
edge_df.to_csv("../../data/neuroscience/expanded_neuroscience_edges.csv", index=False)
print("Saved edge list to 'expanded_neuroscience_edges.csv'")
