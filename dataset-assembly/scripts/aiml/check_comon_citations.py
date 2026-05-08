import pandas as pd
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio

NEUROSCIENCE_PAPERS_JSON = "../../data/neuroscience/papers.json"
NEUROMORPHIC_PAPERS_JSON = "../../data/neuromorphic/neuromorphic_papers_cleaned_enhanced.json"
AI_ML_DOIS = "../../data/aiml/combined_ml_papers_dois.txt"

OPENCITATIONS_API = "https://opencitations.net/index/coci/api/v1"
CONCURRENCY_LIMIT = 10

async def get_references(session, semaphore, doi):
    """Fetch references for a DOI from OpenCitations API."""
    url = f"{OPENCITATIONS_API}/references/{doi}"
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return doi, [ref.get('cited') for ref in data if ref.get('cited')]
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
    return doi, []

async def check_citation_events(source_dois, target_set):
    """Check if papers in source_dois cite any papers in target_set."""
    citation_events = []
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async with aiohttp.ClientSession() as session:
        tasks = [get_references(session, semaphore, doi) for doi in source_dois]
        results = await tqdm_asyncio.gather(*tasks, desc="Fetching references")

        for citing_doi, references in results:
            for cited_doi in references:
                if cited_doi in target_set:
                    citation_events.append((citing_doi, cited_doi))

    return citation_events

async def check_common_citations(aiml_set, neuro_set):
    """Check for citation events between AI/ML and neuroscience paper sets (both directions)."""
    print("Checking AI/ML papers citing neuroscience papers...")
    aiml_cites_neuro = await check_citation_events(aiml_set, neuro_set)
    print(f"Found {len(aiml_cites_neuro)} citation events (AI/ML -> Neuro)")

    print("\nChecking neuroscience papers citing AI/ML papers...")
    neuro_cites_aiml = await check_citation_events(neuro_set, aiml_set)
    print(f"Found {len(neuro_cites_aiml)} citation events (Neuro -> AI/ML)")

    all_events = aiml_cites_neuro + neuro_cites_aiml
    print(f"\nTotal citation events: {len(all_events)}")
    return all_events



with open(NEUROSCIENCE_PAPERS_JSON, 'r') as f:
    neuroscience_papers = pd.read_json(f)

print(f"Total neuroscience papers loaded: {len(neuroscience_papers)}")

with open(NEUROMORPHIC_PAPERS_JSON, 'r') as f:
    neuromorphic_papers = pd.read_json(f)

print(f"Total neuromorphic papers loaded: {len(neuromorphic_papers)}")

neuroscience_dois = set(neuroscience_papers['doi'].dropna().unique().tolist())
neuromorphic_dois = set(neuromorphic_papers['doi'].dropna().unique().tolist())

print(f"Total unique DOIs in neuromorphic papers: {len(neuromorphic_dois)}")
print(f"Total unique DOIs in neuroscience papers: {len(neuroscience_dois)}")

unioned_dois = neuroscience_dois.union(neuromorphic_dois)

print(f"Total unique DOIs in unioned papers: {len(unioned_dois)}")

with open(AI_ML_DOIS, 'r') as f:
    aiml_dois = set(line.strip() for line in f if line.strip())

print(f"Total unique DOIs in AI/ML papers: {len(aiml_dois)}")

# Check for citation events between the two sets
citation_events = asyncio.run(check_common_citations(aiml_dois, unioned_dois))

print(f"\nCitation events (from, to):")
for citing, cited in citation_events[:20]:  # Show first 20
    print(f"  {citing} -> {cited}")
if len(citation_events) > 20:
    print(f"  ... and {len(citation_events) - 20} more")

# Save citation events to a CSV file
citation_events_df = pd.DataFrame(citation_events, columns=['citing_doi', 'cited_doi'])
citation_events_df.to_csv("../../data/aiml/neuro_aiml_citation_events.csv", index=False)
print("\nCitation events saved to '../../data/aiml/neuro_aiml_citation_events.csv'")
