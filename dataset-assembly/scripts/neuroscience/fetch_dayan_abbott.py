"""
Fetch references and citations for Dayan & Abbott "Theoretical Neuroscience"
via the Semantic Scholar API, writing two CSVs:
  - dayan_abbott_references.csv  (works cited by the book)
  - dayan_abbott_citations.csv   (works that cite the book)
"""

import csv
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE = "https://api.semanticscholar.org/graph/v1/paper"
ISBN = "0-262-04199-5"
FIELDS = "externalIds,title,authors,year,venue"
LIMIT = 1000  # max per request


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


SESSION = make_session()


def get(url: str, params: dict) -> dict:
    r = SESSION.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def get_paper_id(isbn: str) -> str:
    # Try ISBN first
    url = f"{API_BASE}/ISBN:{isbn}"
    r = SESSION.get(url, params={"fields": "paperId,title,authors"}, timeout=30)
    if r.status_code == 200:
        data = r.json()
        print(f"Found paper via ISBN: {data.get('title')}")
        print(f"Authors: {', '.join(a['name'] for a in data.get('authors', []))}")
        return data["paperId"]

    # Fall back to title search
    print(f"ISBN lookup failed ({r.status_code}), falling back to title search...")
    time.sleep(2)  # brief pause before hitting another endpoint
    search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": "Theoretical Neuroscience Computational Mathematical Modeling Neural Systems Dayan Abbott",
        "fields": "paperId,title,authors,year",
        "limit": 5,
    }
    results = get(search_url, params).get("data", [])
    if not results:
        raise RuntimeError("No results found via title search.")
    # Pick the first result and show options so user can verify
    for i, p in enumerate(results):
        authors = ", ".join(a["name"] for a in p.get("authors", []))
        print(f"  [{i}] {p.get('year')} — {p.get('title')} | {authors}")
    chosen = results[0]
    print(f"\nUsing: {chosen['title']} (paperId={chosen['paperId']})")
    return chosen["paperId"]


def fetch_all(paper_id: str, edge: str) -> list[dict]:
    """Paginate through references or citations for a given paper."""
    results = []
    offset = 0
    while True:
        url = f"{API_BASE}/{paper_id}/{edge}"
        params = {"fields": FIELDS, "limit": LIMIT, "offset": offset}
        data = get(url, params)
        batch = data.get("data", [])
        results.extend(batch)
        print(f"  {edge}: fetched {len(results)} so far (offset={offset})")
        if len(batch) < LIMIT:
            break
        offset += LIMIT
        time.sleep(0.5)
    return results


def extract_doi(paper: dict) -> str:
    return (paper.get("externalIds") or {}).get("DOI", "")


def write_csv(path: str, rows: list[dict], edge: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["doi", "title", "year", "venue", "authors", "s2_paper_id"]
        )
        writer.writeheader()
        for item in rows:
            # references wrap the paper under "citedPaper"; citations under "citingPaper"
            paper = item.get("citedPaper" if edge == "references" else "citingPaper", item)
            writer.writerow(
                {
                    "doi": extract_doi(paper),
                    "title": paper.get("title", ""),
                    "year": paper.get("year", ""),
                    "venue": paper.get("venue", ""),
                    "authors": "; ".join(a["name"] for a in paper.get("authors", [])),
                    "s2_paper_id": paper.get("paperId", ""),
                }
            )
    print(f"Wrote {len(rows)} rows to {path}")


def main() -> None:
    print("Looking up paper...")
    paper_id = get_paper_id(ISBN)

    print("\nFetching references (works cited by the book)...")
    references = fetch_all(paper_id, "references")

    print("\nFetching citations (works that cite the book)...")
    citations = fetch_all(paper_id, "citations")

    write_csv("dayan_abbott_references.csv", references, "references")
    write_csv("dayan_abbott_citations.csv", citations, "citations")

    refs_with_doi = sum(1 for r in references if extract_doi(r.get("citedPaper", r)))
    cits_with_doi = sum(1 for c in citations if extract_doi(c.get("citingPaper", c)))
    print(f"\nDone. References: {len(references)} ({refs_with_doi} with DOI)")
    print(f"      Citations:  {len(citations)} ({cits_with_doi} with DOI)")


if __name__ == "__main__":
    main()
