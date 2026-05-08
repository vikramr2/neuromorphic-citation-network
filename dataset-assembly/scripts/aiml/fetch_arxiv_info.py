import pandas as pd
import urllib.parse
import xml.etree.ElementTree as ET
import asyncio
import aiohttp
from datetime import datetime
from tqdm.asyncio import tqdm_asyncio

# Concurrency limit to respect arXiv API rate limits
ARXIV_CONCURRENT = 5

def parse_arxiv_response(data):
    """Parse arXiv API XML response"""
    root = ET.fromstring(data)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}

    entry = root.find('atom:entry', ns)
    if entry is not None:
        arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
        paper_title = entry.find('atom:title', ns).text.strip()
        arxiv_id_base = arxiv_id.split('v')[0]
        arxiv_doi = f"10.48550/arXiv.{arxiv_id_base}"
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        published = entry.find('atom:published', ns).text
        pub_date = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ')
        year = pub_date.year
        abstract = entry.find('atom:summary', ns).text.strip()

        return {
            'arxiv_id': arxiv_id,
            'title': paper_title,
            'doi': arxiv_doi,
            'pdf_url': pdf_url,
            'year': year,
            'published_date': published,
            'abstract': abstract
        }
    return None

async def fetch_arxiv_info(session, title, semaphore):
    """Async fetch from arXiv API"""
    async with semaphore:
        query = urllib.parse.quote(f'ti:"{title}"')
        url = f'http://export.arxiv.org/api/query?search_query={query}&max_results=1'
        try:
            async with session.get(url) as response:
                data = await response.text()
                return parse_arxiv_response(data)
        except Exception as e:
            print(f"Error fetching arXiv for '{title[:50]}...': {e}")
            return None

async def fetch_paper_info(session, title, arxiv_sem):
    """Fetch paper info from arXiv"""
    arxiv_info = await fetch_arxiv_info(session, title, arxiv_sem)
    if arxiv_info:
        return arxiv_info
    return {
        'arxiv_id': None, 'title': None, 'doi': None, 'pdf_url': None,
        'year': None, 'published_date': None, 'abstract': None
    }

async def fetch_all_papers(titles):
    """Fetch info for all papers concurrently"""
    arxiv_sem = asyncio.Semaphore(ARXIV_CONCURRENT)

    connector = aiohttp.TCPConnector(limit=20)
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [fetch_paper_info(session, title, arxiv_sem) for title in titles]
        results = await tqdm_asyncio.gather(*tasks, desc="Fetching papers")
        return results

if __name__ == "__main__":
    # Example usage
    # title = "Attention Is All You Need"
    # result = asyncio.run(fetch_all_papers([title]))[0]
    # if result['arxiv_id']:
    #     print(f"Title: {result['title']}")
    #     print(f"arXiv ID: {result['arxiv_id']}")
    #     print(f"DOI: {result['doi']}")
    #     print(f"PDF URL: {result['pdf_url']}")
    #     print(f"Year: {result['year']}")
    #     print(f"Published: {result['published_date']}")
    #     print(f"\nAbstract:\n{result['abstract'][:200]}...")
    # else:
    #     print("Paper not found")

    ARXIV_PATH = "../../data/aiml/arxiv_ml_papers_titles_abstracts.csv"
    df = pd.read_csv(ARXIV_PATH)

    # Fetch all papers concurrently
    results = asyncio.run(fetch_all_papers(df['title'].tolist()))

    detailed_df = pd.DataFrame(results)
    detailed_df.to_csv("arxiv_ml_papers_detailed_info.csv", index=False)
    print("Saved detailed arXiv ML papers info to 'arxiv_ml_papers_detailed_info.csv'")
