import json
import pandas as pd
import requests
from tqdm import tqdm
import time
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Need to get json array with each entry having:
# [
#     'pdf_filename',
#     'title',
#     'sections',
#     'figures_folder',
#     'figure_count',
#     'doi',
#     'crossref_metadata'
# ]

def fetch_crossref_metadata(doi: str) -> Optional[Dict[str, Any]]:
    """
    Fetch metadata from CrossRef API for a given DOI.

    Args:
        doi: The DOI string (e.g., "10.1145/3320288.3320290")

    Returns:
        Dictionary with url, authors, publisher, and publication dates, or None if failed
    """
    if not doi:
        return None

    # CrossRef API endpoint
    url = f"https://api.crossref.org/works/{doi}"

    try:
        # Add a polite user agent as recommended by CrossRef
        headers = {
            'User-Agent': 'NeuromorphicPapersEnhancer/1.0 (mailto:research@example.com)'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        message = data.get('message', {})

        # Extract relevant fields
        metadata = {
            'url': message.get('URL'),
            'authors': [],
            'publisher': message.get('publisher'),
            'publication_dates': {}
        }

        # Extract authors
        if 'author' in message:
            for author in message['author']:
                author_info = {}
                if 'given' in author:
                    author_info['given'] = author['given']
                if 'family' in author:
                    author_info['family'] = author['family']
                if 'name' in author:
                    author_info['name'] = author['name']
                if 'ORCID' in author:
                    author_info['ORCID'] = author['ORCID']

                # Create full name if given and family are available
                if 'given' in author_info and 'family' in author_info:
                    author_info['full_name'] = f"{author_info['given']} {author_info['family']}"

                metadata['authors'].append(author_info)

        # Extract publication dates
        date_fields = [
            'issued',
            'published-print',
            'published-online',
            'created',
            'deposited',
            'indexed',
            'accepted',
            'posted'
        ]

        for field in date_fields:
            if field in message:
                date_data = message[field]
                if isinstance(date_data, dict) and 'date-parts' in date_data:
                    # date-parts is typically [[year, month, day]]
                    date_parts = date_data['date-parts'][0] if date_data['date-parts'] else []
                    if date_parts:
                        metadata['publication_dates'][field] = {
                            'date-parts': date_parts,
                            'formatted': format_date_parts(date_parts)
                        }

        return metadata

    except requests.exceptions.RequestException:
        return None
    except (KeyError, json.JSONDecodeError):
        return None


def format_date_parts(date_parts: list) -> str:
    """
    Format date parts array into a readable string.

    Args:
        date_parts: List like [year, month, day] (month and day optional)

    Returns:
        Formatted date string
    """
    if len(date_parts) == 3:
        return f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
    elif len(date_parts) == 2:
        return f"{date_parts[0]}-{date_parts[1]:02d}"
    elif len(date_parts) == 1:
        return str(date_parts[0])
    return ""

PARENT_DIR = sys.argv[1] if len(sys.argv) > 1 else '.'
FIELD = sys.argv[2] if len(sys.argv) > 2 else 'neuroscience'
CSV_OVERRIDE = sys.argv[3] if len(sys.argv) > 3 else None

def find_papers_csv(parent_dir: str, field: str) -> str:
    """Auto-detect the papers CSV by trying known naming patterns."""
    candidates = [
        Path(parent_dir) / f"final_{field}_papers_with_ids.csv",
        Path(parent_dir) / f"expanded_{field}_nodes.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"No papers CSV found. Tried: {tried}")

if PARENT_DIR == '.':
    PDFS_FOLDER = "../../data/neuroscience/papers/"
    PAPERS_CSV = "../../data/neuroscience/final_neuroscience_papers_with_ids.csv"
    OUTPUT_JSON_PATH = "../../data/neuroscience/neuroscience_json_template.json"
else:
    PDFS_FOLDER = str(Path(PARENT_DIR) / "papers")
    PAPERS_CSV = CSV_OVERRIDE if CSV_OVERRIDE else find_papers_csv(PARENT_DIR, FIELD)
    OUTPUT_JSON_PATH = str(Path(PARENT_DIR) / f"{FIELD}_json_template.json")

papers_df = pd.read_csv(PAPERS_CSV)
json_template = []

for _, row in tqdm(papers_df.iterrows(), total=papers_df.shape[0]):
    paper_entry = {
        "pdf_filename": f"{row['id']}.pdf",
        "title": row['title'],
        "sections": [],
        "figures_folder": f"{row['id']}_figures",
        "figure_count": 0,
        "doi": row['doi'],
        "crossref_metadata": {}
    }

    if pd.notna(row['abstract']):
        abstract_section = {
            "section_title": "Abstract",
            "section_text": row['abstract']
        }
        paper_entry['sections'].append(abstract_section)

    # Fetch CrossRef metadata
    if pd.notna(row['doi']):
        crossref_metadata = fetch_crossref_metadata(row['doi'])
        paper_entry['crossref_metadata'] = crossref_metadata if crossref_metadata else {}
        time.sleep(0.5)  # Be polite to CrossRef API

    json_template.append(paper_entry)

with open(OUTPUT_JSON_PATH, 'w') as json_file:
    json.dump(json_template, json_file, indent=4)
