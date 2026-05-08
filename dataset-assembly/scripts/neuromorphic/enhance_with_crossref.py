#!/usr/bin/env python3
"""
Enhance neuromorphic papers JSON with CrossRef metadata.
Fetches URL, authors, publisher, and publication dates for each DOI.
"""

import json
import requests
import time
from typing import Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm


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


def enhance_papers(input_path: str, output_path: str, delay: float = 0.5):
    """
    Enhance papers JSON with CrossRef metadata.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        delay: Delay between API requests in seconds (to be polite to CrossRef)
    """
    print(f"Loading papers from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    print(f"Found {len(papers)} papers")
    print("Fetching CrossRef metadata...\n")

    enhanced_papers = []
    success_count = 0
    fail_count = 0
    no_doi_count = 0

    for paper in tqdm(papers, desc="Processing papers", unit="paper"):
        # Copy original paper data
        enhanced_paper = paper.copy()

        # Fetch CrossRef metadata if DOI exists
        doi = paper.get('doi')
        if doi:
            metadata = fetch_crossref_metadata(doi)
            if metadata:
                # Add CrossRef metadata to paper
                enhanced_paper['crossref_metadata'] = metadata
                success_count += 1
            else:
                enhanced_paper['crossref_metadata'] = None
                fail_count += 1

            # Be polite to CrossRef API - add delay between requests
            time.sleep(delay)
        else:
            enhanced_paper['crossref_metadata'] = None
            no_doi_count += 1

        enhanced_papers.append(enhanced_paper)

    # Save enhanced data
    print(f"\nSaving enhanced data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_papers, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("Summary:")
    print(f"Total papers: {len(papers)}")
    print(f"Successfully enhanced: {success_count}")
    print(f"Failed to fetch: {fail_count}")
    print(f"No DOI available: {no_doi_count}")
    print(f"Output saved to: {output_path}")
    print("="*60)


def main():
    # Define paths
    input_path = "/home/vr9/approach1/data/neuromorphic/neuromorphic_papers_cleaned.json"
    output_path = "/home/vr9/approach1/data/neuromorphic/neuromorphic_papers_cleaned_enhanced.json"

    # Run enhancement
    enhance_papers(input_path, output_path, delay=0.5)


if __name__ == "__main__":
    main()