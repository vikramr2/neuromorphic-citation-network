#!/usr/bin/env python3
"""
Backfill neuroscience_papers.json with entries from the expanded CSV
that are missing from the current JSON.

For each missing entry:
  - Creates a JSON template entry (pdf_filename, title, doi, etc.)
  - Adds abstract as a section if available
  - Fetches CrossRef metadata

Also updates neuroscience_nodes.csv to include any rows from the
expanded CSV that were previously filtered out.

Usage:
    python backfill_json.py
"""

import csv
import json
import time
import sys
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "neuroscience"
EXPANDED_CSV = DATA_DIR / "archive" / "expanded_neuroscience_papers_with_ids.csv"
NODES_CSV = DATA_DIR / "neuroscience_nodes.csv"
PAPERS_JSON = DATA_DIR / "neuroscience_papers.json"


def fetch_crossref_metadata(doi: str) -> Optional[Dict[str, Any]]:
    if not doi:
        return None
    url = f"https://api.crossref.org/works/{doi}"
    try:
        headers = {
            'User-Agent': 'NeuromorphicPapersEnhancer/1.0 (mailto:research@example.com)'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        message = data.get('message', {})
        metadata = {
            'url': message.get('URL'),
            'authors': [],
            'publisher': message.get('publisher'),
            'publication_dates': {}
        }
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
                if 'given' in author_info and 'family' in author_info:
                    author_info['full_name'] = f"{author_info['given']} {author_info['family']}"
                metadata['authors'].append(author_info)
        date_fields = ['issued', 'published-print', 'published-online',
                       'created', 'deposited', 'indexed', 'accepted', 'posted']
        for field in date_fields:
            if field in message:
                date_data = message[field]
                if isinstance(date_data, dict) and 'date-parts' in date_data:
                    date_parts = date_data['date-parts'][0] if date_data['date-parts'] else []
                    if date_parts:
                        if len(date_parts) == 3:
                            fmt = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
                        elif len(date_parts) == 2:
                            fmt = f"{date_parts[0]}-{date_parts[1]:02d}"
                        else:
                            fmt = str(date_parts[0])
                        metadata['publication_dates'][field] = {
                            'date-parts': date_parts,
                            'formatted': fmt
                        }
        return metadata
    except Exception:
        return None


def main():
    # 1) Load expanded CSV (all entries)
    with open(EXPANDED_CSV, newline="") as f:
        expanded = {r['id']: r for r in csv.DictReader(f)}
    print(f"Expanded CSV: {len(expanded)} entries")

    # 2) Load current nodes CSV
    with open(NODES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        nodes = {r['id']: r for r in reader if r.get('id', '').strip()}
    print(f"Current nodes CSV: {len(nodes)} entries")

    # Also include nodes-only entries (e.g. 757, 758) in the full set
    all_entries = dict(expanded)
    for nid, row in nodes.items():
        if nid not in all_entries:
            all_entries[nid] = row

    # 3) Load current JSON
    with open(PAPERS_JSON) as f:
        papers = json.load(f)
    json_ids = {e.get('pdf_filename', '').replace('.pdf', '')
                for e in papers if 'pdf_filename' in e}
    print(f"Current JSON: {len(papers)} entries")

    # 4) Find missing entries
    papers_dir = DATA_DIR / "papers"
    missing = []
    for eid in sorted(all_entries.keys(), key=lambda x: int(x)):
        if eid in json_ids:
            continue
        # Only add if PDF exists on disk
        if not (papers_dir / f"{eid}.pdf").exists():
            continue
        missing.append(all_entries[eid])

    print(f"Missing from JSON (with PDF on disk): {len(missing)}")
    if not missing:
        print("Nothing to backfill.")
        return

    # 5) Create JSON entries for missing papers
    new_entries = []
    for row in tqdm(missing, desc="Backfilling"):
        entry = {
            "pdf_filename": f"{row['id']}.pdf",
            "title": row.get('title', ''),
            "sections": [],
            "figures_folder": f"{row['id']}_figures",
            "figure_count": 0,
            "doi": row.get('doi', ''),
            "crossref_metadata": {}
        }
        abstract = row.get('abstract', '')
        if abstract and abstract.lower() != 'nan':
            entry['sections'].append({
                "section_title": "Abstract",
                "section_text": abstract
            })
        doi = row.get('doi', '')
        if doi and doi.lower() != 'nan':
            crossref = fetch_crossref_metadata(doi)
            entry['crossref_metadata'] = crossref if crossref else {}
            time.sleep(0.5)
        new_entries.append(entry)

    # 6) Append to JSON and save
    papers.extend(new_entries)
    with open(PAPERS_JSON, 'w') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"\nUpdated JSON: {len(papers)} entries (+{len(new_entries)} new)")

    # 7) Update nodes CSV with any missing rows from expanded
    nodes_missing = [eid for eid in all_entries if eid not in nodes]
    if nodes_missing:
        with open(NODES_CSV, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for eid in sorted(nodes_missing, key=lambda x: int(x)):
                row = all_entries[eid]
                writer.writerow({k: row.get(k, '') for k in fieldnames})
        print(f"Updated nodes CSV: added {len(nodes_missing)} rows")
    else:
        print("Nodes CSV already complete.")


if __name__ == "__main__":
    main()