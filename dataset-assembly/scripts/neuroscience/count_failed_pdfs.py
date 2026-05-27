#!/usr/bin/env python3
"""
Report which PDFs in neuroscience_papers_marker.json failed processing.
"""

import json
from pathlib import Path
import pandas as pd

MARKER_JSON = Path("../../../data/neuroscience/neuroscience_papers_marker.json")
PAPERS_DIR  = Path("../../../data/neuroscience/papers")

NODE_IDS_FILE = "/home/vr9/neuromorphic-citation-network/data/neuroscience/neuroscience_nodes_updated.csv"

def main():
    node_ids_df = pd.read_csv(NODE_IDS_FILE)
    get_pdf_url = lambda node_id: node_ids_df.loc[node_ids_df["id"] == node_id, "pdf_url"].values[0]

    with open(MARKER_JSON) as f:
        data: list[dict] = json.load(f)

    total    = len(data)
    success  = [e for e in data if e.get("marker_processed")]
    failed   = [e for e in data if not e.get("marker_processed") and "pdf_filename" in e]
    on_disk  = [e for e in failed if (PAPERS_DIR / e["pdf_filename"]).exists()]

    print(f"Total entries : {total}")
    print(f"Succeeded     : {len(success)}")
    print(f"Failed        : {len(failed)}")
    print(f"  with PDF on disk: {len(on_disk)}")

    if on_disk:
        print("\nFailed PDFs (with PDF on disk):")
        for e in sorted(on_disk, key=lambda e: int(Path(e["pdf_filename"]).stem)):
            err = e.get("error", "no error recorded")
            print(f"  {e['pdf_filename']:12s}  {err}")

    node_ids = [int(e["pdf_filename"][:-4]) for e in on_disk]
    pdf_urls = [get_pdf_url(node_id) for node_id in node_ids]
    found = [pdf_url == 'db' for pdf_url in pdf_urls] # db means the PDF was found in the database, so it should be extractable
    for node_id, pdf_url, is_found in zip(node_ids, pdf_urls, found):
        if not is_found:
            print(f"PDF for node {node_id} not found in database (URL: {pdf_url})")
        else:
            print(f"PDF for node {node_id} found in database (URL: {pdf_url})")

if __name__ == "__main__":
    main()
