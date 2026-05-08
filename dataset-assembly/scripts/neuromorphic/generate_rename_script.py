#!/usr/bin/env python3
"""
Generates a shell script that copies and renames PDFs and figure directories
using node IDs from the neuromorphic_nodes.csv.

Mapping chain:
  neuromorphic_nodes.csv  (id -> doi)
  neuromorphic_papers.json (doi -> pdf_filename, figures_folder)

Output:
  papers_renamed/{id}.pdf
  figures_renamed/{id}_figures/
"""

import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "neuromorphic"
NODES_CSV = DATA_DIR / "neuromorphic_nodes.csv"
PAPERS_JSON = DATA_DIR / "neuromorphic_papers.json"
OUTPUT_SCRIPT = DATA_DIR / "rename_files.sh"


def main():
    # 1) Load id -> doi from CSV
    id_to_doi = {}
    with open(NODES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_to_doi[row["doi"]] = row["id"]

    print(f"Loaded {len(id_to_doi)} nodes from CSV")

    # 2) Load doi -> (pdf_filename, figures_folder) from JSON
    with open(PAPERS_JSON) as f:
        papers = json.load(f)

    doi_to_files = {}
    for entry in papers:
        doi = entry.get("doi", "")
        pdf = entry.get("pdf_filename", "")
        figs = entry.get("figures_folder", "")
        if doi and pdf:
            doi_to_files[doi] = (pdf, figs)

    print(f"Loaded {len(doi_to_files)} papers with DOIs from JSON")

    # 3) Build matched rename lines + track unmatched PDFs
    matched = 0
    no_json = 0
    missing_nodes = []
    rename_lines = []
    matched_pdfs = set()

    for doi, node_id in sorted(id_to_doi.items(), key=lambda x: int(x[1])):
        if doi not in doi_to_files:
            no_json += 1
            missing_nodes.append((node_id, doi))
            continue

        pdf_filename, figures_folder = doi_to_files[doi]
        matched += 1
        matched_pdfs.add(pdf_filename)

        # Copy PDF
        rename_lines.append(f'cp -- "papers/{pdf_filename}" "papers_renamed/{node_id}.pdf"')

        # Copy figures directory if it exists
        if figures_folder:
            rename_lines.append(f'[ -d "figures/{figures_folder}" ] && cp -r -- "figures/{figures_folder}" "figures_renamed/{node_id}_figures"')

    # 4) Find PDFs that didn't match any node
    all_pdfs = {entry.get("pdf_filename", "") for entry in papers if entry.get("pdf_filename")}
    unmatched_pdfs = sorted(all_pdfs - matched_pdfs)
    unmatched_lines = []
    for pdf in unmatched_pdfs:
        unmatched_lines.append(f'cp -- "papers/{pdf}" "unmatched_pdfs/{pdf}"')

    print(f"Matched: {matched}")
    print(f"Nodes with no JSON entry: {no_json}")
    print(f"Unmatched PDFs (no node): {len(unmatched_pdfs)}")

    # 5) Print missing nodes table
    if missing_nodes:
        print(f"\n{'='*80}")
        print(f"MISSING NODES — {len(missing_nodes)} nodes have no PDF in JSON")
        print(f"{'='*80}")
        print(f"{'Node ID':<10} {'DOI':<70}")
        print(f"{'-'*10} {'-'*70}")
        for node_id, doi in missing_nodes:
            print(f"{node_id:<10} {doi:<70}")

    # 6) Print unmatched PDFs table
    if unmatched_pdfs:
        print(f"\n{'='*80}")
        print(f"UNMATCHED PDFs — {len(unmatched_pdfs)} PDFs have no node")
        print(f"{'='*80}")
        for pdf in unmatched_pdfs:
            print(f"  {pdf}")

    # 7) Write shell script
    with open(OUTPUT_SCRIPT, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        f.write(f"cd \"{DATA_DIR}\"\n\n")
        f.write("mkdir -p papers_renamed figures_renamed unmatched_pdfs\n\n")

        f.write(f"# --- Matched: copy and rename {matched} PDFs ---\n")
        for line in rename_lines:
            f.write(line + "\n")

        f.write(f"\n# --- Unmatched: copy {len(unmatched_pdfs)} PDFs with no node ---\n")
        for line in unmatched_lines:
            f.write(line + "\n")

        f.write(f'\necho "Done. Copied {matched} renamed + {len(unmatched_pdfs)} unmatched PDFs."\n')

    OUTPUT_SCRIPT.chmod(0o755)
    print(f"\nGenerated: {OUTPUT_SCRIPT}")
    print(f"Run it with: bash {OUTPUT_SCRIPT}")


if __name__ == "__main__":
    main()
