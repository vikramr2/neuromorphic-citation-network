#!/usr/bin/env python3
"""
Re-OCR papers using Marker (marker-pdf).

Takes existing JSON files (aiml_papers.json, neuroscience_papers.json,
neuromorphic_papers.json), copies all fields, and replaces sections with
markdown text extracted by Marker.

Usage:
    python ocr_fetch.py <parent_dir> <field>

Examples:
    python ocr_fetch.py ../../data/aiml aiml
    python ocr_fetch.py ../../data/neuroscience neuroscience
    python ocr_fetch.py ../../data/neuromorphic neuromorphic
"""

import json
import sys
import copy
from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARENT_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
FIELD = sys.argv[2] if len(sys.argv) > 2 else "neuroscience"

if PARENT_DIR == ".":
    papers_dir = Path(f"../../data/{FIELD}/papers/")
    input_json = Path(f"../../data/{FIELD}/{FIELD}_papers.json")
    output_json = Path(f"../../data/{FIELD}/{FIELD}_papers_marker.json")
else:
    papers_dir = Path(PARENT_DIR) / "papers"
    input_json = Path(PARENT_DIR) / f"{FIELD}_papers.json"
    output_json = Path(PARENT_DIR) / f"{FIELD}_papers_marker.json"

SAVE_EVERY = 10


# ---------------------------------------------------------------------------
# Parse markdown into sections
# ---------------------------------------------------------------------------

def parse_sections(markdown_text: str) -> list[dict]:
    """
    Parse markdown text into sections list.

    Returns list of dicts with keys: level, heading, body
    Body text is kept in markdown format.
    """
    sections = []
    current_section = None
    current_body = []

    for line in markdown_text.split("\n"):
        if line.startswith("##"):
            if current_section is not None:
                current_section["body"] = "\n".join(current_body).strip()
                sections.append(current_section)
                current_body = []

            level = len(line) - len(line.lstrip("#"))
            heading = line.lstrip("#").strip()
            if heading:
                current_section = {"level": level, "heading": heading, "body": ""}
        else:
            if current_section is not None:
                current_body.append(line)

    if current_section is not None:
        current_section["body"] = "\n".join(current_body).strip()
        sections.append(current_section)

    if not sections and markdown_text.strip():
        sections.append({
            "level": 1,
            "heading": "Full Text",
            "body": markdown_text.strip(),
        })

    return sections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {input_json}")
    with open(input_json, "r") as f:
        papers = json.load(f)
    print(f"Loaded {len(papers)} papers")

    # Resume support: start from input papers, merge in any existing results
    output_data = copy.deepcopy(papers)
    output_by_filename = {e["pdf_filename"]: e for e in output_data if "pdf_filename" in e}
    done = set()

    if output_json.exists():
        with open(output_json, "r") as f:
            existing = json.load(f)
        for e in existing:
            fn = e.get("pdf_filename")
            if fn and e.get("marker_processed") and fn in output_by_filename:
                output_by_filename[fn].update(e)
                done.add(fn)
        print(f"Resuming — {len(done)} already processed")

    # Build work list
    to_process = []
    for entry in papers:
        if "pdf_filename" not in entry:
            continue
        fname = entry["pdf_filename"]
        if fname in done:
            continue
        pdf_path = papers_dir / fname
        if not pdf_path.exists():
            continue
        to_process.append((fname, pdf_path))

    print(f"Papers to process: {len(to_process)}")
    if not to_process:
        print("Nothing to do.")
        return

    # Load Marker models once
    print("Loading Marker models...")
    converter = PdfConverter(artifact_dict=create_model_dict())
    print("Models loaded.")

    processed = 0
    for fname, pdf_path in tqdm(to_process, desc="OCR"):
        entry = output_by_filename[fname]
        try:
            rendered = converter(str(pdf_path))
            markdown_text, _, _ = text_from_rendered(rendered)
            entry["sections"] = parse_sections(markdown_text)
            entry["marker_processed"] = True
        except Exception as e:
            tqdm.write(f"Error on {fname}: {e}")
            entry["error"] = str(e)
            entry["marker_processed"] = False

        processed += 1
        if processed % SAVE_EVERY == 0:
            with open(output_json, "w") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Final save
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    total_sections = sum(len(e.get("sections", [])) for e in output_data)
    errors = sum(1 for e in output_data if "error" in e)
    print(f"\nDone! Processed {processed} papers this run.")
    print(f"Total sections: {total_sections}")
    print(f"Errors: {errors}")
    print(f"Output: {output_json}")


if __name__ == "__main__":
    main()
