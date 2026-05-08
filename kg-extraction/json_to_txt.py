#!/usr/bin/env python3
"""
json_to_txt.py  —  Convert *_papers_marker.json files to per-paper .txt files.

Usage:
    python json_to_txt.py <json_file_or_dir> <output_dir>

    If a directory is given, all *_papers_marker.json files inside are processed.

Each paper becomes one .txt file named by its PDF stem (e.g. 0.txt).
Text = title + all section bodies concatenated.
"""

import json
import sys
from pathlib import Path


def paper_to_text(paper: dict) -> str:
    parts = []
    if paper.get("title"):
        parts.append(paper["title"])
    for section in paper.get("sections", []):
        heading = section.get("heading", "")
        body    = section.get("body", "")
        if heading:
            parts.append(heading)
        if body:
            parts.append(body)
    return "\n\n".join(parts)


def convert(json_path: Path, out_dir: Path) -> int:
    papers = json.loads(json_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for paper in papers:
        stem = Path(paper.get("pdf_filename", f"paper_{count}")).stem
        text = paper_to_text(paper)
        (out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        count += 1
    print(f"  {json_path.name}: {count} papers → {out_dir}")
    return count


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    src  = Path(sys.argv[1])
    dest = Path(sys.argv[2])

    if src.is_file():
        convert(src, dest)
    elif src.is_dir():
        jsons = sorted(src.rglob("*_papers_marker.json"))
        if not jsons:
            sys.exit(f"No *_papers_marker.json found under {src}")
        for j in jsons:
            convert(j, dest)
    else:
        sys.exit(f"ERROR: {src} is not a file or directory")


if __name__ == "__main__":
    main()
