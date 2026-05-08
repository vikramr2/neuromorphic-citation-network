import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "neuromorphic"
MAIN_JSON = DATA_DIR / "neuromorphic_papers.json"
ARCHIVE_JSON = DATA_DIR / "archive" / "papers.json"


def normalize(title: str) -> str:
    """Lowercase, strip HTML tags, collapse whitespace, remove punctuation."""
    title = re.sub(r"<[^>]+>", "", title)  # strip HTML like <sub>
    title = title.lower()
    title = re.sub(r"[^a-z0-9 ]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def main():
    with open(MAIN_JSON) as f:
        main_data = json.load(f)

    with open(ARCHIVE_JSON) as f:
        archive_data = json.load(f)

    # Entries missing pdf_filename
    orphans = [e for e in main_data if "pdf_filename" not in e]
    print(f"Main JSON: {len(main_data)} total, {len(orphans)} missing pdf_filename")
    print(f"Archive JSON: {len(archive_data)} entries\n")

    # Build title -> pdf_filename lookup from archive
    archive_by_title = {}
    for entry in archive_data:
        # Archive entries may have title in sections heading
        title = entry.get("title", "")
        if not title:
            # Try first section heading as fallback
            sections = entry.get("sections", [])
            if sections:
                title = sections[0].get("heading", "")
        key = normalize(title)
        if key:
            archive_by_title[key] = entry.get("pdf_filename", "")

    # Match orphans against archive
    matched = []
    unmatched = []

    for entry in orphans:
        doi = entry.get("doi", "")
        title = entry.get("title", "")
        key = normalize(title)
        pdf = archive_by_title.get(key, "")

        if pdf:
            matched.append((doi, title[:80], pdf))
        else:
            unmatched.append((doi, title[:80]))

    # Print table
    print(f"{'='*120}")
    print(f"MATCHED: {len(matched)} / {len(orphans)}")
    print(f"{'='*120}")
    print(f"{'DOI':<40} {'Title':<50} {'pdf_filename':<30}")
    print(f"{'-'*40} {'-'*50} {'-'*30}")
    for doi, title, pdf in matched:
        print(f"{doi:<40} {title:<50} {pdf:<30}")

    print(f"\n{'='*120}")
    print(f"UNMATCHED: {len(unmatched)} / {len(orphans)}")
    print(f"{'='*120}")
    print(f"{'DOI':<40} {'Title':<80}")
    print(f"{'-'*40} {'-'*80}")
    for doi, title in unmatched:
        print(f"{doi:<40} {title:<80}")

    # Also check: do the matched PDFs actually exist on disk?
    papers_dir = DATA_DIR / "papers"
    if papers_dir.exists() and matched:
        print(f"\n{'='*120}")
        print("PDF FILE CHECK (do matched filenames exist on disk?)")
        print(f"{'='*120}")
        found = 0
        for doi, title, pdf in matched:
            exists = (papers_dir / pdf).exists()
            found += int(exists)
            if not exists:
                print(f"  MISSING: {pdf}  ({doi})")
        print(f"\n  {found}/{len(matched)} matched PDFs exist on disk")


if __name__ == "__main__":
    main()
