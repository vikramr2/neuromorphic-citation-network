#!/usr/bin/env python3
"""
Deduplicate neuroscience_papers_marker.json.

For each pdf_filename that appears more than once:
  - If any copy has marker_processed=True  → keep that copy, drop the rest.
  - If all copies have marker_processed=False → keep the last occurrence.

Writes the result back in-place (backs up the original first).
"""

import json
from pathlib import Path

MARKER_JSON = Path("../../../data/neuroscience/neuroscience_papers_marker.json")


def main():
    with open(MARKER_JSON) as f:
        data: list[dict] = json.load(f)

    print(f"Entries before cleanup: {len(data)}")

    # Track the winning index for each filename (last-seen wins by default;
    # a successful entry overwrites any previous winner).
    winners: dict[str, int] = {}

    for i, entry in enumerate(data):
        fname = entry.get("pdf_filename")
        if fname is None:
            continue
        prev = winners.get(fname)
        if prev is None:
            winners[fname] = i
        else:
            # Prefer a successful entry; otherwise take the later one.
            prev_success = data[prev].get("marker_processed", False)
            curr_success = entry.get("marker_processed", False)
            if curr_success and not prev_success:
                winners[fname] = i
            elif not curr_success and not prev_success:
                winners[fname] = i  # keep later failed entry

    keep_indices = set(winners.values())

    # Entries with no pdf_filename are kept as-is.
    no_fname = [e for e in data if "pdf_filename" not in e]

    cleaned = no_fname + [data[i] for i in sorted(keep_indices)]

    removed = len(data) - len(cleaned)
    success = sum(1 for e in cleaned if e.get("marker_processed"))
    failed  = sum(1 for e in cleaned if not e.get("marker_processed"))

    print(f"Removed {removed} duplicate entries")
    print(f"Entries after cleanup: {len(cleaned)}  (success={success}, failed={failed})")

    backup = MARKER_JSON.with_suffix(".json.pre_cleanup")
    MARKER_JSON.replace(backup)
    print(f"Backup written to {backup.name}")

    tmp = MARKER_JSON.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    tmp.replace(MARKER_JSON)
    print(f"Cleaned JSON written to {MARKER_JSON.name}")


if __name__ == "__main__":
    main()
