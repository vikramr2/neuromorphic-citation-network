#!/usr/bin/env python3
"""
Append docling-parsed results for PDFs not yet in neuroscience_papers_marker.json.

Usage:
    python append_marker_papers.py
"""

import csv
import json
import multiprocessing
import requests
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR    = Path("../../../data/neuroscience")
PAPERS_DIR  = DATA_DIR / "papers"
FIGURES_DIR = DATA_DIR / "figures"
MARKER_JSON = DATA_DIR / "neuroscience_papers_marker.json"
NODES_CSV   = DATA_DIR / "neuroscience_nodes_updated.csv"

START_INDEX     = 750   # process only PDFs with stem >= this value
SAVE_EVERY      = 10
PDF_TIMEOUT_SEC = 600   # 10 min per PDF before giving up

# ---------------------------------------------------------------------------
# Helpers (run in main process)
# ---------------------------------------------------------------------------

def fetch_crossref_metadata(doi: str) -> dict:
    try:
        resp = requests.get(f"https://api.crossref.org/works/{doi}", timeout=15)
        if resp.status_code != 200:
            return {}
        msg = resp.json().get("message", {})
        authors = "; ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in msg.get("author", [])
        )
        pub_date = msg.get("published-print", msg.get("published-online", {}))
        date_parts = pub_date.get("date-parts", [[]])
        date = "-".join(str(p) for p in date_parts[0]) if date_parts and date_parts[0] else ""
        return {
            "url": msg.get("URL", f"https://doi.org/{doi}"),
            "authors": authors,
            "publisher": msg.get("publisher", ""),
            "publication_dates": date,
        }
    except Exception as e:
        print(f"  [crossref warn] {doi}: {e}")
        return {}


def load_nodes_by_index(csv_path: Path) -> dict[int, dict]:
    result = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                result[int(row["id"])] = row
            except (ValueError, KeyError):
                pass
    return result


# ---------------------------------------------------------------------------
# Worker (runs in a fresh subprocess per PDF to isolate native-lib crashes)
# ---------------------------------------------------------------------------

def _subprocess_worker(pdf_path_str: str, csv_title: str, figures_dir_str: str, result_path: str):
    """
    Runs inside a child process. Imports docling here so heap state from a
    previous crash cannot carry over.  Writes JSON result to result_path.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pdf_path    = Path(pdf_path_str)
    figures_dir = Path(figures_dir_str)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 3.0

    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
    )

    result_doc = converter.convert(str(pdf_path))
    document   = result_doc.document

    # ---- title ----
    title = csv_title
    if not title:
        try:
            for line in document.export_to_markdown().split("\n"):
                line = line.strip()
                if line.startswith("##"):
                    title = line.lstrip("#").strip()
                    break
        except Exception:
            pass
    if not title and hasattr(document, "name") and document.name:
        title = document.name
    if not title:
        try:
            for item, _ in document.iterate_items():
                if hasattr(item, "label") and item.label and "section_header" in str(item.label).lower():
                    title = item.text if hasattr(item, "text") else str(item)
                    break
        except Exception:
            pass
    title = title or f"[Untitled - {pdf_path.name}]"

    # ---- sections ----
    sections = []
    try:
        markdown_text = document.export_to_markdown()
        lines = markdown_text.split("\n")
        current_section = None
        current_body: list[str] = []

        for line in lines:
            if line.startswith("##"):
                if current_section is not None:
                    current_section["body"] = "\n".join(current_body).strip()
                    sections.append(current_section)
                    current_body = []
                level = len(line) - len(line.lstrip("#"))
                text  = line.lstrip("#").strip()
                if text:
                    current_section = {"level": level, "heading": text, "body": ""}
            else:
                if current_section is not None:
                    current_body.append(line)

        if current_section is not None:
            current_section["body"] = "\n".join(current_body).strip()
            sections.append(current_section)
    except Exception:
        try:
            current_section = None
            for item, level in document.iterate_items():
                if hasattr(item, "label") and item.label:
                    label_str = str(item.label).lower()
                    text = item.text if hasattr(item, "text") else str(item)
                    if "section_header" in label_str:
                        if current_section is not None:
                            sections.append(current_section)
                        current_section = {"level": level, "heading": text, "body": ""}
                    elif current_section is not None and "text" in label_str:
                        current_section["body"] = (
                            current_section["body"] + "\n" + text
                            if current_section["body"] else text
                        )
            if current_section is not None:
                sections.append(current_section)
        except Exception:
            pass

    # ---- figures ----
    figures_subdir = figures_dir / f"{pdf_path.stem}_figures"
    figures_subdir.mkdir(exist_ok=True)
    figure_count = 0
    for i, picture in enumerate(document.pictures):
        if picture.image and picture.image.pil_image:
            picture.image.pil_image.save(str(figures_subdir / f"figure_{i + 1}.png"))
            figure_count += 1

    output = {
        "title":          title,
        "sections":       sections,
        "figures_folder": str(figures_subdir.relative_to(DATA_DIR)),
        "figure_count":   figure_count,
    }
    with open(result_path, "w") as f:
        json.dump(output, f, ensure_ascii=False)


def process_pdf_isolated(pdf_path: Path, csv_title: str) -> dict:
    """
    Spawn a fresh process for each PDF so that any native-lib heap corruption
    (e.g. 'corrupted double-linked list') kills only the child, not this loop.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        result_path = tmp.name

    p = multiprocessing.Process(
        target=_subprocess_worker,
        args=(str(pdf_path), csv_title, str(FIGURES_DIR), result_path),
    )
    p.start()
    p.join(timeout=PDF_TIMEOUT_SEC)

    if p.is_alive():
        p.kill()
        p.join()
        Path(result_path).unlink(missing_ok=True)
        raise RuntimeError(f"timed out after {PDF_TIMEOUT_SEC}s")

    if p.exitcode != 0:
        Path(result_path).unlink(missing_ok=True)
        raise RuntimeError(f"subprocess exited with code {p.exitcode}")

    rp = Path(result_path)
    if not rp.exists():
        raise RuntimeError("subprocess produced no output file")

    with open(rp) as f:
        data = json.load(f)
    rp.unlink(missing_ok=True)
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(MARKER_JSON) as f:
        marker_data: list[dict] = json.load(f)

    already_done = {e["pdf_filename"] for e in marker_data if e.get("marker_processed")}
    print(f"Successfully processed in marker JSON: {len(already_done)}")

    all_pdfs   = sorted(PAPERS_DIR.glob("*.pdf"), key=lambda p: int(p.stem))
    to_process = [
        p for p in all_pdfs
        if int(p.stem) >= START_INDEX and p.name not in already_done
    ]
    print(f"Total PDFs: {len(all_pdfs)}, to process from index {START_INDEX}: {len(to_process)}")

    if not to_process:
        print("Nothing to do.")
        return

    nodes_by_id = load_nodes_by_index(NODES_CSV)
    print(f"Loaded metadata for {len(nodes_by_id)} nodes\n")

    errors    = []
    processed = 0

    for pdf_path in to_process:
        idx      = int(pdf_path.stem)
        node     = nodes_by_id.get(idx, {})
        doi      = node.get("doi", "")
        csv_title = node.get("title", "")

        crossref_meta = {}
        if doi:
            crossref_meta = fetch_crossref_metadata(doi)
            time.sleep(0.5)

        entry = {
            "pdf_filename":     pdf_path.name,
            "title":            csv_title,
            "sections":         [],
            "figures_folder":   f"{idx}_figures",
            "figure_count":     0,
            "doi":              doi,
            "crossref_metadata": crossref_meta,
            "marker_processed": False,
        }

        try:
            parsed = process_pdf_isolated(pdf_path, csv_title)
            entry.update(parsed)
            entry["marker_processed"] = True
        except Exception as e:
            print(f"Error on {pdf_path.name}: {e}")
            entry["error"] = str(e)
            errors.append((pdf_path.name, str(e)))

        marker_data.append(entry)
        processed += 1

        if processed % SAVE_EVERY == 0:
            temp_file = MARKER_JSON.with_suffix(".json.tmp")
            with open(temp_file, "w") as f:
                json.dump(marker_data, f, indent=2, ensure_ascii=False)
            temp_file.replace(MARKER_JSON)
            print(f"  Saved at {processed}/{len(to_process)}")

    if MARKER_JSON.exists():
        backup_file = MARKER_JSON.with_suffix(".json.backup")
        MARKER_JSON.replace(backup_file)
        print(f"Created backup: {backup_file}")

    temp_file = MARKER_JSON.with_suffix(".json.tmp")
    with open(temp_file, "w") as f:
        json.dump(marker_data, f, indent=2, ensure_ascii=False)
    temp_file.replace(MARKER_JSON)

    print(f"\nDone! Processed {processed} new papers.")
    print(f"Total entries in marker JSON: {len(marker_data)}")
    if errors:
        print(f"Errors: {len(errors)}")
        for name, err in errors:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
