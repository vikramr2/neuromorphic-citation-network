#!/usr/bin/env python3
"""
Retry failed marker JSON entries using LLM-based OCR (fallback for PDFs
that docling could not parse).

Usage:
    python retry_failed_with_ocr.py
"""

import base64
import json
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR    = Path("../../../data/neuroscience")
PAPERS_DIR  = DATA_DIR / "papers"
MARKER_JSON = DATA_DIR / "neuroscience_papers_marker.json"

LLM_MODEL       = "openai/gpt-oss-120b"   # model name as served by vLLM
OLLAMA_BASE_URL = "http://earlsinclair.ornl.gov:8200/v1"
OLLAMA_API_KEY  = "vllm"

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
)

SAVE_EVERY = 10

# ---------------------------------------------------------------------------
# PDF → images
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: Path) -> list[Image.Image]:
    doc = pdfium.PdfDocument(str(pdf_path))
    return [doc[i].render(scale=2).to_pil() for i in range(len(doc))]


def image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# LLM OCR
# ---------------------------------------------------------------------------

def ocr_page(page_image: Image.Image) -> str:
    b64 = image_to_base64(page_image)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Convert this document page to markdown. Preserve headings, lists, tables, and equations. Output only the markdown, no commentary.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Markdown → sections
# ---------------------------------------------------------------------------

def markdown_to_sections(markdown_text: str) -> list[dict]:
    sections = []
    current_section = None
    current_body: list[str] = []

    for line in markdown_text.split("\n"):
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

    return sections


def extract_title(markdown_text: str) -> str | None:
    for line in markdown_text.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(MARKER_JSON) as f:
        marker_data: list[dict] = json.load(f)

    # Build index: pdf_filename → position in list
    idx_map: dict[str, int] = {
        e["pdf_filename"]: i
        for i, e in enumerate(marker_data)
        if "pdf_filename" in e
    }

    failed = [
        e for e in marker_data
        if (not e.get("marker_processed") or not e.get("sections"))
        and "pdf_filename" in e
        and (PAPERS_DIR / e["pdf_filename"]).exists()
    ]
    failed.sort(key=lambda e: int(Path(e["pdf_filename"]).stem))

    print(f"Failed entries to retry: {len(failed)}")
    if not failed:
        print("Nothing to do.")
        return

    errors    = []
    processed = 0

    outer_bar = tqdm(failed, desc="PDFs", unit="pdf")
    for entry in outer_bar:
        pdf_path = PAPERS_DIR / entry["pdf_filename"]
        outer_bar.set_postfix(file=entry["pdf_filename"])

        try:
            pages = pdf_to_images(pdf_path)

            page_markdowns = []
            for page_image in tqdm(pages, desc=f"  {pdf_path.stem} pages", unit="page", leave=False):
                page_markdowns.append(ocr_page(page_image))

            full_markdown = "\n\n".join(page_markdowns)
            sections      = markdown_to_sections(full_markdown)
            ocr_title     = extract_title(full_markdown)

            # Update the entry in-place (keep existing crossref_metadata, doi, etc.)
            pos = idx_map[entry["pdf_filename"]]
            marker_data[pos]["sections"]         = sections
            marker_data[pos]["figure_count"]     = 0
            marker_data[pos]["marker_processed"] = True
            marker_data[pos].pop("error", None)
            if ocr_title and not marker_data[pos].get("title"):
                marker_data[pos]["title"] = ocr_title

        except Exception as e:
            tqdm.write(f"Error on {entry['pdf_filename']}: {e}")
            errors.append((entry["pdf_filename"], str(e)))

        processed += 1
        if processed % SAVE_EVERY == 0:
            tmp = MARKER_JSON.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(marker_data, f, indent=2, ensure_ascii=False)
            tmp.replace(MARKER_JSON)
            tqdm.write(f"  Saved at {processed}/{len(failed)}")

    # Final save
    tmp = MARKER_JSON.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(marker_data, f, indent=2, ensure_ascii=False)
    tmp.replace(MARKER_JSON)

    print(f"\nDone. Retried {processed} PDFs, {processed - len(errors)} succeeded, {len(errors)} failed.")
    if errors:
        for name, err in errors:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
