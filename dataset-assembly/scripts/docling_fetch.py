#!/usr/bin/env python3
"""
Extract markdown from PDFs using an OpenAI-compatible LLM.

Sends each PDF page as a base64 image to the LLM, which converts it to
markdown. Saves each paper as {id}.md in a markdowns/ directory.

Usage:
    python docling_fetch.py <parent_dir> <field>

Examples:
    python docling_fetch.py ../data/aiml aiml
    python docling_fetch.py ../data/neuroscience neuroscience
    python docling_fetch.py ../data/neuromorphic neuromorphic
"""

import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLM_MODEL = "gpt-oss:20b"
client = OpenAI(
    base_url="http://carz1.ornl.gov:11434/v1",
    api_key="ollama",
)

PARENT_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
FIELD = sys.argv[2] if len(sys.argv) > 2 else "neuroscience"

if PARENT_DIR == ".":
    papers_dir = Path(f"../../data/{FIELD}/papers/")
    markdowns_dir = Path(f"../../data/{FIELD}/markdowns/")
else:
    papers_dir = Path(PARENT_DIR) / "papers"
    markdowns_dir = Path(PARENT_DIR) / "markdowns"

markdowns_dir.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# PDF -> page images
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: Path) -> list[Image.Image]:
    """Convert each page of a PDF to a PIL Image."""
    doc = pdfium.PdfDocument(str(pdf_path))
    images = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        bitmap = page.render(scale=2)
        images.append(bitmap.to_pil())
    return images


def image_to_base64(img: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# LLM OCR for a single page
# ---------------------------------------------------------------------------

def ocr_page(page_image: Image.Image) -> str:
    """Send a page image to the LLM and get markdown back."""
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
# Main
# ---------------------------------------------------------------------------

def main():
    pdf_files = sorted(papers_dir.glob("*.pdf"), key=lambda p: int(p.stem))
    print(f"Found {len(pdf_files)} PDFs in {papers_dir}")

    # Resume: skip papers that already have a .md file
    already_done = {p.stem for p in markdowns_dir.glob("*.md")}
    to_process = [p for p in pdf_files if p.stem not in already_done]
    print(f"Already done: {len(already_done)}, to process: {len(to_process)}")

    if not to_process:
        print("Nothing to do.")
        return

    errors = []
    for pdf_path in tqdm(to_process, desc="OCR"):
        paper_id = pdf_path.stem
        try:
            pages = pdf_to_images(pdf_path)
            page_markdowns = []
            for page_image in pages:
                md = ocr_page(page_image)
                page_markdowns.append(md)

            combined = "\n\n".join(page_markdowns)
            out_path = markdowns_dir / f"{paper_id}.md"
            out_path.write_text(combined, encoding="utf-8")
        except Exception as e:
            tqdm.write(f"Error on {pdf_path.name}: {e}")
            errors.append((paper_id, str(e)))

    print(f"\nDone! Processed {len(to_process)} papers.")
    print(f"Errors: {len(errors)}")
    if errors:
        for pid, err in errors:
            print(f"  {pid}: {err}")


if __name__ == "__main__":
    main()