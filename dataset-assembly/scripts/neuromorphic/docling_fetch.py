from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
import os
import json
from pathlib import Path
from tqdm import tqdm
import re

# Setup paths
papers_dir = Path("../data/papers/")
figures_base_dir = Path("../data/figures/")
output_json = Path("../data/papers.json")

# Create figures directory if it doesn't exist
figures_base_dir.mkdir(exist_ok=True)

# Configure pipeline to generate picture images at higher resolution
pipeline_options = PdfPipelineOptions()
pipeline_options.generate_picture_images = True
pipeline_options.images_scale = 3.0  # Increase for higher resolution (default is 1.0)

converter = DocumentConverter(
    format_options={
        'pdf': PdfFormatOption(pipeline_options=pipeline_options)
    }
)

def sanitize_folder_name(name):
    """Sanitize paper title for use as folder name"""
    # Remove invalid characters
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    # Truncate if too long (keep it reasonable for filesystems)
    if len(name) > 100:
        name = name[:100]
    # Remove leading/trailing spaces and dots
    name = name.strip('. ')
    return name or "untitled"

def extract_sections(document):
    """Extract sections with their body text"""
    sections = []

    try:
        # Use export_to_markdown to get structured content
        markdown_text = document.export_to_markdown()
        lines = markdown_text.split('\n')

        current_section = None
        current_body = []

        for line in lines:
            if line.startswith('##'):
                # Save previous section if exists
                if current_section is not None:
                    current_section['body'] = '\n'.join(current_body).strip()
                    sections.append(current_section)
                    current_body = []

                # Start new section
                level = 0
                for char in line:
                    if char == '#':
                        level += 1
                    else:
                        break
                text = line.lstrip('#').strip()

                if text:  # Only add non-empty headings
                    current_section = {
                        'level': level,
                        'heading': text,
                        'body': ''
                    }
            else:
                # Accumulate body text
                if current_section is not None:
                    current_body.append(line)

        # Don't forget the last section
        if current_section is not None:
            current_section['body'] = '\n'.join(current_body).strip()
            sections.append(current_section)

    except Exception as e:
        # If markdown export fails, try iterate_items
        try:
            current_section = None

            for item, level in document.iterate_items():
                if hasattr(item, 'label') and item.label:
                    label_str = str(item.label).lower()
                    text = item.text if hasattr(item, 'text') else str(item)

                    if 'section_header' in label_str:
                        # Save previous section
                        if current_section is not None:
                            sections.append(current_section)

                        # Start new section
                        current_section = {
                            'level': level,
                            'heading': text,
                            'body': ''
                        }
                    elif current_section is not None and 'text' in label_str:
                        # Add body text to current section
                        if current_section['body']:
                            current_section['body'] += '\n' + text
                        else:
                            current_section['body'] = text

            # Add last section
            if current_section is not None:
                sections.append(current_section)
        except:
            pass

    return sections

def process_pdf(pdf_path):
    """Process a single PDF and extract title, sections, and figures"""
    try:
        # Convert the PDF
        result = converter.convert(str(pdf_path))
        document = result.document

        # Extract title from the document content
        title = None

        # Method 1: Try markdown export - get first heading
        try:
            markdown_text = document.export_to_markdown()
            lines = markdown_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('##'):
                    # Extract first heading as title
                    title = line.lstrip('#').strip()
                    break
        except:
            pass

        # Method 2: Check document name attribute
        if not title and hasattr(document, 'name') and document.name:
            title = document.name

        # Method 3: Try iterate_items for section_header
        if not title:
            try:
                for item, level in document.iterate_items():
                    if hasattr(item, 'label') and item.label and 'section_header' in str(item.label).lower():
                        title = item.text if hasattr(item, 'text') else str(item)
                        break
            except:
                pass

        # Last resort: Mark as untitled but keep filename reference
        if not title:
            title = f"[Untitled - {pdf_path.name}]"

        # Extract sections
        sections = extract_sections(document)

        # Create folder for figures using PDF filename (not title)
        # This makes it easier to map JSON entries to figure folders
        folder_name = f"{pdf_path.stem}_figures"
        figures_dir = figures_base_dir / folder_name
        figures_dir.mkdir(exist_ok=True)

        # Extract and save figures
        figure_count = 0
        for i, picture in enumerate(document.pictures):
            image = picture.image

            if image:
                pil_image = image.pil_image
                figure_path = figures_dir / f"figure_{i + 1}.png"
                pil_image.save(str(figure_path))
                figure_count += 1

        # Prepare result
        paper_data = {
            'pdf_filename': pdf_path.name,
            'title': title,
            'sections': sections,
            'figures_folder': str(figures_dir.relative_to(Path("../data"))),
            'figure_count': figure_count
        }

        return paper_data

    except Exception as e:
        print(f"\nError processing {pdf_path.name}: {e}")
        return {
            'pdf_filename': pdf_path.name,
            'title': pdf_path.stem,
            'sections': [],
            'figures_folder': None,
            'figure_count': 0,
            'error': str(e)
        }

# Get all PDF files
pdf_files = sorted(papers_dir.glob("*.pdf"))
print(f"Found {len(pdf_files)} PDF files to process")

# Load existing progress if available
if output_json.exists():
    try:
        with open(output_json, 'r') as f:
            papers_data = json.load(f)
        processed_files = {p['pdf_filename'] for p in papers_data}
        print(f"Resuming from {len(papers_data)} already processed papers")
    except:
        papers_data = []
        processed_files = set()
else:
    papers_data = []
    processed_files = set()

# Process all PDFs
for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
    # Skip already processed files
    if pdf_path.name in processed_files:
        continue

    try:
        paper_data = process_pdf(pdf_path)
        papers_data.append(paper_data)

        # Save progress periodically (every 10 papers for safety)
        if len(papers_data) % 10 == 0:
            # Save to temp file first, then rename (safer)
            temp_file = output_json.with_suffix('.json.tmp')
            with open(temp_file, 'w') as f:
                json.dump(papers_data, f, indent=2)
            temp_file.replace(output_json)
    except Exception as e:
        # Log the error and continue
        print(f"\n\nFATAL ERROR processing {pdf_path.name}: {e}")
        error_data = {
            'pdf_filename': pdf_path.name,
            'title': pdf_path.stem,
            'sections': [],
            'figures_folder': None,
            'figure_count': 0,
            'error': f'FATAL: {str(e)}'
        }
        papers_data.append(error_data)

        # Save immediately after error
        temp_file = output_json.with_suffix('.json.tmp')
        with open(temp_file, 'w') as f:
            json.dump(papers_data, f, indent=2)
        temp_file.replace(output_json)

# Save final results
# Create backup if file exists
if output_json.exists():
    backup_file = output_json.with_suffix('.json.backup')
    output_json.replace(backup_file)
    print(f"Created backup: {backup_file}")

# Save to temp file first, then rename (atomic operation)
temp_file = output_json.with_suffix('.json.tmp')
with open(temp_file, 'w') as f:
    json.dump(papers_data, f, indent=2)
temp_file.replace(output_json)

print(f"\nProcessing complete!")
print(f"Processed {len(papers_data)} papers")
print(f"Results saved to {output_json}")

# Print summary statistics
total_figures = sum(p['figure_count'] for p in papers_data)
errors = sum(1 for p in papers_data if 'error' in p)
print(f"Total figures extracted: {total_figures}")
print(f"Errors encountered: {errors}")
