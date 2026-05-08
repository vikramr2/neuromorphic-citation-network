# Move the out of network PDFs to a separate folder for easier access and management.

import pandas as pd

NEUROSCIENCE_PAPERS_CSV = "../../data/neuroscience/final_neuroscience_papers_with_ids.csv"
NEUROSCIENCE_PAPERS_PDF_FOLDER = "../../data/neuroscience/papers/"
OON_PDFS_FOLDER = "../../data/neuroscience/oon_pdfs/"

# Get the ids of papers in the network
neuroscience_papers_df = pd.read_csv(NEUROSCIENCE_PAPERS_CSV)
network_ids = set(neuroscience_papers_df["id"].tolist())

print(f"Number of neuroscience papers in the network: {len(network_ids)}")

# Get the ids of the pdfs in the pdf folder {id}.pdf
import os
pdf_filenames = os.listdir(NEUROSCIENCE_PAPERS_PDF_FOLDER)
pdf_ids = set()
for filename in pdf_filenames:
    if filename.endswith(".pdf"):
        pdf_id = filename[:-4]  # Remove .pdf extension
        pdf_ids.add(pdf_id)

print(f"Number of PDFs in the pdf folder: {len(pdf_ids)}")

# Identify out of network pdfs
# First convert network_ids to string for comparison
network_ids_str = set(str(id) for id in network_ids)
oon_pdf_ids = pdf_ids - network_ids_str
print(f"Number of out of network PDFs: {len(oon_pdf_ids)}")

# Move the out of network pdfs to the oon_pdfs folder
import shutil

if not os.path.exists(OON_PDFS_FOLDER):
    os.makedirs(OON_PDFS_FOLDER)

for oon_pdf_id in oon_pdf_ids:
    src_path = os.path.join(NEUROSCIENCE_PAPERS_PDF_FOLDER, f"{oon_pdf_id}.pdf")
    dst_path = os.path.join(OON_PDFS_FOLDER, f"{oon_pdf_id}.pdf")
    shutil.move(src_path, dst_path)
    
print("Moved out of network PDFs to the oon_pdfs folder.")
