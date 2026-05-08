import json
import pandas as pd
from tqdm import tqdm
from doi_fetch import get_doi_from_title

with open('../data/papers.json', 'r') as f:
    papers = json.load(f)

pdfs = []
titles = []
dois = []

missing_pdfs = []
missing_titles = []

for paper in tqdm(papers):
    filename = paper['pdf_filename']
    title = paper['title']
    doi = get_doi_from_title(title)

    if doi is not None:
        pdfs.append(filename)
        titles.append(title)
        dois.append(doi)
    else:
        missing_pdfs.append(filename)
        missing_titles.append(title)

print(f"Found {len(titles)} DOIs, missing {len(missing_titles)} DOIs")
df = pd.DataFrame({'pdf_filename': pdfs, 'title': titles, 'doi': dois})
df.to_csv('../data/papers_with_dois.csv', index=False)
print("Saved papers with DOIs to ../data/papers_with_dois.csv")

if missing_titles:
    # Leave an empty column for new titles so I can manually fill them in later
    new_titles = [None] * len(missing_titles)

    df_missing = pd.DataFrame({'pdf_filename': missing_pdfs, 'old_title': missing_titles, 'new_title': new_titles})
    df_missing.to_csv('../data/missing_papers.csv', index=False)
    print("Saved missing titles to ../data/missing_papers.csv for manual review")
