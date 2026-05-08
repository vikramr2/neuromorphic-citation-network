# Dataset Assembly

## Neuromorphic Paper Dataset Assembly

This folder contains all scripts run to assemble our set of papers in Neuromorphic computing. The process is as follows:

1. Fetch data from our GROBID MongoDB pipeline into a JSON file
2. Batch `wget` to download all our Neuromorphic publication pdfs from our DropBox
3. Use Docling to scrape all pdfs' figures and text
4. Manual correction of malformed and missing titles from the Docling scrape
5. Fetch DOIs across both datasets using the CrossRef API
6. Assemble a citation network using the OpenCitations API
7. Cluster with Leiden with resolution 0.0001
8. Get the largest cluster
9. Union data across the Docling and MongoDB scrapes, and filter to the papers within the largest cluster.
10. Enhance the data with authors, dates, and publisher info using the CrossRef API
  
## Scripts in this folder

In alphabetical order...

- `analyze_clusters.py`: Comparison script to see what clustering properly removes noise (not too much, not too little). This is judged fairly subjectively.
- `assemble_network.py`: Connect dois via citationn across both MongoDB and Docling fetch.
- `docling_fetch.py`: Scrape data and figures off of the Dropbox pdfs
- `doi_fetch.py`: Gets dois from our MongoDB set
- `doi_from_docling_fetch.py`: Fetches the DOI from the Docling scrape
- `doi_from_fixed_titles.py`: Re-fetches the DOI from the manually corrected titles.
- `doi_noise_removal.py`: Filters to only dois within the largest cluster of Leiden 0.0001. Treating all other DOIs as 'noise'.
- `enhance_with_crossref.py`: Enhances the cleaned papers JSON with CrossRef API metadata (URL, authors, publisher, publication dates)
- `malformed_titles.py`: Manual Regex entries of and filtration to what consitutes a 'malformed' title.
- `run_docling_safe.sh`: A safe wrapper around `docling_fetch.py` so that I can keep a backup in case the output data gets wiped
- `union_entries.py`: Unions the dois and titles and pdf filenames across both the Docling scrape and MongoDB for easier processing. Docling scrapes prioritized in the union. Filename of the Mongo entries just labeled as 'MongoDB'
  
## Notes
  
**Jan 7, 2026**, I think to start from the MongoDB database and then work towards extracting the paper data might be a bit difficult. I'm currently doing the following:

1. `wget` from Dropbox and get 2,532 pdfs  
2. Docling extract figures and data from the pdfs
3. Extract DOIs and examine citation and community structure
4. GWM traversal and enhabnce KG

Essentially work from the PDFs first, then get the data rather than the other way around.

**Jan 14, 2026**, I clustered the appended data across the GROBID MongoDB data and the scraped Docling data. I got 2555 total articles.

I used the following methods to cluster the citation network.

- Leiden res 0.01
- Leiden res 0.001
- Leiden res 0.0001
- Leiden modularity (mod) (Effectively reduces to Louvain)

I got the following largest cluster sizes:

- **0.01**: 188
- **0.001**: 590
- **0.0001**: 1745
- **mod**: 407

We want to perform a removal of irrelevant data, but we also want to consider the least amount of data as 'irrelevant'. As such I'm going to set res 0.0001 as the noise filter.
