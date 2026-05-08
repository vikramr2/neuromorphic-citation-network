import json
import pandas as pd
from typing import List
from analyze_clusters import (
    DATA_DIR, 
    DOI_MAPPING, 
    get_largest_community, 
    get_doi_from_ids
)

CLUSTERING = DATA_DIR + "neuromorphic_communities_0p0001.tsv"
DOCLING_PAPERS = DATA_DIR + "papers.json"
MONGO_PAPERS = DATA_DIR + "mongo_data/mongo_data_with_dois.json"
TITLE_DOI_MAPPING = DATA_DIR + "combined_neuromorphic_title_dois.csv"

if __name__ == "__main__":
    doi_mapping_df = pd.read_csv(DOI_MAPPING)
    largest_community_nodes = get_largest_community(CLUSTERING)
    largest_community_dois = get_doi_from_ids(largest_community_nodes, doi_mapping_df)
    
    print(f"Largest community has {len(largest_community_nodes)} nodes")
    # print("DOIs in the largest community:")
    # for doi in largest_community_dois:
    #     print(doi)

    # Load docling papers
    with open(DOCLING_PAPERS, "r") as f:
        docling_data = json.load(f)

    print("Docling Entry Keys:", docling_data[0].keys())

    # Load mongo papers
    with open(MONGO_PAPERS, "r") as f:
        mongo_data = json.load(f)

    print("Mongo Entry Keys:", mongo_data[0].keys())
    
    # Load title-DOI mapping
    title_doi_df = pd.read_csv(TITLE_DOI_MAPPING)

    # Filter out entries where pdf_filename is mongoDB
    title_doi_df = title_doi_df[title_doi_df['pdf_filename'] != 'mongoDB']

    # Create pdf_filename to DOI mapping
    pdf_to_doi = dict(zip(title_doi_df['pdf_filename'], title_doi_df['doi']))

    # Apply this to the docling data
    for entry in docling_data:
        pdf_filename = entry.get('pdf_filename', None)
        if pdf_filename and pdf_filename in pdf_to_doi:
            entry['doi'] = pdf_to_doi[pdf_filename]
        else:
            entry['doi'] = None
    
    print("DOI of first docling entry after mapping:", docling_data[0].get('doi', None))

    docling_dois = set(
        entry['doi'] for entry in docling_data if entry.get('doi', None) is not None
    )

    # Filter out mongo data whos DOIs are in docling DOIs
    filtered_mongo_data = [
        entry for entry in mongo_data
        if entry.get('doi', None) not in docling_dois
    ]

    # Combine both datasets
    combined_data = docling_data + filtered_mongo_data

    print(f"Combined dataset has {len(combined_data)} entries.")

    # Filter out entries whose DOIs aren't in the largest community
    final_filtered_data = [
        entry for entry in combined_data
        if entry.get('doi', None) in largest_community_dois
    ]

    print(f"Filtered dataset has {len(final_filtered_data)} entries.")

    # Remove duplicate DOIs
    seen_dois = set()
    unique_filtered_data = []
    for entry in final_filtered_data:
        doi = entry.get('doi', None)
        if doi and doi not in seen_dois:
            seen_dois.add(doi)
            unique_filtered_data.append(entry)
    
    print(f"After removing duplicate DOIs: {len(unique_filtered_data)} entries.")

    # Save the final filtered dataset
    output_path = DATA_DIR + "neuromorphic_papers_cleaned.json"
    with open(output_path, "w") as f:
        json.dump(unique_filtered_data, f, indent=2)
    print(f"Saved cleaned dataset to {output_path}")
    