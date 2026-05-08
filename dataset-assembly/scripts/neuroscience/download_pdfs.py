import pandas as pd

NEUROSCIENCE_PAPERS_CSV = "../../data/neuroscience/expanded_neuroscience_papers_with_ids.csv"

if __name__ == "__main__":
    # Load neuroscience papers data
    neuroscience_df = pd.read_csv(NEUROSCIENCE_PAPERS_CSV)

    print(f"# Total neuroscience papers loaded: {len(neuroscience_df)}")

    # Extract DOIs, ensuring uniqueness and dropping NaN values
    neuroscience_dois = list(set(neuroscience_df["doi"].dropna().tolist()))

    print(f"# Total unique neuroscience DOIs: {len(neuroscience_dois)}")

    # Get all pdf_url entries that are not NaN
    pdf_urls = neuroscience_df["pdf_url"].dropna().tolist()
    ids = neuroscience_df["id"].dropna().tolist()

    print(f"# Total neuroscience papers with PDF URLs: {len(pdf_urls)}")

    blacklist_urls = ['unable', 'dropbox']

    # Print wget commands for downloading PDFs
    for id, url in zip(ids, pdf_urls):
        if any(blacklist in url.lower() for blacklist in blacklist_urls):
            continue
        print(f"wget -O '../data/neuroscience/papers/{id}.pdf' '{url}'")