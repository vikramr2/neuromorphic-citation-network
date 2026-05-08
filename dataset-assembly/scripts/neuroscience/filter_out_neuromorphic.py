import pandas as pd
import json

NEUROMORPHIC_PAPERS_JSON = "../../data/neuromorphic/neuromorphic_papers_cleaned_enhanced.json"
NEUROSCIENCE_PAPERS_CSV = "../../data/neuroscience/expanded_neuroscience_papers_with_ids.csv"

with open(NEUROMORPHIC_PAPERS_JSON, "r") as f:
    neuromorphic_papers = json.load(f)

neuromorphic_dois = {entry["doi"].lower() for entry in neuromorphic_papers}

neuroscience_df = pd.read_csv(NEUROSCIENCE_PAPERS_CSV)
print(f"Original neuroscience papers count: {len(neuroscience_df)}")
# Filter out neuromorphic papers
filtered_neuroscience_df = neuroscience_df[~neuroscience_df["doi"].str.lower().isin(neuromorphic_dois)]
print(f"Filtered neuroscience papers count: {len(filtered_neuroscience_df)}")
filtered_neuroscience_df.to_csv(NEUROSCIENCE_PAPERS_CSV, index=False)
