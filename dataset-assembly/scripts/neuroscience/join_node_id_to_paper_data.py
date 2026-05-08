import pandas as pd

nodelist = pd.read_csv("../../data/neuroscience/expanded_neuroscience_nodes.csv")
paperdata = pd.read_csv("../../data/neuroscience/expanded_neuroscience_papers.csv")

# Add id collumn to paperdata by joining on doi
merged_df = pd.merge(paperdata, nodelist, on="doi", how="left")

# Move id column to the front
cols = merged_df.columns.tolist()
cols = [cols[-1]] + cols[:-1]
merged_df = merged_df[cols]

merged_df.to_csv("../../data/neuroscience/expanded_neuroscience_papers_with_ids.csv", index=False)