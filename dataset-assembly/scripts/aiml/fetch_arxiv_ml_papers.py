import pandas as pd

df = pd.read_csv("hf://datasets/CShorten/ML-ArXiv-Papers/ML-Arxiv-Papers.csv")

df_trimmed = df[["title", "abstract"]]

df_trimmed.to_csv("arxiv_ml_papers_titles_abstracts.csv", index=False)
