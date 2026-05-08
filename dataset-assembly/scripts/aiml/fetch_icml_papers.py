import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
splits = {'icml_2021': 'data/icml_2021-00000-of-00001.parquet', 'icml_2022': 'data/icml_2022-00000-of-00001.parquet', 'icml_2023': 'data/icml_2023-00000-of-00001.parquet', 'icml_2024': 'data/icml_2024-00000-of-00001.parquet'}
# df = pd.read_parquet("hf://datasets/AIM-Harvard/ICML-Accepted-Papers/" + splits["icml_2021"])

necessary_columns = [
    "title",
    "authors",
    "abstract",
    "Download PDF",
    "tags"
]

all_dfs = []
for split_name, split_path in splits.items():
    df = pd.read_parquet("hf://datasets/AIM-Harvard/ICML-Accepted-Papers/" + split_path)
    df_trimmed = df[necessary_columns]
    all_dfs.append(df_trimmed)

final_df = pd.concat(all_dfs, ignore_index=True)
final_df.to_csv("icml_2021_2024_papers_titles_abstracts.csv", index=False)
print("Saved ICML 2021-2024 papers' titles and abstracts to 'icml_2021_2024_papers_titles_abstracts.csv'")
