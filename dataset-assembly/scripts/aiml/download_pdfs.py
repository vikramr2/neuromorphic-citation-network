import pandas as pd

print("mkdir -p ../../data/aiml/papers/")

NODELIST_FILE = "../../data/aiml/expanded_aiml_nodes.csv"

nodelist_df = pd.read_csv(NODELIST_FILE)
id_pdf_pairs = [(row['id'], row['pdf_url']) for _, row in nodelist_df.iterrows() if pd.notna(row['pdf_url'])]

for node_id, pdf_url in id_pdf_pairs:
    safe_pdf_url = pdf_url.replace("/", "_").replace(":", "_")
    output_path = f"../../data/aiml/papers/{node_id}.pdf"
    print(f"wget -O {output_path} {pdf_url}")
