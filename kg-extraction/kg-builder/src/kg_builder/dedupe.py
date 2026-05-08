import pandas as pd
from pathlib import Path
import logging
import json
from .io_utils import normalize_text


def dedupe_neuromorphickgdb_tables(merged_dir: Path, deduped_path: Path, fuzzy_threshold: float, track_duplicates: bool = True):
    """Deduplicate NeuromorphicKGDB tables."""
    # Load merged tables
    entities_path = merged_dir / "merged_entities.txt"
    documents_path = merged_dir / "merged_documents.txt"
    predications_path = merged_dir / "merged_predications.txt"

    entities_df = pd.read_csv(entities_path, sep='|', encoding='utf-8', engine='python') if entities_path.exists() else pd.DataFrame()
    documents_df = pd.read_csv(documents_path, sep='|', encoding='utf-8', engine='python') if documents_path.exists() else pd.DataFrame()
    predications_df = pd.read_csv(predications_path, sep='|', encoding='utf-8', engine='python') if predications_path.exists() else pd.DataFrame()

    # Deduplicate entities by name
    if not entities_df.empty:
        entities_df['name_norm'] = entities_df['entity_name'].apply(normalize_text)
        entities_deduped = entities_df.drop_duplicates(subset=['name_norm'])
        entities_deduped = entities_deduped.drop(columns=['name_norm'])
        logging.info(f"Entity deduplication: {len(entities_df)} -> {len(entities_deduped)}")

        # Update entity name to ID mapping
        entity_name_to_id = dict(zip(entities_deduped['entity_name'], entities_deduped['entity_id']))
    else:
        entities_deduped = entities_df
        entity_name_to_id = {}

    # Deduplicate documents by ID (should already be unique, but just in case)
    if not documents_df.empty:
        documents_deduped = documents_df.drop_duplicates(subset=['document_id'])
        logging.info(f"Document deduplication: {len(documents_df)} -> {len(documents_deduped)}")
    else:
        documents_deduped = documents_df

    # Deduplicate predications
    if not predications_df.empty:
        # Normalize for deduplication
        predications_df['subject_norm'] = predications_df['subject_entity_name'].apply(normalize_text)
        predications_df['predicate_norm'] = predications_df['predicate'].apply(normalize_text)
        predications_df['object_norm'] = predications_df['object_entity_name'].apply(normalize_text)

        # Track duplicates if enabled
        if track_duplicates:
            # Count occurrences of each normalized triple
            duplicate_counts = predications_df.groupby(['subject_norm', 'predicate_norm', 'object_norm']).size().reset_index(name='count')
            duplicates = duplicate_counts[duplicate_counts['count'] > 1].copy()
            
            # Add original entity names back for readability
            duplicates_with_names = duplicates.merge(
                predications_df[['subject_norm', 'predicate_norm', 'object_norm', 'subject_entity_name', 'predicate', 'object_entity_name']].drop_duplicates(),
                on=['subject_norm', 'predicate_norm', 'object_norm'],
                how='left'
            )
            
            # Save duplicates with frequency counts
            duplicates_output = merged_dir / "duplicates.jsonl"
            with open(duplicates_output, 'w', encoding='utf-8') as f:
                for _, row in duplicates_with_names.iterrows():
                    duplicate_entry = {
                        'h': row['subject_entity_name'],
                        'r': row['predicate'],
                        't': row['object_entity_name'],
                        'frequency': int(row['count'])
                    }
                    f.write(json.dumps(duplicate_entry, ensure_ascii=False) + '\n')
            logging.info(f"Saved {len(duplicates_with_names)} duplicate triples with frequencies to {duplicates_output}")

        # Exact deduplication based on normalized subject-predicate-object
        predications_deduped = predications_df.drop_duplicates(
            subset=['subject_norm', 'predicate_norm', 'object_norm']
        )

        # Remove normalization columns
        predications_deduped = predications_deduped.drop(
            columns=['subject_norm', 'predicate_norm', 'object_norm']
        )

        logging.info(f"Predication deduplication: {len(predications_df)} -> {len(predications_deduped)}")

        # Update entity IDs in predications to use deduplicated entity IDs
        if entity_name_to_id:
            predications_deduped['subject_entity_id'] = predications_deduped['subject_entity_name'].map(entity_name_to_id)
            predications_deduped['object_entity_id'] = predications_deduped['object_entity_name'].map(entity_name_to_id)
    else:
        predications_deduped = predications_df

    # Save deduplicated tables
    deduped_dir = deduped_path.parent
    deduped_dir.mkdir(exist_ok=True)

    if not entities_deduped.empty:
        entities_output = deduped_dir / "deduped_entities.txt"
        entities_deduped.to_csv(entities_output, sep='|', index=False, encoding='utf-8')
        logging.info(f"Saved {len(entities_deduped)} deduped entities to {entities_output}")

    if not documents_deduped.empty:
        documents_output = deduped_dir / "deduped_documents.txt"
        documents_deduped.to_csv(documents_output, sep='|', index=False, encoding='utf-8')
        logging.info(f"Saved {len(documents_deduped)} deduped documents to {documents_output}")

    if not predications_deduped.empty:
        predications_output = deduped_dir / "deduped_predications.txt"
        predications_deduped.to_csv(predications_output, sep='|', index=False, encoding='utf-8')
        logging.info(f"Saved {len(predications_deduped)} deduped predications to {predications_output}")

        # Also save as JSONL for compatibility with downstream processes
        jsonl_output = deduped_dir / "deduped.jsonl"
        with open(jsonl_output, 'w', encoding='utf-8') as f:
            for _, row in predications_deduped.iterrows():
                triple = {
                    'h': row['subject_entity_name'],
                    'r': row['predicate'],
                    't': row['object_entity_name'],
                    'document_id': row['document_id'],
                    'model': row['model_name'],
                    'predication_id': row['predication_id']
                }
                # Include _prompt if it exists
                if '_prompt' in row:
                    triple['_prompt'] = row['_prompt']
                # Also include prompt_filename if it exists
                if 'prompt_filename' in row and row['prompt_filename']:
                    triple['_prompt'] = row['prompt_filename']
                f.write(json.dumps(triple, ensure_ascii=False) + '\n')
        logging.info(f"Saved deduped triples to {jsonl_output} for compatibility")


def dedupe_triples(df: pd.DataFrame, fuzzy_threshold: float) -> pd.DataFrame:
    """Legacy deduplication function for backward compatibility."""
    df = df.copy()
    df['h_norm'] = df['h'].apply(normalize_text)
    df['r_norm'] = df['r'].apply(normalize_text)
    df['t_norm'] = df['t'].apply(normalize_text)

    # Exact dedupe
    df_deduped = df.drop_duplicates(subset=['h_norm', 'r_norm', 't_norm'])

    logging.info(f"Deduped from {len(df)} to {len(df_deduped)} triples")
    return df_deduped.drop(columns=['h_norm', 'r_norm', 't_norm'])


def dedupe(merged_dir: Path, deduped_path: Path, fuzzy_threshold: float, track_duplicates: bool = True):
    """Main deduplication function with format detection."""
    # Check if we have NeuromorphicKGDB format files
    neuromorphickgdb_files = list(merged_dir.glob("merged_*.txt"))

    if neuromorphickgdb_files:
        logging.info("Detected NeuromorphicKGDB format files, using new deduplication function")
        dedupe_neuromorphickgdb_tables(merged_dir, deduped_path, fuzzy_threshold, track_duplicates)
    else:
        logging.info("Using legacy deduplication function")
        _dedupe_legacy(merged_dir, deduped_path, fuzzy_threshold)


def _dedupe_legacy(merged_dir: Path, deduped_path: Path, fuzzy_threshold: float):
    """Legacy deduplication for JSONL format."""
    parquet_path = merged_dir / "all_triples.parquet"
    txt_path = merged_dir / "all_triples.txt"
    
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif txt_path.exists():
        df = pd.read_csv(txt_path, sep='|', encoding='utf-8', engine='python')
    else:
        logging.error(f"Neither parquet nor txt file found: {parquet_path} or {txt_path}")
        return

    df_deduped = dedupe_triples(df, fuzzy_threshold)

    # Save as txt
    deduped_txt = deduped_path.with_suffix('.txt')
    df_deduped.to_csv(deduped_txt, sep='|', index=False, encoding='utf-8')

    # Save jsonl
    deduped_jsonl = deduped_path.with_suffix('.jsonl')
    with open(deduped_jsonl, 'w', encoding='utf-8') as f:
        for _, row in df_deduped.iterrows():
            f.write(row.to_json(force_ascii=False) + '\n')

    logging.info(f"Saved deduped triples to {deduped_txt} and {deduped_jsonl}")
