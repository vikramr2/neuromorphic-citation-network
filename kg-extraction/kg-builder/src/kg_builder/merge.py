import pandas as pd
from pathlib import Path
import json
import logging
import uuid
from typing import Dict, List
from .config import load_config
from .postmerging import run_postmerging


def merge_neuromorphickgdb_tables(docs_dirs: list[Path], merged_dir: Path):
    """Merge NeuromorphicKGDB-like tables from all documents across multiple directories into global tables."""
    merged_dir.mkdir(exist_ok=True)

    # Initialize global tables
    global_entities = []
    global_documents = []
    global_predications = []

    # Track entity name to global ID mapping for deduplication
    entity_name_to_global_id = {}
    next_entity_id = 0

    total_entity_files = 0
    total_document_files = 0
    total_predication_files = 0

    # Process each docs directory
    for docs_dir in docs_dirs:
        if not docs_dir.exists():
            logging.warning(f"Docs directory does not exist: {docs_dir}")
            continue

        # Find all table files in this directory
        entity_files = list(docs_dir.glob("*_entities.txt"))
        document_files = list(docs_dir.glob("*_documents.txt"))
        predication_files = list(docs_dir.glob("*_predications.txt"))

        total_entity_files += len(entity_files)
        total_document_files += len(document_files)
        total_predication_files += len(predication_files)

        logging.info(f"Processing {docs_dir}: {len(entity_files)} entity files, {len(document_files)} document files, "
                    f"{len(predication_files)} predication files")

        # Merge documents table
        for doc_file in document_files:
            try:
                df = pd.read_csv(doc_file, sep='|', encoding='utf-8', engine='python')
                global_documents.extend(df.to_dict('records'))
            except Exception as e:
                logging.error(f"Error reading document file {doc_file}: {e}")

        # Merge entities table with deduplication
        for entity_file in entity_files:
            try:
                df = pd.read_csv(entity_file, sep='|', encoding='utf-8', engine='python')
                for _, row in df.iterrows():
                    entity_name = row['entity_name']
                    if entity_name not in entity_name_to_global_id:
                        # Create new global entity ID
                        global_entity_id = f"entity_{next_entity_id}"
                        next_entity_id += 1
                        entity_name_to_global_id[entity_name] = global_entity_id

                        # Add to global entities
                        global_entity = {
                            'entity_id': global_entity_id,
                            'entity_name': entity_name,
                            'entity_type': row.get('entity_type', 'unknown')
                        }
                        global_entities.append(global_entity)
            except Exception as e:
                logging.error(f"Error reading entity file {entity_file}: {e}")

        # Merge predications table with entity ID updates
        for pred_file in predication_files:
            try:
                df = pd.read_csv(pred_file, sep='|', encoding='utf-8', engine='python')
                for _, row in df.iterrows():
                    subject_name = row['subject_entity_name']
                    object_name = row['object_entity_name']

                    # Get or create global entity IDs
                    subject_global_id = entity_name_to_global_id.get(subject_name)
                    if not subject_global_id:
                        subject_global_id = f"entity_{next_entity_id}"
                        next_entity_id += 1
                        entity_name_to_global_id[subject_name] = subject_global_id
                        global_entities.append({
                            'entity_id': subject_global_id,
                            'entity_name': subject_name,
                            'entity_type': 'unknown'
                        })

                    object_global_id = entity_name_to_global_id.get(object_name)
                    if not object_global_id:
                        object_global_id = f"entity_{next_entity_id}"
                        next_entity_id += 1
                        entity_name_to_global_id[object_name] = object_global_id
                        global_entities.append({
                            'entity_id': object_global_id,
                            'entity_name': object_name,
                            'entity_type': 'unknown'
                        })

                    # Add predication with global entity IDs
                    global_predication = {
                        'predication_id': row['predication_id'],  # Keep original predication ID
                        'subject_entity_id': subject_global_id,
                        'object_entity_id': object_global_id,
                        'predicate': row['predicate'],
                        'document_id': row['document_id'],
                        'model_name': row['model_name'],
                        'subject_entity_name': subject_name,
                        'object_entity_name': object_name
                    }
                    global_predications.append(global_predication)

            except Exception as e:
                logging.error(f"Error reading predication file {pred_file}: {e}")

    logging.info(f"Total files processed: {total_entity_files} entity files, {total_document_files} document files, "
                f"{total_predication_files} predication files across {len(docs_dirs)} directories")

    # Convert to DataFrames
    entities_df = pd.DataFrame(global_entities)
    documents_df = pd.DataFrame(global_documents)
    predications_df = pd.DataFrame(global_predications)

    # Save merged tables as human-readable delimited files
    if not entities_df.empty:
        entities_path = merged_dir / "merged_entities.txt"
        entities_df.to_csv(entities_path, sep='|', index=False, encoding='utf-8')
        logging.info(f"Saved {len(entities_df)} merged entities to {entities_path}")

    if not documents_df.empty:
        documents_path = merged_dir / "merged_documents.txt"
        documents_df.to_csv(documents_path, sep='|', index=False, encoding='utf-8')
        logging.info(f"Saved {len(documents_df)} merged documents to {documents_path}")

    if not predications_df.empty:
        predications_path = merged_dir / "merged_predications.txt"
        predications_df.to_csv(predications_path, sep='|', index=False, encoding='utf-8')
        logging.info(f"Saved {len(predications_df)} merged predications to {predications_path}")

        # Also save as JSONL for backward compatibility
        jsonl_path = merged_dir / "all_triples.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for _, row in predications_df.iterrows():
                triple = {
                    'h': row['subject_entity_name'],
                    'r': row['predicate'],
                    't': row['object_entity_name'],
                    'document_id': row['document_id'],
                    'model': row['model_name'],
                    'predication_id': row['predication_id']
                }
                # Include prompt_filename if it exists
                if 'prompt_filename' in row and row['prompt_filename']:
                    triple['_prompt'] = row['prompt_filename']
                # Include _prompt if it exists
                if '_prompt' in row:
                    triple['_prompt'] = row['_prompt']
                f.write(json.dumps(triple, ensure_ascii=False) + '\n')
        logging.info(f"Saved triple format to {jsonl_path} for compatibility")

    logging.info(f"✅ Merge completed: {len(global_entities)} entities, "
                f"{len(global_documents)} documents, {len(global_predications)} predications")

    # Run postmerging to clean up entity names
    try:
        config = load_config()
        postmerging_results = run_postmerging(config, merged_dir)
        if postmerging_results:
            logging.info(f"Postmerging results: {postmerging_results}")
    except Exception as e:
        logging.error(f"Error during postmerging: {e}")


def merge_triples(docs_dirs: list[Path], merged_dir: Path):
    """Merge triples from multiple docs directories."""
    # Check if we have the new NeuromorphicKGDB format files in any directory
    neuromorphickgdb_files = []
    for docs_dir in docs_dirs:
        if docs_dir.exists():
            neuromorphickgdb_files.extend(list(docs_dir.glob("*_entities.txt")) + \
                                         list(docs_dir.glob("*_documents.txt")) + \
                                         list(docs_dir.glob("*_predications.txt")))

    if neuromorphickgdb_files:
        logging.info("Detected NeuromorphicKGDB format files, using new merge function")
        merge_neuromorphickgdb_tables(docs_dirs, merged_dir)
    else:
        logging.info("Using legacy JSONL merge function")
        _merge_legacy_jsonl_multiple(docs_dirs, merged_dir)


def _merge_legacy_jsonl_multiple(docs_dirs: list[Path], merged_dir: Path):
    """Original merge function for JSONL files across multiple directories."""
    merged_dir.mkdir(parents=True, exist_ok=True)
    all_triples = []
    
    total_files = 0
    for docs_dir in docs_dirs:
        if not docs_dir.exists():
            logging.warning(f"Docs directory does not exist: {docs_dir}")
            continue
            
        jsonl_files = list(docs_dir.glob("*.jsonl"))
        total_files += len(jsonl_files)
        
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            triple = json.loads(line)
                            all_triples.append(triple)
            except Exception as e:
                logging.error(f"Error reading {jsonl_file}: {e}")

    if not all_triples:
        logging.warning("No triples found in any directory")
        return

    df = pd.DataFrame(all_triples)
    logging.info(f"Merged {len(df)} triples from {total_files} files across {len(docs_dirs)} directories")

    # Save jsonl
    jsonl_path = merged_dir / "all_triples.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(row.to_json(force_ascii=False) + '\n')

    # Save parquet
    parquet_path = merged_dir / "all_triples.parquet"
    df.to_parquet(parquet_path)
    logging.info(f"Saved merged triples to {jsonl_path} and {parquet_path}")
