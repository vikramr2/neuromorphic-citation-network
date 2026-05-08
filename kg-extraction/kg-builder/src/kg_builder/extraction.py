from pathlib import Path
import json
import logging
import time
import uuid
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import pandas as pd
from openai import NotFoundError
from .mongo_utils import connect_mongo, get_documents, extract_text
from .llm_client import MultiLLMClient
from .io_utils import normalize_entity, normalize_relation


@dataclass
class Entity:
    """Entity table entry similar to NeuromorphicKGDB."""
    entity_id: str
    entity_name: str
    entity_type: str = "unknown"

    def to_dict(self):
        return asdict(self)


@dataclass
class Document:
    """Document/Publication table entry."""
    document_id: str
    title: str = ""
    abstract: str = ""
    source: str = "mongodb"
    content_length: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class Predication:
    """Predication table entry for NeuromorphicKGDB."""
    predication_id: str
    subject_entity_id: str
    object_entity_id: str
    predicate: str
    document_id: str
    model_name: str
    subject_entity_name: str = ""  # For convenience
    object_entity_name: str = ""   # For convenience
    prompt_filename: str = ""      # Prompt file used for extraction

    def to_dict(self):
        return asdict(self)


@dataclass
class NeuromorphicKGDBTables:
    """Container for all NeuromorphicKGDB-like tables."""
    entities: List[Entity]
    documents: List[Document]
    predications: List[Predication]

    def __init__(self, exclusions: list[str] = None):
        if exclusions is None:
            exclusions = []
        self.exclusions = exclusions
        self.entities = []
        self.documents = []
        self.predications = []
        self.entity_name_to_id = {}  # Cache for entity name -> ID mapping

    def get_or_create_entity(self, entity_name: str, entity_type: str = "unknown") -> str:
        """Get existing entity ID or create new entity and return its ID."""
        # Normalize the entity name
        normalized_name = normalize_entity(entity_name, self.exclusions)

        if normalized_name in self.entity_name_to_id:
            return self.entity_name_to_id[normalized_name]

        entity_id = str(uuid.uuid4())
        entity = Entity(
            entity_id=entity_id,
            entity_name=normalized_name,
            entity_type=entity_type
        )
        self.entities.append(entity)
        self.entity_name_to_id[normalized_name] = entity_id
        return entity_id

    def add_document(self, document: Document):
        """Add a document to the documents table."""
        self.documents.append(document)

    def add_predication(self, subject_name: str, predicate: str, object_name: str,
                       document_id: str, model_name: str, prompt_filename: str = ""):
        """Add a predication (triple) to the predications table."""
        # Normalize predicate
        normalized_predicate = normalize_relation(predicate)

        subject_entity_id = self.get_or_create_entity(subject_name)
        object_entity_id = self.get_or_create_entity(object_name)

        predication_id = str(uuid.uuid4())
        predication = Predication(
            predication_id=predication_id,
            subject_entity_id=subject_entity_id,
            object_entity_id=object_entity_id,
            predicate=normalized_predicate,
            document_id=document_id,
            model_name=model_name,
            subject_entity_name=normalize_entity(subject_name, self.exclusions),
            object_entity_name=normalize_entity(object_name, self.exclusions),
            prompt_filename=prompt_filename
        )
        self.predications.append(predication)

    def to_dataframes(self):
        """Convert tables to pandas DataFrames for easy saving/merging."""
        entities_df = pd.DataFrame([e.to_dict() for e in self.entities])
        documents_df = pd.DataFrame([d.to_dict() for d in self.documents])
        predications_df = pd.DataFrame([p.to_dict() for p in self.predications])

        # Debug logging
        logging.debug(f"DataFrames created: entities={len(entities_df)}, documents={len(documents_df)}, predications={len(predications_df)}")

        return {
            'entities': entities_df,
            'documents': documents_df,
            'predications': predications_df
        }

    def save_tables(self, output_dir: Path, doc_id: str, append: bool = False):
        """Save all tables as human-readable delimited files.
        
        Args:
            output_dir: Directory to save tables
            doc_id: Document ID
            append: If True, append to existing files instead of overwriting
        """
        tables = self.to_dataframes()

        for table_name, df in tables.items():
            output_path = output_dir / f"{doc_id}_{table_name}.txt"
            
            if df.empty:
                # For empty DataFrames in append mode, don't create/truncate files
                if not append:
                    # Create empty file with just headers for empty DataFrames
                    if len(df.columns) > 0:
                        df.head(0).to_csv(output_path, sep='|', index=False, encoding='utf-8')
                    else:
                        # If no columns, create empty file
                        output_path.touch()
                    logging.debug(f"Created empty {table_name} file for document {doc_id}")
                # In append mode with empty df, do nothing
                continue
            
            if append and output_path.exists():
                try:
                    # Read existing data
                    existing_df = pd.read_csv(output_path, sep='|', encoding='utf-8')
                    # Append new data
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    # Remove duplicates if any (based on primary key columns)
                    if table_name == 'entities':
                        combined_df = combined_df.drop_duplicates(subset=['entity_id'])
                    elif table_name == 'documents':
                        combined_df = combined_df.drop_duplicates(subset=['document_id'])
                    elif table_name == 'predications':
                        combined_df = combined_df.drop_duplicates(subset=['predication_id'])
                    
                    # Save combined data
                    combined_df.to_csv(output_path, sep='|', index=False, encoding='utf-8')
                    logging.debug(f"Appended {len(df)} {table_name} to {output_path} (total: {len(combined_df)})")
                except Exception as e:
                    logging.warning(f"Could not append to existing {table_name} file {output_path}: {e}. Overwriting instead.")
                    df.to_csv(output_path, sep='|', index=False, encoding='utf-8')
                    logging.debug(f"Saved {len(df)} {table_name} to {output_path}")
            else:
                # Save as delimited file with uncommon delimiter
                df.to_csv(output_path, sep='|', index=False, encoding='utf-8')
                logging.debug(f"Saved {len(df)} {table_name} to {output_path}")


def get_next_begin_idx(input_dir: Path, docs_dir: Path, total_docs: int) -> int:
    """
    Determine the next begin_idx based on processed documents.
    
    Args:
        input_dir: Directory containing input .txt files (sorted by filename)
        docs_dir: Directory containing processed .jsonl files
        total_docs: Total number of documents available
    
    Returns:
        Next begin_idx to use for restarting
    """
    # Get all input files sorted by filename
    input_files = sorted([f for f in input_dir.glob("*.txt")])
    
    if not input_files:
        return 0
    
    # Get all processed document IDs
    processed_ids = set()
    for jsonl_file in docs_dir.glob("*.jsonl"):
        doc_id = jsonl_file.stem
        processed_ids.add(doc_id)
    
    # Find the first unprocessed document in the sorted input list
    for idx, input_file in enumerate(input_files):
        doc_id = input_file.stem
        if doc_id not in processed_ids:
            return idx
    
    # All documents have been processed
    return total_docs


def get_processed_models_for_document(jsonl_path: Path, model_names: list) -> set:
    """
    Check which models have already provided triples for a given document.
    
    Args:
        jsonl_path: Path to the JSONL file for the document
        model_names: List of all configured model names
    
    Returns:
        Set of model names that have already been processed
    """
    processed_models = set()
    
    if not jsonl_path.exists():
        return processed_models
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        triple = json.loads(line)
                        model = triple.get('_model')
                        if model and model in model_names:
                            processed_models.add(model)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logging.warning(f"Could not read JSONL file {jsonl_path}: {e}")
    
    return processed_models


def extract_triples(config, docs_dir: Path, filename: str = None, begin_idx: int = 0, end_idx: int = None):
    """
    Extract triples from documents.

    Args:
        config: Configuration object
        docs_dir: Directory to save extracted triples
        filename: Specific input file to process (optional)
        begin_idx: Starting document index (0-based) for range processing
        end_idx: Ending document index (exclusive) for range processing
    """
    if filename:
        # Process single file
        input_dir = Path(config.base.output_dir) / "input"
        file_path = input_dir / filename

        if not file_path.exists():
            logging.error(f"Input file not found: {file_path}")
            return

        # Read the document from file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Extract document ID from filename (remove .txt extension)
        doc_id = filename.replace('.txt', '') if filename.endswith('.txt') else filename

        # Create a mock document object for processing
        docs = [{
            '_id': doc_id,
            'title': f'Document {doc_id}',
            'abstract': '',
            'sections': [{'content': text}]
        }]
    else:
        # Original behavior: get documents from MongoDB
        collection = connect_mongo(config.mongo.uri, config.mongo.database, config.mongo.collection)
        docs = get_documents(collection, config.mongo.limit)

        # Apply range slicing if specified
        if begin_idx > 0 or end_idx is not None:
            original_count = len(docs)
            docs = docs[begin_idx:end_idx]
            range_end = begin_idx + len(docs) - 1 if docs else begin_idx - 1
            logging.info(f"Processing document range: {begin_idx} to {range_end} (total in range: {len(docs)})")
            if len(docs) == 0:
                logging.warning(f"No documents in the specified range {begin_idx}:{end_idx}. Fetched {original_count} documents total.")

    # Use LLM config directly - MultiLLMClient expects Pydantic model objects

    # Use LLM config directly - MultiLLMClient expects Pydantic model objects
    if hasattr(config.llm, 'models') and config.llm.models:
        # New multi-model configuration
        model_configs = config.llm.models
    else:
        # Fallback to single model configuration for backward compatibility
        # Create a temporary model config object
        from .config import LLMModelConfig
        model_configs = [LLMModelConfig(
            name='default',
            base_url=config.llm.base_url,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            provider='vllm'
        )]

    exclusions = config.pipeline.normalize.get('exclusions', [])
    multi_llm = MultiLLMClient(model_configs, exclusions, default_timeout=config.llm.parallel_execution.request_timeout)
    prompt_files = config.pipeline.extraction.get('prompt_files', ['prompts/triple_extraction.md'])
    if isinstance(prompt_files, str):
        prompt_files = [prompt_files]
    prompt_paths = [Path(__file__).parent / pf for pf in prompt_files]

    docs_dir.mkdir(exist_ok=True)

    # Get all model names for the new loop ordering
    model_names = [client_info['name'] for client_info in multi_llm.clients]
    logging.info(f"🔄 Starting extraction with {len(model_names)} models, {len(prompt_paths)} prompts, and {len(docs)} documents")
    logging.info(f"📊 Total combinations to process: {len(model_names)} × {len(prompt_paths)} × {len(docs)} = {len(model_names) * len(prompt_paths) * len(docs)}")

    # New efficient loop ordering: prompt first, then document, then parallel models
    # This allows parallel execution across servers for each document+prompt combination
    extraction_start_time = time.time()
    total_combinations_processed = 0
    successful_combinations = 0

    for prompt_idx, prompt_path in enumerate(prompt_paths):
        prompt_start_time = time.time()
        logging.info(f"📝 Processing prompt {prompt_idx+1}/{len(prompt_paths)}: {prompt_path.name}")

        # Track statistics for this prompt
        prompt_combinations = 0
        prompt_successful = 0
        prompt_total_triples = 0

        for doc_idx, doc in enumerate(docs):
            doc_id = str(doc.get('_id', f'doc_{doc_idx}'))
            text = extract_text(doc)

            # For token budget, truncate to ~8000 tokens assuming ~4 chars per token
            max_chars = config.pipeline.extraction.get('max_chars', 16000)  # Read from config with default
            original_length = len(text)

            if len(text) > max_chars:
                text = text[:max_chars] + "..."
                logging.debug(f"[{doc_idx+1}/{len(docs)}] Truncated text for doc {doc_id}: {original_length} -> {max_chars} chars")
            else:
                logging.debug(f"[{doc_idx+1}/{len(docs)}] Document {doc_id} length: {original_length} chars (no truncation needed)")

            # Check which models have already been processed for this document
            jsonl_path = docs_dir / f"{doc_id}.jsonl"
            processed_models = get_processed_models_for_document(jsonl_path, model_names)
            
            if len(processed_models) == len(model_names):
                logging.info(f"⏭️  [{doc_idx+1}/{len(docs)}] Skipping doc {doc_id} - all {len(model_names)} models already processed")
                continue
            
            remaining_models = [m for m in model_names if m not in processed_models]
            logging.info(f"🔄 [{doc_idx+1}/{len(docs)}] Processing doc {doc_id} - {len(remaining_models)}/{len(model_names)} models remaining: {remaining_models}")

            try:
                # Extract triples from remaining models in parallel across different servers
                doc_start_time = time.time()
                model_results = multi_llm.extract_triples_multi_parallel(text, prompt_path, config.llm, doc_id, models_to_run=remaining_models)

                doc_time = time.time() - doc_start_time
                doc_combinations = len(model_results)
                doc_successful = sum(1 for triples in model_results.values() if len(triples) > 0)
                doc_total_triples = sum(len(triples) for triples in model_results.values())

                total_combinations_processed += doc_combinations
                successful_combinations += doc_successful

                logging.info(f"✅ [{doc_idx+1}/{len(docs)}] Extracted {doc_total_triples} triples from {doc_successful}/{doc_combinations} models for doc {doc_id} using {prompt_path.name} in {doc_time:.2f}s")

                # Process and save results for each model in this document+prompt combination
                combination_start_time = time.time()

                for model_name, triples in model_results.items():
                    # Create NeuromorphicKGDB-like tables for this document
                    tables = NeuromorphicKGDBTables(exclusions)

                    # Add document to documents table
                    document = Document(
                        document_id=doc_id,
                        title=doc.get('title', ''),
                        abstract=doc.get('abstract', ''),
                        source=f"{config.mongo.database}.{config.mongo.collection}",
                        content_length=len(text)
                    )
                    tables.add_document(document)

                    # Process triples for this document
                    for triple in triples:
                        if 'h' in triple and 'r' in triple and 't' in triple:
                            logging.debug(f"Adding predication: {triple['h']} -> {triple['r']} -> {triple['t']}")
                            tables.add_predication(
                                subject_name=triple['h'],
                                predicate=triple['r'],
                                object_name=triple['t'],
                                document_id=doc_id,
                                model_name=model_name,
                                prompt_filename=prompt_path.name
                            )

                    # Save raw triples as JSONL with correct model name and prompt
                    all_triples = []
                    for triple in triples:
                        triple_with_model = triple.copy()
                        triple_with_model['_model'] = model_name
                        triple_with_model['_prompt'] = prompt_path.name
                        all_triples.append(triple_with_model)

                    if all_triples:
                        jsonl_path = docs_dir / f"{doc_id}.jsonl"
                        # Read existing file if it exists, otherwise create new
                        existing_triples = []
                        if jsonl_path.exists():
                            try:
                                with open(jsonl_path, 'r', encoding='utf-8') as f:
                                    for line in f:
                                        if line.strip():
                                            existing_triples.append(json.loads(line))
                            except Exception as e:
                                logging.warning(f"Could not read existing JSONL file {jsonl_path}: {e}")

                        # Append new triples
                        existing_triples.extend(all_triples)

                        # Save all triples
                        with open(jsonl_path, 'w', encoding='utf-8') as f:
                            for triple in existing_triples:
                                f.write(json.dumps(triple, ensure_ascii=False) + '\n')

                        logging.debug(f"Saved {len(all_triples)} triples from {model_name} to {jsonl_path} (total: {len(existing_triples)})")

                    # Save NeuromorphicKGDB tables (append mode for multiple model-prompt combinations)
                    tables.save_tables(docs_dir, doc_id, append=True)

                combination_time = time.time() - combination_start_time
                logging.info(f"💾 [{doc_idx+1}/{len(docs)}] Saved results for doc {doc_id} + {prompt_path.name}: {doc_total_triples} triples from {doc_combinations} models in {combination_time:.2f}s")

                # Update prompt statistics
                prompt_combinations += doc_combinations
                prompt_successful += doc_successful
                prompt_total_triples += doc_total_triples

            except Exception as e:
                logging.error(f"❌ [{doc_idx+1}/{len(docs)}] Failed to extract triples for doc {doc_id} with {prompt_path.name}: {e}")
                total_combinations_processed += len(remaining_models)

        prompt_time = time.time() - prompt_start_time

        logging.info(f"📊 Prompt {prompt_path.name} completed: {prompt_successful}/{prompt_combinations} model combinations successful, "
                    f"{prompt_total_triples} total triples in {prompt_time:.2f}s")

    total_extraction_time = time.time() - extraction_start_time

    # Summary statistics
    total_docs = len(docs)
    total_models = len(multi_llm.clients)
    total_prompts = len(prompt_paths)
    if total_docs > 0:
        avg_length = sum(len(extract_text(doc)) for doc in docs) / total_docs
        truncated_docs = sum(1 for doc in docs if len(extract_text(doc)) > max_chars)
        truncation_rate = truncated_docs / total_docs * 100

        logging.info(f"📈 Extraction Summary: {total_docs} documents processed | "
                    f"Avg length: {avg_length:.0f} chars | "
                    f"Truncation rate: {truncation_rate:.1f}% ({truncated_docs}/{total_docs} docs) | "
                    f"Max chars allowed: {max_chars}")
        logging.info(f"🤖 Models: {total_models} | Prompts: {total_prompts} | Total combinations: {total_combinations_processed}")
        logging.info(f"✅ Successful combinations: {successful_combinations}/{total_combinations_processed} ({successful_combinations/total_combinations_processed*100:.1f}%)")
        logging.info(f"⏱️  Total extraction time: {total_extraction_time:.2f}s | Avg time per combination: {total_extraction_time/total_combinations_processed:.2f}s")


def deduplicate_triples(triples: list) -> list:
    """Remove duplicate triples based on (head, relation, tail) combination."""
    seen = set()
    unique_triples = []

    for triple in triples:
        key = (triple.get('h', ''), triple.get('r', ''), triple.get('t', ''))
        if key not in seen:
            seen.add(key)
            unique_triples.append(triple)

    return unique_triples


def save_triples(triples: list, path: Path):
    """Save triples to JSONL file with proper Unicode support."""
    with open(path, 'w', encoding='utf-8') as f:
        for triple in triples:
            f.write(json.dumps(triple, ensure_ascii=False) + '\n')


def save_input_documents(config, input_dir: Path, begin_idx: int = 0, end_idx: int = None):
    """
    Save input documents from MongoDB to local files.

    Args:
        config: Configuration object
        input_dir: Directory to save input documents
        begin_idx: Starting document index (0-based) for range processing
        end_idx: Ending document index (exclusive) for range processing
    """
    collection = connect_mongo(config.mongo.uri, config.mongo.database, config.mongo.collection)
    docs = get_documents(collection, config.mongo.limit)

    # Apply range slicing if specified
    if begin_idx > 0 or end_idx is not None:
        original_count = len(docs)
        docs = docs[begin_idx:end_idx]
        range_end = begin_idx + len(docs) - 1 if docs else begin_idx - 1
        logging.info(f"Saving document range: {begin_idx} to {range_end} (total in range: {len(docs)})")
        if len(docs) == 0:
            logging.warning(f"No documents in the specified range {begin_idx}:{end_idx}. Fetched {original_count} documents total.")

    input_dir.mkdir(exist_ok=True, parents=True)
    saved_count = 0

    for doc in docs:
        doc_id = str(doc.get('_id', f'doc_{saved_count}'))
        text = extract_text(doc)

        # Save the document text to a file named after the document ID
        file_path = input_dir / f"{doc_id}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)

        saved_count += 1
        logging.info(f"Saved document {doc_id} to {file_path}")

    logging.info(f"Successfully saved {saved_count} input documents to {input_dir}")
