import asyncio
import aiohttp
import json
import logging
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Set
import faiss
import numpy as np
import pandas as pd
from datetime import datetime
from .config import load_config


class TraceLogger:
    """Singleton class for structured JSONL logging of pipeline steps."""

    _instance = None

    def __new__(cls, log_dir: Path):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.log_dir = log_dir
            cls._instance.log_dir.mkdir(exist_ok=True)
        return cls._instance

    def log_event(self, step: str, input_summary: str, output_summary: str, latency_ms: float, metadata: Dict[str, Any] = None):
        """Log a pipeline event to daily JSONL file."""
        if metadata is None:
            metadata = {}

        event = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "latency_ms": latency_ms,
            "metadata": metadata
        }

        date_str = datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"trace_{date_str}.jsonl"

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')


class AsyncLLMClient:
    """Async client for generating embeddings from LLM providers."""

    def __init__(self, base_url: str, model: str, provider: str = "vllm", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.provider = provider.lower()
        self.timeout = timeout
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text input."""
        if self.provider == "vllm":
            return await self._generate_vllm_embedding(text)
        elif self.provider == "ollama":
            return await self._generate_ollama_embedding(text)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _generate_vllm_embedding(self, text: str) -> List[float]:
        """Generate embedding using vLLM API."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        url = f"{self.base_url}/v1/embeddings"
        payload = {
            "model": self.model,
            "input": text,
            "encoding_format": "float"
        }

        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"vLLM API error {response.status}: {error_text}")

            data = await response.json()
            return data["data"][0]["embedding"]

    async def _generate_ollama_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama API."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text
        }

        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Ollama API error {response.status}: {error_text}")

            data = await response.json()
            return data["embedding"]

    async def generate_text(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
        """Generate text response from LLM."""
        if self.provider == "vllm":
            return await self._generate_vllm_text(prompt, max_tokens, temperature)
        elif self.provider == "ollama":
            return await self._generate_ollama_text(prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _generate_vllm_text(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text using vLLM API."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"vLLM API error {response.status}: {error_text}")

            data = await response.json()
            return data["choices"][0]["message"]["content"]

    async def _generate_ollama_text(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text using Ollama API."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        
        logging.info(f"Ollama request payload: {json.dumps(payload)}")

        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Ollama API error {response.status}: {error_text}")

            data = await response.json()
            logging.info(f"Ollama full response data: {json.dumps(data)}")
            
            text_response = data.get("response", "")
            if not text_response and "thinking" in data:
                logging.warning("Ollama response is empty, falling back to 'thinking' field")
                text_response = data["thinking"]
                
            return text_response


class AsyncMultiModelGraphRAG:
    """Asynchronous GraphRAG implementation with multi-model support."""

    def __init__(self, config):
        self.config = config
        self.embedding_config = config.pipeline.embedding
        self.faiss_config = config.pipeline.faiss
        self.graphrag_config = config.pipeline.graphrag
        self.checkpoint_config = config.pipeline.embedding['checkpointing']
        self.output_dir = Path(config.base.output_dir)
        self.logs_dir = Path(config.base.logs_dir)

        # Initialize checkpoint directory
        self.checkpoint_dir = self.output_dir / self.checkpoint_config['directory']
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Initialize trace logger
        self.logger = TraceLogger(self.logs_dir)

        # Load NetworkX graph
        self.graph = self._load_graph()

        # Load FAISS index and mapping (optional for ingest_and_embed)
        try:
            self.faiss_index, self.entity_mapping = self._load_faiss_index()
        except FileNotFoundError:
            self.faiss_index = None
            self.entity_mapping = None
            logging.info("FAISS index not found - will be created during ingest_and_embed")

        # Initialize LLM clients for different roles
        self.llm_clients = self._init_llm_clients()

    def _load_graph(self) -> nx.DiGraph:
        """Load NetworkX DiGraph from the most refined available triples file."""
        # Check for refined graph (after clustering) first
        graphml_path = self.output_dir / "merged" / "refined_graph.graphml"
        if graphml_path.exists():
            logging.info(f"Loading graph from refined GraphML: {graphml_path}")
            graph = nx.read_graphml(graphml_path)
            # Convert to DiGraph if needed
            if not isinstance(graph, nx.DiGraph):
                graph = graph.to_directed()
            logging.info(f"Loaded refined graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            return graph
        
        # Check for deduped triples (after dedupe)
        deduped_path = self.output_dir / "merged" / "deduped.jsonl"
        if deduped_path.exists():
            logging.info(f"Loading graph from deduped JSONL: {deduped_path}")
            graph = nx.DiGraph()
            with open(deduped_path, 'r', encoding='utf-8') as f:
                for line in f:
                    triple = json.loads(line.strip())
                    head = triple['h']
                    relation = triple['r']
                    tail = triple['t']
                    
                    # Skip triples with invalid (non-string) entities
                    if not isinstance(head, str) or not isinstance(tail, str) or not isinstance(relation, str):
                        logging.warning(f"Skipping triple with invalid entities: {triple}")
                        continue
                        
                    # Skip triples with empty or whitespace-only entities
                    if not head.strip() or not tail.strip() or not relation.strip():
                        logging.warning(f"Skipping triple with empty entities: {triple}")
                        continue
                        
                    # Skip triples with 'nan' entities
                    if head.strip().lower() == 'nan' or tail.strip().lower() == 'nan':
                        logging.warning(f"Skipping triple with 'nan' entities: {triple}")
                        continue
                    
                    graph.add_edge(head, tail, relation=relation, **triple)
            logging.info(f"Loaded deduped graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            return graph
        
        # Fallback to merged triples (after merge/postmerge)
        merged_path = self.output_dir / "merged" / "all_triples.jsonl"
        if merged_path.exists():
            logging.info(f"Loading graph from merged JSONL: {merged_path}")
            graph = nx.DiGraph()
            with open(merged_path, 'r', encoding='utf-8') as f:
                for line in f:
                    triple = json.loads(line.strip())
                    head = triple['h']
                    relation = triple['r']
                    tail = triple['t']
                    
                    # Skip triples with invalid (non-string) entities
                    if not isinstance(head, str) or not isinstance(tail, str) or not isinstance(relation, str):
                        logging.warning(f"Skipping triple with invalid entities: {triple}")
                        continue
                        
                    # Skip triples with empty or whitespace-only entities
                    if not head.strip() or not tail.strip() or not relation.strip():
                        logging.warning(f"Skipping triple with empty entities: {triple}")
                        continue
                        
                    # Skip triples with 'nan' entities
                    if head.strip().lower() == 'nan' or tail.strip().lower() == 'nan':
                        logging.warning(f"Skipping triple with 'nan' entities: {triple}")
                        continue
                    
                    graph.add_edge(head, tail, relation=relation, **triple)
            logging.info(f"Loaded merged graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            return graph
        
        raise FileNotFoundError(f"No graph file found. Checked: {graphml_path}, {deduped_path}, {merged_path}")

    def _load_faiss_index(self) -> tuple:
        """Load FAISS index and entity mapping."""
        index_path = self.output_dir / self.faiss_config['index_path']
        mapping_path = index_path.with_suffix('.mapping.json')

        if not index_path.exists() or not mapping_path.exists():
            raise FileNotFoundError(f"FAISS index or mapping not found: {index_path}, {mapping_path}")

        index = faiss.read_index(str(index_path))
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)

        logging.info(f"Loaded FAISS index with {index.ntotal} vectors")
        return index, mapping

    def _init_llm_clients(self) -> Dict[str, AsyncLLMClient]:
        """Initialize LLM clients for different roles."""
        clients = {}
        endpoints = self.graphrag_config.get('llm_endpoints', {})
        logging.info(f"Initializing LLM clients with endpoints: {endpoints}")

        for role, model_name in endpoints.items():
            # Find model config
            model_config = None
            for mc in self.config.llm.models:
                if mc.name == model_name:
                    model_config = mc
                    break

            if model_config:
                clients[role] = AsyncLLMClient(
                    base_url=model_config.base_url,
                    model=model_config.model,
                    provider=model_config.provider,
                    timeout=getattr(model_config, 'timeout', 30)
                )
            else:
                logging.warning(f"Model config not found for {model_name}, skipping {role}")

        return clients

    def _save_checkpoint(self, embeddings: List[List[float]], entity_names: List[str], batch_idx: int, model_name: str):
        """Save current progress to checkpoint file."""
        if not self.checkpoint_config['enabled']:
            return

        checkpoint_data = {
            'embeddings': embeddings,
            'entity_names': entity_names,
            'batch_idx': batch_idx,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat()
        }

        checkpoint_path = self.checkpoint_dir / f"embedding_checkpoint_{model_name}.pkl"
        try:
            import pickle
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logging.info(f"Saved checkpoint at batch {batch_idx} ({len(embeddings)} embeddings)")
        except Exception as e:
            logging.warning(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self, model_name: str) -> tuple:
        """Load checkpoint if it exists. Returns (embeddings, entity_names, batch_idx) or (None, None, 0)."""
        if not self.checkpoint_config['enabled']:
            return [], [], 0

        checkpoint_path = self.checkpoint_dir / f"embedding_checkpoint_{model_name}.pkl"
        if not checkpoint_path.exists():
            return [], [], 0

        try:
            import pickle
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)

            embeddings = checkpoint_data['embeddings']
            entity_names = checkpoint_data['entity_names']
            batch_idx = checkpoint_data['batch_idx']
            saved_model = checkpoint_data['model_name']
            timestamp = checkpoint_data['timestamp']

            if saved_model != model_name:
                logging.warning(f"Checkpoint model ({saved_model}) differs from current model ({model_name}), ignoring checkpoint")
                return [], [], 0

            logging.info(f"Loaded checkpoint from {timestamp}: {len(embeddings)} embeddings, resuming from batch {batch_idx}")
            return embeddings, entity_names, batch_idx

        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}")
            return [], [], 0

    def _cleanup_checkpoints(self, model_name: str):
        """Remove checkpoint files after successful completion."""
        if not self.checkpoint_config['enabled']:
            return

        checkpoint_path = self.checkpoint_dir / f"embedding_checkpoint_{model_name}.pkl"
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logging.info("Cleaned up checkpoint files")
        except Exception as e:
            logging.warning(f"Failed to cleanup checkpoint: {e}")

    async def ingest_and_embed(self):
        """Ingest entities and generate embeddings for the knowledge graph."""
        model_name = self.embedding_config['model']
        logging.info(f"Starting entity embedding generation using model: {model_name}")

        # Check for existing checkpoint (if checkpointing is enabled)
        existing_embeddings = []
        existing_entity_names = []
        start_batch = 0

        if self.checkpoint_config['enabled']:
            existing_embeddings, existing_entity_names, start_batch = self._load_checkpoint(model_name)
            if existing_embeddings:
                logging.info(f"Resuming from checkpoint: {len(existing_embeddings)} embeddings already generated")

        embeddings = existing_embeddings
        processed_entity_names = existing_entity_names

        # Load entities from the most refined available source
        entities = []
        entity_ids = []
        
        # Check for refined graph (after clustering) first
        refined_graph_path = self.output_dir / "merged" / "refined_graph.graphml"
        if refined_graph_path.exists():
            logging.info(f"Loading entities from refined graph: {refined_graph_path}")
            G_refined = nx.read_graphml(refined_graph_path)
            entities = list(G_refined.nodes())
            # Generate entity IDs for refined entities
            entity_ids = [f"entity_{i}" for i in range(len(entities))]
        else:
            # Check for deduped entities
            deduped_entities_path = self.output_dir / "merged" / "deduped" / "deduped_entities.txt"
            if deduped_entities_path.exists():
                entities_path = deduped_entities_path
                logging.info(f"Using deduped entities file: {entities_path}")
            else:
                # Fallback to merged entities
                merged_entities_path = self.output_dir / "merged" / "merged_entities.txt"
                if merged_entities_path.exists():
                    entities_path = merged_entities_path
                    logging.info(f"Using merged entities file: {entities_path}")
                else:
                    raise FileNotFoundError(f"No entities source found. Checked: {refined_graph_path}, {deduped_entities_path}, {merged_entities_path}")
            
            entities_df = pd.read_csv(entities_path, sep='|', dtype=str)
            entities_df = entities_df.dropna(subset=['entity_name'])  # Remove rows with NaN entity names
            entities_df = entities_df[entities_df['entity_name'].str.strip() != '']  # Remove empty entity names
            entities = entities_df['entity_name'].tolist()
            entity_ids = entities_df['entity_id'].tolist()

        remaining_entities = entities[start_batch * self.embedding_config['batch_size']:]
        remaining_entity_ids = entity_ids[start_batch * self.embedding_config['batch_size']:]

        logging.info(f"Processing {len(remaining_entities)} entities for embedding using model: {model_name}")

        # Create client
        client_config = self._get_embedding_client_config()
        async with AsyncLLMClient(**client_config) as client:
            # Process entities in chunks with checkpointing
            batch_size = self.embedding_config['batch_size']
            checkpoint_interval = self.checkpoint_config['save_interval']
            total_batches = (len(entities) + batch_size - 1) // batch_size  # Total batches for all entities
            remaining_batches = (len(remaining_entities) + batch_size - 1) // batch_size  # Batches remaining

            for i in range(0, len(remaining_entities), batch_size):
                chunk_entities = remaining_entities[i:i + batch_size]
                chunk_entity_ids = remaining_entity_ids[i:i + batch_size]
                current_batch = start_batch + (i // batch_size)

                # Generate embeddings for this chunk
                chunk_embeddings = await self._embed_batch_chunk(client, chunk_entities, model_name, current_batch, total_batches)

                # Add to our collections
                embeddings.extend(chunk_embeddings)
                processed_entity_names.extend(chunk_entities)

                # Save checkpoint periodically (if enabled)
                if self.checkpoint_config['enabled'] and (current_batch + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(embeddings, processed_entity_names, current_batch + 1, model_name)

        # Save embeddings and build FAISS index
        self._save_embeddings_and_index(embeddings, processed_entity_names, model_name)

        # Cleanup checkpoints after successful completion (if enabled)
        if self.checkpoint_config['enabled'] and self.checkpoint_config['cleanup_on_success']:
            self._cleanup_checkpoints(model_name)

        logging.info(f"Successfully generated embeddings for {len(processed_entity_names)} entities using model: {model_name}")

    async def _embed_batch_chunk(self, client: AsyncLLMClient, entities: List[str], model_name: str, batch_num: int, total_batches: int) -> List[List[float]]:
        """Generate embeddings for a single chunk of entities."""
        logging.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(entities)} entities) using model: {model_name}")

        # Generate embeddings with error handling
        embeddings = []
        for entity in entities:
            try:
                embedding = await client.generate_embedding(entity)
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    logging.warning(f"Received None embedding for entity: {entity}")
                    embeddings.append(None)
            except Exception as e:
                logging.error(f"Failed to generate embedding for entity '{entity}': {e}")
                embeddings.append(None)

        # Progress logging
        progress = (batch_num + 1) * len(entities) / 405916 * 100  # Approximate total
        logging.info(f"Batch {batch_num + 1}/{total_batches} complete using model: {model_name}")

        return embeddings

    def _get_embedding_client_config(self) -> Dict[str, Any]:
        """Get configuration for the embedding client."""
        model_name = self.embedding_config['model']

        # Find the model config in LLM models
        for model_config in self.config.llm.models:
            if model_config.name == model_name:
                return {
                    "base_url": model_config.base_url,
                    "model": model_config.model,
                    "provider": model_config.provider,
                    "timeout": getattr(model_config, 'timeout', 30)
                }

        # Fallback: assume it's an Ollama model
        logging.warning(f"Model {model_name} not found in LLM config, assuming Ollama")
        return {
            "base_url": "http://localhost:11434",
            "model": model_name,
            "provider": "ollama",
            "timeout": 30
        }

    async def _embed_batch(self, client: AsyncLLMClient, entities: List[str], batch_size: int, model_name: str, start_batch: int = 0) -> List[List[float]]:
        """Generate embeddings for entities in batches."""
        embeddings = []
        total_entities = len(entities)

        for i in range(0, total_entities, batch_size):
            batch_num = start_batch + (i // batch_size) + 1
            batch = entities[i:i + batch_size]
            logging.info(f"Processing batch {batch_num}/{(total_entities + batch_size - 1)//batch_size + start_batch} "
                        f"({len(batch)} entities) using model: {model_name}")

            # Generate embeddings with error handling
            batch_embeddings = []
            for entity in batch:
                try:
                    embedding = await client.generate_embedding(entity)
                    if embedding is not None:
                        batch_embeddings.append(embedding)
                    else:
                        logging.warning(f"Received None embedding for entity: {entity}")
                        batch_embeddings.append(None)
                except Exception as e:
                    logging.error(f"Failed to generate embedding for entity '{entity}': {e}")
                    batch_embeddings.append(None)

            embeddings.extend(batch_embeddings)

            # Progress logging
            processed = min(i + batch_size, total_entities)
            progress = processed / total_entities * 100
            logging.info(f"{progress:.1f}% complete using model: {model_name}")

        return embeddings

    def _save_embeddings_and_index(self, embeddings: List[List[float]], entity_names: List[str], model_name: str):
        """Save embeddings and build FAISS index."""
        # Filter out None embeddings and corresponding entity names
        valid_embeddings = []
        valid_entity_names = []

        for embedding, entity_name in zip(embeddings, entity_names):
            if embedding is not None:
                valid_embeddings.append(embedding)
                valid_entity_names.append(entity_name)
            else:
                logging.warning(f"Skipping entity {entity_name} due to failed embedding generation")

        if not valid_embeddings:
            raise ValueError("No valid embeddings generated - cannot create FAISS index")

        logging.info(f"Creating FAISS index with {len(valid_embeddings)} valid embeddings (filtered out {len(embeddings) - len(valid_embeddings)} failed embeddings)")

        # Convert to numpy array
        embeddings_array = np.array(valid_embeddings, dtype=np.float32)

        # Build FAISS index
        dimension = embeddings_array.shape[1]
        if self.faiss_config['index_type'] == "IndexFlatIP":
            index = faiss.IndexFlatIP(dimension)
        elif self.faiss_config['index_type'] == "IndexFlatL2":
            index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.faiss_config['index_type']}")

        # Add vectors to index
        index.add(embeddings_array)

        # Save index
        index_path = self.output_dir / self.faiss_config['index_path']
        faiss.write_index(index, str(index_path))
        logging.info(f"Saved FAISS index to {index_path} (generated with model: {model_name})")

        # Save entity name mapping (only for valid embeddings)
        mapping = {str(i): name for i, name in enumerate(valid_entity_names)}
        mapping_path = index_path.with_suffix('.mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        logging.info(f"Saved entity mapping to {mapping_path} (for embeddings from model: {model_name})")

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a GraphRAG query with full pipeline."""
        if not self.faiss_index or not self.entity_mapping:
            raise RuntimeError("FAISS index not loaded. Run ingest_and_embed first.")

        start_time = datetime.now()

        # 1. Entity Extraction
        entities = await self._extract_entities(query)

        # 2. Entry Point Discovery
        entry_nodes = await self._discover_entry_points(entities)

        # 3. Graph Traversal
        subgraph_triples, doc_ids = self._traverse_graph(entry_nodes)

        # 4. Hybrid Reranking
        top_triples, top_docs = await self._hybrid_reranking(query, subgraph_triples, doc_ids)

        # 5. Answer Generation
        answer, prompt = await self._generate_answer(query, top_triples, top_docs)

        # Save intermediate results
        result = {
            "query": query,
            "answer": answer,
            "prompt": prompt,
            "entities": entities,
            "entry_nodes": list(entry_nodes),
            "subgraph_triples": subgraph_triples,
            "top_triples": top_triples,
            "top_docs": top_docs,
            "timestamp": start_time.isoformat()
        }

        result_path = self.output_dir / "latest_query_result.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        # Log completion
        total_latency = (datetime.now() - start_time).total_seconds() * 1000
        self.logger.log_event(
            step="QueryProcessing",
            input_summary=f"Query: {query[:100]}...",
            output_summary=f"Answer: {answer[:100]}...",
            latency_ms=total_latency,
            metadata={"entity_count": len(entities), "triple_count": len(subgraph_triples)}
        )

        return {
            "answer": answer,
            "prompt": prompt,
            "entities": entities,
            "entry_nodes": list(entry_nodes),
            "subgraph_triples": subgraph_triples,
            "top_triples": top_triples,
            "top_docs": top_docs,
            "sources": top_docs
        }

    async def _extract_entities(self, query: str) -> List[str]:
        """Extract key entities from query using extractor model."""
        start_time = datetime.now()

        if 'extractor' not in self.llm_clients:
            logging.warning("Extractor model not configured, using keyword extraction")
            # Simple keyword extraction as fallback
            entities = [word.strip() for word in query.split() if len(word.strip()) > 3]
        else:
            client = self.llm_clients['extractor']

            # Read entity extraction prompt from file
            prompt_template_path = Path(__file__).parent / "prompts" / "entity_extraction.md"
            with open(prompt_template_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()

            prompt = f"{prompt_template}\n\nQuery: {query}"

            # Get configuration parameters for extractor
            extractor_max_tokens = self.graphrag_config.get('extractor_max_tokens', 256)
            extractor_temperature = self.graphrag_config.get('extractor_temperature', 0.2)

            async with client:
                response = await client.generate_text(prompt, max_tokens=extractor_max_tokens, temperature=extractor_temperature)
                logging.info(f"Raw extractor response: {response}")
                try:
                    # Clean the response by removing markdown code blocks if present
                    cleaned_response = response.strip()
                    if cleaned_response.startswith('```json'):
                        cleaned_response = cleaned_response[7:]
                    if cleaned_response.endswith('```'):
                        cleaned_response = cleaned_response[:-3]
                    cleaned_response = cleaned_response.strip()

                    entities = json.loads(cleaned_response)
                    if not isinstance(entities, list):
                        entities = [cleaned_response.strip()]
                except json.JSONDecodeError:
                    # Fallback 1: extract quoted strings (JSON-like but broken)
                    import re
                    quoted_terms = re.findall(r'"([^"]*)"', response)
                    if quoted_terms:
                        entities = quoted_terms
                    else:
                        # Fallback 2: extract bullet points (Markdown list)
                        bullet_terms = re.findall(r'^\s*[-*]\s+(.*)$', response, re.MULTILINE)
                        if bullet_terms:
                            # Clean up markdown formatting like **bold**
                            entities = [term.replace('**', '').strip() for term in bullet_terms]
                        else:
                            # Fallback 3: Split by commas and clean up
                            entities = [term.strip().strip('"').strip("'") for term in response.split(',') if term.strip()]
                            entities = [e for e in entities if len(e) > 2][:8]  # Limit to 8 terms, minimum 3 chars

        latency = (datetime.now() - start_time).total_seconds() * 1000
        self.logger.log_event(
            step="EntityExtraction",
            input_summary=f"Query: {query[:50]}...",
            output_summary=f"Entities: {entities}",
            latency_ms=latency
        )

        return entities

    async def _discover_entry_points(self, entities: List[str]) -> Set[str]:
        """Discover entry points via vector and keyword search."""
        start_time = datetime.now()

        entry_nodes = set()

        # Vector search
        for entity in entities:
            # Generate embedding for entity
            client_config = self._get_embedding_client_config()
            async with AsyncLLMClient(**client_config) as client:
                embedding = await client.generate_embedding(entity)

            # Search FAISS
            embedding_array = np.array([embedding], dtype=np.float32)
            distances, indices = self.faiss_index.search(embedding_array, k=5)

            for idx in indices[0]:
                if str(idx) in self.entity_mapping:
                    node_id = self.entity_mapping[str(idx)]
                    entry_nodes.add(node_id)

        # Keyword search
        for entity in entities:
            for node in self.graph.nodes():
                # Skip non-string nodes
                if not isinstance(node, str):
                    continue
                if entity.lower() in node.lower():
                    entry_nodes.add(node)

        latency = (datetime.now() - start_time).total_seconds() * 1000
        self.logger.log_event(
            step="EntryPointDiscovery",
            input_summary=f"Entities: {entities}",
            output_summary=f"Entry nodes: {len(entry_nodes)}",
            latency_ms=latency
        )

        return entry_nodes

    def _traverse_graph(self, entry_nodes: Set[str]) -> tuple:
        """Perform BFS traversal to collect triples and documents."""
        start_time = datetime.now()

        visited = set()
        subgraph_triples = []
        doc_ids = set()

        k_hops = self.graphrag_config.get('k_hops', 2)

        for start_node in entry_nodes:
            if start_node not in self.graph:
                continue

            # BFS traversal
            queue = [(start_node, 0)]
            visited.add(start_node)

            while queue:
                current_node, depth = queue.pop(0)
                if depth >= k_hops:
                    continue

                for neighbor in self.graph.successors(current_node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

                    # Collect triple
                    edge_data = self.graph.get_edge_data(current_node, neighbor)
                    triple = {
                        'h': current_node,
                        'r': edge_data['relation'],
                        't': neighbor,
                        **{k: v for k, v in edge_data.items() if k not in ['relation']}
                    }
                    subgraph_triples.append(triple)
                    if 'document_id' in edge_data:
                        doc_ids.add(edge_data['document_id'])

        latency = (datetime.now() - start_time).total_seconds() * 1000
        self.logger.log_event(
            step="GraphTraversal",
            input_summary=f"Entry nodes: {len(entry_nodes)}",
            output_summary=f"Triples: {len(subgraph_triples)}, Docs: {len(doc_ids)}",
            latency_ms=latency
        )

        return subgraph_triples, list(doc_ids)

    async def _hybrid_reranking(self, query: str, triples: List[Dict], doc_ids: List[str]) -> tuple:
        """Perform hybrid reranking with graph and semantic scores."""
        start_time = datetime.now()

        if 'ranker' not in self.llm_clients:
            logging.warning("Ranker model not configured, using graph-only scoring")
            # Return top triples by document frequency
            doc_freq = {}
            for triple in triples:
                doc_id = triple.get('document_id', '')
                doc_freq[doc_id] = doc_freq.get(doc_id, 0) + 1

            sorted_docs = sorted(doc_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            top_docs = [doc_id for doc_id, _ in sorted_docs]
            top_triples = [t for t in triples if t.get('document_id') in top_docs][:20]
        else:
            # Implement full hybrid scoring
            # TODO: Implement semantic scoring with ranker model
            top_triples = triples[:20]
            top_docs = doc_ids[:5]

        latency = (datetime.now() - start_time).total_seconds() * 1000
        self.logger.log_event(
            step="HybridReranking",
            input_summary=f"Query: {query[:50]}..., Triples: {len(triples)}",
            output_summary=f"Top triples: {len(top_triples)}, Top docs: {len(top_docs)}",
            latency_ms=latency
        )

        return top_triples, top_docs

    async def _generate_answer(self, query: str, triples: List[Dict], docs: List[str]) -> tuple:
        """Generate final answer using reasoner model. Returns (answer, prompt)."""
        start_time = datetime.now()
        prompt = ""

        if 'reasoner' not in self.llm_clients:
            answer = "Reasoner model not configured. Graph context available but no synthesis performed."
        else:
            client = self.llm_clients['reasoner']
            context = f"Query: {query}\n\nGraph Relations:\n"
            for triple in triples[:10]:  # Limit context
                context += f"{triple['h']} -> {triple['r']} -> {triple['t']}\n"

            context += f"\nDocument IDs: {docs[:5]}"
            
            # Add document content
            context += "\n\nDocument Contents:\n"
            for doc_id in docs[:5]:
                doc_path = self.output_dir / "docs" / f"{doc_id}_documents.txt"
                if doc_path.exists():
                    try:
                        with open(doc_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Truncate content if too long to avoid context window issues
                            # User requested support for large documents (~16k words), so we increase limit to 100k chars
                            if len(content) > 100000:
                                content = content[:100000] + "... (truncated)"
                            context += f"\nDocument {doc_id}:\n{content}\n"
                    except Exception as e:
                        logging.warning(f"Failed to read document {doc_id}: {e}")
                else:
                    logging.warning(f"Document file not found: {doc_path}")

            # Read prompt template from file
            prompt_template_path = Path(__file__).parent / "prompts" / "graphrag_query.md"
            with open(prompt_template_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()

            # Format the prompt with context and query
            prompt = prompt_template.format(context=context, query=query)

            # Get configuration parameters
            max_tokens = self.graphrag_config.get('max_tokens', 1024)
            temperature = self.graphrag_config.get('temperature', 0.2)

            async with client:
                answer = await client.generate_text(prompt, max_tokens=max_tokens, temperature=temperature)

        latency = (datetime.now() - start_time).total_seconds() * 1000
        self.logger.log_event(
            step="AnswerGeneration",
            input_summary=f"Query: {query[:50]}..., Context: {len(triples)} triples",
            output_summary=f"Answer: {answer[:100]}...",
            latency_ms=latency
        )

        return answer, prompt
