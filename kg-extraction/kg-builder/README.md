# KG Builder

A Python project to build knowledge graphs (KGs) from publications stored in MongoDB using LLMs, with GraphRAG capabilities for Q&A.

## Features

- **Save input documents** to local files for debugging and reproducibility
- Extract (head, relation, tail) triples from documents using **multiple LLMs** (vLLM and Ollama)
- **Multi-prompt extraction**: Try all prompt files for each document to maximize triple extraction
- **OpenWebUI-compatible message structure**: Prompt sent as "user" role, document content as "assistant" role
- **Parallel execution**: Query multiple LLM servers concurrently for faster processing
- **LLM regeneration**: Automatic retry with temperature adjustments for failed extractions
- **Enhanced timing analysis**: Detailed breakdowns of LLM vs processing time per prompt
- Merge knowledge graphs from different models per document
- **Post-merge entity cleanup**: Standardize entity names by removing unwanted characters and prefixes
- Merge document-level KGs into global knowledge graph
- Deduplicate triples with text mining
- **Entity embedding generation**: Create vector representations of entities using LLMs
- **FAISS index building**: Build and persist vector search index for efficient entity lookup
- **GraphRAG Q&A**: Asynchronous multi-model GraphRAG for knowledge graph question answering
- Noise reduction via graph clustering
- Ontology generation from refined KG
- Consistency checking of KG vs ontology
- **Graph visualization** (PNG/JPG) with automatic rendering

## Setup

1. **Clone or download the repository.**

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the project:**
   - Edit the YAML files in `configs/` directory.
   - Or set environment variables (see `.env.example`).

## Configuration

- `configs/base.yaml`: General settings
- `configs/mongo.yaml`: MongoDB connection and **document processing limits**
- `configs/llm.yaml`: **Multi-model LLM configuration** with parallel execution and regeneration
- `configs/pipeline.yaml`: Pipeline parameters including multi-prompt extraction
- `configs/postprocessing.yaml`: **Post-merge entity cleanup rules** for standardizing entity names

### Document Processing Limits

Configure how many documents to process by default in `configs/mongo.yaml`:

```yaml
uri: "mongodb://carz1.ornl.gov:27017"
database: "knight"
collection: "knight_documents"
limit: 10  # Default limit (can be overridden by CLI --limit or --all)
           # Set to 0 to process ALL documents by default
```

**Options for controlling document processing:**
- **Config file**: Set `limit: 10` (or any number) in `mongo.yaml`
- **CLI limit**: Use `--limit N` to process N documents
- **Process all**: Use `--all` to process every document in the collection
- **Process missing**: Use `--missing` to process only documents that exist in input but haven't been extracted yet
- **No limit by default**: Set `limit: 0` in config to process all documents by default

### Processing Missing Documents

The `--missing` option for the `extract` command automatically identifies and processes only the documents that:
- Exist in the input directory (`output_triple/input/` by default)
- Do not have a corresponding extracted triples file in the docs directory (`output_multiprompt/docs/`)

This is useful for:
- **Incremental processing**: Resume extraction after interruptions
- **Partial failures**: Reprocess documents that failed during initial extraction
- **Adding new documents**: Process newly added documents without reprocessing existing ones

```bash
# Process only documents that haven't been extracted yet
python -m src.kg_builder.cli extract --missing
```

The system compares `.txt` files in the input directory against `.jsonl` files in the docs directory to determine which documents are missing.

### Multi-Model Configuration

The `llm.yaml` now supports multiple models from different providers with parallel execution:

```yaml
models:
  - name: "gpt-oss-120b-vllm"
    base_url: "http://pc0143857.ornl.gov:8000/v1"
    model: "openai/gpt-oss-120b"
    provider: "vllm"
    temperature: 0.2
    max_tokens: 1024

  - name: "llama2-7b-ollama"
    base_url: "http://localhost:11434"
    model: "llama2:7b"
    provider: "ollama"
    temperature: 0.3
    max_tokens: 1024

# Parallel execution settings
parallel_execution:
  enabled: true
  max_concurrent_servers: 3  # Maximum number of servers to query concurrently
  request_timeout: 30        # Timeout for individual requests (seconds)
  retry_attempts: 2          # Number of retry attempts per request
  retry_delay: 1.0           # Delay between retries (seconds)
  enable_llm_json_fix: true  # Enable LLM-based fixing of malformed JSON responses
```

### Multi-Prompt Extraction

The `pipeline.yaml` supports multiple prompt files that are all tried for each document:

```yaml
extraction:
  max_chars: 16000  # Maximum characters to process per document (~8000 tokens)
  prompt_files:
    - "prompts/neuromorphic_prompt.md"  # List of prompt files for triple extraction
    - "prompts/triple_extraction.md"   # All prompts are tried for each document
```

### Regeneration Settings

Handle empty or malformed LLM responses with automatic retries:

```yaml
regeneration:
  enabled: true          # Enable regeneration for empty responses
  max_attempts: 3        # Maximum regeneration attempts (1 original + 2 retries)
  temperature_increment: 0.3  # Increase temperature by this amount each retry
  max_temperature: 1.0   # Maximum temperature to use (prevents going too high)
  delay_between_attempts: 0.5  # Delay between regeneration attempts (seconds)
```

### Post-Merge Entity Cleanup

Clean up entity names and relations after merging to standardize the knowledge graph:

```yaml
enabled: true
cleanup_rules:
  # Remove starting or ending quotes (single or double)
  - name: "remove_quotes"
    pattern: "^[\"']+|['\"]+$"
    replacement: ""
    description: "Remove leading and trailing single or double quotes"

  # Remove starting or ending special characters
  - name: "remove_special_chars_start"
    pattern: "^[$#*:;.,!?@%^&()\\[\\]{}|\\-_=+~`]+"
    replacement: ""
    description: "Remove leading special characters like $, #, *, :, ;, etc."

  - name: "remove_special_chars_end"
    pattern: "[$#*:;.,!?@%^&()\\[\\]{}|\\-_=+~`]+$"
    replacement: ""
    description: "Remove trailing special characters like $, #, *, :, ;, etc."

  # Remove :node_ or :node- substrings
  - name: "remove_node_prefixes"
    pattern: ":node[_-]"
    replacement: ""
    description: "Remove :node_ or :node- substrings from anywhere in the text"

  # Remove "a " (a followed by space) at the beginning
  - name: "remove_a_space_prefix"
    pattern: "^a\\s+"
    replacement: ""
    description: "Remove 'a ' prefix (a followed by space)"

  # Remove "all " (all followed by space) at the beginning
  - name: "remove_all_space_prefix"
    pattern: "^all\\s+"
    replacement: ""
    description: "Remove 'all ' prefix (all followed by space)"

# Apply cleanup to these fields in the merged data
apply_to_fields:
  - "entity_name"        # In entities table
  - "subject_entity_name" # In predications table
  - "object_entity_name" # In predications table
  - "predicate"          # In predications table
  - "h"                  # In JSONL triples (head)
  - "r"                  # In JSONL triples (relation)
  - "t"                  # In JSONL triples (tail)
```

### Entity Embedding and FAISS Index

Configure entity embedding generation and FAISS vector search:

```yaml
embedding:
  model: "llama3.3:70b-instruct-q2_K"  # Model to use for generating entity embeddings
  batch_size: 10     # Number of entities to embed in each batch
  dimension: 4096    # Expected embedding dimension (adjust based on model)

faiss:
  index_path: "output/faiss_index.idx"  # Path to save/load the FAISS index
  index_type: "IndexFlatIP"  # FAISS index type (IndexFlatIP for inner product, IndexFlatL2 for L2)
```

### GraphRAG Configuration

Configure the GraphRAG query engine:

```yaml
graphrag:
  concurrency_limit: 10  # Max concurrent async requests
  k_hops: 2              # Graph traversal depth for subgraph extraction
  max_subgraph_triples: 50  # Maximum triples to include in subgraph
  max_prompt_triples: 20    # Maximum triples to include in prompt
  max_prompt_documents: 5   # Maximum documents to include in prompt
  llm_endpoints:
    extractor: "llama3.3:70b-instruct-q2_K"  # Fast, structural tasks
    ranker: "ibm-granite-13b"               # Classification/Scoring
    reasoner: "gpt-oss-120b"                # Final synthesis
```

Supported providers:
- **vLLM**: OpenAI-compatible API (uses "user" role for prompts, "assistant" role for document content)
- **Ollama**: Local Ollama API

Environment variables can override config values, e.g., `MONGO_URI=mongodb://...`

## Usage

### Processing Document Limits

By default, commands process a limited number of documents (configurable in `configs/mongo.yaml` or via CLI). You can:

- **Process specific number**: Use `--limit N` to process N documents
- **Process ALL documents**: Use `--all` to process every document in the MongoDB collection
- **Process specific file**: Use `--filename` to process a single saved document

### Quick Reference

| Task | Command |
|------|---------|
| Process 20 documents | `python -m src.kg_builder.cli run-all --limit 20` |
| **Process ALL documents** | `python -m src.kg_builder.cli run-all --all` |
| Process document range | `python -m src.kg_builder.cli run-all --begin-idx 0 --end-idx 100` |
| Extract from specific file | `python -m src.kg_builder.cli extract --filename 67c7090f2ce410f1d2136f39.txt` |
| Process document range (extract only) | `python -m src.kg_builder.cli extract --begin-idx 0 --end-idx 100` |
| **Process missing documents** | `python -m src.kg_builder.cli extract --missing` |
| Save all input documents | `python -m src.kg_builder.cli save-inputs --all` |
| Generate entity embeddings | `python -m src.kg_builder.cli embed` |
| Ask questions with GraphRAG | `python -m src.kg_builder.cli ask "What is quantum computing?"` |
| Generate visualization | `python -m src.kg_builder.cli render --format png` |

**Note:** When using `--begin-idx` and `--end-idx` with `run-all`, only the extraction step is restricted to the specified document range. The `save-inputs` step saves all documents, and downstream steps (`merge`, `dedupe`, `cluster`, `ontology`, `check`) process all available extracted data. This enables incremental processing where you can extract different ranges and accumulate results.

### Run Complete Pipeline

```bash
# Process limited number of documents
python -m src.kg_builder.cli run-all --limit 20

# Process ALL documents in the collection
python -m src.kg_builder.cli run-all --all

# Use default limit from config file
python -m src.kg_builder.cli run-all
```

This runs: `save-inputs` → `extract` → `merge` → `postmerge` → `dedupe` → `cluster` → `embed` → `ontology` → `check`

**Note on range processing:** When using `--begin-idx` and `--end-idx` with `run-all`:
- `save-inputs` saves ALL documents (to ensure all inputs are available)
- `extract` processes only the specified document range
- Downstream steps (`merge`, `postmerge`, `dedupe`, etc.) process ALL accumulated extracted data

This design optimizes for incremental processing where extraction is the bottleneck.

### Pipeline Stages Details

The KG Builder pipeline consists of the following stages, each with specific inputs, outputs, and configuration parameters:

#### 1. `save-inputs` - Save Input Documents
- **Purpose**: Downloads and saves raw document content from MongoDB to local files for reproducibility and debugging.
- **Input**: MongoDB collection (configured in `configs/mongo.yaml`)
- **Output**: `output/input/*.txt` (one file per document)
- **Config Parameters** (`configs/mongo.yaml`):
  - `uri`: MongoDB connection string
  - `database`: Database name
  - `collection`: Collection name
  - `limit`: Default document limit (can be overridden by CLI `--limit` or `--all`)
- **CLI Options**: `--limit N`, `--all`, `--begin-idx`, `--end-idx`

#### 2. `extract` - Extract Triples
- **Purpose**: Uses LLMs to extract (head, relation, tail) triples from documents using multiple prompts and models.
- **Input**: `output/input/*.txt` files
- **Output**: 
  - `output/docs/*.jsonl`: Per-document triples with `_model` and `_prompt` fields
  - `output/docs/*_entities.txt`, `*_documents.txt`, `*_predications.txt`: Per-document tables
- **Config Parameters**:
  - `configs/llm.yaml`: Model configurations (multiple models with parallel execution)
  - `configs/pipeline.yaml`: `extraction.max_chars`, `extraction.prompt_files`
  - `configs/mongo.yaml`: Document selection limits
- **CLI Options**: `--limit N`, `--all`, `--filename`, `--missing`, `--begin-idx`, `--end-idx`

#### 3. `merge` - Merge Knowledge Graphs
- **Purpose**: Combines per-document triples into unified global tables with entity deduplication.
- **Input**: `output/docs/*_entities.txt`, `*_documents.txt`, `*_predications.txt`
- **Output**: 
  - `output/merged/merged_entities.txt`, `merged_documents.txt`, `merged_predications.txt`
  - `output/merged/all_triples.jsonl`
- **Config Parameters**: None specific (uses base output directories from `configs/base.yaml`)
- **CLI Options**: None

#### 4. `postmerge` - Post-Merge Entity Cleanup
- **Purpose**: Applies text cleanup rules to standardize entity names and relations.
- **Input**: `output/merged/merged_entities.txt`, `merged_predications.txt`, `all_triples.jsonl`
- **Output**: Updated `output/merged/merged_*.txt` and `all_triples.jsonl` with cleaned text
- **Config Parameters** (`configs/postprocessing.yaml`):
  - `enabled`: Enable/disable postprocessing
  - `cleanup_rules`: List of regex patterns and replacements
  - `apply_to_fields`: Fields to clean (e.g., `entity_name`, `subject_entity_name`, etc.)
- **CLI Options**: None

#### 5. `dedupe` - Deduplicate Triples
- **Purpose**: Removes duplicate triples using text normalization and exact matching.
- **Input**: `output/merged/merged_*.txt`
- **Output**: 
  - `output/merged/deduped/deduped_entities.txt`, `deduped_documents.txt`, `deduped_predications.txt`
  - `output/merged/deduped/deduped.jsonl`
  - `output/merged/duplicates.jsonl` (if tracking enabled)
- **Config Parameters** (`configs/pipeline.yaml`):
  - `dedupe.fuzzy_threshold`: Threshold for fuzzy matching (currently uses exact matching)
  - `dedupe.track_duplicates`: Whether to save duplicate frequency data
- **CLI Options**: None

#### 6. `cluster` - Graph Clustering
- **Purpose**: Applies graph clustering algorithms to reduce noise and identify core knowledge components.
- **Input**: `output/merged/deduped/deduped_*.txt`
- **Output**: 
  - `output/merged/refined_graph.graphml`: Refined graph in GraphML format
  - `output/merged/refined_graph.png/jpg`: Visualizations (if enabled)
  - `output/merged/deleted_entities.jsonl` (if tracking enabled)
- **Config Parameters** (`configs/pipeline.yaml`):
  - `clustering.min_cluster_size`: Minimum cluster size to retain
  - `clustering.enable_visualization`: Generate PNG/JPG visualizations
  - `clustering.track_deleted_entities`: Save list of removed entities
- **CLI Options**: None

#### 7. `embed` - Generate Entity Embeddings
- **Purpose**: Creates vector embeddings for entities and builds FAISS index for efficient similarity search.
- **Input**: `output/merged/refined_graph.graphml` (or deduped entities if graph not available)
- **Output**: 
  - `output/faiss_index.idx`: FAISS vector index
  - `output/faiss_index.mapping.json`: Entity-to-ID mappings
- **Config Parameters** (`configs/pipeline.yaml`):
  - `embedding.model`: LLM model for generating embeddings
  - `embedding.batch_size`: Batch size for embedding generation
  - `embedding.dimension`: Expected embedding dimension
  - `faiss.index_path`: Path to save FAISS index
  - `faiss.index_type`: FAISS index type (`IndexFlatIP` or `IndexFlatL2`)
- **CLI Options**: None

#### 8. `ontology` - Generate Ontology
- **Purpose**: Creates OWL ontology from the refined knowledge graph.
- **Input**: `output/merged/refined_graph.graphml`
- **Output**: `output/ontology/UnifiedKG.owl`
- **Config Parameters** (`configs/pipeline.yaml`):
  - `ontology.name`: Ontology name
  - `ontology.base_iri`: Base IRI for ontology
- **CLI Options**: None

#### 9. `check` - Consistency Check
- **Purpose**: Validates the ontology against the knowledge graph for consistency.
- **Input**: `output/ontology/UnifiedKG.owl`
- **Output**: `output/reports/consistency_report.md`
- **Config Parameters**: None specific
- **CLI Options**: None

### Run Individual Stages

```bash
# Save input documents
python -m src.kg_builder.cli save-inputs --limit 20    # Save 20 documents
python -m src.kg_builder.cli save-inputs --all         # Save ALL documents

# Extract triples
python -m src.kg_builder.cli extract --limit 20        # Extract from 20 documents
python -m src.kg_builder.cli extract --all             # Extract from ALL documents
python -m src.kg_builder.cli extract --filename 67c7090f2ce410f1d2136f39.txt  # Extract from specific file
python -m src.kg_builder.cli extract --missing         # Extract only missing documents

# Process extracted data (no limit needed - processes all extracted files)
python -m src.kg_builder.cli merge
python -m src.kg_builder.cli postmerge
python -m src.kg_builder.cli dedupe
python -m src.kg_builder.cli cluster
python -m src.kg_builder.cli embed     # Generate entity embeddings and FAISS index
python -m src.kg_builder.cli ontology
python -m src.kg_builder.cli check

### Understanding Merge vs Postmerge

The `merge` and `postmerge` commands serve distinct but complementary purposes in the KG Builder pipeline:

**`merge`** - Data Consolidation:
- Combines triples from all extracted document files into unified global tables
- Creates `merged_entities.txt`, `merged_documents.txt`, and `merged_predications.txt` files
- Performs entity deduplication across documents to create consistent global entity IDs
- Generates `all_triples.jsonl` for compatibility with downstream processing
- **Does NOT** modify entity names or relations - preserves raw extracted data

**`postmerge`** - Entity Cleanup:
- Applies configurable text cleanup rules to standardize entity names and relations
- Removes unwanted characters, quotes, prefixes, and special characters from entity names
- Cleans up malformed text that may have been extracted by LLMs
- Processes the same files created by `merge` but applies postprocessing transformations
- **Must be run after `merge`** to ensure clean, standardized entity names

**Why both are needed:**
- `merge` consolidates the raw data structure
- `postmerge` improves data quality by cleaning up noisy entity names
- Running both ensures you have both consolidated AND clean data
- The pipeline (`run-all`) executes both commands in sequence

**Example of postmerge cleanup:**
- Raw entity: `"quantum computing"`
- After postmerge: `quantum computing` (quotes removed)
- Raw entity: `:node_123 artificial intelligence`
- After postmerge: `artificial intelligence` (prefix removed)

# Ask questions using GraphRAG
python -m src.kg_builder.cli ask "What are the applications of neural networks?" --verbose

# Generate visualizations
python -m src.kg_builder.cli render --format png       # Generate graph visualization

### Knowledge Graph Visualization

The `kg_visualization.py` script creates a subgraph visualization from the merged predications file. It extracts the top entities by frequency, builds a NetworkX graph, and generates a PNG visualization suitable for inclusion in research papers.

#### Usage

```bash
# Basic usage with defaults
python src/kg_builder/utils/kg_visualization.py

# Custom input file and output path
python src/kg_builder/utils/kg_visualization.py --input /path/to/merged_predications.txt --output /path/to/visualization.png

# Control subgraph size
python src/kg_builder/utils/kg_visualization.py --max-nodes 50
```

#### Arguments

- `--input`: Path to the merged predications file (default: `output_triple/merged/merged_predications.txt`)
- `--output`: Output path for the PNG visualization (default: `../../../paper/figs/kg-visualization.png`)
- `--max-nodes`: Maximum number of nodes (entities) in the subgraph (default: 100)

#### Features

- **Automatic Subgraph Selection**: Selects top entities by frequency to create a meaningful subgraph
- **Entity Filtering**: Filters out empty, whitespace-only, and non-human-readable entities (requires at least 2 alphanumeric characters)
- **Isolated Node Removal**: Removes disconnected nodes to prevent empty-looking visualizations
- **NetworkX Graph Construction**: Builds directed graph with relations as edge labels
- **Matplotlib Visualization**: Generates publication-quality PNG with spring layout
- **Complete Labeling**: Shows entity names for all connected nodes
- **Configurable Size**: Adjust subgraph size based on your visualization needs

#### Output

- PNG file suitable for inclusion in LaTeX documents (e.g., `\includegraphics{plots/kg-visualization.png}`)
- The visualization shows entity nodes connected by relation edges, with labels for important entities and relations
```

### GraphRAG Q&A

The KG Builder includes a **production-grade Asynchronous Multi-Model GraphRAG system** for intelligent question answering over knowledge graphs. This system optimizes for both cost and performance by routing different tasks to specialized LLMs.

#### Architecture Overview

**Core Components:**
- **`TraceLogger`**: Singleton observability class that logs structured events to daily JSONL files (`logs/trace_YYYYMMDD.jsonl`)
- **`AsyncLLMClient`**: Multi-provider LLM client supporting vLLM and Ollama with async text generation and embedding
- **`AsyncMultiModelGraphRAG`**: Main orchestrator class handling the complete RAG pipeline

**Multi-Model Routing Strategy:**
- **Extractor LLM**: Fast, cost-effective model for entity extraction (e.g., GPT-OSS-20B)
- **Ranker LLM**: Classification/scoring model for relevance assessment (e.g., IBM Granite)
- **Reasoner LLM**: High-capability model for answer synthesis (e.g., GPT-OSS-120B)

#### Prerequisites

Before using GraphRAG queries, ensure you have completed the full KG Builder pipeline:

```bash
# Complete pipeline (recommended)
python -m src.kg_builder.cli run-all --all

# Or run individual steps
python -m src.kg_builder.cli save-inputs --all
python -m src.kg_builder.cli extract --all
python -m src.kg_builder.cli merge
python -m src.kg_builder.cli postmerge
python -m src.kg_builder.cli dedupe
python -m src.kg_builder.cli cluster
python -m src.kg_builder.cli embed  # ⚠️ REQUIRED: Generates FAISS index
```

#### Usage Examples

```bash
# Basic question
python -m src.kg_builder.cli ask "What is quantum computing?"

# Verbose output with detailed logs
python -m src.kg_builder.cli ask "How do neural networks work?" --verbose

# Complex multi-part questions
python -m src.kg_builder.cli ask "What are the differences between supervised and unsupervised learning?"
```

#### GraphRAG Pipeline Details

The system implements a sophisticated 5-stage pipeline:

1. **Entity Extraction**:
   - Uses extractor LLM to identify key named entities from queries
   - Fallback to keyword extraction if model unavailable
   - Logged with latency tracking

2. **Entry Point Discovery**:
   - **Vector Search**: FAISS similarity search using entity embeddings
   - **Keyword Matching**: Exact string matching against graph nodes
   - Combines results for comprehensive coverage

3. **Graph Traversal**:
   - k-hop BFS traversal (configurable depth, default: 2 hops)
   - Collects related triples and associated document IDs
   - NetworkX-based efficient graph operations

4. **Hybrid Reranking**:
   - **Graph Score**: Based on document frequency in traversal results
   - **Semantic Score**: LLM-based relevance assessment (when ranker available)
   - **Final Score**: Weighted combination (50% graph + 50% semantic)
   - Top-K filtering with configurable limits

5. **Answer Generation**:
   - Synthesizes coherent answers using reasoner LLM
   - Context includes ranked triples and document excerpts
   - Structured output with sources and metadata

#### Configuration

Configure GraphRAG in `configs/pipeline.yaml`:

```yaml
graphrag:
  concurrency_limit: 10  # Max concurrent async requests
  k_hops: 2              # Graph traversal depth
  max_subgraph_triples: 50  # Maximum triples to include in subgraph
  max_prompt_triples: 20    # Maximum triples to include in prompt
  max_prompt_documents: 5   # Maximum documents to include in prompt
  llm_endpoints:
    extractor: "gpt-oss-20b-ollama"  # Fast, structural tasks
    ranker: "granite4-small-h-ollama"      # Classification/Scoring
    reasoner: "gpt-oss-120b-ollama"        # Final synthesis
```

Configure models in `configs/llm.yaml`:

```yaml
models:
  - name: "gpt-oss-20b-ollama"
    base_url: "http://carz1.ornl.gov:11434"
    model: "gpt-oss:20b"
    provider: "ollama"
    temperature: 0.2
    timeout: 1200

  - name: "granite4-small-h-ollama"
    base_url: "http://carz2.ornl.gov:11434"
    model: "ibm/granite4:small-h"
    provider: "ollama"
    temperature: 0.2
    timeout: 900

  - name: "gpt-oss-120b-ollama"
    base_url: "http://medz2.ornl.gov:11434"
    model: "gpt-oss:120b"
    provider: "ollama"
    temperature: 0.2
    timeout: 1200
```

#### Features

- **Asynchronous Processing**: Full asyncio implementation with configurable concurrency
- **Multi-Model Optimization**: Cost-effective routing of tasks to appropriate models
- **Structured Observability**: JSONL logging with timestamps, latency, and metadata
- **Robust Error Handling**: Graceful fallbacks when models are unavailable
- **Scalable Architecture**: Batch processing, vector search, and efficient graph operations
- **Intermediate Results**: Query results saved to `output/latest_query_result.json`
- **Production Ready**: Tested with 405K+ nodes and 665K+ edges

#### Output Files

- `output/latest_query_result.json`: Complete query results with metadata
- `logs/trace_YYYYMMDD.jsonl`: Structured logs for each pipeline step
- `output/faiss_index.idx`: FAISS vector index for entity embeddings
- `output/faiss_index.mapping.json`: Entity-to-ID mappings

## LLM Comparison & Evaluation

The `src/kg_builder/utils/compare_llms.py` script allows you to perform A/B comparison between Naive LLM answers and RAG answers (using the existing KG query engine). It generates metrics (Perplexity, Similarity) and visualizations using seaborn for enhanced aesthetics.

### Usage

```bash
# Compare Naive vs RAG for a single query
python -m src.kg_builder.utils.compare_llms --config configs/compare.yaml --query "What is quantum computing?"

# Compare for a list of queries
python -m src.kg_builder.utils.compare_llms --config configs/compare.yaml --queries queries.txt
```

### Configuration

The script uses `configs/compare.yaml` which references `configs/llm.yaml` and `configs/pipeline.yaml`.

```yaml
# LLM Configuration
llms: "configs/llm.yaml"

# GraphRAG Configuration
kg_rag:
  enabled: true
  python_api: false  # Use CLI ask command instead of direct API
  config_dir: "configs"
  cli_path: "src.kg_builder.cli"
  timeout_sec: 180   # Timeout for CLI command execution
  cli_params:
    max_tokens: 2048
    temperature: 0.2

# Evaluator Configuration
evaluator:
  perplexity_model: "gpt2"
  embedding_model: "granite4-small-h-ollama"

# Visualization Configuration
visualization:
  enabled_plots:
    - "agreement_matrix"
    - "perplexity_matrix"
    - "delta_heatmap"

# Default Settings
defaults:
  temperature_naive: 0.2
  max_output_tokens: 2048
  timeout_s: 600
  output_dir: "results"
```

### Features

- **RAG Integration**: Uses the CLI `ask` command for authentic GraphRAG responses
- **5-Model Comparison**: Compares 4 naive LLM models against 1 RAG system
- **Advanced Metrics**: Perplexity, semantic similarity, and performance deltas
- **Seaborn Visualizations**: Professional heatmaps with proper annotations
- **Error Handling**: Graceful handling of visualization rendering issues
- **Comprehensive Output**: CSV results, PNG plots, and JSON summaries

### Outputs

Results are saved in the `results/` directory (configurable):
- `runs_YYYYMMDD_HHMMSS.csv`: Raw results with metrics for all 5 variants (4 naive + 1 RAG)
- `agreement_matrix_*.png`: 5×5 heatmap of pairwise text similarity (seaborn)
- `perplexity_matrix_*.png`: Perplexity scores for all models (seaborn)
- `delta_heatmap_*.png`: RAG improvement over naive models (seaborn)
- `summary_*.json`: Aggregated statistics and performance metrics

### Metrics Explained

- **Perplexity**: Lower values indicate more coherent/fluent text
- **Semantic Similarity**: Cosine similarity between response embeddings
- **Delta Heatmap**: `RAG_perplexity - naive_perplexity` (negative = RAG improved)
- **Agreement Matrix**: Shows how similar responses are across all model variants

### Processing All Documents - Performance Considerations

When using the `--all` option to process every document in your MongoDB collection:

**⚠️ Important Considerations:**
- **Large collections**: Processing thousands of documents can take hours or days
- **Storage space**: Ensure sufficient disk space for output files (JSONL, tables, graphs)
- **Memory usage**: Large collections may require significant RAM
- **LLM costs**: API-based LLMs will incur costs for every document processed
- **Monitoring**: Use detailed logging to track progress and identify issues

**Recommended approach for large collections:**
1. **Test first**: Run with `--limit 10` to verify your configuration works
2. **Incremental processing**: Use `begin_idx` and `end_idx` parameters for batch processing:
   ```bash
   # First, save all input documents (fast operation)
   python -m src.kg_builder.cli save-inputs --all
   
   # Then extract in batches (slow operation)
   python -m src.kg_builder.cli extract --begin-idx 0 --end-idx 100    # Process docs 0-99
   python -m src.kg_builder.cli extract --begin-idx 100 --end-idx 200  # Process docs 100-199
   
   # Or use run-all for convenience (saves all inputs, extracts range, processes all accumulated data)
   python -m src.kg_builder.cli run-all --begin-idx 0 --end-idx 100    # Process docs 0-99
   python -m src.kg_builder.cli run-all --begin-idx 100 --end-idx 200  # Process docs 100-199
   ```
3. **Monitor resources**: Check disk space, memory usage, and processing times
4. **Resume capability**: If processing fails, you can resume from where you left off

## Enhanced Logging and Timing

The system provides detailed timing breakdowns for performance analysis:

```
⏱️  Timing Breakdown for Document 67c7090f2ce410f1d2136f7d:
   📡 LLM Queries: 334.67s
      • neuromorphic_prompt.md: 167.33s (71 triples)
      • triple_extraction.md: 167.34s (72 triples)
   💾 File I/O: 45.23s
      • JSONL save: 12.45s
      • Tables save: 32.78s
   🔄 Other processing: 45.67s
   📊 Total document time: 1348.70s
```

This helps identify bottlenecks in LLM queries, file I/O, and processing operations.

## Log Timing Analysis

The `log_timing_analysis.py` utility provides comprehensive analysis of KG Builder log files for performance monitoring and debugging:

```bash
# Analyze the most recent run (auto-detects document count)
python -m src.kg_builder.utils.log_timing_analysis

# Specify total documents explicitly
python -m src.kg_builder.utils.log_timing_analysis 2242

# Analyze cumulative across all dates
python -m src.kg_builder.utils.log_timing_analysis --cumulative 2242

# Use custom log file
python -m src.kg_builder.utils.log_timing_analysis --log-file /path/to/custom/logfile.log 2242
```

**Analysis Features:**
- **Document Processing Metrics**: Processing times, distributions, and completion estimates
- **Per-Server Performance**: Request counts, success rates, response times, throughput, and failure breakdowns per server
- **Progress Tracking**: Accurate progress calculation accounting for multi-prompt processing
- **Retry & Failure Analysis**: Regeneration attempts, timeouts, malformed responses, and success rates
- **Log Rotation Support**: Automatically detects and reads all log files (main + rotated backups) for complete analysis
- **Three Key Performance Metrics**: Overall Success Rate, Regeneration Rate, and Regeneration Success Rate
- **Dynamic Configuration**: No hardcoded values - automatically detects server mappings and prompt configurations

**Sample Output:**
```
=== KG Builder Timing Analysis Report ===

📊 DOCUMENT PROCESSING ANALYSIS
----------------------------------------
Documents processed: 1670
Total processing time: 2h 34m 12s
Average time per document: 5.5s

Time Distribution:
  Under 1 minute: 234 documents (14.0%)
  1-5 minutes: 1234 documents (73.9%)
  5-10 minutes: 156 documents (9.3%)
  10min-1hr: 46 documents (2.8%)
  Over 1 hour: 0 documents (0.0%)

🎯 PROGRESS & COMPLETION ESTIMATE
----------------------------------------
Configured prompts: 2
Progress: 37.2% complete (1670/4484 document-prompt combinations)
Estimated time remaining: 4h 21m 48s
Remaining combinations: 2814 (docs × prompts)
Estimated completion: 37%

🖥️  SERVER PERFORMANCE ANALYSIS
----------------------------------------

🔹 medz2.ornl.gov:
   Model: gpt-oss-120b-ollama
   Endpoint: http://medz2.ornl.gov:11434
   Requests: 835 / 2242 expected
   Successful: 835
   Failed: 0
   Regeneration Rate: 15.2%

🔹 carz3.ornl.gov:
   Model: llama3.3-quantized-ollama
   Endpoint: http://carz3.ornl.gov:11434
   Requests: 835 / 2242 expected
   Successful: 835
   Failed: 0
   Regeneration Rate: 12.8%

🔹 carz1.ornl.gov:
   Model: gpt-oss-20b-ollama
   Endpoint: http://carz1.ornl.gov:11434
   Requests: 835 / 2242 expected
   Successful: 835
   Failed: 0
   Regeneration Rate: 18.4%
   Failure breakdown:
     • Regeneration attempts: 153
       → Estimated regeneration rate: 18.4% (153/835 queries)
       → 142 successful regenerations
       → 1.1 attempts needed per successful regeneration

📊 OVERALL PERFORMANCE METRICS
----------------------------------------
Overall Success Rate: 100.0%
  = (2505 successful responses / 2505 total queries) × 100
Regeneration Rate: 15.5%
  = (389 queries with regeneration / 2505 total queries) × 100
Regeneration Success Rate: 97.2%
  = (378 queries succeeded after regeneration / 389 queries with regeneration) × 100

🔄 RETRY & FAILURE ANALYSIS
----------------------------------------
Regeneration attempts: 389
Regeneration successes: 378
Timeouts: 0
Malformed response failures: 0
Regeneration success rate: 97.2%
Note: Regeneration occurs when initial LLM responses are malformed or empty.
      Each regeneration attempt tries different parameters (temperature, max_tokens).
      Success rate = (successful regenerations / total regeneration attempts) × 100

Definitions:
  Overall Success Rate = (successful responses / total queries) × 100
  Regeneration Rate = (queries with regeneration / total queries) × 100
  Regeneration Success Rate = (queries that ultimately succeeded after regeneration / queries that had regeneration) × 100
```

**Arguments:**
- `total_docs`: Total number of documents expected (auto-detected if not provided)
- `--log-file`: Path to the log file (default: `logs_multiprompt/kg_builder.log`)
- `--cumulative`: Analyze cumulative across all dates (default: analyze only most recent run)

**Key Features:**
- **Complete Log Analysis**: Reads all available log files including rotated backups (kg_builder.log, kg_builder.log.1, etc.) to ensure no data is missed
- **Per-Server Metrics**: Detailed performance breakdown for each LLM server with request counts, success rates, and failure analysis
- **Three Performance Metrics**: 
  - *Overall Success Rate*: Percentage of successful LLM responses
  - *Regeneration Rate*: Percentage of queries that required regeneration
  - *Regeneration Success Rate*: Success rate of regeneration attempts
- **Dynamic Detection**: Automatically loads server mappings from `configs/llm.yaml` and prompt configurations from `configs/pipeline.yaml`
- **Progress Estimation**: Accurate progress tracking that accounts for multi-prompt processing

## Outputs

- `output/input/*.txt`: Raw input documents (**save-inputs** stage)
- `output/docs/*.jsonl`: Per-document triples with `_model` and `_prompt` fields (**extract** stage)
- `output/merged/merged_*.txt`: Cleaned merged tables (entities, documents, predications) (**merge** + **postmerge** stages)
- `output/merged/all_triples.jsonl`: Cleaned merged triples in JSONL format (**merge** + **postmerge** stages)
- `output/merged/deduped.*`: Deduplicated triples and entities (**dedupe** stage)
- `output/faiss_index.idx`: FAISS vector index for entity embeddings (**embed** stage)
- `output/faiss_index.mapping.json`: Entity index to entity name mappings for FAISS (**embed** stage)
- `output/latest_query_result.json`: Latest GraphRAG query results (**ask** command)
- `output/merged/refined_graph.graphml`: Refined graph after clustering (**cluster** stage)
- `output/merged/refined_graph.png`: Graph visualization (PNG) (**cluster** stage, if visualization enabled)
- `output/merged/refined_graph.jpg`: Graph visualization (JPG) (**cluster** stage, if visualization enabled)
- `output/ontology/UnifiedKG.owl`: Generated OWL ontology (**ontology** stage)
- `output/reports/consistency_report.md`: Ontology consistency check report (**check** stage)
- `results/runs_*.csv`: LLM comparison results (**compare_llms.py** utility)
- `results/*_matrix_*.png`: Comparison heatmaps (**compare_llms.py** utility)
- `results/summary_*.json`: Comparison statistics (**compare_llms.py** utility)
- `logs/trace_YYYYMMDD.jsonl`: Daily structured logs for pipeline steps (all stages)

## Troubleshooting

- **MongoDB connection fails:** Check `MONGO_URI` and ensure MongoDB is running.
- **LLM request fails:** Verify `LLM_BASE_URL` and model availability.
- **Missing dependencies:** Ensure all packages from `requirements.txt` are installed.
- **Ontology reasoner fails:** Install HermiT reasoner if needed (owlready2 may require it).
- **No triples extracted:** Check LLM response format; ensure JSON lines are valid.
- **Graph visualization fails:** Ensure matplotlib is installed (`pip install matplotlib`).
- **compare_llms.py visualization errors:** The script handles matplotlib LaTeX rendering issues gracefully - plots may fail to generate but CSV results will still be saved.
- **RAG not working in compare_llms.py:** Ensure GraphRAG is properly set up with FAISS index and the CLI ask command works independently.

## Requirements

- Python 3.10+
- MongoDB
- vLLM or Ollama servers for LLM inference
- FAISS (installed via `faiss-cpu`)
- NetworkX for graph operations
- matplotlib (for graph visualization)
- seaborn (for enhanced heatmaps in compare_llms.py)
- aiohttp (for async HTTP requests)
