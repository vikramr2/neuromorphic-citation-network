import typer
from pathlib import Path
from .config import load_config
from .logging_utils import setup_logging
from .extraction import extract_triples, save_input_documents
from .merge import merge_triples
from .dedupe import dedupe as dedupe_func
from .clustering import clustering, render_graphml_to_image
from .ontology import generate_ontology
from .consistency import consistency_check

app = typer.Typer()

# Global config directory option
config_dir_option = typer.Option("configs", "--config-dir", help="Directory containing configuration files")

@app.callback()
def callback(config_dir: str = config_dir_option):
    """
    KG Builder CLI - Build knowledge graphs from MongoDB publications using LLMs.
    """
    # Store the config directory globally for use in commands
    app.config_dir = config_dir

@app.command()
def save_inputs(limit: int = 20, all: bool = False, begin_idx: int = 0, end_idx: int = None):
    """
    Save input documents from MongoDB to local files for processing.

    Args:
        limit: Maximum number of documents to save (ignored if --all is used)
        all: Process all documents in the collection (overrides limit)
        begin_idx: Starting document index (0-based) for range processing
        end_idx: Ending document index (exclusive) for range processing
    """
    config = load_config(app.config_dir)
    # Override config limit with command line parameter
    if all:
        config.mongo.limit = 0  # 0 means no limit (process all)
        typer.echo("Processing ALL documents in the collection...")
    else:
        config.mongo.limit = limit
    # Adjust limit to ensure enough documents are fetched for the range
    if end_idx is not None and not all:
        config.mongo.limit = max(config.mongo.limit, end_idx)
    setup_logging(Path(config.base.logs_dir), "INFO")

    input_dir = Path(config.base.output_dir) / "input"
    save_input_documents(config, input_dir, begin_idx, end_idx)

@app.command()
def extract(limit: int = 20, filename: str = None, begin_idx: int = 0, end_idx: int = None, all: bool = False, missing: bool = False):
    """
    Extract triples from documents.

    Args:
        limit: Maximum number of documents to process (ignored if filename is provided or --all is used)
        filename: Specific input file to process (e.g., '67c7090f2ce410f1d2136f39.txt')
        begin_idx: Starting document index (0-based) for range processing
        end_idx: Ending document index (exclusive) for range processing
        all: Process all documents in the collection (overrides limit)
        missing: Process only documents that are in input_dir but not in docs_dir
    """
    config = load_config(app.config_dir)
    # Override config limit with command line parameter
    if all:
        config.mongo.limit = 0  # 0 means no limit (process all)
        typer.echo("Processing ALL documents in the collection...")
    else:
        config.mongo.limit = limit
    # Adjust limit to ensure enough documents are fetched for the range
    if end_idx is not None and not all:
        config.mongo.limit = max(config.mongo.limit, end_idx)
    setup_logging(Path(config.base.logs_dir), "INFO")
    docs_dir = Path(config.base.output_dir) / "docs"
    
    if missing:
        # Determine input directory
        if config.base.merge_output_dirs:
            input_dir = Path(config.base.project_dir) / config.base.merge_output_dirs[0] / "input"
        else:
            input_dir = Path(config.base.output_dir) / "input"
        
        if not input_dir.exists():
            typer.echo(f"❌ Input directory not found: {input_dir}")
            return
        
        # Get all input files
        input_files = set(f.stem for f in input_dir.glob("*.txt"))
        # Get all processed files
        processed_files = set(f.stem for f in docs_dir.glob("*.jsonl"))
        # Find missing
        missing_files = input_files - processed_files
        
        if not missing_files:
            typer.echo("✅ All documents have been processed.")
            return
        
        typer.echo(f"🔍 Found {len(missing_files)} missing documents to process.")
        
        # Process each missing file
        for missing_id in sorted(missing_files):
            filename = f"{missing_id}.txt"
            typer.echo(f"📄 Processing missing document: {filename}")
            extract_triples(config, docs_dir, filename)
    else:
        extract_triples(config, docs_dir, filename, begin_idx, end_idx)

@app.command()
def merge(input_dirs: list[str] = None):
    """
    Merge triples from multiple document directories.
    
    Args:
        input_dirs: List of input directories containing docs folders (e.g., 'output_triple output_multiprompt').
                   If not provided, uses merge_output_dirs from config.
    """
    config = load_config(app.config_dir)
    setup_logging(Path(config.base.logs_dir), "INFO")
    
    # Determine input directories
    if input_dirs is None or len(input_dirs) == 0:
        # Use configured output directories
        if config.base.merge_output_dirs:
            input_dirs = [Path(config.base.project_dir) / output_dir for output_dir in config.base.merge_output_dirs]
        else:
            # Fallback to single output_dir
            input_dirs = [Path(config.base.project_dir) / config.base.output_dir]
    else:
        # Convert string paths to Path objects
        input_dirs = [Path(d) for d in input_dirs]
    
    # Convert to docs directories
    docs_dirs = [input_dir / "docs" for input_dir in input_dirs]
    
    merged_dir = Path(config.base.project_dir) / config.base.output_dir / "merged"
    merge_triples(docs_dirs, merged_dir)

@app.command()
def postmerge():
    """
    Clean up entity names and relations in merged data using postprocessing rules.
    
    This step standardizes entity names by removing unwanted characters, quotes, 
    and prefixes according to the rules defined in configs/postprocessing.yaml.
    """
    config = load_config(app.config_dir)
    setup_logging(Path(config.base.logs_dir), "INFO")
    
    merged_dir = Path(config.base.project_dir) / config.base.output_dir / "merged"
    
    # Import here to avoid circular imports
    from .postmerging import run_postmerging
    
    results = run_postmerging(config, merged_dir)
    if results:
        typer.echo(f"✅ Postprocessing completed: {sum(results.values())} total records processed")
        for key, count in results.items():
            typer.echo(f"  • {key}: {count} records")
    else:
        typer.echo("ℹ️  Postprocessing is disabled or no data to process")

@app.command()
def dedupe():
    config = load_config(app.config_dir)
    setup_logging(Path(config.base.logs_dir), "INFO")
    
    merged_dir = Path(config.base.project_dir) / config.base.output_dir / "merged"
    deduped_path = merged_dir / "deduped"
    dedupe_func(merged_dir, deduped_path, config.pipeline.dedupe['fuzzy_threshold'], config.pipeline.dedupe.get('track_duplicates', True))

@app.command()
def cluster():
    config = load_config(app.config_dir)
    setup_logging(Path(config.base.logs_dir), "INFO")
    
    merged_dir = Path(config.base.project_dir) / config.base.output_dir / "merged"
    refined_path = merged_dir / "refined_graph.graphml"
    clustering(merged_dir, refined_path, config.pipeline.clustering['min_cluster_size'], 
              config.pipeline.clustering.get('enable_visualization', False),
              config.pipeline.clustering.get('track_deleted_entities', True))

@app.command()
def render(graphml_path: str = None, format: str = "png"):
    """
    Render a GraphML file as an image.

    Args:
        graphml_path: Path to the GraphML file (default: output/merged/refined_graph.graphml)
        format: Image format (png or jpg)
    """
    config = load_config(app.config_dir)
    setup_logging(Path(config.base.logs_dir), "INFO")

    if graphml_path is None:
        merged_dir = Path(config.base.output_dir) / "merged"
        graphml_path = merged_dir / "refined_graph.graphml"
    else:
        graphml_path = Path(graphml_path)

    image_path = graphml_path.with_suffix(f'.{format}')
    title = f"Knowledge Graph from {graphml_path.name}"

    render_graphml_to_image(graphml_path, image_path, title)

@app.command()
def ontology():
    config = load_config(app.config_dir)
    setup_logging(Path(config.base.logs_dir), "INFO")
    
    base_dir = Path(config.base.project_dir) / config.base.output_dir
    merged_dir = base_dir / "merged"
    ontology_path = base_dir / "ontology" / "UnifiedKG.owl"
    generate_ontology(merged_dir, ontology_path, config.pipeline.ontology['name'], config.pipeline.ontology['base_iri'])

@app.command()
def check():
    config = load_config(app.config_dir)
    setup_logging(Path(config.base.logs_dir), "INFO")

    base_dir = Path(config.base.project_dir) / config.base.output_dir
    ontology_path = base_dir / "ontology" / "UnifiedKG.owl"
    report_path = base_dir / "reports" / "consistency_report.md"
    consistency_check(ontology_path, report_path)

@app.command()
def embed():
    """
    Generate entity embeddings and build FAISS index for GraphRAG.
    
    This step should be run after dedupe and cluster to create vector representations
    of all entities in the refined knowledge graph for efficient similarity search.
    """
    config = load_config(app.config_dir)
    setup_logging(Path(config.base.logs_dir), "INFO")
    
    # Import and run GraphRAG ingestion
    from .graphrag import AsyncMultiModelGraphRAG
    import asyncio
    
    async def run_ingest():
        graphrag = AsyncMultiModelGraphRAG(config)
        await graphrag.ingest_and_embed()
    
    asyncio.run(run_ingest())
    typer.echo("✅ Entity embedding and FAISS index creation completed")

@app.command()
def ask(
    query: str, 
    verbose: bool = False,
    max_tokens: int = None,
    temperature: float = None,
    model: str = None
):
    """
    Ask a question using GraphRAG.
    
    Args:
        query: The question to ask about the knowledge graph
        verbose: Print detailed logs to stdout
        max_tokens: Maximum tokens in the response (overrides config)
        temperature: Sampling temperature (overrides config)
        model: Override the reasoner model to use
    """
    config = load_config(app.config_dir)
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(Path(config.base.logs_dir), log_level)
    
    # Override config with CLI parameters
    if max_tokens is not None:
        config.pipeline.graphrag['max_tokens'] = max_tokens
    
    if temperature is not None:
        config.pipeline.graphrag['temperature'] = temperature
        
    if model is not None:
        if 'llm_endpoints' not in config.pipeline.graphrag:
            config.pipeline.graphrag['llm_endpoints'] = {}
        config.pipeline.graphrag['llm_endpoints']['reasoner'] = model
    
    # Import and run GraphRAG query
    from .graphrag import AsyncMultiModelGraphRAG
    import asyncio
    import json
    
    async def run_query():
        graphrag = AsyncMultiModelGraphRAG(config)
        result = await graphrag.process_query(query)
        
        # Print answer
        typer.echo(f"\n🤖 Answer: {result['answer']}\n")
        
        if result['sources']:
            typer.echo("📚 Sources:")
            for i, source in enumerate(result['sources'], 1):
                typer.echo(f"  {i}. {source}")
            typer.echo()
        
        if verbose:
            typer.echo("📊 Details:")
            typer.echo(f"  Entities found: {len(result['entities'])}")
            typer.echo(f"  Entry nodes: {len(result['entry_nodes'])}")
            typer.echo(f"  Subgraph triples: {len(result['subgraph_triples'])}")
            typer.echo(f"  Top triples: {len(result['top_triples'])}")
            typer.echo(f"  Top documents: {len(result['top_docs'])}")
            
            typer.echo("\n📝 Prompt Text:")
            typer.echo("-" * 40)
            typer.echo(result['prompt'])
            typer.echo("-" * 40)
            
            typer.echo("\n🕸️ Subgraph Triples (Top 5):")
            for t in result['subgraph_triples'][:5]:
                typer.echo(f"  {t['h']} -> {t['r']} -> {t['t']}")
                
            typer.echo("\n📄 Top Documents:")
            for d in result['top_docs']:
                typer.echo(f"  {d}")
    
    asyncio.run(run_query())


@app.command()
def run_all(limit: int = 20, all: bool = False, begin_idx: int = 0, end_idx: int = None):
    """
    Run the complete KG Builder pipeline.

    Args:
        limit: Maximum number of documents to process (ignored if --all is used)
        all: Process all documents in the collection (overrides limit)
        begin_idx: Starting document index (0-based) for range processing
        end_idx: Ending document index (exclusive) for range processing
    """
    # Use subprocess to call the commands programmatically
    import subprocess
    import sys

    config_dir = getattr(app, 'config_dir', 'configs')
    
    # Determine the limit/all arguments to pass to subcommands
    if all:
        limit_args = ["--all"]
        typer.echo("Running complete pipeline for ALL documents in the collection...")
    else:
        limit_args = ["--limit", str(limit)]
        typer.echo(f"Running complete pipeline for up to {limit} documents...")
    
    # Adjust limit for range if specified
    if end_idx is not None and not all:
        limit_args = ["--limit", str(max(limit, end_idx))]
    
    # Build range arguments
    range_args = []
    if begin_idx > 0 or end_idx is not None:
        range_args = ["--begin-idx", str(begin_idx)]
        if end_idx is not None:
            range_args.extend(["--end-idx", str(end_idx)])
    
    commands = [
        [sys.executable, "-m", "src.kg_builder.cli", "--config-dir", config_dir, "save-inputs"] + limit_args,
        [sys.executable, "-m", "src.kg_builder.cli", "--config-dir", config_dir, "extract"] + limit_args + range_args,
        [sys.executable, "-m", "src.kg_builder.cli", "--config-dir", config_dir, "merge"],
        [sys.executable, "-m", "src.kg_builder.cli", "--config-dir", config_dir, "postmerge"],
        [sys.executable, "-m", "src.kg_builder.cli", "--config-dir", config_dir, "dedupe"],
        [sys.executable, "-m", "src.kg_builder.cli", "--config-dir", config_dir, "cluster"],
        [sys.executable, "-m", "src.kg_builder.cli", "--config-dir", config_dir, "embed"],
        [sys.executable, "-m", "src.kg_builder.cli", "--config-dir", config_dir, "ontology"],
        [sys.executable, "-m", "src.kg_builder.cli", "--config-dir", config_dir, "check"]
    ]

    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            sys.exit(result.returncode)

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    app()
