# Tests Directory

This directory contains various test files and utilities for the KG Builder project.

## Test Files

### Unit Tests

#### `test_schemas.py`
Tests for data schema validation and creation.

**Tests:**
- `test_triple_creation()`: Validates Triple object creation with valid data
- `test_triple_validation()`: Ensures proper validation of Triple objects (rejects empty head entities)

**Run:**
```bash
pytest tests/test_schemas.py
```

#### `test_utils.py`
Tests for utility functions.

**Tests:**
- `test_normalize_text()`: Tests text normalization function with various inputs (case conversion, accent removal, whitespace handling)

**Run:**
```bash
pytest tests/test_utils.py
```

### Integration Tests

#### `test_ollama.py`
Comprehensive testing harness for Ollama API endpoints. Tests both HTTP API and Python SDK interfaces.

**Features:**
- Tests `/api/generate` and `/api/chat` endpoints
- Compares HTTP API vs Python SDK performance
- Supports multiple models, prompts, and documents
- Configurable test iterations with timing measurements

**Usage:**
```bash
# Test all models with all prompts and documents
python tests/test_ollama.py --models model1 model2 --prompt-dir prompts/ --text-dir documents/

# Test single model with specific files
python tests/test_ollama.py --model llama2 --prompt-file prompts/extraction.md --text-file docs/sample.txt

# Test only HTTP API, generate endpoint, 5 times each
python tests/test_ollama.py --models llama2 --prompt-file prompts/extraction.md --text-file docs/sample.txt --api http --endpoint generate --times 5
```

**Arguments:**
- `--prompt-file` / `--prompt-dir`: Single prompt file or directory of .md files
- `--text-file` / `--text-dir`: Single text file or directory of .txt files
- `--model` / `--models`: Single model or list of models to test
- `--api`: API to test (`http`, `python`, or `all`)
- `--endpoint`: Endpoint to test (`generate`, `chat`, or `all`)
- `--times`: Number of test iterations (default: 3)
- `--host` / `--port`: Ollama server connection details

## Utilities

### `log_timing_analysis.py` (moved to `src/kg_builder/utils/`)
Timing and performance analysis script for KG Builder log files.

**Note:** This utility has been moved to `src/kg_builder/utils/log_timing_analysis.py` for better organization. See the main README.md for usage instructions.

## Running All Tests

To run all unit tests:
```bash
pytest tests/
```

To run tests with coverage:
```bash
pytest --cov=kg_builder tests/
```

## Dependencies

Tests require the following dependencies (install with Poetry):
```bash
poetry install --with dev
```

Key testing dependencies:
- `pytest`: Test framework
- `requests`: HTTP API testing (for Ollama tests)
- `ollama`: Python SDK for Ollama (optional, falls back to HTTP if not available)

## Notes

- The `timing_analysis.py` script is a utility tool, not a traditional unit test
- Ollama tests require a running Ollama server (configured via `--host` and `--port`)
- Server mappings for timing analysis are loaded from `configs/llm.yaml`
- All tests assume the working directory is the `kg-builder/` root directory</content>
<parameter name="filePath">/scratch/ramki/knight/approach1/kg-builder/tests/README.md