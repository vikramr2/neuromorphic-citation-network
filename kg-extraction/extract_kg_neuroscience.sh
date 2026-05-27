#!/usr/bin/env bash
# Extract KG for neuroscience papers.
# Run from: kg-extraction/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
KG_BUILDER="$SCRIPT_DIR/kg-builder"
INPUT_JSON="$REPO_ROOT/data/neuroscience/neuroscience_papers_marker.json"
INPUT_TXT="$REPO_ROOT/data/kg/kg_neuroscience/input"

echo "=== Neuroscience KG Extraction ==="

echo "[1/4] Converting marker JSON to txt files..."
mkdir -p "$INPUT_TXT"
python "$SCRIPT_DIR/json_to_txt.py" "$INPUT_JSON" "$INPUT_TXT"

echo "[2/4] Extracting triples (this will take a while)..."
cd "$KG_BUILDER"
python -m src.kg_builder.cli --config-dir configs_neuroscience extract --missing

echo "[3/4] Merging triples..."
python -m src.kg_builder.cli --config-dir configs_neuroscience merge

echo "[4/4] Deduplicating..."
python -m src.kg_builder.cli --config-dir configs_neuroscience dedupe

echo "=== Done. Output: $REPO_ROOT/data/kg/kg_neuroscience/ ==="
