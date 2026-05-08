#!/usr/bin/env python3
"""
run_neukrag_dspy.py  —  NeuKRAG KG extraction via DSPy + Claude

Usage:
    python run_neukrag_dspy.py <input_path> [--output-dir DIR] [--mode MODE]

Arguments:
    input_path      .txt file or directory of .txt files
    --output-dir    where to write JSONL triple files
                    (default: output_triple/docs)
    --mode          "triples"      — simple (h, r, t) JSONL (default)
                    "neuromorphic" — rich JSON graph with ontology types

The JSONL output is compatible with the existing kg-builder pipeline:
    python -m src.kg_builder.cli merge
    python -m src.kg_builder.cli postmerge
    ... etc.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import dspy
from pydantic import BaseModel, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MAX_CHARS = 16_000
LLM_MODEL       = "openai/openai/gpt-oss-120b"
OLLAMA_BASE_URL = "http://earlsinclair.ornl.gov:8200/v1"
OLLAMA_API_KEY  = "vllm"

# ---------------------------------------------------------------------------
# Pydantic schemas for structured output
# ---------------------------------------------------------------------------

class Triple(BaseModel):
    h: str
    r: str
    t: str

    @field_validator("h", "r", "t")
    @classmethod
    def non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("field must not be empty")
        return v


class KGNode(BaseModel):
    id: str
    name: str
    type: str
    topic: list[str] = []
    evidence_span: str = ""
    confidence: float = 0.8
    polarity: str = "proposes"
    novelty_tag: str = "method"


class KGEdge(BaseModel):
    source: str
    relation: str
    target: str
    evidence_span: str = ""
    confidence: float = 0.8


class KGGraph(BaseModel):
    paper_id: str
    nodes: list[KGNode] = []
    edges: list[KGEdge] = []


# ---------------------------------------------------------------------------
# DSPy signatures
# ---------------------------------------------------------------------------

TRIPLE_PROMPT = (Path(__file__).parent / "approach1/kg-builder/src/kg_builder/prompts/triple_extraction.md").read_text()
NEURO_PROMPT  = (Path(__file__).parent / "approach1/kg-builder/src/kg_builder/prompts/neuromorphic_prompt.md").read_text()


class ExtractTriples(dspy.Signature):
    """Extract factual RDF triples from a scientific text as JSONL.

    Each output line must be valid JSON: {"h": "Head", "r": "relation", "t": "Tail"}
    Use Title Case for entities, snake_case for relations.
    Output ONLY JSON lines — no prose, no code fences.
    """
    extraction_rules: str = dspy.InputField(desc="Extraction rules and examples")
    paper_text: str       = dspy.InputField(desc="Scientific text to process")
    triples_jsonl: str    = dspy.OutputField(desc="JSONL lines, one triple per line")


class ExtractNeuromorphicKG(dspy.Signature):
    """Extract a rich knowledge graph from a neuromorphic computing paper.

    Follow the ontology (node types, relations, topics) provided in the prompt.
    Output a single valid JSON object matching:
      { "paper_id": "...", "nodes": [...], "edges": [...] }
    Output ONLY the JSON object — no prose, no code fences.
    """
    ontology_prompt: str = dspy.InputField(desc="Ontology definitions, relations, and output schema")
    paper_text: str      = dspy.InputField(desc="Full paper text (truncated to fit context)")
    paper_id: str        = dspy.InputField(desc="Short identifier for this paper")
    kg_json: str         = dspy.OutputField(desc="Valid JSON object with paper_id, nodes, edges")


# ---------------------------------------------------------------------------
# DSPy modules
# ---------------------------------------------------------------------------

class TripleExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ExtractTriples)

    def forward(self, paper_text: str) -> list[dict]:
        result = self.predict(
            extraction_rules=TRIPLE_PROMPT,
            paper_text=paper_text[:MAX_CHARS],
        )
        return _parse_jsonl_triples(result.triples_jsonl)


class NeuromorphicKGExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ExtractNeuromorphicKG)

    def forward(self, paper_text: str, paper_id: str) -> tuple[KGGraph, list[dict]]:
        result = self.predict(
            ontology_prompt=NEURO_PROMPT,
            paper_text=paper_text[:MAX_CHARS],
            paper_id=paper_id,
        )
        graph = _parse_kg_json(result.kg_json, paper_id)
        triples = _graph_to_triples(graph)
        return graph, triples


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove markdown code fences that Claude sometimes adds."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text.strip())
    return text.strip()


def _parse_jsonl_triples(raw: str) -> list[dict]:
    triples = []
    for line in _strip_fences(raw).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if all(k in obj for k in ("h", "r", "t")):
                Triple(**obj)  # validate
                triples.append({"h": obj["h"], "r": obj["r"], "t": obj["t"]})
        except Exception:
            pass  # skip malformed lines
    return triples


def _parse_kg_json(raw: str, paper_id: str) -> KGGraph:
    raw = _strip_fences(raw)
    # Sometimes the LM wraps the object in an array; unwrap it
    if raw.startswith("["):
        raw = raw[1:raw.rfind("]")]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning(f"JSON parse error for {paper_id}: {exc}. Returning empty graph.")
        return KGGraph(paper_id=paper_id)

    if not isinstance(data, dict):
        return KGGraph(paper_id=paper_id)

    nodes = [KGNode(**n) for n in data.get("nodes", []) if isinstance(n, dict)]
    edges = [KGEdge(**e) for e in data.get("edges", []) if isinstance(e, dict)]
    return KGGraph(paper_id=data.get("paper_id", paper_id), nodes=nodes, edges=edges)


def _graph_to_triples(graph: KGGraph) -> list[dict]:
    """Convert KGGraph edges to simple (h, r, t) triples using node names."""
    id_to_name = {n.id: n.name for n in graph.nodes}
    triples = []
    for edge in graph.edges:
        h = id_to_name.get(edge.source, edge.source)
        t = id_to_name.get(edge.target, edge.target)
        triples.append({"h": h, "r": edge.relation, "t": t})
    return triples


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def iter_input_files(input_path: Path):
    if input_path.is_file():
        yield input_path
    elif input_path.is_dir():
        for f in sorted(input_path.glob("*.txt")):
            yield f
    else:
        sys.exit(f"ERROR: {input_path} is not a file or directory")


def save_jsonl(triples: list[dict], out_path: Path, model_name: str, prompt_name: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    for t in triples:
        row = {**t, "_model": model_name, "_prompt": prompt_name}
        rows.append(row)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_kg_json(graph: KGGraph, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph.model_dump(), f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NeuKRAG extraction via DSPy + Claude")
    parser.add_argument("input_path", type=Path, help=".txt file or directory of .txt files")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output_triple/docs"),
        help="directory for JSONL output (default: output_triple/docs)"
    )
    parser.add_argument(
        "--mode", choices=["triples", "neuromorphic"], default="triples",
        help="extraction mode: 'triples' (simple JSONL) or 'neuromorphic' (rich JSON graph)"
    )
    args = parser.parse_args()

    # Configure DSPy with the vLLM-served model
    lm = dspy.LM(
        LLM_MODEL,
        api_base=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY,
    )
    dspy.configure(lm=lm)
    log.info(f"DSPy configured with {LLM_MODEL} @ {OLLAMA_BASE_URL}")

    if args.mode == "triples":
        extractor = TripleExtractor()
        prompt_name = "triple_extraction.md"
        model_name  = LLM_MODEL
    else:
        extractor = NeuromorphicKGExtractor()
        prompt_name = "neuromorphic_prompt.md"
        model_name  = LLM_MODEL

    input_files = list(iter_input_files(args.input_path))
    log.info(f"Found {len(input_files)} input file(s)")

    for txt_path in input_files:
        paper_id = txt_path.stem
        text = txt_path.read_text(encoding="utf-8")
        out_jsonl = args.output_dir / f"{paper_id}.jsonl"

        log.info(f"Processing {paper_id} ({len(text):,} chars) ...")

        try:
            if args.mode == "triples":
                triples = extractor(paper_text=text)
            else:
                graph, triples = extractor(paper_text=text, paper_id=paper_id)
                kg_json_path = args.output_dir / f"{paper_id}_kg.json"
                save_kg_json(graph, kg_json_path)
                log.info(f"  Saved rich KG → {kg_json_path} "
                         f"({len(graph.nodes)} nodes, {len(graph.edges)} edges)")

            save_jsonl(triples, out_jsonl, model_name, prompt_name)
            log.info(f"  Saved {len(triples)} triples → {out_jsonl}")

        except Exception as exc:
            log.error(f"  Failed for {paper_id}: {exc}")

    log.info("Done.")
    log.info("Next step: cd kg-builder && python -m src.kg_builder.cli merge")


if __name__ == "__main__":
    main()
