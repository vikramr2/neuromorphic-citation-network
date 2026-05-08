#!/usr/bin/env python3
"""
Pre-build the code index for NeuroGraphStore.

Walks all GitHub repos in data/github_repos/{pytorch,hdl,fugu,superneuro}/,
extracts .py, .v, .vhd, .sv source files, and writes
data/github_repos/code_index.jsonl — one JSON line per chunk.

Python files are split into logical chunks using the ast module:
  - Small files (≤ MAX_CHUNK_LINES):  one chunk, chunk_name="<module>"
  - Larger files:  one chunk per top-level class/function
  - Large classes: one chunk for the class header + one chunk per method

HDL files (Verilog/VHDL) are split into MAX_CHUNK_LINES-line windows.

code_id format: "{repo_type}/{full_name}/{rel_path}::{chunk_name}"

Run from repo root:
    python dataset-assembly/scripts/code/index_code_files.py

Output: data/github_repos/code_index.jsonl
"""

import ast
import json
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3].parent
DATA_DIR  = REPO_ROOT / "data"
REPOS_DIR = DATA_DIR / "github_repos"
REPOS_JSON = REPOS_DIR / "neuromorphic_repos.json"
OUTPUT     = REPOS_DIR / "code_index.jsonl"

EXTENSIONS = {
    ".py":   "python",
    ".v":    "verilog",
    ".sv":   "verilog",
    ".vhd":  "vhdl",
    ".vhdl": "vhdl",
}

SKIP_PATH_SEGMENTS = {
    "__pycache__", ".git", "node_modules", "vendor",
    "build", "dist", ".egg-info", ".tox", ".mypy_cache",
    "site-packages", "migrations",
}

SKIP_NAME_PREFIXES = ("conftest", "setup")
SKIP_NAME_SUFFIXES = ("_test.py", "_tests.py")

MAX_CHUNK_LINES = 300    # max lines in any single emitted chunk
MAX_CHUNK_BYTES = 51200  # 50 KB per chunk — skip if exceeded
MAX_FILE_BYTES  = 1_048_576  # 1 MB — skip whole file above this


# ── Chunking helpers ────────────────────────────────────────────────────────────

def _trim(lines: list[str], label: str = "") -> str:
    """Join lines, truncating to MAX_CHUNK_LINES with a note."""
    if len(lines) > MAX_CHUNK_LINES:
        return "\n".join(lines[:MAX_CHUNK_LINES]) + f"\n# ... [truncated: {label}]"
    return "\n".join(lines)


def get_python_chunks(content: str) -> list[tuple[str, str]]:
    """
    Split a Python source file into (chunk_name, chunk_content) pairs.

    - Files ≤ MAX_CHUNK_LINES lines  → one chunk named "<module>"
    - Larger files                   → header chunk + one chunk per top-level def
    - Large classes                  → class-header chunk + one chunk per method
    """
    lines = content.splitlines()

    if len(lines) <= MAX_CHUNK_LINES:
        return [("<module>", content)]

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return [("<module>", _trim(lines, "unparseable"))]

    top_nodes = [
        n for n in ast.iter_child_nodes(tree)
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    if not top_nodes:
        return [("<module>", _trim(lines, "no top-level defs"))]

    chunks: list[tuple[str, str]] = []

    # Header: imports and module-level constants before the first definition
    first_def_line = top_nodes[0].lineno - 1
    if first_def_line > 0:
        chunks.append(("<module>", "\n".join(lines[:first_def_line])))

    for node in top_nodes:
        node_lines = lines[node.lineno - 1 : node.end_lineno]

        if isinstance(node, ast.ClassDef) and len(node_lines) > MAX_CHUNK_LINES:
            methods = [
                n for n in ast.iter_child_nodes(node)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if methods:
                # Class header = everything up to (but not including) the first method
                first_m_rel = methods[0].lineno - node.lineno
                class_header = node_lines[:first_m_rel]
                if class_header:
                    chunks.append((node.name, "\n".join(class_header)))

                for method in methods:
                    m_start = method.lineno - node.lineno
                    m_end   = method.end_lineno - node.lineno + 1
                    m_lines = node_lines[m_start:m_end]
                    chunks.append((
                        f"{node.name}.{method.name}",
                        _trim(m_lines, f"{node.name}.{method.name}"),
                    ))
            else:
                chunks.append((node.name, _trim(node_lines, node.name)))

        else:
            chunks.append((node.name, _trim(node_lines, node.name)))

    return chunks


def get_hdl_chunks(content: str) -> list[tuple[str, str]]:
    """Split an HDL file into MAX_CHUNK_LINES-line windows."""
    lines = content.splitlines()
    if len(lines) <= MAX_CHUNK_LINES:
        return [("<module>", content)]
    chunks = []
    for i in range(0, len(lines), MAX_CHUNK_LINES):
        part = lines[i : i + MAX_CHUNK_LINES]
        chunks.append((f"lines_{i+1}_{i+len(part)}", "\n".join(part)))
    return chunks


# ── Repo metadata ───────────────────────────────────────────────────────────────

def load_repo_metadata() -> dict[str, dict]:
    with open(REPOS_JSON) as f:
        data = json.load(f)

    mapping: dict[str, dict] = {}
    for repo_type, repos in data["repositories"].items():
        base = REPOS_DIR / repo_type
        for repo in repos:
            full_name = repo["full_name"]
            name      = repo["name"]
            for candidate in [base / full_name.replace("/", "_"), base / name]:
                if candidate.exists() and candidate.is_dir():
                    mapping[str(candidate)] = {
                        "repo_name":   full_name,
                        "repo_type":   repo_type,
                        "stars":       repo.get("stars", 0),
                        "topics":      repo.get("topics", []),
                        "description": repo.get("description", "") or "",
                    }
                    break
    return mapping


def should_skip_path(rel_path: str) -> bool:
    parts = set(Path(rel_path).parts)
    if parts & SKIP_PATH_SEGMENTS:
        return True
    name = Path(rel_path).name
    if name.startswith(SKIP_NAME_PREFIXES):
        return True
    if name.endswith(SKIP_NAME_SUFFIXES):
        return True
    return False


# ── Indexer ─────────────────────────────────────────────────────────────────────

def index_repo(repo_dir: Path, meta: dict, out_file) -> int:
    count = 0
    for fpath in repo_dir.rglob("*"):
        if not fpath.is_file():
            continue
        ext = fpath.suffix.lower()
        if ext not in EXTENSIONS:
            continue

        rel = str(fpath.relative_to(repo_dir))
        if should_skip_path(rel):
            continue

        try:
            if fpath.stat().st_size > MAX_FILE_BYTES:
                continue
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        chunks = (
            get_python_chunks(text) if ext == ".py"
            else get_hdl_chunks(text)
        )

        for chunk_name, chunk_content in chunks:
            if not chunk_content.strip():
                continue
            if len(chunk_content.encode()) > MAX_CHUNK_BYTES:
                continue

            code_id = f"{meta['repo_type']}/{meta['repo_name']}/{rel}::{chunk_name}"
            record = {
                "code_id":     code_id,
                "repo_name":   meta["repo_name"],
                "repo_type":   meta["repo_type"],
                "rel_path":    rel,
                "chunk_name":  chunk_name,
                "language":    EXTENSIONS[ext],
                "stars":       meta["stars"],
                "topics":      meta["topics"],
                "description": meta["description"],
                "content":     chunk_content,
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading repo metadata from {REPOS_JSON}")
    repo_map = load_repo_metadata()
    print(f"Found {len(repo_map)} repo directories\n")

    total = 0
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for repo_dir_str, meta in sorted(repo_map.items()):
            n = index_repo(Path(repo_dir_str), meta, f)
            if n:
                print(f"  [{meta['repo_type']:12s}] {meta['repo_name']:40s}  {n} chunks")
            total += n

    print(f"\nDone. {total} chunks indexed → {OUTPUT}")
    print(f"Index size: {OUTPUT.stat().st_size / 1_048_576:.1f} MB")


if __name__ == "__main__":
    main()
