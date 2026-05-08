#!/usr/bin/env python3
"""
Builds reference documentation for neuromorphic libraries into
agentic_workspace/references/{library}/reference.md.

Approach C (hybrid):
  - Fetch + convert web docs: brian2, snntorch, superneuromat
  - AST-extract API from source:  superneuroabm, fugu

Run from repo root or dataset-assembly/scripts/code/:
    python dataset-assembly/scripts/code/build_lib_references.py
"""

import ast
import re
import sys
import textwrap
from pathlib import Path

import requests

# ── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[3]
REPOS_DIR   = REPO_ROOT / "data" / "github_repos"
OUT_DIR     = REPO_ROOT / "agentic_workspace" / "references"

REPO_PATHS = {
    "brian2":       REPOS_DIR / "brian2"     / "brian-team_brian2",
    "superneuromat": REPOS_DIR / "superneuro" / "superneuromat",
    "superneuroabm": REPOS_DIR / "superneuro" / "superneuroabm",
    "fugu":         REPOS_DIR / "fugu"       / "Fugu",
}

# ── HTML → Markdown ─────────────────────────────────────────────────────────

def html_to_text(html: str) -> str:
    """Convert HTML to readable plain text / rough markdown."""
    try:
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0
        return h.handle(html)
    except ImportError:
        pass

    # Fallback: lightweight manual conversion
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Headings
    for lvl, tag in enumerate(["h1", "h2", "h3", "h4"], start=1):
        text = re.sub(rf"<{tag}[^>]*>(.*?)</{tag}>",
                      lambda m, l=lvl: "\n" + "#" * l + " " + m.group(1).strip() + "\n",
                      text, flags=re.DOTALL | re.IGNORECASE)
    # Code blocks
    text = re.sub(r"<pre[^>]*>(.*?)</pre>",
                  lambda m: "\n```\n" + m.group(1).strip() + "\n```\n",
                  text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<code[^>]*>(.*?)</code>",
                  lambda m: "`" + m.group(1).strip() + "`",
                  text, flags=re.DOTALL | re.IGNORECASE)
    # Lists
    text = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1\n", text, flags=re.DOTALL | re.IGNORECASE)
    # Paragraphs / breaks
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>(.*?)</p>", r"\1\n\n", text, flags=re.DOTALL | re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def fetch_page(url: str, timeout: int = 15) -> str | None:
    """Fetch a URL and return markdown text, or None on failure."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return html_to_text(resp.text)
    except Exception as e:
        print(f"  [WARN] Could not fetch {url}: {e}")
        return None


def fetch_pages(pages: list[tuple[str, str]], max_chars_per_page: int = 12_000) -> str:
    """
    Fetch a list of (title, url) pairs and join into one markdown document.
    Truncates each page to max_chars_per_page to keep files manageable.
    """
    parts = []
    for title, url in pages:
        print(f"  Fetching: {title} ({url})")
        text = fetch_page(url)
        if text:
            truncated = text[:max_chars_per_page]
            if len(text) > max_chars_per_page:
                truncated += f"\n\n... [truncated at {max_chars_per_page} chars] ..."
            parts.append(f"\n\n---\n## {title}\n_Source: {url}_\n\n{truncated}")
    return "\n".join(parts)


# ── AST Source Extractor ────────────────────────────────────────────────────

def get_docstring(node: ast.AST) -> str:
    """Extract docstring from a function/class node."""
    if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        return textwrap.dedent(node.body[0].value.value).strip()
    return ""


def format_args(args: ast.arguments) -> str:
    """Format function arguments as a readable string."""
    parts = []
    defaults_offset = len(args.args) - len(args.defaults)

    for i, arg in enumerate(args.args):
        if arg.arg == "self":
            continue
        annotation = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        default_idx = i - defaults_offset
        default = (f" = {ast.unparse(args.defaults[default_idx])}"
                   if default_idx >= 0 else "")
        parts.append(f"{arg.arg}{annotation}{default}")

    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    for kwarg in args.kwonlyargs:
        parts.append(kwarg.arg)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    return ", ".join(parts)


def extract_api_from_file(path: Path, module_prefix: str = "") -> list[dict]:
    """Parse a Python file and extract public class/function API entries."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []

    entries = []
    module_doc = get_docstring(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            class_doc = get_docstring(node)
            methods = []
            for item in node.body:
                if (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and not item.name.startswith("_")):
                    sig = f"{item.name}({format_args(item.args)})"
                    ret = f" -> {ast.unparse(item.returns)}" if item.returns else ""
                    methods.append({
                        "name": item.name,
                        "signature": sig + ret,
                        "doc": get_docstring(item),
                    })
            entries.append({
                "type": "class",
                "name": node.name,
                "module": module_prefix,
                "doc": class_doc,
                "methods": methods,
            })

        elif (isinstance(node, ast.FunctionDef)
              and not node.name.startswith("_")
              and not any(isinstance(p, ast.ClassDef) and node in ast.walk(p)
                          for p in ast.walk(tree) if isinstance(p, ast.ClassDef))):
            sig = f"{node.name}({format_args(node.args)})"
            ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            entries.append({
                "type": "function",
                "name": node.name,
                "module": module_prefix,
                "signature": sig + ret,
                "doc": get_docstring(node),
            })

    return entries


def entries_to_markdown(entries: list[dict], lib_name: str) -> str:
    """Render extracted API entries as markdown."""
    lines = []
    for e in entries:
        if e["type"] == "class":
            lines.append(f"\n### `{e['name']}`")
            if e["module"]:
                lines.append(f"_Module: `{e['module']}`_\n")
            if e["doc"]:
                lines.append(e["doc"] + "\n")
            if e["methods"]:
                lines.append("**Methods:**\n")
                for m in e["methods"]:
                    lines.append(f"- `{m['signature']}`")
                    if m["doc"]:
                        first_line = m["doc"].split("\n")[0]
                        lines.append(f"  - {first_line}")
                lines.append("")
        elif e["type"] == "function":
            lines.append(f"\n### `{e['signature']}`")
            if e["module"]:
                lines.append(f"_Module: `{e['module']}`_\n")
            if e["doc"]:
                lines.append(e["doc"].split("\n")[0])
            lines.append("")
    return "\n".join(lines)


def extract_source_reference(
    repo_path: Path,
    src_subdirs: list[str],
    lib_name: str,
    readme_glob: str = "README.md",
) -> str:
    """Build a reference doc from source: README + AST API extraction."""
    parts = []

    # README
    readmes = list(repo_path.glob(readme_glob))
    if readmes:
        parts.append("## README\n\n" + readmes[0].read_text(encoding="utf-8", errors="ignore"))

    # Source API
    all_entries = []
    for subdir in src_subdirs:
        src_dir = repo_path / subdir
        if not src_dir.exists():
            continue
        for py_file in sorted(src_dir.rglob("*.py")):
            if py_file.name.startswith("test_") or "test" in py_file.parts:
                continue
            rel = py_file.relative_to(repo_path)
            module = str(rel).replace("/", ".").removesuffix(".py")
            entries = extract_api_from_file(py_file, module_prefix=module)
            all_entries.extend(entries)

    if all_entries:
        parts.append("## API Reference (extracted from source)\n")
        parts.append(entries_to_markdown(all_entries, lib_name))

    return "\n\n".join(parts)


# ── Per-library configs ──────────────────────────────────────────────────────

LIBRARY_CONFIGS = {
    "brian2": {
        "method": "web",
        "header": (
            "# Brian2 Reference\n\n"
            "Brian2 is a clock-driven simulator for spiking neural networks written in Python. "
            "Key abstractions: `NeuronGroup`, `Synapses`, `Network`, `SpikeMonitor`, `StateMonitor`.\n"
        ),
        "pages": [
            ("Overview", "https://brian2.readthedocs.io/en/stable/index.html"),
            ("Introduction to neurons", "https://brian2.readthedocs.io/en/stable/resources/tutorials/1-intro-to-brian-neurons.html"),
            ("Introduction to synapses", "https://brian2.readthedocs.io/en/stable/resources/tutorials/2-intro-to-brian-synapses.html"),
            ("NeuronGroup", "https://brian2.readthedocs.io/en/stable/user/models.html"),
            ("Synapses", "https://brian2.readthedocs.io/en/stable/user/synapses.html"),
            ("Running networks", "https://brian2.readthedocs.io/en/stable/user/running.html"),
            ("Recording", "https://brian2.readthedocs.io/en/stable/user/recording.html"),
            ("Units system", "https://brian2.readthedocs.io/en/stable/user/units.html"),
        ],
    },
    "snntorch": {
        "method": "web",
        "header": (
            "# snnTorch Reference\n\n"
            "snnTorch is a PyTorch-based library for training spiking neural networks. "
            "Key modules: neuron models (Leaky, Synaptic, Alpha, LSTM variants), "
            "`spikegen` for input encoding, `functional` for loss/accuracy, `surrogate` for gradients.\n"
        ),
        "pages": [
            ("Overview", "https://snntorch.readthedocs.io/en/latest/index.html"),
            ("snntorch module API", "https://snntorch.readthedocs.io/en/latest/snntorch.html"),
            ("spikegen", "https://snntorch.readthedocs.io/en/latest/spikegen.html"),
            ("functional", "https://snntorch.readthedocs.io/en/latest/snntorch.functional.html"),
            ("surrogate gradients", "https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html"),
            ("snnTorch tutorials index", "https://snntorch.readthedocs.io/en/latest/tutorials/index.html"),
        ],
    },
    "superneuromat": {
        "method": "web+source",
        "header": (
            "# SuperNeuroMAT Reference\n\n"
            "SuperNeuroMAT (ORNL) is a matrix-based LIF spiking neural network simulator. "
            "Core class: `SNN`. Supports optional JIT (Numba) and GPU (CUDA) acceleration, "
            "STDP learning, and JSON serialization of networks.\n"
        ),
        "pages": [
            ("Overview", "https://ornl.github.io/superneuromat/"),
            ("Quickstart", "https://ornl.github.io/superneuromat/guide/quickstart.html"),
            ("API Reference", "https://ornl.github.io/superneuromat/reference/index.html"),
            ("SNN class", "https://ornl.github.io/superneuromat/reference/snn.html"),
        ],
        "src_subdirs": ["src/superneuromat"],
        "repo_key": "superneuromat",
    },
    "superneuroabm": {
        "method": "source",
        "header": (
            "# SuperNeuroABM Reference\n\n"
            "SuperNeuroABM (ORNL) is a GPU-accelerated multi-agent SNN simulator built on SAGESim. "
            "Core class: `NeuromorphicModel`. Supports Izhikevich and LIF soma models, "
            "single-exponential synapses, STDP, and multi-GPU via MPI.\n"
            "\n_Documentation generated from source code._\n"
        ),
        "src_subdirs": ["superneuroabm"],
        "repo_key": "superneuroabm",
    },
    "fugu": {
        "method": "source",
        "header": (
            "# Fugu Reference\n\n"
            "Fugu (Sandia National Labs) is a Python framework for composing computational graphs "
            "of spiking neural network primitives ('bricks') that compile to neuromorphic backends "
            "(Loihi, STACS, gensa, lava). Core abstractions: `Brick`, `Scaffold`.\n"
            "\n_Documentation generated from source code._\n"
        ),
        "src_subdirs": ["fugu"],
        "repo_key": "fugu",
    },
}


# ── Builder ──────────────────────────────────────────────────────────────────

def build_reference(lib_name: str, config: dict) -> str:
    method = config["method"]
    doc = config["header"]

    if "web" in method:
        print(f"  Fetching web docs...")
        doc += fetch_pages(config["pages"])

    if "source" in method:
        repo_path = REPO_PATHS[config["repo_key"]]
        print(f"  Extracting API from source: {repo_path}")
        doc += "\n\n" + extract_source_reference(
            repo_path=repo_path,
            src_subdirs=config["src_subdirs"],
            lib_name=lib_name,
        )

    return doc


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    libs = sys.argv[1:] or list(LIBRARY_CONFIGS.keys())

    for lib in libs:
        if lib not in LIBRARY_CONFIGS:
            print(f"Unknown library '{lib}'. Available: {list(LIBRARY_CONFIGS.keys())}")
            continue

        print(f"\n{'='*60}")
        print(f"Building reference: {lib}")
        print(f"{'='*60}")

        out_path = OUT_DIR / lib / "reference.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        doc = build_reference(lib, LIBRARY_CONFIGS[lib])
        out_path.write_text(doc, encoding="utf-8")

        size_kb = out_path.stat().st_size / 1024
        print(f"  -> {out_path} ({size_kb:.1f} KB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
