#!/usr/bin/env python3
"""
compare_llms.py

A script to perform A/B comparison between Naive LLM answers and RAG answers
(using the existing KG query engine).

Usage:
    python compare_llms.py --config configs/compare.yaml --query "Your query here"
    python compare_llms.py --config configs/compare.yaml --queries queries.txt

"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'text.usetex': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'mathtext.fontset': 'dejavusans'
})
import matplotlib.pylab as pylab
import numpy as np
import pandas as pd
import requests
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

# Try to import optional dependencies
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    sns.set()
    palette = sns.color_palette('muted')
    sns.set_palette(palette)
except ImportError:
    SEABORN_AVAILABLE = False
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Import project modules
# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
pd.options.display.float_format = '{:.2f}'.format

params = {
    'legend.fontsize': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
}
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
pylab.rcParams.update(params)

from src.kg_builder.llm_client import LLMClient

# Initialize Rich console
console = Console()

# --- Configuration & Setup ---

def load_config(path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return expand_env_vars(cfg)

def expand_env_vars(d: Any) -> Any:
    """Recursively expand environment variables in a dictionary or list."""
    if isinstance(d, dict):
        return {k: expand_env_vars(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [expand_env_vars(v) for v in d]
    elif isinstance(d, str):
        return os.path.expandvars(d)
    else:
        return d

def setup_logging(cfg: dict) -> None:
    """Setup logging based on configuration."""
    log_cfg = cfg.get("logging", {})
    level_str = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

# --- LLM Client Management ---

def get_llm_clients(cfg: dict) -> Dict[str, Any]:
    """
    Return dict[name] -> client with .generate(messages, **kwargs).
    
    We create a wrapper around LLMClient to provide async text generation.
    The cfg should contain an 'llms' section which might be a path to llm.yaml
    or the config dict itself.
    """
    llm_cfg = cfg.get("llms", {})
    
    # If llms is a string, assume it's a path to a yaml file
    if isinstance(llm_cfg, str):
        if os.path.exists(llm_cfg):
            with open(llm_cfg, 'r') as f:
                llm_cfg = yaml.safe_load(f)
        else:
            logging.warning(f"LLM config file {llm_cfg} not found. Using empty config.")
            llm_cfg = {}

    # Extract models list
    models = llm_cfg.get("models", [])
    if not models:
        logging.error("No models found in LLM configuration.")
        return {}

    clients = {}
    
    for model_conf in models:
        name = model_conf["name"]
        # Create LLMClient instance for this model
        client = LLMClient(
            base_url=model_conf["base_url"],
            model=model_conf["model"],
            temperature=model_conf.get("temperature", 0.2),
            max_tokens=model_conf.get("max_tokens", 512),
            provider=model_conf.get("provider", "vllm")
        )
        
        # Create async wrapper
        class ClientWrapper:
            def __init__(self, sync_client: LLMClient, model_config: dict):
                self.sync_client = sync_client
                self.model_config = model_config
                self.name = model_config["name"]
                
            async def generate(self, messages: list, **kwargs) -> dict:
                # Prepare kwargs based on model config and overrides
                gen_kwargs = {
                    "temperature": kwargs.get("temperature", self.model_config.get("temperature", 0.2)),
                    "max_tokens": kwargs.get("max_tokens", self.model_config.get("max_tokens", 512)),
                }
                
                # Construct prompt from messages
                prompt = ""
                for msg in messages:
                    if msg["role"] == "system":
                        prompt += f"System: {msg['content']}\n"
                    elif msg["role"] == "user":
                        prompt += f"User: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        prompt += f"Assistant: {msg['content']}\n"
                
                prompt += "Assistant: "
                
                # Use asyncio.to_thread to run synchronous generation
                try:
                    text = await asyncio.to_thread(
                        self._generate_text_sync,
                        prompt,
                        gen_kwargs
                    )
                    return {"text": text, "usage": {}}
                except Exception as e:
                    logging.error(f"Error generating text for {self.name}: {e}")
                    return {"text": "", "usage": {}, "error": str(e)}
            
            def _generate_text_sync(self, prompt: str, kwargs: dict) -> str:
                """Synchronous text generation using LLMClient's chat completion."""
                try:
                    # Use the existing _extract_vllm or similar method but for general text
                    if self.sync_client.provider == "vllm":
                        messages = [{"role": "user", "content": prompt}]
                        response = self.sync_client.client.chat.completions.create(
                            model=self.sync_client.model,
                            messages=messages,
                            temperature=kwargs.get("temperature", self.sync_client.temperature),
                            max_tokens=kwargs.get("max_tokens", self.sync_client.max_tokens),
                            timeout=self.sync_client.timeout
                        )
                        return response.choices[0].message.content
                    elif self.sync_client.provider == "ollama":
                        payload = {
                            "model": self.sync_client.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": kwargs.get("temperature", self.sync_client.temperature),
                                "num_predict": kwargs.get("max_tokens", self.sync_client.max_tokens)
                            }
                        }
                        response = requests.post(
                            f"{self.sync_client.base_url}/api/generate",
                            json=payload,
                            headers={'Content-Type': 'application/json; charset=utf-8'},
                            timeout=self.sync_client.timeout
                        )
                        if response.status_code == 200:
                            data = response.json()
                            return data.get('response', '')
                        else:
                            raise Exception(f"Ollama API error: {response.status_code}")
                    else:
                        raise ValueError(f"Unsupported provider: {self.sync_client.provider}")
                except Exception as e:
                    logging.error(f"Sync text generation failed: {e}")
                    raise

        clients[name] = ClientWrapper(client, model_conf)
        
    return clients

# --- Execution Functions ---

async def call_naive(
    client: Any, 
    query: str, 
    *, 
    temperature: float, 
    max_tokens: int, 
    timeout_s: int
) -> dict:
    """
    Return {text, prompt_tokens, completion_tokens, total_tokens, latency_ms, logprob_sum(optional), nll_per_token(optional)}
    """
    start_time = time.time()
    messages = [
        {"role": "system", "content": "You are a helpful expert. Answer concisely."},
        {"role": "user", "content": query}
    ]
    
    try:
        # Run with timeout
        response = await asyncio.wait_for(
            client.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            ),
            timeout=timeout_s
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Parse response - assuming response is a dict or object
        # Adjust based on actual AsyncLLMClient return type
        # If it returns just text (str), we wrap it.
        if isinstance(response, str):
            text = response
            usage = {}
        else:
            text = response.get("text", "") or response.get("content", "")
            usage = response.get("usage", {})
            
        return {
            "text": text,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "latency_ms": latency_ms,
            "status": "success"
        }
        
    except asyncio.TimeoutError:
        return {
            "text": "",
            "latency_ms": (time.time() - start_time) * 1000,
            "status": "error:timeout"
        }
    except Exception as e:
        return {
            "text": "",
            "latency_ms": (time.time() - start_time) * 1000,
            "status": f"error:{str(e)}"
        }

async def call_rag(query: str, cfg: dict) -> dict:
    """
    Use GraphRAG to answer the query via CLI ask command.
    """
    start_time = time.time()
    
    try:
        # Use CLI ask command instead of direct API
        import subprocess
        import sys
        
        # Get CLI parameters from config
        cli_path = cfg.get("kg_rag", {}).get("cli_path", "src.kg_builder.cli")
        config_dir = cfg.get("kg_rag", {}).get("config_dir", "configs")
        
        # Build command
        if isinstance(cli_path, list):
            cmd = cli_path + ["--config-dir", config_dir, "ask", query]
        else:
            cmd = [
                sys.executable, "-m", cli_path,
                "--config-dir", config_dir,
                "ask", query
            ]
        
        # Add optional parameters if specified in config
        graphrag_cfg = cfg.get("kg_rag", {}).get("cli_params", {})
        if "max_tokens" in graphrag_cfg:
            cmd.extend(["--max-tokens", str(graphrag_cfg["max_tokens"])])
        if "temperature" in graphrag_cfg:
            cmd.extend(["--temperature", str(graphrag_cfg["temperature"])])
        if "model" in graphrag_cfg:
            cmd.extend(["--model", graphrag_cfg["model"]])
        
        # Run the command and capture output
        timeout_sec = cfg.get("kg_rag", {}).get("timeout_sec", 120)  # Default 2 minutes
        result = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # Project root
            ),
            timeout=timeout_sec
        )
        
        stdout, stderr = await result.communicate()
        
        output = stdout.decode()
        error_output = stderr.decode()
        
        # Prioritize stdout for the answer to avoid capturing stderr logs
        if "🤖 Answer:" in output:
            source_text = output
        elif "🤖 Answer:" in error_output:
            source_text = error_output
        else:
            source_text = output + "\n" + error_output
        
        # Check if we got an answer even if return code is non-zero
        # The CLI might exit with code 1 due to warnings but still produce a valid answer
        if "🤖 Answer:" in source_text:
            logging.debug(f"Found answer in output (length: {len(source_text)})")
            
            # Extract answer from the output
            answer_start = source_text.find("🤖 Answer:")
            
            # Find the end of the answer (before sources or details)
            answer_end = source_text.find("\n\n📚 Sources:", answer_start)
            if answer_end == -1:
                answer_end = source_text.find("\n\n📊 Details:", answer_start)
            if answer_end == -1:
                answer_end = len(source_text)
            
            answer = source_text[answer_start + len("🤖 Answer:"):answer_end].strip()
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Read detailed metrics from latest_query_result.json
            metrics = {
                "entities_found": 0,
                "entry_nodes": 0,
                "subgraph_triples": 0,
                "top_triples": 0,
                "top_documents": 0,
                "top_5_doc_ids": "",
                "top_triples_ids": ""
            }
            
            try:
                # Try to find the result file in standard locations
                possible_paths = [
                    "output_triple/latest_query_result.json",
                    "output/latest_query_result.json"
                ]
                
                result_data = None
                for p in possible_paths:
                    if os.path.exists(p):
                        with open(p, 'r') as f:
                            result_data = json.load(f)
                        break
                
                if result_data:
                    metrics["entities_found"] = len(result_data.get("entities", []))
                    metrics["entry_nodes"] = len(result_data.get("entry_nodes", []))
                    metrics["subgraph_triples"] = len(result_data.get("subgraph_triples", []))
                    metrics["top_triples"] = len(result_data.get("top_triples", []))
                    metrics["top_documents"] = len(result_data.get("top_docs", []))
                    metrics["top_5_doc_ids"] = "|".join(result_data.get("top_docs", [])[:5])
                    
                    # Extract triple IDs or signatures
                    top_triples_list = result_data.get("top_triples", [])
                    triple_ids = []
                    for t in top_triples_list:
                        if 'id' in t:
                            triple_ids.append(str(t['id']))
                        else:
                            triple_ids.append(f"{t.get('h')}|{t.get('r')}|{t.get('t')}")
                    metrics["top_triples_ids"] = ";".join(triple_ids)

            except Exception as e:
                logging.warning(f"Failed to read detailed metrics: {e}")

            return {
                "text": answer,
                "latency_ms": latency_ms,
                "status": "success",
                **metrics
            }
        else:
            # No answer found, treat as error
            logging.error(f"CLI ask command failed with return code {result.returncode}")
            logging.error(f"STDERR: {error_output}")
            logging.error(f"STDOUT: {output[:500]}...")
            return {
                "text": "",
                "latency_ms": (time.time() - start_time) * 1000,
                "status": f"error:cli_failed:{error_output.strip()}"
            }
            
    except Exception as e:
        return {
            "text": "",
            "latency_ms": (time.time() - start_time) * 1000,
            "status": f"error:{str(e)}"
        }

# --- Metrics & Evaluation ---

_perplexity_model = None
_perplexity_tokenizer = None
_embedding_model = None

def get_perplexity_model(model_name: str = "gpt2"):
    global _perplexity_model, _perplexity_tokenizer
    if not TRANSFORMERS_AVAILABLE:
        return None, None
        
    if _perplexity_model is None:
        logging.info(f"Loading perplexity model: {model_name}")
        try:
            _perplexity_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _perplexity_model = AutoModelForCausalLM.from_pretrained(model_name)
            if torch.cuda.is_available():
                _perplexity_model = _perplexity_model.to("cuda")
            _perplexity_model.eval()
        except Exception as e:
            logging.error(f"Failed to load perplexity model: {e}")
            return None, None
            
    return _perplexity_model, _perplexity_tokenizer

def compute_perplexity(text: str, model_name: str = "gpt2") -> dict:
    """
    Use HF transformers to compute NLL/token and perplexity.
    Handle long texts by chunking with stride.
    """
    if not text or not text.strip():
        return {"perplexity": np.nan, "nll_per_token": np.nan}
        
    model, tokenizer = get_perplexity_model(model_name)
    if not model:
        return {"perplexity": np.nan, "nll_per_token": np.nan}
        
    encodings = tokenizer(text, return_tensors="pt")
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    
    device = model.device
    input_ids = encodings.input_ids.to(device)
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        
        input_ids_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_chunk.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids_chunk, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if not nlls:
        return {"perplexity": np.nan, "nll_per_token": np.nan}

    ppl = torch.exp(torch.stack(nlls).mean())
    nll_per_token = torch.stack(nlls).mean()
    
    return {
        "perplexity": ppl.item(),
        "nll_per_token": nll_per_token.item()
    }

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _embedding_model
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
        
    if _embedding_model is None:
        logging.info(f"Loading embedding model: {model_name}")
        try:
            _embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")
            return None
            
    return _embedding_model

def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    """Generate embeddings for texts using either sentence-transformers or Ollama."""
    if not texts:
        return np.array([])
    
    # Check if it's an Ollama model (contains '-ollama' suffix)
    if model_name.endswith('-ollama'):
        # Use Ollama for embedding
        from src.kg_builder.llm_client import LLMClient
        
        # Find the model config from llm.yaml
        llm_cfg = load_config("configs/llm.yaml")
        model_config = None
        for mc in llm_cfg.get("models", []):
            if mc["name"] == model_name:
                model_config = mc
                break
        
        if not model_config:
            logging.error(f"Ollama model config not found for {model_name}")
            return np.array([])
        
        # Create sync client
        client = LLMClient(
            base_url=model_config["base_url"],
            model=model_config["model"],
            temperature=0.0,  # Not used for embeddings
            provider="ollama"
        )
        
        embeddings = []
        for text in texts:
            try:
                embedding = client.generate_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                logging.error(f"Failed to generate embedding for text: {e}")
                embeddings.append(np.zeros(4096))  # Fallback zero vector
        
        return np.array(embeddings)
    
    else:
        # Use sentence-transformers
        model = get_embedding_model(model_name)
        if not model:
            return np.array([])
        
        embeddings = model.encode(texts, batch_size=16, normalize_embeddings=True)
        return embeddings

def pairwise_similarity(emb: np.ndarray) -> np.ndarray:
    if not SKLEARN_AVAILABLE or emb.size == 0:
        return np.array([])
    return cosine_similarity(emb)

def rouge_l_scores(texts: list[str]) -> np.ndarray:
    # Optional: Implement if needed, but cosine similarity is usually sufficient for this
    return np.array([])

# --- Visualization ---

def plot_agreement_matrix(names: list[str], sim: np.ndarray, out_png: str, title: str, xlabel: str = "Models", ylabel: str = "Models", cbar_label: str = "Similarity", annot_fontsize: int = 10) -> None:
    if sim.size == 0:
        return
        
    if SEABORN_AVAILABLE:
        # Use seaborn for better aesthetics
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(sim, dtype=bool))  # Mask upper triangle for cleaner look
        sns.heatmap(sim, mask=mask, annot=True, fmt=".2f", cmap="viridis", 
                   xticklabels=names, yticklabels=names, vmin=0, vmax=1,
                   cbar_kws={'label': cbar_label}, annot_kws={'size': annot_fontsize})
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
    else:
        # Fallback to matplotlib
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(names)), labels=names)
        ax.set_yticks(np.arange(len(names)), labels=names)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        for i in range(len(names)):
            for j in range(len(names)):
                text = ax.text(j, i, f"{sim[i, j]:.2f}",
                               ha="center", va="center", color="w" if sim[i, j] < 0.5 else "k", fontsize=annot_fontsize)
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        plt.colorbar(im, ax=ax, label=cbar_label)
    
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()

def plot_metric_matrix(names: list[str], values: np.ndarray, out_png: str, title: str, xlabel: str = "", ylabel: str = "Models", cbar_label: str = "Value", cmap: str = "viridis", annot_fontsize: int = 10) -> None:
    if values.size == 0:
        return
        
    if values.ndim == 1:
        values = values.reshape(-1, 1)
        
    if SEABORN_AVAILABLE:
        # Use seaborn for horizontal bar chart style heatmap
        plt.figure(figsize=(2, len(names) * 0.5 + 2))
        df = pd.DataFrame(values, index=names, columns=['Value'])
        sns.heatmap(df, annot=True, fmt=".2f", cmap=cmap, cbar=True,
                   yticklabels=True, xticklabels=False, cbar_kws={'label': cbar_label}, annot_kws={'size': annot_fontsize})
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    else:
        # Fallback to matplotlib
        fig, ax = plt.subplots(figsize=(2, len(names) * 0.5 + 2))
        im = ax.imshow(values, cmap=cmap)
        
        ax.set_yticks(np.arange(len(names)), labels=names)
        ax.set_xticks([]) 
        
        for i in range(len(names)):
            text = ax.text(0, i, f"{values[i, 0]:.2f}",
                           ha="center", va="center", color="w", fontsize=annot_fontsize)
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        plt.colorbar(im, ax=ax, label=cbar_label)
    
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()

def plot_delta_heatmap(model_names: list[str], deltas: list[float], out_png: str, title: str, xlabel: str = "", ylabel: str = "Models", cbar_label: str = "Delta", annot_fontsize: int = 10) -> None:
    if not deltas:
        return
        
    deltas_np = np.array(deltas).reshape(-1, 1)
    
    if SEABORN_AVAILABLE:
        # Use seaborn for better diverging colormap
        plt.figure(figsize=(2, len(model_names) * 0.5 + 2))
        df = pd.DataFrame(deltas_np, index=model_names, columns=['Delta'])
        sns.heatmap(df, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                   yticklabels=True, xticklabels=False, cbar=True,
                   cbar_kws={'label': cbar_label}, annot_kws={'size': annot_fontsize})
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    else:
        # Fallback to matplotlib
        fig, ax = plt.subplots(figsize=(2, len(model_names) * 0.5 + 2))
        im = ax.imshow(deltas_np, cmap="RdBu_r", vmin=-max(abs(deltas_np.min()), abs(deltas_np.max())), 
                       vmax=max(abs(deltas_np.min()), abs(deltas_np.max())))
        
        ax.set_yticks(np.arange(len(model_names)), labels=model_names)
        ax.set_xticks([])
        
        for i in range(len(model_names)):
            val = deltas[i]
            text = ax.text(0, i, f"{val:.2f}",
                           ha="center", va="center", color="k", fontsize=annot_fontsize)
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, label=cbar_label)
        fig.tight_layout()
    
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()

def scatter_ppl(ppl_naive: list[float], ppl_rag: list[float], labels: list[str], out_png: str, title: str) -> None:
    if not ppl_naive or not ppl_rag:
        return
        
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(ppl_naive, ppl_rag)
    
    for i, txt in enumerate(labels):
        ax.annotate(txt, (ppl_naive[i], ppl_rag[i]))
        
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    
    ax.set_xlabel("Perplexity (Naive)")
    ax.set_ylabel("Perplexity (RAG)")
    ax.set_title(title)
    ax.grid(True)
    
    plt.savefig(out_png, dpi=200)
    plt.close()

# --- Main Logic ---

async def process_query(
    query: str, 
    clients: Dict[str, Any], 
    cfg: dict, 
    results: list,
    console: Console,
    truncate_len: int,
    timestamp: str,
    date_str: str,
    time_str: str
):
    console.rule(f"[bold]Query: {query}[/bold]")
    
    defaults = cfg.get("defaults", {})
    temp_naive = defaults.get("temperature_naive", 0.2)
    max_tokens = defaults.get("max_output_tokens", 512)
    timeout = defaults.get("timeout_s", 600)
    
    eval_cfg = cfg.get("evaluator", {})
    ppl_model_name = eval_cfg.get("perplexity_model", "gpt2")
    
    # First, run RAG once per query (if enabled)
    rag_result = None
    if cfg.get("kg_rag", {}).get("enabled", False):
        console.print(f"[green]Running RAG query...[/green]")
        rag_result = await call_rag(query, cfg)
        
        if rag_result["status"] == "success":
            ppl_res_rag = compute_perplexity(rag_result["text"], ppl_model_name)
        else:
            ppl_res_rag = {"perplexity": np.nan, "nll_per_token": np.nan}
            
        row_rag = {
            "query": query,
            "provider": "rag_system", 
            "model": "RAG_System",    
            "mode": "rag",
            "text": rag_result["text"],
            "latency_ms": rag_result["latency_ms"],
            "prompt_tokens": 0, 
            "completion_tokens": 0,
            "total_tokens": 0,
            "perplexity": ppl_res_rag["perplexity"],
            "nll_per_token": ppl_res_rag["nll_per_token"],
            "length_chars": len(rag_result["text"]),
            "status": rag_result["status"],
            "timestamp": timestamp,
            "date": date_str,
            "time": time_str,
            "entities_found": rag_result.get("entities_found", 0),
            "entry_nodes": rag_result.get("entry_nodes", 0),
            "subgraph_triples": rag_result.get("subgraph_triples", 0),
            "top_triples": rag_result.get("top_triples", 0),
            "top_documents": rag_result.get("top_documents", 0),
            "top_5_doc_ids": rag_result.get("top_5_doc_ids", ""),
            "top_triples_ids": rag_result.get("top_triples_ids", "")
        }
        results.append(row_rag)
        
        text_display = rag_result["text"]
        if truncate_len > 0 and len(text_display) > truncate_len:
            text_display = text_display[:truncate_len] + "..."
        console.print(Panel(text_display, title="RAG System", border_style="green"))

    # Then run naive queries for each model
    for model_name, client in clients.items():
        # 1. Naive
        console.print(f"[blue]Running Naive for {model_name}...[/blue]")
        naive_res = await call_naive(
            client, query, 
            temperature=temp_naive, 
            max_tokens=max_tokens, 
            timeout_s=timeout
        )
        
        if naive_res["status"] == "success":
            ppl_res = compute_perplexity(naive_res["text"], ppl_model_name)
        else:
            ppl_res = {"perplexity": np.nan, "nll_per_token": np.nan}
            
        row_naive = {
            "query": query,
            "provider": client.model_config.get("provider", "unknown"),
            "model": model_name,
            "mode": "naive",
            "text": naive_res["text"],
            "latency_ms": naive_res["latency_ms"],
            "prompt_tokens": naive_res.get("prompt_tokens", 0),
            "completion_tokens": naive_res.get("completion_tokens", 0),
            "total_tokens": naive_res.get("total_tokens", 0),
            "perplexity": ppl_res["perplexity"],
            "nll_per_token": ppl_res["nll_per_token"],
            "length_chars": len(naive_res["text"]),
            "status": naive_res["status"],
            "timestamp": timestamp,
            "date": date_str,
            "time": time_str,
            "entities_found": 0,
            "entry_nodes": 0,
            "subgraph_triples": 0,
            "top_triples": 0,
            "top_documents": 0,
            "top_5_doc_ids": "",
            "top_triples_ids": ""
        }
        results.append(row_naive)
        
        text_display = naive_res["text"]
        if truncate_len > 0 and len(text_display) > truncate_len:
            text_display = text_display[:truncate_len] + "..."
        console.print(Panel(text_display, title=f"{model_name} (Naive)", border_style="blue"))

async def generate_results_for_queries(query_subset, query_names, clients, cfg, all_results, console, args, timestamp, date_str, time_str, base_out_dir, is_all_queries=False):
    """Generate results for a subset of queries"""
    
    # Filter results for this query subset
    subset_results = [r for r in all_results if r["query"] in query_subset]
    
    if not subset_results:
        return
        
    df = pd.DataFrame(subset_results)
    
    # Create output directory
    if is_all_queries:
        out_dir = os.path.join(base_out_dir, "all_queries")
    else:
        # Create a safe directory name from the first query
        safe_name = "".join(c for c in query_names[0][:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        out_dir = os.path.join(base_out_dir, safe_name)
    
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(out_dir, f"runs_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    console.print(f"[bold]Saved results to {csv_path}[/bold]")
    
    df_success = df[df["status"] == "success"].copy()
    
    if df_success.empty:
        console.print(f"[yellow]No successful runs for {'all queries' if is_all_queries else f'query: {query_names[0]}'}.[/yellow]")
        return

    # Print summary of responses
    if not is_all_queries:
        console.print(f"\n[bold]📊 Response Summary for: {query_names[0]}[/bold]")
    
    for query in query_subset:
        if not is_all_queries or len(query_subset) <= 3:  # Only show individual queries in summary for small sets
            console.print(f"\n[bold cyan]Query: {query}[/bold cyan]")
        query_results = df_success[df_success["query"] == query]
        
        for _, row in query_results.iterrows():
            model_name = row["model"]
            mode = row["mode"]
            text = row["text"][:200] + "..." if len(row["text"]) > 200 else row["text"]
            perplexity = row["perplexity"]
            
            if mode == "naive":
                console.print(f"  🤖 [green]{model_name}[/green] (Naive): {text}")
                console.print(f"     📈 Perplexity: {perplexity:.2f}")
            else:  # RAG
                console.print(f"  🧠 [blue]RAG System[/blue]: {text}")
                console.print(f"     📈 Perplexity: {perplexity:.2f}")

    df_success["variant"] = df_success["model"] + ":" + df_success["mode"]
    variants = sorted(df_success["variant"].unique())
    
    # Create display names for plots
    model_display_names = cfg.get("model_display_names", {})
    display_variants = []
    for v in variants:
        if ":rag" in v:
            display_variants.append("rag")
        else:
            model = v.split(":")[0]
            display_variants.append(model_display_names.get(model, model))
    
    n_variants = len(variants)
    sim_sum = np.zeros((n_variants, n_variants))
    sim_counts = 0
    
    embed_model_name = cfg.get("evaluator", {}).get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Get visualization configuration
    viz_cfg = cfg.get("visualization", {})
    enabled_plots = viz_cfg.get("enabled_plots", ["agreement_matrix", "perplexity_matrix", "delta_heatmap"])
    titles = viz_cfg.get("titles", {})
    axes = viz_cfg.get("axes", {})
    
    for q in query_subset:
        df_q = df_success[df_success["query"] == q]
        if df_q.empty:
            continue
            
        texts = []
        
        for i, v in enumerate(variants):
            row = df_q[df_q["variant"] == v]
            if not row.empty:
                texts.append(row.iloc[0]["text"])
            else:
                texts.append("") 
        
        embeddings = embed_texts(texts, embed_model_name)
        
        if embeddings.size > 0:
            sims = pairwise_similarity(embeddings)
            sim_sum += sims
            sim_counts += 1
            
    if sim_counts > 0:
        avg_sim = sim_sum / sim_counts
        
        if "agreement_matrix" in enabled_plots:
            try:
                axes_cfg = axes.get("agreement_matrix", {})
                xlabel = axes_cfg.get("xlabel", "Models")
                ylabel = axes_cfg.get("ylabel", "Models")
                cbar_label = axes_cfg.get("cbar_label", "Similarity")
                annot_fontsize = axes_cfg.get("annot_fontsize", 10)
                title_suffix = f" ({len(query_subset)} queries)" if len(query_subset) > 1 else ""
                plot_agreement_matrix(
                    display_variants, 
                    avg_sim, 
                    os.path.join(out_dir, f"agreement_matrix_{timestamp}.png"),
                    titles.get("agreement_matrix", "Average Pairwise Text Similarity") + title_suffix,
                    xlabel, ylabel, cbar_label, annot_fontsize
                )
                console.print(f"[green]Generated agreement matrix plot[/green]")
            except Exception as e:
                console.print(f"[yellow]Failed to generate agreement matrix: {e}[/yellow]")
        
    ppl_means = df_success.groupby("variant")["perplexity"].mean()
    ppl_values = ppl_means.reindex(variants).values
    
    if "perplexity_matrix" in enabled_plots:
        try:
            axes_cfg = axes.get("perplexity_matrix", {})
            xlabel = axes_cfg.get("xlabel", "")
            ylabel = axes_cfg.get("ylabel", "Models")
            cbar_label = axes_cfg.get("cbar_label", "Perplexity")
            annot_fontsize = axes_cfg.get("annot_fontsize", 10)
            title_suffix = f" ({len(query_subset)} queries)" if len(query_subset) > 1 else ""
            plot_metric_matrix(
                display_variants,
                ppl_values,
                os.path.join(out_dir, f"perplexity_matrix_{timestamp}.png"),
                titles.get("perplexity_matrix", "Average Perplexity (Lower is Better)") + title_suffix,
                xlabel, ylabel, cbar_label, "magma_r", annot_fontsize
            )
            console.print(f"[green]Generated perplexity matrix plot[/green]")
        except Exception as e:
            console.print(f"[yellow]Failed to generate perplexity matrix: {e}[/yellow]")
    
    rag_variant = "RAG_System:rag"
    if rag_variant in variants and "delta_heatmap" in enabled_plots:
        rag_ppl = ppl_means.get(rag_variant, np.nan)
        deltas = []
        naive_models = []
        
        for v in variants:
            if ":naive" in v:
                model_name = v.split(":")[0]
                naive_ppl = ppl_means.get(v, np.nan)
                delta = rag_ppl - naive_ppl
                deltas.append(delta)
                naive_models.append(model_name)
        
        if naive_models:
            try:
                axes_cfg = axes.get("delta_heatmap", {})
                xlabel = axes_cfg.get("xlabel", "")
                ylabel = axes_cfg.get("ylabel", "Models")
                cbar_label = axes_cfg.get("cbar_label", "Delta")
                annot_fontsize = axes_cfg.get("annot_fontsize", 10)
                title_suffix = f" ({len(query_subset)} queries)" if len(query_subset) > 1 else ""
                naive_models_display = [model_display_names.get(m, m) for m in naive_models]
                plot_delta_heatmap(
                    naive_models_display,
                    deltas,
                    os.path.join(out_dir, f"delta_heatmap_{timestamp}.png"),
                    titles.get("delta_heatmap", "Perplexity Delta (RAG System - Naive Model)\nNegative = RAG Improved") + title_suffix,
                    xlabel, ylabel, cbar_label, annot_fontsize
                )
                console.print(f"[green]Generated delta heatmap plot[/green]")
            except Exception as e:
                console.print(f"[yellow]Failed to generate delta heatmap: {e}[/yellow]")
            
    summary = {
        "timestamp": timestamp,
        "queries_processed": len(query_subset),
        "query_names": query_names,
        "variants": variants,
        "averages": df_success.groupby("variant")[["perplexity", "latency_ms", "length_chars"]].mean().to_dict(),
        "wins_rag_vs_naive": {} 
    }
    
    with open(os.path.join(out_dir, f"summary_{timestamp}.json"), "w") as f:
        json.dump(summary, f, indent=2)

async def main_async():
    parser = argparse.ArgumentParser(description="Compare LLM Naive vs RAG")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--query", help="Single query to run")
    parser.add_argument("--queries", help="Path to file with queries (one per line)")
    parser.add_argument("--truncate", type=int, default=0, help="Truncate console output")
    parser.add_argument("--outdir", help="Output directory")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds")
    parser.add_argument("--max_output_tokens", type=int, help="Max output tokens")
    
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    if args.outdir:
        cfg.setdefault("defaults", {})["output_dir"] = args.outdir
    if args.timeout:
        cfg.setdefault("defaults", {})["timeout_s"] = args.timeout
    if args.max_output_tokens:
        cfg.setdefault("defaults", {})["max_output_tokens"] = args.max_output_tokens
        
    setup_logging(cfg)
    
    base_out_dir = cfg.get("defaults", {}).get("output_dir", "results")
    os.makedirs(base_out_dir, exist_ok=True)
    
    queries = []
    if args.query:
        queries.append(args.query)
    if args.queries and os.path.exists(args.queries):
        with open(args.queries, "r") as f:
            queries.extend([line.strip() for line in f if line.strip()])
            
    if not queries:
        console.print("[red]No queries provided![/red]")
        return

    clients = get_llm_clients(cfg)
    if not clients:
        console.print("[red]No LLM clients initialized![/red]")
        sys.exit(1)
        
    results = []
    
    # Generate timestamp at the start of processing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    
    # Create timestamped base directory
    timestamp_dir = os.path.join(base_out_dir, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)
    
    console.print(f"[bold]Processing {len(queries)} queries...[/bold]")
    
    for query in queries:
        await process_query(query, clients, cfg, results, console, args.truncate, timestamp, date_str, time_str)
    
    # Generate results for all queries combined (existing behavior)
    console.print(f"\n[bold]📊 Generating aggregated results for all queries...[/bold]")
    await generate_results_for_queries(queries, ["all_queries"], clients, cfg, results, console, args, timestamp, date_str, time_str, timestamp_dir, is_all_queries=True)
    
    # Generate results for each individual query
    if len(queries) > 1:
        console.print(f"\n[bold]📊 Generating individual results for each query...[/bold]")
        for i, query in enumerate(queries):
            query_name = f"query_{i+1}"
            console.print(f"\n[bold]Processing {query_name}: {query[:50]}{'...' if len(query) > 50 else ''}[/bold]")
            await generate_results_for_queries([query], [query_name], clients, cfg, results, console, args, timestamp, date_str, time_str, timestamp_dir, is_all_queries=False)
    
    console.print(f"\n[bold green]✅ All results saved to: {timestamp_dir}[/bold green]")

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        traceback.print_exc()

if __name__ == "__main__":
    main()
