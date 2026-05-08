import os
from pathlib import Path
from typing import Any, Dict
import yaml
from pydantic import BaseModel


class BaseConfig(BaseModel):
    project_dir: str = "."
    output_dir: str = "output"
    merge_output_dirs: list[str] = []  # List of output directories to merge from (for multi-directory processing)
    logs_dir: str = "logs"
    kg_path: str = "./data/kg.jsonl"  # Path to the knowledge graph JSONL file
    seed: int = 42


class MongoConfig(BaseModel):
    uri: str = "mongodb://localhost:27017"
    database: str = "knight"
    collection: str = "publications"
    limit: int = 20
    file_hint_path: str = "/Users/rnu/Documents/ornl/knight/data/knight_database"


class LLMModelConfig(BaseModel):
    name: str
    base_url: str
    model: str
    temperature: float = 0.2
    max_tokens: int | None = None
    provider: str = "vllm"  # "vllm" or "ollama"
    timeout: int | None = None  # Model-specific timeout, falls back to parallel_execution.request_timeout


class ParallelExecutionConfig(BaseModel):
    enabled: bool = False
    max_concurrent_servers: int = 3
    request_timeout: int = 30
    retry_attempts: int = 2
    retry_delay: float = 1.0
    enable_llm_json_fix: bool = True  # Enable LLM-based JSON fixing for malformed responses


class RegenerationConfig(BaseModel):
    enabled: bool = True  # Enable regeneration for empty responses
    max_attempts: int = 3  # Maximum regeneration attempts
    temperature_increment: float = 0.3  # Increase temperature by this amount each retry
    max_temperature: float = 1.0  # Maximum temperature to use
    delay_between_attempts: float = 0.5  # Delay between regeneration attempts


class LLMConfig(BaseModel):
    models: list[LLMModelConfig] = [
        LLMModelConfig(
            name="gpt-oss-120b",
            base_url="http://pc0143857.ornl.gov:8000/v1",
            model="openai/gpt-oss-120b",
            provider="vllm"
        )
    ]
    parallel_execution: ParallelExecutionConfig = ParallelExecutionConfig()
    regeneration: RegenerationConfig = RegenerationConfig()
    # Legacy single model config (for backward compatibility) - now optional
    base_url: str = "http://pc0143857.ornl.gov:8000/v1"
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.2
    max_tokens: int = 1024


class PipelineConfig(BaseModel):
    extraction: Dict[str, Any] = {"max_chars": 16000}
    normalize: Dict[str, Any] = {"lower": True, "strip_punct": True, "exclusions": ["cmos", "mos"]}
    dedupe: Dict[str, Any] = {"fuzzy_threshold": 0.92}
    clustering: Dict[str, Any] = {"algorithm": "louvain", "min_cluster_size": 3}
    embedding: Dict[str, Any] = {"model": "llama3.3:70b-instruct-q2_K", "batch_size": 10, "dimension": 4096}
    faiss: Dict[str, Any] = {"index_path": "output/faiss_index.idx", "index_type": "IndexFlatIP"}
    graphrag: Dict[str, Any] = {
        "concurrency_limit": 10,
        "k_hops": 2,
        "max_subgraph_triples": 50,
        "max_prompt_triples": 20,
        "max_prompt_documents": 5,
        "llm_endpoints": {
            "extractor": "llama3.3:70b-instruct-q2_K",
            "ranker": "ibm-granite-13b",
            "reasoner": "gpt-oss-120b"
        }
    }
    ontology: Dict[str, Any] = {"name": "UnifiedKG", "base_iri": "http://ornl.gov/kg#"}
    regeneration: Dict[str, Any] = {
        "enabled": True,
        "max_attempts": 3,
        "temperature_increment": 0.3,
        "max_temperature": 1.0,
        "delay_between_attempts": 0.5
    }


class PostmergingConfig(BaseModel):
    enabled: bool = True
    cleanup_rules: list[Dict[str, Any]] = [
        {
            "name": "remove_quotes",
            "pattern": "^[\"']+|['\"]+$",
            "replacement": "",
            "description": "Remove leading and trailing single or double quotes"
        },
        {
            "name": "remove_special_chars_start",
            "pattern": "^[$#*:;.,!?@%^&()\\[\\]{}|\\-_=+~`]+",
            "replacement": "",
            "description": "Remove leading special characters like $, #, *, :, ;, etc."
        },
        {
            "name": "remove_special_chars_end",
            "pattern": "[$#*:;.,!?@%^&()\\[\\]{}|\\-_=+~`]+$",
            "replacement": "",
            "description": "Remove trailing special characters like $, #, *, :, ;, etc."
        },
        {
            "name": "remove_node_prefixes",
            "pattern": ":node[_-]",
            "replacement": "",
            "description": "Remove :node_ or :node- substrings"
        },
        {
            "name": "remove_a_space_prefix",
            "pattern": "^a\\s+",
            "replacement": "",
            "description": "Remove 'a ' prefix (a followed by space)"
        },
        {
            "name": "remove_all_space_prefix",
            "pattern": "^all\\s+",
            "replacement": "",
            "description": "Remove 'all ' prefix (all followed by space)"
        }
    ]
    apply_to_fields: list[str] = [
        "entity_name",
        "subject_entity_name",
        "object_entity_name",
        "predicate",
        "h", "r", "t"
    ]


class Config:
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.base = self._load_config("base.yaml", BaseConfig)
        self.mongo = self._load_config("mongo.yaml", MongoConfig)
        self.llm = self._load_config("llm.yaml", LLMConfig)
        self.pipeline = self._load_config("pipeline.yaml", PipelineConfig)
        self.postprocessing = self._load_config("postmerging.yaml", PostmergingConfig)

    def _load_config(self, filename: str, model_cls):
        path = self.config_dir / filename
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        # Override with env vars
        data = self._override_with_env(data, filename.split('.')[0])
        return model_cls(**data)

    def _override_with_env(self, data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        def override_dict(d: Dict[str, Any], env_prefix: str):
            for key, value in d.items():
                env_key = f"{env_prefix}_{key}".upper()
                if env_key in os.environ:
                    if isinstance(value, dict):
                        d[key] = override_dict(value, env_key)
                    else:
                        d[key] = os.environ[env_key]
            return d
        return override_dict(data, prefix)


def load_config(config_dir: str = "configs") -> Config:
    return Config(Path(config_dir))
