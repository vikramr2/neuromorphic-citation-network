import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional


class TraceLogger:
    """Singleton class for structured JSONL logging of pipeline steps."""

    _instance = None

    def __new__(cls, logs_dir: Path):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, logs_dir: Path):
        if self._initialized:
            return
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(exist_ok=True)
        self._initialized = True

    def log_event(self, step: str, input_summary: str, output_summary: str,
                  latency_ms: float, metadata: Optional[Dict[str, Any]] = None):
        """Log a structured event to daily JSONL file."""
        timestamp = datetime.now().isoformat()
        event = {
            "timestamp": timestamp,
            "step": step,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "latency_ms": latency_ms,
            "metadata": metadata or {}
        }

        # Create daily log file
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = self.logs_dir / f"trace_{date_str}.jsonl"

        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        except Exception as e:
            logging.error(f"Failed to write trace log: {e}")


def setup_logging(logs_dir: Path, level: str = "INFO"):
    logs_dir.mkdir(exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = RotatingFileHandler(logs_dir / "kg_builder.log", maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
