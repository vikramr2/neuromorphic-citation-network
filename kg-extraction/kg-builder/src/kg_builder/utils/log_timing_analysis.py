#!/usr/bin/env python3
"""
Log-based timing analysis for KG Builder LLM processing.

Analyzes log files to provide detailed metrics on:
- Document processing time distributions
- Per-server performance metrics (requests, successes, timing)
- Progress tracking and completion estimates
- Retry and failure analysis (timeouts, regeneration attempts)
- Server reliability and throughput analysis

Usage: python -m src.kg_builder.utils.log_timing_analysis <total_documents>
Example: python -m src.kg_builder.utils.log_timing_analysis 2242
"""

import sys
import re
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import yaml


class TimingAnalyzer:
    def __init__(self, log_file: str, config_file: str = None):
        self.log_file = log_file
        # Use config from the project root (three levels up from utils)
        project_root = Path(__file__).parent.parent.parent.parent
        self.base_config = self._load_base_config()
        self.config_file = config_file or project_root / 'configs' / 'llm.yaml'
        self.server_mapping, self.model_to_url, self.server_to_model, self.server_to_url = self._load_server_mapping()
        self.num_prompts = self._load_num_prompts()
        self.num_models = self._load_num_models()
        self.request_timeout = self._load_request_timeout()
        # Get all log files (including rotated ones)
        self.log_files = self._get_all_log_files(log_file)

    def _load_base_config(self) -> Dict:
        """Load base configuration from base.yaml."""
        try:
            project_root = Path(__file__).parent.parent.parent.parent
            base_config_path = project_root / 'configs' / 'base.yaml'
            with open(base_config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            return config
        except Exception as e:
            print(f"Warning: Could not load base configuration: {e}")
            return {}

    def _get_all_log_files(self, log_file: str) -> List[str]:
        """Get all log files including rotated ones, sorted by rotation number (newest first)."""
        log_path = Path(log_file)
        log_dir = log_path.parent
        base_name = log_path.name
        
        # Find all log files (main + rotated)
        log_files = []
        for file_path in log_dir.glob(f"{base_name}*"):
            if file_path.is_file():
                log_files.append(str(file_path))
        
        # Sort by rotation number: kg_builder.log, kg_builder.log.1, kg_builder.log.2, etc.
        def sort_key(file_path):
            if file_path == str(log_path):
                return 0  # Main log file comes first
            else:
                # Extract rotation number from .1, .2, etc.
                match = re.search(r'\.(\d+)$', Path(file_path).name)
                if match:
                    return int(match.group(1))
                else:
                    return 999  # Fallback
        
        log_files.sort(key=sort_key)
        return log_files
    def _load_server_mapping(self) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
        """Load server mapping from llm.yaml config file."""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            server_mapping = {}  # model_name -> server
            model_to_url = {}    # model_name -> base_url
            server_to_model = {} # server -> model_name
            server_to_url = {}   # server -> base_url
            
            for model_config in config.get('models', []):
                model_name = model_config['name']
                base_url = model_config['base_url']
                # Extract server from base_url (e.g., "http://medz2.ornl.gov:11434" -> "medz2.ornl.gov")
                server = base_url.replace('http://', '').replace('https://', '').split(':')[0]
                
                server_mapping[model_name] = server
                model_to_url[model_name] = base_url
                server_to_model[server] = model_name
                server_to_url[server] = base_url
            
            return server_mapping, model_to_url, server_to_model, server_to_url
        except Exception as e:
            print(f"Warning: Could not load server mapping from {self.config_file}: {e}")
            print("Using default mapping...")
            # Provide default mappings
            default_server_mapping = {
                'gpt-oss-120b-ollama': 'medz2.ornl.gov',
                'llama3.3-quantized-ollama': 'carz3.ornl.gov',
                'gpt-oss-20b-ollama': 'carz1.ornl.gov',
                'granite4-small-h-ollama': 'carz2.ornl.gov'
            }
            default_model_to_url = {
                'gpt-oss-120b-ollama': 'http://medz2.ornl.gov:11434',
                'llama3.3-quantized-ollama': 'http://carz3.ornl.gov:11434',
                'gpt-oss-20b-ollama': 'http://carz1.ornl.gov:11434',
                'granite4-small-h-ollama': 'http://carz2.ornl.gov:11434'
            }
            default_server_to_model = {v: k for k, v in default_server_mapping.items()}
            default_server_to_url = {k: v for k, v in default_model_to_url.items()}
            
            return default_server_mapping, default_model_to_url, default_server_to_model, default_server_to_url

    def _load_request_timeout(self) -> float:
        """Load the request timeout from llm.yaml config file."""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            parallel_execution = config.get('parallel_execution', {})
            return parallel_execution.get('request_timeout', 600.0)  # Default to 600 seconds (10 minutes)
        except Exception as e:
            print(f"Warning: Could not load request timeout from {self.config_file}: {e}")
            return 600.0  # Default to 600 seconds

    def _load_num_prompts(self) -> int:
        """Load the number of prompts configured for extraction."""
        try:
            project_root = Path(__file__).parent.parent.parent.parent
            pipeline_config = project_root / 'configs' / 'pipeline.yaml'
            with open(pipeline_config, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            prompt_files = config.get('extraction', {}).get('prompt_files', ['prompts/triple_extraction.md'])
            if isinstance(prompt_files, str):
                prompt_files = [prompt_files]
            return len(prompt_files)
        except Exception as e:
            print(f"Warning: Could not load prompt configuration: {e}")
            return 1  # Default to 1 prompt

    def _load_num_models(self) -> int:
        """Load the number of models configured for extraction."""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            models = config.get('models', [])
            return len(models)
        except Exception as e:
            print(f"Warning: Could not load model configuration: {e}")
            return 4  # Default to 4 models

    def _get_current_prompt_from_log(self) -> Tuple[int, int, str]:
        """Get the current prompt being processed from recent log entries.
        
        Returns:
            Tuple of (current_prompt_index, total_prompts, prompt_filename)
            Returns (0, 0, "") if no prompt processing found
        """
        try:
            # Look for the most recent "Processing prompt" line
            recent_prompt_lines = []
            for log_file in self.log_files:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if 'Processing prompt' in line and '/' in line:
                            recent_prompt_lines.append(line)
            
            if recent_prompt_lines:
                # Get the most recent prompt processing line
                most_recent_line = recent_prompt_lines[-1]
                
                # Extract pattern like "Processing prompt 1/1: triple_extraction.md"
                prompt_match = re.search(r'Processing prompt (\d+)/(\d+):?\s*([^\s]+)?', most_recent_line)
                if prompt_match:
                    current = int(prompt_match.group(1))
                    total = int(prompt_match.group(2))
                    filename = prompt_match.group(3) if prompt_match.group(3) else ""
                    return (current, total, filename)
            
            return (0, 0, "")
        except Exception as e:
            print(f"Warning: Could not determine current prompt: {e}")
            return (0, 0, "")

    def _find_most_recent_run_total(self) -> int:
        """Find the total number of documents in the most recent run."""
        try:
            # Look for the most recent document completion line (✅ format) across all log files
            recent_completion_lines = []
            for log_file in self.log_files:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if '✅' in line and 'Extracted' in line and 'triples' in line and 'models' in line and 'for doc' in line:
                            recent_completion_lines.append(line)
            
            if recent_completion_lines:
                # Get the most recent completion line
                most_recent_line = recent_completion_lines[-1]
                # Extract the total from pattern like [1005/2242]
                total_match = re.search(r'\[(\d+)/(\d+)\]', most_recent_line)
                if total_match:
                    return int(total_match.group(2))  # Return the total
            
            # Fallback: look for any completion line and get the maximum total
            max_total = 0
            for line in recent_completion_lines[-50:]:  # Check last 50 completion lines
                total_match = re.search(r'\[(\d+)/(\d+)\]', line)
                if total_match:
                    max_total = max(max_total, int(total_match.group(2)))
            
            return max_total if max_total > 0 else 0  # Return 0 if no total found
        except Exception as e:
            print(f"Warning: Could not determine recent run total: {e}")
            return 0  # Return 0 as fallback

    def _get_total_documents_from_filesystem(self) -> int:
        """Get total number of documents from input files in output_triple/input/."""
        try:
            project_root = Path(__file__).parent.parent.parent.parent
            input_dir = project_root / 'output_triple' / 'input'
            if input_dir.exists():
                # Count .txt files in input directory
                txt_files = list(input_dir.glob('*.txt'))
                return len(txt_files)
            return 0
        except Exception as e:
            print(f"Warning: Could not count input documents: {e}")
            return 0

    def _get_processed_documents_from_filesystem(self) -> int:
        """Get number of fully processed documents from JSONL files in output_triple/docs/."""
        try:
            project_root = Path(__file__).parent.parent.parent.parent
            docs_dir = project_root / 'output_triple' / 'docs'
            if docs_dir.exists():
                # Count .jsonl files in docs directory
                jsonl_files = list(docs_dir.glob('*.jsonl'))
                return len(jsonl_files)
            return 0
        except Exception as e:
            print(f"Warning: Could not count processed documents: {e}")
            return 0

    def _get_completed_combinations_from_filesystem(self) -> int:
        """Get total number of completed (model, prompt) combinations from JSONL files.
        
        This is more accurate than just counting files, as it accounts for partial processing
        when multiple prompts are configured.
        """
        try:
            import json
            project_root = Path(__file__).parent.parent.parent.parent
            docs_dir = project_root / 'output_triple' / 'docs'
            if not docs_dir.exists():
                return 0
            
            total_combinations = 0
            for jsonl_file in docs_dir.glob('*.jsonl'):
                combinations = set()
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    data = json.loads(line)
                                    model = data.get('_model', 'unknown')
                                    prompt = data.get('_prompt', 'unknown')
                                    combinations.add((model, prompt))
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    print(f"Warning: Could not read {jsonl_file}: {e}")
                    continue
                
                total_combinations += len(combinations)
            
            return total_combinations
        except Exception as e:
            print(f"Warning: Could not count completed combinations: {e}")
            return 0

    def _get_computing_documents_from_log(self) -> int:
        """Get number of documents currently being processed from recent log entries."""
        try:
            computing_docs = set()
            
            # Read from all log files (newest first)
            for log_file in self.log_files:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Look for lines indicating documents being processed
                        # Pattern: "Processing doc <doc_id>" or similar
                        processing_patterns = [
                            r'🔄 \[(\d+)/(\d+)\] Processing doc ([a-f0-9]+)',
                            r'Querying model: ([^\s]+).*for doc ([a-f0-9]+)',
                            r'Processing document ([a-f0-9]+)',
                            r'Starting extraction.*doc ([a-f0-9]+)'
                        ]
                        
                        for pattern in processing_patterns:
                            match = re.search(pattern, line)
                            if match:
                                if len(match.groups()) >= 3:  # Pattern with doc count
                                    doc_id = match.group(3)
                                elif 'for doc' in pattern:
                                    doc_id = match.group(2)
                                else:
                                    doc_id = match.group(1)
                                computing_docs.add(doc_id)
            
            return len(computing_docs)
        except Exception as e:
            print(f"Warning: Could not determine computing documents: {e}")
            return 0

    def _find_current_run_start_time(self) -> str:
        """Find the start time of the current run (most recent sequence of document completions)."""
        try:
            # Find all document completion lines across all log files
            completion_lines = []
            for log_file in self.log_files:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if '✅' in line and 'Extracted' in line and 'triples' in line and 'models' in line and 'for doc' in line:
                            # Extract timestamp and document number
                            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}', line)
                            doc_match = re.search(r'\[(\d+)/(\d+)\]', line)
                            if timestamp_match and doc_match:
                                timestamp = timestamp_match.group(1)
                                current_doc = int(doc_match.group(1))
                                total_docs = int(doc_match.group(2))
                                completion_lines.append((timestamp, current_doc, total_docs))
            
            if not completion_lines:
                return ""
            
            # Sort by timestamp (most recent first)
            completion_lines.sort(key=lambda x: x[0], reverse=True)
            
            # Find the most recent total
            most_recent_total = completion_lines[0][2]
            
            # Get all completions with this most recent total, sorted by document number
            recent_run_completions = [(ts, doc) for ts, doc, total in completion_lines if total == most_recent_total]
            recent_run_completions.sort(key=lambda x: x[1])  # Sort by document number
            
            # Find consecutive sequences (allowing for some gaps due to parallel processing)
            if recent_run_completions:
                # Take the most recent completions (last 50 or so) to find the current active sequence
                recent_completions = recent_run_completions[-50:]
                
                # Find the earliest timestamp among the most recent document numbers
                max_doc_num = max(doc for _, doc in recent_completions)
                # Look for documents in the range [max_doc_num - 20, max_doc_num] to find the start of current activity
                current_range_docs = [(ts, doc) for ts, doc in recent_completions if max_doc_num - 20 <= doc <= max_doc_num]
                
                if current_range_docs:
                    # Return the earliest timestamp in the current active range
                    current_range_docs.sort(key=lambda x: x[0])  # Sort by timestamp
                    return current_range_docs[0][0]
            
            # Fallback: return the earliest timestamp of the most recent total
            earliest_timestamp = min(ts for ts, _, total in completion_lines if total == most_recent_total)
            return earliest_timestamp
            
        except Exception as e:
            print(f"Warning: Could not determine current run start time: {e}")
            return ""  # Return empty string to disable time filtering

    def parse_time(self, time_str: str) -> float:
        """Convert time string like '66.20s' to seconds."""
        if time_str.endswith('s'):
            return float(time_str[:-1])
        return float(time_str)

    def format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0 or hours > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")

        return "".join(parts)

    def analyze_logs(self, total_docs: int = None, cumulative: bool = False) -> Dict:
        """Analyze the log file and return comprehensive metrics."""
        results = {
            'documents': {
                'processed': 0,
                'total_expected': total_docs,
                'times': [],
                'distribution': defaultdict(int)
            },
            'servers': defaultdict(lambda: {
                'requests': 0,
                'successful_responses': 0,
                'failed_responses': 0,
                'total_time': 0.0,
                'response_times': [],
                'success_rate': 0.0,
                'timeouts': 0,
                'malformed_response_failures': 0,
                'regeneration_attempts': 0,
                'regeneration_successes': 0,
                'queries_with_regeneration': 0,
                'queries_succeeded_after_regeneration': 0,
                'queries_exceeding_timeout': 0,  # New: queries that took longer than request_timeout
                'other_faults': 0,  # New: other types of faults/errors
                'triples_extracted': 0  # New: total triples extracted by this server
            }),
            'retries': {
                'regeneration_attempts': 0,
                'regeneration_successes': 0,
                'timeouts': 0,
                'malformed_response_failures': 0,
                'other_faults': 0  # New: other types of connection/server faults
            }
        }

        # Regex patterns for different log entry types
        # Match document completion lines (✅ format with actual processing time)
        doc_completion_pattern = r'✅\s*\[(\d+)/(\d+)\].*?(\d+)\s+triples.*?for\s+doc\s+([a-f0-9]+).*?in\s+([\d.]+)s'
        model_extraction_pattern = r'Extracted\s+(\d+)\s+triples\s+from\s+([^\s]+)\s+in\s+([\d.]+)s'
        failure_pattern = r'Failed\s+to\s+extract.*from\s+([^\s]+).*after\s+([\d.]+)s'
        
        # Additional patterns for retries and failures (support both old and new formats)
        regeneration_attempt_pattern = r'Regeneration attempt (\d+)/(\d+)(?: for ([^\s]+))?'
        regeneration_success_pattern = r'Regeneration successful on attempt (\d+)(?: for ([^\s]+))?'
        regeneration_triggered_pattern = r'Regeneration triggered for ([^\s]+)'
        regeneration_completed_pattern = r'Regeneration completed successfully for ([^\s]+)'
        regeneration_failed_pattern = r'Regeneration failed completely for ([^\s]+)'
        timeout_pattern = r'Read timed out'
        malformed_response_failure_pattern = r'Failed to fix malformed response'

        # Find the total for the most recent run and its start time
        current_run_total = self._find_most_recent_run_total()
        current_run_start = self._find_current_run_start_time()
        
        # Get document counts from filesystem
        total_docs_from_filesystem = self._get_total_documents_from_filesystem()
        processed_docs_from_filesystem = self._get_processed_documents_from_filesystem()
        completed_combinations_from_filesystem = self._get_completed_combinations_from_filesystem()
        computing_docs_from_log = self._get_computing_documents_from_log()
        
        # Get current prompt information from logs
        current_prompt_idx, total_prompts_from_log, current_prompt_filename = self._get_current_prompt_from_log()
        
        detected_total = 0
        model_extractions_count = 0
        model_extractions_per_doc = len(self.server_mapping) * self.num_prompts  # Account for multiple prompts
        current_server_context = None  # Track the most recent server mentioned for associating failures

        try:
            # Read from all log files (newest first)
            for log_file in self.log_files:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # For recent-only mode, only include lines from the current run
                        # Temporarily disable time filtering to capture all recent server data
                        if not cumulative and current_run_start and False:  # Disabled for debugging
                            pass  # Skip time filtering
                        # Check for document completion
                        doc_match = re.search(doc_completion_pattern, line)
                        if doc_match:
                            current, total, triples, doc_id, time_str = doc_match.groups()
                            detected_total = max(detected_total, int(total))  # Track the maximum total seen
                            time_seconds = self.parse_time(time_str)

                            results['documents']['processed'] += 1
                            results['documents']['times'].append(time_seconds)

                            # Categorize by time buckets
                            if time_seconds < 60:
                                results['documents']['distribution']['under_1min'] += 1
                            elif time_seconds < 300:
                                results['documents']['distribution']['1_5min'] += 1
                            elif time_seconds < 600:
                                results['documents']['distribution']['5_10min'] += 1
                            elif time_seconds < 3600:
                                results['documents']['distribution']['10min_1hr'] += 1
                            else:
                                results['documents']['distribution']['over_1hr'] += 1

                        # Check for individual model extractions (successful)
                        model_match = re.search(model_extraction_pattern, line)
                        if model_match:
                            triples, model_name, time_str = model_match.groups()
                            triples_count = int(triples)
                            time_seconds = self.parse_time(time_str)
                            model_extractions_count += 1

                            server = self.server_mapping.get(model_name, 'unknown')
                            current_server_context = server  # Update context for subsequent failures

                            results['servers'][server]['requests'] += 1
                            results['servers'][server]['successful_responses'] += 1
                            results['servers'][server]['total_time'] += time_seconds
                            results['servers'][server]['response_times'].append(time_seconds)
                            results['servers'][server]['triples_extracted'] += triples_count  # Add triples count
                            
                            # Track queries that exceeded the timeout
                            if time_seconds > self.request_timeout:
                                results['servers'][server]['queries_exceeding_timeout'] += 1

                        # Check for failures
                        fail_match = re.search(failure_pattern, line)
                        if fail_match:
                            model_name, time_str = fail_match.groups()
                            time_seconds = self.parse_time(time_str)

                            server = self.server_mapping.get(model_name, 'unknown')
                            current_server_context = server  # Update context for subsequent failures

                            results['servers'][server]['requests'] += 1
                            results['servers'][server]['failed_responses'] += 1
                            results['servers'][server]['total_time'] += time_seconds

                        # Check for regeneration attempts
                        regen_attempt_match = re.search(regeneration_attempt_pattern, line)
                        if regen_attempt_match:
                            groups = regen_attempt_match.groups()
                            attempt_num, max_attempts = groups[0], groups[1]
                            server = groups[2] if len(groups) > 2 and groups[2] else None
                            
                            results['retries']['regeneration_attempts'] += 1
                            # Use explicit server if available, otherwise use context
                            target_server = server if server and server in self.server_to_model else current_server_context
                            if target_server:
                                results['servers'][target_server]['regeneration_attempts'] += 1

                        # Check for regeneration successes
                        regen_success_match = re.search(regeneration_success_pattern, line)
                        if regen_success_match:
                            groups = regen_success_match.groups()
                            attempt_num = groups[0]
                            server = groups[1] if len(groups) > 1 and groups[1] else None
                            
                            results['retries']['regeneration_successes'] += 1
                            # Use explicit server if available, otherwise use context
                            target_server = server if server and server in self.server_to_model else current_server_context
                            if target_server:
                                results['servers'][target_server]['regeneration_successes'] += 1

                        # Check for regeneration triggered
                        regen_triggered_match = re.search(regeneration_triggered_pattern, line)
                        if regen_triggered_match:
                            server = regen_triggered_match.group(1)
                            if server in self.server_to_model:
                                results['servers'][server]['queries_with_regeneration'] += 1

                        # Check for regeneration completed successfully
                        regen_completed_match = re.search(regeneration_completed_pattern, line)
                        if regen_completed_match:
                            server = regen_completed_match.group(1)
                            if server in self.server_to_model:
                                results['servers'][server]['queries_succeeded_after_regeneration'] += 1

                        # Check for regeneration failed completely
                        regen_failed_match = re.search(regeneration_failed_pattern, line)
                        if regen_failed_match:
                            server = regen_failed_match.group(1)
                            if server in self.server_to_model:
                                results['servers'][server]['queries_with_regeneration'] += 1  # Still counts as having regeneration

                        # Check for timeouts - extract server from error message
                        if re.search(timeout_pattern, line):
                            results['retries']['timeouts'] += 1
                            # Try to extract server from the error message
                            server_match = re.search(r"host='([^']+)'", line)
                            if server_match:
                                server = server_match.group(1)
                                if server in self.server_to_model:  # Verify it's a known server
                                    results['servers'][server]['timeouts'] += 1

                        # Check for malformed response failures - extract server from error message
                        if re.search(malformed_response_failure_pattern, line):
                            results['retries']['malformed_response_failures'] += 1
                            # Try to extract server from the error message
                            server_match = re.search(r"host='([^']+)'", line)
                            if server_match:
                                server = server_match.group(1)
                                if server in self.server_to_model:  # Verify it's a known server
                                    results['servers'][server]['malformed_response_failures'] += 1

                        # Check for other faults/errors (connection errors, server errors, etc.)
                        other_fault_patterns = [
                            r'ConnectionError',
                            r'RequestException', 
                            r'HTTPError',
                            r'ServerError',
                            r'Connection refused',
                            r'Connection reset',
                            r'Network is unreachable',
                            r'Name or service not known',
                            r'Failed to connect',
                            r'Socket error'
                        ]
                        
                        for pattern in other_fault_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                results['retries']['other_faults'] += 1
                                # Try to extract server from the error message
                                server_match = re.search(r"host='([^']+)'", line) or re.search(r"to ([^:\s]+)", line)
                                if server_match:
                                    server = server_match.group(1)
                                    if server in self.server_to_model:  # Verify it's a known server
                                        results['servers'][server]['other_faults'] += 1
                                break  # Only count once per line

        except FileNotFoundError:
            print(f"Error: Log file {self.log_file} not found.")
            sys.exit(1)

        # If no document completions found, estimate documents processed from model extractions
        if results['documents']['processed'] == 0 and model_extractions_count > 0:
            estimated_docs_processed = model_extractions_count // model_extractions_per_doc
            results['documents']['processed'] = estimated_docs_processed
            print(f"Estimated {estimated_docs_processed} documents processed from {model_extractions_count} model extractions")

        # Set the total documents (use filesystem count if available, otherwise provided value or detected from log)
        if total_docs_from_filesystem > 0:
            results['documents']['total_expected'] = total_docs_from_filesystem
            results['documents']['processed_from_filesystem'] = processed_docs_from_filesystem
            results['documents']['completed_combinations_from_filesystem'] = completed_combinations_from_filesystem
            results['documents']['computing_from_log'] = computing_docs_from_log
            results['documents']['remaining'] = max(0, total_docs_from_filesystem - processed_docs_from_filesystem - computing_docs_from_log)
            # Use filesystem count as the authoritative processed count
            results['documents']['processed'] = processed_docs_from_filesystem
        elif total_docs is None:
            results['documents']['total_expected'] = detected_total
        else:
            results['documents']['total_expected'] = total_docs

        # Add current prompt information to results
        results['documents']['current_prompt_idx'] = current_prompt_idx
        results['documents']['total_prompts_from_log'] = total_prompts_from_log
        results['documents']['current_prompt_filename'] = current_prompt_filename

        # Calculate derived metrics
        for server, metrics in results['servers'].items():
            total_requests = metrics['requests']
            if total_requests > 0:
                metrics['success_rate'] = (metrics['successful_responses'] / total_requests) * 100
                if metrics['response_times']:
                    metrics['avg_response_time'] = sum(metrics['response_times']) / len(metrics['response_times'])
                else:
                    metrics['avg_response_time'] = 0.0

        return results

    def print_report(self, results: Dict):
        """Print the comprehensive analysis report."""
        docs = results['documents']
        servers = results['servers']

        print("=== KG Builder Timing Analysis Report ===\n")

        # Document-level analysis
        print("📊 DOCUMENT PROCESSING ANALYSIS")
        print("-" * 40)

        if docs['processed'] == 0:
            print("No documents found for the specified run.")
            return

        total_time = sum(docs['times'])
        avg_time_per_doc = total_time / docs['processed']

        print(f"Documents processed: {docs['processed']}")
        print(f"Total processing time: {self.format_time(total_time)}")
        print(f"Average time per document: {avg_time_per_doc:.1f}s")
        print()

        # Time distribution
        print("Time Distribution:")
        buckets = [
            ('under_1min', 'Under 1 minute'),
            ('1_5min', '1-5 minutes'),
            ('5_10min', '5-10 minutes'),
            ('10min_1hr', '10min-1hr'),
            ('over_1hr', 'Over 1 hour')
        ]

        for key, label in buckets:
            count = docs['distribution'][key]
            pct = (count / docs['processed']) * 100 if docs['processed'] > 0 else 0
            print(f"  {label}: {count} documents ({pct:.1f}%)")

        print()

        # Progress and completion estimate
        if docs['total_expected'] > 0:
            # Use filesystem-based counts if available
            if 'processed_from_filesystem' in docs:
                processed_count = docs['processed_from_filesystem']
                computing_count = docs.get('computing_from_log', 0)
                # Remaining documents = total - processed (conservative estimate, don't subtract computing since it's unreliable)
                remaining_count = docs['total_expected'] - processed_count
                
                # Get current prompt information for more accurate calculation
                current_prompt_info = ""
                current_prompt_idx = docs.get('current_prompt_idx', 0)
                total_prompts_from_log = docs.get('total_prompts_from_log', 0)
                current_prompt_filename = docs.get('current_prompt_filename', '')
                if current_prompt_idx > 0 and total_prompts_from_log > 0:
                    remaining_prompts = total_prompts_from_log - current_prompt_idx
                    current_prompt_info = f" (currently processing prompt {current_prompt_idx}/{total_prompts_from_log}: {current_prompt_filename})"
                
                # Calculate completed combinations and remaining
                total_expected_combinations = docs['total_expected'] * self.num_models * self.num_prompts
                completed_combinations = docs.get('completed_combinations_from_filesystem', processed_count * self.num_models * self.num_prompts)
                remaining_combinations = total_expected_combinations - completed_combinations
                progress_pct = (completed_combinations / total_expected_combinations) * 100
                
                # Normalize average time per document by models and prompts to get time per combination
                avg_time_per_combination = avg_time_per_doc / (self.num_models * self.num_prompts)
                estimated_remaining_time = remaining_combinations * avg_time_per_combination

                print("🎯 PROGRESS & COMPLETION ESTIMATE")
                print("-" * 40)
                print(f"Configured prompts: {self.num_prompts}, models: {self.num_models}{current_prompt_info}")
                print(f"Total documents: {docs['total_expected']}")
                print(f"Fully processed: {processed_count} documents")
                print(f"Completed combinations: {completed_combinations} ({self.num_models} models × {self.num_prompts} prompts × {processed_count} docs)")
                print(f"Currently computing: {computing_count} documents (estimated from logs)")
                print(f"Remaining: {remaining_count} documents")
                print(f"Progress: {progress_pct:.2f}% complete ({completed_combinations}/{total_expected_combinations} model-prompt-document combinations)")
                print(f"Estimated time remaining: {self.format_time(estimated_remaining_time)} (based on {remaining_combinations} remaining combinations × {avg_time_per_combination:.1f}s avg per combination)")
                print(f"Remaining combinations: {remaining_combinations}")
                # Estimated completion = current progress (conservative, doesn't assume computing will complete)
                estimated_completion_pct = progress_pct
                print(f"Estimated completion: {estimated_completion_pct:.2f}%")
                print()
            else:
                # Fallback to old log-based calculation
                total_expected_combinations = docs['total_expected'] * self.num_prompts
                # Current combinations = documents processed (each represents 1 document-prompt combination)
                current_combinations = docs['processed']
                progress_pct = (current_combinations / total_expected_combinations) * 100
                remaining_combinations = total_expected_combinations - current_combinations
                estimated_remaining_time = remaining_combinations * avg_time_per_doc

                print("🎯 PROGRESS & COMPLETION ESTIMATE")
                print("-" * 40)
                print(f"Configured prompts: {self.num_prompts}")
                print(f"Progress: {progress_pct:.1f}% complete ({current_combinations}/{total_expected_combinations} document-prompt combinations)")
                print(f"Estimated time remaining: {self.format_time(estimated_remaining_time)}")
                print(f"Remaining combinations: {remaining_combinations} (docs × prompts)")
                print(f"Estimated completion: {progress_pct:.0f}%")
                print()

        # Server-level analysis
        print("🖥️  SERVER PERFORMANCE ANALYSIS")
        print("-" * 40)

        if not servers:
            print("No server data found.")
            return

        # Sort servers by total requests
        sorted_servers = sorted(servers.items(), key=lambda x: x[1]['requests'], reverse=True)

        for server, metrics in sorted_servers:
            model_name = self.server_to_model.get(server, 'unknown')
            endpoint_url = self.server_to_url.get(server, 'unknown')
            
            # Calculate expected requests for this server (when all prompts are done)
            expected_requests = docs['total_expected'] * self.num_prompts
            
            print(f"\n🔹 {server}:")
            print(f"   Model: {model_name}")
            print(f"   Endpoint: {endpoint_url}")
            print(f"   Requests: {metrics['requests']} / {expected_requests} expected")
            print(f"   Successful: {metrics['successful_responses']}")
            print(f"   Failed: {metrics['failed_responses']}")
            print(f"   Triples extracted: {metrics['triples_extracted']}")

            # Calculate and display per-server regeneration rate
            # Only show if we have accurate data from new logging patterns
            if metrics['queries_with_regeneration'] > 0 and metrics['requests'] > 0:
                server_regeneration_rate = (metrics['queries_with_regeneration'] / metrics['requests']) * 100
                print(f"   Regeneration Rate: {server_regeneration_rate:.1f}%")

            # Show failure breakdown if there are any failures
            has_failures = (metrics['failed_responses'] > 0 or 
                          metrics['timeouts'] > 0 or 
                          metrics['malformed_response_failures'] > 0 or 
                          metrics['other_faults'] > 0 or
                          metrics['regeneration_attempts'] > 0)
            
            if has_failures:
                print(f"   Failure breakdown:")
                if metrics['failed_responses'] > 0:
                    print(f"     • General failures: {metrics['failed_responses']}")
                if metrics['timeouts'] > 0:
                    print(f"     • Timeouts: {metrics['timeouts']}")
                if metrics['malformed_response_failures'] > 0:
                    print(f"     • Malformed responses: {metrics['malformed_response_failures']}")
                if metrics['other_faults'] > 0:
                    print(f"     • Other faults: {metrics['other_faults']}")
                if metrics['regeneration_attempts'] > 0:
                    print(f"     • Regeneration attempts: {metrics['regeneration_attempts']}")
                    if metrics['regeneration_successes'] > 0:
                        # Estimate queries with regeneration from attempts (each regeneration process = 1 query)
                        estimated_queries_with_regen = metrics['regeneration_attempts']
                        if metrics['requests'] > 0:
                            estimated_regeneration_rate = (estimated_queries_with_regen / metrics['requests']) * 100
                            print(f"       → Estimated regeneration rate: {estimated_regeneration_rate:.1f}% ({estimated_queries_with_regen}/{metrics['requests']} queries)")
                        print(f"       → {metrics['regeneration_successes']} successful regenerations")
                        avg_attempts_per_success = metrics['regeneration_attempts'] / metrics['regeneration_successes']
                        print(f"       → {avg_attempts_per_success:.1f} attempts needed per successful regeneration")

            # Show queries exceeding timeout
            if metrics['queries_exceeding_timeout'] > 0:
                print(f"   Queries exceeding timeout ({self.request_timeout}s): {metrics['queries_exceeding_timeout']}")
                if metrics['requests'] > 0:
                    timeout_rate = (metrics['queries_exceeding_timeout'] / metrics['requests']) * 100
                    print(f"   Timeout exceedance rate: {timeout_rate:.1f}%")

            if metrics['response_times']:
                print(f"   Total response time: {self.format_time(metrics['total_time'])}")
                print(f"   Average response time: {metrics['avg_response_time']:.1f}s")
                print(f"   Throughput: {len(metrics['response_times'])/metrics['total_time']:.1f} req/s")
            else:
                print("   No successful response times recorded")

        # Calculate and display overall metrics
        total_queries_all_servers = sum(metrics['requests'] for metrics in servers.values())
        total_successful_responses = sum(metrics['successful_responses'] for metrics in servers.values())
        total_triples_extracted = sum(metrics['triples_extracted'] for metrics in servers.values())
        total_queries_with_regeneration = sum(metrics['queries_with_regeneration'] for metrics in servers.values())
        total_queries_succeeded_after_regeneration = sum(metrics['queries_succeeded_after_regeneration'] for metrics in servers.values())

        if total_queries_all_servers > 0:
            overall_success_rate = (total_successful_responses / total_queries_all_servers) * 100
            regeneration_rate = (total_queries_with_regeneration / total_queries_all_servers) * 100
            
            print(f"\n📊 OVERALL PERFORMANCE METRICS")
            print("-" * 40)
            print(f"Overall Success Rate: {overall_success_rate:.1f}%")
            print(f"  = ({total_successful_responses} successful responses / {total_queries_all_servers} total queries) × 100")
            print(f"Total Triples Extracted: {total_triples_extracted:,}")
            
            if total_queries_with_regeneration > 0:
                regeneration_success_rate = (total_queries_succeeded_after_regeneration / total_queries_with_regeneration) * 100
                print(f"Regeneration Rate: {regeneration_rate:.1f}%")
                print(f"  = ({total_queries_with_regeneration} queries with regeneration / {total_queries_all_servers} total queries) × 100")
                print(f"Regeneration Success Rate: {regeneration_success_rate:.1f}%")
                print(f"  = ({total_queries_succeeded_after_regeneration} queries succeeded after regeneration / {total_queries_with_regeneration} queries with regeneration) × 100")

        # Retry and failure analysis
        retries = results['retries']
        print("\n🔄 RETRY & FAILURE ANALYSIS")
        print("-" * 40)
        print(f"Regeneration attempts: {retries['regeneration_attempts']}")
        print(f"Regeneration successes: {retries['regeneration_successes']}")
        print(f"Timeouts: {retries['timeouts']}")
        print(f"Malformed response failures: {retries['malformed_response_failures']}")
        print(f"Other faults: {retries['other_faults']}")
        
        if retries['regeneration_attempts'] > 0:
            success_rate = (retries['regeneration_successes'] / retries['regeneration_attempts']) * 100
            print(f"Regeneration success rate: {success_rate:.1f}%")
            print(f"Note: Regeneration occurs when initial LLM responses are malformed or empty.")
            print(f"      Each regeneration attempt tries different parameters (temperature, max_tokens).")
            print(f"      Success rate = (successful regenerations / total regeneration attempts) × 100")
            print(f"")
            print(f"Definitions:")
            print(f"  Overall Success Rate = (successful responses / total queries) × 100")
            print(f"  Regeneration Rate = (queries with regeneration / total queries) × 100")
            print(f"  Regeneration Success Rate = (queries that ultimately succeeded after regeneration / queries that had regeneration) × 100")


def main():
    # Change to project root directory first to load config
    script_dir = Path(__file__).parent.parent.parent.parent
    os.chdir(script_dir)
    
    # Load base config to determine default log file path
    try:
        base_config_path = Path('configs/base.yaml')
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f) or {}
        logs_dir = base_config.get('logs_dir', 'logs_multiprompt')
        default_log_file = f"{logs_dir}/kg_builder.log"
    except Exception as e:
        print(f"Warning: Could not load base config for default log file: {e}")
        default_log_file = 'logs_multiprompt/kg_builder.log'
    
    parser = argparse.ArgumentParser(description='Analyze KG Builder log files for timing and performance metrics')
    parser.add_argument('total_docs', type=int, nargs='?', default=None, help='Total number of documents expected in the run (auto-detected if not provided)')
    parser.add_argument('--log-file', default=default_log_file,
                       help=f'Path to the log file (default: {default_log_file})')
    parser.add_argument('--cumulative', action='store_true',
                       help='Analyze cumulative across all dates (default: analyze only most recent run)')

    args = parser.parse_args()

    # Make log path relative to project root
    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = Path(args.log_file)

    analyzer = TimingAnalyzer(str(log_path))
    results = analyzer.analyze_logs(args.total_docs, args.cumulative)
    analyzer.print_report(results)


if __name__ == '__main__':
    main()