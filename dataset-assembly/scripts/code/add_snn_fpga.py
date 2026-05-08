#!/usr/bin/env python3
"""
Extends the `hdl` category in neuromorphic_repos.json with FPGA/HDL
repositories that implement spiking neural networks, clones new ones into
data/github_repos/hdl/, and rebuilds code_index.jsonl.

Run from repo root OR dataset-assembly/scripts/code/:
    python dataset-assembly/scripts/code/add_snn_fpga.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv


# ── Config ─────────────────────────────────────────────────────────────────────

CATEGORIES = {
    "hdl": {
        "pinned": [],
        "queries": [
            "snn fpga",
            "spiking neural network fpga",
            "spiking neural network verilog",
            "spiking neural network vhdl",
            "neuromorphic fpga",
            "LIF neuron fpga verilog",
            "leaky integrate-and-fire verilog",
            "spike neuromorphic FPGA implementation",
        ],
        "max_per_query": 20,
    },
}


# ── GitHub Fetcher ─────────────────────────────────────────────────────────────

class GitHubRepoFetcher:
    """Fetches GitHub repositories by name or search query."""

    def __init__(self, token: Optional[str] = None):
        self.base_url = "https://api.github.com"
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            self.headers["Authorization"] = f"token {token}"

    def _handle_rate_limit(self, response: requests.Response) -> bool:
        if response.status_code == 403:
            reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
            wait_time = reset_time - int(time.time()) + 10
            if wait_time > 0:
                print(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            return True
        return False

    def _parse_repo(self, item: Dict) -> Dict:
        return {
            "name":        item["name"],
            "full_name":   item["full_name"],
            "description": item.get("description", ""),
            "url":         item["html_url"],
            "stars":       item["stargazers_count"],
            "forks":       item["forks_count"],
            "language":    item.get("language", "Unknown"),
            "created_at":  item["created_at"],
            "updated_at":  item["updated_at"],
            "topics":      item.get("topics", []),
            "license":     item.get("license", {}).get("name", "None") if item.get("license") else "None",
        }

    def fetch_repo(self, full_name: str) -> Optional[Dict]:
        url = f"{self.base_url}/repos/{full_name}"
        try:
            response = requests.get(url, headers=self.headers)
            if self._handle_rate_limit(response):
                return self.fetch_repo(full_name)
            response.raise_for_status()
            return self._parse_repo(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error fetching repo '{full_name}': {e}")
            return None

    def search_repositories(self, query: str, max_results: int = 20) -> List[Dict]:
        repos: List[Dict] = []
        per_page = min(max_results, 100)
        page = 1

        while len(repos) < max_results:
            url = f"{self.base_url}/search/repositories"
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            }
            try:
                response = requests.get(url, headers=self.headers, params=params)
                if self._handle_rate_limit(response):
                    continue
                response.raise_for_status()
                data = response.json()

                if not data.get("items"):
                    break

                for item in data["items"]:
                    if len(repos) >= max_results:
                        break
                    repos.append(self._parse_repo(item))

                page += 1
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                print(f"Error searching for '{query}': {e}")
                break

        return repos


# ── Repo management ────────────────────────────────────────────────────────────

def merge_repos(existing: List[Dict], new: List[Dict]) -> List[Dict]:
    """Merge new repos into existing list, deduplicating by full_name."""
    seen = {repo["full_name"] for repo in existing}
    merged = list(existing)
    added = 0
    for repo in new:
        if repo["full_name"] not in seen:
            merged.append(repo)
            seen.add(repo["full_name"])
            added += 1
            print(f"    Added: {repo['full_name']}")
        else:
            print(f"    Already exists: {repo['full_name']}")
    print(f"  {added} new repos added")
    return merged


def clone_repos(repos: List[Dict], dest_dir: Path) -> None:
    """Clone repos into dest_dir/{owner_repo}/ using a shallow clone."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    for repo in repos:
        full_name = repo["full_name"]
        clone_url = repo["url"] + ".git"
        dir_name = full_name.replace("/", "_")
        target = dest_dir / dir_name

        if target.exists():
            print(f"  Already cloned: {full_name}")
            continue

        print(f"  Cloning: {full_name} -> {target.name}")
        result = subprocess.run(
            ["git", "clone", "--depth=1", clone_url, str(target)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"    OK")
        else:
            print(f"    FAILED: {result.stderr.strip()}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()

    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("Warning: No GITHUB_TOKEN found. Rate limits will be low.\n")

    scripts_dir = Path(__file__).resolve().parent.parent
    repo_root   = scripts_dir.parents[2]
    repos_dir   = repo_root / "data" / "github_repos"
    repos_json  = repos_dir / "neuromorphic_repos.json"
    indexer     = Path(__file__).resolve().parent / "index_code_files.py"

    print(f"Repo root : {repo_root}")
    print(f"JSON file : {repos_json}\n")

    with open(repos_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    repos = data["repositories"]

    fetcher = GitHubRepoFetcher(token=github_token)
    all_new: Dict[str, List[Dict]] = {}

    for category, cfg in CATEGORIES.items():
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"{'='*60}")

        cat_repos: Dict[str, Dict] = {}

        if cfg["pinned"]:
            print("\nFetching pinned repos...")
            for full_name in cfg["pinned"]:
                print(f"  Fetching: {full_name}")
                repo = fetcher.fetch_repo(full_name)
                if repo:
                    cat_repos[repo["full_name"]] = repo
                    print(f"    -> {repo['name']} ⭐ {repo['stars']}")
                else:
                    print(f"    -> FAILED")
                time.sleep(1)

        max_per_query = cfg.get("max_per_query", 20)
        print(f"\nSearching for related repos (cap {max_per_query} per query)...")
        for query in cfg["queries"]:
            print(f"  Query: '{query}'")
            results = fetcher.search_repositories(query, max_results=max_per_query)
            print(f"    Found {len(results)} repos")
            for repo in results:
                cat_repos[repo["full_name"]] = repo
            time.sleep(2)

        print(f"\nTotal unique {category} repos found: {len(cat_repos)}")

        if category not in repos:
            repos[category] = []

        print(f"\nMerging into '{category}'...")
        repos[category] = merge_repos(repos[category], list(cat_repos.values()))
        all_new[category] = list(cat_repos.values())

    data["fetched_at"] = datetime.now().isoformat()
    for key in repos:
        data[f"total_{key}_repos"] = len(repos[key])

    with open(repos_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved updated neuromorphic_repos.json")
    for key, lst in repos.items():
        print(f"  {key}: {len(lst)} repos")

    # Clone new repos into hdl/
    for category, cat_repos in all_new.items():
        if not cat_repos:
            continue
        dest = repos_dir / category
        print(f"\n{'='*60}")
        print(f"Cloning new {category} repos into {dest}")
        print(f"{'='*60}")
        clone_repos(cat_repos, dest)

    # Rebuild code_index.jsonl
    print(f"\n{'='*60}")
    print("Rebuilding code_index.jsonl ...")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(indexer)])
    if result.returncode != 0:
        print("WARNING: Indexer exited with non-zero status.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
