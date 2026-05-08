#!/usr/bin/env python3
"""
Adds Fugu and SuperNeuro repositories to the neuromorphic_repos.json
produced by github_scrape.py.

Fetches:
- The original Fugu and SuperNeuro repos directly
- Any repositories that use or reference Fugu/SuperNeuro via GitHub search
"""

import requests
import json
import time
import os
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv


CATEGORIES = {
    "fugu": {
        "pinned": [
            "sandialabs/Fugu",
        ],
        "queries": [
            "Fugu spiking neural network",
            "Fugu neuromorphic",
            "sandialabs Fugu",
        ],
    },
    "superneuro": {
        "pinned": [
            "ORNL/superneuro",
            "ORNL/superneuromat",
            "ORNL/superneuroabm",
        ],
        "queries": [
            "SuperNeuro neuromorphic",
            "SuperNeuro spiking",
            "superneuromat",
            "superneuroabm",
            "ORNL superneuro",
        ],
    },
}


class GitHubRepoFetcher:
    """Fetches GitHub repositories by name or search query."""

    def __init__(self, token: Optional[str] = None):
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        if token:
            self.headers["Authorization"] = f"token {token}"

    def _handle_rate_limit(self, response: requests.Response):
        """Wait if rate limited, return True if we should retry."""
        if response.status_code == 403:
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            wait_time = reset_time - int(time.time()) + 10
            if wait_time > 0:
                print(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            return True
        return False

    def _parse_repo(self, item: Dict) -> Dict:
        """Parse a GitHub API repo response into our standard format."""
        return {
            "name": item["name"],
            "full_name": item["full_name"],
            "description": item.get("description", ""),
            "url": item["html_url"],
            "stars": item["stargazers_count"],
            "forks": item["forks_count"],
            "language": item.get("language", "Unknown"),
            "created_at": item["created_at"],
            "updated_at": item["updated_at"],
            "topics": item.get("topics", []),
            "license": item.get("license", {}).get("name", "None") if item.get("license") else "None"
        }

    def fetch_repo(self, full_name: str) -> Optional[Dict]:
        """Fetch metadata for a single repository by full name."""
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

    def search_repositories(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """
        Search GitHub repositories for a given query.

        Args:
            query: Search query string
            max_results: Maximum number of results (None = all available, up to 1000)

        Returns:
            List of repository info dicts
        """
        repos = []
        per_page = 100
        page = 1
        max_results = max_results or 1000

        while len(repos) < max_results:
            url = f"{self.base_url}/search/repositories"
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page
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


def load_existing_data(filepath: str) -> Dict:
    """Load existing neuromorphic_repos.json."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


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


def main():
    load_dotenv()

    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("Warning: No GITHUB_TOKEN found. Rate limits will be low.\n")

    input_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "github_repos", "neuromorphic_repos.json"
    )
    input_file = os.path.normpath(input_file)

    print(f"Loading existing data from {input_file}")
    data = load_existing_data(input_file)
    repos = data["repositories"]

    fetcher = GitHubRepoFetcher(token=github_token)

    for category, cfg in CATEGORIES.items():
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"{'='*60}")

        cat_repos = {}

        # 1) Fetch pinned repos directly
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

        # 2) Search for repos that use/reference this framework
        print("\nSearching for related repos...")
        for query in cfg["queries"]:
            print(f"  Searching: {query}")
            results = fetcher.search_repositories(query)
            print(f"    Found {len(results)} repos")
            for repo in results:
                cat_repos[repo["full_name"]] = repo
            time.sleep(2)

        print(f"\nTotal unique {category} repos found: {len(cat_repos)}")

        # 3) Merge into category
        if category not in repos:
            repos[category] = []

        print(f"\nMerging into '{category}'...")
        repos[category] = merge_repos(repos[category], list(cat_repos.values()))

    # Update metadata
    data["fetched_at"] = datetime.now().isoformat()
    data["total_pytorch_repos"] = len(repos.get("pytorch", []))
    data["total_hdl_repos"] = len(repos.get("hdl", []))
    data["total_fugu_repos"] = len(repos.get("fugu", []))
    data["total_superneuro_repos"] = len(repos.get("superneuro", []))

    # Save
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved updated data to {input_file}")
    print(f"  PyTorch repos: {data['total_pytorch_repos']}")
    print(f"  HDL repos: {data['total_hdl_repos']}")
    print(f"  Fugu repos: {data['total_fugu_repos']}")
    print(f"  SuperNeuro repos: {data['total_superneuro_repos']}")


if __name__ == "__main__":
    main()
