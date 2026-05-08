#!/usr/bin/env python3
"""
GitHub repository scraper for neuromorphic computing projects.
Fetches repositories related to:
- PyTorch/snnTorch implementations
- HDL (Verilog, netlist) implementations
"""

import requests
import json
import time
from typing import List, Dict, Optional
from datetime import datetime
import os
from dotenv import load_dotenv


class GitHubNeuromorphicScraper:
    """Scraper for neuromorphic computing repositories on GitHub."""

    def __init__(self, token: Optional[str] = None):
        """
        Initialize the scraper.

        Args:
            token: GitHub personal access token (optional but recommended for higher rate limits)
        """
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        if token:
            self.headers["Authorization"] = f"token {token}"

        # Define search queries for different categories
        self.pytorch_queries = [
            "snnTorch",
            "neuromorphic pytorch",
            "spiking neural network pytorch",
            "SNN pytorch",
            "neuromorphic computing pytorch",
        ]

        self.hdl_queries = [
            "neuromorphic verilog",
            "neuromorphic HDL",
            "spiking neural network verilog",
            "neuromorphic netlist",
            "neuromorphic FPGA verilog",
            "SNN hardware verilog",
            "neuromorphic chip verilog",
            "neuromorphic spice",
            "neuromorphic spice netlist",
            "spiking neural network HDL",
            "neuromorphic VHDL"
        ]

    def search_repositories(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """
        Search GitHub repositories for a given query.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (None = fetch all available, up to 1000 due to GitHub API limits)

        Returns:
            List of repository information dictionaries
        """
        repos = []
        per_page = 100  # Maximum allowed by GitHub API
        page = 1
        # GitHub API limits search results to 1000 items total
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

                if response.status_code == 403:
                    # Rate limit exceeded
                    reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                    wait_time = reset_time - int(time.time()) + 10
                    if wait_time > 0:
                        print(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue

                response.raise_for_status()
                data = response.json()

                if not data.get("items"):
                    break

                for item in data["items"]:
                    if len(repos) >= max_results:
                        break

                    repo_info = {
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
                    repos.append(repo_info)

                page += 1
                time.sleep(1)  # Be nice to the API

            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for query '{query}': {e}")
                break

        return repos

    def fetch_pytorch_repos(self, max_per_query: Optional[int] = None) -> List[Dict]:
        """
        Fetch PyTorch/snnTorch related neuromorphic repositories.

        Args:
            max_per_query: Maximum results per query (None = fetch all available)

        Returns:
            List of unique repositories
        """
        print("Fetching PyTorch/snnTorch neuromorphic repositories...")
        all_repos = {}

        for query in self.pytorch_queries:
            print(f"  Searching: {query}")
            repos = self.search_repositories(query, max_per_query)
            print(f"    Found {len(repos)} repos for this query")
            for repo in repos:
                all_repos[repo["full_name"]] = repo
            time.sleep(2)  # Rate limiting

        print(f"Found {len(all_repos)} unique PyTorch repositories")
        return list(all_repos.values())

    def fetch_hdl_repos(self, max_per_query: Optional[int] = None) -> List[Dict]:
        """
        Fetch HDL (Verilog/netlist) related neuromorphic repositories.

        Args:
            max_per_query: Maximum results per query (None = fetch all available)

        Returns:
            List of unique repositories
        """
        print("Fetching HDL (Verilog/netlist) neuromorphic repositories...")
        all_repos = {}

        for query in self.hdl_queries:
            print(f"  Searching: {query}")
            repos = self.search_repositories(query, max_per_query)
            print(f"    Found {len(repos)} repos for this query")
            for repo in repos:
                all_repos[repo["full_name"]] = repo
            time.sleep(2)  # Rate limiting

        print(f"Found {len(all_repos)} unique HDL repositories")
        return list(all_repos.values())

    def fetch_all_repos(self, max_per_query: Optional[int] = None) -> Dict[str, List[Dict]]:
        """
        Fetch all neuromorphic repositories (both PyTorch and HDL).

        Args:
            max_per_query: Maximum results per query (None = fetch all available, up to 1000 per query due to GitHub API limits)

        Returns:
            Dictionary with 'pytorch' and 'hdl' keys containing repo lists
        """
        pytorch_repos = self.fetch_pytorch_repos(max_per_query)
        hdl_repos = self.fetch_hdl_repos(max_per_query)

        return {
            "pytorch": pytorch_repos,
            "hdl": hdl_repos
        }

    def save_to_json(self, data: Dict, filename: str = "neuromorphic_repos.json"):
        """
        Save repository data to JSON file.

        Args:
            data: Repository data dictionary
            filename: Output filename
        """
        output_data = {
            "fetched_at": datetime.now().isoformat(),
            "total_pytorch_repos": len(data.get("pytorch", [])),
            "total_hdl_repos": len(data.get("hdl", [])),
            "repositories": data
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nData saved to {filename}")

    def print_summary(self, data: Dict):
        """
        Print a summary of fetched repositories.

        Args:
            data: Repository data dictionary
        """
        print("\n" + "="*60)
        print("PYTORCH/SNNTORCH REPOSITORIES")
        print("="*60)

        pytorch_repos = sorted(data.get("pytorch", []), key=lambda x: x["stars"], reverse=True)
        for i, repo in enumerate(pytorch_repos[:10], 1):
            print(f"\n{i}. {repo['full_name']} ⭐ {repo['stars']}")
            print(f"   Language: {repo['language']}")
            print(f"   {repo['description'][:100] if repo['description'] else 'No description'}...")
            print(f"   URL: {repo['url']}")

        print("\n" + "="*60)
        print("HDL (VERILOG/NETLIST) REPOSITORIES")
        print("="*60)

        hdl_repos = sorted(data.get("hdl", []), key=lambda x: x["stars"], reverse=True)
        for i, repo in enumerate(hdl_repos[:10], 1):
            print(f"\n{i}. {repo['full_name']} ⭐ {repo['stars']}")
            print(f"   Language: {repo['language']}")
            print(f"   {repo['description'][:100] if repo['description'] else 'No description'}...")
            print(f"   URL: {repo['url']}")


def main():
    """Main function to run the scraper."""
    # Load environment variables from .env file
    load_dotenv()

    # Get GitHub token from environment variable (optional)
    github_token = os.environ.get("GITHUB_TOKEN")

    if not github_token:
        print("Warning: No GITHUB_TOKEN found in environment variables.")
        print("You'll have lower rate limits (60 requests/hour vs 5000/hour)")
        print("Set GITHUB_TOKEN environment variable for higher limits.\n")

    # Initialize scraper
    scraper = GitHubNeuromorphicScraper(token=github_token)

    # Fetch repositories (fetch all available - up to 1000 per query due to GitHub API limits)
    print("Starting repository fetch...\n")
    print("Note: GitHub API limits search results to 1000 items per query.")
    print("Fetching maximum available results for each search query...\n")
    repos_data = scraper.fetch_all_repos(max_per_query=None)

    # Save to JSON
    scraper.save_to_json(repos_data)

    # Print summary
    scraper.print_summary(repos_data)

    print("\n" + "="*60)
    print(f"Total PyTorch repos: {len(repos_data['pytorch'])}")
    print(f"Total HDL repos: {len(repos_data['hdl'])}")
    print("="*60)


if __name__ == "__main__":
    main()
