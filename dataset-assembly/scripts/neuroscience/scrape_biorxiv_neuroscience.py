#!/usr/bin/env python3
"""
Script to scrape neuroscience papers from bioRxiv.
Fetches DOI, title, abstract, and PDF URL for all neuroscience papers.
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict
import os

class BioRxivScraper:
    def __init__(self, output_dir='../data/biorxiv_papers'):
        self.base_url = "https://api.biorxiv.org/details"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def fetch_papers_by_date_range(self, start_date: str, end_date: str,
                                   server: str = "biorxiv",
                                   category: str = "neuroscience") -> List[Dict]:
        """
        Fetch papers from bioRxiv API for a specific date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            server: 'biorxiv' or 'medrxiv'
            category: Category to filter by (e.g., 'neuroscience')

        Returns:
            List of paper dictionaries
        """
        cursor = 0
        all_papers = []

        while True:
            # Use query parameter for category filtering
            url = f"{self.base_url}/{server}/{start_date}/{end_date}/{cursor}"
            params = {'category': category} if category else {}

            print(f"Fetching: {url}?category={category} (cursor={cursor})")

            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if 'collection' not in data or not data['collection']:
                    print(f"No more papers found for {start_date} to {end_date}")
                    break

                papers = data['collection']
                all_papers.extend(papers)

                print(f"Retrieved {len(papers)} papers (cursor: {cursor}, total: {len(all_papers)})")

                # Check if we've retrieved all papers
                total_count = int(data.get('messages', [{}])[0].get('total', 0))
                if len(all_papers) >= total_count or len(papers) == 0:
                    break

                cursor += len(papers)

                # Be respectful to the API
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching data: {e}")
                break
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                break

        return all_papers

    def extract_paper_info(self, paper: Dict) -> Dict:
        """
        Extract relevant information from a paper entry.

        Args:
            paper: Raw paper dictionary from API

        Returns:
            Cleaned dictionary with DOI, title, abstract, and PDF URL
        """
        doi = paper.get('doi', '')

        return {
            'doi': doi,
            'title': paper.get('title', ''),
            'abstract': paper.get('abstract', ''),
            'pdf_url': f"https://www.biorxiv.org/content/{doi}v{paper.get('version', '1')}.full.pdf",
            'authors': paper.get('authors', ''),
            'date': paper.get('date', ''),
            'category': paper.get('category', ''),
            'version': paper.get('version', ''),
        }

    def scrape_all_neuroscience_papers(self, start_year: int = 2013,
                                      end_date: str = None) -> List[Dict]:
        """
        Scrape all neuroscience papers from bioRxiv.

        Args:
            start_year: Year to start scraping from (bioRxiv launched in 2013)
            end_date: End date in YYYY-MM-DD format (defaults to today)

        Returns:
            List of all papers with extracted information
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        all_papers = []
        current_start = datetime(start_year, 1, 1)
        final_end = datetime.strptime(end_date, '%Y-%m-%d')

        # Fetch papers in 6-month chunks to avoid API limits
        while current_start < final_end:
            current_end = min(current_start + timedelta(days=180), final_end)

            start_str = current_start.strftime('%Y-%m-%d')
            end_str = current_end.strftime('%Y-%m-%d')

            print(f"\n=== Fetching papers from {start_str} to {end_str} ===")

            papers = self.fetch_papers_by_date_range(start_str, end_str)

            for paper in papers:
                paper_info = self.extract_paper_info(paper)
                all_papers.append(paper_info)

            current_start = current_end + timedelta(days=1)

            # Save intermediate results
            self.save_papers(all_papers,
                           f'neuroscience_papers_partial_{len(all_papers)}.json')

        return all_papers

    def save_papers(self, papers: List[Dict], filename: str = 'neuroscience_papers.json'):
        """
        Save papers to JSON file.

        Args:
            papers: List of paper dictionaries
            filename: Output filename
        """
        output_path = os.path.join(self.output_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)

        print(f"\nSaved {len(papers)} papers to {output_path}")

    def save_as_csv(self, papers: List[Dict], filename: str = 'neuroscience_papers.csv'):
        """
        Save papers to CSV file.

        Args:
            papers: List of paper dictionaries
            filename: Output filename
        """
        import csv

        output_path = os.path.join(self.output_dir, filename)

        if not papers:
            print("No papers to save")
            return

        fieldnames = ['doi', 'title', 'abstract', 'pdf_url', 'authors', 'date', 'category', 'version']

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(papers)

        print(f"Saved {len(papers)} papers to {output_path}")


def main():
    """Main function to run the scraper."""
    scraper = BioRxivScraper()

    print("Starting bioRxiv neuroscience paper scraper...")
    print("This will fetch all neuroscience papers from 2013 to present.")
    print("This may take a while depending on the number of papers.\n")

    # Scrape all papers
    papers = scraper.scrape_all_neuroscience_papers(start_year=2013)

    print(f"\n=== Scraping Complete ===")
    print(f"Total papers collected: {len(papers)}")

    # Save to JSON
    scraper.save_papers(papers, 'neuroscience_papers_complete.json')

    # Save to CSV for easier viewing
    scraper.save_as_csv(papers, 'neuroscience_papers_complete.csv')

    # Print some statistics
    if papers:
        print(f"\nFirst paper date: {min(p['date'] for p in papers if p['date'])}")
        print(f"Last paper date: {max(p['date'] for p in papers if p['date'])}")
        print(f"\nSample paper:")
        print(f"  DOI: {papers[0]['doi']}")
        print(f"  Title: {papers[0]['title'][:100]}...")
        print(f"  PDF URL: {papers[0]['pdf_url']}")


if __name__ == "__main__":
    main()
