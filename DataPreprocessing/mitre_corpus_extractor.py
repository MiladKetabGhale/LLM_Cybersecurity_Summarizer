"""
MITRE ATT&CK Description Scraper

This script scrapes the MITRE ATT&CK Enterprise techniques page to extract
technique IDs, titles, and their description paragraphs from individual
technique pages. The output is saved as a JSONL file containing a dataset
of cybersecurity technique definitions.

Key details:
  - Targets the 'Description' section for each technique (or fallback layout)
  - Saves entries as JSONL: {"technique_id": ..., "title": ..., "description": ...}
  - Designed as a preprocessing step for downstream NLP / summarization pipelines.

Usage:
  → Produces 'mitre_descriptions.jsonl' with one JSON object per technique.
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import os

BASE_URL = "https://attack.mitre.org"
TECHNIQUES_URL = f"{BASE_URL}/techniques/enterprise/"
OUTPUT_FILE = "mitre_descriptions.jsonl"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_technique_links():
    """Scrape MITRE ATT&CK Enterprise page to extract URLs of technique entries.

    Returns:
        list[str]: Unique list of technique URLs (excluding sub-techniques).
    """
    print("Fetching technique links...")
    response = requests.get(TECHNIQUES_URL, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for a_tag in soup.select("a[href^='/techniques/T']"):
        href = a_tag["href"]
        if "/T" in href and href.count("/") == 3:  # Avoid subtechniques like /T1557/001/
            full_url = BASE_URL + href
            links.append(full_url)

    return sorted(set(links))  # Remove duplicates

def extract_description_paragraphs(technique_url):
    """Extract the first 3–4 non-empty paragraphs from the Description section of a technique page.

    Args:
        technique_url (str): URL of the MITRE ATT&CK technique page.
    Returns:
        tuple[str, list[str]]: Technique title and list of description paragraphs.
    """
    print(f"\n Scraping: {technique_url}")
    response = requests.get(technique_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    title_tag = soup.find("h1")
    title = title_tag.text.strip() if title_tag else "Unknown Title"
    paragraphs = []

    # Try Enterprise-style layout first
    desc_section = soup.find("h2", string=lambda s: s and "Description" in s)
    if desc_section:
        for sibling in desc_section.find_next_siblings():
            if sibling.name == "h2":
                break
            if sibling.name == "p":
                text = sibling.get_text(strip=True)
                if text:
                    paragraphs.append(text)
            if len(paragraphs) >= 4:
                break

    # If no paragraphs found, try ICS-style layout
    if not paragraphs:
        desc_div = soup.find("div", class_="description-body")
        if desc_div:
            for p in desc_div.find_all("p", limit=4):
                text = p.get_text(strip=True)
                if text:
                    paragraphs.append(text)

    if paragraphs:
        print(f"Extracted {len(paragraphs)} paragraph(s).")
    else:
        print("No description found.")

    return title, paragraphs

def save_entry(technique_id, title, paragraphs):
    """Append a technique entry as JSONL line to OUTPUT_FILE.

    Args:
        technique_id (str): Technique ID (e.g., T1003).
        title (str): Technique title.
        paragraphs (list[str]): List of description paragraphs.
    """

    if not paragraphs:
        return
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "technique_id": technique_id,
            "title": title,
            "description": " ".join(paragraphs)
        }, ensure_ascii=False) + "\n")

def main():
    """Main scraper pipeline: extract technique URLs, fetch descriptions, and save dataset as JSONL."""
    links = get_technique_links()
    print(f"\n Found {len(links)} technique entries. Extracting descriptions...")

    for i, link in enumerate(links, 1):
        technique_id = link.rstrip("/").split("/")[-1] # extract 'Txxxx' ID from URL
        try:
            title, paragraphs = extract_description_paragraphs(link)
            save_entry(technique_id, title, paragraphs)
            if paragraphs:
                print(f"[{i}/{len(links)}] Saved: {technique_id} - {title}")
            else:
                print(f"[{i}/{len(links)}] Skipped (no paragraphs): {technique_id} - {title}")
        except Exception as e:
            print(f"[{i}/{len(links)}] Failed: {technique_id} - {e}")
        time.sleep(1.0)                               # throttle requests (polite crawling)

    print(f"\n Done! File saved to:\n{os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()

