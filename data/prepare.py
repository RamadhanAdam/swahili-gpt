"""
data/prepare.py
Downloads the Swahili News dataset from HuggingFace and saves it as a single
plain-text file (data/swahili.txt) ready for character-level language model training.

Dataset: https://huggingface.co/datasets/swahili_news
~31,000 news articles scraped from Tanzanian news platforms.
License: CC BY 4.0

Usage:
    pip install datasets
    python data/prepare.py
"""

from datasets import load_dataset
from pathlib import Path

OUTPUT = Path(__file__).parent / "swahili.txt"

def main():
    print("Downloading Swahili News dataset from HuggingFace...")
    dataset = load_dataset("swahili_news")

    # Print columns so we always know what we're working with
    first_split = list(dataset.keys())[0]
    print(f"Columns: {dataset[first_split].column_names}")

    articles = []
    for split in dataset:
        for row in dataset[split]:
            # dataset uses 'text' as the main field
            text = (row.get("text") or row.get("content") or "").strip()
            if text:
                articles.append(text)

    full_text = "\n\n---\n\n".join(articles)

    OUTPUT.write_text(full_text, encoding="utf-8")

    chars = len(full_text)
    words = len(full_text.split())
    print(f"Saved {len(articles):,} articles → {OUTPUT}")
    print(f"  {chars:,} characters  |  {words:,} words")

if __name__ == "__main__":
    main()