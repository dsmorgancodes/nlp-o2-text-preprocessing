"""
text_preprocessing_newsgroups_dsmorgan.py - Project script.

Purpose

  Load a text dataset from scikit-learn (20 Newsgroups), preprocess the text,
  compare raw and cleaned tokens, and summarize the results.

Analytical Questions

- What does the newsgroup text look like before preprocessing?
- How does tokenization change the text into analyzable units?
- How do lowercasing, punctuation removal, and stop word removal affect results?
- How much does preprocessing reduce token volume?

Run from root project folder with:

  uv run python -m nlp.text_preprocessing_newsgroups_dsmorgan
"""

# ============================================================
# Section 1. Setup and Imports (includes logging deps)
# ============================================================
import logging
from pathlib import Path
import re
from typing import Any

from datafun_toolkit.logger import get_logger, log_header, log_path
import matplotlib.pyplot as plt
import polars as pl
from sklearn.datasets import (
    fetch_20newsgroups,  # pyright: ignore[reportMissingModuleSource]
)

print("Imports complete.")

# ============================================================
# Configure Logging (script execution only)
# ============================================================

LOG: logging.Logger = get_logger("CI", level="DEBUG")

ROOT_PATH: Path = Path.cwd()
DATA_PATH: Path = ROOT_PATH / "data"
NOTEBOOKS_PATH: Path = ROOT_PATH / "notebooks"
SCRIPTS_PATH: Path = ROOT_PATH / "scripts"

log_header(LOG, "NLP")
LOG.info("START script.....")

log_path(LOG, "ROOT_PATH", ROOT_PATH)
log_path(LOG, "DATA_PATH", DATA_PATH)
log_path(LOG, "NOTEBOOKS_PATH", NOTEBOOKS_PATH)
log_path(LOG, "SCRIPTS_PATH", SCRIPTS_PATH)

# ============================================================
# Section 2. Load the Newsgroups Dataset
# ============================================================

categories: list[str] = [
    "sci.space",
    "rec.sport.baseball",
    "talk.politics.misc",
]

newsgroups: Any = fetch_20newsgroups(
    subset="train",
    categories=categories,
    remove=("headers", "footers", "quotes"),
)

# Clean up records and keep non-empty documents only.
text_list: list[str] = [
    " ".join(doc.split())
    for doc in newsgroups.data
    if isinstance(doc, str) and doc.strip()
]

print("Data loaded successfully.")
print(f"Loaded {len(text_list):,} text records.")

target_names: list[str] = list(newsgroups.target_names)
targets: list[int] = list(newsgroups.target)

category_df: pl.DataFrame = (
    pl.DataFrame(
        {
            "target": targets,
        }
    )
    .with_columns(
        pl.col("target").replace_strict(dict(enumerate(target_names))).alias("category")
    )
    .group_by("category")
    .len()
    .sort("len", descending=True)
)

print("Records by category:")
print(category_df)

raw_text: str = " ".join(text_list)

print(f"Raw text length: {len(raw_text):,} characters")
print("First 500 characters of raw text:")
print(raw_text[:500])

# ============================================================
# Section 3. Inspect the Raw Text
# ============================================================

print("First 5 text records:")
for line in text_list[:5]:
    print("-", line[:160])

print(f"\nLoaded {len(text_list):,} text records.")
print(f"Raw text length: {len(raw_text):,} characters")

print("\nFirst 500 characters of combined text:")
print(raw_text[:500])

# ============================================================
# Section 4. Tokenize the Raw Text
# ============================================================

raw_tokens: list[str] = raw_text.split()
count_of_raw_tokens: int = len(raw_tokens)

print("First 20 raw tokens:")
print(raw_tokens[:20])
print(f"Total raw tokens: {count_of_raw_tokens:,}")

# ============================================================
# Section 5. Normalize the Text
# ============================================================

lower_text: str = raw_text.lower()

print("First 500 characters of lowercase text:")
print(lower_text[:500])

# ============================================================
# Section 6. Remove Punctuation and Tokenize Again
# ============================================================

no_punct_text: str = re.sub(r"[^a-z0-9\s]", " ", lower_text)

tokens_no_punct: list[str] = no_punct_text.split()
count_of_tokens_no_punct: int = len(tokens_no_punct)

print("First 20 tokens after lowercasing and punctuation removal:")
print(tokens_no_punct[:20])
print(f"Total tokens after punctuation removal: {count_of_tokens_no_punct:,}")

# ============================================================
# Section 7. Remove Stop Words
# ============================================================

STOP_WORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}

clean_tokens: list[str] = [
    token for token in tokens_no_punct if len(token) > 2 and token not in STOP_WORDS
]

count_of_clean_tokens: int = len(clean_tokens)

vocab_reduction_pct: float = (
    (count_of_raw_tokens - count_of_clean_tokens) / count_of_raw_tokens * 100
    if count_of_raw_tokens > 0
    else 0.0
)

print("First 20 cleaned tokens:")
print(clean_tokens[:20])
print(f"Total cleaned tokens: {count_of_clean_tokens:,}")

# ============================================================
# Section 8. Build a Before/After Summary Table
# ============================================================

summary_df: pl.DataFrame = pl.DataFrame(
    {
        "stage": [
            "raw tokens",
            "after punctuation removal",
            "after stop word removal",
        ],
        "count": [
            count_of_raw_tokens,
            count_of_tokens_no_punct,
            count_of_clean_tokens,
        ],
        "metric": [
            "",
            "",
            f"{vocab_reduction_pct:.1f}% reduction from raw",
        ],
    }
)

print("Preprocessing summary:")
print(summary_df)
print(f"Vocabulary reduction: {vocab_reduction_pct:.1f}%")

# ============================================================
# Section 9. Build a Frequency Table with Polars
# ============================================================

token_df: pl.DataFrame = pl.DataFrame({"token": clean_tokens})

freq_df: pl.DataFrame = token_df.group_by("token").len().sort("len", descending=True)

print("Top 20 most frequent cleaned tokens:")
print(freq_df.head(20))

# ============================================================
# Section 10. Build a "Most Frequent Cleaned Tokens" Bar Chart
# ============================================================

top_df: pl.DataFrame = freq_df.head(10)

plt.figure(figsize=(10, 5))
plt.bar(top_df["token"], top_df["len"])

ax = plt.gca()
ax.tick_params(axis="x", labelrotation=45)

plt.title("Most Frequent Cleaned Tokens (20 Newsgroups)")
plt.xlabel("Token")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ============================================================
# Section 11. Compare Raw vs Clean Token Counts
# ============================================================

plt.figure(figsize=(8, 5))
plt.bar(summary_df["stage"], summary_df["count"])

ax = plt.gca()
ax.tick_params(axis="x", labelrotation=20)

plt.title(
    f"Token Counts Across Preprocessing Stages ({vocab_reduction_pct:.1f}% reduction)"
)
plt.xlabel("Stage")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ============================================================
# LOG THE END (only in the script)
# ============================================================

LOG.info("========================")
LOG.info("Pipeline executed successfully!")
LOG.info("========================")
LOG.info("END main()")
