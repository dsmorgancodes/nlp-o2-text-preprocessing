# Newsgroups Preprocessing Summary

## What Was Completed

- Added a new script pipeline for scikit-learn's 20 Newsgroups dataset:
  - `src/nlp/text_preprocessing_newsgroups_dsmorgan.py`
- Added a matching notebook pipeline with the same section structure:
  - `notebooks/text_preprocessing_newsgroups_dsmorgan.ipynb`
- Added a new dependency in `pyproject.toml`:
  - `scikit-learn`

## Dataset and Scope

This new workflow uses the built-in scikit-learn 20 Newsgroups dataset, so no custom file is required.

Included categories:

- `sci.space`
- `rec.sport.baseball`
- `talk.politics.misc`

The loader removes headers, footers, and quoted text to focus analysis on message body content.

## Script Walkthrough

The script follows the same instructional pattern as the existing preprocessing scripts.

1. **Setup and Imports**
   - Imports core libraries: `re`, `pathlib`, `polars`, `matplotlib`, and `fetch_20newsgroups`.
   - Configures logging with `datafun_toolkit` utilities.

2. **Load Dataset**
   - Fetches the train subset for selected categories.
   - Cleans each document into single-line records.
   - Prints record count and category distribution table.

3. **Inspect Raw Text**
   - Prints a few sample records and a preview of combined raw text.

4. **Tokenize Raw Text**
   - Splits combined text using whitespace and counts raw tokens.

5. **Normalize Text**
   - Converts all text to lowercase.

6. **Remove Punctuation and Tokenize Again**
   - Uses regex to remove punctuation/special characters.
   - Re-tokenizes and counts tokens.

7. **Remove Stop Words**
   - Filters out short/common terms with a stop-word list.
   - Computes `vocab_reduction_pct` as a technical metric.

8. **Build Summary Table**
   - Creates `summary_df` with stage counts and reduction metric.

9. **Build Frequency Table**
   - Builds token frequency table with Polars.

10. **Top Token Chart**
   - Plots top 10 cleaned tokens.

11. **Stage Comparison Chart**
   - Plots token counts across preprocessing stages.
   - Includes reduction percentage in chart title.

## How to Run

From the project root:

```shell
uv sync --extra dev --extra docs --upgrade
uv run python -m nlp.text_preprocessing_newsgroups_dsmorgan
```

Then open and run:

- `notebooks/text_preprocessing_newsgroups_dsmorgan.ipynb`

## Expected Outputs

- Console summaries for token counts and frequencies
- Category distribution table
- Two charts:
  - Most Frequent Cleaned Tokens (20 Newsgroups)
  - Token Counts Across Preprocessing Stages

## Notes

- The first run may download the 20 Newsgroups dataset over the network.
- If download fails (offline environment), rerun when internet access is available.
