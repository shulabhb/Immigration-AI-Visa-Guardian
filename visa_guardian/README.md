# visa_guardian

This repo scrapes authoritative immigration sources into chunked JSONL for retrieval/training and provides a tiny Q&A curation path.

## Setup

```bash
cd visa_guardian
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Scrape

```bash
python scripts/scrape_sources.py
```

Outputs per-source JSONL files in `data/cleaned/` and raw HTML in `data/raw/`.

## Build merged corpus

```bash
python scripts/build_corpus.py
```

Creates `data/laws/clauses.jsonl`.

## Q&A conversion (optional)

Edit `data/qna/qna_seed.csv`, then run:

```bash
python scripts/make_qna_jsonl.py
```

Produces `data/qna/qna.jsonl`.
