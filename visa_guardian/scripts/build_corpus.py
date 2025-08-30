# scripts/build_corpus.py
import json, pathlib

BASE = pathlib.Path(__file__).resolve().parents[1]
CLEAN_DIR = BASE / "data" / "cleaned"
LAWS_DIR = BASE / "data" / "laws"
LAWS_DIR.mkdir(parents=True, exist_ok=True)
OUT = LAWS_DIR / "clauses.jsonl"

count = 0
with open(OUT, "w", encoding="utf-8") as out:
    for p in CLEAN_DIR.glob("*.jsonl"):
        for line in p.open(encoding="utf-8"):
            out.write(line)
            count += 1

print(f"Wrote {count} clauses → {OUT}")
