# scripts/make_qna_jsonl.py
import csv, json, pathlib

BASE = pathlib.Path(__file__).resolve().parents[1]
IN = BASE / "data" / "qna" / "qna_seed.csv"
OUT = BASE / "data" / "qna" / "qna.jsonl"

with open(IN, newline="", encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as g:
    r = csv.DictReader(f)
    for row in r:
        rec = {
            "question": row["question"].strip(),
            "answer": row["answer"].strip(),
            "law_ref": row["law_ref"].strip(),
            "visa_type": row["visa_type"].strip(),
            "risk_level": row.get("risk_level", "").strip() or None,
            "notes": row.get("notes", "").strip() or None,
        }
        g.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Wrote → {OUT}")
