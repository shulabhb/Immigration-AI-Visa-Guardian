# scripts/convert_qna_csv.py
import csv, json, sys, pathlib

if len(sys.argv) < 3:
    print("Usage: python scripts/convert_qna_csv.py <input_csv> <output_jsonl>")
    sys.exit(1)

BASE = pathlib.Path(__file__).resolve().parents[1]
IN = pathlib.Path(sys.argv[1])
if not IN.is_absolute():
    IN = BASE / IN
OUT = pathlib.Path(sys.argv[2])
if not OUT.is_absolute():
    OUT = BASE / OUT

with IN.open(newline="", encoding="utf-8") as f, OUT.open("w", encoding="utf-8") as g:
    r = csv.DictReader(f)
    for row in r:
        if not row.get("question", "").strip():
            continue
        rec = {
            "question": row.get("question", "").strip(),
            "answer": (row.get("answer", "") or "").strip() or None,
            "law_ref": (row.get("law_ref", "") or "").strip() or None,
            "visa_type": (row.get("visa_type", "") or "").strip() or None,
            "risk_level": (row.get("risk_level", "") or "").strip() or None,
            "notes": (row.get("notes", "") or "").strip() or None,
        }
        g.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Wrote → {OUT}")
