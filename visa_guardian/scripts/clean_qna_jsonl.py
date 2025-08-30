# scripts/clean_qna_jsonl.py
import json, sys, pathlib

path = pathlib.Path(sys.argv[1])
seen = set()
clean = []
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        key = (
            obj.get("question", "").strip(),
            (obj.get("law_ref", "") or "").strip(),
            (obj.get("visa_type", "") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        clean.append(obj)

out = path.with_suffix(".clean.jsonl")
with open(out, "w", encoding="utf-8") as g:
    for obj in clean:
        g.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Input:  {path} ({len(seen)} unique)")
print(f"Output: {out} ({len(clean)} kept)")
