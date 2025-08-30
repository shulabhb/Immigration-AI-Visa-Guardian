import json, pathlib, re
from sentence_transformers import SentenceTransformer
import numpy as np

BASE = pathlib.Path(__file__).resolve().parents[1]
INP = BASE / "data" / "laws" / "clauses.jsonl"
OUT = BASE / "data" / "laws" / "clauses_dedup.jsonl"

print(f"Reading: {INP}")
texts, metas = [], []
with open(INP, encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        texts.append(rec["text"])
        metas.append(rec)
print(f"Loaded {len(texts)} clauses")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(texts, normalize_embeddings=True, batch_size=64)
emb = np.asarray(emb, dtype="float32")

keep = []
kept_idx = []
for i, e in enumerate(emb):
    if not kept_idx:
        keep.append(True); kept_idx.append(i); continue
    sims = emb[kept_idx] @ e
    if float(np.max(sims)) > 0.95:
        keep.append(False)
    else:
        keep.append(True); kept_idx.append(i)

kept = 0
with open(OUT, "w", encoding="utf-8") as g:
    for k, rec in zip(keep, metas):
        if not k:
            continue
        raw_text = rec.get("text", "")
        t = raw_text.lower()
        tags = set(rec.get("visa_tags", []) or [])
        source_id = rec.get("source_id") or ""
        # Force-tag F2 for known F-2 specific sources
        f2_slugs = {
            "uscis_f_chapter9",
            "ecfr_8_214_2_f2",
            "study_states_dependents",
            "study_states_f2_study",
            "ice_dependents_overview",
            "state_student_visa",
        }
        if source_id in f2_slugs:
            tags.add("F2")
        # Strong F-2/dependents signals
        f2_patterns = [
            r"\bF-2\b", r"\bF2\b", r"\bspouse\b", r"\bdependent(s)?\b", r"minor child(ren)?",
            r"may not (accept|engage in) employment", r"part[- ]time study", r"change of status"
        ]
        if any(re.search(p, raw_text, flags=re.IGNORECASE) for p in f2_patterns):
            tags.add("F2")
        # Infer visa labels when obvious in text
        for pat, vtag in [
            ("f-1", "F1"), ("f1", "F1"), ("f-2", "F2"), ("f2", "F2"),
            ("j-1", "J1"), ("j1", "J1"), ("j-2", "J2"), ("j2", "J2"),
            ("h-1b", "H1B"), ("h1b", "H1B"), ("h-4", "H4"), ("h4", "H4"),
        ]:
            if pat in t:
                tags.add(vtag)
        for kw, tag in [
            ("opt", "OPT"), ("cpt", "CPT"), ("on-campus", "on-campus"),
            ("grace period", "grace-period"), ("portability", "portability"),
            ("unlawful presence", "unlawful-presence"), ("dependent", "dependent"), ("spouse", "spouse"), ("child", "child"), ("children", "child")
        ]:
            if kw in t:
                tags.add(tag)
        rec["visa_tags"] = sorted(tags)
        g.write(json.dumps(rec, ensure_ascii=False) + "\n")
        kept += 1

print(f"Kept {kept} / {len(keep)} -> {OUT}")
