# scripts/debug_f2_misses.py
import json, pathlib, re, faiss
from sentence_transformers import SentenceTransformer

BASE = pathlib.Path(__file__).resolve().parents[1]
LAWS = BASE / "data" / "laws"
IDX = LAWS / "faiss_F2.index"  # per-visa index
META = LAWS / "faiss_F2_meta.json"
QNA = BASE / "data" / "qna" / "f2_qna_50.clean.jsonl"

meta = json.load(open(META, encoding="utf-8"))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index(str(IDX))

def looks_like(doc, law_ref):
    if not law_ref: return False
    hay = " ".join([doc.get("title",""), doc.get("section_hint",""), doc.get("url","")]).lower()
    ref = law_ref.lower()
    toks = [t for t in re.split(r'[\s,;:()]+', ref) if t]
    return (ref in hay) or sum(t in hay for t in toks) >= 2

def search(q, k=5):
    qv = model.encode([q], normalize_embeddings=True).astype("float32")
    s, ids = index.search(qv, k)
    return [(int(i), float(sv)) for i, sv in zip(ids[0], s[0]) if i >= 0]

misses = []
with open(QNA, encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        if (ex.get("visa_type") or "").upper() != "F2":
            continue
        hits = search(ex["question"], 5)
        ok = any(looks_like(meta[i], ex.get("law_ref","")) for i,_ in hits)
        if not ok:
            misses.append((ex, hits))

for ex, hits in misses[:10]:
    print("\nQ:", ex["question"]) 
    print("law_ref:", ex.get("law_ref",""))
    for rank,(i,score) in enumerate(hits,1):
        d = meta[i]
        print(f"  [{rank}] {score:.3f} | {d.get('title')} | {d.get('section_hint')}")
        print("      ", d.get("url"))

print(f"\nTotal misses: {len(misses)} / {sum(1 for _ in open(QNA, encoding='utf-8'))}")
