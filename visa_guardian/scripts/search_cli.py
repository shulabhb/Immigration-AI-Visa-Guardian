import json, pathlib, sys
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

BASE = pathlib.Path(__file__).resolve().parents[1]
LAWS = BASE / "data" / "laws"
IDX = LAWS / "faiss.index"
META = LAWS / "faiss_meta.json"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index(str(IDX))
meta = json.load(open(META, encoding="utf-8"))

def search(q, k=5):
    qv = model.encode([q], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(qv, k)
    out = []
    for rank in range(k):
        i = int(ids[0][rank])
        if i < 0: continue
        out.append({
            "rank": rank+1,
            "score": float(scores[0][rank]),
            "title": meta[i].get("title"),
            "url": meta[i].get("url"),
            "section_hint": meta[i].get("section_hint"),
            "visa_tags": meta[i].get("visa_tags"),
            "preview": meta[i]["text"][:300].replace("\n"," ")
        })
    return out

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Can F-1 students work more than 20 hours during school?"
    results = search(q, k=5)
    for r in results:
        print(f"[{r['rank']}] {r['score']:.3f} | {r['title']} | {r['section_hint']}")
        print(f"    {r['url']}")
        print(f"    tags={r['visa_tags']}")
        print(f"    {r['preview']}…\n")
