import json, pathlib
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

BASE = pathlib.Path(__file__).resolve().parents[1]
LAWS = BASE / "data" / "laws"
INP = (LAWS / "clauses_dedup.jsonl") if (LAWS / "clauses_dedup.jsonl").exists() else (LAWS / "clauses.jsonl")

def build_for_tag(tag: str):
    tag_upper = tag.upper()
    docs, texts = [], []
    with open(INP, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            vt = set(d.get("visa_tags") or [])
            if tag_upper in vt:
                docs.append(d)
                texts.append(d["text"])
    if not texts:
        raise SystemExit(f"No documents found for tag {tag_upper}")
    print(f"Building index for {tag_upper}: {len(texts)} docs")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")
    idx_path = LAWS / f"faiss_{tag_upper}.index"
    meta_path = LAWS / f"faiss_{tag_upper}_meta.json"
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(idx_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(docs, f)
    print(f"Wrote index -> {idx_path}")
    print(f"Wrote meta  -> {meta_path}")

if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "F2"
    build_for_tag(tag)
