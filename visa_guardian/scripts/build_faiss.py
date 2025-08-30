import json, pathlib
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

BASE = pathlib.Path(__file__).resolve().parents[1]
LAWS = BASE / "data" / "laws"
INP = (LAWS / "clauses_dedup.jsonl") if (LAWS / "clauses_dedup.jsonl").exists() else (LAWS / "clauses.jsonl")
IDX = LAWS / "faiss.index"
META = LAWS / "faiss_meta.json"

print(f"Reading corpus from: {INP}")
docs, texts = [], []
with open(INP, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        docs.append(d); texts.append(d["text"])

print(f"Loaded {len(texts)} clauses.")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(texts, batch_size=64, normalize_embeddings=True)
emb = np.asarray(emb, dtype="float32")

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, str(IDX))

with open(META, "w", encoding="utf-8") as f:
    json.dump(docs, f)

print(f"Wrote index -> {IDX}")
print(f"Wrote metadata -> {META}")
