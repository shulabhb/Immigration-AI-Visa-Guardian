import json, pathlib, sys
import numpy as np, faiss
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

BASE = pathlib.Path(__file__).resolve().parents[1]
LAWS = BASE / "data" / "laws"
IDX = LAWS / "faiss.index"
META = LAWS / "faiss_meta.json"
import sys
QNA = BASE / "data" / "qna" / "qna.jsonl"
if len(sys.argv) > 1:
    arg = sys.argv[1]
    QNA = Path(arg) if arg.startswith("/") else (BASE / arg)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index(str(IDX))
meta = json.load(open(META, encoding="utf-8"))

def looks_like(doc, law_ref: str) -> bool:
    if not law_ref:
        return False
    hay = " ".join([doc.get("title",""), doc.get("section_hint",""), doc.get("url","")]).lower()
    ref = law_ref.lower().strip()
    tokens = [t for t in re.split(r"[\s,;:()]+", ref) if t]
    strong_hits = sum(1 for t in tokens if t in hay)
    if strong_hits >= 2:
        return True
    return ref in hay

def search(q, k=5, ex=None):
    qv = model.encode([q], normalize_embeddings=True).astype("float32")
    # Retrieve a larger candidate set for simple reranking
    scores, ids = index.search(qv, 50)
    candidates = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if int(i) >= 0]

    # Visa-aware, dependent-aware prefilter + boosting for reranking
    vt_raw = ((ex or {}).get("visa_type") or "").strip().upper()
    vt_map = {"F1": "F-1", "F2": "F-2", "J1": "J-1", "J2": "J-2", "H1B": "H-1B", "H4": "H-4"}
    vt_std = vt_map.get(vt_raw, vt_raw)

    # Prefilter for dependent visas: require either visa label mention or dependent keywords
    if vt_std in ("F-2", "J-2", "H-4"):
        filtered = []
        for i, s in candidates:
            doc = meta[i]
            hay = (doc.get("title", "") + " " + doc.get("section_hint", "") + " " + doc.get("text", ""))
            low = hay.lower()
            if (vt_std and vt_std in hay) or any(kw in low for kw in ["dependent", "dependents", "spouse", "spouses"]):
                filtered.append((i, s))
        # fallback if filter removes too many
        if len(filtered) >= 5:
            candidates = filtered

    # If visa raw tag exists (e.g., F2), prefer candidates explicitly tagged with it
    if vt_raw:
        tagged = [(i, s) for i, s in candidates if vt_raw in (set(meta[i].get("visa_tags") or []))]
        if len(tagged) >= 5:
            candidates = tagged

    # If visa_type is provided, prefer candidates that already carry that tag
    if vt_raw:
        tagged = [(i, s) for i, s in candidates if vt_raw in (set(meta[i].get("visa_tags") or []))]
        if tagged:
            candidates = tagged

    def boost(doc: dict) -> float:
        bonus = 0.0
        tags = set(doc.get("visa_tags") or [])
        # Prefer docs explicitly tagged with the raw visa label
        if vt_raw and vt_raw in tags:
            bonus += 0.2
        # Prefer docs whose title/section/text mention the standardized visa label (e.g., F-2)
        hay = (doc.get("title", "") + " " + doc.get("section_hint", "") + " " + doc.get("text", ""))
        if vt_std and vt_std in hay:
            bonus += 0.1
        # For dependent visas, lightly boost docs mentioning dependent/spouse terms
        if vt_std in ("F-2", "J-2", "H-4"):
            low = hay.lower()
            for kw in ["dependent", "dependents", "spouse", "spouses"]:
                if kw in low:
                    bonus += 0.05
        return bonus

    # Lightweight TF-IDF rerank blended in
    cand_texts = [meta[i].get("title", "") + "\n" + (meta[i].get("text", "") or "") for i, _ in candidates]
    sims = None
    if cand_texts:
        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform([q] + cand_texts)
        sims = cosine_similarity(mat[0], mat[1:]).ravel()

    blended = []
    sim_list = sims.tolist() if sims is not None else [0.0] * len(candidates)
    for (i, s), sim in zip(candidates, sim_list):
        blended.append((i, s + 0.15*float(sim) + boost(meta[i])))

    reranked = sorted(blended, key=lambda x: x[1], reverse=True)
    return [i for i, _ in reranked[:k]]

data = [json.loads(l) for l in open(QNA, encoding="utf-8")]
hits = 0
for ex in data:
    q = ex["question"]
    vt_raw = (ex.get("visa_type") or "").strip().upper()
    vt_map = {"F1":"F-1","F2":"F-2","J1":"J-1","J2":"J-2","H1B":"H-1B","H4":"H-4"}
    vt = vt_map.get(vt_raw, vt_raw)
    if vt:
        q = f"{vt}: {q}"
        if vt in ("F-2","J-2","H-4"):
            q = q + " dependents spouse"
    ids = search(q, k=5, ex=ex)
    ok = any(looks_like(meta[i], ex.get("law_ref","")) for i in ids)
    hits += int(ok)

p_at_5 = hits / max(len(data), 1)
print(f"Eval size: {len(data)}")
print(f"Precision@5: {p_at_5:.2f}")
