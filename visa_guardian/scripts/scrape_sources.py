# scripts/scrape_sources.py
import os, re, time, json, hashlib, pathlib, datetime as dt, csv
import requests
from bs4 import BeautifulSoup

BASE = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"
CLEAN_DIR = BASE / "data" / "cleaned"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "visa-guardian/0.1 (contact: you@example.com)"}

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def fetch(url: str, retries=3, backoff=2) -> str:
    last_exc = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=45)
            if r.ok:
                return r.text
            last_exc = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last_exc = e
        time.sleep(backoff * (i + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_exc}")

def clean_text(html: str, selector: str | None, drop_selectors: list[str]) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    node = soup.select_one(selector) if selector else soup
    if not node:
        node = soup
    for sel in drop_selectors:
        for el in node.select(sel):
            el.decompose()
    title = (soup.title.get_text(" ", strip=True) if soup.title else "")
    text = node.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, title

def chunk_text(text: str, max_chars=600):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    out, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = f"{buf}\n{p}" if buf else p
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out

def save_raw(url: str, html: str):
    name = f"{sha1(url)}.html"
    (RAW_DIR / name).write_text(html, encoding="utf-8")

def save_clean(slug: str, url: str, title: str, chunks: list[str], meta: dict):
    stamp = dt.datetime.utcnow().isoformat()
    out_path = CLEAN_DIR / f"{slug}.jsonl"
    visa_tags = meta.get("visa_tags")
    if isinstance(visa_tags, str):
        try:
            visa_tags = json.loads(visa_tags.replace("'", '"'))
        except Exception:
            try:
                visa_tags = eval(visa_tags)
            except Exception:
                visa_tags = []
    visa_tags = visa_tags or []
    with open(out_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            rec = {
                "clause_id": f"{slug}_{i}",
                "source_id": slug,
                "url": url,
                "title": title or meta.get("section_hint") or slug,
                "text": c,
                "visa_tags": visa_tags,
                "section_hint": meta.get("section_hint"),
                "retrieved_at": stamp,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def process_row(row: dict):
    url = row["url"].strip()
    selector = row.get("selector", "").strip() or None
    drop = [s.strip() for s in row.get("drop", "").split(",") if s.strip()]
    slug = row.get("slug").strip() or sha1(url)

    print(f"→ Scraping: {slug} | {url}")
    html = fetch(url)
    save_raw(url, html)
    text, title = clean_text(html, selector, drop)
    chunks = chunk_text(text, max_chars=800)
    if not chunks:
        raise RuntimeError("No chunks extracted (check selector/drop).")
    save_clean(slug, url, title, chunks, row)

if __name__ == "__main__":
    csv_path = BASE / "sources.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("enabled", "1") != "1":
                continue
            try:
                process_row(row)
                time.sleep(1.0)
            except Exception as e:
                print(f"!! Failed {row.get('slug') or row.get('url')}: {e}")
