"""Supplier master as a vector database (Chroma + Ollama embeddings).

The master is the set of supplier names we have already seen: the company labels of the SROIE *train* split.
It never contains test-split suppliers, so evaluating on the test split is honest.

    python supplier_index.py build          # embed the names into supplier_db/
    python supplier_index.py search "BOOK TA .K(TAMAN DAYA) SDN BND"
"""
import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

EMBED_MODEL = "nomic-embed-text"    # ollama pull nomic-embed-text
DB_DIR = "supplier_db"
HEADER_LINES = 8                    # the supplier name is printed near the top of a receipt
MIN_LINE = 4                        # skip lines too short to be a name
CUTOFF = 0.22                       # best-hit distance up to here counts as a confident match (see evaluate_supplier_rag.py)


def alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.casefold())


def load_master(truth_path: str = "data_real/truth.json") -> list[str]:
    """Distinct supplier names from the SROIE train split."""
    names: dict[str, str] = {}
    for t in json.loads(Path(truth_path).read_text()):
        name = t["expected"].get("supplier_name")
        if t.get("source", "").startswith("sroie/train/") and name:
            names.setdefault(alnum(name), name)
    return sorted(names.values())


def _db(persist_dir: str = DB_DIR) -> Chroma:
    return Chroma(
        collection_name="suppliers",
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},   # distance = 1 - cosine similarity, 0 is identical
    )


def build_index(names: list[str], persist_dir: str = DB_DIR) -> Chroma:
    db = _db(persist_dir)
    db.reset_collection()   # rebuild from scratch, so the index always matches the master
    # nomic-embed-text expects a task prefix on both sides
    db.add_texts([f"search_document: {n}" for n in names], metadatas=[{"name": n} for n in names])
    return db


def search(db: Chroma, text: str, k: int = 3) -> list[tuple[str, float]]:
    """The k nearest known suppliers to a name, as (name, distance). Smaller distance is a closer match."""
    hits = db.similarity_search_with_score(f"search_query: {text}", k=k)
    return [(doc.metadata["name"], float(dist)) for doc, dist in hits]


def header_lines(text: str) -> list[str]:
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) >= MIN_LINE]
    return lines[:HEADER_LINES]


def search_receipt(db: Chroma, text: str, k: int = 3) -> list[tuple[str, float]]:
    """Known suppliers that match the top lines of a receipt: each line is a query, the best distance per name wins."""
    best: dict[str, float] = {}
    for line in header_lines(text):
        for name, dist in search(db, line, k):
            best[name] = min(dist, best.get(name, 9.0))
    return sorted(best.items(), key=lambda x: x[1])[:k]


def known_supplier(db: Chroma, text: str, cutoff: float = CUTOFF) -> str | None:
    """The known supplier this receipt is from, or None when no known supplier is close enough (a new supplier)."""
    hits = search_receipt(db, text, k=1)
    return hits[0][0] if hits and hits[0][1] <= cutoff else None


def fuzzy_receipt(names: list[str], text: str, k: int = 3) -> list[tuple[str, float]]:
    """The no-vector-database baseline: plain string similarity between the header lines and every known name.
    Returns (name, 1 - similarity) so it reads like a distance."""
    lines = [alnum(l) for l in header_lines(text)]
    candidates = lines + [a + b for a, b in zip(lines, lines[1:])]   # names often wrap onto two lines
    scored = []
    for name in names:
        want = alnum(name)
        scored.append((name, 1 - max((SequenceMatcher(None, want, c).ratio() for c in candidates), default=0)))
    return sorted(scored, key=lambda x: x[1])[:k]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "search"])
    ap.add_argument("query", nargs="?")
    a = ap.parse_args()
    if a.command == "build":
        names = load_master()
        build_index(names)
        print(f"indexed {len(names)} suppliers into {DB_DIR}/")
    else:
        for name, dist in search(_db(), a.query or "", k=5):
            print(f"{dist:.3f}  {name}")
