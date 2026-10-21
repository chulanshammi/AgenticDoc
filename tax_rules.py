"""Tax rules as a vector index (Chroma + Ollama embeddings), with date and country as metadata filters.

A vector search cannot tell that 2018-07-15 falls inside "1 June to 31 August 2018". So each rule carries
its validity period as metadata: Chroma first keeps the rules valid on the receipt's date and country, and
the embedding search then ranks what is left. This is the usual shape of RAG over rules that change over time.

    python tax_rules.py build
    python tax_rules.py search MY 2018-07-15 "which tax rate applies?"
"""
import argparse
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

EMBED_MODEL = "nomic-embed-text"
RULES_FILE = "rules/tax_rules.json"
DB_DIR = "rules_db"
NO_START, NO_END = 0, 99999999      # an open start or end of validity, as yyyymmdd integers


def day(iso: str | None, default: int) -> int:
    return int(iso.replace("-", "")) if iso else default


def load_rules(path: str = RULES_FILE) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _db(persist_dir: str = DB_DIR) -> Chroma:
    return Chroma(
        collection_name="tax_rules",
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_index(rules: list[dict], persist_dir: str = DB_DIR) -> Chroma:
    db = _db(persist_dir)
    db.reset_collection()
    db.add_texts(
        [f"search_document: {r['text']}" for r in rules],
        metadatas=[{"id": r["id"], "country": r["country"],
                    "from_int": day(r["valid_from"], NO_START), "to_int": day(r["valid_to"], NO_END)} for r in rules],
        ids=[r["id"] for r in rules],
    )
    return db


def retrieve(db: Chroma, country: str, on: tuple[int, int, int] | None, query: str, k: int = 3,
             filter_by_date: bool = True) -> list[tuple[str, float]]:
    """The rule ids that best answer `query` for a receipt from `country` dated `on` (year, month, day).
    Returns (rule id, distance), closest first. Without a date, or with filter_by_date=False, only the country filters."""
    conditions: list[dict] = [{"country": {"$eq": country}}]
    if on is not None and filter_by_date:
        d = on[0] * 10000 + on[1] * 100 + on[2]
        conditions += [{"from_int": {"$lte": d}}, {"to_int": {"$gte": d}}]
    where = {"$and": conditions} if len(conditions) > 1 else conditions[0]
    hits = db.similarity_search_with_score(f"search_query: {query}", k=k, filter=where)
    return [(doc.metadata["id"], float(dist)) for doc, dist in hits]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "search"])
    ap.add_argument("country", nargs="?")
    ap.add_argument("date", nargs="?", help="YYYY-MM-DD")
    ap.add_argument("query", nargs="?", default="which tax rate applies?")
    a = ap.parse_args()
    if a.command == "build":
        rules = load_rules()
        build_index(rules)
        print(f"indexed {len(rules)} rules into {DB_DIR}/")
    else:
        y, m, d = (int(x) for x in a.date.split("-"))
        for rid, dist in retrieve(_db(), a.country, (y, m, d), a.query):
            print(f"{dist:.3f}  {rid}")
