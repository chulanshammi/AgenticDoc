"""Does the supplier index find the right supplier from the top lines of a receipt?

Queries are SROIE *test* receipts. The index only holds *train* suppliers (see supplier_index.py), so a test
supplier is either already known (it should be found) or new (the index can only return a wrong neighbour).
Both cases matter: the second one tells us how to set a "no known supplier matches" threshold.

The plain string-similarity baseline answers: is a vector database needed here at all?
"""
import argparse
import json
import statistics
from difflib import SequenceMatcher
from pathlib import Path

from supplier_index import _db, alnum, fuzzy_receipt, load_master, search_receipt

SAME_NAME = 0.9   # two spellings of one company; the labels are human-typed


def same(a: str, b: str) -> bool:
    return SequenceMatcher(None, alnum(a), alnum(b)).ratio() >= SAME_NAME


def rank_of(truth: str, hits: list[tuple[str, float]]) -> int | None:
    return next((i for i, (n, _) in enumerate(hits) if same(n, truth)), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_real")
    ap.add_argument("--n", type=int, default=0, help="test documents to use (0 = all 347), evenly spaced")
    a = ap.parse_args()

    master = load_master(f"{a.data}/truth.json")
    docs = [t for t in json.loads(Path(f"{a.data}/truth.json").read_text())
            if t.get("source", "").startswith("sroie/test/") and t["expected"].get("supplier_name")]
    if a.n and a.n < len(docs):
        docs = [docs[int(i * len(docs) / a.n)] for i in range(a.n)]
    db = _db()
    print(f"{len(master)} suppliers in the index, {len(docs)} test receipts")

    known, unknown = [], []   # per receipt: ranks and the distance of the best hit
    for t in docs:
        text = (Path(a.data) / f"{t['id']}.txt").read_text(encoding="utf-8")
        truth = t["expected"]["supplier_name"]
        vec, fuz = search_receipt(db, text), fuzzy_receipt(master, text)
        row = {"id": t["id"], "truth": truth, "vec_rank": rank_of(truth, vec), "fuz_rank": rank_of(truth, fuz),
               "vec_d": vec[0][1], "fuz_d": fuz[0][1], "vec_top": vec[0][0], "fuz_top": fuz[0][0]}
        (known if any(same(m, truth) for m in master) else unknown).append(row)

    n = len(docs)
    print(f"\nthe true supplier is already in the index for {len(known)}/{n} = {len(known) / n:.0%} of test receipts")
    print("\nAmong those, is the true supplier found?")
    print(f"  {'':22s}{'top-1':>8s}{'top-3':>8s}")
    for label, key in (("vector (Chroma)", "vec_rank"), ("string similarity", "fuz_rank")):
        r = [x[key] for x in known]
        print(f"  {label:22s}{sum(v == 0 for v in r) / len(r):8.0%}{sum(v is not None for v in r) / len(r):8.0%}")

    print("\nDistance of the best hit (smaller = closer). Can a threshold tell known from new suppliers?")
    for label, key in (("vector (Chroma)", "vec_d"), ("string similarity", "fuz_d")):
        hit = [x[key] for x in known if x[key.replace("_d", "_rank")] == 0]
        miss = [x[key] for x in unknown]
        # best single cut-off: accept the match when distance <= cut
        cuts = sorted(set(hit + miss))
        best = max(((sum(d <= c for d in hit) + sum(d > c for d in miss)) / (len(hit) + len(miss)), c) for c in cuts)
        print(f"  {label:22s}correct hits median {statistics.median(hit):.3f} | new suppliers median "
              f"{statistics.median(miss):.3f} | best cut-off {best[1]:.3f} separates them {best[0]:.0%} of the time")

    print("\nA few receipts where the vector search missed a known supplier:")
    for x in [x for x in known if x["vec_rank"] != 0][:5]:
        print(f"  {x['id']}: true {x['truth']!r} -> found {x['vec_top']!r} (d={x['vec_d']:.3f})")


if __name__ == "__main__":
    main()
