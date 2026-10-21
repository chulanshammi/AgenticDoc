"""Baseline evaluation on the real receipts in data_real/ (SROIE + CORD).

Only labelled fields are scored (null in truth.json = no label, skipped). An escalated or failed document
counts as a miss for every labelled field, because nothing was extracted. The number to watch is
SILENT ERRORS: documents that were saved even though a labelled field is wrong.
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import graph_agent
from common import date_of, to_number
from graph_agent import run_receipt_agent
from multi_agent import run_multi_agent
from receipt_agent import alnum

SCORED = {"sroie": ["supplier_name", "invoice_date", "invoice_number", "total"],
          "cord": ["total", "subtotal", "tax"]}
UNSUPPORTED = ["supplier_name", "invoice_date", "invoice_number"]   # CORD text never prints these in its labelled lines


def correct(field: str, got, want: str) -> bool:
    if got is None:
        return False
    if field in ("total", "subtotal", "tax"):
        w = to_number(want)   # labels carry prefixes such as '$8.20' and 'Rp 59.400'
        return w is not None and abs(float(got) - w) <= 0.01
    if field == "invoice_date":
        g, w = date_of(str(got)), date_of(want)
        return g is not None and g == w if w else alnum(str(got)) == alnum(want)
    if field == "supplier_name":   # the label itself is human-typed, so allow small spelling differences
        return SequenceMatcher(None, alnum(str(got)), alnum(want)).ratio() >= 0.9
    return alnum(str(got)) == alnum(want)


def sample(truths: list[dict], source: str, n: int, split: str = "all", offset: int = 0) -> list[dict]:
    pool = [t for t in truths if t["id"].startswith(source + "_")]
    if split != "all":   # SROIE only: truth.json says which split a receipt came from
        pool = [t for t in pool if t.get("source", "").startswith(f"sroie/{split}/")] if source == "sroie" else pool
    if n <= 0 or n >= len(pool):
        return pool
    step = len(pool) / n
    return [pool[int(i * step + offset)] for i in range(n)]   # evenly spaced: deterministic, no random seed to remember


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_real")
    ap.add_argument("--n", type=int, default=20, help="documents per source (0 = all)")
    ap.add_argument("--split", choices=["all", "test", "train"], default="all",
                    help="SROIE split. Use test with --rag: the supplier master is built from the train split")
    ap.add_argument("--offset", type=int, default=0,
                    help="shift the evenly spaced sample, for a second set of receipts that does not overlap the first (try 3)")
    ap.add_argument("--multi", action="store_true", help="use the supervisor + extractor + reviewer team (multi_agent.py)")
    ap.add_argument("--rag", action="store_true", help="give the agent the supplier master (needs supplier_index.py build)")
    ap.add_argument("--sources", nargs="+", default=["sroie", "cord"])
    ap.add_argument("--out", default="real_results.json")
    ap.add_argument("--rescore", metavar="RESULTS_JSON", help="score a saved run again without calling the model")
    ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    graph_agent.VERBOSE = a.verbose
    if a.rag and a.split != "test" and "sroie" in a.sources:
        ap.error("--rag needs --split test: the supplier master holds every train-split supplier, so train receipts would be answered by lookup")
    cached = {x["id"]: x["result"] for x in json.loads(Path(a.rescore).read_text())["results"]} if a.rescore else {}

    truths = json.loads((Path(a.data) / "truth.json").read_text())
    docs = [(src, t) for src in a.sources for t in sample(truths, src, a.n, a.split, a.offset)]

    hits, labelled = defaultdict(int), defaultdict(int)
    status, steps = Counter(), 0
    silent, unsupported, results = [], Counter(), []
    start = time.time()

    for k, (src, t) in enumerate(docs, 1):
        try:
            r = (cached[t["id"]] if cached else run_multi_agent(t["id"], a.data, use_rag=a.rag) if a.multi
                 else run_receipt_agent(t["id"], a.data, use_rag=a.rag))
        except Exception as e:   # connection problems etc.
            r = {"doc_id": t["id"], "status": "failed", "reason": f"{type(e).__name__}: {e}", "steps": 0, "calls": []}
        status[(src, r["status"])] += 1
        steps += r.get("steps", 0)
        got = r.get("fields") or {}
        wrong = []
        for f in SCORED[src]:
            want = t["expected"].get(f)
            if not want:
                continue
            labelled[(src, f)] += 1
            if r["status"] == "saved" and correct(f, got.get(f), want):
                hits[(src, f)] += 1
            elif r["status"] == "saved":
                wrong.append((f, got.get(f), want))
        if src == "cord" and r["status"] == "saved":
            for f in UNSUPPORTED:
                if got.get(f):
                    unsupported[f] += 1
        if wrong:
            silent.append({"id": t["id"], "wrong": wrong})
        results.append({"id": t["id"], "source": src, "result": r, "expected": t["expected"]})
        print(f"[{k}/{len(docs)}] {t['id']}: {r['status']}" + (f"  WRONG {wrong}" if wrong and a.show_errors else ""),
              flush=True)

    print(f"\n{len(docs)} documents in {time.time() - start:.0f}s, {steps / max(len(docs), 1):.1f} steps on average")
    for src in a.sources:
        n_src = sum(v for (s, _), v in status.items() if s == src)
        print(f"\n{src.upper()} ({n_src} docs): " + ", ".join(f"{st} {v}" for (s, st), v in sorted(status.items()) if s == src))
        for f in SCORED[src]:
            if labelled[(src, f)]:
                print(f"  {f:15s} {hits[(src, f)]:3d}/{labelled[(src, f)]:3d} = {hits[(src, f)] / labelled[(src, f)]:.0%}")
    if a.multi:
        revised = [x for x in results if x["result"].get("revisions")]
        reviewer_stopped = [x for x in results if str(x["result"].get("reason", "")).startswith("reviewer:")]
        print(f"\nTEAM: {len(revised)} receipts went back to the extractor at least once, "
              f"{len(reviewer_stopped)} were escalated because the reviewer's objections survived the revisions")
    saved = sum(v for (_, st), v in status.items() if st == "saved")
    print(f"\nSILENT ERRORS: {len(silent)} of {saved} saved documents have a wrong labelled field")
    if unsupported:
        print("CORD fields filled in although the text prints no such line:", dict(unsupported))
    Path(a.out).write_text(json.dumps({"results": results, "silent_errors": silent}, indent=2))
    print(f"details written to {a.out}")


if __name__ == "__main__":
    main()
