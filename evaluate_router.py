import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import agent
from evaluator import FIELDS as PAYSLIP_FIELDS
from evaluator import field_correct as payslip_field_correct
from evaluator import norm_currency, norm_text
from invoice_agent import date_of
from graph_agent import route

INVOICE_FIELDS = ["supplier_name", "invoice_number", "invoice_date", "subtotal", "vat", "total", "currency"]
FIELDS = {"payslip": PAYSLIP_FIELDS, "invoice": INVOICE_FIELDS}


def invoice_field_correct(field, got, want):
    if field in ("subtotal", "vat", "total"):
        return got is not None and abs(float(got) - float(want)) <= 0.01
    if field == "invoice_date":
        return date_of(str(got)) is not None and date_of(str(got)) == date_of(want)
    if field == "currency":
        return norm_currency(got) == want
    return norm_text(got) == norm_text(want)


def field_correct(doc_type, field, got, want):
    return (payslip_field_correct if doc_type == "payslip" else invoice_field_correct)(field, got, want)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_mixed")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("--rules-only", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    agent.VERBOSE = args.verbose

    data = Path(args.data)
    truths = json.loads((data / "truth.json").read_text())
    if args.limit:
        truths = truths[: args.limit]

    confusion = Counter()                       # (true type, predicted type or 'unsure')
    n_good = Counter(); good_saved = Counter(); good_exact = Counter(); good_not_saved = Counter()
    field_hits = defaultdict(int)
    n_bad = bad_kept = bad_missed = 0
    bad_how = Counter()
    steps_total = 0
    results = []
    start = time.time()

    for k, t in enumerate(truths, 1):
        try:
            r = route(t["id"], str(data), use_model=not args.rules_only)
        except Exception as e:   # connection problems etc.
            r = {"doc_id": t["id"], "doc_type": None, "status": "failed", "steps": 0, "calls": [],
                 "reason": f"{type(e).__name__}: {e}", "classifier": {}}
        results.append(r)
        steps_total += r["steps"]
        confusion[(t["doc_type"], r["doc_type"] or "unsure")] += 1
        saved = r["status"] == "saved"
        tp = t["doc_type"]

        if t["expect_escalation"]:
            n_bad += 1
            if saved:
                bad_missed += 1
                if args.show_errors:
                    print(f"  MISSED {t['id']} ({tp}/{t['kind']}) saved as {r['doc_type']}")
            else:
                bad_kept += 1
                bad_how[r["status"]] += 1
        else:
            n_good[tp] += 1
            if not saved:
                good_not_saved[tp] += 1
                if args.show_errors:
                    print(f"  {t['id']} ({tp}) not saved: {r.get('reason', '')[:160]}")
            else:
                good_saved[tp] += 1
                ok_all = r["doc_type"] == tp
                if r["doc_type"] != tp and args.show_errors:
                    print(f"  {t['id']} is a {tp} but was handled as {r['doc_type']}")
                for f in FIELDS[tp]:
                    ok = r["doc_type"] == tp and field_correct(tp, f, r["fields"].get(f), t["expected"][f])
                    field_hits[(tp, f)] += ok
                    if not ok:
                        ok_all = False
                        if args.show_errors and r["doc_type"] == tp:
                            print(f"  {t['id']} {f}: got {r['fields'].get(f)!r} want {t['expected'][f]!r}")
                good_exact[tp] += ok_all
        print(f"[{k}/{len(truths)}] {t['id']} {tp:<8} {t['kind']:<17} -> {r['doc_type'] or 'unsure':<8} {r['status']}")

    elapsed = time.time() - start
    Path("router_results.json").write_text(json.dumps(results, indent=2))
    n = max(len(truths), 1)
    types = ["payslip", "invoice", "other"]

    print("\n" + "=" * 64)
    print(f"Documents: {len(truths)}   time: {elapsed:.0f}s ({elapsed / n:.1f}s per doc)   "
          f"classifier: {'rules only' if args.rules_only else 'model + rules'}")
    print(f"Average agent model calls per document (classifier not counted): {steps_total / n:.1f}")

    print("\nCLASSIFICATION  (rows: what it really is, columns: what the system decided)")
    cols = types + ["unsure"]
    print(f"  {'':<10}" + "".join(f"{c:>9}" for c in cols))
    right = 0
    for tp in types:
        row = [confusion[(tp, c)] for c in cols]
        right += confusion[(tp, tp)]
        print(f"  {tp:<10}" + "".join(f"{v:>9}" for v in row))
    wrong_type = sum(v for (a, b), v in confusion.items() if b not in (a, "unsure"))
    print(f"  correct: {right}/{len(truths)}   wrong type: {wrong_type}   unsure (sent to a human): "
          f"{sum(v for (a, b), v in confusion.items() if b == 'unsure')}")

    for tp in ("payslip", "invoice"):
        print(f"\nGOOD {tp.upper()}S ({n_good[tp]})")
        print(f"  not saved (sent to human) : {good_not_saved[tp]}")
        print(f"  exact match (all fields)  : {good_exact[tp]}/{n_good[tp]}")
        if good_saved[tp]:
            for f in FIELDS[tp]:
                h = field_hits[(tp, f)]
                print(f"    {f:<18}{h:>3}/{good_saved[tp]}  {100 * h / good_saved[tp]:5.1f}%")

    print(f"\nBAD DOCUMENTS ({n_bad}: missing name, wrong total, or not a payslip/invoice)")
    print(f"  kept out of the system    : {bad_kept}  {dict(bad_how)}")
    print(f"  MISSED (saved anyway)     : {bad_missed}")
    print("\nPer-document results saved to router_results.json")


if __name__ == "__main__":
    main()