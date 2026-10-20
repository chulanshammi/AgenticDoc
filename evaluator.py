import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

FIELDS = ["employer_name", "employee_name", "pay_period",
          "gross_pay", "total_deductions", "net_pay", "currency"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
CUR_ALIASES = {"£": "GBP", "$": "USD", "RS.": "LKR", "RS": "LKR", "LKR": "LKR",
               "GBP": "GBP", "USD": "USD"}


def norm_text(s):
    return re.sub(r"\s+", " ", (s or "").strip().casefold())


def norm_period(s):
    """Return (year, month) from 'March 2026', '03/2026', 'Mar-2026', 'MARCH 2026'."""
    s = norm_text(s)
    year = re.search(r"(20\d\d)", s)
    if not year:
        return None
    month = None
    iso = re.search(r"\b(20\d\d)\s*[/\-]\s*(\d{1,2})\b", s)   # 2025-08, 2025/06
    m = re.search(r"\b(\d{1,2})\s*/", s)
    if iso and 1 <= int(iso.group(2)) <= 12:
        return (int(iso.group(1)), int(iso.group(2)))
    full = re.search(r"\b\d{1,2}\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(20\d\d)\b", s)   # 31/03/2026
    if full and 1 <= int(full.group(1)) <= 12:
        return (int(full.group(2)), int(full.group(1)))
    if m:
        month = int(m.group(1))
    else:
        for idx, name in enumerate(MONTHS, 1):
            if name in s or name[:3] in s:
                month = idx
                break
    return (int(year.group(1)), month)


def norm_currency(s):
    return CUR_ALIASES.get((s or "").strip().upper())


def field_correct(field, got, want):
    if field in ("gross_pay", "total_deductions", "net_pay"):
        return got is not None and abs(float(got) - float(want)) <= 0.01
    if field == "pay_period":
        return norm_period(got) == norm_period(want)
    if field == "currency":
        return norm_currency(got) == want
    return norm_text(got) == norm_text(want)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show-errors", action="store_true")
    args = ap.parse_args()

    from extracter import Escalate, extract_payslip  # imported here so the scoring code can be tested alone

    data = Path(args.data)
    truths = json.loads((data / "truth.json").read_text())
    if args.limit:
        truths = truths[: args.limit]

    field_hits = defaultdict(int)
    n_normal = n_normal_extracted = exact = false_escalations = 0
    n_trap = trap_caught = trap_missed = 0
    results, errors = [], []
    start = time.time()

    for k, t in enumerate(truths, 1):
        text = (data / f"{t['id']}.txt").read_text(encoding="utf-8")
        got: dict | None = None
        try:
            got = extract_payslip(text).model_dump()
            escalated = False
        except Escalate as e:
            got, escalated = None, True
            if t["expect_escalation"] is False and args.show_errors:
                print(f"  {t['id']} escalated: {str(e)[:300]}")
        except Exception as e:  # timeouts, connection errors
            got, escalated = None, True
            errors.append(f"{t['id']}: {type(e).__name__}: {e}")

        if t["expect_escalation"]:
            n_trap += 1
            if escalated:
                trap_caught += 1
            else:
                trap_missed += 1
        else:
            n_normal += 1
            if escalated:
                false_escalations += 1
            else:
                assert got is not None  # not escalated, so extraction succeeded
                n_normal_extracted += 1
                ok_all = True
                for f in FIELDS:
                    ok = field_correct(f, got.get(f), t["expected"][f])
                    field_hits[f] += ok
                    if not ok:
                        ok_all = False
                        if args.show_errors:
                            print(f"  {t['id']} {f}: got {got.get(f)!r} want {t['expected'][f]!r}")
                exact += ok_all

        results.append({"id": t["id"], "kind": t["kind"], "escalated": escalated, "got": got})
        print(f"[{k}/{len(truths)}] {t['id']} {t['kind']:<16} {'ESCALATED' if escalated else 'extracted'}")

    elapsed = time.time() - start
    Path("results.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60)
    print(f"Documents: {len(truths)}   time: {elapsed:.0f}s ({elapsed / max(len(truths), 1):.1f}s per doc)")
    print(f"\nNORMAL documents ({n_normal})")
    print(f"  sent to human by mistake : {false_escalations}")
    print(f"  exact match (all fields) : {exact}/{n_normal}")
    if n_normal_extracted:
        print("  field accuracy (of those extracted):")
        for f in FIELDS:
            print(f"    {f:<18}{field_hits[f]:>3}/{n_normal_extracted}  {100 * field_hits[f] / n_normal_extracted:5.1f}%")
    print(f"\nTRAP documents ({n_trap}: missing name or wrong total)")
    print(f"  correctly escalated      : {trap_caught}")
    print(f"  MISSED (slipped through) : {trap_missed}")
    if errors:
        print(f"\nErrors ({len(errors)}), first 3:")
        for e in errors[:3]:
            print("  ", e)
    print("\nPer-document results saved to results.json")


if __name__ == "__main__":
    main()