"""Does the tax-rule index return the rule that applied on the receipt's date?

Queries are the 945 SROIE receipts whose label has a parsable date (Malaysia). The right answer comes from the rules
themselves: before 2018-06-01 the 6% GST rule, from 2018-06-01 to 2018-08-31 the 0% rule, from 2018-09-01 the two SST rules.

Two modes: country filter only (the embedding search has to work out the date from the words), and country + date filter.
Reported per period, because the 6% period is 88% of the receipts and a search that always answers "6%" would look good.
"""
import json
from collections import defaultdict
from pathlib import Path

from common import date_of
from tax_rules import _db, retrieve


def gold(d: tuple[int, int, int]) -> tuple[str, set[str]]:
    if d <= (2018, 5, 31):
        return "GST 6% (to May 2018)", {"my_gst_standard"}
    if d <= (2018, 8, 31):
        return "GST 0% (Jun-Aug 2018)", {"my_gst_zero"}
    return "SST (Sep 2018 on)", {"my_sst_sales", "my_sst_service"}


def main():
    truths = [t for t in json.loads(Path("data_real/truth.json").read_text()) if t["id"].startswith("sroie_")]
    receipts = [(t["expected"]["invoice_date"], date_of(t["expected"]["invoice_date"] or "")) for t in truths]
    receipts = [(s, d) for s, d in receipts if d]
    db = _db()

    cache: dict[tuple, list[str]] = {}   # the query depends only on the date, so one search per distinct date and mode
    def top3(s, d, filtered):
        key = (s, filtered)
        if key not in cache:
            cache[key] = [rid for rid, _ in retrieve(db, "MY", d, f"Malaysia receipt dated {s}: which tax rate applies?",
                                                     k=3, filter_by_date=filtered)]
        return cache[key]

    stats = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))   # mode -> period -> [receipts, top-1 right, all right in top-3]
    for s, d in receipts:
        period, want = gold(d)
        for mode, filtered in (("country filter only", False), ("country + date filter", True)):
            got = top3(s, d, filtered)
            row = stats[mode][period]
            row[0] += 1
            row[1] += bool(got) and got[0] in want
            row[2] += want <= set(got)

    print(f"{len(receipts)} receipts with a date, {len({s for s, _ in receipts})} distinct dates\n")
    for mode, periods in stats.items():
        print(mode)
        print(f"  {'period':26s}{'receipts':>9s}{'top-1 right':>13s}{'all right in top-3':>20s}")
        for period, (n, t1, t3) in sorted(periods.items()):
            print(f"  {period:26s}{n:9d}{t1 / n:13.0%}{t3 / n:20.0%}")
        print()


if __name__ == "__main__":
    main()
