import argparse
import json
import random
from pathlib import Path

import payslip_generator as gp

SUPPLIERS = [
    "Colombo Office Supplies", "Island Packaging Co", "Metro IT Services Ltd",
    "Greenline Cleaning Services", "Orient Print House", "Zenith Freight Ltd",
    "Riverside Catering", "Sunrise Stationers",
]
ITEMS = [
    "Printer paper (box)", "Consulting hours", "Cloud hosting", "Courier delivery",
    "Cleaning service", "Laptop repair", "Printed brochures", "Software licence",
]
UNIT_RANGE = {"LKR": (2_000, 60_000), "GBP": (20, 900), "USD": (20, 900)}
VAT_RATES = {"LKR": [18, 15], "GBP": [20], "USD": [10, 15]}
NUM_LABELS = ["Invoice No", "Invoice #", "Inv. Number", "Invoice number"]
DATE_LABELS = ["Date", "Invoice date", "Issued"]
SUB_LABELS = ["Subtotal", "Net amount", "Total before VAT"]
TOT_LABELS = ["Total", "Total due", "Amount due", "Grand total"]


def fmt_date(y: int, m: int, d: int, rng: random.Random) -> str:
    return rng.choice([
        f"{d} {gp.MONTHS[m - 1]} {y}",
        f"{y}-{m:02d}-{d:02d}",
        f"{d:02d}/{m:02d}/{y}",
        f"{d:02d}-{gp.MONTHS[m - 1][:3]}-{y}",
    ])


def make_invoice(i: int, rng: random.Random, kind: str):
    supplier = rng.choice(SUPPLIERS)
    customer = rng.choice(gp.EMPLOYERS)
    cur = rng.choice(list(gp.CURRENCIES))
    lo, hi = UNIT_RANGE[cur]
    items = []
    for desc in rng.sample(ITEMS, rng.randint(2, 3)):
        qty = rng.randint(1, 10)
        unit = float(rng.randrange(lo, hi, 50 if cur == "LKR" else 5))
        items.append((desc, qty, unit, qty * unit))
    subtotal = float(sum(x[3] for x in items))
    rate = rng.choice(VAT_RATES[cur])
    vat = round(subtotal * rate / 100, 2)
    total = round(subtotal + vat, 2)

    shown_total = total
    if kind == "inconsistent":
        shown_total = round(total + rng.choice([-1500, -250, 400, 2000]) * (1 if cur == "LKR" else 0.01), 2)

    plain = rng.random() < 0.25   # plain numbers, currency stated once in a header line
    s, v, t = (gp.fmt_money(x, cur, rng, plain) for x in (subtotal, vat, shown_total))
    number = f"{rng.choice(['INV', 'SI', 'BILL'])}-{rng.choice([2024, 2025])}-{rng.randint(1, 9999):04d}"
    y, m, d = rng.choice([2024, 2025]), rng.randint(1, 12), rng.randint(1, 28)
    date = fmt_date(y, m, d, rng)
    nl, dl, sl, tl = rng.choice(NUM_LABELS), rng.choice(DATE_LABELS), rng.choice(SUB_LABELS), rng.choice(TOT_LABELS)
    vl = f"VAT ({rate}%)" if rng.random() < 0.5 else f"VAT @ {rate}%"
    item_lines = [f"{desc:<24}{qty} x {unit:,.2f} = {amount:,.2f}" for desc, qty, unit, amount in items]
    money_lines = [f"{sl}: {s}", f"{vl}: {v}", f"{tl}: {t}"]

    layout = rng.choice(["A", "B", "C"])
    if layout == "A":
        lines = [supplier, "INVOICE", f"{nl}: {number}", f"{dl}: {date}", f"Bill to: {customer}", "",
                 *item_lines, "", *money_lines]
    elif layout == "B":
        lines = ["TAX INVOICE", f"From: {supplier}", f"Billed to: {customer}", f"{nl}: {number}",
                 f"{dl}: {date}", "", *item_lines, "", *money_lines]
    else:
        lines = ["Invoice", f"Supplier: {supplier}", f"Customer: {customer}", f"{dl}: {date}",
                 f"{nl}: {number}", "", *item_lines, "", *money_lines]
    if plain:
        lines.insert(1, f"Currency: {cur}")
    if kind == "missing_supplier":
        lines = [ln for ln in lines if supplier not in ln]

    truth = {
        "id": f"invoice_{i:03d}",
        "kind": kind,
        "expect_escalation": kind != "normal",
        "expected": {
            "supplier_name": supplier,
            "invoice_number": number,
            "invoice_date": f"{y}-{m:02d}-{d:02d}",
            "subtotal": subtotal,
            "vat": vat,
            "total": total,
            "currency": cur,
        } if kind == "normal" else None,
    }
    return "\n".join(lines) + "\n", truth


def make_other(i: int, rng: random.Random):
    company = rng.choice(gp.EMPLOYERS + SUPPLIERS)
    people = [f"{rng.choice(gp.FIRST)} {rng.choice(gp.LAST)}" for _ in range(3)]
    y, m, d = rng.choice([2024, 2025]), rng.randint(1, 12), rng.randint(1, 28)
    date = fmt_date(y, m, d, rng)
    which = rng.choice(["meeting", "email", "delivery"])
    if which == "meeting":
        lines = [f"Meeting notes - {date}", f"Attendees: {', '.join(people)}", "Agenda:",
                 "1. Review of last week's tasks", "2. Office move planning", "3. Any other business", "",
                 f"Action: {people[0]} to confirm the venue by Friday.",
                 f"Action: {people[1]} to update the team calendar."]
    elif which == "email":
        lines = [f"From: {people[0]}", f"To: {people[1]}", f"Date: {date}", "Subject: Quick question", "",
                 f"Hi {people[1].split()[0]},",
                 rng.choice(["Could you send me the latest version of the staff rota?",
                             "Please remind me what time the supplier visit is on Thursday.",
                             "Can we move our catch-up to the afternoon?"]),
                 "Thanks,", people[0].split()[0]]
    else:
        lines = ["DELIVERY NOTE", f"Supplier: {rng.choice(SUPPLIERS)}", f"Delivered to: {company}",
                 f"Date: {date}", "Items: 4 boxes, 2 pallets", f"Received by: {people[2]}"]
    truth = {"id": f"other_{i:03d}", "kind": "other", "expect_escalation": True, "expected": None}
    return "\n".join(lines) + "\n", truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data_mixed")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    truths = []
    for i in range(args.n):
        doc_type = rng.choices(["payslip", "invoice", "other"], weights=[40, 40, 20])[0]
        r = rng.random()
        if doc_type == "other":
            text, truth = make_other(i, rng)
        elif doc_type == "payslip":
            kind = "normal" if r < 0.8 else ("missing_employee" if r < 0.9 else "inconsistent")
            text, truth = gp.make_doc(i, rng, kind)
        else:
            kind = "normal" if r < 0.8 else ("missing_supplier" if r < 0.9 else "inconsistent")
            text, truth = make_invoice(i, rng, kind)
        truth["id"] = f"doc_{i:03d}"
        truth["doc_type"] = doc_type
        (out / f"{truth['id']}.txt").write_text(text, encoding="utf-8")
        truths.append(truth)

    (out / "truth.json").write_text(json.dumps(truths, indent=2), encoding="utf-8")
    counts: dict[str, int] = {}
    for t in truths:
        key = f"{t['doc_type']}/{t['kind']}"
        counts[key] = counts.get(key, 0) + 1
    print(f"Wrote {len(truths)} documents to {out}/")
    for k in sorted(counts):
        print(f"  {k:<28}{counts[k]}")


if __name__ == "__main__":
    main()