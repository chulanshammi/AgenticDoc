import argparse
import json
import random
from pathlib import Path

EMPLOYERS = [
    "ACME Textiles (Pvt) Ltd", "Lanka Tea Exports", "Brightway Logistics Ltd",
    "Harbour View Hotels", "Greenfield Foods PLC", "Nova Digital Solutions",
    "Kandy Hardware Stores", "Ceylon Spice Traders", "Thames Bridge Accounting",
]
FIRST = ["Nimal", "Kasun", "Dilani", "Sanduni", "Ruwan", "Ayesha", "Tharindu",
         "Priya", "Oliver", "Hannah", "Amal", "Sarah", "Imran", "Chamari"]
LAST = ["Perera", "Fernando", "Silva", "Jayasinghe", "Wickramasinghe", "Khan",
        "Smith", "Patel", "Bandara", "De Mel", "Rajapaksa", "Taylor"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
CURRENCIES = {"LKR": (80_000, 450_000), "GBP": (1_800, 6_000), "USD": (2_000, 7_000)}
SYMBOL = {"GBP": "£", "USD": "$", "LKR": "Rs."}

GROSS_LABELS = ["Gross pay", "Gross salary", "Total earnings", "Gross"]
DED_LABELS = ["Total deductions", "Deductions (EPF, tax)", "Deductions", "Tax & contributions"]
NET_LABELS = ["Net pay", "Net salary", "Take-home pay", "Amount paid"]


def lakh_format(n: float) -> str:
    """Sri Lankan / Indian grouping: 1,50,000.00"""
    s = f"{n:.2f}"
    whole, dec = s.split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return f"{whole}.{dec}"


def fmt_money(n: float, cur: str, rng: random.Random, plain: bool = False) -> str:
    if plain:
        return f"{n:,.2f}"
    style = rng.choice(["code_after", "code_before", "symbol", "lakh", "short"])
    if style == "short" and n % 1000 == 0 and n >= 1000:
        return f"{cur} {int(n // 1000)}k"
    if style == "lakh" and cur == "LKR":
        return f"{lakh_format(n)} LKR"
    if style == "symbol":
        return f"{SYMBOL[cur]}{n:,.2f}"
    if style == "code_before":
        return f"{cur} {n:,.2f}"
    return f"{n:,.2f} {cur}"


def fmt_period(month: int, year: int, rng: random.Random) -> str:
    return rng.choice([
        f"{MONTHS[month - 1]} {year}",
        f"{month:02d}/{year}",
        f"{MONTHS[month - 1][:3]}-{year}",
        f"{MONTHS[month - 1].upper()} {year}",
    ])


def make_doc(i: int, rng: random.Random, kind: str):
    employer = rng.choice(EMPLOYERS)
    employee = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    month, year = rng.randint(1, 12), rng.choice([2024, 2025])
    cur = rng.choice(list(CURRENCIES))
    lo, hi = CURRENCIES[cur]
    gross = float(rng.randrange(lo, hi, 500 if cur == "LKR" else 50))
    ded = round(gross * rng.uniform(0.08, 0.28), 2)
    ded = float(round(ded / 50) * 50) if rng.random() < 0.5 else ded
    net = round(gross - ded, 2)

    shown_net = net
    if kind == "inconsistent":
        shown_net = round(net + rng.choice([-5000, -1200, 800, 3500]) * (1 if cur == "LKR" else 0.01), 2)

    # Currency note: use a symbol-only layout sometimes (extractor must infer the code).
    plain = rng.random() < 0.25  # plain numbers, currency stated once in a header line
    g, d, n = (fmt_money(x, cur, rng, plain) for x in (gross, ded, shown_net))
    period = fmt_period(month, year, rng)
    gl, dl, nl = rng.choice(GROSS_LABELS), rng.choice(DED_LABELS), rng.choice(NET_LABELS)

    emp_line = f"Employee: {employee}"
    layout = rng.choice(["A", "B", "C"])
    if layout == "A":
        lines = [employer, f"PAYSLIP - {period}", emp_line, f"{gl}: {g}", f"{dl}: {d}", f"{nl}: {n}"]
    elif layout == "B":
        lines = [f"{employer.upper()}", "Monthly salary statement", f"Period: {period}",
                 f"Name: {employee}", f"{gl:<26}{g}", f"{dl:<26}{d}", f"{nl:<26}{n}"]
    else:
        lines = ["PAY ADVICE", f"Company: {employer}", f"Staff member: {employee}",
                 f"Pay period: {period}", "", f"Earnings .... {gl}: {g}",
                 f"Less ........ {dl}: {d}", f"Paid ........ {nl}: {n}"]

    if plain:
        lines.insert(1, f"Currency: {cur}")

    if kind == "missing_employee":
        lines = [l for l in lines if employee not in l]

    truth = {
        "id": f"payslip_{i:03d}",
        "kind": kind,
        "expect_escalation": kind != "normal",
        "expected": {
            "employer_name": employer,
            "employee_name": employee,
            "pay_period": f"{MONTHS[month - 1]} {year}",
            "gross_pay": gross,
            "total_deductions": ded,
            "net_pay": net,
            "currency": cur,
        } if kind == "normal" else None,
    }
    return "\n".join(lines) + "\n", truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    truths = []
    for i in range(args.n):
        r = rng.random()
        kind = "normal" if r < 0.8 else ("missing_employee" if r < 0.9 else "inconsistent")
        text, truth = make_doc(i, rng, kind)
        (out / f"{truth['id']}.txt").write_text(text, encoding="utf-8")
        truths.append(truth)

    (out / "truth.json").write_text(json.dumps(truths, indent=2), encoding="utf-8")
    counts = {k: sum(t["kind"] == k for t in truths) for k in ("normal", "missing_employee", "inconsistent")}
    print(f"Wrote {len(truths)} documents to {out}/  {counts}")


if __name__ == "__main__":
    main()