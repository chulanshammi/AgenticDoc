"""Reviewer checks: is an answer plausible, not just printed?

validate_receipt (grounding) asks "is this value in the document?". A wrong value is usually in the document too: the
cash handed over instead of the total, "DOC NO" instead of the number after it. These checks ask whether the value
plays the right role on the receipt. They are plain code, no model, so they behave the same on every run.

    review(fields, text) -> list of Issue
"""
import re
from dataclasses import dataclass

from common import AMOUNT_RE, date_of, numbers_in, parse_amount

CASH_RE = re.compile(r"^(?!.*change).*\b(cash|tendered|tender|amount paid)\b", re.IGNORECASE)
CHANGE_RE = re.compile(r"\bchange\b", re.IGNORECASE)
ROUNDING = 0.06     # Malaysian receipts round to the nearest 5 sen
MAX_INVOICE_LEN = 30
MAX_INVOICE_TOKENS = 3


@dataclass
class Issue:
    field: str
    message: str


def labelled_amount(text: str, label: re.Pattern) -> float | None:
    """The amount printed on the first line that carries the label ('CASH 50.00' gives 50.0). Needs the row layout."""
    for line in text.splitlines():
        if label.search(line):
            amounts = [parse_amount(m) for m in re.findall(AMOUNT_RE, line)]
            amounts = [a for a in amounts if a is not None]
            if amounts:
                return amounts[-1]
    return None


def total_issue(total: float, text: str) -> Issue | None:
    """The amount to pay is the cash handed over minus the change. A small model often copies the cash line."""
    cash, change = labelled_amount(text, CASH_RE), labelled_amount(text, CHANGE_RE)
    if cash is None or change is None:
        return None
    expected = round(cash - change, 2)
    if abs(total - expected) <= ROUNDING:
        return None
    printed = numbers_in(text)
    if not any(abs(expected - x) <= 0.01 for x in printed):
        return None   # cash - change is not an amount the receipt prints, so this is not a clear case; do not guess
    why = "equals the cash handed over" if abs(total - cash) <= 0.01 else "does not match the payment lines"
    return Issue("total", f"total {total} {why}. The receipt shows cash {cash} and change {change}, "
                          f"so the amount to pay is {expected}.")


def invoice_number_issue(value: str) -> Issue | None:
    v = value.strip()
    if re.search(r"[\n,]", v) or len(v) > MAX_INVOICE_LEN:
        return Issue("invoice_number", f"invoice_number {v[:40]!r} looks like a sentence or an address. "
                                       "It is a short code of letters and digits printed next to a label such as INV NO.")
    if not re.search(r"\d", v):
        return Issue("invoice_number", f"invoice_number {v!r} has no digit: it is a label, not the number after the label.")
    if len(v.split()) > MAX_INVOICE_TOKENS:
        return Issue("invoice_number", f"invoice_number {v!r} has too many words to be a number. Copy only the code.")
    if m := re.match(r"^[A-Za-z][A-Za-z .]*[:#]\s*(\S.*)$", v):
        return Issue("invoice_number", f"invoice_number {v!r} includes its label. Give only {m.group(1)!r}.")
    return None


def date_issue(value: str) -> Issue | None:
    v = value.strip()
    if date_of(v) is None or "\n" in v or len(v) > 30:
        return Issue("invoice_date", f"invoice_date {v[:40]!r} is not a date. Copy the date as printed, for example 25/03/2018.")
    return None


def review(fields: dict, text: str) -> list[Issue]:
    issues = []
    if fields.get("total"):
        issues += [i for i in [total_issue(float(fields["total"]), text)] if i]
    if fields.get("invoice_number"):
        issues += [i for i in [invoice_number_issue(str(fields["invoice_number"]))] if i]
    if fields.get("invoice_date"):
        issues += [i for i in [date_issue(str(fields["invoice_date"]))] if i]
    return issues
