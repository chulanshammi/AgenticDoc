import argparse
import json
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

import agent
from agent import RunState, make_common_tools, to_number
from extracter import PLACEHOLDERS, _norm, currencies_in, numbers_in, on_a_line


class Invoice(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    supplier_name: str = Field(min_length=1, description="The company that issued the invoice")
    invoice_number: str = Field(min_length=1)
    invoice_date: str = Field(min_length=1, description="Invoice date copied exactly as printed")
    subtotal: float = Field(gt=0)
    vat: float = Field(ge=0, description="The VAT amount of money, not the percentage")
    total: float = Field(gt=0)
    currency: Optional[str] = Field(default=None, description="3-letter code")

    @model_validator(mode="after")
    def check_totals(self):
        expected = round(self.subtotal + self.vat, 2)
        if abs(expected - self.total) > 0.01:
            raise ValueError(f"total {self.total} != subtotal + vat ({expected})")
        return self


# ---- dates -----------------------------------------------------------------------------------
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_DATE_RE = (r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{4}"
            r"|\d{1,2}[ \-][A-Za-z]{3,9}\.?[ \-,]*\d{4}|[A-Za-z]{3,9}\.? \d{1,2},? \d{4}")


def _valid(y: int, m: int, d: int):
    return (y, m, d) if 1 <= m <= 12 and 1 <= d <= 31 else None


def date_of(s: str):
    """(year, month, day) from '2025-03-12', '12/03/2025' (day first), '12 March 2025', '12-Mar-2025',
    'March 12, 2025'. None if not recognised."""
    s = s.strip().casefold()
    if m := re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", s):
        return _valid(int(m[1]), int(m[2]), int(m[3]))
    if m := re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", s):
        return _valid(int(m[3]), int(m[2]), int(m[1]))
    if m := re.search(r"\b(\d{1,2})[ \-]([a-z]{3,9})\.?[ \-,]*(\d{4})\b", s):
        if m[2][:3] in _MONTHS:
            return _valid(int(m[3]), _MONTHS.index(m[2][:3]) + 1, int(m[1]))
    if m := re.search(r"\b([a-z]{3,9})\.? (\d{1,2}),? (\d{4})\b", s):
        if m[1][:3] in _MONTHS:
            return _valid(int(m[3]), _MONTHS.index(m[1][:3]) + 1, int(m[2]))
    return None


def date_strings_in(text: str) -> list[str]:
    """The dates as printed in the document, without duplicates, in order."""
    out: list[str] = []
    for m in re.finditer(_DATE_RE, text):
        raw = m.group(0).strip()
        if date_of(raw) and raw not in out:
            out.append(raw)
    return out


# ---- grounding ---------------------------------------------------------------------------------
CUSTOMER_LABELS = {"bill to", "billed to", "customer", "client", "to"}


def customers_in(text: str) -> set[str]:
    """Names printed after labels like 'Bill to:' or 'Customer:'. Used to catch supplier/customer mix-ups."""
    out = set()
    for line in text.splitlines():
        if ":" in line:
            label, value = line.split(":", 1)
            if _norm(label) in CUSTOMER_LABELS:
                out.add(_norm(value))
    return out


def summary_lines(text: str) -> list[str]:
    """The printed subtotal / VAT / total lines (not the item lines), so a hint can point at them."""
    out = []
    for line in text.splitlines():
        if re.search(r"vat|subtotal|net amount|total|amount due", line, re.IGNORECASE) and not re.search(r" x .* = ", line):
            out.append(line.strip())
    return out[:5]


def grounding_problems(inv: Invoice, text: str) -> list[str]:
    problems: list[str] = []
    haystack = _norm(text)

    # supplier: a whole line (or everything after a label), not a placeholder, and not the customer
    supplier = _norm(inv.supplier_name)
    if supplier in PLACEHOLDERS:
        problems.append(
            f"supplier_name is the placeholder {inv.supplier_name!r}. If the document has no supplier, use an empty string."
        )
    elif supplier not in haystack:
        problems.append(
            f"supplier_name {inv.supplier_name!r} does not appear in the document. Copy it exactly, or use an empty string if absent."
        )
    elif not on_a_line(inv.supplier_name, text):
        problems.append(
            f"supplier_name {inv.supplier_name!r} is only part of a line. Copy the complete name as printed on its line."
        )
    elif supplier in customers_in(text):
        problems.append(
            f"supplier_name {inv.supplier_name!r} is the customer being billed. The supplier is the other company, "
            "the one that issued the invoice. If no other company is named, use an empty string."
        )

    # invoice number: printed exactly, as a whole token
    number = inv.invoice_number
    if _norm(number) in PLACEHOLDERS:
        problems.append(f"invoice_number is the placeholder {number!r}.")
    elif not re.search(rf"(?<![\w\-]){re.escape(number)}(?![\w\-])", text, re.IGNORECASE):
        problems.append(f"invoice_number {number!r} does not appear in the document. Copy it exactly as printed.")

    # date: must be a date the document actually prints
    doc_dates = {d for d in (date_of(s) for s in date_strings_in(text)) if d}
    if doc_dates:
        got = date_of(inv.invoice_date)
        shown = ", ".join(repr(x) for x in date_strings_in(text)[:3])
        if got is None:
            problems.append(
                f"invoice_date {inv.invoice_date!r} is not a recognisable date. The document shows: {shown}. Copy it exactly as printed."
            )
        elif got not in doc_dates:
            problems.append(
                f"invoice_date {inv.invoice_date!r} does not match the document, which shows: {shown}. Copy it exactly as printed."
            )

    # amounts: printed in the document, never calculated
    present = numbers_in(text)
    for field in ("subtotal", "vat", "total"):
        value = getattr(inv, field)
        if not any(abs(value - x) <= 0.01 for x in present):
            problems.append(
                f"{field} {value} does not appear in the document. Copy the printed amount; do not calculate it."
            )

    # currency
    doc_currencies = currencies_in(text)
    if doc_currencies:
        got_cur = (inv.currency or "").strip().upper()
        if not got_cur:
            problems.append(f"currency is missing. The document shows: {', '.join(sorted(doc_currencies))}.")
        elif got_cur not in doc_currencies:
            problems.append(
                f"currency {got_cur!r} does not match the document, which shows: {', '.join(sorted(doc_currencies))}."
            )
    return problems


# ---- the agent -------------------------------------------------------------------------------
SYSTEM_PROMPT = """You process one invoice document by calling tools. Follow these steps:
1. Call read_document with the document id.
2. Read the text and call validate_invoice with the fields exactly as printed in the document.
3. If validate_invoice reports PROBLEMS, fix them and call validate_invoice again. Most problems are fixable:
   a wrong number format, a wrong date, a wrong currency. Correct the value and try again.
4. When validate_invoice says OK, call save_result.
Call escalate only if, after trying to fix the problem, a required value is truly missing from the document
(for example there is no supplier name) or the printed numbers contradict each other.
Rules: never invent values; copy numbers exactly as printed (no calculating). The supplier is the company
that issued the invoice, not the customer who is billed. vat is the amount of money, not the percentage.
Pass amounts as plain numbers with no currency symbols or commas, and expand shorthand (150k means 150000).
Currency is a 3-letter code such as LKR, GBP or USD. Copy the invoice number and date exactly as printed.
Always answer by calling a tool, never with plain text."""


def make_invoice_tools(data_dir: Path, state: RunState, out_dir: Path | None):
    check_id, reject, read_document, save_result, escalate = make_common_tools(data_dir, state, out_dir)

    def validate_invoice(
        doc_id: str,
        supplier_name: str,
        invoice_number: str,
        invoice_date: str,
        subtotal: float,
        vat: float,
        total: float,
        currency: str = "",
    ) -> str:
        """Check extracted invoice fields against the schema and against the document text.

        Args:
            doc_id (str): The document id.
            supplier_name (str): Name of the company that issued the invoice, exactly as printed, or empty string if absent.
            invoice_number (str): The invoice number exactly as printed.
            invoice_date (str): The invoice date copied exactly as printed in the document.
            subtotal (float): Amount before VAT, as a plain number.
            vat (float): The VAT amount of money (not the percentage), as a plain number.
            total (float): Total to pay, as a plain number.
            currency (str): Three-letter currency code such as LKR, GBP or USD.

        Returns:
            str: OK, or a list of problems to fix.
        """
        if (err := check_id(doc_id)) is not None:
            return err
        state.validated = None
        sub, tax, tot = to_number(subtotal), to_number(vat), to_number(total)
        if sub is None or tax is None or tot is None:
            raw = {"subtotal": subtotal, "vat": vat, "total": total}
            return reject([
                f"{k} {v!r} is not a number. Give the amount as printed, as a plain number such as 18000.50."
                for k, v in raw.items() if to_number(v) is None
            ])
        text = (data_dir / f"{doc_id}.txt").read_text(encoding="utf-8")
        try:
            invoice = Invoice(
                supplier_name=supplier_name,
                invoice_number=invoice_number,
                invoice_date=invoice_date,
                subtotal=sub,
                vat=tax,
                total=tot,
                currency=currency or None,
            )
        except ValidationError as e:
            msgs = [f"{'.'.join(str(x) for x in d['loc']) or 'document'}: {d['msg']}" for d in e.errors()]
            if any(" != " in m for m in msgs):
                # The usual cause is copying the wrong numbers (an item amount, a quantity, the VAT rate).
                # Point at the printed summary lines. Only if the model already copied from them is it escalate-worthy.
                msgs.append(
                    "Copy subtotal, vat and total from the summary lines of the document: "
                    + " | ".join(summary_lines(text))
                    + ". vat is the amount of money, not the percentage. Do not use item line amounts or quantities. "
                    "If you already copied from these lines and the numbers still do not add up, "
                    "the document itself is inconsistent: call escalate."
                )
            return reject(msgs, escalate_hint=False)
        problems = grounding_problems(invoice, text)
        if problems:
            return reject(problems)
        state.validated = invoice.model_dump()
        return "OK. Now call save_result."

    return [read_document, validate_invoice, save_result, escalate]


def run_invoice_agent(doc_id: str, data_dir: str = "data_mixed", out_dir: str | None = None) -> dict:
    return agent.run_agent(doc_id, data_dir, out_dir,
                           tools_factory=make_invoice_tools, system_prompt=SYSTEM_PROMPT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_id")
    ap.add_argument("--data", default="data_mixed")
    ap.add_argument("--out", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    agent.VERBOSE = a.verbose
    print(json.dumps(run_invoice_agent(a.doc_id, a.data, a.out), indent=2))