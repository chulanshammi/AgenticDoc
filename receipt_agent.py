"""Receipt / tax-invoice extraction (SROIE + CORD): schema, grounding checks, prompt and tools.
The tools are LangChain StructuredTools; the LangGraph loop that drives them is in graph_agent.py.

Real receipts are messy: OCR errors, no supplier line on some, no date on others. So only `total` is required.
Every other field is optional, but if the model gives a value it must be printed in the document.
"""
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated, Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from common import (PLACEHOLDERS, RunState, customers_in, date_of, date_strings_in, make_common_tools, norm,
                    numbers_in, to_number)

SUPPLIER_MATCH = 0.85   # OCR noise: 'SDN BND' for 'SDN BHD' must still count as printed
SAME_SUPPLIER = 0.9     # two spellings of one company


class Receipt(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    total: float = Field(gt=0, description="Final amount to pay")
    supplier_name: Optional[str] = Field(default=None, description="Shop or company that issued the receipt")
    invoice_date: Optional[str] = Field(default=None, description="Date copied exactly as printed")
    invoice_number: Optional[str] = None
    subtotal: Optional[float] = Field(default=None, gt=0)
    tax: Optional[float] = Field(default=None, ge=0)


def alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.casefold())


def supplier_printed(name: str, text: str) -> bool:
    """True if the name is in the text, allowing for OCR noise. Compares letters and digits only."""
    want = alnum(name)
    if not want:
        return False
    if want in alnum(text):
        return True
    lines = [alnum(l) for l in text.splitlines() if l.strip()]
    candidates = lines + [a + b for a, b in zip(lines, lines[1:])]   # names often wrap onto two lines
    return any(SequenceMatcher(None, want, c).ratio() >= SUPPLIER_MATCH for c in candidates)


def same_supplier(a: str, b: str) -> bool:
    return SequenceMatcher(None, alnum(a), alnum(b)).ratio() >= SAME_SUPPLIER


def grounding_problems(r: Receipt, text: str, known_supplier: str | None = None) -> list[str]:
    problems: list[str] = []
    present = numbers_in(text)

    if not any(abs(r.total - x) <= 0.01 for x in present):
        problems.append(f"total {r.total} does not appear in the document. Copy the printed amount; do not calculate it.")
    for field in ("subtotal", "tax"):
        value = getattr(r, field)
        if value is not None and not any(abs(value - x) <= 0.01 for x in present):
            problems.append(f"{field} {value} does not appear in the document. Leave it out if it is not printed.")

    if known_supplier:
        # The supplier master matched this receipt to a known company. That is a stronger source than the model's
        # pick from the text, which is how a cashier name or a promo line ends up as the supplier.
        if not (r.supplier_name and same_supplier(r.supplier_name, known_supplier)):
            problems.append(f"supplier_name must be {known_supplier!r}: the supplier master matched this receipt to that company.")
    elif r.supplier_name:
        if norm(r.supplier_name) in PLACEHOLDERS:
            problems.append(f"supplier_name is the placeholder {r.supplier_name!r}. Leave it empty if the document has none.")
        elif not supplier_printed(r.supplier_name, text):
            problems.append(f"supplier_name {r.supplier_name!r} does not appear in the document. "
                            "Copy it as printed, or leave it empty.")
        elif norm(r.supplier_name) in customers_in(text):
            problems.append(f"supplier_name {r.supplier_name!r} is the customer being billed, not the issuer.")

    if r.invoice_number:
        if not re.search(rf"(?<![\w\-]){re.escape(r.invoice_number)}(?![\w\-])", text, re.IGNORECASE):
            problems.append(f"invoice_number {r.invoice_number!r} does not appear in the document. "
                            "Copy it exactly as printed, or leave it empty.")

    if r.invoice_date:
        doc_dates = {d for d in (date_of(s) for s in date_strings_in(text)) if d}
        got = date_of(r.invoice_date)
        if not (norm(r.invoice_date) in norm(text) or (got and got in doc_dates)):
            shown = ", ".join(repr(x) for x in date_strings_in(text)[:3]) or "no recognisable date"
            problems.append(f"invoice_date {r.invoice_date!r} does not match the document, which shows: {shown}. "
                            "Copy it as printed, or leave it empty.")
    return problems


SYSTEM_PROMPT = """You process one scanned receipt or tax invoice by calling tools. The text comes from OCR, so it can be noisy. Follow these steps:
1. Call read_document with the document id.
2. Read the text and call validate_receipt with the fields exactly as printed in the document.
3. If validate_receipt reports PROBLEMS, fix them and call validate_receipt again.
4. When validate_receipt says OK, call save_result.
Call escalate only if the final total cannot be found in the document at all.
Rules: never invent values. total is the final amount to pay (after tax and discounts), not a line item and not the cash
handed over or the change. supplier_name is the shop or company that issued the receipt (not the customer, not a cashier).
Leave supplier_name, invoice_date, invoice_number, subtotal and tax out (empty or 0) when the document does not print them.
Copy dates and numbers exactly as printed. Pass amounts as plain numbers.
Always answer by calling a tool, never with plain text."""


def _amount(v: Any) -> float:
    """A small model sends '', null, '59,500' or 'Rp 59.400' for a number. Read it like to_number does;
    anything unreadable becomes 0, which validate_receipt treats as 'not given'."""
    return to_number(v) or 0.0


def _text(v: Any) -> Any:
    return "" if v is None else v


# The JSON schema the model sees still says number / string. Only the validation is lenient, so a sloppy value
# reaches validate_receipt (which counts failures and gives feedback) instead of failing before it.
Amount = Annotated[float, BeforeValidator(_amount)]
Text = Annotated[str, BeforeValidator(_text)]


class ValidateReceiptArgs(BaseModel):
    doc_id: str = Field(description="The document id.")
    total: Amount = Field(description="Final amount to pay, as a plain number.")
    supplier_name: Text = Field(default="", description="Shop or company that issued the receipt, as printed, or empty string if not printed.")
    invoice_date: Text = Field(default="", description="The date copied exactly as printed, or empty string if not printed.")
    invoice_number: Text = Field(default="", description="The invoice or receipt number exactly as printed, or empty string if not printed.")
    subtotal: Amount = Field(default=0, description="Amount before tax, as a plain number, or 0 if not printed.")
    tax: Amount = Field(default=0, description="Tax amount of money (not the percentage), as a plain number, or 0 if not printed.")


def make_receipt_tools(data_dir: Path, state: RunState, out_dir: Path | None):
    check_id, reject, read_document, save_result, escalate = make_common_tools(data_dir, state, out_dir)

    def validate_receipt(
        doc_id: str,
        total: float,
        supplier_name: str = "",
        invoice_date: str = "",
        invoice_number: str = "",
        subtotal: float = 0,
        tax: float = 0,
    ) -> str:
        """Check extracted receipt fields against the schema and against the document text."""
        if (err := check_id(doc_id)) is not None:
            return err
        state.validated = None
        tot, sub, tx = to_number(total), to_number(subtotal), to_number(tax)
        if not tot:
            return reject(["total is missing or not a number. Give the amount as printed, as a plain number such as 18000.50."])
        text = (data_dir / f"{doc_id}.txt").read_text(encoding="utf-8")
        try:
            receipt = Receipt(
                total=tot,
                supplier_name=supplier_name or None,
                invoice_date=invoice_date or None,
                invoice_number=invoice_number or None,
                subtotal=sub or None,   # 0 means 'not printed'
                tax=tx or None,
            )
        except ValidationError as e:
            return reject([f"{'.'.join(str(x) for x in d['loc']) or 'document'}: {d['msg']}" for d in e.errors()])
        if problems := grounding_problems(receipt, text, state.known_supplier):
            return reject(problems)
        state.validated = receipt.model_dump()
        return "OK. Now call save_result."

    return [
        StructuredTool.from_function(read_document, parse_docstring=True),
        StructuredTool.from_function(
            validate_receipt, args_schema=ValidateReceiptArgs,
            description="Check extracted receipt fields against the schema and against the document text. "
                        "Returns OK, or a list of problems to fix.",
        ),
        StructuredTool.from_function(save_result, parse_docstring=True),
        StructuredTool.from_function(escalate, parse_docstring=True),
    ]
