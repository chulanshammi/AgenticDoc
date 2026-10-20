import re
from typing import Optional

import ollama
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

MODEL = "qwen3.5:0.8b"  # must match `ollama list`
MAX_ATTEMPTS = 3
VERBOSE = False

_client = ollama.Client(timeout=90)


class Payslip(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    employer_name: str = Field(min_length=1)
    employee_name: str = Field(min_length=1)
    pay_period: str = Field(min_length=1, description="Pay period copied exactly as printed in the document")
    gross_pay: float = Field(gt=0)
    total_deductions: float = Field(ge=0)
    net_pay: float = Field(gt=0)
    currency: Optional[str] = Field(default=None, description="3-letter code, e.g. LKR")

    @model_validator(mode="after")
    def check_totals(self):
        expected = round(self.gross_pay - self.total_deductions, 2)
        if abs(expected - self.net_pay) > 0.01:
            raise ValueError(
                f"net_pay {self.net_pay} != gross_pay - total_deductions ({expected})"
            )
        return self


SYSTEM_PROMPT = (
    "You extract data from payslip text. "
    "Return ONLY JSON matching the schema. "
    "Numbers must be plain numbers: no currency symbols, no thousands separators, "
    "and expand shorthand like 150k to 150000. "
    "currency is the 3-letter code (LKR, GBP, USD). "
    "If a value is not in the text, use an empty string for text fields. Never invent values."
)


class Escalate(Exception):
    """Raised when extraction fails validation after all attempts."""


def _chat(messages):
    try:
        return _client.chat(
            model=MODEL,
            messages=messages,
            format=Payslip.model_json_schema(),
            think=False,
            options={"temperature": 0, "num_predict": 400},
        )
    except (TypeError, ollama.ResponseError):
        # Older ollama library, or model doesn't support `think`: retry without it.
        return _client.chat(
            model=MODEL,
            messages=messages,
            format=Payslip.model_json_schema(),
            options={"temperature": 0, "num_predict": 400},
        )


PLACEHOLDERS = {"n/a", "na", "none", "null", "unknown", "not provided", "not available",
                "not specified", "missing", "-", "--", "?", "tbd", "nil"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().casefold())


def on_a_line(value: str, text: str) -> bool:
    """True if value is a whole line of the document, or everything after a 'Label:' on a line.
    Stops 'Lanka Tea' passing for 'LANKA TEA EXPORTS' (a substring check would accept it)."""
    want = _norm(value)
    for line in text.splitlines():
        line = _norm(line)
        if line == want or ":" in line and _norm(line.split(":", 1)[1]) == want:
            return True
    return False


def numbers_in(text: str) -> set[float]:
    """Every number that appears in the text, in any common format.
    Handles 150,000.00 / 1,50,000.00 / 150k / 150.00, with or without a currency mark."""
    found = set()
    for m in re.finditer(r"(?<!\w)(\d[\d,]*(?:\.\d+)?)(\s?[kK])?(?!\w)", text):
        try:
            value = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        found.add(value * 1000 if m.group(2) else value)
    return found


_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_CODES = ["LKR", "GBP", "USD", "EUR", "INR", "AUD", "CAD"]


def period_of(s: str):
    """(year, month) from 'March 2026', '03/2026', 'Mar-2026', 'MARCH 2026', '31/03/2026'. None if not recognised."""
    s = s.strip().casefold()
    m = re.search(r"\b([a-z]{3,9})\.?[\s\-/,]*(20\d\d)\b", s)
    if m and m.group(1)[:3] in _MONTHS:
        return int(m.group(2)), _MONTHS.index(m.group(1)[:3]) + 1
    m = re.search(r"\b(\d{1,2})\s*[/\-]\s*(20\d\d)\b", s)
    if m and 1 <= int(m.group(1)) <= 12:
        return int(m.group(2)), int(m.group(1))
    m = re.search(r"\b\d{1,2}\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(20\d\d)\b", s)
    if m and 1 <= int(m.group(1)) <= 12:
        return int(m.group(2)), int(m.group(1))
    m = re.search(r"\b(20\d\d)\s*[/\-]\s*(\d{1,2})\b", s)   # year first: 2025-08, 2025/06, 2025-08-14
    if m and 1 <= int(m.group(2)) <= 12:
        return int(m.group(1)), int(m.group(2))
    return None


_PERIOD_RE = r"[A-Za-z]{3,9}\.?[\s\-/,]*20\d\d|\d{1,2}\s*[/\-]\s*20\d\d|\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*20\d\d|20\d\d\s*[/\-]\s*\d{1,2}\b"


def period_strings_in(text: str) -> list[str]:
    """The period strings as printed in the document (for example '12/2025'), without duplicates, in order."""
    out: list[str] = []
    for m in re.finditer(_PERIOD_RE, text):
        raw = m.group(0).strip()
        if period_of(raw) and raw not in out:
            out.append(raw)
    return out


def periods_in(text: str) -> set:
    found = set()
    for m in re.finditer(_PERIOD_RE, text):
        p = period_of(m.group(0))
        if p:
            found.add(p)
    return found


def currencies_in(text: str) -> set:
    """Currency codes the document shows, from explicit codes or symbols."""
    found = {c for c in _CODES if re.search(rf"\b{c}\b", text)}
    if "£" in text:
        found.add("GBP")
    if "€" in text:
        found.add("EUR")
    if "$" in text and not found & {"AUD", "CAD"}:
        found.add("USD")
    if re.search(r"\bRs\b\.?", text):
        found.add("LKR")
    return found


def grounding_problems(p: Payslip, text: str) -> list[str]:
    """Checks that the model copied from the document instead of making things up."""
    problems = []
    haystack = _norm(text)
    for field in ("employer_name", "employee_name"):
        value = getattr(p, field)
        if _norm(value) in PLACEHOLDERS:
            problems.append(
                f"{field} is the placeholder {value!r}. If the document has no such value, use an empty string."
            )
        elif _norm(value) not in haystack:
            problems.append(
                f"{field} {value!r} does not appear in the document. Copy it exactly, or use an empty string if absent."
            )
        elif not on_a_line(value, text):
            problems.append(
                f"{field} {value!r} is only part of a line. Copy the complete name as printed on its line."
            )
    if _norm(p.employer_name) == _norm(p.employee_name) and p.employer_name:
        problems.append(
            "employer_name and employee_name are the same. The employer is the company; the employee is the person paid."
        )
    present = numbers_in(text)
    for field in ("gross_pay", "total_deductions", "net_pay"):
        value = getattr(p, field)
        if not any(abs(value - x) <= 0.01 for x in present):
            problems.append(
                f"{field} {value} does not appear in the document. Copy the printed number; do not calculate it."
            )
    # pay period: must be a month+year that the document actually shows
    doc_periods = periods_in(text)
    if doc_periods:
        got_period = period_of(p.pay_period)
        shown = ", ".join(repr(x) for x in period_strings_in(text)[:3])
        if got_period is None:
            problems.append(
                f"pay_period {p.pay_period!r} is not a recognisable month and year. "
                f"The document shows: {shown}. Copy it exactly as printed."
            )
        elif got_period not in doc_periods:
            problems.append(
                f"pay_period {p.pay_period!r} does not match the document, which shows: {shown}. Copy it exactly as printed."
            )
    # currency: required when the document shows one, and must agree with it
    doc_currencies = currencies_in(text)
    if doc_currencies:
        got_cur = (p.currency or "").strip().upper()
        if not got_cur:
            problems.append(
                f"currency is missing. The document shows: {', '.join(sorted(doc_currencies))}."
            )
        elif got_cur not in doc_currencies:
            problems.append(
                f"currency {got_cur!r} does not match the document, which shows: {', '.join(sorted(doc_currencies))}."
            )
    return problems


def extract_payslip(text: str) -> Payslip:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Payslip text:\n\n{text}"},
    ]
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = _chat(messages)["message"]["content"]
        if VERBOSE:
            print(f"[attempt {attempt}] {raw}")
        try:
            payslip = Payslip.model_validate_json(raw)
        except ValidationError as e:
            last_error = f"schema/arithmetic: {e}"
            feedback = f"Invalid:\n{e}\nReturn corrected JSON only."
        else:
            problems = grounding_problems(payslip, text)
            if not problems:
                return payslip
            last_error = "grounding: " + "; ".join(problems)
            feedback = "Problems:\n- " + "\n- ".join(problems) + "\nReturn corrected JSON only."
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": feedback})
    raise Escalate(str(last_error))