"""Shared building blocks for the receipt agent: amounts, dates, text helpers, per-run state and the common tools."""
import json
import os
import re
from pathlib import Path
from typing import Any

MODEL = os.environ.get("AGENT_MODEL", "qwen3.5:0.8b")   # must match `ollama list`. Try: AGENT_MODEL=deepseek-r1:7b
MAX_VALIDATION_FAILURES = 3     # after this many failed validations, code escalates the document itself

PLACEHOLDERS = {"n/a", "na", "none", "null", "unknown", "not provided", "not available",
                "not specified", "missing", "-", "--", "?", "tbd", "nil"}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().casefold())


# ---- amounts -----------------------------------------------------------------------------------
AMOUNT_RE = r"\d[\d.,]*\d|\d"


def parse_amount(token: str) -> float | None:
    """One printed amount to a float, whichever separator convention it uses.
    18,000.50 / 1,50,000.00 / 18.000,50 / 23.000 (dot = thousands) / 12,50 (comma = decimal) / 9.00 / 1500"""
    t = token.strip()
    if not re.fullmatch(AMOUNT_RE, t):
        return None
    dots, commas = t.count("."), t.count(",")
    if dots and commas:                                  # the separator that comes last is the decimal point
        decimal = "." if t.rfind(".") > t.rfind(",") else ","
        thousands = "," if decimal == "." else "."
        t = t.replace(thousands, "").replace(decimal, ".")
    elif commas:
        after = t.rsplit(",", 1)[1]
        t = t.replace(",", ".") if commas == 1 and len(after) != 3 else t.replace(",", "")
    elif dots:
        after = t.rsplit(".", 1)[1]
        if dots > 1 or (len(after) == 3 and not t.startswith("0.")):   # 23.000 and 1.500.000 are thousands
            t = t.replace(".", "")
    try:
        return float(t)
    except ValueError:
        return None


def to_number(value: Any) -> float | None:
    """Accept 18000, '18,000.00', 'Rs.18,000.00', '£1,234.50', '1,50,000.00', '150k', 'Rp 39.600'.
    None if it is not a number. The grounding check still proves the value is printed in the document."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(rf"({AMOUNT_RE})\s?([kK])?(?![\w])", value)
        if m and (amount := parse_amount(m.group(1))) is not None:
            return amount * (1000 if m.group(2) else 1)
    return None


def numbers_in(text: str) -> set[float]:
    """Every number that appears in the text, in any common format.
    A token that can be read two ways (23.000) contributes both readings: this is a 'was it printed' check."""
    found = set()
    for m in re.finditer(rf"(?<!\w)({AMOUNT_RE})(\s?[kK])?(?!\w)", text):
        values: set[float] = set()
        if (parsed := parse_amount(m.group(1))) is not None:
            values.add(parsed)
        try:
            values.add(float(m.group(1).replace(",", "")))
        except ValueError:
            pass
        for value in values:
            found.add(value * 1000 if m.group(2) else value)
    return found


# ---- dates -------------------------------------------------------------------------------------
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_DATE_RE = (r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/\-.]\d{1,2}[/\-.]\d{4}"
            r"|\d{1,2}[ \-][A-Za-z]{3,9}\.?[ \-,]*\d{4}|[A-Za-z]{3,9}\.? \d{1,2},? \d{4}")


def _valid(y: int, m: int, d: int):
    return (y, m, d) if 1 <= m <= 12 and 1 <= d <= 31 else None


def date_of(s: str):
    """(year, month, day) from '2025-03-12', '12/03/2025' (day first), '12-03-25', '12 March 2025', '12-Mar-2025',
    'March 12, 2025'. A time after the date is ignored. None if not recognised."""
    s = s.strip().casefold()
    if m := re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", s):
        return _valid(int(m[1]), int(m[2]), int(m[3]))
    if m := re.search(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4}|\d{2})\b", s):   # day first; 05-01-16 means 2016
        year = int(m[3]) + (2000 if len(m[3]) == 2 else 0)
        return _valid(year, int(m[2]), int(m[1]))
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


# ---- text --------------------------------------------------------------------------------------
CUSTOMER_LABELS = {"bill to", "billed to", "customer", "client", "to"}


def customers_in(text: str) -> set[str]:
    """Names printed after labels like 'Bill to:' or 'Customer:'. Used to catch supplier/customer mix-ups."""
    out = set()
    for line in text.splitlines():
        if ":" in line:
            label, value = line.split(":", 1)
            if norm(label) in CUSTOMER_LABELS:
                out.add(norm(value))
    return out


# ---- per-run state and the tools every run shares ----------------------------------------------
class RunState:
    """What the tools share during one document. Lives in code, not in the model's head."""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.validated: dict | None = None   # set only when validation passed
        self.outcome: dict | None = None     # set by save_result or escalate
        self.failures = 0                    # failed validation calls so far
        self.known_supplier: str | None = None   # set by the retrieve step when the supplier master matched confidently


def make_common_tools(data_dir: Path, state: RunState, out_dir: Path | None):
    """read_document, save_result and escalate, plus two helpers the validator uses.
    Returns (check_id, reject, read_document, save_result, escalate)."""

    def check_id(doc_id: str) -> str | None:
        # Untrusted input: the model chose this string. Allow only the document we are working on.
        if not isinstance(doc_id, str) or not re.fullmatch(r"[A-Za-z0-9_\-]+", doc_id):
            return "ERROR: invalid document id."
        if doc_id != state.doc_id:
            return f"ERROR: you are working on {state.doc_id!r}, not {doc_id!r}."
        return None

    def reject(problems: list[str]) -> str:
        """Common path for every failed validation. Counts failures and escalates in code if stuck."""
        state.failures += 1
        text = "PROBLEMS:\n- " + "\n- ".join(problems)
        if state.failures >= MAX_VALIDATION_FAILURES:
            state.outcome = {
                "status": "escalated",
                "reason": f"validation failed {state.failures} times. Last problems: " + "; ".join(problems)[:250],
            }
            text += "\nToo many failed attempts: the document was escalated to a human reviewer automatically."
        return text

    def read_document(doc_id: str) -> str:
        """Read the full text of a document.

        Args:
            doc_id (str): The document id given in the task.

        Returns:
            str: The text of the document.
        """
        if (err := check_id(doc_id)) is not None:
            return err
        path = data_dir / f"{doc_id}.txt"
        if not path.is_file():
            return f"ERROR: document {doc_id!r} not found."
        return path.read_text(encoding="utf-8")

    def save_result(doc_id: str) -> str:
        """Save the validated result. Only works after the validate tool returned OK.

        Args:
            doc_id (str): The document id.

        Returns:
            str: Confirmation, or an error.
        """
        if (err := check_id(doc_id)) is not None:
            return err
        if state.validated is None:
            return "ERROR: nothing validated yet. Call the validate tool first and get OK."
        state.outcome = {"status": "saved", "fields": state.validated}
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{doc_id}.json").write_text(json.dumps(state.validated, indent=2))
        return "SAVED. You are done."

    def escalate(doc_id: str, reason: str) -> str:
        """Send the document to a human reviewer instead of saving it.

        Args:
            doc_id (str): The document id.
            reason (str): A short reason, for example total missing.

        Returns:
            str: Confirmation.
        """
        if (err := check_id(doc_id)) is not None:
            return err
        state.outcome = {"status": "escalated", "reason": str(reason)[:300]}
        return "ESCALATED to a human reviewer. You are done."

    return check_id, reject, read_document, save_result, escalate
