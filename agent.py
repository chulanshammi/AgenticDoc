import argparse
import json
import re
from pathlib import Path
from typing import Any

import ollama
from pydantic import ValidationError

from extracter import MODEL, Payslip, grounding_problems

MAX_STEPS = 8      # model calls per document
MAX_NUDGES = 2     # times we remind a model that answered in text instead of calling a tool
MAX_VALIDATION_FAILURES = 3   # after this many failed validations, code escalates the document itself
VERBOSE = False

_client = ollama.Client(timeout=90)

SYSTEM_PROMPT = """You process one payslip document by calling tools. Follow these steps:
1. Call read_document with the document id.
2. Read the text and call validate_payslip with the fields exactly as printed in the document.
3. If validate_payslip reports PROBLEMS, fix them and call validate_payslip again. Most problems are fixable:
   a wrong number format, a wrong month or year, a wrong currency. Correct the value and try again.
4. When validate_payslip says OK, call save_result.
Call escalate only if, after trying to fix the problem, a required value is truly missing from the document
(for example there is no employee name) or the printed numbers contradict each other.
Rules: never invent values; copy numbers exactly as printed (no calculating). Pass amounts as plain numbers
with no currency symbols or commas, and expand shorthand (150k means 150000). Currency is a 3-letter code
such as LKR, GBP or USD. Copy the pay period exactly as printed in the document, in the same format.
Always answer by calling a tool, never with plain text."""


def to_number(value: Any) -> float | None:
    """Accept 18000, '18,000.00', 'Rs.18,000.00', '£1,234.50', '1,50,000.00', '150k'. None if it is not a number.
    Code normalises the format; the grounding check still proves the value is printed in the document."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s?([kK])?(?![\w])", value)
        if m:
            try:
                return float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)
            except ValueError:
                return None
    return None


class RunState:
    """What the tools share during one document. Lives in code, not in the model's head."""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.validated: dict | None = None   # set only when validation passed
        self.outcome: dict | None = None     # set by save_result or escalate
        self.failures = 0                    # failed validate_payslip calls so far


def make_common_tools(data_dir: Path, state: RunState, out_dir: Path | None):
    """The tools every document type shares: read, save, escalate. Plus two helpers the validators use.
    Returns (check_id, reject, read_document, save_result, escalate)."""

    def check_id(doc_id: str) -> str | None:
        # Untrusted input: the model chose this string. Allow only the document we are working on.
        if not isinstance(doc_id, str) or not re.fullmatch(r"[A-Za-z0-9_\-]+", doc_id):
            return "ERROR: invalid document id."
        if doc_id != state.doc_id:
            return f"ERROR: you are working on {state.doc_id!r}, not {doc_id!r}."
        return None

    def reject(problems: list[str], escalate_hint: bool = True) -> str:
        """Common path for every failed validation. Counts failures and escalates in code if stuck.
        escalate_hint=False when the caller supplies its own, more specific guidance."""
        state.failures += 1
        text = "PROBLEMS:\n- " + "\n- ".join(problems)
        if escalate_hint and any(" != " in p for p in problems):   # the arithmetic check failed
            text += (
                "\nHINT: if these are the numbers printed in the document, the document itself is "
                "inconsistent. Do not recalculate or change printed numbers. Call escalate."
            )
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
            reason (str): A short reason, for example employee name missing.

        Returns:
            str: Confirmation.
        """
        if (err := check_id(doc_id)) is not None:
            return err
        state.outcome = {"status": "escalated", "reason": str(reason)[:300]}
        return "ESCALATED to a human reviewer. You are done."

    return check_id, reject, read_document, save_result, escalate


def make_tools(data_dir: Path, state: RunState, out_dir: Path | None):
    """Payslip tools for one run. They are closures so they can see `state`."""
    check_id, reject, read_document, save_result, escalate = make_common_tools(data_dir, state, out_dir)

    def validate_payslip(
        doc_id: str,
        employer_name: str,
        employee_name: str,
        pay_period: str,
        gross_pay: float,
        total_deductions: float,
        net_pay: float,
        currency: str = "",
    ) -> str:
        """Check extracted payslip fields against the schema and against the document text.

        Args:
            doc_id (str): The document id.
            employer_name (str): Employer name exactly as printed.
            employee_name (str): Employee name exactly as printed, or empty string if absent.
            pay_period (str): The pay period copied exactly as printed in the document.
            gross_pay (float): Gross pay as a plain number.
            total_deductions (float): Total deductions as a plain number.
            net_pay (float): Net pay as a plain number.
            currency (str): Three-letter currency code such as LKR, GBP or USD.

        Returns:
            str: OK, or a list of problems to fix.
        """
        if (err := check_id(doc_id)) is not None:
            return err
        state.validated = None  # any new attempt invalidates the old one
        gross, deductions, net = to_number(gross_pay), to_number(total_deductions), to_number(net_pay)
        if gross is None or deductions is None or net is None:
            raw = {"gross_pay": gross_pay, "total_deductions": total_deductions, "net_pay": net_pay}
            unreadable = [k for k, v in raw.items() if to_number(v) is None]
            return reject([
                f"{k} {raw[k]!r} is not a number. Give the amount as printed, as a plain number such as 18000.50."
                for k in unreadable
            ])
        try:
            payslip = Payslip(
                employer_name=employer_name,
                employee_name=employee_name,
                pay_period=pay_period,
                gross_pay=gross,
                total_deductions=deductions,
                net_pay=net,
                currency=currency or None,
            )
        except ValidationError as e:
            msgs = [f"{'.'.join(str(x) for x in d['loc']) or 'document'}: {d['msg']}" for d in e.errors()]
            return reject(msgs)
        text = (data_dir / f"{doc_id}.txt").read_text(encoding="utf-8")
        problems = grounding_problems(payslip, text)
        if problems:
            return reject(problems)
        state.validated = payslip.model_dump()
        return "OK. Now call save_result."

    return [read_document, validate_payslip, save_result, escalate]


def _chat(messages: list[Any], tools: list[Any]):
    try:
        return _client.chat(
            model=MODEL,
            messages=messages,
            tools=tools,
            think=False,
            options={"temperature": 0, "num_predict": 500},
        )
    except (TypeError, ollama.ResponseError):
        return _client.chat(
            model=MODEL,
            messages=messages,
            tools=tools,
            options={"temperature": 0, "num_predict": 500},
        )


def _short(value: Any, n: int = 160) -> str:
    s = str(value).replace("\n", " | ")
    return s if len(s) <= n else s[: n - 3] + "..."


def run_agent(doc_id: str, data_dir: str = "data", out_dir: str | None = None,
              tools_factory: Any = None, system_prompt: str | None = None) -> dict:
    """Process one document. Always returns a dict with status saved / escalated / failed.
    tools_factory / system_prompt let another document type (invoices) reuse this same loop."""
    state = RunState(doc_id)
    tools = (tools_factory or make_tools)(Path(data_dir), state, Path(out_dir) if out_dir else None)
    registry = {fn.__name__: fn for fn in tools}

    messages: list[Any] = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": f"Process document {doc_id}."},
    ]
    calls: list[str] = []
    nudges = 0

    for step in range(1, MAX_STEPS + 1):
        response = _chat(messages, tools)
        message = response.message
        messages.append(message)

        if not message.tool_calls:
            # Model answered in plain text instead of acting.
            if VERBOSE:
                print(f"  step {step}: model said (no tool): {_short(message.content)}")
            nudges += 1
            if nudges > MAX_NUDGES:
                return {"doc_id": doc_id, "status": "failed", "reason": "model did not call a tool",
                        "steps": step, "calls": calls}
            if state.failures:
                nudge = ("Do not reply with text. Call the validate tool with corrected values, or call escalate "
                         "if the printed numbers contradict each other or a required value is missing.")
            else:
                nudge = "Call one of the tools now. Do not reply with text."
            messages.append({"role": "user", "content": nudge})
            continue

        for call in message.tool_calls:
            name = call.function.name
            args = dict(call.function.arguments or {})
            calls.append(name)
            fn = registry.get(name)
            if fn is None:
                result = f"ERROR: unknown tool {name!r}. Available: {', '.join(registry)}."
            else:
                try:
                    result = fn(**args)
                except TypeError as e:   # wrong or missing arguments
                    result = f"ERROR: bad arguments for {name}: {e}"
                except Exception as e:   # a tool bug must not crash the whole batch
                    result = f"ERROR: {name} failed: {type(e).__name__}: {e}"
            if VERBOSE:
                print(f"  step {step}: {name}({_short(args)})\n           -> {_short(result)}")
            messages.append({"role": "tool", "tool_name": name, "content": str(result)})

        if state.outcome is not None:
            return {"doc_id": doc_id, **state.outcome, "steps": step, "calls": calls}

    return {"doc_id": doc_id, "status": "failed", "reason": f"hit step limit ({MAX_STEPS})",
            "steps": MAX_STEPS, "calls": calls}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_id")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=None, help="folder to write saved JSON into")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    VERBOSE = a.verbose
    print(json.dumps(run_agent(a.doc_id, a.data, a.out), indent=2))