import argparse
import json
import re
from pathlib import Path
from typing import Literal

import ollama
from pydantic import BaseModel, Field, ValidationError

from extracter import MODEL, _norm

MAX_ATTEMPTS = 2
_client = ollama.Client(timeout=90)


class Classification(BaseModel):
    doc_type: Literal["payslip", "invoice", "other"]
    evidence: str = Field(min_length=1, description="A short phrase copied exactly from the document that shows its type")


SYSTEM_PROMPT = (
    "You classify one business document. Answer with exactly one doc_type: "
    "payslip (a salary statement for an employee), "
    "invoice (a bill from a supplier asking a customer to pay), "
    "or other (anything else). "
    "evidence must be a short phrase of at most eight words copied exactly from the document. "
    "Return ONLY JSON matching the schema."
)

# Rules: each pattern counts once if it appears anywhere in the document.
PAYSLIP_CUES = [r"pay ?slip", r"pay advice", r"salary", r"pay period|period:", r"\bemployee\b|staff member",
                r"\bgross\b|total earnings", r"deductions|tax & contributions", r"net pay|take-home|amount paid",
                r"\bepf\b"]
INVOICE_CUES = [r"invoice", r"\bvat\b", r"subtotal|net amount", r"bill(?:ed)? to", r"\bcustomer\b",
                r"amount due|total due|grand total", r"\bsupplier\b|\bfrom:"]


def _count(patterns: list[str], text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))


def rules_label(text: str) -> str | None:
    """payslip / invoice / other from word counts alone. None when the rules themselves are unsure."""
    p, i = _count(PAYSLIP_CUES, text), _count(INVOICE_CUES, text)
    if p >= 3 and p > i:
        return "payslip"
    if i >= 3 and i > p:
        return "invoice"
    if p < 3 and i < 3:
        return "other"
    return None


def _ask_model(text: str) -> tuple[str | None, str]:
    """Returns (doc_type, evidence), or (None, reason) if the model never gave a usable answer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Document:\n\n{text}"},
    ]
    haystack = _norm(text)
    reason = "no answer"
    for _ in range(MAX_ATTEMPTS):
        try:
            resp = _client.chat(model=MODEL, messages=messages, format=Classification.model_json_schema(),
                                think=False, options={"temperature": 0, "num_predict": 250})
        except (TypeError, ollama.ResponseError):
            resp = _client.chat(model=MODEL, messages=messages, format=Classification.model_json_schema(),
                                options={"temperature": 0, "num_predict": 250})
        raw = resp["message"]["content"]
        try:
            c = Classification.model_validate_json(raw)
        except ValidationError as e:
            reason = f"invalid answer: {e.errors()[0]['msg']}"
            feedback = "Invalid answer. Return JSON with doc_type and evidence only."
        else:
            if _norm(c.evidence) in haystack:
                return c.doc_type, c.evidence
            reason = f"evidence {c.evidence!r} is not in the document"
            feedback = "The evidence is not in the document. Copy a short phrase exactly as printed."
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": feedback})
    return None, reason


def classify(text: str, use_model: bool = True) -> dict:
    """Always returns a dict: doc_type (payslip / invoice / other, or None when unsure), ok, model, rules, reason."""
    rules = rules_label(text)
    if rules is not None or not use_model:
        return {"doc_type": rules, "ok": rules is not None, "model": None, "rules": rules,
                "reason": "" if rules else "rules unsure"}
    model, info = _ask_model(text)     # only reached when the rules could not decide
    if model is None:
        return {"doc_type": None, "ok": False, "model": None, "rules": None, "reason": f"rules unsure; model: {info}"}
    return {"doc_type": model, "ok": True, "model": model, "rules": None, "reason": "", "evidence": info}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_id")
    ap.add_argument("--data", default="data_mixed")
    ap.add_argument("--rules-only", action="store_true")
    a = ap.parse_args()
    doc = (Path(a.data) / f"{a.doc_id}.txt").read_text(encoding="utf-8")
    print(json.dumps(classify(doc, use_model=not a.rules_only), indent=2))