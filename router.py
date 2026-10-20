import argparse
import json
from pathlib import Path

import agent
from classifier import classify
from invoice_agent import run_invoice_agent

HANDLERS = {
    "payslip": agent.run_agent,
    "invoice": run_invoice_agent,
}


def route(doc_id: str, data_dir: str = "data_mixed", out_dir: str | None = None, use_model: bool = True) -> dict:
    """Always returns a dict with status saved / escalated / failed, plus doc_type and the classifier's answer."""
    text = (Path(data_dir) / f"{doc_id}.txt").read_text(encoding="utf-8")
    c = classify(text, use_model=use_model)
    doc_type = c["doc_type"] if c["ok"] else None
    handler = HANDLERS.get(doc_type or "")
    if handler is None:
        why = c["reason"] if not c["ok"] else f"classified as {doc_type}: no specialist agent for this type"
        return {"doc_id": doc_id, "doc_type": doc_type, "status": "escalated", "reason": why,
                "steps": 0, "calls": [], "classifier": c}
    result = handler(doc_id, data_dir, out_dir)
    return {**result, "doc_type": doc_type, "classifier": c}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_id")
    ap.add_argument("--data", default="data_mixed")
    ap.add_argument("--out", default=None, help="folder to write saved JSON into")
    ap.add_argument("--rules-only", action="store_true", help="classify with the rules only (no model call)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    agent.VERBOSE = a.verbose
    print(json.dumps(route(a.doc_id, a.data, a.out, use_model=not a.rules_only), indent=2))