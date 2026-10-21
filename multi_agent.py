"""Multi-agent receipt processing: a supervisor, an extractor and a reviewer, as one LangGraph graph.

    START -> supervisor -> extractor -> supervisor -> reviewer -> supervisor -> finalize -> END
                  ^                                                  |
                  +------------------ revise (with feedback) --------+

supervisor  decides who works next. Plain code, no model: the routing is a few rules, and a model that routes is one
            more thing that can go wrong. It also enforces the limit on revisions.
extractor   the LLM agent from graph_agent.py (read, validate, save). On a revision it also gets the reviewer's objections.
reviewer    reviewer.py: checks that each answer plays the right role on the receipt (not just that it is printed).
            Code, not a model: these checks are exact, and they must give the same verdict on every run.
finalize    writes the accepted result, or escalates when the reviewer's objections survive the revisions.

The workers never call each other. They read and write the shared state and hand control back to the supervisor.
"""
import argparse
import json
import operator
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

import graph_agent
from reviewer import review

MAX_REVISIONS = 2   # extra attempts the extractor gets after the reviewer objects


class TeamState(TypedDict):
    doc_id: str
    data_dir: str
    out_dir: str | None
    use_rag: bool
    next: str                                      # the supervisor's decision
    extraction: dict | None                        # the extractor's latest result
    reviewed: bool                                 # has the reviewer seen that extraction?
    issues: list[dict]                             # the reviewer's objections to it
    revisions: int
    steps: int
    calls: Annotated[list[str], operator.add]
    final: dict | None


def feedback_text(extraction: dict, issues: list[dict]) -> str:
    return ("A reviewer rejected your previous answer.\n"
            f"Previous answer: {json.dumps(extraction['fields'])}\n"
            "Problems:\n" + "\n".join(f"- {i['message']}" for i in issues) + "\n"
            "Call validate_receipt again with ALL fields (keep the ones that were fine), corrected where needed.")


def build_team():
    def supervisor(s: TeamState) -> dict:
        ex = s["extraction"]
        if ex is None:
            return {"next": "extractor"}
        if not s["reviewed"]:
            return {"next": "reviewer" if ex["status"] == "saved" else "finalize"}   # nothing to review if it did not save
        if not s["issues"]:
            return {"next": "finalize"}
        return {"next": "extractor" if s["revisions"] < MAX_REVISIONS else "finalize"}

    def extractor(s: TeamState) -> dict:
        revising = s["extraction"] is not None
        feedback = feedback_text(s["extraction"], s["issues"]) if revising else None
        # out_dir is None: the extractor only proposes. The result is written once, by finalize, after the review.
        r = graph_agent.run_receipt_agent(s["doc_id"], s["data_dir"], None, s["use_rag"], feedback)
        return {"extraction": r, "reviewed": False, "issues": [], "revisions": s["revisions"] + revising,
                "steps": s["steps"] + r.get("steps", 0), "calls": r.get("calls", [])}

    def reviewer(s: TeamState) -> dict:
        text = (Path(s["data_dir"]) / f"{s['doc_id']}.txt").read_text(encoding="utf-8")
        issues = [{"field": i.field, "message": i.message} for i in review(s["extraction"]["fields"], text)]
        if graph_agent.VERBOSE:
            print(f"  reviewer (after attempt {s['revisions'] + 1}): " + (" | ".join(i["message"][:90] for i in issues) or "no objections"))
        return {"reviewed": True, "issues": issues}

    def finalize(s: TeamState) -> dict:
        ex, issues = s["extraction"], s["issues"]
        if ex["status"] == "saved" and issues:   # the objections survived every revision: a human decides
            result = {"doc_id": s["doc_id"], "status": "escalated",
                      "reason": "reviewer: " + "; ".join(i["message"] for i in issues)[:300]}
        else:
            result = {k: v for k, v in ex.items() if k not in ("steps", "calls")}
            if ex["status"] == "saved" and s["out_dir"]:
                out = Path(s["out_dir"])
                out.mkdir(parents=True, exist_ok=True)
                (out / f"{s['doc_id']}.json").write_text(json.dumps(ex["fields"], indent=2))
        return {"final": {**result, "steps": s["steps"], "calls": s["calls"], "revisions": s["revisions"]}}

    g = StateGraph(TeamState)
    g.add_node("supervisor", supervisor)
    g.add_node("extractor", extractor)
    g.add_node("reviewer", reviewer)
    g.add_node("finalize", finalize)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["next"], ["extractor", "reviewer", "finalize"])
    g.add_edge("extractor", "supervisor")
    g.add_edge("reviewer", "supervisor")
    g.add_edge("finalize", END)
    return g.compile()


TEAM = build_team()


def run_multi_agent(doc_id: str, data_dir: str = "data_rows", out_dir: str | None = None, use_rag: bool = False) -> dict:
    """Same contract as run_receipt_agent: a dict with status saved / escalated / failed."""
    final = TEAM.invoke({"doc_id": doc_id, "data_dir": data_dir, "out_dir": out_dir, "use_rag": use_rag, "next": "",
                         "extraction": None, "reviewed": False, "issues": [], "revisions": 0, "steps": 0, "calls": [],
                         "final": None})
    return final["final"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_id", nargs="?")
    ap.add_argument("--data", default="data_rows")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rag", action="store_true")
    ap.add_argument("--graph", action="store_true", help="print the graph as a Mermaid diagram and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if a.graph:
        print(TEAM.get_graph().draw_mermaid())
    elif a.doc_id:
        graph_agent.VERBOSE = a.verbose
        print(json.dumps(run_multi_agent(a.doc_id, a.data, a.out, a.rag), indent=2))
    else:
        ap.error("give a doc_id, or --graph")
