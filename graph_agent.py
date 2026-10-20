"""LangGraph version of agent.py + router.py.

Same tools, same prompts, same guardrails. Only the control flow moved from a hand-written
`for` loop into explicit graph nodes and edges.

Router graph:   classify -> (payslip_agent | invoice_agent | no_handler) -> END
Agent graph:    call_model -> run_tools -> call_model ... -> END
                call_model -> nudge (model replied in text) -> call_model
"""
import argparse
import json
import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

import agent
from agent import MAX_NUDGES, MAX_STEPS, RunState, _chat, _short
from classifier import classify
from invoice_agent import SYSTEM_PROMPT as INVOICE_PROMPT, make_invoice_tools

RECURSION_LIMIT = 60   # MAX_STEPS model calls x up to 3 nodes per step, with headroom


class AgentState(TypedDict):
    messages: Annotated[list[Any], operator.add]   # appended to, never replaced
    calls: Annotated[list[str], operator.add]
    step: int
    nudges: int
    result: dict | None                            # set when the run is over


def build_agent_graph(doc_id: str, data_dir: str, out_dir: str | None, tools_factory, system_prompt: str):
    """One compiled graph per document: the tools close over this document's RunState."""
    run = RunState(doc_id)
    tools = tools_factory(Path(data_dir), run, Path(out_dir) if out_dir else None)
    registry = {fn.__name__: fn for fn in tools}

    def call_model(s: AgentState) -> dict:
        step = s["step"] + 1
        message = _chat(s["messages"], tools).message
        return {"messages": [message], "step": step}

    def run_tools(s: AgentState) -> dict:
        out, names = [], []
        for call in s["messages"][-1].tool_calls:
            name = call.function.name
            args = dict(call.function.arguments or {})
            names.append(name)
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
            if agent.VERBOSE:
                print(f"  step {s['step']}: {name}({_short(args)})\n           -> {_short(result)}")
            out.append({"role": "tool", "tool_name": name, "content": str(result)})
        update: dict = {"messages": out, "calls": names}
        if run.outcome is not None:   # save_result or escalate ran (or too many failed validations)
            update["result"] = {"doc_id": doc_id, **run.outcome, "steps": s["step"], "calls": s["calls"] + names}
        return update

    def nudge(s: AgentState) -> dict:
        # Model answered in plain text instead of acting.
        if agent.VERBOSE:
            print(f"  step {s['step']}: model said (no tool): {_short(s['messages'][-1].content)}")
        nudges = s["nudges"] + 1
        if nudges > MAX_NUDGES:
            return {"nudges": nudges, "result": {"doc_id": doc_id, "status": "failed",
                                                 "reason": "model did not call a tool",
                                                 "steps": s["step"], "calls": s["calls"]}}
        if run.failures:
            text = ("Do not reply with text. Call the validate tool with corrected values, or call escalate "
                    "if the printed numbers contradict each other or a required value is missing.")
        else:
            text = "Call one of the tools now. Do not reply with text."
        return {"nudges": nudges, "messages": [{"role": "user", "content": text}]}

    def give_up(s: AgentState) -> dict:
        return {"result": {"doc_id": doc_id, "status": "failed", "reason": f"hit step limit ({MAX_STEPS})",
                           "steps": MAX_STEPS, "calls": s["calls"]}}

    def after_model(s: AgentState) -> str:
        return "run_tools" if s["messages"][-1].tool_calls else "nudge"

    def continue_or_stop(s: AgentState) -> str:
        if s["result"] is not None:
            return END
        return "give_up" if s["step"] >= MAX_STEPS else "call_model"

    g = StateGraph(AgentState)
    g.add_node("call_model", call_model)
    g.add_node("run_tools", run_tools)
    g.add_node("nudge", nudge)
    g.add_node("give_up", give_up)
    g.add_edge(START, "call_model")
    g.add_conditional_edges("call_model", after_model, ["run_tools", "nudge"])
    g.add_conditional_edges("run_tools", continue_or_stop, [END, "give_up", "call_model"])
    g.add_conditional_edges("nudge", continue_or_stop, [END, "give_up", "call_model"])
    g.add_edge("give_up", END)

    initial: AgentState = {
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": f"Process document {doc_id}."}],
        "calls": [], "step": 0, "nudges": 0, "result": None,
    }
    return g.compile(), initial


def run_graph_agent(doc_id: str, data_dir: str, out_dir: str | None, tools_factory, system_prompt: str) -> dict:
    graph, initial = build_agent_graph(doc_id, data_dir, out_dir, tools_factory, system_prompt)
    final = graph.invoke(initial, {"recursion_limit": RECURSION_LIMIT})
    return final["result"]


# ---------------------------------------------------------------- router graph

class RouterState(TypedDict):
    doc_id: str
    data_dir: str
    out_dir: str | None
    use_model: bool
    doc_type: str | None
    classifier: dict | None
    result: dict | None


def build_router_graph():
    def classify_node(s: RouterState) -> dict:
        text = (Path(s["data_dir"]) / f"{s['doc_id']}.txt").read_text(encoding="utf-8")
        c = classify(text, use_model=s["use_model"])
        return {"classifier": c, "doc_type": c["doc_type"] if c["ok"] else None}

    def payslip_agent(s: RouterState) -> dict:
        r = run_graph_agent(s["doc_id"], s["data_dir"], s["out_dir"], agent.make_tools, agent.SYSTEM_PROMPT)
        return {"result": {**r, "doc_type": s["doc_type"], "classifier": s["classifier"]}}

    def invoice_agent(s: RouterState) -> dict:
        r = run_graph_agent(s["doc_id"], s["data_dir"], s["out_dir"], make_invoice_tools, INVOICE_PROMPT)
        return {"result": {**r, "doc_type": s["doc_type"], "classifier": s["classifier"]}}

    def no_handler(s: RouterState) -> dict:
        c = s["classifier"]
        why = c["reason"] if not c["ok"] else f"classified as {s['doc_type']}: no specialist agent for this type"
        return {"result": {"doc_id": s["doc_id"], "doc_type": s["doc_type"], "status": "escalated",
                           "reason": why, "steps": 0, "calls": [], "classifier": c}}

    def pick_agent(s: RouterState) -> str:
        return {"payslip": "payslip_agent", "invoice": "invoice_agent"}.get(s["doc_type"] or "", "no_handler")

    g = StateGraph(RouterState)
    g.add_node("classify", classify_node)
    g.add_node("payslip_agent", payslip_agent)
    g.add_node("invoice_agent", invoice_agent)
    g.add_node("no_handler", no_handler)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", pick_agent, ["payslip_agent", "invoice_agent", "no_handler"])
    for n in ("payslip_agent", "invoice_agent", "no_handler"):
        g.add_edge(n, END)
    return g.compile()


ROUTER = build_router_graph()


def route(doc_id: str, data_dir: str = "data_mixed", out_dir: str | None = None, use_model: bool = True) -> dict:
    """Drop-in replacement for router.route."""
    final = ROUTER.invoke({"doc_id": doc_id, "data_dir": data_dir, "out_dir": out_dir, "use_model": use_model,
                           "doc_type": None, "classifier": None, "result": None})
    return final["result"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_id", nargs="?")
    ap.add_argument("--data", default="data_mixed")
    ap.add_argument("--out", default=None, help="folder to write saved JSON into")
    ap.add_argument("--rules-only", action="store_true", help="classify with the rules only (no model call)")
    ap.add_argument("--graph", action="store_true", help="print the router graph as a Mermaid diagram and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if a.graph:
        print(ROUTER.get_graph().draw_mermaid())
    else:
        agent.VERBOSE = a.verbose
        print(json.dumps(route(a.doc_id, a.data, a.out, use_model=not a.rules_only), indent=2))
