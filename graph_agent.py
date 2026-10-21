"""LangGraph agent loop for one receipt, with a LangChain chat model and tools.

    [retrieve] -> call_model -> run_tools -> check -> call_model ... -> END
                  call_model -> nudge (model replied in text instead of calling a tool) -> call_model

`retrieve` (only with use_rag) looks the receipt up in the supplier master, a Chroma vector index of known suppliers.

The tools and their guardrails are in receipt_agent.py and common.py. This file is only the control flow.
"""
import argparse
import functools
import json
import os
from pathlib import Path
from typing import Annotated, Any, TypedDict

import ollama
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from common import MODEL, RunState
from receipt_agent import SYSTEM_PROMPT, make_receipt_tools

MAX_STEPS = 8      # model calls per document
MAX_NUDGES = 2     # times we remind a model that answered in text instead of calling a tool
RECURSION_LIMIT = 60   # MAX_STEPS model calls x up to 4 nodes per step, with headroom
VERBOSE = False

# A thinking model (deepseek-r1) spends tokens on its reasoning before it answers: give it room, and time.
NUM_PREDICT = int(os.environ.get("AGENT_NUM_PREDICT", 500))
THINKING = os.environ.get("AGENT_THINKING", "off") == "on"
_OPTIONS = dict(model=MODEL, temperature=0, num_predict=NUM_PREDICT, client_kwargs={"timeout": int(os.environ.get("AGENT_TIMEOUT", 90))})
_llm = ChatOllama(reasoning=THINKING, **_OPTIONS)
_llm_no_think = ChatOllama(**_OPTIONS)   # for a model that rejects the `think` option


@functools.cache
def _supplier_db():
    from supplier_index import _db   # imported here so the agent runs without Chroma when RAG is off
    return _db()


def _short(value: Any, n: int = 160) -> str:
    s = str(value).replace("\n", " | ")
    return s if len(s) <= n else s[: n - 3] + "..."


def _calls(messages: list[AnyMessage]) -> list[str]:
    return [c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]   # appended to, never replaced
    step: int
    nudges: int
    result: dict | None                                   # set when the run is over


def build_graph(doc_id: str, data_dir: str, out_dir: str | None, use_rag: bool = False, feedback: str | None = None):
    """One compiled graph per document: the tools close over this document's RunState."""
    run = RunState(doc_id)
    tools = make_receipt_tools(Path(data_dir), run, Path(out_dir) if out_dir else None)
    registry = {t.name: t for t in tools}
    model, model_no_think = _llm.bind_tools(tools), _llm_no_think.bind_tools(tools)

    def finish(s: AgentState, **fields) -> dict:
        return {"result": {"doc_id": doc_id, **fields, "steps": s["step"], "calls": _calls(s["messages"])}}

    def retrieve(s: AgentState) -> dict:
        """RAG step: ask the supplier master which known company this receipt is from."""
        from supplier_index import known_supplier
        text = (Path(data_dir) / f"{doc_id}.txt").read_text(encoding="utf-8")
        run.known_supplier = known_supplier(_supplier_db(), text)   # validate_receipt reads it from the RunState
        if VERBOSE:
            print(f"  retrieve: supplier master -> {run.known_supplier!r}")
        # The model is not told. A first version added a message ("use exactly this supplier_name") and the model
        # then sent a minimal call without date and invoice number (date 87% -> 62% on 60 receipts). The rule in
        # validate_receipt gives the same name as feedback, and only when the model's supplier is different.
        return {}

    def call_model(s: AgentState) -> dict:
        try:
            message = model.invoke(s["messages"])
        except ollama.ResponseError:
            message = model_no_think.invoke(s["messages"])
        return {"messages": [message], "step": s["step"] + 1}

    def run_tools(s: AgentState) -> dict:
        # One at a time, in order. LangGraph's prebuilt ToolNode runs the calls of one message in parallel,
        # which is wrong here: validate_receipt and save_result share a RunState and save needs validate first.
        out = []
        for call in s["messages"][-1].tool_calls:
            tool = registry.get(call["name"])
            ids = dict(tool_call_id=call["id"], name=call["name"])
            if tool is None:
                out.append(ToolMessage(f"ERROR: unknown tool {call['name']!r}. Available: {', '.join(registry)}.", **ids))
                continue
            try:
                out.append(tool.invoke(call))   # a ToolCall in, a ToolMessage out
            except Exception as e:   # bad arguments, or a tool bug: neither may crash the whole batch
                out.append(ToolMessage(f"ERROR: {call['name']} failed: {type(e).__name__}: {e}", status="error", **ids))
        return {"messages": out}

    def check(s: AgentState) -> dict:
        """After the tools ran: print them if asked, and stop if save_result / escalate (or the failure limit) ended the run."""
        if VERBOSE:
            asked = next(m for m in reversed(s["messages"]) if isinstance(m, AIMessage))
            answers = {m.tool_call_id: m.content for m in s["messages"] if isinstance(m, ToolMessage)}
            for c in asked.tool_calls:
                print(f"  step {s['step']}: {c['name']}({_short(c['args'])})\n           -> {_short(answers.get(c['id'], ''))}")
        return finish(s, **run.outcome) if run.outcome is not None else {}

    def nudge(s: AgentState) -> dict:
        if VERBOSE:
            print(f"  step {s['step']}: model said (no tool): {_short(s['messages'][-1].content)}")
        nudges = s["nudges"] + 1
        if nudges > MAX_NUDGES:
            return {"nudges": nudges, **finish(s, status="failed", reason="model did not call a tool")}
        if run.failures:
            text = ("Do not reply with text. Call the validate tool with corrected values, or call escalate "
                    "if the total is missing from the document.")
        else:
            text = "Call one of the tools now. Do not reply with text."
        return {"nudges": nudges, "messages": [HumanMessage(text)]}

    def give_up(s: AgentState) -> dict:
        return finish(s, status="failed", reason=f"hit step limit ({MAX_STEPS})")

    def after_model(s: AgentState) -> str:
        return "run_tools" if s["messages"][-1].tool_calls else "nudge"

    def continue_or_stop(s: AgentState) -> str:
        if s["result"] is not None:
            return END
        return "give_up" if s["step"] >= MAX_STEPS else "call_model"

    g = StateGraph(AgentState)
    g.add_node("call_model", call_model)
    g.add_node("run_tools", run_tools)
    g.add_node("check", check)
    g.add_node("nudge", nudge)
    g.add_node("give_up", give_up)
    if use_rag:
        g.add_node("retrieve", retrieve)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "call_model")
    else:
        g.add_edge(START, "call_model")
    g.add_conditional_edges("call_model", after_model, ["run_tools", "nudge"])
    g.add_edge("run_tools", "check")
    g.add_conditional_edges("check", continue_or_stop, [END, "give_up", "call_model"])
    g.add_conditional_edges("nudge", continue_or_stop, [END, "give_up", "call_model"])
    g.add_edge("give_up", END)

    initial: AgentState = {
        "messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(f"Process document {doc_id}.")]
                    + ([HumanMessage(feedback)] if feedback else []),   # a reviewer's objections, on a second attempt
        "step": 0, "nudges": 0, "result": None,
    }
    return g.compile(), initial


def run_receipt_agent(doc_id: str, data_dir: str = "data_real", out_dir: str | None = None, use_rag: bool = False,
                      feedback: str | None = None) -> dict:
    """Process one receipt. Always returns a dict with status saved / escalated / failed."""
    graph, initial = build_graph(doc_id, data_dir, out_dir, use_rag, feedback)
    return graph.invoke(initial, {"recursion_limit": RECURSION_LIMIT})["result"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_id", nargs="?")
    ap.add_argument("--data", default="data_real")
    ap.add_argument("--out", default=None, help="folder to write saved JSON into")
    ap.add_argument("--rag", action="store_true", help="look the supplier up in the supplier master (needs supplier_index.py build)")
    ap.add_argument("--graph", action="store_true", help="print the graph as a Mermaid diagram and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if a.graph:
        print(build_graph("doc", a.data, None, a.rag)[0].get_graph().draw_mermaid())
    elif a.doc_id:
        VERBOSE = a.verbose
        print(json.dumps(run_receipt_agent(a.doc_id, a.data, a.out, a.rag), indent=2))
    else:
        ap.error("give a doc_id, or --graph")
