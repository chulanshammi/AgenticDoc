# AgenticDoc

A small agent that reads text documents (payslips and invoices), extracts structured fields with a local LLM, and checks every answer against the document before saving it. When a document cannot be validated, it is escalated to a human reviewer instead of being guessed.

It runs fully locally with [Ollama](https://ollama.com) (default model `qwen3.5:0.8b`). No API keys.

## How it works

```
document.txt -> classify -> payslip agent -> read -> validate -> save
                         \-> invoice agent         \-> (problems? fix and retry) -> escalate
                         \-> unknown type -> escalate
```

1. **Classify** decides if the document is a payslip or an invoice (rules first, model as a second opinion).
2. A **specialist agent** gets four tools and must answer only by calling them:
   - `read_document` returns the document text
   - `validate_payslip` / `validate_invoice` checks the fields against a Pydantic schema and against the text itself
   - `save_result` writes the JSON, and only works after validation returned OK
   - `escalate` sends the document to a human with a reason
3. **Guardrails live in code, not in the prompt.** Every extracted value must actually appear in the document ("grounding"), totals must add up, the model can only touch the document it was given, and after 3 failed validations the code escalates on its own.

Function calling uses Ollama's native tool support: plain Python functions are passed as `tools=[...]`, and the schema is built from their type hints and docstrings.

## Two implementations of the same agent

| | Files | Control flow |
|---|---|---|
| Plain Python | `agent.py`, `router.py` | a hand-written `for` loop |
| LangGraph | `graph_agent.py` | explicit graph nodes and edges |

Both share the same tools, prompts and validators, and `graph_agent.route` is a drop-in replacement for `router.route`. Print the graph with `python graph_agent.py --graph` (Mermaid output).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install ollama pydantic langgraph

ollama pull qwen3.5:0.8b     # the name must match MODEL in extracter.py
ollama serve                 # if it is not already running
```

## Usage

Run these from the project folder (use `.venv/bin/python` if the venv is not activated).

```bash
# one document, auto-detects payslip or invoice (-v prints every tool call)
python router.py doc_000 -v
python graph_agent.py doc_000 -v

# save the extracted JSON
python graph_agent.py doc_000 --out output

# classifier rules only, no model call
python graph_agent.py doc_000 --rules-only
```

Each run returns a JSON result with `status` of `saved`, `escalated` or `failed`, plus the steps taken and the tools called.

## Test data and evaluation

Documents are synthetic and generated with a fixed seed, with the expected answers in `truth.json`.

```bash
python payslip_generator.py --n 60 --out data          # payslips only
python generator.py --n 100 --out data_mixed           # payslips and invoices mixed

python evaluate_agent.py --data data --show-errors     # payslip agent
python evaluate_router.py --data data_mixed_test --show-errors   # full router
python evaluator.py --data data                        # one-shot extractor (no agent)
```

Add `--limit 10` for a quick run. The model is small, so a full run takes a while.

To evaluate the LangGraph version, change `from router import route` to `from graph_agent import route` in `evaluate_router.py`.

## Project layout

| File | Purpose |
|---|---|
| `agent.py` | Payslip agent, shared tools and the plain-Python agent loop |
| `invoice_agent.py` | Invoice schema, validator and tools. Reuses the loop in `agent.py` |
| `router.py` | Classifies a document and hands it to the right agent |
| `graph_agent.py` | LangGraph version of the agent loop and the router |
| `classifier.py` | Rules plus model document-type classifier |
| `extracter.py` | `Payslip` schema, grounding checks and a one-shot (non-agent) extractor |
| `evaluator.py`, `evaluate_agent.py`, `evaluate_router.py` | Accuracy evaluation at each level |
| `generator.py`, `payslip_generator.py` | Synthetic document generators |
| `data*/` | Generated documents and `truth.json` |
| `*_results.json` | Saved evaluation outputs |

## Tuning

Limits are constants at the top of `agent.py`: `MAX_STEPS` (model calls per document), `MAX_NUDGES` (reminders when the model replies in text) and `MAX_VALIDATION_FAILURES` (failed validations before automatic escalation). The model name is `MODEL` in `extracter.py`.
