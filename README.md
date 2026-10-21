# AgenticDoc

An agent that reads real scanned receipts and tax invoices (OCR text), extracts the key fields with a local LLM, and checks every answer against the document before saving it. If it cannot validate a document, it escalates it to a human instead of guessing.

Runs fully locally with [Ollama](https://ollama.com) (default model `qwen3.5:0.8b`). No API keys.

> **Status: baseline.** The agent works end to end on real data, but accuracy is low (see below). The project is being built up step by step: a vector database with RAG next, then a multi-agent setup. The model layer and tools already use LangChain, the control flow uses LangGraph. The earlier synthetic-data version (payslips, generated invoices) is in `legacy/` and is no longer used.

## How it works

```
receipt.txt -> LangGraph loop (LangChain model + tools) -> read_document -> validate_receipt -> save_result
                                                              |  problems? fix and retry (max 3)
                                                              +-> escalate to a human
```

The model answers only by calling tools:

| Tool | What it does |
|---|---|
| `read_document` | returns the OCR text |
| `validate_receipt` | checks the fields against a Pydantic schema and against the text itself |
| `save_result` | writes the JSON. Only works after validation returned OK |
| `escalate` | sends the document to a human with a reason |

**Guardrails are in code, not in the prompt** ([receipt_agent.py](receipt_agent.py), [common.py](common.py)): every value the model returns must be printed in the document (amounts in any separator convention, dates, invoice numbers, supplier with an OCR-tolerant match), the model can only touch the document it was given, and after 3 failed validations the code escalates on its own.

Only `total` is required. Supplier, date, invoice number, subtotal and tax are optional, because many real receipts do not print them. If the model gives a value, it must be grounded in the text.

The loop is a LangGraph graph ([graph_agent.py](graph_agent.py)); `python graph_agent.py --graph` prints it as Mermaid. The model is `ChatOllama` and the tools are LangChain `StructuredTool`s. Tool arguments are validated leniently (a small model sends `''`, `null` or `"59,500"` for a number), so a sloppy value reaches `validate_receipt`, which counts the failure and answers with feedback.

Tool calls run one at a time, in order, in our own `run_tools` node. LangGraph's prebuilt `ToolNode` runs the calls of one message in parallel, and `save_result` must not run before `validate_receipt` has finished.

## Data

Real data, not generated. Two public sets, converted to one text file per document plus `truth.json`:

| Source | Docs | Labelled fields |
|---|---|---|
| [SROIE 2019](https://huggingface.co/datasets/ryanznie/SROIE_2019_with_labels) (Malaysian receipts) | 973 | supplier_name, invoice_date, total, invoice_number |
| [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (Indonesian receipts, validation split) | 100 | total, subtotal, tax, item_count |

A field with no label is `null` in `truth.json` and is skipped by the evaluation. Licences: CORD is CC-BY-4.0; the SROIE copy's card says `unknown`, so treat it as research/learning use.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install ollama pydantic langgraph langchain-ollama langchain-chroma chromadb huggingface_hub pyarrow

ollama pull qwen3.5:0.8b     # must match MODEL in common.py
ollama pull nomic-embed-text # embeddings for the supplier index
ollama serve                 # if it is not already running

python download_data.py      # into real_raw/ (git-ignored)
python prepare_sroie.py      # writes data_real/
python prepare_cord.py
python prepare_sroie.py --layout rows --out data_rows   # same receipts, text rebuilt as printed rows
python prepare_cord.py --out data_rows
python supplier_index.py build   # embed the known suppliers into supplier_db/ (git-ignored)
python tax_rules.py build        # embed the tax rules into rules_db/ (git-ignored)
```

## Usage

```bash
python graph_agent.py sroie_000 -v          # one receipt, prints every tool call
python graph_agent.py cord_007 --out output # also save the extracted JSON

python multi_agent.py sroie_121 -v --data data_rows   # the team: supervisor, extractor, reviewer

python evaluate_real.py --n 20 --show-errors   # baseline: 20 SROIE + 20 CORD, evenly spaced
python evaluate_real.py --rescore real_results.json   # score a saved run again, no model calls
```

A run returns `status` of `saved`, `escalated` or `failed`, plus the steps and tools used.

## Baseline (qwen3.5:0.8b, 20 SROIE + 20 CORD, LangChain version)

| | Field | Correct |
|---|---|---|
| SROIE | supplier_name | 45% |
| | invoice_date | 90% |
| | invoice_number | 67% |
| | total | 80% |
| CORD | total | 55% |
| | subtotal | 50% |
| | tax | 17% |

**Silent errors: 23 of 34 saved documents have a wrong labelled field.** That is the number to bring down: a document that is saved but wrong is worse than one that is escalated. Details per document are in [real_results.json](real_results.json).

Read the percentages as rough. The sample is 40 documents, and a tiny model changes its answers when small things change: the same documents scored 20 of 30 silent errors before the move to LangChain (supplier 45%, date 75%, total 70%). The field accuracies moved, the silent-error rate (about two thirds) did not. Compare future changes on this rate, and on a larger sample.

Known failure modes the grounding check cannot catch, because the wrong value *is* printed in the text:
- tax given as the percentage (`10`) instead of the amount
- supplier picked from the wrong line (cashier name, branch, promo line)
- supplier filled in on CORD receipts that print none
- `Rp 59.400` read as `59.4` (the checker accepts both readings of `59.400`)
- invoice number copied together with its label (`INVOICE :581355`)

## Supplier index (RAG, standalone)

The idea: the model's worst field is `supplier_name` (45%), and it often picks a printed but wrong line. A *supplier master* of suppliers we have already seen can say which company a receipt probably comes from. The master is the 227 distinct company labels of the SROIE **train** split, embedded with `nomic-embed-text` into Chroma. It is evaluated on the **test** split only, so it never contains the answer by construction. Each receipt is searched with its top 8 lines.

| 347 test receipts | Vector (Chroma) | Plain string similarity |
|---|---|---|
| supplier already in the index (258 = 74%): found at top-1 | 96% | **99%** |
| top-3 | 97% | 100% |
| new supplier vs known, told apart by a distance cut-off | 98% | 97% |

Two things to take from this:
- The ceiling is 74%: a quarter of the test receipts come from suppliers not seen in train, and an index of known names cannot find those. It can only say "no known supplier is close" (distance above about 0.22).
- **On this data plain string similarity is as good as the vector search, slightly better.** Receipt names differ from the master only by OCR noise, which character matching handles well. A vector database earns its place when the match is about meaning rather than spelling, or when the master is too big to scan. That is the case for business rules (roadmap step 4), not for supplier names.

### Using it in the agent (`--rag`)

With `--rag` the graph gets a `retrieve` node first. It looks the receipt up in the supplier index; if the best hit is within `CUTOFF` (0.22) the receipt counts as from that known supplier, and `validate_receipt` rejects any other `supplier_name` with the right name in its feedback. For a new supplier (no close hit) nothing changes. The model is **not** told the supplier in the prompt: a first version did that, and the model then answered with a minimal call that left out date and invoice number (date 87% to 62%).

```bash
python evaluate_real.py --sources sroie --split test --n 60            # RAG off
python evaluate_real.py --sources sroie --split test --n 60 --rag      # RAG on, same 60 receipts
```
`--rag` is refused on the train split: the index holds every train supplier, so those receipts would be answered by lookup.

Result on 60 SROIE test receipts, same receipts in both runs ([results/](results/)):

| | RAG off | RAG on |
|---|---|---|
| supplier_name correct | 45% | **72%** |
| supplier fixed / broken, among documents saved in both runs | | 17 fixed, 1 broken |
| date, invoice number, total in documents saved in both runs | | no change at all |
| saved / escalated | 56 / 4 | 53 / 7 |
| **silent errors** | 40 of 56 (71%) | **32 of 53 (60%)** |

Read with care:
- The cut-off 0.22 was chosen on the same 347 test receipts (`evaluate_supplier_rag.py`) that the 60 are drawn from, so the RAG numbers are somewhat optimistic. A clean check needs the cut-off set on other data.
- Three more receipts were escalated. That is the safe direction, but it is a cost: when the index is confidently wrong, the model can only fail the rule three times and the receipt goes to a human. (Example: `sroie_040` is from POPULAR BOOK CO. and the index says COSWAY at distance 0.205.)
- Single run on 60 receipts. The supplier gain (17 vs 1) is large enough to trust; the silent-error change is not.
- 60% silent errors is still high. The rest are invoice numbers, totals and dates the supplier index cannot touch.

## Tax rules (RAG with metadata filters, standalone)

[rules/tax_rules.json](rules/tax_rules.json) holds six short rule chunks written for this project from public pages: the Malaysian GST 6% period, the 0% period from 1 June 2018, SST (sales tax 5%/10%, service tax 6%) from 1 September 2018, and the Indonesian restaurant tax (cap 10%, tax = total x rate). Every chunk lists its source URLs and a `note` saying what the sources do and do not support. The text that gets retrieved only contains what the sources say, apart from one sentence per period about what a receipt "normally shows", which we checked against the data: of the SROIE receipts that print a rate, 480 of 487 print 6% up to May 2018 and 44 of 47 print 0% from June to August 2018.

Known weak points of the corpus: the SST pages were written *before* SST started ("planned", "expected"); the end of the 0% period (31 August 2018) is inferred from SST starting on 1 September, and one source says 6% GST was deactivated from 1 August; the exact day GST started is not in the sources. Rates have changed since, so this covers receipts of 2016 to early 2019 only.

A vector search cannot tell that 2018-07-15 lies inside "1 June to 31 August 2018", so each chunk also carries its validity period and country as metadata. Chroma filters on those first, then the embedding search ranks what is left. Evaluated on the 945 dated SROIE receipts (does the search return the rule that applied on the receipt's date?):

| Period | Receipts | Country filter only: top-1 | Country + date filter: top-1 |
|---|---|---|---|
| GST 6% (to May 2018) | 836 | 42% | 100% |
| GST 0% (Jun-Aug 2018) | 99 | **0%** | 100% |
| SST (from Sep 2018) | 10 | 50% | 100% |

The 100% on the right is by construction: after the date filter only the valid rule is left, so it proves the filter works, not that the search is clever. The left column is the informative one: embeddings alone get the date wrong, and for the 0% period they never find the right rule. Dates in the SST period are only 10 receipts, so read nothing into that row.

Not measured: whether using these rules lowers the agent's silent errors. SROIE has no tax amount labels, and CORD (the receipts with tax labels) has no dates and one fixed Indonesian rate. So this part is a retrieval demo, plus a way to flag odd receipts, not a proven accuracy gain.

## Row layout

The SROIE OCR boxes come as separate pieces: a label and its amount are different boxes, and often all the labels are listed first and the amounts after (`NET TOTAL / CASH / CHANGE`, then `39.80 / 40.00 / 0.20`). `prepare_sroie.py --layout rows` uses the box coordinates to join boxes that sit on one printed row (`NET TOTAL 39.80`, `CASH 40.00`). The text goes to `data_rows/`; `data_real/` keeps the original one-box-per-line text.

On the 60 test receipts (set A, same receipts in all four runs): total 68% to 80% (RAG off), silent errors 71% to 64% (off) and 60% to 53% (RAG on). But date 87% to 78% (off), and more receipts were escalated (4 to 7 off, 3 to 9 on). Net: better totals, mixed elsewhere. The two earlier runs ([results/sroie_test60_rag_off.json](results/sroie_test60_rag_off.json), `..._on.json`) are the one-box-per-line text.

## Multi-agent team

`multi_agent.py` is a LangGraph graph with a supervisor, an extractor and a reviewer.

```
START -> supervisor -> extractor -> supervisor -> reviewer -> supervisor -> finalize -> END
               ^                                                  |
               +------------ revise, with the objections ---------+   (at most 2 revisions)
```

- **extractor**: the LLM agent from `graph_agent.py`. On a revision it also receives the reviewer's objections and its own previous answer.
- **reviewer** ([reviewer.py](reviewer.py)): checks that an answer plays the right role, not just that it is printed. Total must equal cash minus change when the receipt prints both (a small model often copies the cash line); an invoice number must look like a short code, not an address, a label or a sentence; a date must be a date. **Plain code, not a model**: the checks are exact and give the same verdict every run.
- **supervisor**: plain code that routes and caps the revisions. The workers never call each other; they read and write the shared state and return to the supervisor.
- If the reviewer still objects after the revisions, the receipt is **escalated** to a human.

Offline, on the answers the single agent had saved for set A: the reviewer flagged 5 of 6 wrong totals and 10 of 12 wrong invoice numbers, and 0 of 77 correct ones. Those rules were written after looking at set A's errors, so that number is optimistic. The real test is a second set of 60 test receipts that does not overlap (set B, `--offset 3`), with the same settings (row layout, RAG on):

| Set B, 60 receipts | Single agent | Team |
|---|---|---|
| saved / escalated | 39 / 21 | 27 / 33 |
| saved with a wrong labelled field ("silent errors") | 24 of 39 (62%) | 12 of 27 (44%) |
| **receipts with every labelled field right** | **15** | **15** |
| total correct | 48% | 40% |

How to read it:
- **The team does not produce more correct receipts (15 and 15). It stops wrong ones from being saved:** of the 24 receipts the single agent saved wrongly, 11 were escalated and 1 became right. One receipt the single agent had right was escalated too.
- **Revising rarely works with this model.** 17 receipts went back to the extractor; 5 were accepted afterwards and 1 of those was fully right. The useful part of the reviewer is the gate, not the repair. 9 of its 11 escalations are invoice numbers.
- **Set B is harder than set A** for the same single-agent settings (21 escalations and 62% silent errors, against 9 and 53% on set A). So differences of 10 points between runs are normal here; 60 receipts is a small sample. Only the paired comparisons (same receipts in both runs) carry weight.
- Per-field accuracy of the team looks lower because an escalated receipt scores as a miss on every field.
- The extractor is the bottleneck: only 15 of 60 receipts come out fully right, with either setup. A 0.8B model is small for this task.

Results: [results/](results/).

## Files

| File | Purpose |
|---|---|
| `common.py` | amount/date parsing, text helpers, per-run state, the shared tools |
| `receipt_agent.py` | `Receipt` schema, grounding checks, prompt, `validate_receipt` |
| `graph_agent.py` | LangGraph agent loop and CLI |
| `evaluate_real.py` | evaluation, including silent errors |
| `supplier_index.py`, `evaluate_supplier_rag.py` | supplier master as a Chroma vector index, and its evaluation |
| `multi_agent.py`, `reviewer.py` | the supervisor / extractor / reviewer team, and the reviewer's checks |
| `tax_rules.py`, `evaluate_tax_rag.py`, `rules/` | tax rules as a Chroma index with date/country filters, and its evaluation |
| `download_data.py`, `prepare_sroie.py`, `prepare_cord.py` | get the data and convert it |
| `results/` | saved evaluation runs (RAG off / on) |
| `data_rows/` | the same receipts with the text rebuilt as printed rows |
| `data_real/` | converted documents and `truth.json` |
| `legacy/` | the old synthetic-data version, kept for reference. Safe to delete |

## Roadmap

1. ~~Real data and a baseline~~ (this)
2. ~~Move the model layer to LangChain (`ChatOllama`, `StructuredTool`) inside the LangGraph graph~~
3. ~~RAG: embeddings + vector database (Chroma), standalone first~~ (supplier index, see above)
4. ~~Use RAG in the graph~~: supplier master done and in the graph. Tax rules: indexed and evaluated, not yet used by the agent
5. ~~Multi-agent: supervisor, extractor, reviewer~~ (above): fewer wrong saves, not more right ones
6. Re-measure against this baseline
