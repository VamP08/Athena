# Athena — AI Research Analyst

[![Tests](https://github.com/VamP08/Athena/actions/workflows/test.yml/badge.svg)](https://github.com/VamP08/Athena/actions)

> **A privacy-first, local-first AI research analyst.** Enter a topic. Three AI
> agents research the web, draft a structured report, and **pause for your
> approval** before finalising. Provide feedback to trigger a targeted revision.
> All model inference runs on-premise with Ollama — or on free cloud tiers.

**Why local-first?** 54% of German companies say data protection hinders their AI
adoption (Bitkom, 2026). German data protection authorities explicitly prefer
self-hosted AI systems. Athena is built for exactly that constraint: every model
call — the agents, drafting, even the evaluation judge and embeddings — can run
locally, so prompts, drafts, and internal context never leave the machine. In
web-research mode, only the search queries themselves reach external engines;
the planned document mode (reports over your own PDFs) removes even that,
running fully air-gapped.

---

## What This Demonstrates

| Skill | Implementation |
|---|---|
| **Retrieval (RAG) over a real archive** | Mixed-format ingestion (PDF · XLSX · DOCX · CSV · MD) into **one SQLite file** — `sqlite-vec` vectors + FTS5 lexical, fused with Reciprocal Rank Fusion, filtered by document type and year |
| **Multi-agent orchestration** | LangGraph 1.2 Supervisor pattern — deterministic routing rules first, Pydantic structured-output LLM routing for ambiguous cases |
| **Human-in-the-Loop (HITL)** | `interrupt()` pauses the graph mid-execution; `Command(resume=)` restarts it with approval or feedback; feedback loops back through the writer |
| **Custom MCP server** | FastMCP 2.x server exposing `search_web` + `fetch_page` over streamable HTTP — any MCP client (Claude Desktop, Cursor) connects without code changes |
| **Local LLM** | Ollama + Qwen3.5:9b — fits consumer hardware, fully offline |
| **Cloud LLM** | Groq API (`openai/gpt-oss-120b`, free tier) — swap via one env var, zero code changes |
| **Production service layer** | FastAPI REST API with bearer-token auth, durable SQLite checkpointing (a paused review survives a server restart — verified), an approval **audit trail**, and worker-pool job execution |
| **Containerized** | One `docker compose up` brings up API + UI, with an optional local-LLM (Ollama) profile |
| **Observability** | LangSmith auto-tracing — every tool call, token count, and routing decision recorded |
| **Evaluation** | Ragas LLM-as-judge: answer_relevancy · faithfulness · context_precision — with a fully local judge+embeddings option |
| **Testing** | pytest with mocked LLM — deterministic, no network, runs in CI on every push |

---

## Architecture

```
User → Streamlit UI → LangGraph StateGraph
                            │
                       Supervisor (deterministic rules + Pydantic structured output)
                      /         \
               Researcher      Writer ──► Human Review (interrupt())
              (tool loop)    (report)          │
                   │                    approve / revise
          web_search (ddgs)                    │
           or MCP server               InMemorySaver (persists state
        (FastMCP, streamable HTTP)      across the HITL pause)
```

The researcher is tool-agnostic: in direct mode it uses an in-process `ddgs`
metasearch tool; with `MCP_MODE=true` it discovers the same capabilities from a
standalone MCP server. The graph doesn't change — that's the point of MCP.

---

## Quick Start

```bash
git clone https://github.com/VamP08/Athena
cd Athena

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # defaults run fully local — no keys needed

# Local LLM setup (skip if using Groq — set GROQ_API_KEY in .env instead)
ollama pull qwen3.5:9b
ollama pull nomic-embed-text     # used by the evaluation suite

streamlit run app.py
```

Verify the engine without the UI:

```bash
pytest tests/ -v                 # mocked — fast, no network
python test_graph_manual.py      # real end-to-end run (LLM + live search)
```

### Run as a service (FastAPI)

```bash
uvicorn api.main:app --port 8000
```

```
POST /research                      {"topic": "..."}            → 202 {thread_id}
GET  /research/{thread_id}          status + draft when ready
POST /research/{thread_id}/resume   {"action": "approve"} or
                                    {"action": "revise", "feedback": "..."}
GET  /research/{thread_id}/audit    who approved/revised what, when
GET  /health
```

Set `ATHENA_API_TOKEN` to require bearer-token auth. With
`ATHENA_CHECKPOINTER=sqlite` (default in `.env.example`), a review that is
awaiting approval **survives an API restart** — kill the server mid-review,
start it again, and approve the same thread.

### Run with Docker

```bash
docker compose up --build           # API on :8000, UI on :8501
docker compose --profile local-llm up   # + Ollama as the local model backend
```

---

## Document Mode — reports over your own financial archive

Point Athena at a folder of the organisation's own documents and it stops using the
web entirely. The same three agents research, draft and pause for approval — but the
evidence now comes from your files, and **no web tool is even loaded**, so
"nothing leaves this machine" is literal rather than aspirational.

```bash
ollama pull bge-m3                # multilingual embeddings (German + English)
python corpus_tools/generate_corpus.py   # optional: a synthetic demo archive
python ingest.py                  # parse + embed ./corpus -> athena_index.db

ATHENA_MODE=documents streamlit run app.py
```

Ask it *"Wie hoch war der Umsatz in Q3 2025 und wer hat den Jahresabschluss geprüft?"*
and it retrieves the exact spreadsheet row and the auditor's name from two different
documents in two different formats, then writes a cited report.

| Handles | How |
|---|---|
| **Tables, not just prose** | Every row is indexed with its column headers attached, so `Umsatz (EUR): 4821000` stays interpretable instead of becoming a naked number |
| **Exact figures and IDs** | Lexical BM25 runs alongside embeddings — invoice numbers, account codes and `Q3 2025` are precisely what dense vectors retrieve badly |
| **German + English** | A question in one language finds documents in the other |
| **Legacy exports** | cp1252 semicolon-delimited CSVs (what German accounting software emits) decode correctly |
| **What it *cannot* read** | A scanned PDF or a formula-only spreadsheet total is reported as a **visible limitation**, never silently treated as absent — in a financial archive, "unreadable" and "not there" are different answers |
| **Counting questions** | A `list_documents` tool exposes the corpus inventory, because semantic search structurally cannot answer "how many invoices are there" |

The demo corpus is **synthetic by design** — the archive of a fictional logistics
company, generated from a fixed seed. Committing real financial records to a public
repo would contradict the premise this project argues for, and a generated corpus
gives the evaluation exact ground truth.

### Grounding evaluation (deterministic, no LLM judge)

Because the demo corpus is generated, every answer has one provably correct value —
so accuracy is scored exactly, offline, with no judge and no tokens
(`python eval/run_grounding_eval.py`).

| Metric | Local qwen3.5:9b | |
|---|---|---|
| answer accuracy | 6/6 | ✅ |
| **fabricated figures** | **0** | ✅ |
| misattributed answers | 0/6 | ✅ |
| retrieval hit rate | 6/6 | ✅ |
| correct refusal on an unanswerable question | 1/1 | ✅ |

<sup>7-case golden set · fully offline · qwen3.5:9b, reasoning on, num_ctx 16384 ·
2026-08-03. "Fabricated" = a figure in the report appearing in none of the retrieved
passages — the failure that matters most in a financial archive, since it looks
authoritative and is silently wrong. Figures the report *derives* from grounded ones
(a year-on-year difference, say) are classified separately: computing is analysis,
not invention.</sup>

**What the context window was hiding.** At `num_ctx=8192` this suite scored 5/6, and
the missing case returned a *completely empty report* despite retrieving the correct
figure at rank 1 — the generation failed silently while retrieval was perfect. The
cause is that Ollama truncates an over-long prompt from the **front**, discarding the
system prompt first: the model loses the instruction *"do not include any claim not
supported by the research"* while keeping partial evidence. Raising the window to
16384 is therefore an accuracy fix, not a performance tweak, and it costs ~1.6x
latency. Measured, with the model's real ceiling being 262144:

| num_ctx | 12-passage prompt | VRAM | on GPU |
|---|---|---|---|
| 8192 | **empty report** | 7230/8188 | 100% |
| 16384 | 553 words, correct | 6375/8188 | 84% |
| 32768 | 439 words, correct | 6393/8188 | 76% |

This harness also settled a design question against intuition. Disabling the local
model's reasoning trace is **~4x faster** (48 s vs 208 s per case) and was briefly
adopted for that reason — until measurement showed it produced **5 fabricated figures
and 2 misattributed answers** where the slower setting produced none. Retrieval was
identical (6/6) in both, so the difference was purely in how the model used evidence
it already had. The speed was reverted; accuracy is not a thing this project trades.

Measured on a consumer laptop, fully offline: 13 documents → 447 indexed passages in
**18.5 s**; a complete research → draft → review cycle in **~3.5 min** on a local 9B
model with reasoning enabled.

## Two-Environment Strategy

| Environment | Model | Purpose |
|---|---|---|
| **Local (Ollama)** | `qwen3.5:9b` — free, offline, private | Development, plumbing, fully on-premise demo |
| **Cloud (Groq)** | `openai/gpt-oss-120b` — free tier | Prompt tuning, deployment, stronger eval judge |

Switch by setting `GROQ_API_KEY` in `.env`. No code changes. This mirrors the
hybrid architecture German enterprises actually deploy: sensitive workloads
local, scale-out in the cloud.

---

## MCP Server

The researcher can consume tools from a standalone Model Context Protocol server
instead of in-process functions:

```bash
# Terminal 1 — start the MCP server (streamable HTTP on :8001/mcp)
python -m mcp_servers.web_server

# Terminal 2 — run the app in MCP mode
MCP_MODE=true streamlit run app.py
```

Any MCP-compatible client can connect to `http://localhost:8001/mcp` and discover
`search_web` and `fetch_page` automatically.

---

## Evaluation (Ragas)

```bash
python eval/run_eval.py
```

Runs every golden-dataset topic through the full graph (auto-approving the HITL
gate), then scores with LLM-as-judge. Judge and embeddings can run fully locally
(Ollama) — the evaluation itself honours the local-first constraint.

| Metric | Score | Target | |
|---|---|---|---|
| answer_relevancy | **0.87** | > 0.85 | ✅ |
| faithfulness | **0.83** | > 0.80 | ✅ |
| context_precision | **1.00*** | > 0.75 | ✅ |

<sup>5-topic golden dataset · judge: `llama-3.3-70b-versatile` (Groq) · embeddings:
local `nomic-embed-text` · 2026-07-22. Judge migration to `gpt-oss-120b` is
scheduled — its free tier's tighter token budgets need the leaner judging
prompts that the citations feature will bring.
*context_precision currently scores a single synthesized context chunk, so it
reads near-binary — the planned citations feature (raw per-source contexts) will
turn it into a discriminating signal. Weakest cell: faithfulness 0.64 on the
fast-moving topic (SpaceX Starship) — the current tuning target.</sup>

---

## Project Structure

```
athena/
├── app.py                    # Streamlit UI — chat, streaming, HITL review panel
├── api/
│   ├── main.py               # FastAPI service — jobs, review gate, auth, health
│   └── registry.py           # Thread registry + approval audit trail (SQLite)
├── Dockerfile · docker-compose.yml
├── test_graph_manual.py      # Real end-to-end validation script
├── core/
│   ├── state.py              # ResearchState TypedDict (add-reducer lists)
│   ├── llm.py                # get_llm() — Groq → Ollama fallback
│   ├── tools.py              # web_search tool on ddgs (retry/backoff)
│   ├── nodes.py              # supervisor / researcher / writer / review nodes
│   └── graph.py              # StateGraph assembly + InMemorySaver
├── mcp_servers/
│   ├── web_server.py         # FastMCP server (streamable HTTP)
│   └── connect.py            # MCP → LangChain tools client helper
├── tests/                    # mocked unit + integration tests (CI)
├── eval/                     # Ragas golden dataset + evaluation runner
└── docs/                     # PROGRESS.md · ROADMAP.md · DECISIONS.md
```

---

## Engineering Log

Development status, phase gates, and the decision log live in
[docs/PROGRESS.md](docs/PROGRESS.md), [docs/ROADMAP.md](docs/ROADMAP.md), and
[docs/DECISIONS.md](docs/DECISIONS.md) — including why this project dropped
langchain-community (sunset June 2026), migrated MCP off SSE, and moved off
Groq's deprecated llama-3.3-70b before its 2026-08-16 shutdown.
