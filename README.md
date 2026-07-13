# Athena — AI Research Analyst

[![Tests](https://github.com/VamP08/Athena/actions/workflows/test.yml/badge.svg)](https://github.com/VamP08/Athena/actions)

> **A privacy-first, local-first AI research analyst.** Enter a topic. Three AI
> agents research the web, draft a structured report, and **pause for your
> approval** before finalising. Provide feedback to trigger a targeted revision.
> Runs entirely on-premise with Ollama — no data leaves the machine — or on free
> cloud tiers for deployment.

**Why local-first?** 54% of German companies say data protection hinders their AI
adoption (Bitkom, 2026). German data protection authorities explicitly prefer
self-hosted AI systems. Athena is built for exactly that constraint: the whole
pipeline — agents, search synthesis, drafting, even the evaluation judge and
embeddings — can run with zero external API calls.

---

## What This Demonstrates

| Skill | Implementation |
|---|---|
| **Multi-agent orchestration** | LangGraph 1.2 Supervisor pattern — deterministic routing rules first, Pydantic structured-output LLM routing for ambiguous cases |
| **Human-in-the-Loop (HITL)** | `interrupt()` pauses the graph mid-execution; `Command(resume=)` restarts it with approval or feedback; feedback loops back through the writer |
| **Custom MCP server** | FastMCP 2.x server exposing `search_web` + `fetch_page` over streamable HTTP — any MCP client (Claude Desktop, Cursor) connects without code changes |
| **Local LLM** | Ollama + Qwen3.5:9b — fits consumer hardware, fully offline |
| **Cloud LLM** | Groq API (`openai/gpt-oss-120b`, free tier) — swap via one env var, zero code changes |
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

---

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
gate), then scores with LLM-as-judge. Judge and embeddings run locally by default
(Ollama) — the evaluation itself honours the zero-external-calls constraint.

| Metric | Score | Target |
|---|---|---|
| answer_relevancy | _pending M3_ | > 0.85 |
| faithfulness | _pending M3_ | > 0.80 |
| context_precision | _pending M3_ | > 0.75 |

---

## Project Structure

```
athena/
├── app.py                    # Streamlit UI — chat, streaming, HITL review panel
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
