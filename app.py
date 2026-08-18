"""
app.py
Athena — Research Analyst
Streamlit frontend: topic in, live pipeline progress, report draft held for
human review before it is finalised.

Run:
    streamlit run app.py

MCP mode (requires the tool server, see mcp_servers/web_server.py):
    MCP_MODE=true streamlit run app.py
"""

import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Athena — Research Analyst",
    page_icon=":material/local_library:",
    layout="centered",
)

st.markdown(
    """
    <style>
      /* Product chrome: hide Streamlit's own header/deploy UI */
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stAppDeployButton"], #MainMenu, [data-testid="stDecoration"] { display: none; }

      .block-container { max-width: 46rem; padding-top: 2.5rem; }

      /* Editorial serif for the wordmark */
      .athena-wordmark {
        font-family: Georgia, 'Iowan Old Style', 'Times New Roman', serif;
        font-size: 3.2rem; font-weight: 700; letter-spacing: -0.02em;
        color: #E7E9EC; line-height: 1.1; margin: 0;
      }
      .athena-wordmark .dot { color: #E3A94F; }
      .athena-tagline { color: #9AA3AF; font-size: 1.02rem; margin: 0.5rem 0 1.4rem 0; }

      .athena-chips { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 2.2rem; }
      .athena-chip {
        border: 1px solid #262C36; border-radius: 999px;
        padding: 0.28rem 0.75rem; font-size: 0.78rem; color: #B9C0C9;
        background: #12161C;
      }
      .athena-chip::before { content: "•"; color: #E3A94F; margin-right: 0.45rem; }

      .athena-try { color: #6E7681; font-size: 0.8rem; text-transform: uppercase;
                    letter-spacing: 0.08em; margin-bottom: 0.6rem; }

      /* Sidebar */
      [data-testid="stSidebar"] { border-right: 1px solid #1C222B; }
      .sb-brand { font-family: Georgia, serif; font-size: 1.35rem; font-weight: 700;
                  color: #E7E9EC; margin-bottom: 0.15rem; }
      .sb-label { color: #6E7681; font-size: 0.72rem; text-transform: uppercase;
                  letter-spacing: 0.1em; margin: 1.1rem 0 0.45rem 0; }
      .sb-row { font-size: 0.85rem; color: #B9C0C9; line-height: 1.9; }
      .sb-row b { color: #E7E9EC; font-weight: 600; }

      /* Chat + review */
      div[data-testid="stChatMessage"] {
        background: #12161C; border: 1px solid #1C222B;
        border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: 0.4rem;
      }
      .review-flag { display: inline-block; background: rgba(227, 169, 79, 0.12);
                     color: #E3A94F; border: 1px solid rgba(227, 169, 79, 0.35);
                     border-radius: 5px; padding: 0.15rem 0.6rem;
                     font-size: 0.72rem; font-weight: 600; letter-spacing: 0.07em;
                     text-transform: uppercase; margin-bottom: 0.7rem; }
      .stage-line { color: #B9C0C9; font-size: 0.9rem; }
      .stage-line::before { content: "→"; color: #E3A94F; margin-right: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Graph initialisation ──────────────────────────────────────────────────────

@st.cache_resource
def get_graph():
    """
    Build the LangGraph graph once and cache it for the whole process.

    Without cache_resource every Streamlit rerun (each button click) would
    rebuild the graph and discard the checkpointer — killing the paused
    review state that human-in-the-loop depends on.
    """
    from core.graph import build_graph
    return build_graph()


graph = get_graph()

# ── Model selection (sidebar) ─────────────────────────────────────────────────
# Process-wide operator control: choices are written to env, and get_llm()
# re-reads env on every node execution, so a switch applies from the next run.

GROQ_MODELS = [
    "openai/gpt-oss-120b",   # production, tool-capable — default
    "openai/gpt-oss-20b",    # faster, lighter
    "qwen/qwen3.6-27b",      # Groq's other recommended replacement
]


@st.cache_data(ttl=30)
def list_ollama_models() -> list[str] | None:
    """Chat-capable models on the local Ollama server, or None if unreachable.

    None and [] are different answers: unreachable means the whole local
    backend must not be offered — silently substituting a default model name
    here is what once let a cloud deployment show "Ollama (local)" in the
    picker and then greet the user with a raw httpx traceback.
    """
    import httpx

    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        tags = httpx.get(f"{base}/api/tags", timeout=3).json()
        names = [m["name"] for m in tags.get("models", [])]
        return [n for n in names if "embed" not in n] or [os.getenv("OLLAMA_MODEL", "qwen3.5:9b")]
    except Exception:
        return None


def model_picker():
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    ollama_models = list_ollama_models()

    backends = []
    if has_groq:
        backends.append("Groq (cloud)")
    if ollama_models:
        backends.append("Ollama (local)")
    if not backends:
        st.error(
            "No model backend is reachable. Set GROQ_API_KEY for the cloud "
            "backend, or start Ollama for the local one."
        )
        return
    if not ollama_models and has_groq:
        st.caption("Local backend (Ollama) not reachable here — cloud only.")
    from core.llm import resolve_backend
    active_backend, _ = resolve_backend()
    default_ix = 0 if (active_backend == "groq" and has_groq) else len(backends) - 1

    backend = st.selectbox("Backend", backends, index=default_ix)

    if backend.startswith("Groq"):
        current = os.getenv("GROQ_MODEL", GROQ_MODELS[0])
        options = [current] + [m for m in GROQ_MODELS if m != current]
        model = st.selectbox("Model", options)
        os.environ["ATHENA_LLM_BACKEND"] = "groq"
        os.environ["GROQ_MODEL"] = model
    else:
        options = ollama_models
        current = os.getenv("OLLAMA_MODEL", options[0])
        ix = options.index(current) if current in options else 0
        model = st.selectbox("Model", options, index=ix)
        os.environ["ATHENA_LLM_BACKEND"] = "ollama"
        os.environ["OLLAMA_MODEL"] = model
        if not has_groq:
            st.caption("Add GROQ_API_KEY in .env to enable the cloud backend.")


def runtime_info() -> dict:
    # Document mode swaps the tool set entirely and never touches the web, so
    # reporting "web search" there is not a cosmetic slip — the runtime panel is
    # where a reader checks the claim that nothing left the machine.
    if os.getenv("ATHENA_MODE", "web").lower() == "documents":
        tools = "Local archive: search + inventory + exact aggregation"
    elif os.getenv("MCP_MODE", "false").lower() == "true":
        tools = "MCP server"
    else:
        tools = "In-process web search"
    return {
        "Tools": tools,
        "State": os.getenv("ATHENA_CHECKPOINTER", "memory"),
    }


NODE_LABELS = {
    "supervisor": "Planning next step",
    "researcher": "Researching sources",
    "writer": "Drafting report",
    "review": "Preparing review",
}

EXAMPLE_TOPICS = [
    "EU AI Act — what it means for medical device manufacturers",
    "Germany's hydrogen strategy: state of play in 2026",
    "Post-quantum cryptography: how banks are preparing",
]

# ── Session state ─────────────────────────────────────────────────────────────

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_review" not in st.session_state:
    st.session_state.awaiting_review = False


def get_config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def reset_session():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.awaiting_review = False


def finalise(report: str):
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"**Final report**\n\n{report}",
    })
    st.session_state.awaiting_review = False
    st.session_state.thread_id = str(uuid.uuid4())


def _model_backend_error(e: Exception) -> None:
    """A human-readable failure instead of a traceback in the chat pane."""
    name = type(e).__name__
    if "Connect" in name or "Connection" in name:
        st.error(
            "The selected model backend is not reachable from this machine. "
            "If you chose **Ollama (local)**, it is not running here — switch "
            "the backend in the sidebar and try again."
        )
    else:
        st.error(
            f"The model backend failed mid-run ({name}). "
            "Try again, or switch the backend in the sidebar."
        )


def run_pipeline(topic: str):
    """Run the research graph for a new topic and stream progress."""
    st.session_state.thread_id = str(uuid.uuid4())
    config = get_config()

    st.session_state.messages.append({"role": "user", "content": topic})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(topic)

    from core.graph import make_initial_state
    from core.ui_documents import bind_current_thread, ensure_session

    # Attached documents belong to the BROWSER SESSION, not to this thread —
    # a new thread_id is minted for every question, so keying on it would drop
    # the user's uploads after their first message. The thread is recorded so
    # teardown can scrub its checkpointed passages too.
    session_id = ensure_session(st)
    bind_current_thread(st)

    with st.chat_message("assistant", avatar=":material/local_library:"):
        with st.status("Working on it…", expanded=True) as status:
            try:
                for chunk in graph.stream(
                    make_initial_state(topic, session_id=session_id),
                    config=config,
                    stream_mode="updates",
                ):
                    for node_name in chunk:
                        label = NODE_LABELS.get(node_name)
                        if label:
                            st.markdown(
                                f'<div class="stage-line">{label}</div>',
                                unsafe_allow_html=True,
                            )
            except Exception as e:  # noqa: BLE001 - a backend outage must not print a traceback at the user
                status.update(label="Run failed", state="error")
                _model_backend_error(e)
                return

            status.update(label="Draft ready for your review", state="complete")

        current = graph.get_state(config)

        if current.next:  # paused at the review gate
            st.session_state.awaiting_review = True
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Draft ready — review it below.",
            })
            st.rerun()
        else:
            final = current.values.get("draft_report", "Research complete.")
            st.markdown(final)
            st.session_state.messages.append({"role": "assistant", "content": final})


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sb-brand">Athena<span style="color:#E3A94F">.</span></div>',
                unsafe_allow_html=True)
    st.caption("Research analyst with a human approval gate.")

    st.markdown('<div class="sb-label">Model</div>', unsafe_allow_html=True)
    model_picker()

    st.markdown('<div class="sb-label">Runtime</div>', unsafe_allow_html=True)
    rows = "".join(
        f'<div class="sb-row">{k} · <b>{v}</b></div>' for k, v in runtime_info().items()
    )
    st.markdown(rows, unsafe_allow_html=True)

    from core.ui_documents import render_sidebar

    render_sidebar(st)

    st.markdown('<div class="sb-label">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sb-row">Research → Draft → <b>Your review</b> → Final</div>'
        '<div class="sb-row" style="color:#6E7681; font-size:0.78rem; line-height:1.5; margin-top:0.3rem;">'
        'Nothing is published without explicit approval. Request changes and only '
        'the sections you name are rewritten.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-label"></div>', unsafe_allow_html=True)
    if st.button("New session", use_container_width=True):
        reset_session()
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────

topic_input = st.chat_input("Research topic or question")
pending = st.session_state.pop("pending_topic", None)
topic = topic_input or pending

if st.session_state.awaiting_review:
    # ── Review gate ───────────────────────────────────────────────────────────
    config = get_config()
    current = graph.get_state(config)
    draft = current.values.get("draft_report", "*No draft found.*")

    for msg in st.session_state.messages:
        avatar = ":material/person:" if msg["role"] == "user" else ":material/local_library:"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    with st.container(border=True):
        st.markdown('<span class="review-flag">Awaiting your approval</span>',
                    unsafe_allow_html=True)
        st.markdown(draft)
        st.divider()

        feedback = st.text_area(
            "Revision notes",
            placeholder="Example: expand the Analysis section with figures from the sources.",
            key="hitl_feedback",
        )

        approve_col, revise_col = st.columns(2)

        if approve_col.button("Approve report", type="primary", use_container_width=True):
            with st.spinner("Finalising…"):
                try:
                    graph.invoke(Command(resume="approve"), config=config)
                except Exception as e:  # noqa: BLE001
                    _model_backend_error(e)
                    st.stop()
            finalise(draft)
            st.rerun()

        if revise_col.button(
            "Request revision",
            use_container_width=True,
            disabled=not feedback.strip(),
            help=None if feedback.strip() else "Enter revision notes first.",
        ):
            with st.spinner("Revising draft…"):
                try:
                    graph.invoke(Command(resume=feedback.strip()), config=config)
                except Exception as e:  # noqa: BLE001
                    _model_backend_error(e)
                    st.stop()

            new_state = graph.get_state(config)
            if new_state.next:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Revised draft ready — feedback applied: *{feedback.strip()}*",
                })
                st.rerun()
            else:
                finalise(new_state.values.get("draft_report", draft))
                st.rerun()

elif topic:
    # ── Active run ────────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        avatar = ":material/person:" if msg["role"] == "user" else ":material/local_library:"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
    # On a public URL every visitor shares one free-tier API budget; 0 = off.
    max_runs = int(os.getenv("ATHENA_MAX_RUNS_PER_SESSION", "0"))
    if max_runs and st.session_state.get("run_count", 0) >= max_runs:
        st.warning(
            f"Demo limit reached ({max_runs} research runs per session). "
            "Run Athena locally for unlimited use — see the README."
        )
    else:
        st.session_state.run_count = st.session_state.get("run_count", 0) + 1
        run_pipeline(topic)

elif st.session_state.messages:
    # ── History ───────────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        avatar = ":material/person:" if msg["role"] == "user" else ":material/local_library:"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

else:
    # ── Hero / empty state ────────────────────────────────────────────────────
    st.markdown(
        '<h1 class="athena-wordmark">Athena<span class="dot">.</span></h1>'
        '<p class="athena-tagline">Multi-agent research, drafted for you — '
        'final only when <em>you</em> approve it.</p>'
        '<div class="athena-chips">'
        '<span class="athena-chip">Runs on your machine</span>'
        '<span class="athena-chip">Human approval gate</span>'
        '<span class="athena-chip">Live web research</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="athena-try">Try a topic</div>', unsafe_allow_html=True)
    for example in EXAMPLE_TOPICS:
        if st.button(example, use_container_width=True):
            st.session_state.pending_topic = example
            st.rerun()
