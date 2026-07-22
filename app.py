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
    page_icon=":material/search:",
    layout="centered",
)

# Light typographic polish; stable selectors only.
st.markdown(
    """
    <style>
      .block-container { max-width: 52rem; }
      [data-testid="stSidebar"] .stCaption { line-height: 1.5; }
      div[data-testid="stChatMessage"] { border-radius: 6px; }
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

# ── Runtime facts (shown in the sidebar — real values, not marketing) ─────────

def runtime_info() -> dict:
    if os.getenv("GROQ_API_KEY"):
        model = f"Groq · {os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')}"
    else:
        model = f"Ollama (local) · {os.getenv('OLLAMA_MODEL', 'qwen3.5:9b')}"
    tools = (
        "MCP server" if os.getenv("MCP_MODE", "false").lower() == "true"
        else "In-process web search"
    )
    checkpointer = os.getenv("ATHENA_CHECKPOINTER", "memory")
    return {"Model": model, "Search tools": tools, "Checkpointer": checkpointer}


# Human-readable labels for pipeline stages streamed from the graph.
NODE_LABELS = {
    "supervisor": "Planning next step",
    "researcher": "Researching sources",
    "writer": "Drafting report",
    "review": "Preparing review",
}

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


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Athena")
    st.caption(
        "Multi-agent research analyst. Drafts are never final without "
        "your explicit approval."
    )
    st.divider()

    st.markdown("**Runtime**")
    for key, value in runtime_info().items():
        st.caption(f"{key}: {value}")

    st.divider()

    st.markdown("**Pipeline**")
    st.caption(
        "Supervisor routes the work — a researcher gathers sources, a writer "
        "drafts the report, and execution pauses at a review gate until you "
        "approve or request changes."
    )

    st.divider()
    if st.button("New session", use_container_width=True):
        reset_session()
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Athena")
st.caption("Enter a topic. Review the draft. Nothing is final until you approve it.")

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Review panel ──────────────────────────────────────────────────────────────

if st.session_state.awaiting_review:
    config = get_config()
    current = graph.get_state(config)
    draft = current.values.get("draft_report", "*No draft found.*")

    with st.container(border=True):
        st.subheader("Draft for review")
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
                graph.invoke(Command(resume="approve"), config=config)
            finalise(draft)
            st.rerun()

        if revise_col.button(
            "Request revision",
            use_container_width=True,
            disabled=not feedback.strip(),
            help="Enter revision notes first." if not feedback.strip() else None,
        ):
            with st.spinner("Revising draft…"):
                graph.invoke(Command(resume=feedback.strip()), config=config)

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

# ── Topic input ───────────────────────────────────────────────────────────────

elif prompt := st.chat_input("Research topic or question"):
    st.session_state.thread_id = str(uuid.uuid4())
    config = get_config()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    from core.graph import make_initial_state

    with st.chat_message("assistant"):
        with st.status("Working…", expanded=True) as status:
            for chunk in graph.stream(
                make_initial_state(prompt),
                config=config,
                stream_mode="updates",
            ):
                for node_name in chunk:
                    label = NODE_LABELS.get(node_name)
                    if label:
                        st.write(label)

            status.update(label="Draft ready for review", state="complete")

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
