"""
app.py
Athena — AI Research Analyst
Streamlit frontend with real-time agent streaming and HITL review panel.

Run:
    streamlit run app.py

For MCP mode (requires web_server.py running first):
    MCP_MODE=true streamlit run app.py
"""

import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Athena — AI Research Analyst",
    page_icon="🔬",
    layout="wide",
)

# ── Graph initialisation ──────────────────────────────────────────────────────

@st.cache_resource
def get_graph():
    """
    Build the LangGraph graph once and cache it for the entire Streamlit session.

    CRITICAL: @st.cache_resource is non-optional here.
    Without it, every Streamlit rerun (triggered by every button click)
    creates a new graph instance, destroying the InMemorySaver and losing
    all HITL state — the interrupt, the draft, the thread ID — everything.
    """
    from core.graph import build_graph
    return build_graph()


graph = get_graph()

# ── Session state defaults ────────────────────────────────────────────────────

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


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔬 Athena")
    st.caption("AI Research Analyst")
    st.divider()

    st.markdown("""
**Stack**
- 🔗 LangGraph 1.2 — multi-agent graph
- 🛠️ FastMCP — custom tool server
- 🦙 Ollama qwen3.5:9b — local LLM
- ⚡ Groq — cloud LLM (prod)
- 🔍 ddgs metasearch — free web search
- 👁️ LangSmith — observability
- 📊 Ragas — evaluation

**Agents**
1. **Supervisor** — routes via structured output
2. **Researcher** — tool-calling loop
3. **Writer** — structured report
4. **Review** — HITL interrupt()
""")

    st.divider()
    if st.button("🔄 Reset conversation", use_container_width=True):
        reset_session()
        st.rerun()

    import os
    mcp_status = "✅ MCP mode" if os.getenv("MCP_MODE", "false").lower() == "true" else "📡 Direct tools"
    st.caption(mcp_status)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🔬 Athena — AI Research Analyst")
st.caption("Enter a topic · watch three agents work · review the draft · approve or revise")

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── HITL review panel ─────────────────────────────────────────────────────────

if st.session_state.awaiting_review:
    config = get_config()
    current = graph.get_state(config)
    draft = current.values.get("draft_report", "*(no draft found)*")

    with st.container(border=True):
        st.subheader("📋 Draft Ready for Review")
        st.info(
            "Read the report below. Click **Approve** to finalise, or provide "
            "specific feedback and click **Request Revisions**.",
            icon="ℹ️",
        )

        st.markdown(draft)
        st.divider()

        feedback = st.text_area(
            "Feedback (leave empty to approve):",
            placeholder=(
                "e.g. 'Add more detail on the economic impact in section 3' "
                "or leave blank and click Approve"
            ),
            key="hitl_feedback",
        )

        col1, col2 = st.columns(2)

        # ── Approve ──
        if col1.button("✅ Approve & Finalise", type="primary", use_container_width=True):
            with st.spinner("Finalising..."):
                graph.invoke(Command(resume="approve"), config=config)

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"✅ **Report finalised.**\n\n{draft}",
            })
            st.session_state.awaiting_review = False
            st.session_state.thread_id = str(uuid.uuid4())  # fresh thread for next topic
            st.rerun()

        # ── Revise ──
        if col2.button("🔄 Request Revisions", use_container_width=True):
            revision_text = feedback.strip() or "Please improve clarity and add more specific detail."
            with st.spinner("Revising..."):
                graph.invoke(Command(resume=revision_text), config=config)

            new_state = graph.get_state(config)

            if new_state.next:
                # Graph interrupted again — new draft is ready for review
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"🔄 Revised based on: *{revision_text}*",
                })
                st.rerun()  # re-render shows the updated draft in the panel above
            else:
                # Unusual path — graph finished without another review
                final = new_state.values.get("draft_report", draft)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"✅ **Report finalised.**\n\n{final}",
                })
                st.session_state.awaiting_review = False
                st.session_state.thread_id = str(uuid.uuid4())
                st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────

elif prompt := st.chat_input("Enter a research topic or question..."):
    # Fresh thread for every new topic
    st.session_state.thread_id = str(uuid.uuid4())
    config = get_config()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    from core.graph import make_initial_state

    initial_state = make_initial_state(prompt)

    with st.chat_message("assistant"):
        with st.status("🔍 Researching...", expanded=True) as status:
            # Stream updates — shows which agent is active in real time
            for chunk in graph.stream(
                initial_state,
                config=config,
                stream_mode="updates",
            ):
                for node_name in chunk:
                    if node_name not in ("__end__", "__interrupt__"):
                        st.write(f"→ **{node_name}** working...")

            status.update(label="✅ Draft ready for your review", state="complete")

        current = graph.get_state(config)

        if current.next:  # graph is paused at the review node
            st.session_state.awaiting_review = True
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Research complete. Draft ready for your review above ↑",
            })
            st.markdown("*Draft ready — see review panel above.*")
            st.rerun()
        else:
            # Graph finished without a review interrupt (edge case)
            final = current.values.get("draft_report", "Research complete.")
            st.markdown(final)
            st.session_state.messages.append({"role": "assistant", "content": final})
