"""
core/ui_documents.py
Streamlit surface for the two document tiers.

Kept out of app.py so the UI logic stays testable and app.py keeps its shape.

THE TWO TIERS, and why they are presented as clearly separate things:

  ARCHIVE   a folder indexed once, persistent across restarts, shared by every
            chat. This is the institution's knowledge base.
  ATTACHED  files dropped into THIS chat. They never touch the persistent index
            and disappear when the chat ends.

A reader must never have to guess which one an answer came from — "this is in
our archive" and "this is in the file you just gave me" are different claims
about provenance, so the UI labels them separately and so do the citations.

THE STREAMLIT RERUN TRAP
  Streamlit re-executes this whole script on every interaction, and
  st.file_uploader hands back the SAME files each time. A naive implementation
  therefore re-parses and re-embeds every attachment on every button click — for
  a 300-page PDF that is minutes of work per keystroke. Attachments are keyed by
  content hash in st.session_state, so each file is ingested exactly once.

WHY NOT thread_id
  app.py mints a NEW thread_id for every question and again when a report is
  finalised, so a chat spans many threads. Keying attachments on thread_id would
  destroy them after the first question. They are keyed on a session id that
  lives as long as the browser session, and every thread the chat uses is
  recorded so teardown can scrub the checkpointer too.
"""

from __future__ import annotations

import hashlib
import os
import uuid

_SUPPORTED = ("pdf", "xlsx", "xlsm", "docx", "csv", "md", "txt")
_MAX_UPLOAD_MB = int(os.getenv("ATHENA_MAX_UPLOAD_MB", "25"))


def ensure_session(st) -> str:
    """Stable per-browser-session id, independent of the graph's thread_id."""
    if "athena_session_id" not in st.session_state:
        st.session_state.athena_session_id = f"ui-{uuid.uuid4().hex[:16]}"
    if "attached_hashes" not in st.session_state:
        st.session_state.attached_hashes = {}
    return st.session_state.athena_session_id


def bind_current_thread(st) -> None:
    """
    Record the thread this chat is currently using.

    Attachments outlive any single thread, but their passages end up in the
    LangGraph checkpointer keyed BY thread, so every thread must be remembered
    or teardown cannot scrub them all.
    """
    from core import sessions

    sid = ensure_session(st)
    tid = st.session_state.get("thread_id")
    if tid and sessions.REGISTRY.get(sid) is not None:
        sessions.bind_thread(sid, tid)


def documents_active(st) -> bool:
    """True when the researcher should search documents rather than the web."""
    return bool(st.session_state.get("documents_mode", False))


def _fmt_bytes(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / (1024 * 1024):.1f} MB"


def render_sidebar(st) -> None:
    """Render both tiers. Called from app.py's existing sidebar block."""
    from core import index as idx
    from core import sessions

    sid = ensure_session(st)

    st.markdown("### Documents")
    mode = st.toggle(
        "Search documents instead of the web",
        value=st.session_state.get("documents_mode", False),
        help="Answers come from the archive and any files attached to this chat. "
             "Nothing is sent to a search engine.",
    )
    st.session_state.documents_mode = mode
    # The researcher reads this when choosing its tools.
    os.environ["ATHENA_MODE"] = "documents" if mode else "web"

    # ── Archive ──────────────────────────────────────────────────────────────
    try:
        stats = idx.index_stats()
        has_index = stats["chunks"] > 0
    except Exception as e:  # noqa: BLE001
        stats, has_index = None, False
        st.caption(f"Archive unavailable: {e}")

    if has_index:
        # Tables are shown next to passages because they are a different
        # capability, not a subset: passages are what search can find, rows are
        # what counting is exact over.
        countable = (
            f" · {stats['fact_tables']} countable tables ({stats['fact_rows']} rows)"
            if stats.get("fact_tables") else ""
        )
        st.caption(
            f"**Archive** · {stats['documents']} documents · "
            f"{stats['chunks']} passages{countable} · `{stats['embed_model']}`"
        )
        if stats.get("years"):
            st.caption("Years: " + ", ".join(y for y in stats["years"] if y))
    else:
        st.caption(
            "**Archive** · not indexed yet. Put documents in the corpus folder "
            "and run `python ingest.py`."
        )

    # ── Attached to this chat ────────────────────────────────────────────────
    st.caption("**Attached to this chat** — never added to the archive, gone when you clear it.")
    uploads = st.file_uploader(
        "Attach documents",
        type=list(_SUPPORTED),
        accept_multiple_files=True,
        key="doc_uploader",
        label_visibility="collapsed",
    )

    if uploads:
        store = sessions.get_or_create(sid)
        for up in uploads:
            data = up.getvalue()
            digest = hashlib.sha256(data).hexdigest()
            # Ingest each file ONCE. Streamlit replays the uploader on every
            # rerun, so without this the same PDF is re-embedded continuously.
            if digest in st.session_state.attached_hashes:
                continue
            if len(data) > _MAX_UPLOAD_MB * 1024 * 1024:
                st.warning(f"{up.name} exceeds {_MAX_UPLOAD_MB} MB and was not attached.")
                st.session_state.attached_hashes[digest] = {"name": up.name, "error": "too large"}
                continue
            with st.spinner(f"Reading {up.name}…"):
                res = store.add_document(up.name, data)
            st.session_state.attached_hashes[digest] = {
                "name": up.name,
                "chunks": res.get("chunks", 0),
                "error": None if res.get("ok") else res.get("error"),
                "notices": res.get("notices", []),
            }

    attached = list(st.session_state.attached_hashes.values())
    if attached:
        for a in attached:
            if a.get("error"):
                st.caption(f"· {a['name']} — not attached: {a['error']}")
            else:
                st.caption(f"· {a['name']} — {a.get('chunks', 0)} passages")
            # A file that only partly parsed must say so. "Not in the documents"
            # and "that file could not be read" are different answers.
            for n in (a.get("notices") or [])[:2]:
                st.caption(f"  ⚠ {n[:120]}")

        if st.button("Clear attached documents", use_container_width=True):
            report = sessions.end_session(sid)
            st.session_state.attached_hashes = {}
            st.session_state.pop("doc_uploader", None)
            st.session_state.athena_session_id = f"ui-{uuid.uuid4().hex[:16]}"
            st.caption(
                f"Removed. Checkpointed state scrubbed for "
                f"{report.get('threads', 0)} thread(s)."
            )
            st.rerun()

    if mode and not has_index and not attached:
        st.warning(
            "Document mode is on, but there is no archive and nothing is attached — "
            "the researcher has nothing to search."
        )
