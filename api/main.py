"""
api/main.py
Athena as a service: the research pipeline behind a REST API.

    POST /research                     start a research job          → 202
    GET  /research/{thread_id}         status + draft when available
    POST /research/{thread_id}/resume  approve or request a revision → 202
    GET  /research/{thread_id}/audit   the human-decision audit trail
    GET  /research                     recent jobs
    GET  /health                       liveness + runtime config

Run:
    uvicorn api.main:app --port 8000

Auth: set ATHENA_API_TOKEN to require `Authorization: Bearer <token>` on all
/research endpoints. Unset = auth disabled (local development).

Jobs run on a worker pool — graph execution is synchronous LangGraph code, so
the API stays responsive while research runs. Durable state (checkpointer +
registry) means a paused review survives an API restart when
ATHENA_CHECKPOINTER=sqlite.

ATHENA_API_SYNC=1 runs jobs inline instead of on the pool — for deterministic
tests only.
"""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langgraph.types import Command
from pydantic import BaseModel, Field

from api.registry import Registry
from core.graph import build_graph, make_initial_state

app = FastAPI(
    title="Athena API",
    description="Multi-agent research analyst with a human approval gate.",
    version="0.2.0",
)

graph = build_graph()
registry = Registry(os.getenv("ATHENA_DB_PATH", "athena.db"))
executor = ThreadPoolExecutor(max_workers=int(os.getenv("ATHENA_WORKERS", "4")))

# ── Auth ──────────────────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


def require_auth(credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    expected = os.getenv("ATHENA_API_TOKEN", "")
    if not expected:
        return  # auth disabled — local development
    if credentials is None or credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token",
        )


# ── Schemas ───────────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=500)
    # Optional chat scope. When present, the run may also search the documents
    # attached to THAT session and no other — the binding is a closure over one
    # store, never a filter, so a wrong or stale id degrades to archive-only
    # rather than exposing someone else's files.
    session_id: str = Field(default="", max_length=64)


class ResumeRequest(BaseModel):
    action: Literal["approve", "revise"]
    feedback: str = Field(default="", max_length=4000)


class JobResponse(BaseModel):
    thread_id: str
    status: str


class StatusResponse(BaseModel):
    thread_id: str
    topic: str
    status: str
    draft_report: str | None = None
    error: str | None = None
    created_at: str
    updated_at: str


# ── Job execution ─────────────────────────────────────────────────────────────

def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _finish(thread_id: str) -> None:
    """Set registry status from the graph's post-run state."""
    state = graph.get_state(_config(thread_id))
    if state.next:  # paused at the review gate
        registry.set_status(thread_id, "awaiting_review")
    else:
        registry.set_status(thread_id, "completed")


def _run_research(thread_id: str, topic: str, session_id: str = "") -> None:
    try:
        graph.invoke(
            make_initial_state(topic, session_id=session_id),
            config=_config(thread_id),
        )
        _finish(thread_id)
    except Exception as e:  # surface the failure to clients, don't swallow it
        registry.set_status(thread_id, "failed", error=f"{type(e).__name__}: {e}")


def _run_resume(thread_id: str, resume_value: str) -> None:
    try:
        graph.invoke(Command(resume=resume_value), config=_config(thread_id))
        _finish(thread_id)
    except Exception as e:
        registry.set_status(thread_id, "failed", error=f"{type(e).__name__}: {e}")


def _submit(fn, *args) -> None:
    if os.getenv("ATHENA_API_SYNC") == "1":
        fn(*args)
    else:
        executor.submit(fn, *args)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/research",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_auth)],
)
def start_research(body: ResearchRequest):
    thread_id = str(uuid.uuid4())
    registry.create(thread_id, body.topic)
    registry.log_action(thread_id, "created", detail=body.topic)
    if body.session_id:
        # Record the pairing so ending the chat can scrub this thread's
        # checkpointed passages too: destroying the store alone is not enough,
        # because retrieved text also lands in the checkpointer.
        from core import sessions

        sessions.bind_thread(body.session_id, thread_id)
    _submit(_run_research, thread_id, body.topic, body.session_id)
    return JobResponse(thread_id=thread_id, status="running")


@app.get(
    "/research/{thread_id}",
    response_model=StatusResponse,
    dependencies=[Depends(require_auth)],
)
def get_research(thread_id: str):
    row = registry.get(thread_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown thread_id")

    draft = None
    if row["status"] in ("awaiting_review", "completed"):
        draft = graph.get_state(_config(thread_id)).values.get("draft_report")

    return StatusResponse(**row, draft_report=draft)


@app.post(
    "/research/{thread_id}/resume",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_auth)],
)
def resume_research(thread_id: str, body: ResumeRequest):
    row = registry.get(thread_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown thread_id")
    if row["status"] != "awaiting_review":
        raise HTTPException(
            status_code=409,
            detail=f"Thread is '{row['status']}', not awaiting review",
        )
    if body.action == "revise" and not body.feedback.strip():
        raise HTTPException(status_code=422, detail="Revision requires feedback")

    resume_value = "approve" if body.action == "approve" else body.feedback.strip()
    registry.log_action(thread_id, body.action, detail=body.feedback.strip())
    registry.set_status(thread_id, "running")
    _submit(_run_resume, thread_id, resume_value)
    return JobResponse(thread_id=thread_id, status="running")


@app.get("/research/{thread_id}/audit", dependencies=[Depends(require_auth)])
def get_audit(thread_id: str):
    if registry.get(thread_id) is None:
        raise HTTPException(status_code=404, detail="Unknown thread_id")
    return {"thread_id": thread_id, "audit": registry.audit_trail(thread_id)}


@app.get("/research", dependencies=[Depends(require_auth)])
def list_research(limit: int = 20):
    return {"threads": registry.list_recent(limit=min(limit, 100))}


# ── Documents ─────────────────────────────────────────────────────────────────
#
# Two tiers, deliberately separate endpoints:
#   /documents/archive   the persistent knowledge base — read-only here, built
#                        by ingest.py. Uploading to it through the API is NOT
#                        offered: the archive is an operator-curated folder, and
#                        an endpoint that silently grows it would make "what is
#                        in our records" untrackable.
#   /sessions/{id}/...   documents attached to one chat. Ephemeral by design.

_UPLOAD_SUFFIXES = {".pdf", ".xlsx", ".xlsm", ".docx", ".csv", ".md", ".txt"}
_MAX_UPLOAD_BYTES = int(os.getenv("ATHENA_MAX_UPLOAD_MB", "25")) * 1024 * 1024


@app.get("/documents/archive", dependencies=[Depends(require_auth)])
def archive_status():
    """What the permanent archive currently holds."""
    from core import index as idx

    try:
        return idx.index_stats()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Archive unavailable: {e}") from e


@app.post(
    "/sessions/{session_id}/documents",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_auth)],
)
async def attach_document(session_id: str, file: UploadFile = File(...)):
    """
    Attach one document to a chat. It never enters the persistent archive.

    The filename is used only for display and for its suffix; it never builds a
    path, so a name like "../../athena_index.db" cannot escape anywhere.
    """
    from core import sessions

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported type '{suffix}'. Allowed: {sorted(_UPLOAD_SUFFIXES)}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Empty file")
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    store = sessions.get_or_create(session_id)
    # A multipart part can arrive with no filename at all; the parsers dispatch
    # on the suffix, so a None here would crash the attach instead of degrading
    # to the unsupported-format notice that "no idea what this is" deserves.
    result = store.add_document(file.filename or "unnamed_upload", data)
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=result.get("error", "Attach failed"))
    return result


@app.get("/sessions/{session_id}/documents", dependencies=[Depends(require_auth)])
def list_session_documents(session_id: str):
    """What is attached to this chat. 404 if the chat has none or has expired."""
    from core import sessions

    store = sessions.REGISTRY.get(session_id)
    if store is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session_id")
    return {"session_id": session_id, "stats": store.stats(),
            "documents": store.list_documents()}


@app.delete("/sessions/{session_id}", dependencies=[Depends(require_auth)])
def end_session(session_id: str):
    """
    End a chat: drop its documents AND scrub the checkpointed passages.

    Both halves are required. Retrieved text also lives in graph state and
    message history, so destroying the store alone would leave verbatim copies
    of an uploaded document in the checkpointer — and "it disappears when the
    chat ends" would be false at the byte level.
    """
    from core import sessions

    if sessions.REGISTRY.get(session_id) is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session_id")
    return sessions.end_session(session_id, checkpointer=getattr(graph, "checkpointer", None))


@app.get("/health")
def health():
    from core.llm import resolve_backend

    backend, model = resolve_backend()
    out = {
        "status": "ok",
        "checkpointer": os.getenv("ATHENA_CHECKPOINTER", "memory"),
        "model_backend": backend,
        "model": model,
        "auth": "enabled" if os.getenv("ATHENA_API_TOKEN") else "disabled",
    }

    # A caller integrating against this API cannot see the Streamlit warning, so
    # the operating regime is reported here too. /health failing when there is
    # simply no archive yet would be wrong — document mode is optional.
    try:
        from core import index as idx

        stats = idx.index_stats()
        out["archive"] = {
            "documents": stats["documents"],
            "chunks": stats["chunks"],
            "tested_chunk_limit": stats["tested_chunk_limit"],
            "within_tested_envelope": stats["within_tested_envelope"],
        }
    except Exception:  # noqa: BLE001 - no index is a valid state, not an outage
        pass
    return out
