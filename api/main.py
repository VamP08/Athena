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
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, HTTPException, status
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


def _run_research(thread_id: str, topic: str) -> None:
    try:
        graph.invoke(make_initial_state(topic), config=_config(thread_id))
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
    _submit(_run_research, thread_id, body.topic)
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


@app.get("/health")
def health():
    from core.llm import resolve_backend

    backend, model = resolve_backend()
    return {
        "status": "ok",
        "checkpointer": os.getenv("ATHENA_CHECKPOINTER", "memory"),
        "model_backend": backend,
        "model": model,
        "auth": "enabled" if os.getenv("ATHENA_API_TOKEN") else "disabled",
    }
