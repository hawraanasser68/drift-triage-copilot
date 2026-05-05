"""
FastAPI app for the triage agent.
  - POST /drift-event     receives webhook from the model service
  - POST /approve/{id}    dashboard calls this after human approves/rejects
  - GET  /investigations  dashboard polls this for open/resolved investigations
"""

import json
import os
from contextlib import asynccontextmanager

import psycopg2
import psycopg2.extras
from fastapi import BackgroundTasks, FastAPI, HTTPException
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import BaseModel

from agent.supervisor import create_graph, resume_investigation, start_investigation

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/mlops")

# App-level state
state: dict = {}


# ---------------------------------------------------------------------------
# Lifespan — set up DB tables and compile the graph once on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create the investigations table (comms sub-agent writes to it)
    conn = psycopg2.connect(DATABASE_URL)
    with conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS investigations (
                    investigation_id TEXT PRIMARY KEY,
                    severity         TEXT,
                    recommended_action TEXT,
                    human_approved   BOOLEAN,
                    job_id           TEXT,
                    summary          TEXT,
                    status           TEXT DEFAULT 'open',
                    created_at       TIMESTAMPTZ DEFAULT NOW(),
                    updated_at       TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pending_approvals (
                    investigation_id   TEXT PRIMARY KEY,
                    recommended_action TEXT,
                    reason             TEXT,
                    interrupt_data     JSONB,
                    created_at         TIMESTAMPTZ DEFAULT NOW()
                )
            """)
    conn.close()

    # Compile the LangGraph graph with a Postgres checkpointer
    checkpointer = PostgresSaver.from_conn_string(DATABASE_URL)
    checkpointer.setup()  # creates LangGraph's internal checkpoint tables
    state["graph"] = create_graph(checkpointer)

    yield

    state.clear()


app = FastAPI(title="Drift Triage Agent", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class DriftEvent(BaseModel):
    severity:     str
    window_size:  int
    output_drift: float


class ApproveRequest(BaseModel):
    approved: bool


# ---------------------------------------------------------------------------
# POST /drift-event
# Starts a new investigation in the background so the webhook returns fast.
# ---------------------------------------------------------------------------
@app.post("/drift-event", status_code=202)
def receive_drift_event(event: DriftEvent, background_tasks: BackgroundTasks):
    drift_payload = event.model_dump()
    background_tasks.add_task(_run_investigation, drift_payload)
    return {"message": "Investigation started."}


def _run_investigation(drift_event: dict):
    try:
        investigation_id, is_paused, interrupt_data = start_investigation(drift_event, state["graph"])
        print(f"Investigation {investigation_id} started (severity={drift_event['severity']}, paused={is_paused})")

        if is_paused and interrupt_data:
            conn = psycopg2.connect(DATABASE_URL)
            with conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO pending_approvals
                            (investigation_id, recommended_action, reason, interrupt_data)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (investigation_id) DO NOTHING
                    """, (
                        investigation_id,
                        interrupt_data.get("recommended_action"),
                        interrupt_data.get("reason"),
                        json.dumps(interrupt_data),
                    ))
            conn.close()
    except Exception as exc:
        print(f"Investigation failed to start: {exc}")


# ---------------------------------------------------------------------------
# POST /approve/{investigation_id}
# Dashboard calls this after the human clicks Approve or Reject.
# ---------------------------------------------------------------------------
@app.post("/approve/{investigation_id}")
def approve(investigation_id: str, body: ApproveRequest):
    # Check the investigation exists and is waiting for approval
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT status FROM investigations WHERE investigation_id = %s",
            (investigation_id,),
        )
        row = cur.fetchone()
    conn.close()

    # Row may not exist yet if comms hasn't written it — that's fine,
    # it means the graph is still in the action node waiting for approval.
    # We resume regardless; the graph will handle an unknown ID gracefully.

    try:
        resume_investigation(investigation_id, body.approved, state["graph"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Remove from pending approvals now that a decision was made
    conn = psycopg2.connect(DATABASE_URL)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM pending_approvals WHERE investigation_id = %s",
                (investigation_id,),
            )
    conn.close()

    return {
        "investigation_id": investigation_id,
        "approved": body.approved,
        "message": "Investigation resumed.",
    }


# ---------------------------------------------------------------------------
# GET /investigations
# Dashboard reads this to show open and resolved investigations.
# ---------------------------------------------------------------------------
@app.get("/pending-approvals")
def list_pending_approvals():
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM pending_approvals ORDER BY created_at DESC")
        rows = cur.fetchall()
    conn.close()
    return {"pending_approvals": [dict(r) for r in rows]}


@app.get("/investigations")
def list_investigations(status: str | None = None):
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if status:
            cur.execute(
                "SELECT * FROM investigations WHERE status = %s ORDER BY created_at DESC",
                (status,),
            )
        else:
            cur.execute("SELECT * FROM investigations ORDER BY created_at DESC")
        rows = cur.fetchall()
    conn.close()
    return {"investigations": [dict(r) for r in rows]}
