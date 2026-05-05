"""
Comms sub-agent.
Asks the LLM to write a plain-English summary of the investigation outcome,
then persists it to Postgres so the dashboard can display it.
"""

import json
import os
from pathlib import Path

import anthropic
import psycopg2

PROMPT_PATH  = Path(__file__).parent.parent / "prompts" / "comms.md"
MODEL        = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/mlops")


def _get_db():
    return psycopg2.connect(DATABASE_URL)


def run(
    investigation_id: str,
    severity: str,
    recommended_action: str,
    human_approved: bool | None,
    job_id: str | None,
    llm_client: anthropic.Anthropic | None = None,
    db_conn=None,
) -> dict:
    """
    Args:
        investigation_id: unique ID.
        severity:         drift severity that triggered the investigation.
        recommended_action: what was recommended by triage.
        human_approved:   True / False / None (None = no approval required).
        job_id:           Redis job ID if dispatched, else None.
        llm_client:       injectable for tests.
        db_conn:          injectable psycopg2 connection for tests.
    Returns:
        dict with keys: summary, status
    """
    if llm_client is None:
        llm_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Determine outcome label
    if human_approved is False:
        outcome = "rejected"
    elif job_id is not None:
        outcome = "dispatched"
    else:
        outcome = "monitored"

    system_prompt = PROMPT_PATH.read_text()

    user_input = json.dumps({
        "investigation_id":   investigation_id,
        "severity":           severity,
        "recommended_action": recommended_action,
        "human_approved":     human_approved,
        "job_id":             job_id,
        "outcome":            outcome,
    })

    response = llm_client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=system_prompt,
        messages=[{"role": "user", "content": user_input}],
    )

    raw = response.content[0].text.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "summary": f"Investigation {investigation_id} completed with outcome: {outcome}.",
            "status":  "resolved" if outcome in ("dispatched", "monitored") else "rejected",
        }

    # Persist to DB so the dashboard can read it
    close_conn = db_conn is None
    if db_conn is None:
        db_conn = _get_db()

    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO investigations (investigation_id, severity, recommended_action,
                                           human_approved, job_id, summary, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (investigation_id) DO UPDATE
                    SET summary = EXCLUDED.summary,
                        status  = EXCLUDED.status
            """, (
                investigation_id, severity, recommended_action,
                human_approved, job_id,
                result["summary"], result["status"],
            ))

    if close_conn:
        db_conn.close()

    return result
