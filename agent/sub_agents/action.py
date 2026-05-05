"""
Action sub-agent.
Asks the LLM to confirm job details, then pushes the job onto the Redis queue.
This node only runs after a human has approved the action.
"""

import json
import os
import uuid
from pathlib import Path

import anthropic
import redis

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "action.md"
MODEL       = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
REDIS_URL   = os.getenv("REDIS_URL", "redis://localhost:6379")
QUEUE_KEY   = "queue:jobs"


def _get_redis() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)


def run(
    investigation_id: str,
    recommended_action: str,
    reason: str,
    llm_client: anthropic.Anthropic | None = None,
    redis_client: redis.Redis | None = None,
) -> dict:
    """
    Args:
        investigation_id:   unique ID for this investigation.
        recommended_action: action approved by the human.
        reason:             why it was recommended (from triage).
        llm_client:         injectable for tests.
        redis_client:       injectable for tests.
    Returns:
        dict with keys: job_id, job_type, idempotency_key, queued
    """
    if llm_client is None:
        llm_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if redis_client is None:
        redis_client = _get_redis()

    system_prompt = PROMPT_PATH.read_text()

    user_input = json.dumps({
        "investigation_id":   investigation_id,
        "recommended_action": recommended_action,
        "reason":             reason,
        "approved_by":        "human",
    })

    response = llm_client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=system_prompt,
        messages=[{"role": "user", "content": user_input}],
    )

    raw = response.content[0].text.strip()

    try:
        job_spec = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: build the job spec ourselves
        job_spec = {
            "job_type":        recommended_action,
            "idempotency_key": f"{investigation_id}-{recommended_action}",
            "payload": {
                "investigation_id": investigation_id,
                "triggered_by":     "drift_triage_agent",
            },
        }

    # Add a unique job_id for tracking
    job_spec["job_id"] = str(uuid.uuid4())

    # Idempotency check — don't enqueue the same job twice
    idempotency_key = job_spec["idempotency_key"]
    already_queued  = redis_client.sismember("queue:processed_keys", idempotency_key)

    if already_queued:
        return {**job_spec, "queued": False, "reason": "duplicate — idempotency key already seen"}

    # Push to queue
    redis_client.rpush(QUEUE_KEY, json.dumps(job_spec))

    return {**job_spec, "queued": True}
