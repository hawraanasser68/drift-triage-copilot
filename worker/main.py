"""
Redis queue worker.
Pulls jobs from queue:jobs, runs the appropriate handler,
retries with exponential backoff, and moves failed jobs to the DLQ.
"""

import json
import os
import time

import redis

from worker.jobs import replay, retrain, rollback

REDIS_URL  = os.getenv("REDIS_URL", "redis://localhost:6379")
QUEUE_KEY  = "queue:jobs"
DLQ_KEY    = "queue:dlq"
RETRY_KEY  = "queue:retries"        # Redis hash: idempotency_key → retry_count
DONE_KEY   = "queue:processed_keys" # Redis set:  idempotency keys that succeeded
MAX_RETRIES = int(os.getenv("WORKER_MAX_RETRIES", "3"))

HANDLERS = {
    "replay_test": replay.run,
    "retrain":     retrain.run,
    "rollback":    rollback.run,
}


def run_worker(redis_client: redis.Redis | None = None):
    """Main blocking loop. Pass a redis_client for tests."""
    r = redis_client or redis.from_url(REDIS_URL, decode_responses=True)
    print(f"Worker started. Listening on {QUEUE_KEY} ...")

    while True:
        # BLPOP blocks until a job arrives (timeout=0 = wait forever)
        item = r.blpop(QUEUE_KEY, timeout=5)
        if item is None:
            continue  # timeout with no job — loop again

        _, raw = item
        try:
            job = json.loads(raw)
        except json.JSONDecodeError:
            print(f"Malformed job, sending to DLQ: {raw}")
            r.rpush(DLQ_KEY, raw)
            continue

        _process(job, r)


def _process(job: dict, r: redis.Redis):
    idempotency_key = job.get("idempotency_key", job.get("job_id", "unknown"))
    job_type        = job.get("job_type")

    # Idempotency: skip if already processed successfully
    if r.sismember(DONE_KEY, idempotency_key):
        print(f"[{idempotency_key}] Already processed — skipping.")
        return

    handler = HANDLERS.get(job_type)
    if handler is None:
        print(f"[{idempotency_key}] Unknown job type '{job_type}' — sending to DLQ.")
        r.rpush(DLQ_KEY, json.dumps(job))
        return

    retry_count = int(r.hget(RETRY_KEY, idempotency_key) or 0)

    try:
        print(f"[{idempotency_key}] Running {job_type} (attempt {retry_count + 1}) ...")
        handler(job.get("payload", {}))

        # Success — mark as processed
        r.sadd(DONE_KEY, idempotency_key)
        r.hdel(RETRY_KEY, idempotency_key)
        print(f"[{idempotency_key}] Done.")

    except Exception as exc:
        retry_count += 1
        print(f"[{idempotency_key}] Failed (attempt {retry_count}): {exc}")

        if retry_count >= MAX_RETRIES:
            print(f"[{idempotency_key}] Max retries reached — sending to DLQ.")
            r.rpush(DLQ_KEY, json.dumps({**job, "error": str(exc), "retries": retry_count}))
            r.hdel(RETRY_KEY, idempotency_key)
        else:
            # Exponential backoff: 2^retry seconds (2s, 4s, 8s)
            backoff = 2 ** retry_count
            print(f"[{idempotency_key}] Retrying in {backoff}s ...")
            time.sleep(backoff)
            r.hset(RETRY_KEY, idempotency_key, retry_count)
            r.rpush(QUEUE_KEY, json.dumps(job))  # re-queue


if __name__ == "__main__":
    run_worker()
