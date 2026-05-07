"""
Retrain job: re-run train.py and register a new model version.
The new version lands in Staging — promotion to Production still
requires a human via the normal gate.
"""

import os
import subprocess
import sys

import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/mlops")


def _reset_drift_state():
    """Reset drift state to none so the next drift episode can be detected."""
    conn = psycopg2.connect(DATABASE_URL)
    with conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE drift_state SET severity = 'none' WHERE id = 1")
    conn.close()


def run(payload: dict):
    """
    payload keys (all optional):
      - investigation_id
      - triggered_by
    """
    print(f"Starting retrain (triggered by investigation {payload.get('investigation_id')}) ...")

    result = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True,
        text=True,
        cwd=os.getenv("APP_DIR", "."),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"train.py exited with code {result.returncode}.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    print("Retrain complete.")
    print(result.stdout)

    _reset_drift_state()
    print("Drift state reset to 'none' — system ready to detect the next drift episode.")
