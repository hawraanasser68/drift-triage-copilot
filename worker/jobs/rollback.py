"""
Rollback job: find the most recent Staging or Archived version
and promote it to Production, archiving the current Production version.
"""

import os

import mlflow
import psycopg2
from mlflow import MlflowClient

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/mlops")


def _reset_drift_state():
    """Reset drift state to none so the next drift episode can be detected."""
    conn = psycopg2.connect(DATABASE_URL)
    with conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE drift_state SET severity = 'none' WHERE id = 1")
    conn.close()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "bank-marketing-classifier")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def run(payload: dict):
    """
    payload keys (all optional):
      - investigation_id
      - triggered_by
    """
    client = MlflowClient()

    # Find current Production version
    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not prod_versions:
        raise RuntimeError("No Production version to roll back from.")
    current_prod = prod_versions[0]

    # Find the previous version to restore:
    # look for the highest version number that is NOT the current Production one
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    candidates = [
        v for v in all_versions
        if int(v.version) < int(current_prod.version)
    ]

    if not candidates:
        raise RuntimeError("No previous version available to roll back to.")

    # Pick the most recent candidate
    previous = max(candidates, key=lambda v: int(v.version))

    print(f"Rolling back: v{current_prod.version} → v{previous.version}")

    # Archive current Production, promote previous
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=str(current_prod.version),
        stage="Archived",
    )
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=str(previous.version),
        stage="Production",
        archive_existing_versions=False,
    )

    print(f"Rollback complete. v{previous.version} is now Production.")

    _reset_drift_state()
    print("Drift state reset to 'none' — system ready to detect the next drift episode.")
