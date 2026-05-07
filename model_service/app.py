"""
FastAPI model service.
  - POST /predict          serve predictions
  - GET  /drift-report     compute drift over rolling window
  - POST /promote/{version} promote a model version to Production
"""

import json
import pickle
import tempfile
from contextlib import asynccontextmanager

import mlflow
import numpy as np
import pandas as pd
import psycopg2
from fastapi import FastAPI
from mlflow import MlflowClient

from model_service.config import DATABASE_URL, MODEL_NAME
from model_service.router import router
from model_service.state import state


def get_db():
    return psycopg2.connect(DATABASE_URL)


# ---------------------------------------------------------------------------
# Lifespan — runs once on startup and shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Create predictions table if it does not exist
    conn = get_db()
    with conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id          SERIAL PRIMARY KEY,
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    input_json  JSONB NOT NULL,
                    probability FLOAT NOT NULL,
                    prediction  INT   NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS drift_state (
                    id       INT PRIMARY KEY DEFAULT 1,
                    severity TEXT NOT NULL DEFAULT 'none'
                )
            """)
            cur.execute("""
                INSERT INTO drift_state (id, severity)
                VALUES (1, 'none')
                ON CONFLICT (id) DO NOTHING
            """)
    conn.close()

    # 2. Load the Production model from MLflow (fall back to latest version)
    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        versions = client.get_latest_versions(MODEL_NAME)
    if not versions:
        raise RuntimeError(f"No registered versions found for model '{MODEL_NAME}'")

    mv = versions[0]
    run_id = mv.run_id

    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="model", dst_path=tmpdir
        )
        with open(f"{tmpdir}/model/pipeline.pkl", "rb") as f:
            state["model"] = pickle.load(f)

        with open(f"{tmpdir}/model/model_card.json") as f:
            card = json.load(f)
            state["threshold"] = card["operating_threshold"]
            state["metrics"]   = card["metrics"]

        with open(f"{tmpdir}/model/schema.json") as f:
            schema = json.load(f)
            state["numeric_cols"]     = schema["numeric_cols"]
            state["categorical_cols"] = schema["categorical_cols"]

        with open(f"{tmpdir}/model/reference_stats.json") as f:
            ref = json.load(f)
            state["ref_numeric"]     = {k: np.array(v["values"]) for k, v in ref["numeric"].items()}
            state["ref_categorical"] = {k: pd.Series(v) for k, v in ref["categorical"].items()}
            state["ref_scores"]      = np.array(ref["scores"])

    state["model_version"] = mv.version
    print(f"Loaded {MODEL_NAME} v{mv.version} (threshold={state['threshold']})")

    yield  # app runs here

    # Shutdown — nothing to clean up
    state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Bank Marketing Model Service", lifespan=lifespan)
app.include_router(router)
