"""
FastAPI model service.
  - POST /predict          serve predictions
  - GET  /drift-report     compute drift over rolling window
  - POST /promote/{version} promote a model version to Production
"""

import json
import os
import pickle
import tempfile
from contextlib import asynccontextmanager

import httpx
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import mlflow
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient

from model_service.drift import compute_drift_report
from model_service.schemas import (
    DriftReport,
    PredictRequest,
    PredictResponse,
    PromoteResponse,
)

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "bank-marketing-classifier")
DATABASE_URL        = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/mlops")
AGENT_WEBHOOK_URL   = os.getenv("AGENT_WEBHOOK_URL", "http://agent:8001/drift-event")
DRIFT_WINDOW        = int(os.getenv("DRIFT_WINDOW", "500"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---------------------------------------------------------------------------
# App-level state (populated during lifespan startup)
# ---------------------------------------------------------------------------
state: dict = {}


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
        # Download all model artifacts
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


# ---------------------------------------------------------------------------
# Feature engineering — mirrors what was done in train.py
# ---------------------------------------------------------------------------
def engineer_features(req: PredictRequest) -> pd.DataFrame:
    row = {
        "age":           req.age,
        "job":           req.job,
        "marital":       req.marital,
        "education":     req.education,
        "default":       req.default,
        "housing":       req.housing,
        "loan":          req.loan,
        "contact":       req.contact,
        "month":         req.month,
        "day_of_week":   req.day_of_week,
        "campaign":      req.campaign,
        "previous":      req.previous,
        "poutcome":      req.poutcome,
        "emp.var.rate":  req.emp_var_rate,
        "cons.price.idx": req.cons_price_idx,
        "cons.conf.idx": req.cons_conf_idx,
        "euribor3m":     req.euribor3m,
        "nr.employed":   req.nr_employed,
        # engineered from pdays
        "no_previous_contact": int(req.pdays == 999),
        "pdays_clean":         -1 if req.pdays == 999 else req.pdays,
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    X = engineer_features(req)
    probability = float(state["model"].predict_proba(X)[0, 1])
    prediction  = int(probability >= state["threshold"])

    # Log to DB
    conn = get_db()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO predictions (input_json, probability, prediction) VALUES (%s, %s, %s)",
                (json.dumps(req.model_dump(by_alias=True)), probability, prediction),
            )
    conn.close()

    return PredictResponse(
        prediction=prediction,
        probability=round(probability, 4),
        threshold=state["threshold"],
    )


# ---------------------------------------------------------------------------
# GET /drift-report
# ---------------------------------------------------------------------------
@app.get("/drift-report", response_model=DriftReport)
async def drift_report():
    # Fetch the last DRIFT_WINDOW predictions from DB
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT input_json, probability FROM predictions ORDER BY created_at DESC LIMIT %s",
            (DRIFT_WINDOW,),
        )
        rows = cur.fetchall()
    conn.close()

    if len(rows) < 10:
        raise HTTPException(status_code=400, detail="Not enough predictions yet for drift analysis (need >= 10).")

    current_df     = pd.DataFrame([r["input_json"] for r in rows])
    current_scores = np.array([r["probability"] for r in rows])

    # Build reference DataFrames from stored training distributions
    ref_numeric_df = pd.DataFrame(state["ref_numeric"])
    ref_cat_df     = pd.DataFrame(state["ref_categorical"])

    # Align column names (dots vs underscores from JSON)
    current_df.rename(columns={
        "emp.var.rate":  "emp.var.rate",
        "cons.price.idx": "cons.price.idx",
        "cons.conf.idx": "cons.conf.idx",
        "nr.employed":   "nr.employed",
    }, inplace=True)

    report = compute_drift_report(
        reference_df=ref_numeric_df,
        current_df=current_df,
        reference_scores=state["ref_scores"],
        current_scores=current_scores,
        numeric_cols=state["numeric_cols"],
        categorical_cols=state["categorical_cols"],
    )

    # Emit webhook if severity changed
    conn = get_db()
    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT severity FROM drift_state WHERE id = 1")
            last_severity = cur.fetchone()[0]

            if report.severity != last_severity:
                cur.execute(
                    "UPDATE drift_state SET severity = %s WHERE id = 1",
                    (report.severity,),
                )
                await _emit_drift_webhook(report)

    conn.close()
    return report


async def _emit_drift_webhook(report: DriftReport):
    payload = {
        "severity":    report.severity,
        "window_size": report.window_size,
        "output_drift": report.output_drift,
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(AGENT_WEBHOOK_URL, json=payload)
    except Exception as exc:
        # Webhook failure must never crash the drift endpoint
        print(f"Webhook delivery failed: {exc}")


# ---------------------------------------------------------------------------
# POST /promote/{version}
# ---------------------------------------------------------------------------
@app.post("/promote/{version}", response_model=PromoteResponse)
def promote(version: int):
    client = MlflowClient()

    # Fetch model version details
    try:
        mv = client.get_model_version(MODEL_NAME, str(version))
    except Exception:
        raise HTTPException(status_code=404, detail=f"Version {version} not found in registry.")

    # Download model card to run the checklist
    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.artifacts.download_artifacts(
            run_id=mv.run_id, artifact_path="model/model_card.json", dst_path=tmpdir
        )
        with open(f"{tmpdir}/model/model_card.json") as f:
            card = json.load(f)

    _assert_promotion_checklist(card, mv)

    # Promote
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=str(version),
        stage="Production",
        archive_existing_versions=True,
    )

    return PromoteResponse(
        version=version,
        stage="Production",
        message=f"{MODEL_NAME} v{version} promoted to Production.",
    )


def _assert_promotion_checklist(card: dict, mv) -> None:
    """Raise HTTPException if any gate fails."""
    errors = []

    metrics = card.get("metrics", {})

    if metrics.get("test_auc", 0) < 0.75:
        errors.append(f"test_auc {metrics.get('test_auc')} < 0.75")

    if metrics.get("test_recall", 0) < 0.75:
        errors.append(f"test_recall {metrics.get('test_recall')} < 0.75")

    if mv.current_stage == "Production":
        errors.append("Version is already in Production.")

    if not card.get("dataset_hash"):
        errors.append("model_card missing dataset_hash.")

    if not card.get("sklearn_version"):
        errors.append("model_card missing sklearn_version.")

    if errors:
        raise HTTPException(status_code=422, detail={"promotion_checklist_failures": errors})
