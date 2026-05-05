"""
Replay job: load the Production model and re-score X_test.
Asserts that test AUC matches the registered value within 1e-12.
"""

import json
import os
import pickle
import tempfile

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import roc_auc_score

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "bank-marketing-classifier")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def run(payload: dict):
    """
    payload keys (all optional — we load from MLflow if not provided):
      - investigation_id
      - triggered_by
    """
    client = MlflowClient()

    # Load the Production model
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError("No Production model found in registry.")
    mv = versions[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.artifacts.download_artifacts(
            run_id=mv.run_id, artifact_path="model", dst_path=tmpdir
        )

        with open(f"{tmpdir}/model/pipeline.pkl", "rb") as f:
            model = pickle.load(f)

        with open(f"{tmpdir}/model/model_card.json") as f:
            card = json.load(f)

        X_test = pd.read_parquet(f"{tmpdir}/model/../../../X_test.parquet"
                                 if os.path.exists(f"{tmpdir}/model/../../../X_test.parquet")
                                 else _find_test_fixtures(tmpdir, mv.run_id))
        y_test = pd.read_parquet(_find_test_fixtures(tmpdir, mv.run_id, label=True))

    registered_auc = card["metrics"]["test_auc"]
    live_proba     = model.predict_proba(X_test)[:, 1]
    live_auc       = float(roc_auc_score(y_test["target"], live_proba))

    delta = abs(live_auc - registered_auc)
    if delta > 1e-12:
        raise AssertionError(
            f"Fidelity check failed: registered AUC={registered_auc}, "
            f"live AUC={live_auc}, delta={delta}"
        )

    print(f"Replay passed. AUC={live_auc} (delta={delta})")


def _find_test_fixtures(tmpdir: str, run_id: str, label: bool = False) -> str:
    """Download X_test / y_test parquet from MLflow run artifacts."""
    client = MlflowClient()
    filename = "y_test.parquet" if label else "X_test.parquet"
    local = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=filename, dst_path=tmpdir
    )
    return local
