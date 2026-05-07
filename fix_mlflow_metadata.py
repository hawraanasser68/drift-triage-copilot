"""
One-shot script: backfill description + metric tags on every existing
model version for MODEL_NAME.  Run this once from inside any container
that can reach the MLflow server, or from the host with MLflow reachable.

Usage (from host):
    MLFLOW_TRACKING_URI=http://localhost:5001 python fix_mlflow_metadata.py
"""

import json
import os
import tempfile

import mlflow
from mlflow import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = os.getenv("MODEL_NAME", "bank-marketing-classifier")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Update top-level registered model description
client.update_registered_model(
    name=MODEL_NAME,
    description=(
        "GradientBoostingClassifier trained on UCI Bank Marketing dataset. "
        "Binary classification: predicts term deposit subscription. "
        "Threshold tuned for recall ≥ 0.75 on validation set."
    ),
)
print(f"Updated registered model description for '{MODEL_NAME}'")

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
for mv in versions:
    run_id = mv.run_id
    print(f"\nProcessing v{mv.version} (run_id={run_id[:8]}…)")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="model/model_card.json",
                dst_path=tmpdir,
            )
            with open(f"{tmpdir}/model/model_card.json") as f:
                card = json.load(f)
    except Exception as exc:
        print(f"  Could not download model_card.json: {exc} — skipping")
        continue

    metrics = card.get("metrics", {})

    client.update_model_version(
        name=MODEL_NAME,
        version=mv.version,
        description=json.dumps(card, indent=2),
    )

    tags = {
        "test_auc":            f"{metrics.get('test_auc', 0):.6f}",
        "test_recall":         f"{metrics.get('test_recall', 0):.6f}",
        "test_precision":      f"{metrics.get('test_precision', 0):.6f}",
        "test_f1":             f"{metrics.get('test_f1', 0):.6f}",
        "operating_threshold": str(card.get("operating_threshold", "")),
        "dataset_hash":        card.get("dataset_hash", ""),
        "sklearn_version":     card.get("sklearn_version", ""),
        "python_version":      card.get("python_version", ""),
    }
    for k, v in tags.items():
        client.set_model_version_tag(MODEL_NAME, mv.version, k, v)

    print(f"  Set description + {len(tags)} tags on v{mv.version}")

print("\nDone.")
