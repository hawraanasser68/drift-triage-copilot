"""
Train a GradientBoosting classifier on the UCI Bank Marketing dataset,
tune the operating threshold (recall >= 0.75), and register the fitted
pipeline in MLflow with the standard artifact triple.
"""

import hashlib
import json
import os
import platform
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = os.getenv("DATA_PATH", "bank-additional-full.csv")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "bank-marketing-classifier"
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ---------------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")

    # Encode target
    df["target"] = df["y"].map({"yes": 1, "no": 0})

    # pdays==999 means never contacted — preserve that signal as a flag
    df["no_previous_contact"] = (df["pdays"] == 999).astype(int)
    df["pdays_clean"] = df["pdays"].where(df["pdays"] != 999, -1)

    # Drop leakage (duration) and original columns we replaced
    df = df.drop(columns=["y", "duration", "pdays"])

    # Drop duplicates
    df = df.drop_duplicates()

    return df


# ---------------------------------------------------------------------------
# 2. Build preprocessor
# ---------------------------------------------------------------------------
def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    binary_cols = [
        col for col in X_train.columns
        if X_train[col].dropna().nunique() <= 2
        and set(X_train[col].dropna().unique()).issubset({0, 1})
    ]
    numeric_cols = [
        col for col in X_train.select_dtypes(include="number").columns
        if col not in binary_cols
    ]
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    binary_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("bin", binary_transformer, binary_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]), numeric_cols, binary_cols, categorical_cols


# ---------------------------------------------------------------------------
# 3. Tune threshold on validation set (highest threshold with recall >= 0.75)
# ---------------------------------------------------------------------------
def tune_threshold(proba: np.ndarray, y_true: pd.Series, min_recall: float = 0.75) -> float:
    best_threshold = 0.5
    for t in np.arange(0.99, 0.00, -0.01):
        preds = (proba >= t).astype(int)
        if recall_score(y_true, preds, zero_division=0) >= min_recall:
            best_threshold = round(float(t), 2)
            break
    return best_threshold


# ---------------------------------------------------------------------------
# 4. Main training run
# ---------------------------------------------------------------------------
def train():
    # Load data
    df = load_and_clean(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]

    # Stratified 60/20/20 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Build preprocessor + classifier pipeline
    preprocessor, numeric_cols, binary_cols, categorical_cols = build_preprocessor(X_train)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    # Tune threshold on validation set
    val_proba = model.predict_proba(X_val)[:, 1]
    operating_threshold = tune_threshold(val_proba, y_val)

    # Evaluate on test set
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= operating_threshold).astype(int)

    metrics = {
        "test_auc": float(roc_auc_score(y_test, test_proba)),
        "test_avg_precision": float(average_precision_score(y_test, test_proba)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "operating_threshold": operating_threshold,
    }

    print(json.dumps(metrics, indent=2))

    # Save test fixtures for the 1e-12 fidelity replay test later
    X_test.to_parquet(ARTIFACTS_DIR / "X_test.parquet", index=False)
    y_test.to_frame().to_parquet(ARTIFACTS_DIR / "y_test.parquet", index=False)

    # Save reference statistics from the training set for drift detection
    reference_stats = {
        "numeric": {
            col: {"mean": float(X_train[col].mean()), "std": float(X_train[col].std()),
                  "values": X_train[col].tolist()}
            for col in numeric_cols
        },
        "categorical": {
            col: X_train[col].tolist()
            for col in categorical_cols
        },
        "scores": model.predict_proba(X_train)[:, 1].tolist(),
    }
    ref_stats_path = ARTIFACTS_DIR / "reference_stats.json"
    ref_stats_path.write_text(json.dumps(reference_stats))

    # ------------------------------------------------------------------
    # Artifact triple
    # ------------------------------------------------------------------

    # 1. Pipeline binary
    pipeline_path = ARTIFACTS_DIR / "pipeline.pkl"
    with open(pipeline_path, "wb") as f:
        pickle.dump(model, f)

    # 2. Schema — feature names and dtypes
    schema = {
        "features": {col: str(dtype) for col, dtype in X_train.dtypes.items()},
        "numeric_cols": numeric_cols,
        "binary_cols": binary_cols,
        "categorical_cols": categorical_cols,
    }
    schema_path = ARTIFACTS_DIR / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=2))

    # 3. Model card — dataset hash, environment, metrics, threshold
    dataset_hash = hashlib.md5(Path(DATA_PATH).read_bytes()).hexdigest()
    model_card = {
        "model_name": MODEL_NAME,
        "dataset_hash": dataset_hash,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "operating_threshold": operating_threshold,
        "threshold_rule": "highest threshold where recall >= 0.75 on validation set",
        "metrics": metrics,
    }
    model_card_path = ARTIFACTS_DIR / "model_card.json"
    model_card_path.write_text(json.dumps(model_card, indent=2))

    # ------------------------------------------------------------------
    # Register in MLflow
    # ------------------------------------------------------------------
    mlflow.set_experiment("bank-marketing")

    with mlflow.start_run(run_name=MODEL_NAME) as run:
        mlflow.log_params({
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 3,
            "operating_threshold": operating_threshold,
        })
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "model_type":   "GradientBoostingClassifier",
            "dataset":      "UCI Bank Marketing",
            "dataset_hash": model_card["dataset_hash"],
        })

        # Log model with the native sklearn flavour so MLflow links the run to
        # the registered model name in the Experiments view (fixes the "—" dash).
        signature = mlflow.models.infer_signature(
            X_train.head(5),
            model.predict_proba(X_train.head(5))[:, 1],
        )
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn_model",
            signature=signature,
            input_example=X_train.head(3),
        )

        # Keep supplementary artifacts at model/ for model-service / replay compat
        mlflow.log_artifact(str(pipeline_path),   artifact_path="model")
        mlflow.log_artifact(str(schema_path),     artifact_path="model")
        mlflow.log_artifact(str(model_card_path), artifact_path="model")
        mlflow.log_artifact(str(ref_stats_path),  artifact_path="model")
        mlflow.log_artifact(str(ARTIFACTS_DIR / "X_test.parquet"), artifact_path="model")
        mlflow.log_artifact(str(ARTIFACTS_DIR / "y_test.parquet"), artifact_path="model")

        # Register via the sklearn model URI (not the raw pickle path)
        mv = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/sklearn_model",
            name=MODEL_NAME,
        )

    # Annotate the registry entry with description and per-metric tags so the
    # model card is visible directly in the MLflow Model Registry UI.
    client = MlflowClient()
    client.update_registered_model(
        name=MODEL_NAME,
        description=(
            "GradientBoostingClassifier trained on UCI Bank Marketing dataset. "
            "Binary classification: predicts term deposit subscription. "
            "Threshold tuned for recall ≥ 0.75 on validation set."
        ),
    )
    client.update_model_version(
        name=MODEL_NAME,
        version=mv.version,
        description=json.dumps(model_card, indent=2),
    )
    for tag_key, tag_val in {
        "test_auc":            f"{metrics['test_auc']:.6f}",
        "test_recall":         f"{metrics['test_recall']:.6f}",
        "test_precision":      f"{metrics['test_precision']:.6f}",
        "test_f1":             f"{metrics['test_f1']:.6f}",
        "operating_threshold": str(operating_threshold),
        "dataset_hash":        model_card["dataset_hash"],
        "sklearn_version":     model_card["sklearn_version"],
        "python_version":      model_card["python_version"],
    }.items():
        client.set_model_version_tag(MODEL_NAME, mv.version, tag_key, tag_val)

    print(f"\nRegistered: {MODEL_NAME} v{mv.version}  (run_id={run.info.run_id})")
    print(f"Test AUC: {metrics['test_auc']:.4f} | Recall: {metrics['test_recall']:.4f}")


if __name__ == "__main__":
    train()
