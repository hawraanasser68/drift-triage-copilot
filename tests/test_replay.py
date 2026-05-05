"""
1e-12 fidelity replay test.
Loads the trained pipeline and test fixtures from the local artifacts/ folder,
re-scores X_test, and asserts the AUC matches the registered value within 1e-12.

In CI this runs after train.py has been executed in a prior step.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

ARTIFACTS = Path("artifacts")


def _artifacts_exist() -> bool:
    return (
        (ARTIFACTS / "pipeline.pkl").exists()
        and (ARTIFACTS / "model_card.json").exists()
        and (ARTIFACTS / "X_test.parquet").exists()
        and (ARTIFACTS / "y_test.parquet").exists()
    )


@pytest.mark.skipif(
    not _artifacts_exist(),
    reason="artifacts/ not found — run train.py first",
)
def test_replay_fidelity():
    # Load pipeline
    with open(ARTIFACTS / "pipeline.pkl", "rb") as f:
        model = pickle.load(f)

    # Load registered metrics
    with open(ARTIFACTS / "model_card.json") as f:
        card = json.load(f)
    registered_auc = card["metrics"]["test_auc"]

    # Load test fixtures
    X_test = pd.read_parquet(ARTIFACTS / "X_test.parquet")
    y_test = pd.read_parquet(ARTIFACTS / "y_test.parquet")

    # Re-score
    live_proba = model.predict_proba(X_test)[:, 1]
    live_auc   = float(roc_auc_score(y_test["target"], live_proba))

    delta = abs(live_auc - registered_auc)

    assert delta <= 1e-12, (
        f"Fidelity check FAILED.\n"
        f"  registered AUC : {registered_auc}\n"
        f"  live AUC       : {live_auc}\n"
        f"  delta          : {delta}  (allowed: 1e-12)"
    )
