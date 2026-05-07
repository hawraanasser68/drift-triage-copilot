"""
End-to-end drift scenario test.

Steps:
  1. Promote the latest model version to Production (if not already there)
  2. Send 30 normal predictions  → establishes a clean baseline in the window
  3. Send 30 drifted predictions → pushes the distribution far from training
  4. Call /drift-report           → triggers real PSI/chi² calculation + webhook
  5. Print a summary so you know what to look for in the dashboard

Run from the project root:
    python test_drift_scenario.py
"""

import os
import sys
import time

import requests

MODEL_SERVICE = os.getenv("MODEL_SERVICE_URL", "http://model-service:8000")
AGENT_URL     = os.getenv("AGENT_URL",         "http://agent:8001")


# ── Data templates ─────────────────────────────────────────────────────────────

# "Normal" prediction — values close to training-set means
NORMAL = {
    "age": 40, "job": "admin.", "marital": "married",
    "education": "university.degree", "default": "no",
    "housing": "yes", "loan": "no", "contact": "cellular",
    "month": "may", "day_of_week": "thu",
    "campaign": 1, "pdays": 999, "previous": 0, "poutcome": "nonexistent",
    "emp_var_rate": -1.8, "cons_price_idx": 93.444,
    "cons_conf_idx": -36.1, "euribor3m": 1.313, "nr_employed": 5099.1,
}

# "Drifted" prediction — economic indicators at the opposite extreme,
# unusual demographics; causes high PSI on numeric features.
DRIFTED = {
    "age": 70, "job": "retired", "marital": "divorced",
    "education": "basic.4y", "default": "unknown",
    "housing": "no", "loan": "yes", "contact": "telephone",
    "month": "dec", "day_of_week": "mon",
    "campaign": 20, "pdays": 999, "previous": 0, "poutcome": "failure",
    "emp_var_rate": 1.4,  "cons_price_idx": 94.767,
    "cons_conf_idx": -26.9, "euribor3m": 4.961, "nr_employed": 5228.1,
}


def check_services():
    for name, url in [("Model service", MODEL_SERVICE + "/docs"),
                      ("Agent",         AGENT_URL     + "/docs")]:
        try:
            r = requests.get(url, timeout=3)
            print(f"  {name}: OK ({r.status_code})")
        except Exception as e:
            print(f"  {name}: UNREACHABLE — {e}")
            sys.exit(1)


def promote_to_production():
    from mlflow import MlflowClient
    import mlflow, os
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = MlflowClient()

    # Already in Production?
    prod = client.get_latest_versions("bank-marketing-classifier", stages=["Production"])
    if prod:
        print(f"  Model already in Production: v{prod[0].version} — skipping promotion.")
        return

    # Find the latest version and promote it
    all_v = client.search_model_versions("name='bank-marketing-classifier'")
    if not all_v:
        print("  No model versions found. Run the trainer first.")
        sys.exit(1)

    latest = max(all_v, key=lambda v: int(v.version))
    resp = requests.post(f"{MODEL_SERVICE}/promote/{latest.version}", timeout=30)
    if resp.ok:
        print(f"  Promoted v{latest.version} to Production.")
    else:
        print(f"  Promotion failed: {resp.text}")
        sys.exit(1)


def send_predictions(template: dict, count: int, label: str):
    ok = fail = 0
    for _ in range(count):
        try:
            r = requests.post(f"{MODEL_SERVICE}/predict", json=template, timeout=5)
            if r.ok:
                ok += 1
            else:
                fail += 1
        except Exception:
            fail += 1
    print(f"  {label}: {ok} sent, {fail} failed")


def trigger_drift_report():
    try:
        r = requests.get(f"{MODEL_SERVICE}/drift-report", timeout=15)
        if r.ok:
            report = r.json()
            print(f"  Severity : {report.get('severity','?').upper()}")
            print(f"  Output drift : {report.get('output_drift', 0):.4f}")
            print(f"  Window size  : {report.get('window_size','?')}")

            drifted_features = [
                f["feature"] for f in
                report.get("numeric_drift", []) + report.get("categorical_drift", [])
                if f.get("drifted")
            ]
            if drifted_features:
                print(f"  Drifted features: {', '.join(drifted_features)}")
            else:
                print("  No individual features above threshold yet.")
            return report.get("severity", "none")
        else:
            # 400 = not enough predictions yet
            print(f"  Drift report error: {r.status_code} — {r.text[:120]}")
            return "none"
    except Exception as e:
        print(f"  Could not reach drift report: {e}")
        return "none"


def check_pending():
    try:
        r = requests.get(f"{AGENT_URL}/pending-approvals", timeout=5)
        items = r.json().get("pending_approvals", []) if r.ok else []
        return items
    except Exception:
        return []


# ── Main ───────────────────────────────────────────────────────────────────────

print("\n=== Drift Scenario Test ===\n")

print("[1/5] Checking services...")
check_services()

print("\n[2/5] Promoting model to Production...")
promote_to_production()

print("\n[3/5] Sending 30 normal predictions (baseline)...")
send_predictions(NORMAL, 30, "Normal")

print("\n[4/5] Sending 30 drifted predictions (shifted distribution)...")
send_predictions(DRIFTED, 30, "Drifted")

print("\n[5/5] Triggering drift report (real PSI / chi² calculation)...")
severity = trigger_drift_report()

print("\n=== Result ===")
if severity == "none":
    print(
        "Severity is NONE — the 60 predictions were not enough to push past the threshold.\n"
        "Try running the script a second time to add more drifted predictions to the window."
    )
elif severity == "warning":
    print(
        "Severity is WARNING — drift is detected but not critical.\n"
        "Run the script again to keep adding drifted data until severity becomes CRITICAL,\n"
        "or check the Triage Queue in the dashboard to see the open investigation."
    )
else:
    print("Severity is CRITICAL — the webhook has been sent to the agent.")
    print("Waiting 4 s for the agent to finish triage and pause at HIL...")
    time.sleep(4)

    pending = check_pending()
    if pending:
        print(f"\nHIL inbox has {len(pending)} pending approval(s).")
        print("Open the dashboard at http://localhost:8501 → HIL Inbox to review and approve.")
    else:
        print(
            "\nNo pending approvals yet — the agent may still be processing.\n"
            "Open the dashboard → HIL Inbox in a few seconds."
        )

print()
