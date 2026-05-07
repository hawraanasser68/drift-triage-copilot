"""
Full end-to-end test for all drift scenarios.

Cases covered:
  1. WARNING  — mild drift → agent dispatches replay_test automatically (no HIL)
  2. CRITICAL — heavy drift → agent pauses for HIL → human APPROVES
  3. CRITICAL — heavy drift → agent pauses for HIL → human REJECTS

Run inside the worker container:
    docker-compose exec worker python3 test_all_cases.py
"""

import json
import os
import time

import psycopg2
import requests

MODEL_SERVICE = os.getenv("MODEL_SERVICE_URL", "http://model-service:8000")
AGENT_URL     = os.getenv("AGENT_URL",         "http://agent:8001")
DATABASE_URL  = os.getenv("DATABASE_URL",       "postgresql://user:password@postgres:5432/mlops")

# ── Prediction templates ────────────────────────────────────────────────────

NORMAL = {
    "age": 40, "job": "admin.", "marital": "married",
    "education": "university.degree", "default": "no",
    "housing": "yes", "loan": "no", "contact": "cellular",
    "month": "may", "day_of_week": "thu",
    "campaign": 1, "pdays": 999, "previous": 0, "poutcome": "nonexistent",
    "emp_var_rate": -1.8, "cons_price_idx": 93.444,
    "cons_conf_idx": -36.1, "euribor3m": 1.313, "nr_employed": 5099.1,
}

MILD = {
    "age": 52, "job": "technician", "marital": "single",
    "education": "high.school", "default": "no",
    "housing": "no", "loan": "no", "contact": "cellular",
    "month": "jun", "day_of_week": "mon",
    "campaign": 3, "pdays": 999, "previous": 0, "poutcome": "nonexistent",
    "emp_var_rate": -0.5, "cons_price_idx": 93.9,
    "cons_conf_idx": -40.0, "euribor3m": 2.1, "nr_employed": 5150.0,
}

DRIFTED = {
    "age": 70, "job": "retired", "marital": "divorced",
    "education": "basic.4y", "default": "unknown",
    "housing": "no", "loan": "yes", "contact": "telephone",
    "month": "dec", "day_of_week": "mon",
    "campaign": 20, "pdays": 999, "previous": 0, "poutcome": "failure",
    "emp_var_rate": 1.4, "cons_price_idx": 94.767,
    "cons_conf_idx": -26.9, "euribor3m": 4.961, "nr_employed": 5228.1,
}

# ── Helpers ─────────────────────────────────────────────────────────────────

def header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print('='*60)

def step(text):
    print(f"\n  >> {text}")

def ok(text):
    print(f"     OK  {text}")

def fail(text):
    print(f"     FAIL  {text}")

def reset_state():
    conn = psycopg2.connect(DATABASE_URL)
    with conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE drift_state SET severity = 'none' WHERE id = 1")
            cur.execute("DELETE FROM predictions")
            cur.execute("DELETE FROM pending_approvals")
    conn.close()

def send_predictions(template, count):
    ok_count = 0
    for _ in range(count):
        try:
            r = requests.post(f"{MODEL_SERVICE}/predict", json=template, timeout=5)
            if r.ok:
                ok_count += 1
        except Exception:
            pass
    print(f"     Sent {ok_count}/{count} predictions")

def trigger_drift_report():
    r = requests.get(f"{MODEL_SERVICE}/drift-report", timeout=30)
    if not r.ok:
        fail(f"drift-report returned {r.status_code}: {r.text[:100]}")
        return None
    report = r.json()
    sev = report.get("severity", "none")
    print(f"     Severity : {sev.upper()}")
    print(f"     Output drift : {report.get('output_drift', 0):.4f}")
    drifted = [f['feature'] for f in report.get('numeric_drift', []) + report.get('categorical_drift', []) if f.get('drifted')]
    if drifted:
        print(f"     Drifted features : {', '.join(drifted)}")
    return sev

def get_pending():
    try:
        r = requests.get(f"{AGENT_URL}/pending-approvals", timeout=5)
        return r.json().get("pending_approvals", []) if r.ok else []
    except Exception:
        return []

def get_investigations():
    try:
        r = requests.get(f"{AGENT_URL}/investigations", timeout=5)
        return r.json().get("investigations", []) if r.ok else []
    except Exception:
        return []

def wait_for_pending(timeout=20):
    for _ in range(timeout):
        pending = get_pending()
        if pending:
            return pending
        time.sleep(1)
    return []

def wait_for_state(expected, timeout=30):
    for _ in range(timeout):
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("SELECT severity FROM drift_state WHERE id=1")
            state = cur.fetchone()[0]
        conn.close()
        if state == expected:
            return True
        time.sleep(1)
    return False

def approve(investigation_id, approved: bool):
    r = requests.post(
        f"{AGENT_URL}/approve/{investigation_id}",
        json={"approved": approved},
        timeout=10,
    )
    return r.ok

# ── TEST 1: WARNING ──────────────────────────────────────────────────────────

header("TEST 1 — WARNING (replay_test, no HIL required)")

step("Resetting state and clearing predictions...")
reset_state()
ok("Clean slate")

step("Sending 40 normal + 10 mild-drift predictions...")
send_predictions(NORMAL, 40)
send_predictions(MILD, 10)

step("Triggering drift report...")
severity = trigger_drift_report()

if severity == "warning":
    ok("Severity is WARNING as expected")
elif severity == "critical":
    ok("Severity is CRITICAL (mild drift pushed past threshold — acceptable)")
elif severity == "none":
    fail("Severity is NONE — not enough drift. The test may need more drifted predictions.")
else:
    fail(f"Unexpected severity: {severity}")

if severity in ("warning", "critical"):
    step("Waiting for agent to create investigation (up to 10s)...")
    time.sleep(5)
    invs = get_investigations()
    if invs:
        latest = sorted(invs, key=lambda i: i.get("created_at", ""), reverse=True)[0]
        ok(f"Investigation created: action={latest.get('recommended_action')} status={latest.get('status')}")
    else:
        fail("No investigation found after 10s")

    step("Verifying NO pending HIL approval (replay_test skips HIL)...")
    pending = get_pending()
    if not pending:
        ok("No HIL approval required — correct for replay_test")
    else:
        fail(f"Unexpected HIL approval found: {pending[0].get('recommended_action')}")

    step("Waiting for drift state to reset to 'none' (replay_test resets it)...")
    if wait_for_state("none", timeout=40):
        ok("Drift state reset to 'none' automatically")
    else:
        fail("Drift state did not reset within 40s")

# ── TEST 2: CRITICAL + APPROVE ───────────────────────────────────────────────

header("TEST 2 — CRITICAL + HUMAN APPROVES")

step("Resetting state and clearing predictions...")
reset_state()
ok("Clean slate")

step("Sending 20 normal + 40 heavily-drifted predictions...")
send_predictions(NORMAL, 20)
send_predictions(DRIFTED, 40)

step("Triggering drift report...")
severity = trigger_drift_report()

if severity == "critical":
    ok("Severity is CRITICAL as expected")
else:
    fail(f"Expected CRITICAL, got {severity} — results below may be unreliable")

step("Waiting for HIL approval to appear (up to 20s)...")
pending = wait_for_pending(timeout=20)

if pending:
    item = pending[0]
    inv_id = item["investigation_id"]
    ok(f"HIL approval pending: investigation={inv_id[:12]}...")
    ok(f"Recommended action : {item.get('recommended_action')}")
    ok(f"Reason             : {item.get('reason', '')[:80]}")

    step("Approving the investigation...")
    if approve(inv_id, approved=True):
        ok("Approval sent successfully")
    else:
        fail("Approval request failed")

    step("Waiting for drift state to reset to 'none' (job resets it)...")
    if wait_for_state("none", timeout=60):
        ok("Drift state reset to 'none' — job completed successfully")
    else:
        fail("Drift state did not reset within 60s (job may still be running)")
else:
    fail("No HIL approval appeared within 20s")

# ── TEST 3: CRITICAL + REJECT ────────────────────────────────────────────────

header("TEST 3 — CRITICAL + HUMAN REJECTS")

step("Resetting state and clearing predictions...")
reset_state()
ok("Clean slate")

step("Sending 20 normal + 40 heavily-drifted predictions...")
send_predictions(NORMAL, 20)
send_predictions(DRIFTED, 40)

step("Triggering drift report...")
severity = trigger_drift_report()

if severity == "critical":
    ok("Severity is CRITICAL as expected")
else:
    fail(f"Expected CRITICAL, got {severity}")

step("Waiting for HIL approval to appear (up to 20s)...")
pending = wait_for_pending(timeout=20)

if pending:
    item = pending[0]
    inv_id = item["investigation_id"]
    ok(f"HIL approval pending: investigation={inv_id[:12]}...")

    step("Rejecting the investigation...")
    if approve(inv_id, approved=False):
        ok("Rejection sent successfully")
    else:
        fail("Rejection request failed")

    step("Verifying drift state reset to 'none' after rejection...")
    if wait_for_state("none", timeout=10):
        ok("Drift state reset to 'none' — rejection handled correctly")
    else:
        fail("Drift state did not reset after rejection")

    step("Verifying investigation is marked as rejected...")
    time.sleep(2)
    invs = get_investigations()
    rejected = [i for i in invs if i.get("investigation_id") == inv_id]
    if rejected and rejected[0].get("status") == "rejected":
        ok("Investigation status = rejected")
    else:
        status = rejected[0].get("status") if rejected else "not found"
        fail(f"Investigation status = {status} (expected 'rejected')")
else:
    fail("No HIL approval appeared within 20s")

# ── SUMMARY ──────────────────────────────────────────────────────────────────

header("ALL TESTS COMPLETE")
print("  Check the dashboard at http://localhost:8501 to see all investigations.\n")
