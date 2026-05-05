"""
Streamlit dashboard — Drift Triage Co-Pilot.
Four panels:
  1. Registry   — MLflow model versions and stages
  2. Investigations — agent's open and resolved investigations
  3. Queue      — Redis job queue depth and DLQ contents
  4. HIL Inbox  — pending human approvals
"""

import os
import time

import mlflow
import redis
import requests
import streamlit as st

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "bank-marketing-classifier")
AGENT_URL           = os.getenv("AGENT_URL", "http://agent:8001")
REDIS_URL           = os.getenv("REDIS_URL", "redis://localhost:6379")
QUEUE_KEY           = "queue:jobs"
DLQ_KEY             = "queue:dlq"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

st.set_page_config(page_title="Drift Triage Co-Pilot", layout="wide")
st.title("Drift Triage Co-Pilot")

# Auto-refresh every 15 seconds
st.markdown(
    "<meta http-equiv='refresh' content='15'>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(["Registry", "Investigations", "Queue", "HIL Inbox"])


# ---------------------------------------------------------------------------
# Tab 1 — Registry
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("MLflow Model Registry")
    try:
        from mlflow import MlflowClient
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if versions:
            rows = [
                {
                    "Version":    v.version,
                    "Stage":      v.current_stage,
                    "Run ID":     v.run_id[:8] + "...",
                    "Created":    v.creation_timestamp,
                }
                for v in sorted(versions, key=lambda v: int(v.version), reverse=True)
            ]
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No model versions registered yet.")
    except Exception as e:
        st.error(f"Could not reach MLflow: {e}")


# ---------------------------------------------------------------------------
# Tab 2 — Investigations
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Agent Investigations")
    status_filter = st.selectbox("Filter by status", ["all", "open", "resolved", "rejected"])
    try:
        params = {} if status_filter == "all" else {"status": status_filter}
        resp = requests.get(f"{AGENT_URL}/investigations", params=params, timeout=5)
        resp.raise_for_status()
        investigations = resp.json().get("investigations", [])
        if investigations:
            st.dataframe(investigations, use_container_width=True)
        else:
            st.info("No investigations found.")
    except Exception as e:
        st.error(f"Could not reach agent: {e}")


# ---------------------------------------------------------------------------
# Tab 3 — Queue
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Redis Queue")
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)

        col1, col2 = st.columns(2)
        with col1:
            depth = r.llen(QUEUE_KEY)
            st.metric("Jobs in queue", depth)
        with col2:
            dlq_depth = r.llen(DLQ_KEY)
            st.metric("Dead-letter queue", dlq_depth, delta_color="inverse")

        if dlq_depth > 0:
            st.warning(f"{dlq_depth} job(s) in the DLQ — manual intervention may be needed.")
            dlq_items = r.lrange(DLQ_KEY, 0, 9)  # show up to 10
            st.write("**DLQ contents (latest 10):**")
            for item in dlq_items:
                st.code(item, language="json")
    except Exception as e:
        st.error(f"Could not reach Redis: {e}")


# ---------------------------------------------------------------------------
# Tab 4 — HIL Inbox
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Human-in-the-Loop Inbox")
    st.caption("These investigations are paused and waiting for your approval before touching Production.")

    try:
        resp = requests.get(f"{AGENT_URL}/pending-approvals", timeout=5)
        resp.raise_for_status()
        pending = resp.json().get("pending_approvals", [])

        if not pending:
            st.success("No pending approvals.")
        else:
            for item in pending:
                inv_id  = item["investigation_id"]
                action  = item["recommended_action"]
                reason  = item["reason"] or "No reason provided."

                with st.expander(f"Investigation {inv_id[:8]}...  |  Action: **{action}**"):
                    st.write(f"**Reason:** {reason}")
                    st.write(f"**Full ID:** `{inv_id}`")

                    col_approve, col_reject = st.columns(2)
                    with col_approve:
                        if st.button("Approve", key=f"approve_{inv_id}"):
                            r = requests.post(
                                f"{AGENT_URL}/approve/{inv_id}",
                                json={"approved": True},
                                timeout=10,
                            )
                            if r.status_code == 200:
                                st.success("Approved. Investigation resumed.")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Failed: {r.text}")

                    with col_reject:
                        if st.button("Reject", key=f"reject_{inv_id}"):
                            r = requests.post(
                                f"{AGENT_URL}/approve/{inv_id}",
                                json={"approved": False},
                                timeout=10,
                            )
                            if r.status_code == 200:
                                st.warning("Rejected. Investigation will close.")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Failed: {r.text}")

    except Exception as e:
        st.error(f"Could not reach agent: {e}")
