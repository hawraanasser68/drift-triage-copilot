"""
Streamlit dashboard — Drift Triage Co-Pilot.
Performance-optimised: data loaded lazily per-page, all calls cached.
"""

import json
import os
import time
from collections import Counter
from datetime import datetime, timezone

import mlflow
import plotly.express as px
import plotly.graph_objects as go
import redis
import requests
import streamlit as st
from mlflow import MlflowClient

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME          = os.getenv("MODEL_NAME",           "bank-marketing-classifier")
AGENT_URL           = os.getenv("AGENT_URL",            "http://agent:8001")
REDIS_URL           = os.getenv("REDIS_URL",            "redis://localhost:6379")
MODEL_SERVICE_URL   = os.getenv("MODEL_SERVICE_URL",    "http://model-service:8000")
QUEUE_KEY = "queue:jobs"
DLQ_KEY   = "queue:dlq"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

st.set_page_config(
    page_title="Drift Triage Co-Pilot",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stAppViewContainer"] > .main { padding-top: 0.5rem; }

[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1623 0%, #1a2236 100%);
    border-right: 1px solid #1e2d45;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #c8d6e5 !important; }
[data-testid="stSidebar"] hr   { border-color: #1e2d45 !important; }

.kpi-card {
    background: #ffffff;
    border: 1px solid #e8edf2;
    border-top: 3px solid #6366f1;
    border-radius: 10px;
    padding: 16px 20px 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    margin-bottom: 8px;
}
.kpi-card.danger  { border-top-color: #ef4444; }
.kpi-card.warning { border-top-color: #f59e0b; }
.kpi-card.success { border-top-color: #10b981; }
.kpi-card.info    { border-top-color: #6366f1; }
.kpi-label { font-size: 11px; font-weight: 600; color: #8896a5;
             text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 4px; }
.kpi-value { font-size: 26px; font-weight: 700; color: #1a2236; line-height: 1.2; }
.kpi-sub   { font-size: 11px; color: #8896a5; margin-top: 4px; }

.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.badge.critical { background: #fee2e2; color: #b91c1c; }
.badge.warning  { background: #fef3c7; color: #92400e; }
.badge.none, .badge.healthy { background: #d1fae5; color: #065f46; }
.badge.resolved { background: #dbeafe; color: #1e40af; }
.badge.rejected { background: #f1f5f9; color: #475569; }
.badge.open     { background: #fff7ed; color: #c2410c; }

.triage-card {
    border: 1px solid #e8edf2; border-left: 4px solid #e8edf2;
    border-radius: 0 8px 8px 0; padding: 14px 16px 10px;
    margin-bottom: 2px; background: #ffffff;
}
.triage-card.critical { border-left-color: #ef4444; }
.triage-card.warning  { border-left-color: #f59e0b; }
.triage-card.none     { border-left-color: #10b981; }
.triage-card.selected { background: #f8faff; border-left-color: #6366f1; }
.tc-header { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.tc-action { font-family: monospace; font-size: 12px; background: #f1f5f9;
             color: #334155; padding: 2px 8px; border-radius: 4px; }
.tc-meta   { font-size: 11px; color: #94a3b8; }

.detail-header { padding: 16px 0 12px; border-bottom: 1px solid #e8edf2; margin-bottom: 16px; }
.detail-title  { font-size: 20px; font-weight: 700; color: #1a2236; }
.detail-meta   { font-size: 12px; color: #8896a5; margin-top: 4px; }

.ai-box {
    background: linear-gradient(135deg, #f0f4ff 0%, #eef2ff 100%);
    border: 1px solid #c7d2fe; border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0; padding: 10px 14px;
    font-size: 13px; color: #3730a3; margin: 10px 0; line-height: 1.5;
}

.section-label {
    font-size: 11px; font-weight: 600; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.8px; margin: 20px 0 8px;
}

.alert-row {
    display: grid;
    grid-template-columns: 110px 90px 1fr 90px 80px;
    gap: 8px; align-items: center;
    padding: 9px 12px; border-bottom: 1px solid #f1f5f9;
    font-size: 12px; color: #334155; min-width: 0;
}
.alert-row.header {
    font-size: 10px; font-weight: 600; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.6px;
    background: #f8fafc; border-radius: 8px 8px 0 0;
}
.alert-container {
    border: 1px solid #e8edf2; border-radius: 8px;
    overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    width: 100%;
}
.summary-text {
    white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; min-width: 0;
}
.evt-tag {
    display: inline-block; font-size: 10px; font-weight: 700;
    padding: 2px 7px; border-radius: 4px; letter-spacing: 0.5px;
    font-family: monospace; background: #f1f5f9; color: #334155;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
IMPACT_MAP = {
    "retrain":     "A new model version will be trained and promoted to Production.",
    "rollback":    "Production will revert to the previous model version.",
    "replay_test": "AUC fidelity check runs against the registered test set.",
    "monitor":     "No Production changes — metrics tracked only.",
}
SEV_COLOR = {"critical": "#ef4444", "warning": "#f59e0b", "none": "#10b981"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _badge(s: str) -> str:
    s = s.lower()
    return f'<span class="badge {s}">● {s.upper()}</span>'

def _time_ago(iso_ts: str) -> str:
    try:
        ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        mins = int((datetime.now(timezone.utc) - ts).total_seconds() // 60)
        if mins < 1:  return "just now"
        if mins < 60: return f"{mins}m ago"
        return f"{mins // 60}h {mins % 60}m ago"
    except Exception:
        return "—"

def _kpi(col, label, value, sub="", style="info"):
    with col:
        st.markdown(f"""
        <div class="kpi-card {style}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

def _ai(text: str):
    st.markdown(f'<div class="ai-box">{text}</div>', unsafe_allow_html=True)

# ── Cached fetchers — TTL 60 s, called only on the pages that need them ───────
@st.cache_data(ttl=60)
def fetch_investigations(status=None):
    try:
        params = {} if not status else {"status": status}
        r = requests.get(f"{AGENT_URL}/investigations", params=params, timeout=5)
        return r.json().get("investigations", []) if r.ok else []
    except Exception:
        return []

@st.cache_data(ttl=60)
def fetch_pending():
    try:
        r = requests.get(f"{AGENT_URL}/pending-approvals", timeout=5)
        return r.json().get("pending_approvals", []) if r.ok else []
    except Exception:
        return []

@st.cache_data(ttl=60)
def fetch_drift():
    try:
        r = requests.get(f"{MODEL_SERVICE_URL}/drift-report", timeout=5)
        return r.json() if r.ok else {}
    except Exception:
        return {}

@st.cache_data(ttl=120)
def fetch_versions():
    try:
        return MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    except Exception:
        return []

@st.cache_data(ttl=120)
def fetch_prod_version():
    try:
        vs = MlflowClient().get_latest_versions(MODEL_NAME, stages=["Production"])
        return f"v{vs[0].version}" if vs else "—"
    except Exception:
        return "—"

def fetch_live(url: str, key: str) -> list:
    """Uncached fetch used by the Live Monitor page."""
    try:
        r = requests.get(url, timeout=3)
        return r.json().get(key, []) if r.ok else []
    except Exception:
        return []

@st.cache_data(ttl=60)
def fetch_queue():
    try:
        rc = redis.from_url(REDIS_URL, decode_responses=True)
        return {
            "depth":     rc.llen(QUEUE_KEY),
            "dlq":       rc.llen(DLQ_KEY),
            "dlq_items": rc.lrange(DLQ_KEY, 0, 9),
            "processed": rc.scard("queue:processed_keys"),
        }
    except Exception:
        return {"depth": 0, "dlq": 0, "dlq_items": [], "processed": 0}

# ── Sidebar — navigation only, no API calls ───────────────────────────────────
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## Drift Co-Pilot")
    st.caption("Bank Marketing Classifier")
    st.markdown("---")

    page = st.radio(
        "nav",
        options=[
            "Overview",
            "Triage Queue",
            "Job Queue",
            "Registry",
            "HIL Inbox",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ════════════════════════════════════════════════════════════════════════
# OVERVIEW  — fetches: investigations + queue + prod_version
# ════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("# Overview")
    st.caption("Real-time health of the bank marketing classifier pipeline.")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI cards ─────────────────────────────────────────────────────────
    with st.spinner("Loading…"):
        all_inv      = fetch_investigations()
        queue        = fetch_queue()
        prod_version = fetch_prod_version()

    n_critical = sum(1 for i in all_inv if i.get("severity") == "critical")
    n_open     = sum(1 for i in all_inv if i.get("status")   == "open")
    n_pending  = len(fetch_pending())

    k = st.columns(5)
    _kpi(k[0], "Production Model",     prod_version,      "Currently serving",                      "info")
    _kpi(k[1], "Total Investigations", len(all_inv),       f"{n_open} open",                         "info")
    _kpi(k[2], "Critical Alerts",      n_critical,
         "Needs attention" if n_critical else "All clear",
         "danger" if n_critical else "success")
    _kpi(k[3], "Pending Approvals",    n_pending,
         "Awaiting review" if n_pending else "Inbox clear",
         "warning" if n_pending else "success")
    _kpi(k[4], "Jobs Processed",       queue["processed"], "Total completed",                        "success")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="section-label">Investigations by severity</div>', unsafe_allow_html=True)
        if all_inv:
            sev_counts = Counter(i.get("severity", "unknown") for i in all_inv)
            fig = px.bar(
                x=list(sev_counts.keys()), y=list(sev_counts.values()),
                color=list(sev_counts.keys()),
                color_discrete_map=SEV_COLOR,
            )
            fig.update_layout(
                showlegend=False, height=220,
                margin=dict(t=0, b=0, l=0, r=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="", showgrid=False),
                yaxis=dict(title="Count", showgrid=True, gridcolor="#f1f5f9"),
                font=dict(size=12),
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No investigation data yet.")

    with col_r:
        st.markdown('<div class="section-label">By recommended action</div>', unsafe_allow_html=True)
        if all_inv:
            ac = Counter(i.get("recommended_action", "unknown") for i in all_inv)
            fig2 = go.Figure(go.Pie(
                labels=list(ac.keys()), values=list(ac.values()),
                hole=0.6,
                marker=dict(colors=["#6366f1","#f59e0b","#10b981","#ef4444"],
                            line=dict(color="white", width=2)),
                textinfo="percent+label", textfont_size=11,
            ))
            fig2.update_layout(height=220, margin=dict(t=0,b=0,l=0,r=0),
                               paper_bgcolor="white", showlegend=False, font=dict(size=11))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data yet.")

    st.divider()

    # ── Pipeline cards ────────────────────────────────────────────────────
    st.markdown("#### Investigation Pipeline")
    STEP_LABELS = ["Started", "Triage", "HIL Check", "Dispatched", "Resolved"]

    def _current_step(inv: dict) -> int:
        if inv.get("status") in ("resolved", "rejected"): return 5
        if inv.get("job_id"):                             return 4
        if inv.get("recommended_action"):                 return 2
        return 1

    shown_invs = sorted(all_inv, key=lambda i: i.get("created_at", ""), reverse=True)[:6]

    if not shown_invs:
        st.info("No investigations yet. Send a drift event to begin.")
    else:
        for inv in shown_invs:
            inv_id     = inv.get("investigation_id", "")
            sev        = inv.get("severity", "unknown")
            action     = inv.get("recommended_action", "—")
            status     = inv.get("status", "open")
            step       = _current_step(inv)
            step_label = STEP_LABELS[min(step, 5) - 1]

            with st.container(border=True):
                st.markdown(
                    f"**{sev.upper()}** &nbsp;·&nbsp; `{action}` &nbsp;·&nbsp; "
                    f"`{inv_id[:10]}…` &nbsp;·&nbsp; status: **`{status}`**",
                    unsafe_allow_html=True,
                )
                st.progress(step / 5, text=f"Step {step} / 5 — {step_label}")


# ════════════════════════════════════════════════════════════════════════
# TRIAGE QUEUE  — fetches: investigations
# ════════════════════════════════════════════════════════════════════════
elif page == "Triage Queue":
    st.markdown("# Triage Queue")

    if "sel" not in st.session_state:
        st.session_state.sel = None

    with st.spinner("Loading investigations…"):
        all_inv = fetch_investigations()

    col_list, col_detail = st.columns([5, 7], gap="large")

    with col_list:
        fc1, fc2 = st.columns(2)
        sev_f    = fc1.selectbox("Filter by severity", ["All","critical","warning","none"])
        status_f = fc2.selectbox("Filter by status",   ["All","open","resolved","rejected"])

        shown = all_inv
        if sev_f    != "All": shown = [i for i in shown if i.get("severity") == sev_f]
        if status_f != "All": shown = [i for i in shown if i.get("status")   == status_f]
        shown = sorted(shown, key=lambda i: i.get("created_at",""), reverse=True)

        st.markdown(f'<div class="section-label">{len(shown)} case(s)</div>',
                    unsafe_allow_html=True)

        for inv in shown:
            sev    = inv.get("severity", "unknown")
            action = inv.get("recommended_action", "—")
            status = inv.get("status", "unknown")
            inv_id = inv.get("investigation_id", "")
            is_sel = st.session_state.sel == inv_id
            card_cls = f"triage-card {sev}" + (" selected" if is_sel else "")

            st.markdown(f"""
            <div class="{card_cls}">
                <div class="tc-header">
                    {_badge(sev)}
                    <span class="tc-action">{action}</span>
                    {_badge(status)}
                </div>
                <div class="tc-meta">
                    {_time_ago(inv.get('created_at',''))} · {inv_id[:12]}…
                </div>
            </div>""", unsafe_allow_html=True)

            if st.button(
                "Viewing" if is_sel else "Open detail",
                key=f"sel_{inv_id}",
                use_container_width=True,
                type="primary" if is_sel else "secondary",
            ):
                st.session_state.sel = inv_id
                st.rerun()

    with col_detail:
        sel_id = st.session_state.sel
        sel    = next((i for i in all_inv if i.get("investigation_id") == sel_id), None)

        if sel is None:
            st.markdown("""
            <div style="text-align:center;padding:60px 40px;background:#f8fafc;
                        border-radius:12px;border:1px dashed #cbd5e1;margin-top:48px">
                <div style="font-size:15px;font-weight:600;color:#475569">No case selected</div>
                <div style="font-size:13px;color:#94a3b8;margin-top:6px">
                    Select a case from the list to view its details
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            sev     = sel.get("severity",  "unknown")
            status  = sel.get("status",    "unknown")
            action  = sel.get("recommended_action", "—")
            summary = sel.get("summary",   "No summary.")
            inv_id  = sel.get("investigation_id", "")
            approved = sel.get("human_approved")
            appr_str = "Approved" if approved is True else "Rejected" if approved is False else "Pending"

            st.markdown(f"""
            <div class="detail-header">
                <div class="detail-title">{_badge(sev)} &nbsp; {action.upper()}</div>
                <div class="detail-meta">
                    {status.upper()} · {appr_str} · {_time_ago(sel.get('created_at',''))}
                    · <code>{inv_id[:16]}…</code>
                </div>
            </div>""", unsafe_allow_html=True)

            tab_a, tab_b = st.tabs(["Summary", "Drift Analysis"])

            with tab_a:
                st.markdown(f"<p style='font-size:14px;color:#334155;line-height:1.6'>{summary}</p>",
                            unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(f"""
                    <div class="kpi-card info" style="margin-top:0">
                        <div class="kpi-label">Recommended action</div>
                        <div class="kpi-value" style="font-size:18px;font-family:monospace">{action}</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    job_id = sel.get("job_id", "—")
                    st.markdown(f"""
                    <div class="kpi-card {'success' if sel.get('job_id') else 'info'}" style="margin-top:0">
                        <div class="kpi-label">Job dispatched</div>
                        <div class="kpi-value" style="font-size:14px;font-family:monospace">
                            {str(job_id)[:16] + '…' if len(str(job_id)) > 16 else job_id}
                        </div>
                    </div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                _ai(f"Severity <strong>{sev}</strong> → recommended <code>{action}</code>. "
                    + ("Human approved — job dispatched." if approved is True
                       else "Human rejected — no Production change." if approved is False
                       else "Waiting for human approval in the HIL Inbox."))

            with tab_b:
                with st.spinner("Loading drift report…"):
                    drift = fetch_drift()
                if not drift:
                    st.info(
                        "Drift report unavailable. The model service needs at least "
                        "10 predictions before it can calculate drift. Send some "
                        "predictions to POST /predict first."
                    )
                else:
                    # ── Overview metrics ──────────────────────────────────
                    d1, d2, d3 = st.columns(3)
                    d1.metric(
                        "Overall severity",
                        drift.get("severity", "—").upper(),
                        help="critical = immediate action needed · warning = monitor closely · none = model is healthy",
                    )
                    d2.metric(
                        "Output drift score",
                        f"{drift.get('output_drift', 0):.4f}",
                        help="Measures how much the model's prediction distribution has shifted compared to training. Above 0.10 is notable; above 0.20 is severe.",
                    )
                    d3.metric(
                        "Predictions analysed",
                        f"{drift.get('window_size', '—'):,}" if isinstance(drift.get('window_size'), int) else "—",
                        help="Number of recent predictions used to compute this report.",
                    )

                    st.markdown("<br>", unsafe_allow_html=True)
                    _ai(
                        "Each bar below shows how much a feature has shifted compared to "
                        "what the model saw during training. "
                        "<strong>Red bars</strong> are above the alert threshold — these features "
                        "are behaving unusually and are contributing to the drift. "
                        "<strong>Grey bars</strong> are within normal range."
                    )
                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── Numeric features (PSI) ────────────────────────────
                    numeric = drift.get("numeric_drift", [])
                    if numeric:
                        st.markdown("**Numeric features** — scored with PSI (Population Stability Index)")
                        st.caption(
                            "PSI measures how much a number-based feature has shifted. "
                            "Below 0.10 is normal · 0.10–0.20 is a warning · above 0.20 means significant drift."
                        )
                        num_sorted = sorted(numeric, key=lambda x: x.get("statistic", 0), reverse=True)
                        names  = [x["feature"] for x in num_sorted]
                        scores = [x.get("statistic", 0) for x in num_sorted]
                        colors = ["#ef4444" if x.get("drifted") else "#cbd5e1" for x in num_sorted]
                        fig = go.Figure(go.Bar(
                            x=scores, y=names, orientation="h",
                            marker_color=colors,
                            text=[f"{s:.3f}" for s in scores],
                            textposition="outside",
                        ))
                        fig.update_layout(
                            height=max(200, len(names) * 28),
                            margin=dict(t=10, b=0, l=0, r=60),
                            xaxis=dict(title="PSI score", showgrid=True,
                                       gridcolor="#f1f5f9", zeroline=False),
                            yaxis=dict(showgrid=False, autorange="reversed"),
                            plot_bgcolor="white", paper_bgcolor="white",
                            font=dict(size=11),
                            shapes=[dict(
                                type="line", x0=0.1, x1=0.1, y0=-0.5,
                                y1=len(names) - 0.5,
                                line=dict(color="#f59e0b", width=1.5, dash="dot"),
                            )],
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Dotted line = alert threshold (PSI 0.10)")

                    # ── Categorical features (Chi²) ───────────────────────
                    categorical = drift.get("categorical_drift", [])
                    if categorical:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("**Categorical features** — scored with Chi-squared test")
                        st.caption(
                            "The chi-squared test checks whether a category-based feature "
                            "has an unusual distribution compared to training. "
                            "A higher score means the distribution has shifted more."
                        )
                        cat_sorted = sorted(categorical, key=lambda x: x.get("statistic", 0), reverse=True)
                        names  = [x["feature"] for x in cat_sorted]
                        scores = [x.get("statistic", 0) for x in cat_sorted]
                        colors = ["#ef4444" if x.get("drifted") else "#cbd5e1" for x in cat_sorted]
                        fig2 = go.Figure(go.Bar(
                            x=scores, y=names, orientation="h",
                            marker_color=colors,
                            text=[f"{s:.1f}" for s in scores],
                            textposition="outside",
                        ))
                        fig2.update_layout(
                            height=max(200, len(names) * 28),
                            margin=dict(t=10, b=0, l=0, r=60),
                            xaxis=dict(title="Chi-squared score", showgrid=True,
                                       gridcolor="#f1f5f9", zeroline=False),
                            yaxis=dict(showgrid=False, autorange="reversed"),
                            plot_bgcolor="white", paper_bgcolor="white",
                            font=dict(size=11),
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                    if not numeric and not categorical:
                        st.info("No per-feature breakdown available in the drift report.")


# ════════════════════════════════════════════════════════════════════════
# JOB QUEUE  — fetches: queue
# ════════════════════════════════════════════════════════════════════════
elif page == "Job Queue":
    st.markdown("# Job Queue")
    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner("Loading…"):
        queue = fetch_queue()

    k = st.columns(3)
    _kpi(k[0], "Waiting in Queue",  queue["depth"],
         "Jobs pending execution", "info")
    _kpi(k[1], "Dead-letter (DLQ)", queue["dlq"],
         "Manual review needed" if queue["dlq"] else "No failed jobs",
         "danger" if queue["dlq"] else "success")
    _kpi(k[2], "Total Processed",   queue["processed"],
         "Successfully completed", "success")

    st.markdown("<br>", unsafe_allow_html=True)

    if queue["dlq"] > 0:
        _ai(f"<strong>{queue['dlq']}</strong> job(s) exceeded the retry limit. "
            "Fix the root cause, then delete the DLQ entry and re-trigger.")
        st.markdown('<div class="section-label">Dead-letter queue items</div>',
                    unsafe_allow_html=True)
        for raw in queue["dlq_items"]:
            try:
                p = json.loads(raw)
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f"**`{p.get('job_type','?')}`** — {p.get('error','unknown error')}")
                    c2.markdown(f"Retries: **{p.get('retries','?')}**")
                    st.caption(f"ID: `{p.get('job_id','?')}`")
            except Exception:
                st.code(raw, language="json")
    elif queue["depth"] == 0:
        st.success("Queue is empty — all jobs processed.")
        _ai(f"<strong>{queue['processed']}</strong> total jobs completed successfully.")
    else:
        _ai(f"<strong>{queue['depth']}</strong> job(s) queued — the worker will process them shortly.")


# ════════════════════════════════════════════════════════════════════════
# REGISTRY  — fetches: versions
# ════════════════════════════════════════════════════════════════════════
elif page == "Registry":
    st.markdown("# Model Registry")
    st.markdown("<br>", unsafe_allow_html=True)
    _ai("Models in <strong>Production</strong> serve live predictions. "
        "<strong>Staging</strong> models are candidates awaiting promotion.")
    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner("Loading registry…"):
        versions = fetch_versions()

    STAGE_COLOR = {
        "Production": "#10b981",
        "Staging":    "#f59e0b",
        "Archived":   "#94a3b8",
        "None":       "#94a3b8",
    }
    STAGE_STYLE = {
        "Production": "success",
        "Staging":    "warning",
        "Archived":   "info",
        "None":       "info",
    }
    if versions:
        for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
            stage = v.current_stage
            color = STAGE_COLOR.get(stage, "#94a3b8")
            style = STAGE_STYLE.get(stage, "info")
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([1, 2, 2, 3])
                c1.markdown(
                    f"<div style='text-align:center;font-size:26px;font-weight:700;"
                    f"color:{color}'>v{v.version}</div>",
                    unsafe_allow_html=True,
                )
                c2.markdown(f"**Stage**  \n`{stage}`")
                c3.markdown(f"**Run ID**  \n`{v.run_id[:12]}…`")
                c4.markdown(f"**Model**  \n`{MODEL_NAME}`")
    else:
        st.info("No model versions registered yet.")


# ════════════════════════════════════════════════════════════════════════
# HIL INBOX  — fetches: pending
# ════════════════════════════════════════════════════════════════════════
elif page == "HIL Inbox":
    st.markdown("# Human-in-the-Loop Inbox")
    st.caption("The AI agent has paused and is waiting for your approval before making any changes to the live model.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Plain-English explanations keyed by severity
    SEV_PLAIN = {
        "critical": (
            "The live model is behaving significantly differently from how it was trained. "
            "The patterns in recent data have shifted enough that the model may now be "
            "producing unreliable predictions. Without action, prediction quality will continue to degrade."
        ),
        "warning": (
            "Early signs of drift are appearing. The model is starting to encounter data "
            "it hasn't seen before. It isn't failing yet, but the trend needs attention "
            "before accuracy drops further."
        ),
    }

    # Plain-English explanations of each action (approve path)
    ACTION_PLAIN = {
        "retrain": (
            "A new version of the model will be trained using recent data to restore accuracy. "
            "The current model keeps serving predictions while training runs in the background. "
            "Once the new version passes quality checks it takes over automatically."
        ),
        "rollback": (
            "The live model will immediately switch back to the previous version. "
            "Use this when you need a fast fix and the previous version was working correctly."
        ),
        "replay_test": (
            "The current model will be tested against a reference dataset to verify its accuracy "
            "hasn't changed. No changes are made to the live model — this is a diagnostic check only."
        ),
        "monitor": (
            "No changes will be made. The system will continue monitoring the model and alert "
            "you if the situation worsens."
        ),
    }

    with st.spinner("Loading inbox…"):
        pending = fetch_pending()

    if not pending:
        st.markdown("""
        <div style="text-align:center;padding:60px 40px;background:#f0fdf4;
                    border-radius:12px;border:1px solid #bbf7d0">
            <div style="font-size:16px;font-weight:600;color:#166534">Inbox clear</div>
            <div style="font-size:13px;color:#166534;margin-top:6px;opacity:0.8">
                No pending approvals — all investigations have been decided.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;
                    padding:12px 18px;margin-bottom:24px;color:#991b1b;font-weight:600">
            {len(pending)} action(s) waiting for your decision
        </div>""", unsafe_allow_html=True)

        for item in pending:
            inv_id  = item["investigation_id"]
            action  = item.get("recommended_action", "unknown")

            # interrupt_data is stored as JSONB — psycopg2 returns it as a dict
            idata       = item.get("interrupt_data") or {}
            severity    = idata.get("severity", "critical")
            output_drift = idata.get("output_drift")
            window_size  = idata.get("window_size")

            sev_badge = _badge(severity)

            with st.container(border=True):

                # ── Card header ──────────────────────────────────────────
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;
                            padding-bottom:14px;border-bottom:1px solid #f1f5f9;margin-bottom:18px">
                    <span style="font-size:18px;font-weight:700;color:#1a2236">
                        Recommended action: <code style="font-size:16px">{action.upper()}</code>
                    </span>
                    {sev_badge}
                    <span style="font-size:11px;color:#94a3b8;margin-left:auto">{inv_id[:16]}…</span>
                </div>""", unsafe_allow_html=True)

                # ── Section 1: What is drift? ─────────────────────────────
                st.markdown("**What is model drift?**")
                st.markdown(
                    "The AI model was trained on historical data. Over time, the real world changes — "
                    "customer behaviour shifts, economic conditions evolve, or the type of data coming "
                    "in changes. When the gap between what the model was trained on and what it sees "
                    "today grows too large, its predictions become less reliable. "
                    "This is called **model drift**.",
                )
                st.markdown("<br>", unsafe_allow_html=True)

                # ── Section 2: What happened ──────────────────────────────
                st.markdown("**What has been detected?**")
                sev_text = SEV_PLAIN.get(severity, SEV_PLAIN["critical"])
                if severity == "critical":
                    st.error(sev_text)
                else:
                    st.warning(sev_text)

                # Drift numbers in plain context
                if output_drift is not None or window_size is not None:
                    m1, m2, m3 = st.columns(3)
                    m1.metric(
                        "Severity level",
                        severity.upper(),
                        help="critical = significant drift requiring immediate action; warning = early signs of drift",
                    )
                    if output_drift is not None:
                        drift_level = "High" if output_drift > 0.2 else "Moderate" if output_drift > 0.1 else "Low"
                        m2.metric(
                            "Drift score",
                            f"{output_drift:.3f}",
                            delta=f"{drift_level} — threshold is 0.10",
                            delta_color="inverse",
                            help="Measures how much the model's output distribution has shifted. Above 0.10 is notable; above 0.20 is severe.",
                        )
                    if window_size is not None:
                        m3.metric(
                            "Predictions analysed",
                            f"{window_size:,}",
                            help="Number of recent predictions used to calculate drift.",
                        )
                    st.markdown("<br>", unsafe_allow_html=True)

                # ── Section 3: What the AI recommends ─────────────────────
                st.markdown("**What is the AI recommending and why?**")
                action_text = ACTION_PLAIN.get(action, "A corrective action is recommended.")
                _ai(action_text)
                st.markdown("<br>", unsafe_allow_html=True)

                # ── Section 4: Consequences ───────────────────────────────
                col_approve_info, col_reject_info = st.columns(2)
                with col_approve_info:
                    st.markdown("**If you approve**")
                    st.info(
                        IMPACT_MAP.get(action, "The recommended action will be executed on the live model.")
                    )
                with col_reject_info:
                    st.markdown("**If you reject**")
                    st.info(
                        "No changes will be made to the live model. "
                        "The investigation will be closed and recorded. "
                        "The model will continue running as-is."
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                st.divider()

                # ── Decision buttons ──────────────────────────────────────
                col_a, col_r, _ = st.columns([1, 1, 4])

                with col_a:
                    if st.button("Approve", key=f"approve_{inv_id}",
                                 type="primary", use_container_width=True):
                        r = requests.post(f"{AGENT_URL}/approve/{inv_id}",
                                          json={"approved": True}, timeout=10)
                        if r.status_code == 200:
                            st.success("Approved — the agent will now proceed.")
                            time.sleep(1)
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error(f"Failed: {r.text}")

                with col_r:
                    if st.button("Reject", key=f"reject_{inv_id}",
                                 use_container_width=True):
                        r = requests.post(f"{AGENT_URL}/approve/{inv_id}",
                                          json={"approved": False}, timeout=10)
                        if r.status_code == 200:
                            st.warning("Rejected — investigation closed, no changes made.")
                            time.sleep(1)
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error(f"Failed: {r.text}")

                st.caption(f"Case ID: `{inv_id}`")
