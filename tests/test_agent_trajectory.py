"""
Agent snapshot trajectory tests.
Runs the LangGraph supervisor with a MemorySaver checkpointer and a mocked
LLM so CI never needs an ANTHROPIC_API_KEY.

Three scenarios are tested:
  1. severity=none  → monitor  → comms  (no HIL, no queue)
  2. severity=critical → retrain → HIL approved → action → comms
  3. severity=critical → retrain → HIL rejected → comms (no queue)
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent.supervisor import InvestigationState, create_graph

# ---------------------------------------------------------------------------
# Recorded fixtures (the expected outputs for each scenario)
# ---------------------------------------------------------------------------
FIXTURE_TRIAGE_MONITOR = {
    "recommended_action": "monitor",
    "reason": "No significant drift detected.",
    "touches_production": False,
}

FIXTURE_TRIAGE_RETRAIN = {
    "recommended_action": "retrain",
    "reason": "Critical drift in euribor3m — retrain required.",
    "touches_production": True,
}

FIXTURE_ACTION_RETRAIN = {
    "job_type": "retrain",
    "idempotency_key": "test-id-retrain",
    "payload": {"investigation_id": "test-id", "triggered_by": "drift_triage_agent"},
}

FIXTURE_COMMS_RESOLVED = {
    "summary": "Investigation complete. Action dispatched.",
    "status": "resolved",
}

FIXTURE_COMMS_REJECTED = {
    "summary": "Investigation closed. Action rejected by human.",
    "status": "rejected",
}

FIXTURE_COMMS_MONITORED = {
    "summary": "No drift action needed. Monitoring continues.",
    "status": "resolved",
}

DRIFT_EVENT_NONE     = {"severity": "none",     "window_size": 500, "output_drift": 0.05}
DRIFT_EVENT_CRITICAL = {"severity": "critical",  "window_size": 500, "output_drift": 0.30,
                        "numeric_drift": [], "categorical_drift": []}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_llm_mock(response_dict: dict) -> MagicMock:
    """Return a mock Anthropic client whose .messages.create() returns response_dict as JSON."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value.content = [
        MagicMock(text=json.dumps(response_dict))
    ]
    return mock_client


def _run_graph_to_interrupt(graph, drift_event: dict, investigation_id: str):
    """Stream the graph until it hits an interrupt or completes."""
    config = {"configurable": {"thread_id": investigation_id}}
    initial: InvestigationState = {
        "investigation_id":   investigation_id,
        "drift_event":        drift_event,
        "triage_result":      None,
        "recommended_action": None,
        "touches_production": False,
        "human_approved":     None,
        "action_result":      None,
        "comms_result":       None,
        "next":               "supervisor",
    }
    for _ in graph.stream(initial, config=config):
        pass
    snapshot = graph.get_state(config)
    return snapshot


def _resume_graph(graph, investigation_id: str, approved: bool):
    """Resume a paused graph with the human decision."""
    from langgraph.types import Command
    config = {"configurable": {"thread_id": investigation_id}}
    for _ in graph.stream(Command(resume={"approved": approved}), config=config):
        pass
    return graph.get_state(config)


# ---------------------------------------------------------------------------
# Scenario 1 — monitor (no HIL, no queue)
# ---------------------------------------------------------------------------
@patch("agent.sub_agents.comms.psycopg2.connect")
@patch("agent.sub_agents.comms.anthropic.Anthropic")
@patch("agent.sub_agents.triage.anthropic.Anthropic")
def test_monitor_trajectory(mock_triage_cls, mock_comms_cls, mock_db_connect):
    mock_triage_cls.return_value = _make_llm_mock(FIXTURE_TRIAGE_MONITOR)
    mock_comms_cls.return_value  = _make_llm_mock(FIXTURE_COMMS_MONITORED)

    # Mock DB so comms sub-agent doesn't need a real Postgres
    mock_db_connect.return_value.__enter__ = lambda s: s
    mock_db_connect.return_value.__exit__  = MagicMock(return_value=False)
    mock_db_connect.return_value.cursor.return_value.__enter__ = lambda s: s
    mock_db_connect.return_value.cursor.return_value.__exit__  = MagicMock(return_value=False)

    graph = create_graph(MemorySaver())
    snapshot = _run_graph_to_interrupt(graph, DRIFT_EVENT_NONE, "test-monitor")

    state = snapshot.values
    # Routing assertions
    assert state["recommended_action"] == "monitor",            "triage must recommend monitor"
    assert state["touches_production"] is False,                "monitor must not touch production"
    assert state["action_result"] is None,                      "action node must be skipped"
    assert state["comms_result"] is not None,                   "comms must have run"
    assert state["comms_result"]["status"] == "resolved",       f"got: {state['comms_result']}"
    # No interrupt — graph completed
    assert not snapshot.tasks,                                  "graph must have completed with no pending interrupt"


# ---------------------------------------------------------------------------
# Scenario 2 — retrain, HIL approved
# ---------------------------------------------------------------------------
@patch("agent.sub_agents.comms.psycopg2.connect")
@patch("agent.sub_agents.comms.anthropic.Anthropic")
@patch("agent.sub_agents.action.redis.from_url")
@patch("agent.sub_agents.action.anthropic.Anthropic")
@patch("agent.sub_agents.triage.anthropic.Anthropic")
def test_retrain_approved_trajectory(
    mock_triage_cls, mock_action_cls, mock_redis_from_url,
    mock_comms_cls, mock_db_connect,
):
    mock_triage_cls.return_value = _make_llm_mock(FIXTURE_TRIAGE_RETRAIN)
    mock_action_cls.return_value = _make_llm_mock(FIXTURE_ACTION_RETRAIN)
    mock_comms_cls.return_value  = _make_llm_mock(FIXTURE_COMMS_RESOLVED)

    # Redis mock — idempotency key not seen before
    mock_redis = MagicMock()
    mock_redis.sismember.return_value = False
    mock_redis_from_url.return_value  = mock_redis

    # DB mock
    mock_db_connect.return_value.__enter__ = lambda s: s
    mock_db_connect.return_value.__exit__  = MagicMock(return_value=False)
    mock_db_connect.return_value.cursor.return_value.__enter__ = lambda s: s
    mock_db_connect.return_value.cursor.return_value.__exit__  = MagicMock(return_value=False)

    graph = create_graph(MemorySaver())

    # Phase 1 — runs triage, then pauses before action
    snapshot = _run_graph_to_interrupt(graph, DRIFT_EVENT_CRITICAL, "test-retrain-approved")
    state = snapshot.values
    assert state["recommended_action"] == "retrain",  "triage must recommend retrain"
    assert state["touches_production"] is True,       "retrain must touch production"
    assert snapshot.tasks,                            "graph must be paused at HIL interrupt"

    # Phase 2 — human approves
    snapshot = _resume_graph(graph, "test-retrain-approved", approved=True)
    state = snapshot.values
    assert state["human_approved"] is True,                              "approval must be recorded"
    assert state["action_result"] is not None,                           "action must have run"
    assert state["action_result"]["queued"] is True,                     "job must have been queued"
    assert state["comms_result"]["status"] == "resolved",                f"got: {state['comms_result']}"
    # Redis rpush must have been called once
    mock_redis.rpush.assert_called_once()


# ---------------------------------------------------------------------------
# Scenario 3 — retrain, HIL rejected
# ---------------------------------------------------------------------------
@patch("agent.sub_agents.comms.psycopg2.connect")
@patch("agent.sub_agents.comms.anthropic.Anthropic")
@patch("agent.sub_agents.action.redis.from_url")
@patch("agent.sub_agents.action.anthropic.Anthropic")
@patch("agent.sub_agents.triage.anthropic.Anthropic")
def test_retrain_rejected_trajectory(
    mock_triage_cls, mock_action_cls, mock_redis_from_url,
    mock_comms_cls, mock_db_connect,
):
    mock_triage_cls.return_value = _make_llm_mock(FIXTURE_TRIAGE_RETRAIN)
    mock_action_cls.return_value = _make_llm_mock(FIXTURE_ACTION_RETRAIN)
    mock_comms_cls.return_value  = _make_llm_mock(FIXTURE_COMMS_REJECTED)

    mock_redis = MagicMock()
    mock_redis_from_url.return_value = mock_redis

    mock_db_connect.return_value.__enter__ = lambda s: s
    mock_db_connect.return_value.__exit__  = MagicMock(return_value=False)
    mock_db_connect.return_value.cursor.return_value.__enter__ = lambda s: s
    mock_db_connect.return_value.cursor.return_value.__exit__  = MagicMock(return_value=False)

    graph = create_graph(MemorySaver())
    _run_graph_to_interrupt(graph, DRIFT_EVENT_CRITICAL, "test-retrain-rejected")

    # Human rejects
    snapshot = _resume_graph(graph, "test-retrain-rejected", approved=False)
    state = snapshot.values
    assert state["human_approved"] is False,                               "rejection must be recorded"
    assert state["action_result"]["queued"] is False,                      "job must NOT have been queued"
    assert state["comms_result"]["status"] == "rejected",                  f"got: {state['comms_result']}"
    # Redis rpush must NOT have been called
    mock_redis.rpush.assert_not_called()
