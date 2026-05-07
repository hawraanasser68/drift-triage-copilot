"""
LangGraph supervisor graph — Drift Triage Co-Pilot.

Topology (true supervisor, not a chain):
  supervisor → triage → supervisor → action (HIL interrupt) → supervisor → comms → END

State is persisted in Postgres via PostgresSaver so the agent survives restarts.
"""

import os
import uuid
from typing import Literal
from typing_extensions import TypedDict

import anthropic
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.errors import NodeInterrupt

from agent.sub_agents import triage as triage_agent
from agent.sub_agents import action as action_agent
from agent.sub_agents import comms  as comms_agent

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/mlops")


# ---------------------------------------------------------------------------
# Shared state for one investigation
# ---------------------------------------------------------------------------
class InvestigationState(TypedDict):
    investigation_id:   str
    drift_event:        dict
    triage_result:      dict | None    # set by triage node
    recommended_action: str | None     # copied from triage_result for easy access
    touches_production: bool           # True → needs human approval
    human_approved:     bool | None    # None = not yet decided
    action_result:      dict | None    # set by action node
    comms_result:       dict | None    # set by comms node
    next:               str            # supervisor routing signal


# ---------------------------------------------------------------------------
# Node: supervisor
# Decides which sub-agent to call next based on current state.
# ---------------------------------------------------------------------------
def supervisor_node(state: InvestigationState) -> InvestigationState:
    if state.get("triage_result") is None:
        return {**state, "next": "triage"}

    if state.get("recommended_action") == "monitor":
        # No action needed — skip straight to comms
        if state.get("comms_result") is None:
            return {**state, "next": "comms"}
        return {**state, "next": "end"}

    if state.get("action_result") is None:
        return {**state, "next": "action"}

    if state.get("comms_result") is None:
        return {**state, "next": "comms"}

    return {**state, "next": "end"}


def _route(state: InvestigationState) -> Literal["triage", "action", "comms", "__end__"]:
    return state["next"] if state["next"] != "end" else "__end__"


# ---------------------------------------------------------------------------
# Node: triage
# ---------------------------------------------------------------------------
def triage_node(
    state: InvestigationState,
    llm_client: anthropic.Anthropic | None = None,
) -> InvestigationState:
    result = triage_agent.run(state["drift_event"], llm_client=llm_client)
    return {
        **state,
        "triage_result":      result,
        "recommended_action": result["recommended_action"],
        "touches_production": result.get("touches_production", False),
        "next":               "supervisor",
    }


# ---------------------------------------------------------------------------
# Node: action  (with HIL interrupt for Production-touching actions)
# ---------------------------------------------------------------------------
def action_node(
    state: InvestigationState,
    llm_client: anthropic.Anthropic | None = None,
    redis_client=None,
) -> InvestigationState:
    # If this action touches Production and we haven't asked the human yet → pause
    if state["touches_production"] and state["human_approved"] is None:
        drift = state["drift_event"]
        raise NodeInterrupt({
            "investigation_id":   state["investigation_id"],
            "recommended_action": state["recommended_action"],
            "reason":             state["triage_result"]["reason"],
            "message":            "Human approval required before dispatching to Production.",
            "severity":           drift.get("severity", "unknown"),
            "output_drift":       drift.get("output_drift"),
            "window_size":        drift.get("window_size"),
        })

    # Human rejected → skip dispatching, head to comms
    if state["human_approved"] is False:
        return {**state, "action_result": {"queued": False, "reason": "rejected by human"}, "next": "supervisor"}

    result = action_agent.run(
        investigation_id=state["investigation_id"],
        recommended_action=state["recommended_action"],
        reason=state["triage_result"]["reason"],
        llm_client=llm_client,
        redis_client=redis_client,
    )
    return {**state, "action_result": result, "next": "supervisor"}


# ---------------------------------------------------------------------------
# Node: comms
# ---------------------------------------------------------------------------
def comms_node(
    state: InvestigationState,
    llm_client: anthropic.Anthropic | None = None,
    db_conn=None,
) -> InvestigationState:
    action_result = state.get("action_result") or {}
    result = comms_agent.run(
        investigation_id=state["investigation_id"],
        severity=state["drift_event"].get("severity", "unknown"),
        recommended_action=state["recommended_action"] or "monitor",
        human_approved=state.get("human_approved"),
        job_id=action_result.get("job_id"),
        llm_client=llm_client,
        db_conn=db_conn,
    )
    return {**state, "comms_result": result, "next": "end"}


# ---------------------------------------------------------------------------
# Build the graph
# (takes a checkpointer so tests can inject a MemorySaver instead)
# ---------------------------------------------------------------------------
def create_graph(checkpointer):
    builder = StateGraph(InvestigationState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("triage",     triage_node)
    builder.add_node("action",     action_node)
    builder.add_node("comms",      comms_node)

    builder.set_entry_point("supervisor")

    # Supervisor uses conditional edges to route
    builder.add_conditional_edges("supervisor", _route, {
        "triage":    "triage",
        "action":    "action",
        "comms":     "comms",
        "__end__":   END,
    })

    # All sub-agents report back to supervisor
    builder.add_edge("triage",  "supervisor")
    builder.add_edge("action",  "supervisor")
    builder.add_edge("comms",   "supervisor")

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Convenience functions used by app.py
# ---------------------------------------------------------------------------
def start_investigation(drift_event: dict, graph) -> tuple[str, bool, dict | None]:
    """
    Kick off a new investigation.
    Returns (investigation_id, is_paused, interrupt_payload).
    is_paused=True means the graph stopped at the HIL interrupt and needs human approval.
    """
    investigation_id = str(uuid.uuid4())
    initial_state: InvestigationState = {
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
    config = {"configurable": {"thread_id": investigation_id}}
    for _ in graph.stream(initial_state, config=config):
        pass

    # Check if graph paused at a NodeInterrupt
    snapshot       = graph.get_state(config)
    is_paused      = bool(snapshot.next)
    interrupt_data = None
    if is_paused and snapshot.tasks:
        for task in snapshot.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupt_data = task.interrupts[0].value
                break

    return investigation_id, is_paused, interrupt_data


def resume_investigation(investigation_id: str, approved: bool, graph) -> None:
    """Resume a paused investigation after human approval / rejection."""
    config = {"configurable": {"thread_id": investigation_id}}
    # Update human_approved in the checkpoint, then let LangGraph re-run the
    # interrupted action node with the full restored state.
    graph.update_state(config, {"human_approved": approved})
    for _ in graph.stream(None, config=config):
        pass
