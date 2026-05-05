"""
Triage sub-agent.
Reads the drift event, calls the LLM with the triage prompt,
and returns a recommended action + whether it touches Production.
"""

import json
import os
from pathlib import Path

import anthropic

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "triage.md"
MODEL       = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


def run(drift_event: dict, llm_client: anthropic.Anthropic | None = None) -> dict:
    """
    Args:
        drift_event: the payload received from the model service webhook.
        llm_client:  injectable Anthropic client (pass a mock in tests).
    Returns:
        dict with keys: recommended_action, reason, touches_production
    """
    if llm_client is None:
        llm_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    system_prompt = PROMPT_PATH.read_text()

    # Build the user message from the drift event
    drifted_features = [
        f["feature"]
        for f in drift_event.get("numeric_drift", []) + drift_event.get("categorical_drift", [])
        if f.get("drifted")
    ]
    user_input = json.dumps({
        "severity":         drift_event.get("severity"),
        "window_size":      drift_event.get("window_size"),
        "output_drift":     drift_event.get("output_drift"),
        "drifted_features": drifted_features,
    })

    response = llm_client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=system_prompt,
        messages=[{"role": "user", "content": user_input}],
    )

    raw = response.content[0].text.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: safe default if LLM returns unexpected output
        result = {
            "recommended_action": "monitor",
            "reason": "Triage LLM returned unparseable output; defaulting to monitor.",
            "touches_production": False,
        }

    return result
