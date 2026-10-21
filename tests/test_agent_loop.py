"""Agent-loop tests: the real Strands agent and guardrail, with a fake decider.

Run: python tests/test_agent_loop.py   (also collected by pytest)

The decider is replaced by a stub that returns a chosen grounding score per
tool, so these tests need no model download and no AWS. They cover the paths
the real-model test can't force on demand: Deny, Confirm approved and resumed,
Confirm declined, and planning arguments kept out of the grounding check.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from strands import Agent  # noqa: E402

from oncall_decider.agent import MockGenerativeModel, run_with_confirmations, strip_thinking  # noqa: E402
from oncall_decider.interventions import RemediationGuardrail  # noqa: E402
from oncall_decider.tools import TOOLS  # noqa: E402

ALERT = {
    "id": "ALT-1001",
    "service": "checkout-api",
    "title": "Error budget burn: checkout-api serving 12% errors",
    "description": "checkout-api's 5xx rate is 12% after the latest deploy.",
    "labels": {"severity": "P1", "team": "payments", "namespace": "prod"},
    "signals": {"http_requests_5xx_rate": 0.12},
}


class FakeDecider:
    """Returns a fixed grounding score per tool name and records each state it saw."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.states: list[str] = []

    def decide(self, state, questions):
        self.states.append(state)
        tool = state.rsplit("Proposed tool call: ", 1)[1].split("(", 1)[0]
        p = self.scores[tool]
        return {"grounded": {"noul": p, "decision": p >= 0.5, "confidence": max(p, 1 - p)}}


def call(name, tid, **inp):
    return {"tool_use": {"name": name, "id": tid, "input": inp}}


def make_agent(decider, script):
    guard = RemediationGuardrail(ALERT, decider=decider)
    agent = Agent(model=MockGenerativeModel(script=script), tools=TOOLS,
                  interventions=[guard], callback_handler=None)
    return agent, guard


def tool_results(agent):
    return {b["toolResult"]["toolUseId"]: b["toolResult"]
            for m in agent.messages for b in m.get("content", []) if "toolResult" in b}


def text_of(result):
    return " ".join(c.get("text", "") for c in result.get("content", []) if isinstance(c, dict))


def test_deny_cancels_the_call_and_names_the_value():
    dec = FakeDecider({"restart_service": 0.10})
    agent, guard = make_agent(dec, [call("restart_service", "t1", service="checkout-api", namespace="staging"),
                                    {"text": "done"}])
    _, confirms = run_with_confirmations(agent, "go", asker=lambda p: True)
    res = tool_results(agent)["t1"]
    assert res["status"] == "error" and text_of(res).startswith("DENIED:"), res
    assert "staging" in text_of(res)
    assert confirms == [] and guard.records[0]["action"] == "deny"


def test_confirm_approved_resumes_and_runs_the_tool():
    dec = FakeDecider({"restart_service": 0.50})
    agent, guard = make_agent(dec, [call("restart_service", "t1", service="checkout-api", namespace="prod"),
                                    {"text": "restarted"}])
    asked = []
    result, confirms = run_with_confirmations(agent, "go", asker=lambda p: asked.append(p) or True)
    assert len(asked) == 1 and "Approve?" in asked[0]
    assert confirms == [{"prompt": asked[0], "approved": True}]
    assert result.stop_reason != "interrupt"
    assert tool_results(agent)["t1"]["status"] == "success"
    # Resuming re-runs the hook; the guardrail must not ask the model twice.
    assert len(dec.states) == 1 and len(guard.records) == 1


def test_confirm_declined_cancels_and_agent_finishes():
    dec = FakeDecider({"restart_service": 0.50})
    agent, _ = make_agent(dec, [call("restart_service", "t1", service="checkout-api", namespace="prod"),
                                {"text": "left it alone"}])
    result, confirms = run_with_confirmations(agent, "go", asker=lambda p: False)
    assert confirms[0]["approved"] is False
    assert result.stop_reason != "interrupt"
    res = tool_results(agent)["t1"]
    assert res["status"] == "error" and "CONFIRMATION_FAILED" in text_of(res)


def test_planning_args_are_not_grounded():
    dec = FakeDecider({"scale_deployment": 0.90, "open_incident_ticket": 0.90})
    agent, guard = make_agent(dec, [
        call("scale_deployment", "t1", service="checkout-api", replicas=3, namespace="prod"),
        call("open_incident_ticket", "t2", title="anything", severity="P1", summary="free text"),
        {"text": "done"},
    ])
    run_with_confirmations(agent, "go", asker=lambda p: True)
    scale_state = dec.states[0]
    assert "replicas" not in scale_state and "service='checkout-api'" in scale_state
    # The ticket's title and summary are prose the agent writes; only severity is checked.
    assert any(s.endswith("open_incident_ticket(severity='P1')") for s in dec.states), dec.states
    assert all(r["action"] == "proceed" for r in guard.records)


def test_strip_thinking():
    assert strip_thinking("<thinking>plan\nmore</thinking>\n\nSummary.") == "Summary."
    assert strip_thinking("No tags here.") == "No tags here."


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS: {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL: {name}: {exc!r}")
    print("\nALL PASS" if not failures else f"\n{failures} FAILURE(S)")
    sys.exit(1 if failures else 0)
