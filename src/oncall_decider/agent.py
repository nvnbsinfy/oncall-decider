"""Wiring for the on-call agent: model selection, the mock generative
model, and the storm loop.

Two halves, same as the contest entry:
- Decisions (triage policy, remediation grounding) come from Decider 2B
  via the sibling modules. That code runs locally, offline.
- Generative writing comes from Bedrock when credentials exist, and
  from an explicit dev-only mock otherwise. This VM has no AWS
  credentials, so the mock is what runs here.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Any, Iterable

from strands import Agent
from strands.models.model import Model
from strands.types.streaming import StreamEvent

try:
    from oncall_decider.decider_client import get_client
    from oncall_decider.tools import (
        restart_service,
        scale_deployment,
        silence_alert,
        open_incident_ticket,
        RECORDED_TICKETS,
    )
    from oncall_decider.servicenow import maybe_create_validated_incident
    from oncall_decider.interventions import (
        TriageDecider,
        RemediationGuardrail,
        TRIAGE_CLASSES,
    )
    _SIBLINGS_OK = True
except ImportError as exc:  # sibling agent's modules land later; fail loudly, not weirdly
    _SIBLINGS_OK = False
    _SIBLING_IMPORT_ERROR = exc

from oncall_decider.alerts import build_storm

logger = logging.getLogger(__name__)

BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")

# How a guardrail Confirm is answered:
#   ask     - prompt the person at the terminal (default). Without a terminal
#             (CI, piped stdin) there is nobody to ask, so the call is declined.
#   approve - approve every Confirm (only for unattended dry-run demos)
#   deny    - decline every Confirm
GUARDRAIL_CONFIRM = os.environ.get("GUARDRAIL_CONFIRM", "ask").strip().lower()
MAX_CONFIRM_ROUNDS = 5

_THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.S)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def _aws_credentials_present() -> bool:
    """True when boto3 can find any AWS credential on this machine."""
    try:
        import boto3

        return boto3.Session().get_credentials() is not None
    except Exception:
        return False


def get_model() -> tuple[Model, str]:
    """Pick the generative model for this run.

    Returns (model, mode) where mode is "bedrock" or "mock".
    Bedrock wins when credentials exist; otherwise we fall back to the
    mock and say so loudly, because the mock is dev-only and must never
    be mistaken for a real model in a write-up.
    """
    if _aws_credentials_present():
        try:
            from strands.models.bedrock import BedrockModel

            model = BedrockModel(model_id=BEDROCK_MODEL_ID)
            logger.info("Using Bedrock generative model: %s", BEDROCK_MODEL_ID)
            print(f"Using Bedrock generative model: {BEDROCK_MODEL_ID}")
            return model, "bedrock"
        except Exception as exc:  # Bedrock import/config failure: say so, then mock
            logger.warning("Bedrock requested but unusable (%s); falling back to mock.", exc)

    print("=" * 70)
    print("MOCK GENERATIVE MODEL ACTIVE (dev-only)")
    print("No AWS credentials found on this machine, so Bedrock is unreachable.")
    print("A scripted stand-in will play the role of the generative model.")
    print("Decider 2B triage and guardrail decisions are still real.")
    print("=" * 70)
    logger.warning("MockGenerativeModel active: no AWS credentials; Bedrock unreachable.")
    return MockGenerativeModel(script=[]), "mock"


# ---------------------------------------------------------------------------
# Mock generative model
# ---------------------------------------------------------------------------

class MockGenerativeModel(Model):
    """A scripted stand-in for the generative model. Dev and demo use only.

    Built against the Model ABC in the installed strands SDK: the Agent
    only ever calls ``stream()``, which yields Bedrock-shaped stream
    events. Each call pops the next scripted response off the queue, so
    the Agent's tool-use loop (model -> tool -> interventions -> model)
    runs exactly as it would with a real provider.

    Script items are dicts:
        {"tool_use": {"name": "restart_service",
                      "input": {"service": "checkout-api", "namespace": "prod"},
                      "id": "tu-1"}}
        {"text": "final summary..."}
    """

    def __init__(self, script: list[dict] | None = None, **config: Any):
        self.script = list(script or [])
        self._config = {"model_id": "mock-generative-model", **config}
        self.calls = 0  # how many times stream() was invoked

    # -- Model ABC ---------------------------------------------------------
    @property
    def stateful(self) -> bool:
        return False

    @property
    def context_window_limit(self) -> int | None:
        return None

    def update_config(self, **model_config: Any) -> None:
        self._config.update(model_config)

    def get_config(self) -> Any:
        return self._config

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        # Not used by the demo agent path; satisfy the ABC with a text event.
        yield {"contentBlockDelta": {"delta": {"text": "{}"}}}
        yield {"messageStop": {"stopReason": "end_turn"}}

    async def stream(
        self,
        messages,
        tool_specs=None,
        system_prompt=None,
        *,
        tool_choice=None,
        system_prompt_content=None,
        invocation_state=None,
        cancel_signal=None,
        agent_metadata=None,
        **kwargs,
    ) -> Iterable[StreamEvent]:
        """Yield one scripted response as Bedrock-style stream events."""
        self.calls += 1
        item = self.script.pop(0) if self.script else None
        if item is None:
            item = {"text": "Done. No further steps needed."}

        yield {"messageStart": {"role": "assistant"}}

        if "tool_use" in item:
            tu = item["tool_use"]
            tool_use_id = tu.get("id", f"mock-tu-{self.calls}")
            yield {
                "contentBlockStart": {
                    "start": {"toolUse": {"toolUseId": tool_use_id, "name": tu["name"]}}
                }
            }
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(tu["input"])}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockDelta": {"delta": {"text": item.get("text", "")}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}

        yield {
            "metadata": {
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "metrics": {"latencyMs": 0},
            }
        }


# ---------------------------------------------------------------------------
# Storm loop
# ---------------------------------------------------------------------------

def _triage_class_names() -> set[str]:
    """Known triage class names, tolerating TRIAGE_CLASSES being a set, list, or dict."""
    fallback = {"page_now", "investigate", "suppress_noise", "false_positive"}
    try:
        names = list(TRIAGE_CLASSES)
    except TypeError:
        return fallback
    if names and all(isinstance(n, str) for n in names):
        return set(names) | fallback
    return fallback


def _render_alert(alert: dict) -> str:
    labels = alert.get("labels", {})
    lines = [
        f"ALERT {alert['id']} [{labels.get('severity', '?')}] "
        f"{alert['service']} (namespace: {labels.get('namespace', '?')}, "
        f"team: {labels.get('team', '?')})",
        f"Title: {alert['title']}",
        f"Description: {alert['description']}",
        "Signals: " + ", ".join(f"{k}={v}" for k, v in alert.get("signals", {}).items()),
    ]
    return "\n".join(lines)


def _ticket_input(alert: dict) -> dict:
    # Matches the sibling tools' signature: open_incident_ticket(title, severity, summary).
    labels = alert.get("labels", {})
    return {
        "title": f"{alert['id']}: {alert['title']}",
        "severity": labels.get("severity", "P3"),
        "summary": alert["description"],
    }


def _mock_script_for_alert(alert: dict, triage_class: str) -> list[dict]:
    """Build the mock model's scripted responses for one alert.

    The checkout-api P1 is the guardrail demo: the scripted model first
    tries to restart in "staging" — a namespace invented out of thin air
    that the alert never mentions — and the guardrail denies it. The agent
    then falls back to opening an incident ticket with the alert's own
    details (grounded, so it proceeds) instead of forcing the restart.
    Every other alert gets a single grounded ticket call plus a summary.
    """
    aid = alert["id"].replace("ALT-", "")
    if alert["service"] == "checkout-api" and triage_class == "page_now":
        return [
            {"tool_use": {"name": "restart_service", "id": f"tu-{aid}-bad",
                          "input": {"service": "checkout-api", "namespace": "staging"}}},
            {"tool_use": {"name": "open_incident_ticket", "id": f"tu-{aid}-ticket",
                          "input": _ticket_input(alert)}},
            {"text": ("Incident summary: checkout-api error-budget burn. "
                      "The agent tried to restart the service in namespace "
                      "'staging', which the guardrail denied — 'staging' "
                      "appears nowhere in the alert. It fell back to opening "
                      "an incident ticket with the alert's own details instead "
                      "of forcing the restart. Human follow-up: confirm the "
                      "deploy that started the 5xx climb before re-enabling "
                      "traffic.")},
        ]
    return [
        {"tool_use": {"name": "open_incident_ticket", "id": f"tu-{aid}-ticket",
                      "input": _ticket_input(alert)}},
        {"text": (f"Incident summary: {alert['service']} ({alert['id']}). "
                  f"Triaged as {triage_class}; incident ticket opened with the "
                  f"alert's own details. Nothing was changed on any real system.")},
    ]


def _collect_tool_calls(agent: Agent) -> tuple[list[dict], list[dict]]:
    """Pull tool calls and guardrail denials out of the agent's history."""
    calls: list[dict] = []
    denied: list[dict] = []
    for message in agent.messages:
        for block in message.get("content", []):
            if "toolUse" in block:
                calls.append({"name": block["toolUse"]["name"],
                              "input": block["toolUse"].get("input", {})})
            if "toolResult" in block:
                result = block["toolResult"]
                for item in result.get("content", []):
                    text = item.get("text", "") if isinstance(item, dict) else ""
                    if text.startswith("DENIED:"):
                        denied.append({"tool_use_id": result.get("toolUseId"),
                                       "reason": text[len("DENIED:"):].strip()})
    return calls, denied


def strip_thinking(text: str) -> str:
    """Remove <thinking>...</thinking> blocks that some Bedrock models emit."""
    return _THINKING_RE.sub("", text).strip()


def _final_text(result) -> str:
    message = getattr(result, "message", {}) or {}
    parts = []
    content = message.get("content", []) if isinstance(message, dict) else []
    for block in content:
        if isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    return strip_thinking("\n".join(parts))


def ask_human(prompt: str, mode: str | None = None) -> bool:
    """Answer one guardrail Confirm. Returns True to approve the tool call."""
    mode = (mode or GUARDRAIL_CONFIRM)
    if mode == "approve":
        print(f"  guardrail confirm (auto-approved, GUARDRAIL_CONFIRM=approve): {prompt}", flush=True)
        return True
    if mode == "deny" or not sys.stdin.isatty():
        why = "GUARDRAIL_CONFIRM=deny" if mode == "deny" else "no terminal to ask"
        print(f"  guardrail confirm (declined, {why}): {prompt}", flush=True)
        return False
    answer = input(f"  GUARDRAIL: {prompt} ").strip().lower()
    return answer in {"y", "yes"}


def run_with_confirmations(agent: Agent, prompt: Any, asker=ask_human) -> tuple[Any, list[dict]]:
    """Invoke the agent; answer each guardrail Confirm and resume until it finishes.

    A Strands Confirm without a preset response pauses the agent with an
    interrupt. This loop asks the human, passes the answer back as an
    interruptResponse, and resumes. An approved call runs; a declined one is
    cancelled and the model is told, so it can choose another step.
    """
    result = agent(prompt)
    decisions: list[dict] = []
    for _ in range(MAX_CONFIRM_ROUNDS):
        if getattr(result, "stop_reason", "") != "interrupt" or not result.interrupts:
            break
        responses = []
        for intr in result.interrupts:
            approved = bool(asker(str(intr.reason)))
            decisions.append({"prompt": str(intr.reason), "approved": approved})
            responses.append({"interruptResponse": {"interruptId": intr.id,
                                                    "response": "yes" if approved else "no"}})
        result = agent(responses)
    return result, decisions


def run_storm(alerts: list[dict] | None = None,
              triage: Any | None = None,
              model: Any | None = None) -> list[dict]:
    """Run the full storm: triage every alert, act per class, summarize.

    Returns one result dict per alert with id, service, triage class,
    confidence, outcome, summary, tool calls, and any guardrail denials.
    """
    if not _SIBLINGS_OK:
        raise ImportError(
            "oncall_decider sibling modules are not importable yet "
            f"(decider_client/tools/interventions): {_SIBLING_IMPORT_ERROR}"
        )

    alerts = alerts if alerts is not None else build_storm()

    if model is None:
        model, mode = get_model()
        mock_mode = mode == "mock"
    else:
        mock_mode = isinstance(model, MockGenerativeModel)
        mode = "mock" if mock_mode else "bedrock"

    if triage is None:
        decider = None
        try:
            decider = get_client()
        except Exception as exc:
            logger.warning("get_client() failed (%s); TriageDecider gets decider=None.", exc)
        triage = TriageDecider(decider=decider, confirm_threshold=0.60)

    known = _triage_class_names()
    results: list[dict] = []

    for alert in alerts:
        decision = triage.decide(alert)
        cls = decision.get("class", "investigate")
        confidence = float(decision.get("confidence", 0.0))
        action = decision.get("action", "auto")

        print(f"{alert['id']} {alert['service']}: {cls} (confidence {confidence:.2f})", flush=True)

        result: dict = {
            "id": alert["id"],
            "service": alert["service"],
            "triage_class": cls,
            "confidence": confidence,
            "triage_reason": decision.get("reason", ""),
            "action": action,
            "tool_calls": [],
            "denied_calls": [],
            "summary": "",
            "summary_source": "bedrock" if mode == "bedrock" else "mock-template",
        }

        if action == "confirm":
            result["outcome"] = "needs human"
            result["summary"] = (
                f"Low-confidence triage ({confidence:.2f}); routed to a human "
                f"instead of acting. Decider said: {decision.get('reason', 'no reason given')}"
            )
            results.append(result)
            continue

        if cls not in known:
            result["outcome"] = "unrecognized triage class"
            result["summary"] = f"Triage returned unknown class {cls!r}; no action taken."
            results.append(result)
            continue

        if cls in ("suppress_noise", "false_positive"):
            result["outcome"] = "no action"
            why = ("known noise pattern" if cls == "suppress_noise"
                   else "self-resolved before triage")
            result["summary"] = f"Suppressed ({why}); no tools run, nothing paged."
            results.append(result)
            continue

        # page_now and investigate both run an agent; page_now gets the guardrail.
        if cls == "page_now":
            agent_model = (MockGenerativeModel(script=_mock_script_for_alert(alert, cls))
                           if mock_mode else model)
            guardrail = RemediationGuardrail(alert)
            agent = Agent(
                tools=[restart_service, scale_deployment, silence_alert, open_incident_ticket],
                interventions=[guardrail],
                model=agent_model,
            )
        else:  # investigate: plain agent, grounded ticket call, no guardrail needed
            agent_model = (MockGenerativeModel(script=_mock_script_for_alert(alert, cls))
                           if mock_mode else model)
            agent = Agent(tools=[open_incident_ticket], model=agent_model)

        prompt = _render_alert(alert) + "\nAll tools are dry-run. Decide on remediation and execute it."
        tickets_before = len(RECORDED_TICKETS)
        agent_result, confirmations = run_with_confirmations(agent, prompt)
        interrupted = getattr(agent_result, "stop_reason", "") == "interrupt"
        result["confirmations"] = confirmations
        if cls == "page_now":
            result["guardrail_checks"] = list(guardrail.records)

        tool_calls, denied = _collect_tool_calls(agent)
        result["tool_calls"] = tool_calls
        result["denied_calls"] = denied
        for d in denied:
            print(f"  guardrail denied a call: {d['reason']}", flush=True)
        for c in confirmations:
            verdict = "approved" if c["approved"] else "declined"
            print(f"  human {verdict} a guardrail confirm", flush=True)
        if interrupted:
            print(f"  agent still waiting after {MAX_CONFIRM_ROUNDS} confirm rounds; stopped", flush=True)

        if mode == "bedrock":
            result["summary"] = _final_text(agent_result) or "(model returned no text)"
        elif confirmations:
            approved = sum(c["approved"] for c in confirmations)
            result["summary"] = (
                f"[mock summary] {alert['service']}: the guardrail denied "
                f"{len(denied)} ungrounded call(s) and asked a human about "
                f"{len(confirmations)} uncertain one(s) ({approved} approved). "
                f"No real systems touched."
            )
        elif tool_calls:
            last = tool_calls[-1]
            result["summary"] = (
                f"[mock summary] {alert['service']}: ran {last['name']} "
                f"with grounded args; {len(denied)} call(s) denied by the guardrail. "
                "No real systems touched."
            )
        else:
            result["summary"] = f"[mock summary] {alert['service']}: no tool calls executed."

        declined = sum(1 for c in confirmations if not c["approved"])
        if interrupted:
            result["outcome"] = "stopped: unanswered guardrail confirm"
        elif declined:
            result["outcome"] = f"remediated (dry-run), human declined {declined} call(s)"
        elif cls == "page_now":
            result["outcome"] = "remediated (dry-run)"
        else:
            result["outcome"] = "ticket opened (dry-run)"

        # Optional layer (off unless SNOW_* is configured): if the agent
        # actually filed a ticket during this turn, a human validates the
        # exact payload and only then is a real ServiceNow incident made.
        new_tickets = RECORDED_TICKETS[tickets_before:]
        if new_tickets:
            maybe_create_validated_incident(result, new_tickets[-1])

        results.append(result)

    return results


if __name__ == "__main__":
    for r in run_storm():
        print(f"{r['id']}: {r['outcome']}")
