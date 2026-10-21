"""Decision handlers for the on-call agent.

- TriageDecider: a pre-agent policy gate. It classifies an alert into a
  fixed triage class with the local decision model and decides whether an
  agent is created at all, so it runs before any big-model budget is spent.
  It is a plain class, not an InterventionHandler: the decision it makes is
  "should an agent run?", which has to happen before the agent exists.
- RemediationGuardrail (a Strands InterventionHandler): asks "are these tool args grounded?" right before a
  remediation tool runs; denies hallucinated args, confirms on uncertainty,
  proceeds when grounded.

Both handlers defer to get_client() (the lazy DeciderClient singleton) so
the model loads once, on first use.
"""

from __future__ import annotations

from typing import Any

from strands.interventions import Confirm, Deny, InterventionHandler, Proceed

from .decider_client import DeciderClient, get_client

# The four mutually exclusive triage classes. Wording is deliberate: the
# model picks among these descriptions, so each one must be unmistakable.
TRIAGE_CLASSES = ("page_now", "investigate", "suppress_noise", "false_positive")

TRIAGE_QUESTION = "Which triage class does this monitoring alert belong to?"

TRIAGE_OPTIONS = {
    "page_now": (
        "wake someone up now: customer-facing outage, error budget burning "
        "fast, or a crashlooping service"
    ),
    "investigate": (
        "needs an engineer soon but not a page: latency regression, growing "
        "backlog, certificate expiring"
    ),
    "suppress_noise": (
        "known noisy or low-value alert: high-but-stable disk usage, "
        "flapping monitor"
    ),
    "false_positive": (
        "self-resolved or canary blip with no customer impact"
    ),
}


def render_alert(alert: dict[str, Any]) -> str:
    """Render an alert dict (keys: id, service, title, description, labels,
    signals) into the state string the decision model reads."""
    return (
        f"Alert {alert.get('id', '?')} [{alert.get('service', '?')}]: "
        f"{alert.get('title', '?')}\n"
        f"Description: {alert.get('description', '')}\n"
        f"Labels: {alert.get('labels', '')}\n"
        f"Signals: {alert.get('signals', '')}"
    )


def triage_state_and_questions(alert: dict[str, Any]) -> tuple[str, dict]:
    """Build the (state, questions) pair for alert triage.

    The question spec matches the dict format DeciderClient.decide expects,
    so the caller can pass it straight through.
    """
    state = render_alert(alert)
    questions = {
        "triage": {
            "type": "choice",
            "question": TRIAGE_QUESTION,
            "options": dict(TRIAGE_OPTIONS),
        }
    }
    return state, questions


class TriageDecider:
    """Pre-agent policy gate: classify each alert; route low-confidence triage to a human."""

    name = "triage-decider"

    def __init__(
        self,
        decider: DeciderClient | None = None,
        confirm_threshold: float = 0.60,
    ) -> None:
        self.decider = decider or get_client()
        self.confirm_threshold = confirm_threshold

    def decide(self, alert: dict) -> dict:
        """Classify one alert.

        Returns {"class", "confidence", "action", "reason"} where action is
        "confirm" when confidence falls below the threshold, else "auto".
        """
        state, questions = triage_state_and_questions(alert)
        triage = self.decider.decide(state, questions)["triage"]
        cls, conf = triage["decision"], triage["confidence"]
        if conf < self.confirm_threshold:
            action, reason = "confirm", (
                f"Low-confidence triage ({conf:.2f}) for alert "
                f"{alert.get('id', '?')}: '{alert.get('title', '?')}'. "
                f"Best guess: {cls}. A human should confirm the class."
            )
        else:
            action, reason = "auto", (
                f"Confident triage ({conf:.2f}): alert {alert.get('id', '?')} "
                f"classified as {cls}."
            )
        return {"class": cls, "confidence": conf, "action": action, "reason": reason}



# Note on wording: the pattern doc phrases this as "Are the tool's argument
# values grounded in facts from the alert payload?". Measured against the
# real model, that phrasing scores a blatant hallucination ("staging", never
# mentioned) at 0.41 -- the uncertain zone, so it would Confirm instead of
# Deny. The version below is the same checking question in the shape decision
# models answer best ("does value X appear in facts Y?"): measured 0.28 on
# the same negative (deny zone) and 0.67 on the grounded positive
# (proceed zone). Same semantics, sharper boundary.
GROUNDING_QUESTION = (
    "Does every value in the proposed tool call appear in the alert payload "
    "above? Answer yes only if all of them do; answer no if any value is invented."
)


# Arguments that are the agent's own choices, not facts it should copy from
# the alert. A replica count or a silence duration is a plan; a ticket title
# or summary is prose the agent writes. Checking these for "does it appear in
# the alert?" flags correct plans as hallucinations (and, worse, a score near
# the boundary lets invented values through by luck). They are left out of
# the grounding check; identifying values (service, namespace, alert_id,
# severity) are always checked.
PLANNING_ARGS: dict[str, set[str]] = {
    "scale_deployment": {"replicas"},
    "silence_alert": {"duration_minutes", "reason"},
    "open_incident_ticket": {"title", "summary"},
}


class RemediationGuardrail(InterventionHandler):
    """Block or confirm remediation tool calls whose args aren't in the alert.

    Three zones on the grounding score (noul = P("args are grounded")):
      score < deny_threshold    -> Deny (confidently hallucinated args)
      score < confirm_threshold -> Confirm (uncertain, ask a human)
      otherwise                 -> Proceed (grounded)
    """

    name = "remediation-guardrail"

    def __init__(
        self,
        alert: dict,
        decider: DeciderClient | None = None,
        deny_threshold: float = 0.35,
        confirm_threshold: float = 0.60,
        planning_args: dict[str, set[str]] | None = None,
    ) -> None:
        self.alert = alert
        self.planning_args = PLANNING_ARGS if planning_args is None else planning_args
        self.records: list[dict] = []  # one entry per checked call, for the demo table
        # Strands re-runs before_tool_call when an interrupted (Confirm) call is
        # resumed. Reuse the first decision for that toolUseId so the model is
        # asked once and the call is recorded once.
        self._decided: dict[str, Any] = {}
        self.decider = decider or get_client()
        self.deny_threshold = deny_threshold
        self.confirm_threshold = confirm_threshold

    # -- helpers ---------------------------------------------------------

    def _state_text(self, tool_name: str, tool_args: dict) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
        return (
            f"{render_alert(self.alert)}\n\n"
            f"Proposed tool call: {tool_name}({args})"
        )

    @staticmethod
    def _ungrounded_values(tool_args: dict, alert_text: str) -> list[str]:
        """Argument values that don't appear anywhere in the alert text.

        A cheap, explainable cross-check used to name names in the Deny
        reason. (The actual decision comes from the model, not this check.)
        """
        haystack = alert_text.lower()
        bad: list[str] = []
        for key, value in tool_args.items():
            if value is None:
                continue
            needle = str(value).strip().lower()
            if not needle or needle not in haystack:
                bad.append(f"{key}={value!r}")
        return bad

    # -- the guardrail ----------------------------------------------------

    def before_tool_call(self, event, **kwargs):  # noqa: ANN001, ANN202
        use_id = event.tool_use.get("toolUseId")
        if use_id is not None and use_id in self._decided:
            return self._decided[use_id]
        action = self._check(event)
        if use_id is not None:
            self._decided[use_id] = action
        return action

    def _check(self, event):  # noqa: ANN001, ANN202
        tool_name = event.tool_use["name"]
        all_args = dict(event.tool_use.get("input") or {})
        exempt = self.planning_args.get(tool_name, set())
        tool_args = {k: v for k, v in all_args.items() if k not in exempt}
        if not tool_args:
            self.records.append({"tool": tool_name, "args": all_args, "score": None, "action": "proceed"})
            return Proceed(reason=f"{tool_name}: only planning arguments, nothing to ground")

        state = self._state_text(tool_name, tool_args)
        result = self.decider.decide(
            state,
            {"grounded": {"type": "noul", "question": GROUNDING_QUESTION}},
        )["grounded"]
        score = result["noul"]
        alert_id = self.alert.get("id", "?")
        suspect = self._ungrounded_values(tool_args, render_alert(self.alert))

        def record(action: str) -> None:
            self.records.append({"tool": tool_name, "args": all_args, "score": score, "action": action})

        if score < self.deny_threshold:
            record("deny")
            # Confidently ungrounded: the call never runs. The reason goes
            # back to the model, so the agent can self-correct instead of
            # just being told no.
            named = f" Ungrounded argument(s): {', '.join(suspect)}." if suspect else ""
            return Deny(
                reason=(
                    f"Refusing {tool_name}: argument values are not grounded in "
                    f"alert {alert_id} (grounding score {score:.2f}).{named} "
                    f"Use only service names, namespaces, ids, and values that "
                    f"appear in the alert payload."
                )
            )

        if score < self.confirm_threshold:
            record("confirm")
            return Confirm(
                prompt=(
                    f"Uncertain grounding ({score:.2f}) for {tool_name} with args "
                    f"{all_args} on alert {alert_id}. Approve? [y/N]"
                ),
                reason=(
                    f"Guardrail is not confident that the arguments to "
                    f"{tool_name} come from alert {alert_id}."
                ),
            )

        record("proceed")
        return Proceed(reason=f"grounding score {score:.2f} for {tool_name}")
