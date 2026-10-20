# Pattern 02: Remediation Guardrail — "are these tool args grounded?"

Here's the failure this pattern exists to prevent: the agent reads an
alert, decides on a remediation, and then calls a tool with arguments it
made up. A restart with the wrong pod name. A scaling action against a
service that doesn't exist. The agent's plan sounded right; the arguments
were hallucinations.

The fix is a second, much cheaper question asked right before the tool
runs: given the facts in the alert payload, do the tool's argument values
actually appear there — or did the agent invent them?

## What decision it checks

`before_tool_call` fires with the tool name and the arguments the agent
proposed. The guardrail builds a state string from two things:

1. The alert payload (id, service, title, description, labels, signals) —
   the facts.
2. The proposed tool name and its argument values — the claim.

Then it asks one typed `noul` (yes/no) question. The exact wording matters —
I learned this the hard way. My first version was:

> Are the tool's argument values grounded in facts from the alert payload?

Against the real model, that phrasing scored a blatant hallucination (a
namespace the alert never mentions) at 0.41 — the uncertain zone, so it
would have asked a human instead of denying. The version below asks the
same thing as a checking question ("does value X appear in facts Y?"),
which is the shape decision models answer best:

> **Does every value in the proposed tool call appear in the alert payload
> above? Answer yes only if all of them do; answer no if any value is
> invented.**

Measured: 0.28 on the hallucinated call (deny zone), 0.67 on the grounded
call (proceed zone). Same semantics, much sharper boundary. If you adapt
this pattern, budget an hour to probe 4–5 phrasings against your own
examples — it moves the numbers more than any threshold tweak.

```python
result = client.decide(state, {
    "grounded": {
        "type": "noul",
        "question": (
            "Does every value in the proposed tool call appear in the alert "
            "payload above? Answer yes only if all of them do; answer no if "
            "any value is invented."
        ),
    }
})
# result == {"grounded": {"noul": <0..1>, "decision": bool, "confidence": ...}}
```

`noul` is a score from 0 to 1: low means "not grounded", high means
"grounded", middle means "can't tell".

## The wiring: three zones

```python
from strands.interventions import InterventionHandler, Proceed, Deny, Confirm
from oncall_decider.decider_client import get_client

DENY_BELOW = 0.35      # confidently ungrounded
CONFIRM_BELOW = 0.60   # uncertain


class RemediationGuardrail(InterventionHandler):
    """Block or confirm remediation tool calls that aren't grounded in the alert."""

    name = "remediation-guardrail"

    def __init__(self, alert):
        self.alert = alert
        self.client = get_client()

    def before_tool_call(self, event):
        tool_name = event.tool_use["name"]
        tool_args = dict(event.tool_use.get("input") or {})
        state = self._state_text(tool_name, tool_args)
        score = self.client.decide(state, {
            "grounded": {
                "type": "noul",
                "question": (
                    "Does every value in the proposed tool call appear in the "
                    "alert payload above? Answer yes only if all of them do; "
                    "answer no if any value is invented."
                ),
            }
        })["grounded"]["noul"]

        if score < DENY_BELOW:
            # Confidently ungrounded: stop the call and tell the model why.
            return Deny(
                reason=(
                    f"Refusing {tool_name}: argument values are not grounded "
                    f"in alert {self.alert['id']} (grounding score {score:.2f}). "
                    f"Use only service names, namespaces, ids, and values "
                    f"that appear in the alert payload."
                )
            )

        if score < CONFIRM_BELOW:
            # Uncertain: ask a human before anything runs.
            return Confirm(
                prompt=(
                    f"Uncertain grounding ({score:.2f}) for {tool_name} "
                    f"with args {tool_args}. Approve?"
                ),
                reason="Guardrail is not confident the args come from the alert.",
            )

        return Proceed(reason=f"grounding score {score:.2f} for {tool_name}")

    def _state_text(self, tool_name, tool_args):
        alert = self.alert
        return (
            f"Alert {alert['id']} [{alert['service']}]: {alert['title']}\n"
            f"Description: {alert['description']}\n"
            f"Labels: {alert['labels']}\nSignals: {alert['signals']}\n\n"
            f"Proposed tool call: {tool_name}({tool_args})"
        )
```

Three zones, three behaviors:

| Grounding score | Zone | Action |
|---|---|---|
| `< 0.35` | confidently ungrounded | **Deny** — the call never runs |
| `0.35 – 0.60` | uncertain | **Confirm** — a human approves or rejects |
| `≥ 0.60` | grounded | **Proceed** |

The Deny path matters more than it looks: `Deny(reason=...)` feeds the
reason string back into the model, so the agent gets to see *why* it was
blocked and can self-correct — propose args that actually come from the
alert — instead of just being told no. The guardrail isn't a wall; it's a
wall that talks back.

## Why this shape of question works

You're not asking the model "is this a good idea?" — that's open-ended and
opinion-shaped. You're asking "does value X appear in facts Y?" — a
checking question with a right answer. Decision models are at their best on
questions a careful human could grade in seconds. Groundedness is exactly
that: point at each argument value, point at the fact it came from. If you
can't point, it fails.

## Adapting it to any tool, any state

The pattern is:

```
state (facts) + proposed tool args (claim) → "grounded?"
```

Nothing about it is on-call specific. The two inputs change per use case:

- **Any tool**: database writes, shell commands, ticket updates, API calls.
  The question stays the same; only the proposed args change.
- **Any state**: swap the alert payload for whatever your facts are — a
  user request, a config file, the output of the last tool. The "facts"
  side is whatever you'd show a human reviewer and say "check against this".

Threshold guidance is the same as Pattern 01: start at deny `< 0.35`,
confirm `< 0.60`, then watch what gets blocked. If legit calls are getting
denied, your state text is probably missing the facts the agent legitimately
used — fix the state, not the threshold.
