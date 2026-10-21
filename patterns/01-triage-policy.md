# Pattern 01: Triage Policy — "which class does this alert belong to?"

This is the oldest decision in on-call: an alert fires, and somebody has to
say what kind of thing it is. "Page someone right now" is a very different
answer from "file a ticket for Monday". Humans do it by pattern-matching
alerts against rules they've built up over years. This pattern turns that
pattern-matching into a small, local decision model call so your agent can
triage an alert storm without phoning a large model for every single alert.

## What decision it makes

One alert goes in, one triage class comes out. That's it — no prose, no
explanation paragraph, just a choice plus a confidence score. It fits a
decision model because the call is:

- **Bounded**: the answer must be one of a fixed set, nothing else is legal.
- **Repeated**: a storm is the same question asked 8, 80, 800 times.
- **Confidence-sensitive**: a low-confidence triage should go to a human,
  and you can't do that without a real probability, not a vibe.

If any of those three break (open-ended answer, asked once, no threshold),
you don't need a decision model — just ask the big model.

## The exact question and option set

State is the alert rendered as text: id, service, title, description,
labels, and signals. The question is a typed `choice` question with this
exact wording:

> **Which triage class does this monitoring alert belong to?**

```python
from oncall_decider.decider_client import get_client
from oncall_decider.interventions import triage_state_and_questions

client = get_client()
alert = {...}  # keys: id, service, title, description, labels, signals
state, questions = triage_state_and_questions(alert)

result = client.decide(state, {"triage": questions["triage"]})
# result == {"triage": {"decision": ..., "confidence": ..., "distribution": {...}}}
```

`questions["triage"]` has the shape:

```python
{
    "type": "choice",
    "question": "Which triage class does this monitoring alert belong to?",
    "options": {
        "page_now": (
            "wake someone up now: customer-facing outage, error budget "
            "burning fast, or a crashlooping service"
        ),
        "investigate": (
            "needs an engineer soon but not a page: latency regression, "
            "growing backlog, certificate expiring"
        ),
        "suppress_noise": (
            "known noisy or low-value alert: high-but-stable disk usage, "
            "flapping monitor"
        ),
        "false_positive": (
            "self-resolved or canary blip with no customer impact"
        ),
    },
}
```

Copy the keys verbatim — the model picks among these exact strings.

In this repo, what each class leads to is fixed in code (`agent.run_storm`):
`page_now` gets an agent with remediation tools behind the guardrail,
`investigate` gets an agent that can only open a ticket, and
`suppress_noise` / `false_positive` get no agent at all.

## The wiring: triage as a pre-agent policy gate

The gate sits in front of the agent and classifies the alert before any
agent is created, so no big-model budget is spent on alerts that don't need
one. It is a plain class rather than a Strands `InterventionHandler` on
purpose: its decision is "should an agent run at all, and with which
tools?", and that has to be made before the agent exists. (The guardrail in
[Pattern 02](02-remediation-guardrail.md) is the part that lives inside the
agent loop.) Low confidence routes to a human instead of letting a guess
through.

```python
import os
from oncall_decider.decider_client import get_client
from oncall_decider.interventions import triage_state_and_questions

CONFIRM_THRESHOLD = float(os.environ.get("TRIAGE_CONFIRM_THRESHOLD", "0.60"))


class TriageDecider:
    """Pre-agent policy gate: classify each alert; ask a human when unsure."""

    def __init__(self, confirm_threshold=CONFIRM_THRESHOLD):
        self.client = get_client()
        self.confirm_threshold = confirm_threshold

    def decide(self, alert):
        state, questions = triage_state_and_questions(alert)
        triage = self.client.decide(state, {"triage": questions["triage"]})["triage"]

        if triage["confidence"] < self.confirm_threshold:
            return {
                "class": triage["decision"],
                "confidence": triage["confidence"],
                "action": "confirm",  # route to a human instead of guessing
                "reason": (
                    f"Low-confidence triage ({triage['confidence']:.2f}) "
                    f"for alert {alert['id']}: '{alert['title']}'. "
                    f"Best guess: {triage['decision']}."
                ),
            }

        return {
            "class": triage["decision"],
            "confidence": triage["confidence"],
            "action": "auto",
            "reason": f"Confident triage ({triage['confidence']:.2f}).",
        }
```

The gate classifies the alert before the agent spends any big-model
budget on it. The caller checks `action`: `"auto"` means carry on,
`"confirm"` means stop and ask a human. The `distribution` field on the
raw result is there if you want to log near-misses (two classes
neck-and-neck usually means the alert text was ambiguous — worth a
human look).

## Picking the threshold

Start at **0.60**. Then run it against real alerts for a week and watch two
numbers:

1. **Confirm rate** — what fraction of alerts asked a human. If it's over
   ~20%, you're paging people instead of the model; lower the threshold or
   improve the option descriptions.
2. **Wrong guesses that sailed through** — confident decisions that turned
   out wrong. If any of them were page-now misclassifications, raise the
   threshold.

The direction of the mistake matters. A false "page now" wakes someone up;
a false "noise" hides an incident. If your team is more afraid of one than
the other, tune the threshold per class: demand higher confidence for the
dangerous answer, accept lower confidence for the safe one.

## Adapting it to another agent

Swap the option set, keep the shape. The pattern is:

```
state (text) + "which class?" + fixed options → (choice, confidence)
```

- Routing support tickets? Options become `billing`, `bug`, `how-to`, `spam`.
- Severity scoring? Options become `sev1` … `sev4`.
- Anything where the answer is "pick one of N, and tell me how sure you are"
  works, and the gate (confirm below the threshold, act above it) transfers
  unchanged. The only part that needs real work is
  writing option descriptions that the model can't misread — that's where
  your domain knowledge goes.
