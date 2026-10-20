"""Guardrail test: real Decider 2B, fabricated BeforeToolCall events.

Run: PYTHONPATH=src python tests/test_guardrail.py

The handler only reads event.tool_use, so a SimpleNamespace with that field
is enough to exercise before_tool_call without standing up an agent.

Cases (both use the REAL model -- the decisions are deterministic):
  1. NEGATIVE: tool args name a namespace ("staging") the alert never
     mentions -> the guardrail must Deny.
  2. POSITIVE: tool args only use values from the alert payload
     ("checkout-api", "prod") -> the guardrail must Proceed.
"""

import os
import sys
from types import SimpleNamespace

# Make the package importable when run as `python tests/test_guardrail.py`
# from the repo root without PYTHONPATH set.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from oncall_decider.interventions import RemediationGuardrail
from strands.interventions import Confirm, Deny, Proceed

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name} {detail}")
        failures.append(name)


ALERT = {
    "id": "ALT-20261018-001",
    "service": "checkout-api",
    "title": "Checkout API crashlooping in prod",
    "description": (
        "The checkout-api deployment in namespace prod has restarted 5 times "
        "in the last 10 minutes. Liveness probe failures. Error budget is "
        "burning fast and checkout errors are customer-facing."
    ),
    "labels": {"env": "prod", "team": "checkout", "namespace": "prod"},
    "signals": ["restart_count=5 in 10m", "liveness_probe_failing", "error_rate=12%"],
}


def make_event(tool_name, tool_input):
    return SimpleNamespace(tool_use={"name": tool_name, "input": tool_input})


def main():
    guardrail = RemediationGuardrail(ALERT)

    # 1. NEGATIVE: "staging" never appears in the alert payload.
    bad = make_event(
        "restart_service", {"service": "checkout-api", "namespace": "staging"}
    )
    action = guardrail.before_tool_call(bad)
    check("negative: handler returns a Deny", isinstance(action, Deny),
          f"got {type(action).__name__}")
    if isinstance(action, Deny):
        check("negative: reason names the ungrounded value",
              "staging" in action.reason.lower(), f"reason={action.reason!r}")
        print(f"      noul detail: {action.reason}")

    # 2. POSITIVE: every arg value comes straight from the alert payload.
    good = make_event(
        "restart_service", {"service": "checkout-api", "namespace": "prod"}
    )
    action = guardrail.before_tool_call(good)
    check("positive: handler returns a Proceed", isinstance(action, Proceed),
          f"got {type(action).__name__}: {getattr(action, 'reason', '')!r}")

    # 3. Handler contract: it must not be a Confirm in the positive case.
    check("positive: not a Confirm", not isinstance(action, Confirm))

    if failures:
        print(f"\n{len(failures)} FAILURE(S): {failures}")
        sys.exit(1)
    print("\nAll guardrail tests passed.")


if __name__ == "__main__":
    main()
