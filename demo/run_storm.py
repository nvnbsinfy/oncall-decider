"""Scripted storm demo entry point.

Runs the full eight-alert storm through triage and the agent loop,
prints a footer triage table, exits 0. From the repo root:

    python -m demo
"""

from __future__ import annotations

import os
import sys

# Make the package importable no matter where we are invoked from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from oncall_decider.agent import run_storm  # noqa: E402


def main() -> int:
    print("On-Call Decider — scripted storm demo (mock generative model)")
    print("Triage and the remediation guardrail run on Decider 2B; the")
    print("generative half is a scripted stand-in (no AWS credentials here).")
    print()

    results = run_storm()

    print()
    print("Triage table")
    print("-" * 88)
    print(f"{'alert':<10}{'service':<16}{'class':<16}{'conf':<7}{'outcome'}")
    print("-" * 88)
    for r in results:
        print(f"{r['id']:<10}{r['service']:<16}{r['triage_class']:<16}"
              f"{r['confidence']:<7.2f}{r['outcome']}")
    print("-" * 88)
    denied = sum(len(r["denied_calls"]) for r in results)
    print(f"{len(results)} alerts triaged, {denied} ungrounded call(s) denied by the guardrail.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
