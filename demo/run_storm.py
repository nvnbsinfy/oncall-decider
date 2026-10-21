"""Scripted storm demo entry point.

Runs the full eight-alert storm through triage and the agent loop,
prints a footer triage table, exits 0. From the repo root:

    python -m demo
    python demo/run_storm.py --log bedrock-run.txt   # also save a UTF-8 transcript

Guardrail confirms are asked at the terminal. Set GUARDRAIL_CONFIRM=approve
or =deny for an unattended run.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the package importable no matter where we are invoked from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from oncall_decider.agent import run_storm  # noqa: E402


class _Tee:
    """Write to the console and to a UTF-8 file at the same time."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return self.stream.isatty()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the scripted alert storm.")
    ap.add_argument("--log", help="also write the transcript to this file (UTF-8)")
    args = ap.parse_args(argv)
    if args.log:
        fh = open(args.log, "w", encoding="utf-8", newline="\n")
        sys.stdout = _Tee(sys.__stdout__, fh)
        sys.stderr = _Tee(sys.__stderr__, fh)
    try:
        return _run()
    finally:
        if args.log:
            sys.stdout.flush()
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
            fh.close()


def _run() -> int:
    print("On-Call Decider — storm demo")
    print("Decider 2B runs triage and the remediation guardrail locally.")
    print("The generative model used for this run is named on the next line:")
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
    confirms = [c for r in results for c in r.get("confirmations", [])]
    approved = sum(c["approved"] for c in confirms)
    checks = [g for r in results for g in r.get("guardrail_checks", [])]
    print(f"{len(results)} alerts triaged. Guardrail: {len(checks)} call(s) checked, "
          f"{denied} denied as ungrounded, {len(confirms)} sent to a human "
          f"({approved} approved, {len(confirms) - approved} declined).")
    if checks:
        print()
        print("Guardrail checks")
        print("-" * 88)
        for g in checks:
            score = "  -  " if g["score"] is None else f"{g['score']:.2f}"
            args_txt = ", ".join(f"{k}={v!r}" for k, v in g["args"].items())
            print(f"{g['action']:<9}{score:<7}{g['tool']}({args_txt[:60]})")
        print("-" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
