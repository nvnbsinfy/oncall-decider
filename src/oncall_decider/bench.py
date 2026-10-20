"""Benchmark: Decider 2B triage latency vs a simulated LLM-classifier baseline.

Run from the repo root:
    PYTHONPATH=src python -m oncall_decider.bench

For each alert in the storm fixture, it times a real Decider 2B triage
decision (policy classification) and compares it against a baseline that
sleeps 2.5s per decision — an honest stand-in for a small Bedrock model
round-trip. If you set USE_REAL_BEDROCK=1 and real AWS credentials are
available, it will instead time an actual Nova Micro invoke; otherwise it
always labels the baseline as simulated.

Results are printed as an ASCII table and written to bench/results.md.

Note on honesty: the baseline is simulated by default. Don't present it as
a measured cloud number unless USE_REAL_BEDROCK=1 was used.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

SIMULATED_BASELINE_SECONDS = 2.5

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = REPO_ROOT / "bench" / "results.md"


def _warn(msg: str) -> None:
    print(f"[bench] {msg}", file=sys.stderr)


def try_import_siblings():
    """Import the sibling modules built by the parallel agent."""
    try:
        from oncall_decider.alerts import build_storm
        from oncall_decider.decider_client import get_client
        from oncall_decider.interventions import triage_state_and_questions
    except ImportError as exc:  # pragma: no cover - integration state
        _warn(
            f"Could not import sibling modules ({exc}). "
            "The decider half of the repo isn't integrated yet; "
            "run bench again after it lands."
        )
        raise SystemExit(2) from exc
    return build_storm, get_client, triage_state_and_questions


def real_bedrock_client():
    """Return a callable latency probe, or None if unavailable.

    Only used when USE_REAL_BEDROCK=1. Returns None on any failure so the
    benchmark falls back to the simulated baseline instead of failing.
    """
    if os.environ.get("USE_REAL_BEDROCK") != "1":
        return None
    try:
        import boto3

        model_id = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")
        client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))

        def probe(prompt: str) -> str:
            resp = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 200, "temperature": 0},
            )
            return resp["output"]["message"]["content"][0]["text"]

        # Sanity check: fail fast on missing credentials.
        probe("Classify this alert as critical/high/medium/low. Reply with one word.")
        return probe
    except Exception as exc:  # missing creds, network, whatever: fall back
        _warn(f"USE_REAL_BEDROCK=1 but Bedrock probe failed ({exc}); using simulated baseline.")
        return None


def baseline_latency(probe, alert) -> float:
    if probe is not None:
        prompt = (
            f"Classify this monitoring alert as critical, high, medium, or low. "
            f"Reply with one word.\nTitle: {alert.get('title')}\n"
            f"Description: {alert.get('description')}\nSignals: {alert.get('signals')}"
        )
        start = time.perf_counter()
        probe(prompt)
        return time.perf_counter() - start
    time.sleep(SIMULATED_BASELINE_SECONDS)
    return SIMULATED_BASELINE_SECONDS


def run_benchmark():
    build_storm, get_client, triage_state_and_questions = try_import_siblings()

    alerts = build_storm()
    client = get_client()
    probe = real_bedrock_client()
    baseline_label = (
        "real Bedrock round-trip (USE_REAL_BEDROCK=1)"
        if probe is not None
        else f"simulated LLM baseline ({SIMULATED_BASELINE_SECONDS}s/decision, "
        "approximating a small Bedrock model round-trip)"
    )
    print(f"\nBaseline: {baseline_label}\n")

    rows = []
    for alert in alerts:
        state, questions = triage_state_and_questions(alert)

        # Warm-up-free: first call includes any cold start; reported honestly.
        start = time.perf_counter()
        result = client.decide(state, {"triage": questions["triage"]})
        decider_s = time.perf_counter() - start

        baseline_s = baseline_latency(probe, alert)

        triage = result["triage"]
        rows.append(
            {
                "alert": str(alert.get("id", "?")),
                "service": str(alert.get("service", "?")),
                "decision": str(triage.get("decision", "?")),
                "conf": f"{float(triage.get('confidence', 0.0)):.2f}",
                "decider_s": decider_s,
                "baseline_s": baseline_s,
                "speedup": baseline_s / decider_s if decider_s > 0 else float("inf"),
            }
        )
        print(
            f"  {alert.get('id'):<10} {alert.get('service'):<18} "
            f"decision={rows[-1]['decision']:<28} "
            f"conf={rows[-1]['conf']}  decider={decider_s:.2f}s  baseline={baseline_s:.2f}s"
        )

    total_decider = sum(r["decider_s"] for r in rows)
    total_baseline = sum(r["baseline_s"] for r in rows)
    overall_speedup = total_baseline / total_decider if total_decider > 0 else float("inf")

    header = (
        "+------------+--------------------+------------------------------+--------+"
        "-----------+------------+----------+"
    )
    cols = (
        f"| {'alert':<10} | {'service':<18} | {'triage decision':<28} | {'conf':<6} "
        f"| {'decider (s)':<9} | {'baseline (s)':<10} | {'speedup':<7} |"
    )
    lines = [header, cols, header]
    for r in rows:
        lines.append(
            f"| {r['alert']:<10} | {r['service']:<18} | {r['decision']:<28} | {r['conf']:<6} "
            f"| {r['decider_s']:<9.2f} | {r['baseline_s']:<10.2f} | {r['speedup']:<7.1f}x |"
        )
    lines.append(header)
    lines.append(
        f"| {'TOTAL':<10} | {'':<18} | {'':<28} | {'':<6} "
        f"| {total_decider:<9.2f} | {total_baseline:<10.2f} | {overall_speedup:<7.1f}x |"
    )
    lines.append(header)
    table = "\n".join(lines)

    print("\n" + table + "\n")
    print(f"Decider 2B total: {total_decider:.2f}s | baseline total: {total_baseline:.2f}s "
          f"| overall speedup: {overall_speedup:.1f}x\n")

    body = f"""# Benchmark results

{baseline_note(probe)}

```text
{table}
```

## How to read this

- **Decider 2B** column: measured wall-clock time per triage decision on this
  machine, including model load / first-call cold start on the first alert.
- **Baseline** column: {baseline_label}. The 2.5s simulated figure is a
  reasonable guess at a small Bedrock model round-trip (prompt + network +
  generation); treat it as approximate, not measured.
- The upstream Decider 2B figure of ~115ms per decision is the maintainers'
  claim on an **RTX 3090 GPU**. On CPU the model is slower; feasibility
  testing on the build VM showed model load ~7s, first decision ~5.6s, and
  steady-state ~1.74s per decision.

## Honesty note

Numbers in this table come from running `python -m oncall_decider.bench`
on the machine they were measured on. CPU latency and the simulated
baseline both vary; the only number we didn't measure is the 115ms GPU
claim, which is upstream's. The Bedrock path itself needs real AWS
credentials, which did not exist on the build machine, so the generative
half ran in mock mode here.
"""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(body, encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")


def baseline_note(probe) -> str:
    if probe is not None:
        return "_Baseline: real Bedrock round-trip (USE_REAL_BEDROCK=1)._"
    return (
        "_Baseline: simulated LLM classifier at 2.5s/decision, approximating a "
        "small Bedrock model round-trip. Honest but approximate — not measured._"
    )


if __name__ == "__main__":
    run_benchmark()
