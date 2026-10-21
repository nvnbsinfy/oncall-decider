"""Benchmark: the same triage decision made by Decider 2B and by a Bedrock LLM.

Run from the repo root:
    PYTHONPATH=src python -m oncall_decider.bench                      # simulated baseline
    USE_REAL_BEDROCK=1 PYTHONPATH=src python -m oncall_decider.bench   # measured Bedrock

With USE_REAL_BEDROCK=1 and AWS credentials, the baseline is a real Bedrock
Converse call (BEDROCK_MODEL_ID, default Nova Micro) asked the SAME question
with the SAME four classes Decider 2B gets, so latency, token use and
agreement are all measured on one decision. Without it, the baseline sleeps
2.5 s per decision and is labelled as simulated everywhere it appears.

Model load and one warm-up decision happen before timing starts, and are
reported on their own line instead of being mixed into the per-alert numbers.

Results are printed and written to bench/results.md.
"""

from __future__ import annotations

import os
import platform
import statistics
import sys
import time
from pathlib import Path

SIMULATED_BASELINE_SECONDS = 2.5

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = REPO_ROOT / "bench" / "results.md"


def _warn(msg: str) -> None:
    print(f"[bench] {msg}", file=sys.stderr)


def baseline_prompt(alert: dict) -> str:
    from oncall_decider.interventions import TRIAGE_OPTIONS, TRIAGE_QUESTION, render_alert

    options = "\n".join(f"- {name}: {desc}" for name, desc in TRIAGE_OPTIONS.items())
    return (
        f"{TRIAGE_QUESTION}\nClasses:\n{options}\n\n{render_alert(alert)}\n\n"
        "Reply with exactly one class name and nothing else."
    )


def parse_class(text: str) -> str:
    from oncall_decider.interventions import TRIAGE_CLASSES

    low = text.lower()
    hits = [c for c in TRIAGE_CLASSES if c in low]
    return hits[0] if len(hits) == 1 else "unparsed"


def real_bedrock_client():
    """Return a probe(prompt) -> (text, in_tokens, out_tokens), or None.

    Only used when USE_REAL_BEDROCK=1. Returns None on any failure so the
    benchmark falls back to the simulated baseline instead of failing.
    """
    if os.environ.get("USE_REAL_BEDROCK") != "1":
        return None
    try:
        import boto3

        model_id = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")
        client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))

        def probe(prompt: str) -> tuple[str, int, int]:
            resp = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 20, "temperature": 0},
            )
            usage = resp.get("usage", {})
            text = resp["output"]["message"]["content"][0]["text"]
            return text, int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))

        probe("Reply with the single word: ready")  # fail fast, and warm the connection
        probe.model_id = model_id  # type: ignore[attr-defined]
        return probe
    except Exception as exc:  # missing creds, network, model access: fall back
        _warn(f"USE_REAL_BEDROCK=1 but Bedrock probe failed ({exc}); using simulated baseline.")
        return None


def run_benchmark() -> None:
    from oncall_decider.alerts import build_storm
    from oncall_decider.decider_client import get_client
    from oncall_decider.interventions import triage_state_and_questions

    alerts = build_storm()
    client = get_client()
    probe = real_bedrock_client()
    real = probe is not None
    baseline_label = (
        f"measured Bedrock Converse call ({probe.model_id}), same question and classes"
        if real else f"SIMULATED LLM baseline ({SIMULATED_BASELINE_SECONDS}s sleep per decision, not measured)"
    )
    print(f"\nBaseline: {baseline_label}\n")

    # Cold start, timed separately: model load + first decision.
    state, questions = triage_state_and_questions(alerts[0])
    t0 = time.perf_counter()
    client.decide(state, {"triage": questions["triage"]})
    cold_s = time.perf_counter() - t0

    rows = []
    for alert in alerts:
        state, questions = triage_state_and_questions(alert)
        t = time.perf_counter()
        triage = client.decide(state, {"triage": questions["triage"]})["triage"]
        decider_s = time.perf_counter() - t

        if real:
            t = time.perf_counter()
            text, tin, tout = probe(baseline_prompt(alert))
            base_s = time.perf_counter() - t
            base_cls = parse_class(text)
        else:
            time.sleep(SIMULATED_BASELINE_SECONDS)
            base_s, base_cls, tin, tout = SIMULATED_BASELINE_SECONDS, "-", 0, 0

        rows.append({
            "alert": alert["id"], "service": alert["service"], "decider": triage["decision"],
            "conf": float(triage["confidence"]), "decider_s": decider_s, "baseline": base_cls,
            "baseline_s": base_s, "in_tokens": tin, "out_tokens": tout,
        })
        r = rows[-1]
        print(f"  {r['alert']:<10} {r['service']:<16} decider={r['decider']:<15} conf={r['conf']:.2f} "
              f"{decider_s:6.2f}s | baseline={base_cls:<15} {base_s:6.2f}s")

    dec_times = [r["decider_s"] for r in rows]
    base_times = [r["baseline_s"] for r in rows]
    header = f"| {'alert':<9} | {'service':<15} | {'Decider 2B':<15} | {'conf':<4} | {'dec (s)':<7} | {'LLM':<15} | {'LLM (s)':<7} |"
    sep = "|" + "|".join("-" * (len(c)) for c in header.split("|")[1:-1]) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append(f"| {r['alert']:<9} | {r['service']:<15} | {r['decider']:<15} | {r['conf']:<4.2f} | "
                     f"{r['decider_s']:<7.2f} | {r['baseline']:<15} | {r['baseline_s']:<7.2f} |")
    table = "\n".join(lines)

    summary = [
        f"- Machine: {platform.system()} {platform.machine()}, Decider on `{client.device}`",
        f"- Decider cold start (model load + first decision, excluded below): {cold_s:.1f} s",
        f"- Decider warm, per decision: median {statistics.median(dec_times):.2f} s, "
        f"max {max(dec_times):.2f} s, total {sum(dec_times):.1f} s",
        f"- Baseline ({'measured' if real else 'simulated'}), per decision: median "
        f"{statistics.median(base_times):.2f} s, total {sum(base_times):.1f} s",
    ]
    if real:
        agree = sum(1 for r in rows if r["baseline"] == r["decider"])
        confident = [r for r in rows if r["conf"] >= 0.60]
        agree_conf = sum(1 for r in confident if r["baseline"] == r["decider"])
        summary += [
            f"- Agreement with Bedrock: {agree}/{len(rows)} alerts; "
            f"{agree_conf}/{len(confident)} where Decider was confident (>= 0.60)",
            f"- Bedrock tokens per decision: {statistics.mean(r['in_tokens'] for r in rows):.0f} in / "
            f"{statistics.mean(r['out_tokens'] for r in rows):.0f} out (Decider: none billed)",
        ]
    print("\n" + table + "\n\n" + "\n".join(summary) + "\n")

    body = f"""# Benchmark results

_Baseline: {baseline_label}._

{table}

{chr(10).join(summary)}

## How to read this

- Both columns answer the same question ("{questions['triage']['question']}")
  over the same four classes, one alert at a time.
- Decider times are wall-clock on the machine named above, after the model
  was loaded and warmed. Upstream's ~115 ms figure is for an RTX 3090 GPU; on
  CPU, expect seconds.
- {"The LLM column is a real Bedrock call: network round-trip plus generation." if real else
   "The LLM column is a fixed 2.5 s sleep, not a measurement. Re-run with USE_REAL_BEDROCK=1 to measure it."}
"""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(body, encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    run_benchmark()
