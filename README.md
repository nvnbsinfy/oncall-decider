# On-Call Decider

An on-call agent that triages alert storms with a small local decision
model and refuses to run remediation it can't ground. The triage policy and
the remediation guardrail both run on Decider 2B, locally, on CPU — the
large generative model only writes summaries, and it can be a mock.

Contest tags: **#strands-decider-2b** **#policy**

## How it works

```
alert stream
    |
    v
Decider 2B triage (policy: which class does this alert belong to?)
    |  confidence < 0.60 --> human Confirm
    v
Strands Agent (plans remediation)
    |
    v
RemediationGuardrail — Decider 2B grounding check:
    "Does every value in the proposed tool call appear in the alert payload
     above? Answer yes only if all of them do; answer no if any value is invented."
    |  noul < 0.35  --> Deny  (reason fed back so the model can self-correct)
    |  0.35 - 0.60  --> human Confirm
    |  >= 0.60      --> Proceed
    v
dry-run tools (nothing here touches real infrastructure)
    |
    v
Bedrock (or mock) writes the summary
```

Two small decisions do all the safety-critical work, and both run in about
two seconds on a CPU. The big model never classifies, never checks
arguments, never runs tools — it just explains what happened.

## Quickstart

```bash
# 1. Install. On machines with a tiny /tmp or a restrictive proxy:
export TMPDIR=$HOME/workspace/tmp
export HF_HUB_DISABLE_XET=1
pip install -r requirements.txt

# 2. Configure.
cp .env.example .env

# 3. Run the alert-storm demo.
python demo/run_storm.py

# 4. Benchmark Decider 2B triage against a simulated LLM baseline.
PYTHONPATH=src python -m oncall_decider.bench

# 5. Run the guardrail tests.
PYTHONPATH=src python tests/test_guardrail.py
```

The first run downloads the Decider 2B weights (~4.5 GB) from Hugging Face.
CPU inference needs ~4.5 GB of free RAM; the loader skips the bf16→fp32
torso upcast, without which it OOMs.

## The two decision patterns

The reusable recipes live in [`patterns/`](patterns/). They're written to
be lifted out and reused in any agent:

1. **[Triage Policy](patterns/01-triage-policy.md)** — one `choice` question
   ("Which triage class does this monitoring alert belong to?") answered
   per alert, wired in as a Strands `InterventionHandler`. Below 0.60
   confidence it asks a human to confirm instead of guessing.
2. **[Remediation Guardrail](patterns/02-remediation-guardrail.md)** — one
   `noul` question ("Does every value in the proposed tool call appear in
   the alert payload above?") asked in `before_tool_call`. Confidently
   ungrounded calls are denied with a reason the model can act on;
   uncertain ones ask a human; grounded ones proceed.

## Benchmark results

Measured on the build VM (CPU-only, no GPU, no optimized kernels), via
`PYTHONPATH=src python -m oncall_decider.bench`. Full table in
[`bench/results.md`](bench/results.md).

| what | number |
|---|---|
| model load | ~8s |
| triage decision, steady-state (4-way choice over a full alert payload) | ~4–7s |
| first decisions (includes warmup) | 12–22s |
| simulated LLM baseline (2.5s/decision) | 20s total for 8 alerts vs 77s for Decider 2B |

Read that honestly: on this CPU-only box, Decider 2B does **not** beat a
small LLM round-trip on raw latency for these long triage questions. (The
1.7s figure from feasibility testing was a short yes/no question; a 4-way
choice over a full alert payload is heavier work. Upstream's 115ms claim is
for an RTX 3090, which we don't have here.)

So I'm not claiming a latency win — the wins here are cost and shape:

- **$0 per decision.** No API, no tokens, no per-call charge. An alert storm
  of 800 alerts costs the same as 8.
- **Nothing leaves the machine.** Alert payloads stay local, which matters
  for incident data.
- **Calibrated confidence with every decision.** That's what lets the
  0.60 threshold route unsure triage to a human — you can't do that with a
  vibe.

The contest rubric counts a measured latency *or cost* improvement as a
bonus. Ours is cost. The numbers above are the receipts.

## Optional: human-validated ServiceNow incidents

Everything above is dry-run. One layer can be switched on to make exactly
one thing real: incident creation in ServiceNow — and only after a human
says so, per incident.

With `SNOW_ENABLED=1` and `SN_INSTANCE` / `SN_USER` / `SN_PASSWORD` set,
whenever the agent files a ticket the run stops, shows the exact payload
(title, severity, summary), and asks the human to choose: **[c]reate**
the incident in ServiceNow, or **[n]o incident**. Only an explicit
"create" calls the Table API, and the returned incident number lands in
the triage table. The incident's Caller is set to the API user the layer
connects with (looked up from `SN_USER`). Anything else creates nothing, and the decision is
logged in the outcome.

Two deliberate limits:

- The layer engages only where the agent proposed a ticket. Alerts
  routed to a human by low triage confidence, and calls paused by the
  guardrail, never turn into incidents here — a human judging an alert
  is not the same act as filing an incident.
- Creating is never the default. No answer, no incident.

The pattern is written up in
[patterns/03-human-validated-action.md](patterns/03-human-validated-action.md)
and tested against a fake Table API server in `tests/test_servicenow.py`.
Verified live as well: on the recorded run against a real ServiceNow
developer instance, the expiring-cert ticket was approved and became
incident **INC0010001**, caller set to the integration user.

## Notes

- **Generative half:** the summaries come from Bedrock (Nova Micro) or from
  a mock when there are no AWS credentials. No credentials existed on the
  build machine, so the live Bedrock path was verified end-to-end on the
  author's laptop instead (Nova Micro drives the agent and writes the
  summaries; run log shown in the demo video). The mock is dev-only; don't
  demo it as the real thing.
- **`strands-decider serve`:** if you'd rather not load the model
  in-process, `strands-decider serve` exposes it over HTTP and
  `decider_client` can point at that endpoint instead.
- **Everything is dry-run.** The remediation tools in this repo don't touch
  real infrastructure. The guardrail exists so that if they ever do, the
  arguments get checked first.
