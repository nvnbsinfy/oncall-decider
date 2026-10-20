# Benchmark results

_Baseline: simulated LLM classifier at 2.5s/decision, approximating a small Bedrock model round-trip. Honest but approximate — not measured._

```text
+------------+--------------------+------------------------------+--------+-----------+------------+----------+
| alert      | service            | triage decision              | conf   | decider (s) | baseline (s) | speedup |
+------------+--------------------+------------------------------+--------+-----------+------------+----------+
| ALT-1001   | checkout-api       | page_now                     | 0.83   | 22.01     | 2.50       | 0.1    x |
| ALT-1002   | auth-service       | page_now                     | 0.96   | 14.53     | 2.50       | 0.2    x |
| ALT-1003   | search-api         | investigate                  | 0.19   | 12.20     | 2.50       | 0.2    x |
| ALT-1004   | edge-gateway       | investigate                  | 0.97   | 4.85      | 2.50       | 0.5    x |
| ALT-1005   | worker-pool        | suppress_noise               | 0.87   | 5.42      | 2.50       | 0.5    x |
| ALT-1006   | cache-layer        | suppress_noise               | 0.86   | 4.41      | 2.50       | 0.6    x |
| ALT-1007   | recommender        | false_positive               | 0.87   | 7.47      | 2.50       | 0.3    x |
| ALT-1008   | order-worker       | page_now                     | 0.31   | 6.06      | 2.50       | 0.4    x |
+------------+--------------------+------------------------------+--------+-----------+------------+----------+
| TOTAL      |                    |                              |        | 76.95     | 20.00      | 0.3    x |
+------------+--------------------+------------------------------+--------+-----------+------------+----------+
```

## How to read this

- **Decider 2B** column: measured wall-clock time per triage decision on this
  machine, including model load / first-call cold start on the first alert.
- **Baseline** column: simulated LLM baseline (2.5s/decision, approximating a small Bedrock model round-trip). The 2.5s simulated figure is a
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
