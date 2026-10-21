# Benchmark results

_Baseline: measured Bedrock Converse call (amazon.nova-micro-v1:0), same question and classes._

| alert     | service         | Decider 2B      | conf | dec (s) | LLM             | LLM (s) |
|-----------|-----------------|-----------------|------|---------|-----------------|---------|
| ALT-1001  | checkout-api    | page_now        | 0.84 | 33.09   | page_now        | 0.56    |
| ALT-1002  | auth-service    | page_now        | 0.96 | 48.88   | page_now        | 0.44    |
| ALT-1003  | search-api      | investigate     | 0.19 | 52.65   | investigate     | 0.48    |
| ALT-1004  | edge-gateway    | investigate     | 0.97 | 40.90   | investigate     | 0.48    |
| ALT-1005  | worker-pool     | suppress_noise  | 0.87 | 41.87   | suppress_noise  | 0.41    |
| ALT-1006  | cache-layer     | suppress_noise  | 0.86 | 39.31   | suppress_noise  | 0.42    |
| ALT-1007  | recommender     | false_positive  | 0.87 | 38.52   | false_positive  | 0.39    |
| ALT-1008  | order-worker    | page_now        | 0.30 | 39.55   | page_now        | 0.59    |

- Machine: Windows AMD64, Decider on `cpu`
- Decider cold start (model load + first decision, excluded below): 80.4 s
- Decider warm, per decision: median 40.23 s, max 52.65 s, total 334.8 s
- Baseline (measured), per decision: median 0.46 s, total 3.8 s
- Agreement with Bedrock: 8/8 alerts; 6/6 where Decider was confident (>= 0.60)
- Bedrock tokens per decision: 289 in / 4 out (Decider: none billed)

## How to read this

- Both columns answer the same question ("Which triage class does this monitoring alert belong to?")
  over the same four classes, one alert at a time.
- Decider times are wall-clock on the machine named above, after the model
  was loaded and warmed. Upstream's ~115 ms figure is for an RTX 3090 GPU; on
  CPU, expect seconds.
- The LLM column is a real Bedrock call: network round-trip plus generation.
