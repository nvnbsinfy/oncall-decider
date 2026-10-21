# Pattern 3: Human-validated action

**The problem.** An agent that can write to a real system of record will
eventually write something a human would have stopped — not because the
arguments were ungrounded, but because filing the record was the wrong
call for that situation. Grounding checks can't judge that. A person
can, in about five seconds, if you show them the exact thing and ask
before it happens.

**The pattern.** Put a human decision between "the agent proposes a
real-world write" and "the write happens":

1. The proposal is already machine-checked (in this repo, the triage
   policy and the grounding guardrail ran first). The human is the last
   gate, not the only one.
2. Show the human the exact payload that will be sent — same fields,
   same values. Approving a summary of the action is not approval of
   the action.
3. Make it a choice, not a rubber stamp: create it, or no — handle it
   another way. Both answers are legitimate, and the "no" is logged
   with the same care as the "yes".
4. Default to not doing it. No answer, an unclear answer, or no
   interactive input at all means the write does not happen.
5. Record the outcome where the run is reported, including the real
   system's identifier when the write goes through (here, the
   ServiceNow incident number in the triage table).

**Where it applies.** Only at genuine write points. Human touchpoints
that exist for judgment — a low-confidence classification, an uncertain
tool call — are not write points, and bolting record creation onto
them trains people to click through. One gate, at the one moment
something becomes real.

**Wiring in this repo.** `src/oncall_decider/servicenow.py`.
`RECORDED_TICKETS` in `tools.py` captures ticket calls that actually
executed — a guardrail-denied call never reaches the tool body, so it
can never be validated into a real incident. After each agent turn,
`maybe_create_validated_incident` runs the console validator and, on
an explicit "create", posts to the ServiceNow Table API
(`POST /api/now/table/incident`, severity mapped to urgency/impact).
The layer is flag-gated (`SNOW_ENABLED` plus credentials) and off by
default; with it off, behavior is byte-for-byte the dry-run demo.

**Failure modes to expect.** If the write fails after approval
(instance down, bad credentials), say so in the outcome — never let a
failed write masquerade as a created record. If the layer is enabled
but misconfigured, fall back to dry-run with a loud warning rather
than failing the run.
