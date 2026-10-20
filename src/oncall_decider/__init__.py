"""On-call decider package: local decision-model client, simulated remediation
tools, and Strands intervention handlers for alert triage and remediation
guardrails.

Pieces:
- decider_client: lazy, CPU-safe client for the local Decider 2B model.
- tools: simulated (dry-run) remediation tools the agent may call.
- interventions: TriageDecider (policy classification) and
  RemediationGuardrail (grounding checks) as Strands InterventionHandlers.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
