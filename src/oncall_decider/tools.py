"""Simulated remediation tools for the on-call agent.

Every tool here is strictly dry-run: each one logs what it *would* have done
and returns a result dict. None of them touches real infrastructure, real
alerting, or real ticketing. The point is to exercise the decision pipeline
(triage -> agent -> guardrail -> tool) without any blast radius.
"""

from __future__ import annotations

from strands import tool


@tool
def restart_service(service: str, namespace: str = "default") -> dict:
    """Simulate restarting a service. DRY-RUN: no real restart happens."""
    action = f"DRY-RUN restart_service(service={service!r}, namespace={namespace!r})"
    print(f"[oncall-decider] {action}")
    return {
        "status": "dry_run",
        "action": "restart_service",
        "service": service,
        "namespace": namespace,
        "detail": f"Would restart {service} in namespace {namespace}.",
    }


@tool
def scale_deployment(service: str, replicas: int, namespace: str = "default") -> dict:
    """Simulate scaling a deployment. DRY-RUN: no real scaling happens."""
    action = (
        f"DRY-RUN scale_deployment(service={service!r}, replicas={replicas!r}, "
        f"namespace={namespace!r})"
    )
    print(f"[oncall-decider] {action}")
    return {
        "status": "dry_run",
        "action": "scale_deployment",
        "service": service,
        "replicas": replicas,
        "namespace": namespace,
        "detail": f"Would scale {service} to {replicas} replicas in namespace {namespace}.",
    }


@tool
def silence_alert(alert_id: str, duration_minutes: int, reason: str) -> dict:
    """Simulate silencing an alert. DRY-RUN: no real silence is created."""
    action = (
        f"DRY-RUN silence_alert(alert_id={alert_id!r}, "
        f"duration_minutes={duration_minutes!r}, reason={reason!r})"
    )
    print(f"[oncall-decider] {action}")
    return {
        "status": "dry_run",
        "action": "silence_alert",
        "alert_id": alert_id,
        "duration_minutes": duration_minutes,
        "reason": reason,
        "detail": f"Would silence {alert_id} for {duration_minutes} minutes: {reason}.",
    }


@tool
def open_incident_ticket(title: str, severity: str, summary: str) -> dict:
    """Simulate opening an incident ticket. DRY-RUN: no real ticket is filed."""
    action = (
        f"DRY-RUN open_incident_ticket(title={title!r}, severity={severity!r}, "
        f"summary={summary!r})"
    )
    print(f"[oncall-decider] {action}")
    return {
        "status": "dry_run",
        "action": "open_incident_ticket",
        "title": title,
        "severity": severity,
        "summary": summary,
        "detail": f"Would open a {severity} incident ticket: {title}.",
    }


TOOLS = [restart_service, scale_deployment, silence_alert, open_incident_ticket]
