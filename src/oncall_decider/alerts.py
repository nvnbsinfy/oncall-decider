"""Vendor-neutral alert stream simulator.

Builds a scripted eight-alert storm that exercises every path of the
on-call agent: page-worthy incidents, things to investigate, noise to
suppress, and one false positive. No real monitoring vendor is named
anywhere here; the alerts are plain dictionaries shaped like the
payloads an on-call engineer would actually see.
"""

from __future__ import annotations


def build_storm() -> list[dict]:
    """Return the scripted alert storm: 8 alerts, one dict each.

    Alert shape:
        {
            "id": "ALT-1001",
            "service": "checkout-api",
            "title": "...",
            "description": "...",        # one realistic paragraph
            "labels": {"severity": ..., "team": ..., "namespace": ...},
            "signals": {metric_name: value, ...},
        }
    """
    return [
        {
            "id": "ALT-1001",
            "service": "checkout-api",
            "title": "Error budget burn: checkout-api serving 12% errors",
            "description": (
                "checkout-api's 5xx rate has been climbing for 25 minutes and is now "
                "sitting at 12% of requests. The burn rate is 14x the budgeted rate, "
                "which eats roughly two weeks of the 30-day error budget in under "
                "two hours if it holds. The failures started minutes after the "
                "latest deploy of the payments path, and the failure mode looks "
                "consistent across pods rather than a single bad host."
            ),
            "labels": {"severity": "P1", "team": "payments", "namespace": "prod"},
            "signals": {
                "http_requests_5xx_rate": 0.12,
                "error_budget_burn_rate": 14.0,
                "deploy_age_minutes": 25,
            },
        },
        {
            "id": "ALT-1002",
            "service": "auth-service",
            "title": "auth-service CrashLoopBackOff: 47 restarts in the last hour",
            "description": (
                "auth-service pods keep dying within seconds of starting and have "
                "restarted 47 times in the last hour. Liveness probes never go green. "
                "The logs show the same panic on every start, right after config "
                "load, so this is a bad config or a bad image rather than load. "
                "Logins are failing over to the cached-token path, which only buys "
                "a few hours before sessions start expiring."
            ),
            "labels": {"severity": "P1", "team": "identity", "namespace": "prod"},
            "signals": {
                "container_restarts_1h": 47,
                "pod_ready_ratio": 0.0,
                "login_failure_rate": 0.68,
            },
        },
        {
            "id": "ALT-1003",
            "service": "search-api",
            "title": "p99 latency regression on search-api after deploy",
            "description": (
                "search-api p99 latency jumped from 450ms to 2100ms about 40 minutes "
                "ago, right after a deploy that touched the ranking query planner. "
                "Throughput is unchanged and error rates are flat, so the service "
                "is still serving — just slowly. Users are noticing: support has "
                "three tickets mentioning slow search in the last half hour."
            ),
            "labels": {"severity": "P2", "team": "search", "namespace": "prod"},
            "signals": {
                "p99_latency_ms": 2100,
                "p99_latency_ms_baseline": 450,
                "deploy_age_minutes": 40,
                "error_rate": 0.002,
            },
        },
        {
            "id": "ALT-1004",
            "service": "edge-gateway",
            "title": "TLS certificate on edge-gateway expires in 6 days",
            "description": (
                "The public-facing certificate for edge-gateway expires in 6 days. "
                "The auto-renewal job ran but failed to complete the challenge, so "
                "nothing is scheduled to fix this on its own. If it lapses, every "
                "external request hits a certificate error. Six days is enough to "
                "renew by hand, but short enough that it shouldn't wait for the "
                "next sprint."
            ),
            "labels": {"severity": "P2", "team": "platform", "namespace": "prod"},
            "signals": {
                "cert_expiry_days": 6,
                "autorenew_last_status": "failed",
            },
        },
        {
            "id": "ALT-1005",
            "service": "worker-pool",
            "title": "Disk usage 82% on worker-pool nodes",
            "description": (
                "Disk usage on worker-pool nodes is at 82%, which tripped the "
                "warning threshold. It's been sitting between 78% and 83% for "
                "three weeks — log rotation keeps it flat — so this is a stable "
                "plateau, not a leak. It deserves a look during the next capacity "
                "review, but nobody needs to wake up for it tonight."
            ),
            "labels": {"severity": "P3", "team": "platform", "namespace": "prod"},
            "signals": {
                "disk_usage_percent": 82,
                "weeks_stable": 3,
                "disk_growth_gb_per_day": 0.1,
            },
        },
        {
            "id": "ALT-1006",
            "service": "cache-layer",
            "title": "Cache hit-rate monitor flapping",
            "description": (
                "The cache hit-rate alert has fired and cleared four times in the "
                "last two hours. Each dip lines up with a batch job that runs "
                "cold reads against keys that were never cached, so the hit rate "
                "drops, the job finishes, and it recovers on its own. The alert "
                "threshold doesn't know about the batch schedule."
            ),
            "labels": {"severity": "P3", "team": "platform", "namespace": "prod"},
            "signals": {
                "alert_transitions_2h": 4,
                "cache_hit_rate": 0.91,
                "cache_hit_rate_baseline": 0.97,
            },
        },
        {
            "id": "ALT-1007",
            "service": "recommender",
            "title": "Canary deploy blip on recommender (self-resolved)",
            "description": (
                "During the recommender canary deploy, the error-rate monitor fired "
                "for about 90 seconds and then cleared by itself. The canary "
                "analysis already rolled the deploy forward because the blip was "
                "within the allowed noise band. Nothing is failing now; this is "
                "the monitor telling us what happened, not what is happening."
            ),
            "labels": {"severity": "P3", "team": "ml-platform", "namespace": "prod"},
            "signals": {
                "blip_duration_seconds": 90,
                "current_error_rate": 0.001,
                "canary_status": "promoted",
            },
        },
        {
            "id": "ALT-1008",
            "service": "order-worker",
            "title": "order-worker queue depth growing: 500 to 8000 messages",
            "description": (
                "The order-worker queue has grown from 500 to 8000 messages in "
                "the last 30 minutes while consumer lag keeps climbing. The "
                "workers aren't erroring — they're just not keeping up, which "
                "points at a throughput problem rather than a crash. If the trend "
                "holds, checkout starts timing out for customers in about an hour."
            ),
            "labels": {"severity": "P2", "team": "payments", "namespace": "prod"},
            "signals": {
                "queue_depth": 8000,
                "queue_depth_baseline": 500,
                "consumer_lag_seconds": 420,
                "worker_error_rate": 0.0,
            },
        },
    ]
