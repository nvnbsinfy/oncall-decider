"""Human-validated ServiceNow incident creation (optional layer).

The agent's tools are dry-run by design: nothing the model proposes
touches a real system. This module is the one deliberate exception, and
it is built so a human is always the last step before anything real
happens:

    agent proposes a ticket (grounded, guardrail-checked)
        -> a human reads the exact payload and decides:
             create the incident in ServiceNow, or
             no incident — handle it another way (decision is logged)
        -> only an explicit "create" calls the ServiceNow Table API.

Not everything that reaches a human ends in an incident, and this layer
respects that: it engages only where the agent actually proposed a
ticket. Alerts routed to a human by low triage confidence, and tool
calls paused by the guardrail, are human judgment calls — no incident
is proposed there, so none can be created from them here.

The layer is off unless every one of these environment variables is set:

    SNOW_ENABLED=1
    SN_INSTANCE=https://dev123456.service-now.com   (or bare: dev123456)
    SN_USER=<user with incident create rights>
    SN_PASSWORD=<that user's password>

With the flag off — or the credentials incomplete — every run behaves
exactly as before: tickets stay dry-run and no prompt ever appears.
Stdlib only, so enabling this needs no extra packages.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


class ServiceNowError(RuntimeError):
    """Raised when ServiceNow rejects or cannot take an incident."""


@dataclass
class ServiceNowConfig:
    instance_url: str
    user: str
    password: str


_TRUTHY = {"1", "true", "yes", "on"}


def snow_config(environ: dict | None = None) -> ServiceNowConfig | None:
    """Resolve the ServiceNow config from the environment.

    Returns None when the layer is disabled. When it is enabled but the
    credentials are incomplete, say so loudly once and still return
    None — a half-configured run must fall back to dry-run, never fail
    halfway through a demo and never prompt for an incident it cannot
    create.
    """
    env = os.environ if environ is None else environ
    if str(env.get("SNOW_ENABLED", "")).strip().lower() not in _TRUTHY:
        return None
    instance = str(env.get("SN_INSTANCE", "")).strip()
    user = str(env.get("SN_USER", "")).strip()
    password = str(env.get("SN_PASSWORD", "")).strip()
    if not (instance and user and password):
        print("[oncall-decider] SNOW_ENABLED is set but SN_INSTANCE, SN_USER "
              "and SN_PASSWORD are not all present — ServiceNow layer OFF, "
              "tickets stay dry-run.")
        return None
    if not instance.startswith("http"):
        instance = f"https://{instance}.service-now.com" if "." not in instance \
            else f"https://{instance}"
    return ServiceNowConfig(instance_url=instance.rstrip("/"),
                            user=user, password=password)


_SEVERITY_MAP = {
    "p1": (1, 1), "critical": (1, 1), "high": (1, 2),
    "p2": (2, 2), "medium": (2, 2), "moderate": (2, 2),
    "p3": (3, 3), "low": (3, 3),
}


def severity_to_urgency_impact(severity: str) -> tuple[int, int]:
    """Map the agent's severity wording onto ServiceNow urgency/impact."""
    return _SEVERITY_MAP.get(str(severity).strip().lower(), (2, 2))


class ServiceNowClient:
    """Minimal Table API client: create an incident, return its number.

    The incident's Caller is set to the API user itself (looked up
    once from sys_user): an incident filed by an integration should
    still name a caller, and the honest caller here is the account
    doing the filing. If that lookup finds nothing, creation proceeds
    without a caller rather than failing over a cosmetic field.
    """

    def __init__(self, config: ServiceNowConfig, opener=None, timeout: int = 30):
        self._config = config
        self._opener = opener if opener is not None else urllib.request.urlopen
        self._timeout = timeout
        self._caller_sys_id: str | None = None

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        url = f"{self._config.instance_url}{path}"
        creds = base64.b64encode(
            f"{self._config.user}:{self._config.password}".encode()).decode()
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Basic {creds}",
            },
        )
        try:
            with self._opener(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise ServiceNowError(
                f"ServiceNow returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ServiceNowError(f"ServiceNow unreachable: {exc.reason}") from exc

    def caller_sys_id(self) -> str:
        """sys_id of the API user ('' if the lookup finds nothing)."""
        if self._caller_sys_id is None:
            self._caller_sys_id = ""
            query = urllib.parse.urlencode({
                "sysparm_fields": "sys_id",
                "sysparm_query": f"user_name={self._config.user}",
            })
            try:
                payload = self._call("GET", f"/api/now/table/sys_user?{query}")
                rows = payload.get("result", [])
                if rows:
                    self._caller_sys_id = rows[0].get("sys_id", "")
            except ServiceNowError:
                pass  # caller is best-effort; creation must not fail over it
        return self._caller_sys_id

    def create_incident(self, title: str, severity: str, summary: str) -> dict:
        urgency, impact = severity_to_urgency_impact(severity)
        body = {
            "short_description": title,
            "description": summary,
            "urgency": urgency,
            "impact": impact,
            "work_notes": "Created by oncall-decider after human validation "
                          "of the agent's proposed ticket.",
        }
        caller = self.caller_sys_id()
        if caller:
            body["caller_id"] = caller
        payload = self._call("POST", "/api/now/table/incident", body)
        result = payload.get("result", {})
        number = result.get("number")
        if not number:
            raise ServiceNowError(f"ServiceNow response had no incident number: {payload}")
        return {"number": number, "sys_id": result.get("sys_id", ""),
                "urgency": urgency, "impact": impact}


class ConsoleValidator:
    """Ask the human what should happen with the proposed incident.

    The payload shown is the payload that would be sent — same title,
    same severity, same summary — so the decision is about the real
    thing, not a sketch of it. Two outcomes:

      "create"      — file the incident in ServiceNow.
      "no_incident" — the human takes it from here (or judges it not
                      incident-worthy); the decision is logged and
                      nothing is created.

    Only an explicit create counts as create. Anything else — any
    other input, just pressing Enter, a closed stdin (EOF) — is
    "no_incident": creating real tickets is never the default.
    """

    def validate(self, payload: dict, config: ServiceNowConfig) -> str:
        print()
        print("=" * 70)
        print("HUMAN VALIDATION — the agent proposes this incident:")
        print(f"  title    : {payload['title']}")
        print(f"  severity : {payload['severity']}")
        print(f"  summary  : {payload['summary']}")
        print(f"  target   : {config.instance_url} (ServiceNow)")
        print("=" * 70)
        try:
            answer = input("[c]reate this incident in ServiceNow, or "
                           "[n]o incident? (default: no incident) ")
        except EOFError:
            print("[oncall-decider] no interactive input available — no incident.")
            return "no_incident"
        if answer.strip().lower() in {"c", "create", "y", "yes"}:
            return "create"
        return "no_incident"


def maybe_create_validated_incident(result: dict, ticket: dict, *,
                                    environ: dict | None = None,
                                    client: ServiceNowClient | None = None,
                                    validator: ConsoleValidator | None = None) -> bool:
    """Run the validation layer for one executed ticket call.

    `ticket` is the exact payload the agent's open_incident_ticket ran
    with (captured at the tool, so guardrail-denied calls never reach
    here). Mutates `result` with the outcome. Returns True only when a
    real incident was created.
    """
    config = snow_config(environ)
    if config is None:
        return False
    validator = validator if validator is not None else ConsoleValidator()
    if validator.validate(ticket, config) != "create":
        _annotate(result,
                  "Human-validated: no incident created (handled outside ServiceNow)",
                  "human-validated: no incident created")
        print("[oncall-decider] human chose NO incident — nothing created in ServiceNow.")
        return False
    client = client if client is not None else ServiceNowClient(config)
    try:
        created = client.create_incident(ticket["title"], ticket["severity"],
                                         ticket["summary"])
    except ServiceNowError as exc:
        note = f"ServiceNow creation FAILED after human approval ({exc})"
        _annotate(result, note, note)
        print(f"[oncall-decider] ServiceNow creation FAILED after approval: {exc}")
        return False
    result["servicenow"] = created
    note = f"ServiceNow incident {created['number']} created (human-validated)"
    _annotate(result, note, note)
    print(f"[oncall-decider] ServiceNow incident created: {created['number']} "
          f"({config.instance_url}/nav_to.do?uri=incident.do?sys_id={created['sys_id']})")
    return True


def _annotate(result: dict, replace_text: str, append_text: str) -> None:
    """Use replace_text when the alert's outcome is the plain dry-run
    ticket outcome; otherwise append append_text, since the outcome
    already says something else (e.g. the run also paused at the
    guardrail) and both facts belong in the table."""
    if result.get("outcome") == "ticket opened (dry-run)":
        result["outcome"] = replace_text
    else:
        result["outcome"] = f"{result.get('outcome', '')}; {append_text}"
