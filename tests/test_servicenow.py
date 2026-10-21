"""ServiceNow human-validation layer tests: fake Table API server.

Run: PYTHONPATH=src python tests/test_servicenow.py

No real ServiceNow and no credentials: a stdlib HTTP server plays the
Table API, records what it receives, and answers like ServiceNow does.
The Decider model is not involved — this layer starts after the agent
has already filed its (dry-run) ticket.

Cases:
  1. Config gating: off by default; bare instance names normalize;
     enabled-but-incomplete falls back to off.
  2. Severity mapping onto urgency/impact.
  3. Client posts the right shape (path, Basic auth, fields) and
     parses the incident number.
  4. Human chooses "create" -> incident created, outcome names the INC.
  5. Human chooses "no incident" -> no HTTP request is made at all.
  6. Layer disabled -> validator is never even asked.
  7. ServiceNow errors after approval -> failure is reported, not hidden.
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from oncall_decider.servicenow import (
    ServiceNowClient,
    maybe_create_validated_incident,
    severity_to_urgency_impact,
    snow_config,
)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}")
    else:
        print(f"  FAIL {name} {detail}")
        failures.append(name)


class FakeServiceNow(BaseHTTPRequestHandler):
    requests = []
    fail_with = None
    empty_user_lookup = False

    def do_GET(self):
        if "/api/now/table/sys_user" in self.path:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            rows = [] if FakeServiceNow.empty_user_lookup else [{"sys_id": "usersysid123"}]
            self.wfile.write(json.dumps({"result": rows}).encode())
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        FakeServiceNow.requests.append({
            "path": self.path,
            "authorization": self.headers.get("Authorization", ""),
            "body": json.loads(body) if body else {},
        })
        if FakeServiceNow.fail_with:
            self.send_response(FakeServiceNow.fail_with)
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"nope"}}')
            return
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(
            {"result": {"number": "INC0010123", "sys_id": "abc123"}}).encode())

    def log_message(self, *args):
        pass


class StubValidator:
    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    def validate(self, payload, config):
        self.calls += 1
        return self.decision


TICKET = {"title": "TLS certificate on edge-gateway expires in 6 days",
          "severity": "P2",
          "summary": "The public-facing certificate expires in 6 days."}


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeServiceNow)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {"SNOW_ENABLED": "1", "SN_INSTANCE": f"http://127.0.0.1:{port}",
           "SN_USER": "admin", "SN_PASSWORD": "test-only"}

    print("config gating")
    check("off by default", snow_config({}) is None)
    cfg = snow_config({"SNOW_ENABLED": "1", "SN_INSTANCE": "dev123456",
                       "SN_USER": "u", "SN_PASSWORD": "p"})
    check("bare instance normalizes",
          cfg is not None and cfg.instance_url == "https://dev123456.service-now.com")
    check("incomplete creds -> off",
          snow_config({"SNOW_ENABLED": "1", "SN_INSTANCE": "x"}) is None)

    print("severity mapping")
    check("P1 -> (1,1)", severity_to_urgency_impact("P1") == (1, 1))
    check("P2 -> (2,2)", severity_to_urgency_impact("P2") == (2, 2))
    check("High -> (1,2)", severity_to_urgency_impact("High") == (1, 2))
    check("unknown -> (2,2)", severity_to_urgency_impact("whenever") == (2, 2))

    print("client request shape")
    FakeServiceNow.requests = []
    client = ServiceNowClient(snow_config(env))
    created = client.create_incident(TICKET["title"], TICKET["severity"], TICKET["summary"])
    req = FakeServiceNow.requests[-1]
    check("number parsed", created["number"] == "INC0010123")
    check("path", req["path"] == "/api/now/table/incident")
    check("basic auth sent", req["authorization"].startswith("Basic "))
    check("short_description", req["body"]["short_description"] == TICKET["title"])
    check("urgency/impact", req["body"]["urgency"] == 2 and req["body"]["impact"] == 2)
    check("caller set to API user", req["body"].get("caller_id") == "usersysid123")

    print("caller lookup finds nobody -> create anyway, without caller")
    FakeServiceNow.empty_user_lookup = True
    FakeServiceNow.requests = []
    created = ServiceNowClient(snow_config(env)).create_incident(
        TICKET["title"], TICKET["severity"], TICKET["summary"])
    check("created anyway", created["number"] == "INC0010123")
    check("no caller_id sent", "caller_id" not in FakeServiceNow.requests[-1]["body"])
    FakeServiceNow.empty_user_lookup = False

    print("human chooses create")
    FakeServiceNow.requests = []
    result = {"outcome": "ticket opened (dry-run)"}
    ok = maybe_create_validated_incident(result, TICKET, environ=env,
                                         validator=StubValidator("create"))
    check("created", ok and result.get("servicenow", {}).get("number") == "INC0010123")
    check("outcome names the incident",
          result["outcome"] == "ServiceNow incident INC0010123 created (human-validated)")
    check("exactly one POST", len(FakeServiceNow.requests) == 1)

    print("human chooses no incident")
    FakeServiceNow.requests = []
    result = {"outcome": "ticket opened (dry-run)"}
    ok = maybe_create_validated_incident(result, TICKET, environ=env,
                                         validator=StubValidator("no_incident"))
    check("not created", not ok and "servicenow" not in result)
    check("no HTTP request made", len(FakeServiceNow.requests) == 0)
    check("outcome says no incident",
          result["outcome"] == "Human-validated: no incident created (handled outside ServiceNow)")

    print("layer disabled")
    stub = StubValidator("create")
    result = {"outcome": "ticket opened (dry-run)"}
    ok = maybe_create_validated_incident(result, TICKET, environ={}, validator=stub)
    check("no-op", not ok and result["outcome"] == "ticket opened (dry-run)")
    check("validator never asked", stub.calls == 0)

    print("servicenow errors after approval")
    FakeServiceNow.fail_with = 401
    result = {"outcome": "ticket opened (dry-run)"}
    ok = maybe_create_validated_incident(result, TICKET, environ=env,
                                         validator=StubValidator("create"))
    check("failure reported", not ok and "FAILED" in result["outcome"])
    FakeServiceNow.fail_with = None

    server.shutdown()
    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {failures}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
