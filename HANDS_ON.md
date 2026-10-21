# Running it yourself (Windows laptop)

This is the hands-on guide. Everything runs on your own machine — the decision
model is local, so there is no cloud bill for the Decider half. Use native
Windows Python for all of this, not WSL. My WSL2 is capped at 2 GB of RAM and
the model needs about 4.5 GB, so WSL will just OOM.

## What you need

- Windows 10/11, 64-bit
- Python 3.11 or newer from python.org (tick "Add python.exe to PATH" during install).
  Verified end to end on 3.12; 3.14 also works — every pinned package ships a
  3.14 wheel. Skip 3.15 for now, it's brand new and the wheels lag behind.
- About 8 GB free disk (model weights are ~4.5 GB plus Python packages)
- About 5 GB free RAM while it runs — close Docker Desktop first
- git — only needed later, when you publish the repo to GitHub. You do NOT
  need git to run the demo. If `git` isn't recognized on your machine, get
  Git for Windows from https://git-scm.com/download/win before the publish step.

Check your Python: open PowerShell and run `python --version`. You want 3.11+.

## Setup

Get the code first: unzip the repo zip into a normal folder (your Desktop or
home folder — not C:\WINDOWS\system32). Once the repo is public on GitHub you
can also `git clone https://github.com/nvnbsinfy/oncall-decider.git` instead.
Then, in PowerShell:

```powershell
cd oncall-decider
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The torch wheel is large, so the install takes a few minutes. If PowerShell
refuses to run the activate script, run this once first:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## First run

Set this before the first run. It works around a download stall that happens
with the model weights on some networks:

```powershell
$env:HF_HUB_DISABLE_XET = "1"
```

Then:

```powershell
copy .env.example .env
python demo\run_storm.py
```

The first run downloads the model (~4.5 GB). After that it is cached and starts
fast. Expect each triage decision to take a few seconds on CPU (simple yes/no
checks run ~1.7s; the 4-way triage over a full alert takes ~4-7s once warm, longer
on the first couple while it warms up). The whole scripted storm takes a couple
of minutes. You will see the triage table, then a P1 where
the agent tries a bad remediation call, the guardrail blocks it, and the agent
corrects itself.

Other things you can run:

```powershell
$env:PYTHONPATH = "src"
python -m oncall_decider.bench        # latency benchmark, writes bench\results.md
python tests\test_guardrail.py        # guardrail unit test (uses the real model)
```

## Where to tweak things

**The triage question and option set** — `src\oncall_decider\interventions.py`,
function `triage_state_and_questions`. The four classes and their descriptions
are plain strings. Edit the descriptions to match how your team actually talks
about severity, and the decisions will follow.

**The triage confidence threshold** — `.env`, `TRIAGE_CONFIRM_THRESHOLD=0.60`.
Below this number the agent asks a human instead of acting on its own. Lower it
and the agent acts more independently; raise it and it asks more often. 0.60 is
a sane starting point.

**The guardrail question** — same file, `RemediationGuardrail.before_tool_call`.
The yes/no question is `"Are the tool's argument values grounded in facts from
the alert payload?"`. The two thresholds in `__init__` (`deny_threshold=0.35`,
`confirm_threshold=0.60`) control the three outcomes: confidently ungrounded
gets denied, uncertain goes to a human, the rest proceed.

**Pointing it at Bedrock** — `src\oncall_decider\agent.py`, function
`get_model()`. It uses Bedrock automatically when it finds AWS credentials, and
falls back to the built-in mock otherwise (the mock is dev-only, it just lets
you test the loop without AWS). To use Bedrock:

1. Create an AWS account if you don't have one. Bedrock models like Nova
   Micro are enabled automatically — there is no model-access page to visit
   anymore (AWS retired it; the models just work once you have credentials).
   **Do this in the us-east-1 region** — the Bedrock console region selector
   must show us-east-1 before you request access; access granted in one region
   only works there, so asking in the wrong region gives a "model not available"
   error later.
2. Create an access key and set the env vars (PowerShell):

```powershell
$env:AWS_ACCESS_KEY_ID = "your-key"
$env:AWS_SECRET_ACCESS_KEY = "your-secret"
$env:AWS_REGION = "us-east-1"
$env:AWS_DEFAULT_REGION = "us-east-1"
```

To make them permanent, use `setx` instead of `$env:`. You can also change the
model in `.env` via `BEDROCK_MODEL_ID`. Demo-scale usage costs cents.

## Optional: real ServiceNow incidents (human-validated)

Everything so far is dry-run. If you want the demo to be able to end with a
real incident in ServiceNow — created only when you approve it — do this:

1. Get a free Personal Developer Instance at developer.servicenow.com
   (sign in, request an instance; takes a few minutes). Note the instance
   URL (https://devXXXXX.service-now.com) and the admin username/password.
2. In the same PowerShell window (venv active), set:

```
$env:SNOW_ENABLED = "1"
$env:SN_INSTANCE = "https://devXXXXX.service-now.com"
$env:SN_USER = "admin"
$env:SN_PASSWORD = "your instance password"
```

3. Run the demo again: `python demo\run_storm.py`

Now, each time the agent files a ticket, the run stops and shows you the
exact incident it proposes. Type `c` to create it in ServiceNow — the demo
prints the real incident number and it shows in the triage table — or `n`
(or just Enter) for no incident: the decision is logged and nothing is
created.

Notes:

- Not everything that reaches a human becomes an incident. The
  low-confidence alerts and the guardrail pause stay human judgment calls;
  this prompt only appears for tickets the agent actually proposed.
- Your password lives only in that PowerShell window's environment. Never
  put it in `.env` or anywhere else git can see it.
- To switch the layer back off: `$env:SNOW_ENABLED = "0"`, or just open a
  fresh PowerShell window.

## If something breaks

- `python` not recognized: reinstall Python and tick the PATH box, then open a
  new PowerShell window.
- Download stalls on first run: make sure `HF_HUB_DISABLE_XET=1` is set in the
  same window before running.
- Out of memory / process dies: close Docker Desktop and anything heavy. The
  model needs ~5 GB free. Do not run it inside WSL.
- Bench numbers look odd on your machine: CPU speed varies a lot between
  machines. The 1.7s figure is from the build VM; yours will differ. The
  comparison that matters is Decider vs the simulated LLM baseline on the same
  box.
