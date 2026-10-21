"""Lazy, CPU-safe singleton client for the Decider 2B decision model.

Usage:
    from oncall_decider.decider_client import get_client

    client = get_client()          # no model loaded yet
    result = client.decide(
        state="Alert CPU-01 [api]: CPU above 95% for 10m",
        questions={
            "triage": {
                "type": "choice",
                "question": "Which triage class does this monitoring alert belong to?",
                "options": {"page_now": "...", "investigate": "..."},
            },
            "grounded": {
                "type": "noul",
                "question": "Does every value in the proposed tool call appear "
                            "in the alert payload above? Answer yes only if all "
                            "of them do; answer no if any value is invented.",
            },
        },
    )
    # result == {
    #   "triage": {"decision": "page_now", "confidence": 0.87,
    #              "distribution": {"page_now": 0.87, "investigate": 0.13}},
    #   "grounded": {"noul": 0.21, "decision": False, "confidence": 0.79},
    # }

CPU-memory note: the stock SystemOneEngine upcasts the bf16 torso to fp32 on
CPU (about 9 GB of RAM on this 2B model), which OOMs on small VMs. This
client subclasses the engine to skip that upcast, keeping inference at
~4.5 GB. On a GPU machine you can leave device="cuda" and get the stock
behaviour.

HTTP alternative: if you'd rather run the model as a local server instead
of in-process, strands-decider ships one:

    strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8080

then POST {"state": ..., "model": ..., "questions": {...}} to /v1/systemone
(see strands_decider.server for the exact route/shape). The question
dictionaries used by decide() mirror that wire format, so swapping between
the two is mechanical.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

# The schema only needs pydantic. torch/transformers (via infer/modeling) are
# imported on first decide(), so importing this module, the handlers, or the
# tests never pulls in the model stack.
from strands_decider.schema import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    SystemOneRequest,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "StrandsAgents/strands-decider-2B-hobson-v19"
ENV_MODEL_ID = "DECIDER_MODEL_ID"


def _cpu_bf16_engine_class():
    """SystemOneEngine subclass that keeps the torso in bf16 on CPU.

    The stock engine upcasts the torso bf16->fp32 on CPU for speed (~2x on
    Apple silicon) at the cost of ~9 GB RAM. On memory-constrained machines
    that is fatal, and bf16 inference gives the same answers, just slower.
    Overriding the hook is a no-op by design, not a monkeypatch:
    _upcast_torso_for_cpu is a real method on SystemOneEngine. Built lazily so
    torch is imported only when the model is actually loaded.
    """
    from strands_decider.infer import SystemOneEngine

    class _CpuBf16Engine(SystemOneEngine):
        def _upcast_torso_for_cpu(self) -> None:  # noqa: D102 - deliberate no-op
            return

    return _CpuBf16Engine


class DeciderClient:
    """Thin, typed wrapper around a lazily-loaded SystemOneEngine.

    Construction never touches the model; the first call to decide() loads
    it (once) and logs how long that took.
    """

    def __init__(self, model_id: str | None = None, device: str = "cpu") -> None:
        self.model_id = model_id or os.environ.get(ENV_MODEL_ID, DEFAULT_MODEL_ID)
        self.device = device
        self._engine: Any = None
        self._lock = threading.Lock()

    # -- loading -----------------------------------------------------------

    def _resolve_checkpoint(self) -> str:
        """Resolve the model id to a local directory.

        A local directory is used as-is. A Hub id is fetched through the Hub
        cache: downloaded on first use, read from the cache afterwards. Set
        HF_HUB_OFFLINE=1 yourself to forbid network access once it is cached,
        and HF_HUB_DISABLE_XET=1 if downloads hang behind a proxy.
        """
        if os.path.isdir(self.model_id):
            return self.model_id
        from huggingface_hub import snapshot_download

        return str(snapshot_download(self.model_id))

    def _load(self) -> Any:
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine
            t0 = time.perf_counter()
            from strands_decider.infer import EngineConfig
            from strands_decider.modeling import StrandsDeciderModel

            checkpoint = self._resolve_checkpoint()
            model = StrandsDeciderModel.load(checkpoint)
            engine_cls = _cpu_bf16_engine_class()
            engine = engine_cls(model, EngineConfig(device=self.device))
            self._engine = engine
            dt = time.perf_counter() - t0
            log.info(
                "Decider 2B loaded in %.1fs (model=%s, device=%s)",
                dt, self.model_id, self.device,
            )
            print(f"[oncall-decider] Decider 2B loaded in {dt:.1f}s "
                  f"(model={self.model_id}, device={self.device})")
            return engine

    # -- asking ------------------------------------------------------------

    @staticmethod
    def _to_question(spec: dict[str, Any]) -> Question:
        """Translate our plain-dict question spec into a strands_decider Question.

        Spec shapes:
            {"type": "choice", "question": "...", "options": {"name": "desc", ...}}
            {"type": "noul", "question": "..."}
        """
        qtype = spec.get("type")
        if qtype == "choice":
            return ChoiceQuestion(
                instructions=spec["question"],
                criteria=dict(spec["options"]),
            )
        if qtype == "noul":
            kwargs = {}
            if spec.get("criteria"):
                kwargs["criteria"] = dict(spec["criteria"])
            return NoulQuestion(instructions=spec["question"], **kwargs)
        raise ValueError(f"unsupported question type: {qtype!r}")

    def decide(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Ask one or more typed questions about the same state.

        Asking several questions together is cheap: the state is encoded once
        and the questions evaluate in parallel. Returns a plain-dict result:

            choice -> {"decision", "confidence", "distribution"}
            noul   -> {"noul", "decision" (bool), "confidence"}
        """
        engine = self._load()
        request = SystemOneRequest(
            state=state,
            questions={name: self._to_question(spec) for name, spec in questions.items()},
        )
        response = engine.evaluate(request)

        out: dict[str, Any] = {}
        for name, answer in response.answers.items():
            if answer.type == "choice":
                out[name] = {
                    "decision": answer.choice,
                    "confidence": answer.confidence,
                    "distribution": dict(answer.probabilities),
                }
            elif answer.type == "noul":
                p = answer.noul
                out[name] = {
                    "noul": p,
                    "decision": bool(p >= 0.5),
                    "confidence": max(p, 1.0 - p),
                }
            else:  # pragma: no cover - we only issue choice/noul questions
                raise ValueError(f"unexpected answer type: {answer.type!r}")
        return out


_client: DeciderClient | None = None
_client_lock = threading.Lock()


def get_client() -> DeciderClient:
    """Module-level lazy singleton. First decide() call loads the model."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = DeciderClient()
    return _client


def reset_client() -> None:
    """Drop the singleton (mainly for tests)."""
    global _client
    with _client_lock:
        _client = None
