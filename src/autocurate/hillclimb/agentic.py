"""Agentic (LLM) candidate proposer — same interface as the offline one.

Given the incumbent gate, its held-out score, and a few *failing cases* (defects
it misses and clean docs it wrongly flags), an LLM proposes a new parameter
vector. The proposal is then subject to the *identical* verifier-gated accept
rule — the LLM can suggest anything, but only a held-out, guard-confirmed
improvement is adopted, so the anti-reward-hack guarantee is unchanged whether
the proposer is a random walk or a frontier model.

This backend is optional (``pip install autocurate[llm]``). The default loop
uses the deterministic ``OfflineProposer`` so everything runs in CI.
"""

from __future__ import annotations

import json
import random
from typing import List, Optional, Sequence

from ..schema import Document


_SYSTEM = (
    "You tune a reference-free data-extraction quality gate. You are given the "
    "gate's signal names, its current non-negative weights and accept threshold, "
    "its held-out F1 and clean false-positive rate, and example failures. Propose "
    "ONE improved parameter vector. Respond with strict JSON: "
    '{"weights": {<signal>: <float>, ...}, "threshold": <float in 0.05..0.95>}. '
    "Raise weights on signals that distinguish the missed defects; keep the "
    "threshold from over-flagging clean text."
)


class AgenticProposer:
    """LLM-backed proposer. Falls back to perturbation if no backend is set."""

    def __init__(self, llm=None, n_failures: int = 4, temperature: float = 0.8,
                 seed: int = 7):
        self.llm = llm                 # any callable(system, user) -> str, or None
        self.n_failures = n_failures
        self.temperature = temperature
        self.rng = random.Random(seed)

    def _failures(self, gate, clean_docs: Sequence[Document], modality: str) -> List[str]:
        from ..verify.corruptions import corrupt
        out: List[str] = []
        for d in clean_docs[: self.n_failures * 4]:
            corrupted, _ = corrupt(d, modality, self.rng, "heldout")
            if gate.evaluate(corrupted).passed:                 # a miss
                out.append(f"MISSED DEFECT: {corrupted.text[:160]!r}")
            elif not gate.evaluate(d).passed:                   # a false alarm
                out.append(f"FALSE ALARM ON CLEAN: {d.text[:160]!r}")
            if len(out) >= self.n_failures:
                break
        return out

    def _prompt(self, gate, clean_docs, modality) -> str:
        return json.dumps({
            "signals": gate.SIGNALS,
            "weights": {k: round(gate.weights[k], 3) for k in gate.SIGNALS},
            "threshold": round(gate.threshold, 3),
            "failures": self._failures(gate, clean_docs, modality),
        }, indent=2)

    def propose(self, gate, clean_docs: Optional[Sequence[Document]] = None,
                modality: str = "html") -> object:
        if self.llm is None or clean_docs is None:
            return self._perturb(gate)                          # graceful fallback
        try:
            reply = self.llm(_SYSTEM, self._prompt(gate, clean_docs, modality))
            spec = json.loads(reply[reply.index("{"): reply.rindex("}") + 1])
            cand = gate.clone()
            for k, v in spec.get("weights", {}).items():
                if k in cand.weights:
                    cand.weights[k] = max(0.0, float(v))
            if "threshold" in spec:
                cand.threshold = min(0.95, max(0.05, float(spec["threshold"])))
            return cand
        except Exception:
            return self._perturb(gate)

    def _perturb(self, gate) -> object:
        cand = gate.clone()
        params = cand.get_params()
        bounds = cand.bounds()
        i = self.rng.randrange(len(params))
        lo, hi = bounds[i]
        params[i] = min(hi, max(lo, params[i] + self.rng.gauss(0.0, 0.1 * (hi - lo))))
        cand.set_params(params)
        return cand

    def propose_many(self, gate, k: int, clean_docs=None, modality: str = "html") -> List[object]:
        return [self.propose(gate, clean_docs, modality) for _ in range(k)]
