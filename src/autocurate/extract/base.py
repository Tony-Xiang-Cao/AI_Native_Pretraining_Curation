"""Base class for reference-free extraction-quality gates.

A gate estimates how cleanly a document was *extracted* — not whether its
content is good, but whether the HTML→text / OCR / JSON step produced faithful
text. It does this **without a gold reference**: each gate computes a handful of
cheap defect *signals* in ``[0, 1]`` (1 == strong evidence of that defect) and
combines them into a single quality estimate.

Extraction defects are typically *channel-local* — a truncated JSON record trips
only ``parse_fail``; a leaked nav bar trips only ``boilerplate``. A weighted mean
would average a single strong defect away, so we combine the signals as a
**noisy-AND** (independent veto): each signal ``k`` passes with probability
``1 − ŵ_k·s_k`` and the gate's quality is the product

    quality = Π_k ( 1 − clamp(w_k · s_k) )                                (eq. gate)

so any one channel with a strong defect drives quality toward 0, while a clean
document (all ``s_k ≈ 0``) scores ≈ 1. The per-signal sensitivities ``w`` and the
accept ``threshold`` form the gate's **parameter vector** — exactly the knobs the
self-improvement loop (``hillclimb``) climbs against the mutation oracle
(``verify``). Subclasses only implement ``_signals`` and declare ``SIGNALS`` +
``DEFAULTS``; all the parameter plumbing, combination, and flagging live here so
every gate is hill-climbable through one uniform interface.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from ..schema import Document, GateResult, clamp

# A signal value at or above this counts as a fired defect flag.
_FLAG_AT = 0.5


class Gate:
    name: str = "gate"
    #: Ordered defect-signal names this gate computes.
    SIGNALS: List[str] = []
    #: Default non-negative weight per signal.
    DEFAULTS: Dict[str, float] = {}

    def __init__(self, weights: Dict[str, float] = None, threshold: float = 0.5):
        self.weights: Dict[str, float] = dict(self.DEFAULTS)
        if weights:
            self.weights.update(weights)
        self.threshold = float(threshold)

    # -- subclass hook ----------------------------------------------------- #
    def _signals(self, doc: Document) -> Dict[str, float]:
        """Return raw defect signals in [0,1] (1 == worst). Override me."""
        raise NotImplementedError

    # -- scoring ----------------------------------------------------------- #
    def quality(self, doc: Document) -> float:
        sig = self._signals(doc)
        return self._combine(sig)

    def _combine(self, sig: Dict[str, float]) -> float:
        quality = 1.0
        for k in self.SIGNALS:
            w = max(0.0, self.weights.get(k, 0.0))
            quality *= (1.0 - clamp(w * clamp(sig.get(k, 0.0))))
        return clamp(quality)

    def _channel_defect(self, k: str, sig: Dict[str, float]) -> float:
        return clamp(max(0.0, self.weights.get(k, 0.0)) * clamp(sig.get(k, 0.0)))

    def evaluate(self, doc: Document) -> GateResult:
        sig = self._signals(doc)
        q = self._combine(sig)
        flags = [k for k in self.SIGNALS if self._channel_defect(k, sig) >= _FLAG_AT]
        return GateResult(
            gate=self.name, quality=q, passed=q >= self.threshold,
            flags=flags, signals={k: round(sig.get(k, 0.0), 4) for k in self.SIGNALS},
        )

    # -- hill-climb parameter interface ------------------------------------ #
    @property
    def param_names(self) -> List[str]:
        return list(self.SIGNALS) + ["threshold"]

    def get_params(self) -> List[float]:
        """Flat parameter vector: [w_1 .. w_n, threshold]."""
        return [self.weights[k] for k in self.SIGNALS] + [self.threshold]

    def set_params(self, vector: Sequence[float]) -> "Gate":
        n = len(self.SIGNALS)
        for k, v in zip(self.SIGNALS, vector[:n]):
            self.weights[k] = max(0.0, float(v))
        if len(vector) > n:
            self.threshold = clamp(float(vector[n]), 0.05, 0.95)
        return self

    def bounds(self) -> List[tuple]:
        """(lo, hi) per parameter, for the offline proposer."""
        return [(0.0, 5.0)] * len(self.SIGNALS) + [(0.05, 0.95)]

    def clone(self) -> "Gate":
        return self.__class__(weights=dict(self.weights), threshold=self.threshold)


def combine_gate(results: Sequence[GateResult]) -> float:
    """Weakest-link aggregation of several gates' quality estimates."""
    return min((r.quality for r in results), default=1.0)
