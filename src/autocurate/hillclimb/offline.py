"""Deterministic, offline candidate proposer (no API key, fully reproducible).

Mutates a gate's parameter vector with bounded Gaussian steps on a random
subset of coordinates — a simple, seedable evolutionary move. This is the
default proposer so the whole self-improvement loop runs in CI.
"""

from __future__ import annotations

import random
from typing import List


class OfflineProposer:
    """Propose mutated copies of a gate by perturbing its parameter vector."""

    def __init__(self, step: float = 0.08, seed: int = 7):
        self.step = step
        self.rng = random.Random(seed)

    def propose(self, gate) -> "object":
        params = gate.get_params()
        bounds = gate.bounds()
        n = len(params)
        # mutate a random subset (at least one) of coordinates
        k = max(1, self.rng.randint(1, max(1, n // 2)))
        idx = self.rng.sample(range(n), k)
        new = list(params)
        for i in idx:
            lo, hi = bounds[i]
            span = hi - lo
            new[i] = min(hi, max(lo, new[i] + self.rng.gauss(0.0, self.step * span)))
        cand = gate.clone()
        cand.set_params(new)
        return cand

    def propose_many(self, gate, k: int) -> List["object"]:
        return [self.propose(gate) for _ in range(k)]
