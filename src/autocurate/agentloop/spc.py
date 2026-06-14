"""Statistical process control for daily throughput and quality streams.

We run two complementary detectors per stream: an **EWMA** chart (smooths noise,
flags gradual drift) and a **CUSUM** chart (fast on small sustained shifts). The
baseline mean/scale is estimated robustly from a clean warm-up window
(σ = 1.4826·MAD), so a few bad days during warm-up do not poison the control
limits. Quality is monitored for *downward* drift (degradation) and throughput
two-sided (an outage or a spam flood are both worth an alert).

Formulas (per stream, daily sample xₜ; baseline μ₀, σ):
  EWMA:  zₜ = λxₜ + (1−λ)zₜ₋₁;  σ_z = σ·sqrt(λ/(2−λ)·(1−(1−λ)^{2t}));  alarm |zₜ−μ₀|>Lσ_z
  CUSUM: Sₜ⁺ = max(0, Sₜ₋₁⁺ + (xₜ−μ₀−k));  Sₜ⁻ = max(0, Sₜ₋₁⁻ − (xₜ−μ₀+k));  alarm S>h
         with slack k = kσ·σ and decision interval h = hσ·σ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..utils import mad, median


@dataclass
class DriftSignal:
    day: int
    value: float
    ewma: float
    cusum_pos: float
    cusum_neg: float
    ewma_alarm: bool
    cusum_alarm: bool
    direction: str          # "up" | "down" | "none"

    @property
    def alarm(self) -> bool:
        return self.ewma_alarm or self.cusum_alarm


class StreamMonitor:
    """EWMA + CUSUM monitor for one daily metric stream."""

    def __init__(self, alpha: float = 0.2, L: float = 3.0,
                 k_sigma: float = 0.5, h_sigma: float = 4.0,
                 watch: str = "two_sided"):
        self.alpha = alpha
        self.L = L
        self.k_sigma = k_sigma
        self.h_sigma = h_sigma
        self.watch = watch          # "two_sided" | "down" | "up"
        self.mu0 = 0.0
        self.sigma = 1.0
        self.reset_state()

    def reset_state(self) -> None:
        self.z = self.mu0
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.t = 0

    def fit(self, baseline: Sequence[float]) -> "StreamMonitor":
        vals = list(baseline)
        self.mu0 = median(vals)
        self.sigma = max(1e-9, 1.4826 * mad(vals, self.mu0))
        # guard against a degenerate near-constant warm-up
        self.sigma = max(self.sigma, 0.02 * (abs(self.mu0) + 1e-6))
        self.reset_state()
        return self

    def update(self, x: float, day: int = -1) -> DriftSignal:
        self.t += 1
        self.z = self.alpha * x + (1 - self.alpha) * self.z
        sigma_z = self.sigma * math.sqrt(
            self.alpha / (2 - self.alpha) * (1 - (1 - self.alpha) ** (2 * self.t))
        )
        dev = self.z - self.mu0
        ewma_alarm = abs(dev) > self.L * sigma_z

        k = self.k_sigma * self.sigma
        h = self.h_sigma * self.sigma
        self.s_pos = max(0.0, self.s_pos + (x - self.mu0 - k))
        self.s_neg = max(0.0, self.s_neg - (x - self.mu0 + k))
        cusum_alarm = (self.s_pos > h) or (self.s_neg > h)

        direction = "none"
        if dev < 0 or self.s_neg > h:
            direction = "down"
        elif dev > 0 or self.s_pos > h:
            direction = "up"

        # restrict alarms to the watched direction
        if self.watch == "down":
            ewma_alarm = ewma_alarm and dev < 0
            cusum_alarm = self.s_neg > h
        elif self.watch == "up":
            ewma_alarm = ewma_alarm and dev > 0
            cusum_alarm = self.s_pos > h

        return DriftSignal(day, x, self.z, self.s_pos, self.s_neg,
                           ewma_alarm, cusum_alarm, direction)

    def state(self) -> dict:
        return {"mu0": self.mu0, "sigma": self.sigma, "z": self.z,
                "s_pos": self.s_pos, "s_neg": self.s_neg, "t": self.t}
