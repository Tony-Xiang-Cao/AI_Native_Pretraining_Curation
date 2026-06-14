"""Verifier-gated self-improvement: propose -> evaluate -> verify -> select.

A capability (here, an extraction gate's parameter vector) is treated as the
*evolving artifact*. Each round proposes candidate configurations, evaluates
them against the mutation oracle, and — in the ``verified`` regime — adopts a
candidate only if it passes the held-out, guard-protected accept rule. The
``naive`` regime instead climbs the in-sample (train-vocabulary) metric with no
guard, and reward-hacks; comparing the two is experiment E3. The proposer is
pluggable: a deterministic offline search by default, or an LLM agent.
"""

from __future__ import annotations

from .base import HillclimbResult, hillclimb
from .offline import OfflineProposer

__all__ = ["hillclimb", "HillclimbResult", "OfflineProposer"]
