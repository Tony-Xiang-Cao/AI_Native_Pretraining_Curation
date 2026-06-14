"""The harness-agnostic autonomous operating loop.

One ``Routine.tick()`` is one Observe→Orient→Decide→Act pass over a day's
crawler output: curate it, append provenance, update statistical-process-control
monitors on throughput and quality, and — if quality drift is detected —
trigger a self-improvement (hill-climb) episode. The same ``Routine`` runs
unchanged under a plain ``cron`` loop, cloud routines, or any other
coding-agent harness, because all scheduling and side-effects go through the
small ``AgentHarness`` protocol.
"""

from __future__ import annotations

from .ledger import ProvenanceLedger
from .routine import (
    AgentHarness,
    AgentCliHarness,
    CrawlMonitorRoutine,
    LocalCronHarness,
    TickResult,
)
from .spc import DriftSignal, StreamMonitor

__all__ = [
    "StreamMonitor", "DriftSignal", "ProvenanceLedger",
    "AgentHarness", "LocalCronHarness", "AgentCliHarness",
    "CrawlMonitorRoutine", "TickResult",
]
