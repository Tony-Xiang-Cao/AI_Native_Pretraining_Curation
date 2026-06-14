"""Harness-agnostic operating loop: ``AgentHarness`` protocol + ``Routine``.

The harness owns *scheduling and side-effects*; the routine owns *logic*. One
``tick()`` is a full Observe→Orient→Decide→Act pass, and the same ``Routine``
object runs unchanged whether it is driven by a plain ``cron`` line, a persistent
cloud scheduler, a headless agent CLI under any external scheduler, or a
different coding-agent runner — only the harness adapter changes.

Scheduling/execution options for an agent-CLI backend:
  * **cloud routines** (persistent) — the production "run daily" scheduler.
  * **in-session loops** — self-paced; good for interactive babysitting, not
    production.
  * **headless CLI** (``<cli> -p "…" --output-format stream-json``) — drive ONE
    tick from OS cron / CI on hosts without a cloud scheduler.
  * **hooks** — emit alerts/ledger side-effects.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Sequence

from ..pipeline import CurationLoop
from ..schema import Document
from ..utils import mean
from .ledger import ProvenanceLedger
from .spc import StreamMonitor


@dataclass
class TickResult:
    day: int
    ok: bool
    metrics: dict = field(default_factory=dict)     # throughput, quality
    alerts: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# The harness protocol and three reference adapters
# --------------------------------------------------------------------------- #

class AgentHarness(Protocol):
    def schedule(self, cron: str, routine_id: str) -> str: ...
    def run_step(self, prompt: str, headless: bool = True) -> str: ...
    def report(self, result: TickResult) -> None: ...
    def now(self) -> float: ...


class LocalCronHarness:
    """Zero-dependency backend: a crontab line + JSONL ledger + stdout. No agent."""

    def __init__(self, ledger: Optional[ProvenanceLedger] = None, verbose: bool = False):
        self.ledger = ledger or ProvenanceLedger()
        self.verbose = verbose
        self._clock = 0.0

    def schedule(self, cron: str, routine_id: str) -> str:
        return f"{cron} python -m autocurate run --routine {routine_id}"

    def run_step(self, prompt: str, headless: bool = True) -> str:
        # No LLM here: the deterministic offline hill-climb is invoked in-process
        # by the routine itself. This hook exists for parity with agent backends.
        return f"[local] noted: {prompt[:60]}"

    def report(self, result: TickResult) -> None:
        self.ledger.append_summary({
            "day": result.day, "ok": result.ok, **result.metrics,
            "alerts": result.alerts, "actions": result.actions,
        })
        if self.verbose:
            print(f"day {result.day}: {result.metrics} alerts={result.alerts} "
                  f"actions={result.actions}")

    def now(self) -> float:
        self._clock += 1.0
        return self._clock


class AgentCliHarness(LocalCronHarness):
    """Agent-CLI backend: schedule via a cloud routine, act via a headless CLI."""

    def __init__(self, ledger=None, model: str = "claude-haiku-4-5", verbose=False,
                 cli: str = "claude"):
        super().__init__(ledger, verbose)
        self.model = model
        self.cli = cli

    def schedule(self, cron: str, routine_id: str) -> str:
        # In practice: register a cloud routine with this cron + a prompt that
        # runs `python -m autocurate run --routine <id>`. Returned string documents it.
        return (f"{self.cli} routine create --cron '{cron}' "
                f"--prompt 'run autocurate routine {routine_id}'")

    def run_step(self, prompt: str, headless: bool = True) -> str:
        try:
            out = subprocess.run(
                [self.cli, "-p", prompt, "--output-format", "stream-json",
                 "--model", self.model],
                capture_output=True, text=True, timeout=600,
            )
            return out.stdout or out.stderr
        except (FileNotFoundError, subprocess.SubprocessError):
            return f"[{self.cli} CLI unavailable; no-op]"


# --------------------------------------------------------------------------- #
# The crawler-monitoring routine
# --------------------------------------------------------------------------- #

class CrawlMonitorRoutine:
    """Track daily throughput + extraction quality; act on quality drift.

    ``remediator`` (optional) is the Act invoked on a *downward quality* alarm —
    e.g. re-extract / clean the flagged shards, or a hill-climb episode that
    re-tunes the gates. Once triggered it is applied to subsequent days' input
    until quality returns in-control, modelling closed-loop recovery.
    """

    def __init__(self, loop: CurationLoop, ledger: Optional[ProvenanceLedger] = None,
                 warmup: int = 10,
                 quality_monitor: Optional[StreamMonitor] = None,
                 throughput_monitor: Optional[StreamMonitor] = None,
                 remediator: Optional[Callable[[Sequence[Document]], List[Document]]] = None,
                 recovery_ratio: float = 0.95,
                 routine_id: str = "crawl-monitor"):
        self.loop = loop
        self.ledger = ledger or ProvenanceLedger()
        self.warmup = warmup
        self.q_mon = quality_monitor or StreamMonitor(watch="down")
        self.t_mon = throughput_monitor or StreamMonitor(watch="two_sided")
        self.remediator = remediator
        self.recovery_ratio = recovery_ratio
        self.routine_id = routine_id
        self._baseline_q: List[float] = []
        self._baseline_t: List[float] = []
        self._fitted = False
        self._remediating = False
        self.day = -1

    def tick(self, harness: AgentHarness, day_docs: Sequence[Document],
             fetch_time: str = "1970-01-01T00:00:00Z") -> TickResult:
        self.day += 1
        if self._remediating and self.remediator is not None:
            day_docs = self.remediator(day_docs)

        decisions, records = self.loop.curate_batch(day_docs, fetch_time)
        for rec in records:
            self.ledger.append_record(rec)

        throughput = float(len(day_docs))
        quality = mean([r.gate_quality for r in records]) if records else 1.0

        alerts: List[str] = []
        actions: List[str] = []

        if self.day < self.warmup:
            self._baseline_q.append(quality)
            self._baseline_t.append(throughput)
        else:
            if not self._fitted:
                self.q_mon.fit(self._baseline_q)
                self.t_mon.fit(self._baseline_t)
                self._fitted = True
            qs = self.q_mon.update(quality, self.day)
            ts = self.t_mon.update(throughput, self.day)
            recovered = self._remediating and quality >= self.recovery_ratio * self.q_mon.mu0
            if recovered:
                # remediation restored quality: clear the alarm state (keep the
                # better extractor in place) so the monitor stops latching.
                self.q_mon.reset_state()
                actions.append("recovered")
            elif qs.alarm and qs.direction == "down":
                alerts.append("quality_drift_down")
                actions.append("hillclimb")
                harness.run_step(
                    f"Quality regression on {self.routine_id}: re-extract/re-tune gates.")
                self._remediating = True
            if ts.alarm:
                alerts.append(f"throughput_drift_{ts.direction}")

        result = TickResult(self.day, ok=not alerts,
                            metrics={"throughput": throughput, "quality": round(quality, 4)},
                            alerts=alerts, actions=actions)
        harness.report(result)
        return result
