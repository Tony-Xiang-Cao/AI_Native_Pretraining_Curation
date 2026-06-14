"""The hill-climbing search loop, in three regimes, with disjoint document sets.

The climber optimizes a gate's parameter vector on ``climb_docs`` and is
evaluated / guarded on ``eval_docs`` and ``guard_docs`` that are **disjoint
clean documents** (different corpus draws), so the reported held-out F1 and
clean-guard FPR are genuinely out-of-sample, not the documents the climber saw.

Regimes:
- **verified** (proposed): rank candidates by the held-out verified objective on
  ``climb_docs``, then adopt the best only if it passes the guard-protected
  accept rule on the disjoint ``eval_docs`` / ``guard_docs``.
- **naive_recall** (ablation): maximize **in-sample defect recall** on the
  train-vocabulary mutations, no guard, adopt any improvement. The canonical
  Goodhart setup — an objective that rewards catching defects but never penalizes
  destroying clean data, so the optimizer **raises the accept threshold** (a
  document is flagged when ``quality < threshold``) until it over-flags.
- **naive_f1** (ablation): maximize **in-sample balanced F1** on the train
  mutations, still no clean guard. A fairer un-guarded objective than recall;
  whether it also reward-hacks is an empirical question E3 answers rather than
  assumes.

Every regime logs, each iteration: in-sample recall on the train mutations, the
out-of-sample held-out F1 on ``eval_docs``, and the clean-guard FPR on
``guard_docs``. The reward-hack is visible as the guard-FPR gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..schema import Document, HillclimbConfig
from ..utils import mean
from ..verify.harness import accept_candidate, evaluate_gate, guard_fpr, verified_objective
from .offline import OfflineProposer


@dataclass
class HillclimbResult:
    regime: str
    gate: object
    trajectory: List[Dict[str, float]] = field(default_factory=list)

    @property
    def final(self) -> Dict[str, float]:
        return self.trajectory[-1] if self.trajectory else {}


def _mean_f1(gate, docs, modality, seeds, which) -> float:
    return mean(evaluate_gate(gate, docs, modality, s, which).f1 for s in seeds)


def _mean_recall(gate, docs, modality, seeds, which) -> float:
    return mean(evaluate_gate(gate, docs, modality, s, which).recall for s in seeds)


def hillclimb(gate0, climb_docs: Sequence[Document], modality: str,
              regime: str = "verified", config: Optional[HillclimbConfig] = None,
              eval_docs: Optional[Sequence[Document]] = None,
              guard_docs: Optional[Sequence[Document]] = None,
              opt_seeds: Sequence[int] = (0, 1, 2, 3, 4),
              test_seeds: Sequence[int] = (20, 21, 22, 23, 24),
              proposer=None) -> HillclimbResult:
    cfg = config or HillclimbConfig()
    eval_docs = eval_docs if eval_docs is not None else climb_docs
    guard_docs = guard_docs if guard_docs is not None else eval_docs
    proposer = proposer or OfflineProposer(step=cfg.step, seed=cfg.seed)
    incumbent = gate0.clone()

    def snapshot(it: int) -> Dict[str, float]:
        return {
            "iter": it,
            "recall_train": _mean_recall(incumbent, climb_docs, modality, opt_seeds, "train"),
            "f1_heldout": _mean_f1(incumbent, eval_docs, modality, test_seeds, "heldout"),
            "guard_fpr": guard_fpr(incumbent, guard_docs),
        }

    traj: List[Dict[str, float]] = [snapshot(0)]

    for it in range(1, cfg.iterations + 1):
        candidates = proposer.propose_many(incumbent, cfg.population)
        if regime == "naive_recall":
            obj = lambda c: _mean_recall(c, climb_docs, modality, opt_seeds, "train")  # noqa: E731
            best = max(candidates, key=obj)
            if obj(best) > obj(incumbent):
                incumbent = best
        elif regime == "naive_f1":
            obj = lambda c: _mean_f1(c, climb_docs, modality, opt_seeds, "train")  # noqa: E731
            best = max(candidates, key=obj)
            if obj(best) > obj(incumbent):
                incumbent = best
        else:  # verified
            scored = [(c, verified_objective(c, climb_docs, modality, opt_seeds, "heldout"))
                      for c in candidates]
            best, _ = max(scored, key=lambda t: t[1])
            decision = accept_candidate(
                best, incumbent, eval_docs, modality, guard_docs,
                seeds=opt_seeds, delta=0.005, eps=cfg.guard_tolerance,
            )
            if decision.accept:
                incumbent = best
        traj.append(snapshot(it))

    return HillclimbResult(regime=regime, gate=incumbent, trajectory=traj)
