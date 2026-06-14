"""Mutation-oracle evaluation, held-out floor/upper bracketing, and the
self-improvement accept rule.

A gate predicts "defect" when it does *not* pass a document (``quality <
threshold``). Against the corruption answer key this yields ordinary
precision / recall / F1 — but *verifiable*, because we created the labels. We
report a (held-out floor, train upper-bound) pair, and we accept a hill-climb
candidate only if its held-out F1 improves with a margin whose **lower
confidence bound** (a small-sample normal/t interval over the per-seed
improvements) clears ``delta`` **and** its false-positive rate on a clean guard
set does not rise by more than ``eps``. That FPR clause is the anti-reward-hack
lock: a candidate cannot win by flagging everything.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..schema import Document
from ..utils import mean, std
from .corruptions import build_mutation_set

# Two-sided 95% Student-t multipliers t_{.975, n-1} for small n (the per-seed
# improvement sample is tiny, so the z=1.96 normal multiplier is too optimistic).
_T_975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
          8: 2.365, 9: 2.306, 10: 2.262}


def _t_mult(n: int) -> float:
    return _T_975.get(n, 1.96)

Gate = "autocurate.extract.base.Gate"   # structural; avoid import cycle


@dataclass
class GateScore:
    precision: float
    recall: float
    f1: float
    fpr: float
    n: int


def _score_labeled(gate, labeled: Sequence[Tuple[Document, int]]) -> GateScore:
    tp = fp = fn = tn = 0
    for doc, defect in labeled:
        pred_defect = not gate.evaluate(doc).passed
        if defect and pred_defect:
            tp += 1
        elif defect and not pred_defect:
            fn += 1
        elif not defect and pred_defect:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return GateScore(precision, recall, f1, fpr, len(labeled))


def evaluate_gate(gate, clean_docs: Sequence[Document], modality: str,
                  seed: int = 7, which: str = "heldout") -> GateScore:
    rng = random.Random(seed)
    labeled = build_mutation_set(clean_docs, modality, rng, which)
    return _score_labeled(gate, labeled)


def _f1_over_seeds(gate, clean_docs, modality, seeds, which) -> List[float]:
    return [evaluate_gate(gate, clean_docs, modality, s, which).f1 for s in seeds]


def floor_and_upper(gate, clean_docs: Sequence[Document], modality: str,
                    seeds: Sequence[int] = range(5)) -> Tuple[float, float]:
    """(held-out-floor F1, train-upper-bound F1), each averaged over seeds."""
    floor = mean(_f1_over_seeds(gate, clean_docs, modality, seeds, "heldout"))
    upper = mean(_f1_over_seeds(gate, clean_docs, modality, seeds, "train"))
    return floor, upper


def guard_fpr(gate, clean_docs: Sequence[Document]) -> float:
    """Fraction of known-clean documents the gate wrongly flags."""
    if not clean_docs:
        return 0.0
    flagged = sum(1 for d in clean_docs if not gate.evaluate(d).passed)
    return flagged / len(clean_docs)


def verified_objective(gate, clean_docs: Sequence[Document], modality: str,
                       seeds: Sequence[int] = range(3), which: str = "heldout",
                       fpr_penalty: float = 0.5) -> float:
    """Scalar hill-climb objective: held-out F1 minus a clean-guard FPR penalty."""
    f1 = mean(_f1_over_seeds(gate, clean_docs, modality, seeds, which))
    return f1 - fpr_penalty * guard_fpr(gate, clean_docs)


@dataclass
class AcceptDecision:
    accept: bool
    delta_f1: float          # mean held-out F1 improvement (candidate - incumbent)
    lower_bound: float       # small-sample t lower bound of the improvement
    delta_fpr: float         # guard FPR change (candidate - incumbent)
    reason: str


def accept_candidate(candidate, incumbent, eval_docs: Sequence[Document], modality: str,
                     guard_docs: Sequence[Document], seeds: Sequence[int] = range(5),
                     delta: float = 0.005, eps: float = 0.005) -> AcceptDecision:
    """Verifier-gated acceptance: held-out improvement + clean-guard no-regression.

    The held-out F1 improvement is evaluated on ``eval_docs`` (ideally disjoint
    from the documents the climber optimizes on); its lower confidence bound uses
    a small-sample Student-t multiplier ``t_{.975, n-1}`` over the per-seed
    improvements. The guard FPR is measured on ``guard_docs``.
    """
    cand = _f1_over_seeds(candidate, eval_docs, modality, seeds, "heldout")
    inc = _f1_over_seeds(incumbent, eval_docs, modality, seeds, "heldout")
    diffs = [c - i for c, i in zip(cand, inc)]
    m = mean(diffs)
    n = len(diffs)
    sem = std(diffs) / (n ** 0.5) if n > 1 else 0.0
    lb = m - _t_mult(n) * sem
    dfpr = guard_fpr(candidate, guard_docs) - guard_fpr(incumbent, guard_docs)
    ok = (lb > delta) and (dfpr <= eps)
    if ok:
        reason = "held-out F1 up (t-significant) and guard FPR not worse"
    elif lb <= delta:
        reason = "held-out gain not significant"
    else:
        reason = "rejected: guard FPR regression (reward-hack lock)"
    return AcceptDecision(ok, m, lb, dfpr, reason)
