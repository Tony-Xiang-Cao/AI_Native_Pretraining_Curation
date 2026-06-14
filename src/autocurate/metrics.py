"""Evaluation metrics (pure stdlib).

Macro-F1 of the three-way curation decision and simple agreement, plus
precision/recall/F1 helpers shared by the experiments. Kept dependency-free and
deterministic so the committed ``results/*.json`` are reproducible and the test
suite can assert them.
"""

from __future__ import annotations

from typing import Dict, Sequence

from .schema import FILTER, RETAIN, REVIEW
from .utils import mean


def agreement(preds: Sequence[int], golds: Sequence[int]) -> float:
    if not preds:
        return 0.0
    return sum(1 for p, g in zip(preds, golds) if p == g) / len(preds)


def macro_f1(preds: Sequence[int], golds: Sequence[int],
             classes: Sequence[int] = (FILTER, REVIEW, RETAIN)) -> float:
    f1s = []
    for c in classes:
        tp = sum(1 for p, g in zip(preds, golds) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, golds) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, golds) if p != c and g == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return mean(f1s)


def binary_prf(preds: Sequence[int], golds: Sequence[int]) -> Dict[str, float]:
    """Precision/recall/F1 for a binary (1 = positive) labelling."""
    tp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(preds, golds) if p == 0 and g == 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}
