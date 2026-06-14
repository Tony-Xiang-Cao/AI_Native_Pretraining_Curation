"""The verification harness — the scientific heart of AutoCurate.

Every quality claim is checked against ground truth we *manufacture*: we take a
clean document, inject a defect whose type and parameters we record, and ask
whether the gate catches it. Because the corruption log is the answer key, a
reference-free estimator becomes a classifier with a known label, so its
precision / recall / F1 are *verifiable* rather than asserted. Held-out
corruption vocabularies force generalization over read-back (the floor), and
the accept rule gates self-improvement on held-out, guard-protected gains.
"""

from __future__ import annotations

from .corruptions import (
    CORRUPTORS,
    DefectRecord,
    build_mutation_set,
    corrupt,
)
from .harness import (
    AcceptDecision,
    GateScore,
    accept_candidate,
    evaluate_gate,
    floor_and_upper,
    guard_fpr,
    verified_objective,
)

__all__ = [
    "DefectRecord", "CORRUPTORS", "corrupt", "build_mutation_set",
    "GateScore", "evaluate_gate", "floor_and_upper", "guard_fpr",
    "verified_objective", "accept_candidate", "AcceptDecision",
]
