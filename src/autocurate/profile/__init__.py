"""Cheap, reference-free text profiling and distributional outlier detection.

This is the first, model-free pass of the loop: it computes a vector of
transparent text heuristics for every document (gzip ratio, token length,
lexical diversity, repetition, character-class fractions, ...) and flags
documents that sit in the extreme tail of their *source cohort* via robust
(median/MAD) z-scores. It needs no API key and runs in microseconds per
document.
"""

from __future__ import annotations

from .heuristics import FEATURE_NAMES, text_features
from .outliers import CohortOutlierDetector, profile_corpus

__all__ = ["FEATURE_NAMES", "text_features", "CohortOutlierDetector", "profile_corpus"]
