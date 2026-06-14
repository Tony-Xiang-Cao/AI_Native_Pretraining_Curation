"""Distributional outlier detection over the heuristic feature vectors.

We fit a robust (median / MAD) location-scale summary **per source cohort** for
every feature, then flag any document whose robust z-score lands in the extreme
tail on one or more features. Per-cohort fitting matters: an academic-PDF
corpus and a forum-scrape corpus have very different "normal" digit fractions
and line lengths, so a single global threshold would drown real anomalies in
benign cross-source variation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from ..schema import Document, ProfileConfig, QualityProfile
from ..utils import mad, median, robust_z
from .heuristics import FEATURE_NAMES, text_features


class CohortOutlierDetector:
    """Fit per-cohort robust location/scale, then score documents for outliers.

    A cohort is keyed by ``Document.source``. Cohorts smaller than
    ``config.min_cohort`` fall back to the pooled (all-documents) statistics so
    tiny sources still get a reasonable reference distribution.
    """

    def __init__(self, config: Optional[ProfileConfig] = None,
                 features: Sequence[str] = FEATURE_NAMES):
        self.config = config or ProfileConfig()
        self.features = list(features)
        # cohort -> feature -> (median, mad)
        self._stats: Dict[str, Dict[str, tuple]] = {}
        self._pooled: Dict[str, tuple] = {}

    def fit(self, docs: Sequence[Document],
            feats: Optional[Sequence[Dict[str, float]]] = None) -> "CohortOutlierDetector":
        feats = feats if feats is not None else [text_features(d.text) for d in docs]
        by_cohort: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        for d, f in zip(docs, feats):
            by_cohort[d.source].append(f)

        self._pooled = self._fit_group(list(feats))
        self._stats = {c: self._fit_group(fs) for c, fs in by_cohort.items()}
        self._cohort_sizes = {c: len(fs) for c, fs in by_cohort.items()}
        return self

    def _fit_group(self, feats: List[Dict[str, float]]) -> Dict[str, tuple]:
        out: Dict[str, tuple] = {}
        for name in self.features:
            vals = [f.get(name, 0.0) for f in feats]
            med = median(vals)
            out[name] = (med, mad(vals, med))
        return out

    def score(self, doc: Document,
              feat: Optional[Dict[str, float]] = None) -> QualityProfile:
        feat = feat if feat is not None else text_features(doc.text)
        cohort = doc.source
        use_pooled = (
            cohort not in self._stats
            or self._cohort_sizes.get(cohort, 0) < self.config.min_cohort
        )
        stats = self._pooled if use_pooled else self._stats[cohort]

        flags: List[str] = []
        worst = 0.0
        for name in self.features:
            med, mad_val = stats.get(name, (0.0, 0.0))
            z = robust_z(feat.get(name, 0.0), med, mad_val)
            if abs(z) >= self.config.z_threshold:
                flags.append(name)
                worst = max(worst, abs(z))
        return QualityProfile(
            doc_id=doc.id, features=feat, outlier_flags=flags, outlier_score=worst
        )

    def score_all(self, docs: Sequence[Document],
                  feats: Optional[Sequence[Dict[str, float]]] = None) -> List[QualityProfile]:
        feats = feats if feats is not None else [text_features(d.text) for d in docs]
        return [self.score(d, f) for d, f in zip(docs, feats)]


def profile_corpus(docs: Sequence[Document],
                   config: Optional[ProfileConfig] = None) -> List[QualityProfile]:
    """One-shot convenience: fit on the corpus and score every document.

    (Fitting and scoring on the same corpus is the intended "screen this batch"
    use; for streaming, fit once on a reference window and reuse the detector.)
    """
    feats = [text_features(d.text) for d in docs]
    det = CohortOutlierDetector(config).fit(docs, feats)
    return det.score_all(docs, feats)
