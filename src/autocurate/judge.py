"""Semantic adjudication tier — a thin adapter over the sibling ``judgecurate``.

AutoCurate does **not** re-implement LLM-as-Judge curation. The cheap tiers
(profiler, extraction gates, a surface governance score) auto-decide the easy
cases; only the genuinely ambiguous middle band is escalated to ``judgecurate``,
whose calibrated, risk-penalized five-attribute decision is reused unchanged.
This is the only module that imports ``judgecurate``, and it degrades to a
conservative "escalate → review" stub if the package is not installed, so the
pure-stdlib core still runs in CI.

The decision-label scheme (FILTER=0 / REVIEW=1 / RETAIN=2) is shared with
``judgecurate`` by construction, so a ``judgecurate.Decision`` maps to an
``autocurate.Decision`` with no translation.
"""

from __future__ import annotations

import re

from .profile.heuristics import text_features
from .schema import Decision, Document, REVIEW, clamp

_NUM_RE = re.compile(r"\b\d[\d,.]*\b")
_ENTITY_RE = re.compile(r"(?<=[a-z]\s)[A-Z][a-z]{2,}")   # capitalized non-sentence-start word

try:  # optional dependency
    from judgecurate import CurationPipeline
    from judgecurate.judges import get_judge
    _HAVE_JC = True
except Exception:  # pragma: no cover - exercised only without the extra
    _HAVE_JC = False


# Obvious-spam blocklist the *cheap* tier can filter without a judge call.
_RISK_WORDS = (
    "click here", "free money", "act now", "buy now", "viagra", "lottery",
    "wire the payment", "miracle", "hot singles", "download this crack",
    "claim your prize", "no diet", "guaranteed to cure",
)


def heuristic_governance_score(text: str) -> float:
    """Cheap surface estimate of content governance value in ``[0, 1]``.

    Deliberately weaker than the judge: it reads only surface statistics
    (function-word density, lexical diversity, length, repetition) plus an
    obvious-spam blocklist. Its job is to *route* — confidently keep clearly
    good text and drop clearly bad text — and escalate everything else.
    """
    f = text_features(text)
    n_words = max(1.0, f["token_length"])
    natural = min(1.0, f["stopword_fraction"] / 0.30)
    diversity = f["type_token_ratio"]
    length = min(1.0, f["token_length"] / 120.0)
    # surface knowledge-density proxy: numbers + proper-noun entities per ~8 words
    facts = len(_NUM_RE.findall(text)) + len(_ENTITY_RE.findall(text))
    knowledge = min(1.0, facts / (n_words / 8.0))
    repetition = f["bigram_repetition"] + f["dup_line_fraction"] + f["top_ngram_fraction"]
    base = (0.28 * natural + 0.16 * diversity + 0.20 * length
            + 0.26 * knowledge - 0.5 * repetition)
    low = text.lower()
    hits = sum(1 for w in _RISK_WORDS if w in low)
    base -= 0.6 * min(1.0, hits)
    return clamp(base)


class JudgeAdapter:
    """Adjudicate the ambiguous middle of the cascade with ``judgecurate``."""

    def __init__(self, backend: str = "heuristic", n_rounds: int = 3, **judge_kwargs):
        if not _HAVE_JC:
            raise RuntimeError(
                "Semantic adjudication needs the judge extra: pip install autocurate[judge]"
            )
        judge = get_judge(backend, **judge_kwargs)
        self.pipe = CurationPipeline(judge=judge)
        self.pipe.config.n_rounds = n_rounds

    def adjudicate(self, doc: Document, gate_quality: float = 1.0) -> Decision:
        d = self.pipe.curate_text(doc.text)
        return Decision(
            doc_id=doc.id, label=d.label, score=d.score, stage="judge",
            reasons=[d.reason] if d.reason else [], gate_quality=gate_quality,
            risk=d.risk_norm,
        )


def stub_review(doc: Document, gate_quality: float = 1.0) -> Decision:
    """Fallback used when ``judgecurate`` is absent: route ambiguous → review."""
    return Decision(doc_id=doc.id, label=REVIEW, score=0.5, stage="judge-stub",
                    reasons=["judgecurate not installed; escalated to human review"],
                    gate_quality=gate_quality)
