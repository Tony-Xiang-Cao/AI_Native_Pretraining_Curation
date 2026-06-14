"""Core data structures for the AutoCurate operating loop.

Everything here is **pure standard library** so the engine runs anywhere — a
laptop, a CI box, or a fresh agent sandbox — with zero third-party
dependencies. The optional ``judgecurate`` adjudicator and real LLM backends
are loaded lazily and only when explicitly requested.

The decision labels deliberately mirror the sibling ``judgecurate`` package so
the two pipelines compose without translation:

    FILTER = 0   REVIEW = 1   RETAIN = 2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Curation decision labels (shared with judgecurate)
# --------------------------------------------------------------------------- #

FILTER: int = 0   # drop the document
REVIEW: int = 1   # ambiguous -> escalate (to the LLM judge, or to a human)
RETAIN: int = 2   # keep the document

LABEL_NAMES: Dict[int, str] = {FILTER: "filter", REVIEW: "review", RETAIN: "retain"}

#: The four extraction "modalities" AutoCurate gates.
MODALITIES: List[str] = ["html", "ocr", "json", "text"]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp ``x`` into ``[lo, hi]`` (the unit interval by default)."""
    return max(lo, min(hi, float(x)))


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #

@dataclass
class Document:
    """A single unit of pretraining data flowing through the loop.

    ``text`` is the *extracted* text the language model would actually train
    on. ``raw`` (optional) holds the pre-extraction artifact — HTML markup, an
    OCR engine's raw dump, or a JSON record's source string — which the
    extraction gates inspect to estimate extraction quality reference-free.
    """

    id: str
    text: str
    source: str = "unknown"          # provenance: which crawl / feed / corpus
    modality: str = "text"           # one of MODALITIES
    raw: Optional[str] = None        # pre-extraction artifact, if available
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            self.modality = "text"


# --------------------------------------------------------------------------- #
# Quality profile (cheap heuristics + outlier flags)
# --------------------------------------------------------------------------- #

@dataclass
class QualityProfile:
    """Cheap, reference-free text features for one document.

    ``features`` is the raw heuristic vector (e.g. gzip_ratio, token_length,
    type_token_ratio, ...). ``outlier_flags`` names the features whose robust
    z-score put this document in the extreme tail of its source cohort, and
    ``outlier_score`` is the max absolute robust-z over flagged features.
    """

    doc_id: str
    features: Dict[str, float] = field(default_factory=dict)
    outlier_flags: List[str] = field(default_factory=list)
    outlier_score: float = 0.0

    @property
    def is_outlier(self) -> bool:
        return bool(self.outlier_flags)


# --------------------------------------------------------------------------- #
# Extraction-gate result
# --------------------------------------------------------------------------- #

@dataclass
class GateResult:
    """The verdict of one extraction-quality gate on one document.

    ``quality`` is a reference-free estimate in ``[0, 1]`` (1 == clean
    extraction). ``passed`` is ``quality >= gate.threshold``. ``flags`` are
    human-readable defect tags (e.g. ``"boilerplate_leak"``, ``"ocr_garble"``)
    and ``signals`` the raw numeric evidence behind the score.
    """

    gate: str
    quality: float
    passed: bool
    flags: List[str] = field(default_factory=list)
    signals: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Final curation decision
# --------------------------------------------------------------------------- #

@dataclass
class Decision:
    """The end-to-end curation decision for one document."""

    doc_id: str
    label: int                       # FILTER / REVIEW / RETAIN
    score: float = 0.0               # comprehensive governance score in [0, 1]
    stage: str = "heuristic"         # which stage decided: heuristic|gate|judge
    reasons: List[str] = field(default_factory=list)
    gate_quality: float = 1.0        # min extraction-gate quality seen
    risk: float = 0.0                # risk penalty in [0, 1] (from the judge)

    @property
    def label_name(self) -> str:
        return LABEL_NAMES.get(self.label, "review")


# --------------------------------------------------------------------------- #
# Provenance ledger record  ("traceable sources" + "divisible responsibility")
# --------------------------------------------------------------------------- #

@dataclass
class ProvenanceRecord:
    """One append-only ledger line: who handled this document, and how.

    Operationalizes two of the four governance directions from the source
    framework: *traceable sources* (source/fetch_time/extractor) and
    *divisible responsibility* (responsible_stage records which stage made the
    final call, so a downstream defect can be traced to an accountable step).
    """

    doc_id: str
    source: str
    fetch_time: str                  # ISO8601 string (passed in; no wall-clock)
    modality: str
    extractor: str                   # extractor name + version that produced text
    gate_quality: float
    decision: str                    # label_name
    responsible_stage: str           # heuristic|gate|judge|human
    config_hash: str = ""            # hash of the config that decided this doc

    def to_json(self) -> Dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "fetch_time": self.fetch_time,
            "modality": self.modality,
            "extractor": self.extractor,
            "gate_quality": round(self.gate_quality, 4),
            "decision": self.decision,
            "responsible_stage": self.responsible_stage,
            "config_hash": self.config_hash,
        }


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class ProfileConfig:
    """Outlier detection thresholds for the heuristic profiler."""

    z_threshold: float = 3.5         # robust-z (MAD) tail cutoff
    min_cohort: int = 12             # need this many docs in a cohort to flag


@dataclass
class GateConfig:
    """A single extraction gate's accept threshold (the hill-climb target)."""

    threshold: float = 0.5           # quality >= threshold => passed


@dataclass
class CascadeConfig:
    """Routing thresholds for the heuristic -> judge cascade.

    Documents with a heuristic governance score above ``retain_above`` are kept
    without spending a judge call; below ``filter_below`` they are dropped; the
    ambiguous middle band is escalated to the LLM judge.
    """

    retain_above: float = 0.62
    filter_below: float = 0.30


@dataclass
class HillclimbConfig:
    """Self-improvement loop settings (verifier-gated)."""

    iterations: int = 40
    population: int = 8              # candidates proposed per round (offline)
    step: float = 0.08               # mutation scale for offline proposer
    guard_tolerance: float = 0.0     # max allowed regression on the guard set
    seed: int = 7


@dataclass
class LoopConfig:
    """Daily operating-loop / drift-detection settings."""

    ewma_alpha: float = 0.3          # EWMA smoothing for quality/throughput
    cusum_k: float = 0.5             # CUSUM slack (in MADs)
    cusum_h: float = 4.0             # CUSUM decision interval (in MADs)
    drift_triggers_hillclimb: bool = True


@dataclass
class AutoCurateConfig:
    """Top-level configuration for the whole operating loop."""

    profile: ProfileConfig = field(default_factory=ProfileConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    gates: Dict[str, GateConfig] = field(
        default_factory=lambda: {m: GateConfig() for m in ("html", "ocr", "json")}
    )
    hillclimb: HillclimbConfig = field(default_factory=HillclimbConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    seed: int = 7

    def config_hash(self) -> str:
        """Short, stable hash of the gate thresholds for the ledger."""
        import hashlib

        payload = "|".join(
            f"{m}:{self.gates[m].threshold:.4f}" for m in sorted(self.gates)
        )
        payload += f"|cas:{self.cascade.retain_above:.3f},{self.cascade.filter_below:.3f}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
