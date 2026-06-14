"""autocurate — an AI-native, self-verifying, self-improving curation loop.

A small, dependency-free reference implementation that turns a four-direction
data-governance theory (traceable sources, assessable quality, graded risk,
divisible responsibility) into a runnable operating loop:

  * **profile**   — cheap reference-free text heuristics + robust-z outliers
  * **extract**   — reference-free HTML / OCR / JSON extraction-quality gates
  * **judge**     — a thin adapter reusing ``judgecurate`` for the ambiguous middle
  * **verify**    — a mutation-oracle harness that makes every claim falsifiable
  * **hillclimb** — verifier-gated self-improvement of the gates
  * **agentloop** — a harness-agnostic daily routine + provenance ledger + SPC drift

Pure standard library; ``judgecurate`` and real LLM backends are optional extras.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .schema import (
    AutoCurateConfig,
    Decision,
    Document,
    FILTER,
    GateResult,
    ProvenanceRecord,
    QualityProfile,
    RETAIN,
    REVIEW,
)
from .profile import CohortOutlierDetector, profile_corpus, text_features
from .extract import GATES, HTMLGate, JSONGate, OCRGate
from .judge import heuristic_governance_score
from .pipeline import CurationLoop
from .verify import (
    accept_candidate,
    build_mutation_set,
    corrupt,
    evaluate_gate,
    floor_and_upper,
    verified_objective,
)
from .hillclimb import HillclimbResult, OfflineProposer, hillclimb
from .agentloop import (
    CrawlMonitorRoutine,
    LocalCronHarness,
    ProvenanceLedger,
    StreamMonitor,
)
from .report import build_report, render_markdown

__all__ = [
    "__version__",
    # schema
    "Document", "Decision", "GateResult", "QualityProfile", "ProvenanceRecord",
    "AutoCurateConfig", "FILTER", "REVIEW", "RETAIN",
    # profile
    "text_features", "CohortOutlierDetector", "profile_corpus",
    # extract
    "HTMLGate", "OCRGate", "JSONGate", "GATES",
    # judge + pipeline
    "heuristic_governance_score", "CurationLoop",
    # verify
    "corrupt", "build_mutation_set", "evaluate_gate", "floor_and_upper",
    "verified_objective", "accept_candidate",
    # hillclimb
    "hillclimb", "HillclimbResult", "OfflineProposer",
    # agentloop
    "StreamMonitor", "ProvenanceLedger", "CrawlMonitorRoutine", "LocalCronHarness",
    # report
    "build_report", "render_markdown",
]
