"""Quality-report generation (structured JSON + human-readable markdown).

The report is organized by the four governance directions so it doubles as an
audit artifact: *traceable sources* (per-source provenance), *assessable
quality* (decision + gate-quality + outlier counts), *graded risk* (risk
buckets), *divisible responsibility* (which stage decided what).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

from .schema import Decision, LABEL_NAMES, ProvenanceRecord, QualityProfile
from .utils import mean


def build_report(decisions: Sequence[Decision], records: Sequence[ProvenanceRecord],
                 profiles: Optional[Sequence[QualityProfile]] = None,
                 judge_calls: int = 0) -> Dict[str, object]:
    n = max(1, len(decisions))
    labels = Counter(d.label_name for d in decisions)
    by_stage = Counter(d.stage for d in decisions)
    by_source = Counter(r.source for r in records)
    by_modality = Counter(r.modality for r in records)

    risks = [d.risk for d in decisions]
    risk_buckets = Counter(_risk_level(r) for r in risks)

    outliers = 0
    if profiles is not None:
        outliers = sum(1 for p in profiles if p.is_outlier)

    return {
        "n_documents": len(decisions),
        "decisions": {k: labels.get(k, 0) for k in LABEL_NAMES.values()},
        "retain_rate": round(labels.get("retain", 0) / n, 4),
        "filter_rate": round(labels.get("filter", 0) / n, 4),
        "review_rate": round(labels.get("review", 0) / n, 4),
        "mean_gate_quality": round(mean([r.gate_quality for r in records]) if records else 1.0, 4),
        "responsibility": dict(by_stage),         # divisible responsibility
        "by_source": dict(by_source),             # traceable sources
        "by_modality": dict(by_modality),
        "risk_grades": dict(risk_buckets),        # graded risk
        "outliers_flagged": outliers,
        "judge_calls": judge_calls,
        "judge_call_rate": round(judge_calls / n, 4),
    }


def _risk_level(r: float) -> str:
    for cut, name in ((0.15, "none"), (0.40, "low"), (0.65, "medium"), (0.85, "high")):
        if r < cut:
            return name
    return "critical"


def render_markdown(report: Dict[str, object]) -> str:
    d = report
    lines: List[str] = []
    lines.append("# AutoCurate quality report\n")
    lines.append(f"Documents processed: **{d['n_documents']}**  ")
    lines.append(f"Judge calls: **{d['judge_calls']}** "
                 f"({d['judge_call_rate']:.0%} of documents)\n")

    lines.append("## 2. Assessable quality")
    dec = d["decisions"]
    lines.append(f"- retain {dec.get('retain', 0)} "
                 f"({d['retain_rate']:.1%}) · review {dec.get('review', 0)} "
                 f"({d['review_rate']:.1%}) · filter {dec.get('filter', 0)} "
                 f"({d['filter_rate']:.1%})")
    lines.append(f"- mean extraction-gate quality: {d['mean_gate_quality']:.3f}")
    lines.append(f"- distributional outliers flagged: {d['outliers_flagged']}\n")

    lines.append("## 3. Graded risk")
    for lvl in ("none", "low", "medium", "high", "critical"):
        if d["risk_grades"].get(lvl):
            lines.append(f"- {lvl}: {d['risk_grades'][lvl]}")
    lines.append("")

    lines.append("## 1. Traceable sources")
    for src, c in sorted(d["by_source"].items(), key=lambda t: -t[1]):
        lines.append(f"- {src}: {c}")
    lines.append("")

    lines.append("## 4. Divisible responsibility (deciding stage)")
    for stage, c in sorted(d["responsibility"].items(), key=lambda t: -t[1]):
        lines.append(f"- {stage}: {c}")
    lines.append("")
    return "\n".join(lines)
