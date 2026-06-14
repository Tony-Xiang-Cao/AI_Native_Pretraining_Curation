"""The end-to-end curation cascade.

A document flows: extraction gate (is the *text* faithfully extracted?) →
cheap surface governance score (is the *content* worth keeping?) → and only the
ambiguous middle band is escalated to the LLM-as-Judge. Each document leaves
with a ``Decision`` and an append-only ``ProvenanceRecord`` naming the stage
accountable for it — wiring the four governance directions into one pass:
traceable sources + divisible responsibility (provenance), assessable quality
(gate + score + judge), graded risk (the judge's risk penalty, surfaced).

``mode`` selects the routing policy, which the experiments compare directly:
  "cascade"  — gates + score route; judge only the middle  (the proposed system)
  "heuristic"— gates + score only; middle → REVIEW, no judge call
  "judge"    — escalate every (non-gate-filtered) document to the judge
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .extract import GATES
from .judge import JudgeAdapter, heuristic_governance_score, stub_review
from .schema import (
    AutoCurateConfig, Decision, Document, FILTER, ProvenanceRecord, RETAIN, REVIEW,
)


class CurationLoop:
    def __init__(self, config: Optional[AutoCurateConfig] = None,
                 gates: Optional[Dict[str, object]] = None,
                 judge: Optional[JudgeAdapter] = None, mode: str = "cascade"):
        self.config = config or AutoCurateConfig()
        # one gate instance per modality, threshold taken from config
        self.gates: Dict[str, object] = gates or {
            m: GATES[m](threshold=self.config.gates[m].threshold) for m in GATES
        }
        self.judge = judge
        self.mode = mode
        self.judge_calls = 0

    # -- single document --------------------------------------------------- #
    def curate(self, doc: Document, fetch_time: str = "1970-01-01T00:00:00Z"
               ) -> Tuple[Decision, ProvenanceRecord]:
        gate = self.gates.get(doc.modality)
        gate_quality = 1.0
        extractor = "surface@v1"

        if gate is not None:
            gr = gate.evaluate(doc)
            gate_quality = gr.quality
            extractor = f"{doc.modality}-gate@v1"
            if not gr.passed:
                dec = Decision(doc.id, FILTER, score=gate_quality, stage="gate",
                               reasons=["extraction_defect:" + ",".join(gr.flags)],
                               gate_quality=gate_quality)
                return dec, self._record(doc, fetch_time, extractor, dec)

        if self.mode == "judge":
            dec = self._escalate(doc, gate_quality)
            return dec, self._record(doc, fetch_time, extractor, dec)

        score = heuristic_governance_score(doc.text)
        cfg = self.config.cascade

        if score >= cfg.retain_above:
            dec = Decision(doc.id, RETAIN, score=score, stage="heuristic",
                           reasons=["surface_governance>=retain"], gate_quality=gate_quality)
        elif score <= cfg.filter_below:
            dec = Decision(doc.id, FILTER, score=score, stage="heuristic",
                           reasons=["surface_governance<=filter"], gate_quality=gate_quality)
        elif self.mode == "heuristic":
            dec = Decision(doc.id, REVIEW, score=score, stage="heuristic",
                           reasons=["ambiguous; no judge in heuristic mode"],
                           gate_quality=gate_quality)
        else:  # escalate to the judge (cascade middle band, or judge mode)
            dec = self._escalate(doc, gate_quality)

        return dec, self._record(doc, fetch_time, extractor, dec)

    def _escalate(self, doc: Document, gate_quality: float) -> Decision:
        self.judge_calls += 1
        if self.judge is not None:
            return self.judge.adjudicate(doc, gate_quality)
        return stub_review(doc, gate_quality)

    def _record(self, doc: Document, fetch_time: str, extractor: str,
                dec: Decision) -> ProvenanceRecord:
        return ProvenanceRecord(
            doc_id=doc.id, source=doc.source, fetch_time=fetch_time,
            modality=doc.modality, extractor=extractor, gate_quality=dec.gate_quality,
            decision=dec.label_name, responsible_stage=dec.stage,
            config_hash=self.config.config_hash(),
        )

    # -- batch ------------------------------------------------------------- #
    def curate_batch(self, docs: Sequence[Document], fetch_time: str = "1970-01-01T00:00:00Z"
                     ) -> Tuple[List[Decision], List[ProvenanceRecord]]:
        decisions: List[Decision] = []
        records: List[ProvenanceRecord] = []
        for d in docs:
            dec, rec = self.curate(d, fetch_time)
            decisions.append(dec)
            records.append(rec)
        return decisions, records
