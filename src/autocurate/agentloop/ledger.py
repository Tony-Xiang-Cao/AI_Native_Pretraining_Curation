"""Append-only provenance ledger (JSONL).

One never-mutated record per document per stage, plus a daily ``summary``
record holding the SPC state so the loop's memory survives across ticks without
a separate database. This is the concrete artifact behind two governance
directions: *traceable sources* (every kept token traces back to source +
fetch time + extractor version) and *divisible responsibility* (every decision
names the single stage answerable for it). Corrections are appended as new
records that ``supersede`` an earlier one — never edits — so the trail is
auditable.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..schema import ProvenanceRecord


class ProvenanceLedger:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._buffer: List[dict] = []      # in-memory mirror (and store when path is None)

    def _write(self, obj: dict) -> None:
        self._buffer.append(obj)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def append_record(self, rec: ProvenanceRecord) -> None:
        obj = rec.to_json()
        obj["kind"] = "decision"
        self._write(obj)

    def append_summary(self, summary: Dict[str, object]) -> None:
        obj = dict(summary)
        obj["kind"] = "summary"
        self._write(obj)

    def supersede(self, doc_id: str, rec: ProvenanceRecord) -> None:
        obj = rec.to_json()
        obj["kind"] = "decision"
        obj["supersedes"] = doc_id
        self._write(obj)

    def records(self, kind: Optional[str] = None) -> List[dict]:
        if kind is None:
            return list(self._buffer)
        return [r for r in self._buffer if r.get("kind") == kind]

    def trace(self, doc_id: str) -> List[dict]:
        """Every ledger line touching a document — its full lineage."""
        return [r for r in self._buffer
                if r.get("doc_id") == doc_id or r.get("supersedes") == doc_id]
