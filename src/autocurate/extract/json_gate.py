"""Reference-free JSON / JSONL structural-quality gate.

"Bespoke" JSON/JSONL pretraining shards fail in structural ways a content
judge never sees: a record truncated mid-write, a renamed/typed-wrong field, a
control character that breaks the parser, two records mashed onto one line. In
the spirit of production data validation (TFDV, Great Expectations), this gate
infers a schema from a clean batch and then scores each record on structural
conformance — with ``json.loads`` success itself acting as a near-perfect
ground-truth signal that validates the softer schema checks.

Signals (1 == strong evidence of a broken record):
  parse_fail     json.loads raised (truncation / delimiter break)
  truncated      unbalanced braces / brackets / quotes
  schema_missing fraction of expected keys absent
  type_mismatch  fraction of present keys with the wrong value type
  empty_value    fraction of expected keys that are null / empty
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List, Optional, Sequence

from ..schema import Document, clamp
from .base import Gate

_TYPE_NAMES = {
    str: "string", bool: "bool", int: "int", float: "float",
    list: "list", dict: "dict", type(None): "null",
}


def _type_name(v) -> str:
    return _TYPE_NAMES.get(type(v), "other")


def infer_schema(records: Sequence[dict], required_frac: float = 0.8) -> Dict[str, str]:
    """Infer ``{key: expected_type}`` from a batch of clean records.

    A key is *required* if it appears in at least ``required_frac`` of records;
    its expected type is the modal non-null type seen for that key.
    """
    n = max(1, len(records))
    key_counts: Counter = Counter()
    key_types: Dict[str, Counter] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        for k, v in rec.items():
            key_counts[k] += 1
            key_types.setdefault(k, Counter())
            if v is not None:
                key_types[k][_type_name(v)] += 1
    schema: Dict[str, str] = {}
    for k, c in key_counts.items():
        if c / n >= required_frac:
            types = key_types.get(k)
            schema[k] = types.most_common(1)[0][0] if types else "string"
    return schema


def _balanced(s: str) -> bool:
    """Brace/bracket/quote balance check (catches truncation)."""
    depth = 0
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not in_str


class JSONGate(Gate):
    name = "json"
    SIGNALS = ["parse_fail", "truncated", "schema_missing", "type_mismatch", "empty_value"]
    DEFAULTS = {"parse_fail": 2.0, "truncated": 1.5, "schema_missing": 1.4,
                "type_mismatch": 1.2, "empty_value": 1.6}

    def __init__(self, weights=None, threshold: float = 0.5,
                 schema: Optional[Dict[str, str]] = None):
        super().__init__(weights, threshold)
        self.schema: Dict[str, str] = dict(schema or {})

    def fit_schema(self, raws: Sequence[str]) -> "JSONGate":
        recs: List[dict] = []
        for r in raws:
            try:
                obj = json.loads(r)
                if isinstance(obj, dict):
                    recs.append(obj)
            except (json.JSONDecodeError, TypeError):
                continue
        self.schema = infer_schema(recs)
        return self

    def _signals(self, doc: Document) -> Dict[str, float]:
        raw = doc.raw if doc.raw is not None else doc.text or ""
        sig = {k: 0.0 for k in self.SIGNALS}

        try:
            obj = json.loads(raw)
            parsed = True
        except (json.JSONDecodeError, TypeError, ValueError):
            obj = None
            parsed = False

        if not parsed:
            sig["parse_fail"] = 1.0
            sig["truncated"] = 0.0 if _balanced(raw) else 1.0
            return sig

        if not _balanced(raw):
            sig["truncated"] = 1.0
        if not isinstance(obj, dict) or not self.schema:
            return sig

        expected = self.schema
        n_exp = max(1, len(expected))
        missing = sum(1 for k in expected if k not in obj)
        mism = 0
        empty = 0
        for k, t in expected.items():
            if k not in obj:
                continue
            v = obj[k]
            if v is not None and _type_name(v) != t and not (t == "float" and _type_name(v) == "int"):
                mism += 1
            if v is None or v == "" or v == [] or v == {}:
                empty += 1
        # A single missing/wrong-typed required key is a serious structural break,
        # so a non-zero count floors the signal at 0.6 rather than 1/n_keys.
        sig["schema_missing"] = _present_floor(missing, n_exp)
        sig["type_mismatch"] = _present_floor(mism, n_exp)
        sig["empty_value"] = _present_floor(empty, n_exp, floor=0.5)
        return sig


def _present_floor(count: int, n: int, floor: float = 0.6) -> float:
    return 0.0 if count == 0 else clamp(max(floor, count / max(1, n)))
