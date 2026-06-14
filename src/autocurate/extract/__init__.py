"""Reference-free extraction-quality gates for HTML, OCR, and JSON/JSONL.

Extraction quality is *upstream* of content quality: a perfectly written page
is worthless training data if the HTML→text step leaked the nav bar, the OCR
mangled every third character, or the JSON record was truncated mid-field. Each
gate estimates extraction quality **without a gold reference**, from cheap
structural signals, and exposes a tunable parameter vector so the
self-improvement loop (``hillclimb``) can climb it against the mutation oracle
(``verify``).
"""

from __future__ import annotations

from .base import Gate, GateResult, combine_gate
from .html_gate import HTMLGate
from .json_gate import JSONGate, infer_schema
from .ocr_gate import OCRGate

GATES = {"html": HTMLGate, "ocr": OCRGate, "json": JSONGate}

__all__ = [
    "Gate", "GateResult", "combine_gate",
    "HTMLGate", "OCRGate", "JSONGate", "infer_schema", "GATES",
]
