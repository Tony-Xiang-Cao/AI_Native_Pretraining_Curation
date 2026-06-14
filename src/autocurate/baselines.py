"""External reference-free baselines, scored on the same mutation oracle.

So the extraction gates are not only compared to themselves (floor vs. upper),
we implement transparent stand-ins for the prior-art reference-free families and
score them on the identical corrupt-and-recover oracle:

- ``GzipBaseline`` — a compression-ratio screen (the ZIP-FIT / entropy-law family):
  flag a document whose gzip ratio sits in the robust tail of the clean cohort.
- ``GopherHTMLBaseline`` — the Gopher/C4 line: stop-word deficit + symbol/link
  density, no markup or main-content modelling.
- ``DictGarbageOCRBaseline`` — the Alex & Burns dictionary/garbage ratio: fraction
  of tokens absent from a lexicon fitted on clean reference text.
- ``ParseOnlyJSONBaseline`` — the trivial structural check: ``json.loads`` success
  only (no schema), so it misses schema-valid-but-wrong records.

Each is a ``Gate`` subclass, so it plugs straight into ``verify.evaluate_gate``.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Sequence

from .extract.base import Gate
from .profile.heuristics import text_features
from .schema import Document, clamp
from .utils import gzip_ratio, mad, median, robust_z, safe_div

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


class GzipBaseline(Gate):
    name = "gzip"
    SIGNALS = ["gzip_dev"]
    DEFAULTS = {"gzip_dev": 1.0}

    def fit(self, clean_docs: Sequence[Document]) -> "GzipBaseline":
        rs = [gzip_ratio(d.text) for d in clean_docs]
        self.med = median(rs)
        self.mad_ = mad(rs, self.med)
        return self

    def _signals(self, doc: Document) -> Dict[str, float]:
        z = abs(robust_z(gzip_ratio(doc.text), getattr(self, "med", 0.4),
                         getattr(self, "mad_", 0.05)))
        return {"gzip_dev": clamp(z / 4.0)}        # |robust-z| >= 4 == full defect


class GopherHTMLBaseline(Gate):
    name = "gopher"
    SIGNALS = ["stopword_deficit", "symbol_link"]
    DEFAULTS = {"stopword_deficit": 1.0, "symbol_link": 1.0}

    def _signals(self, doc: Document) -> Dict[str, float]:
        f = text_features(doc.text)
        sw = clamp(1.0 - min(1.0, f["stopword_fraction"] / 0.15))
        symlink = clamp(f["symbol_word_ratio"] * 1.5
                        + doc.text.lower().count("http") / max(1.0, f["token_length"]) * 4.0)
        return {"stopword_deficit": sw, "symbol_link": symlink}


class DictGarbageOCRBaseline(Gate):
    name = "ocr_dict"
    SIGNALS = ["oov_rate"]
    DEFAULTS = {"oov_rate": 2.0}

    def fit(self, clean_docs: Sequence[Document]) -> "DictGarbageOCRBaseline":
        self.vocab = set()
        for d in clean_docs:
            self.vocab.update(w.lower() for w in _WORD_RE.findall(d.text))
        return self

    def _signals(self, doc: Document) -> Dict[str, float]:
        vocab = getattr(self, "vocab", set())
        toks = _WORD_RE.findall(doc.text)
        if not toks:
            return {"oov_rate": 0.0}
        oov = sum(1 for t in toks if t.lower() not in vocab)
        return {"oov_rate": clamp(safe_div(oov, len(toks)) * 2.0)}


class ParseOnlyJSONBaseline(Gate):
    name = "parse_only"
    SIGNALS = ["parse_fail"]
    DEFAULTS = {"parse_fail": 2.0}

    def _signals(self, doc: Document) -> Dict[str, float]:
        raw = doc.raw if doc.raw is not None else doc.text or ""
        try:
            json.loads(raw)
            return {"parse_fail": 0.0}
        except (json.JSONDecodeError, ValueError, TypeError):
            return {"parse_fail": 1.0}


def fitted_baselines(modality: str, clean_docs: Sequence[Document]) -> Dict[str, Gate]:
    """The baselines applicable to a modality, fitted on clean reference docs."""
    gz = GzipBaseline().fit(clean_docs)
    if modality == "html":
        return {"gzip": gz, "gopher": GopherHTMLBaseline()}
    if modality == "ocr":
        return {"gzip": gz, "ocr_dict": DictGarbageOCRBaseline().fit(clean_docs)}
    if modality == "json":
        return {"gzip": gz, "parse_only": ParseOnlyJSONBaseline()}
    return {"gzip": gz}
