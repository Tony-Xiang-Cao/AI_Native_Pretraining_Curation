"""Reference-free HTML→text extraction-quality gate.

A good HTML extractor (trafilatura, Resiliparse, Readability) returns the main
article text and drops nav bars, cookie banners, link farms, and inline
JS/CSS. We estimate, *without a gold main-text reference*, whether extraction
succeeded, from five structural defect signals. Crucially the boilerplate
detector only knows the **train** phrase slice (`BOILERPLATE_TRAIN`); the
mutation oracle injects **held-out** phrases, so a gate that merely memorizes
the phrase list scores at its conservative floor and must rely on the
structural signals (markup leak, link density, short-line bursts, stop-word
deficit) to generalize.

Signals (1 == strong evidence of a bad extraction):
  markup_leak     residual tags / HTML entities / JS-CSS tokens in the text
  boilerplate_leak density of known nav/legal boilerplate phrases (train slice)
  link_density    URL / markdown-link tokens per line
  nonprose_lines  fraction of short, punctuation-less, stop-word-poor lines
                  (nav chrome / button labels — generalizes past the phrase list)
  stopword_deficit shortfall of function words vs. natural prose
"""

from __future__ import annotations

import re
from typing import Dict

from ..lexicons import BOILERPLATE_TRAIN, CODE_TOKENS
from ..schema import Document, clamp
from ..utils import safe_div
from .base import Gate

_TAG_RE = re.compile(r"</?[a-zA-Z][^>]{0,80}>")
_ENTITY_RE = re.compile(r"&[a-zA-Z]{2,10};|&#\d{1,5};")
_LINK_RE = re.compile(r"https?://|www\.|\]\(http|\[[^\]]{1,40}\]\([^)]+\)")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_STOP = {"the", "be", "to", "of", "and", "that", "have", "with", "a", "in",
         "is", "it", "for", "on", "as", "are", "was", "this", "by", "from"}


class HTMLGate(Gate):
    name = "html"
    SIGNALS = ["markup_leak", "boilerplate_leak", "link_density",
               "nonprose_lines", "stopword_deficit"]
    DEFAULTS = {"markup_leak": 2.0, "boilerplate_leak": 1.6, "link_density": 1.6,
                "nonprose_lines": 1.4, "stopword_deficit": 0.8}

    def __init__(self, weights=None, threshold: float = 0.5):
        super().__init__(weights, threshold)
        self._bp_lower = [p.lower() for p in BOILERPLATE_TRAIN]

    def _signals(self, doc: Document) -> Dict[str, float]:
        text = doc.text or ""
        low = text.lower()
        n_chars = max(1, len(text))
        lines = text.split("\n")
        n_lines = max(1, len(lines))

        tag_chars = sum(len(m.group(0)) for m in _TAG_RE.finditer(text))
        entity_hits = len(_ENTITY_RE.findall(text))
        code_hits = sum(low.count(tok.lower()) for tok in CODE_TOKENS)
        markup_leak = clamp((tag_chars / n_chars) * 6 + (entity_hits + code_hits) / n_lines)

        bp_hits = sum(low.count(p) for p in self._bp_lower)
        boilerplate_leak = clamp(bp_hits / n_lines * 2.0)

        link_hits = len(_LINK_RE.findall(text))
        link_density = clamp(link_hits / n_lines)

        nonprose_lines = clamp(_nonprose_fraction(lines) * 1.2)

        words = _WORD_RE.findall(low)
        sw_frac = safe_div(sum(1 for w in words if w in _STOP), len(words)) if words else 0.0
        stopword_deficit = clamp(1.0 - min(1.0, sw_frac / 0.12))

        return {
            "markup_leak": markup_leak,
            "boilerplate_leak": boilerplate_leak,
            "link_density": link_density,
            "nonprose_lines": nonprose_lines,
            "stopword_deficit": stopword_deficit,
        }


def _nonprose_fraction(lines) -> float:
    """Fraction of non-empty lines that look like nav chrome rather than prose.

    A line is "non-prose" if it is short (<= 8 words), does not end in sentence
    punctuation, and is stop-word-poor — the structural fingerprint of menus,
    button labels, and legal one-liners, independent of the exact phrase, so it
    catches *held-out* boilerplate the phrase list never saw.
    """
    nonempty = [ln.strip() for ln in lines if ln.strip()]
    if not nonempty:
        return 0.0
    bad = 0
    for s in nonempty:
        ws = s.split()
        if len(ws) <= 8 and not s.endswith((".", "?", "!", ":", ";")):
            sw = sum(1 for w in ws if w.lower() in _STOP)
            if sw / max(1, len(ws)) < 0.2:
                bad += 1
    # Tolerate one stray short line (a heading is normal prose structure); only a
    # *burst* of punctuation-less, stop-word-poor lines is boilerplate.
    return max(0, bad - 1) / len(nonempty)
