"""Reference-free OCR-quality gate (post-OCR quality estimation).

Following the no-reference OCR-assessment line of work — dictionary / garbage
ratio (Alex & Burns 2014), the "Rerunning OCR" output-only feature set
(arXiv:2110.01661), and classic garbage-string detectors (Taghva & Nartker) —
we estimate OCR fidelity from the text alone, with no ground-truth transcript.
The signals look for the fingerprints of bad recognition: implausible
("garbage") tokens, character-confusion artifacts, spurious word breaks,
mojibake, and scan-speckle line noise. The confusion map the gate "knows" is
the **train** slice only; held-out confusions test that the structural garble
signals generalize.

Signals (1 == strong evidence of bad OCR):
  garbage_rate    fraction of word tokens that look implausible (no vowel, etc.)
  confusion_rate  digit-in-word and isolated 1/0/| artifacts
  broken_word     hyphenation breaks + single/double-char token bursts
  mojibake        encoding-damage fragment density
  line_noise      isolated non-word symbol glyph density
"""

from __future__ import annotations

import re
from typing import Dict

from ..schema import Document, clamp
from ..utils import safe_div
from .base import Gate

_TOKEN_RE = re.compile(r"\S+")
_ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_VOWELS = set("aeiouyAEIOUYàâäéèêëïîôöùûüáíóúñ")
_NOISE_GLYPHS = set("¬|~^`¦§¤°±¶•◊�")
# Legitimate 1-2 char English words, so word-break fragments stand out.
_COMMON_SHORT = {
    "a", "i", "an", "of", "to", "is", "in", "it", "on", "or", "as", "by",
    "we", "he", "be", "do", "no", "so", "up", "at", "my", "me", "us", "am",
    "if", "go", "ok", "us", "the", "and",
}


def _is_garbage(tok: str) -> bool:
    """Heuristic garbage-token test (no dictionary needed)."""
    letters = _ALPHA_RE.findall(tok)
    if len(letters) >= 3:
        if not any(c in _VOWELS for c in tok):
            return True                                   # consonant soup
        # implausible vowel ratio (char substitutions skew it either way)
        if len(letters) >= 4:
            vr = sum(1 for c in letters if c in _VOWELS) / len(letters)
            if vr < 0.18 or vr > 0.85:
                return True
        # >=4 identical consecutive letters: "aaaa", "lll"
        run = 1
        for i in range(1, len(tok)):
            run = run + 1 if tok[i] == tok[i - 1] and tok[i].isalpha() else 1
            if run >= 4:
                return True
        # interior case flip in an otherwise-lower word: "wOrd", "tHe"
        core = "".join(letters)
        if len(core) >= 3 and core[1:-1] != core[1:-1].lower() and core != core.upper():
            return True
    return False


class OCRGate(Gate):
    name = "ocr"
    SIGNALS = ["garbage_rate", "confusion_rate", "broken_word", "mojibake", "line_noise"]
    DEFAULTS = {"garbage_rate": 1.5, "confusion_rate": 1.1, "broken_word": 1.0,
                "mojibake": 1.4, "line_noise": 0.8}

    def _signals(self, doc: Document) -> Dict[str, float]:
        text = doc.text or ""
        n_chars = max(1, len(text))
        tokens = _TOKEN_RE.findall(text)
        n_tok = max(1, len(tokens))
        lines = text.split("\n")
        n_lines = max(1, len(lines))

        garbage = sum(1 for t in tokens if _is_garbage(t))
        garbage_rate = clamp(safe_div(garbage, n_tok) * 2.5)

        digit_in_word = sum(
            1 for t in tokens if _ALPHA_RE.search(t) and any(c.isdigit() for c in t)
        )
        isolated = sum(1 for t in tokens if t in {"1", "0", "l", "|", "I"})
        confusion_rate = clamp(safe_div(digit_in_word + isolated, n_tok) * 3.0)

        hyphen_breaks = sum(1 for ln in lines if ln.rstrip().endswith("-"))
        tiny_tokens = sum(
            1 for t in tokens
            if 1 <= len(t) <= 2 and _ALPHA_RE.search(t) and t.lower() not in _COMMON_SHORT
        )
        broken_word = clamp(
            safe_div(hyphen_breaks, n_lines) * 2.0 + safe_div(tiny_tokens, n_tok) * 3.0
        )

        # Structural mojibake detection (no fixed fragment list to read back):
        # generic non-ASCII token density + the Unicode replacement char.
        nonascii_tok = sum(1 for t in tokens if any(ord(c) > 127 for c in t))
        repl = sum(1 for c in text if c == "�")
        mojibake = clamp(safe_div(nonascii_tok, n_tok) * 5.0 + safe_div(repl, n_tok) * 8.0)

        glyph_tok = sum(1 for t in tokens if any(c in _NOISE_GLYPHS for c in t))
        noise = sum(1 for c in text if c in _NOISE_GLYPHS)
        line_noise = clamp(safe_div(glyph_tok, n_tok) * 5.0 + safe_div(noise, n_chars) * 30.0)

        return {
            "garbage_rate": garbage_rate,
            "confusion_rate": confusion_rate,
            "broken_word": broken_word,
            "mojibake": mojibake,
            "line_noise": line_noise,
        }
