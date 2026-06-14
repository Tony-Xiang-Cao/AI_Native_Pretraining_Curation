"""Shared vocabularies for gates and the mutation oracle, with a held-out split.

This module is the single point that enforces **de-circularization** (the
lesson inherited from judgecurate): a gate that "knows" the exact strings the
oracle injects would be scored on read-back, not detection. So every finite
vocabulary is partitioned into a ``train`` slice — the only strings a gate is
allowed to pattern-match — and a ``heldout`` slice the oracle uses to measure
*generalization*. Reporting the (held-out floor, train upper-bound) pair makes
the memorization gap explicit, exactly as judgecurate brackets its judge.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

HELD_OUT_RATE = 0.25
_SPLIT_SEED = 0


def split_vocab(vocab: Sequence, held_out_rate: float = HELD_OUT_RATE,
                seed: int = _SPLIT_SEED) -> Dict[str, list]:
    """Deterministically split a vocabulary into train / held-out slices."""
    rng = random.Random(seed)
    v = sorted(vocab, key=lambda x: str(x))
    rng.shuffle(v)
    cut = max(1, round(len(v) * held_out_rate))
    return {"heldout": v[:cut], "train": v[cut:]}


# --------------------------------------------------------------------------- #
# HTML boilerplate / navigation chrome
# --------------------------------------------------------------------------- #

BOILERPLATE_PHRASES: List[str] = [
    "skip to content", "cookie preferences", "accept all cookies",
    "subscribe to our newsletter", "all rights reserved", "privacy policy",
    "terms of service", "sign in", "log in", "create an account",
    "share on twitter", "share on facebook", "follow us", "read more",
    "advertisement", "sponsored content", "back to top", "main navigation",
    "you may also like", "related articles", "leave a comment",
    "click here to subscribe", "menu", "search this site", "add to cart",
]
_BP = split_vocab(BOILERPLATE_PHRASES)
BOILERPLATE_TRAIN: List[str] = _BP["train"]
BOILERPLATE_HELDOUT: List[str] = _BP["heldout"]

# CSS/JS tokens that signal <script>/<style> leakage into extracted text.
CODE_TOKENS: List[str] = [
    "function(", "var ", "const ", "let ", "{", "}", "px;", "@media",
    "addEventListener", "document.", "window.", "</", "/>", "&nbsp;", "&amp;",
]


# --------------------------------------------------------------------------- #
# OCR character confusions (a -> b applied at a rate)
# --------------------------------------------------------------------------- #

OCR_CONFUSIONS: List[Tuple[str, str]] = [
    ("rn", "m"), ("m", "rn"), ("l", "1"), ("I", "l"), ("O", "0"), ("0", "O"),
    ("S", "5"), ("B", "8"), ("cl", "d"), ("vv", "w"), ("ii", "ü"), ("fi", "ti"),
    ("e", "c"), ("a", "o"), ("g", "q"), ("h", "b"),
]
_OCR = split_vocab(OCR_CONFUSIONS)
OCR_CONFUSIONS_TRAIN: List[Tuple[str, str]] = _OCR["train"]
OCR_CONFUSIONS_HELDOUT: List[Tuple[str, str]] = _OCR["heldout"]

# Mojibake fragments produced by UTF-8 misdecoded as Latin-1.
MOJIBAKE_FRAGMENTS: List[str] = [
    "Ã©", "Ã¨", "Ã¼", "Ã±", "â€™", "â€œ", "â€\x9d", "â€“", "Â ", "Â·", "ï¿½", "�",
]


# --------------------------------------------------------------------------- #
# Content-level risk / contradiction markers (semantic defects)
# --------------------------------------------------------------------------- #

RISK_MARKERS: List[str] = [
    "buy now and click here for free money", "act now limited time offer",
    "guaranteed to cure all diseases", "wire the payment immediately",
    "send your password and social security number", "this is not financial advice but invest all",
    "miracle weight loss no diet", "you have won the lottery claim your prize",
    "hot singles in your area", "download this crack for free",
]
_RISK = split_vocab(RISK_MARKERS)
RISK_TRAIN: List[str] = _RISK["train"]
RISK_HELDOUT: List[str] = _RISK["heldout"]

CONTRADICTION_MARKERS: List[str] = [
    "however the opposite is also definitely true",
    "this never happened although it clearly did",
    "the figure is exactly 10 and also exactly 1000",
    "it is both completely safe and extremely dangerous",
    "the study found no effect and a huge effect simultaneously",
    "always avoid this and always do this",
    "the temperature was below zero and boiling at once",
    "nobody agrees and everybody agrees on this point",
]
_CONTRA = split_vocab(CONTRADICTION_MARKERS)
CONTRA_TRAIN: List[str] = _CONTRA["train"]
CONTRA_HELDOUT: List[str] = _CONTRA["heldout"]
