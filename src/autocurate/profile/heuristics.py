"""Reference-free text-quality heuristics (pure stdlib).

These are the cheap, model-free signals that real pretraining pipelines use as
a first screen (Gopher's repetition/symbol filters, C4's heuristics, RefinedWeb
/ FineWeb cleaning) plus a compression-ratio redundancy signal. Every feature
is a transparent function of the text, so the profile is fully auditable and
deterministic.

``text_features(text)`` returns a flat ``{name: value}`` dict; ``FEATURE_NAMES``
is the canonical ordered list. Values are kept on comparable scales (fractions
in ``[0, 1]`` where possible) so robust z-scores across features are meaningful.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

from ..utils import gzip_ratio, mean, safe_div

# A small, high-frequency English stop-word set. Gopher requires a document to
# contain at least two of {the, be, to, of, and, that, have, with}; the *frac*
# of tokens that are stop-words is a cheap "is this natural language?" signal.
_STOPWORDS = {
    "the", "be", "to", "of", "and", "that", "have", "with", "a", "in", "is",
    "it", "for", "on", "as", "are", "was", "this", "by", "an", "or", "from",
}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)         # alphabetic word runs
_TOKEN_RE = re.compile(r"\S+")
# Common mojibake / encoding-damage signatures (UTF-8 misdecoded as Latin-1).
_MOJIBAKE_RE = re.compile(r"Ã|Â|â€|ï¿½|�|Ð|Ñ")

#: Canonical ordered feature list (the profile vector).
FEATURE_NAMES: List[str] = [
    "char_length",
    "token_length",
    "mean_word_length",
    "gzip_ratio",
    "type_token_ratio",
    "bigram_repetition",
    "top_ngram_fraction",
    "dup_line_fraction",
    "stopword_fraction",
    "alpha_fraction",
    "digit_fraction",
    "punct_fraction",
    "uppercase_fraction",
    "non_ascii_fraction",
    "symbol_word_ratio",
    "bullet_line_fraction",
    "ellipsis_line_fraction",
    "short_line_fraction",
    "mean_line_length",
    "mojibake_fraction",
]


def text_features(text: str) -> Dict[str, float]:
    """Compute the full reference-free feature vector for one document."""
    text = text or ""
    n_chars = len(text)
    tokens = _TOKEN_RE.findall(text)
    words = _WORD_RE.findall(text.lower())
    n_tokens = len(tokens)
    n_words = len(words)
    lines = text.split("\n")
    n_lines = max(1, len(lines))

    # --- length / compression ------------------------------------------- #
    feats: Dict[str, float] = {}
    feats["char_length"] = float(n_chars)
    feats["token_length"] = float(n_tokens)
    feats["mean_word_length"] = mean([len(w) for w in words]) if words else 0.0
    feats["gzip_ratio"] = gzip_ratio(text)

    # --- lexical diversity / repetition --------------------------------- #
    feats["type_token_ratio"] = safe_div(len(set(tokens)), n_tokens)
    feats["bigram_repetition"] = _bigram_repetition(tokens)
    feats["top_ngram_fraction"] = _top_ngram_fraction(tokens, n=2)
    feats["dup_line_fraction"] = _dup_line_fraction(lines)

    # --- natural-language signal ---------------------------------------- #
    sw = sum(1 for w in words if w in _STOPWORDS)
    feats["stopword_fraction"] = safe_div(sw, n_words)

    # --- character-class fractions -------------------------------------- #
    alpha = sum(1 for c in text if c.isalpha())
    digit = sum(1 for c in text if c.isdigit())
    upper = sum(1 for c in text if c.isupper())
    punct = sum(1 for c in text if not c.isalnum() and not c.isspace())
    non_ascii = sum(1 for c in text if ord(c) > 127)
    feats["alpha_fraction"] = safe_div(alpha, n_chars)
    feats["digit_fraction"] = safe_div(digit, n_chars)
    feats["punct_fraction"] = safe_div(punct, n_chars)
    feats["uppercase_fraction"] = safe_div(upper, alpha) if alpha else 0.0
    feats["non_ascii_fraction"] = safe_div(non_ascii, n_chars)

    # Gopher symbol-to-word ratio (hashes + ellipses to words).
    n_hash = text.count("#")
    n_ellipsis = text.count("...") + text.count("…")
    feats["symbol_word_ratio"] = safe_div(n_hash + n_ellipsis, n_words)

    # --- line-structure signals (boilerplate / nav) --------------------- #
    feats["bullet_line_fraction"] = safe_div(
        sum(1 for ln in lines if ln.lstrip()[:1] in {"-", "*", "•", "‣"}), n_lines
    )
    feats["ellipsis_line_fraction"] = safe_div(
        sum(1 for ln in lines if ln.rstrip().endswith(("...", "…"))), n_lines
    )
    feats["short_line_fraction"] = safe_div(
        sum(1 for ln in lines if 0 < len(ln.strip()) < 12), n_lines
    )
    feats["mean_line_length"] = mean([len(ln) for ln in lines]) if lines else 0.0

    # --- encoding damage ------------------------------------------------ #
    feats["mojibake_fraction"] = safe_div(len(_MOJIBAKE_RE.findall(text)), max(1, n_words))

    return feats


def _bigram_repetition(tokens: List[str]) -> float:
    """Fraction of adjacent token bigrams that are immediate repeats."""
    if len(tokens) < 2:
        return 0.0
    repeats = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1])
    return safe_div(repeats, len(tokens) - 1)


def _top_ngram_fraction(tokens: List[str], n: int = 2) -> float:
    """Fraction of tokens covered by the single most frequent n-gram.

    A Gopher-style repetition signal: spammy / templated text is dominated by
    one repeated phrase.
    """
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    top = Counter(grams).most_common(1)[0][1]
    return safe_div(top * n, len(tokens))


def _dup_line_fraction(lines: List[str]) -> float:
    """Fraction of non-empty lines that are exact duplicates of an earlier line."""
    seen = set()
    dup = 0
    total = 0
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        total += 1
        if s in seen:
            dup += 1
        else:
            seen.add(s)
    return safe_div(dup, total)
