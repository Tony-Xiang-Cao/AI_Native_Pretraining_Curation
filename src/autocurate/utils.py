"""Small pure-stdlib numeric and text helpers (no NumPy).

Robust statistics (median / MAD / robust z-score) are used throughout for
outlier detection and drift monitoring because pretraining-corpus feature
distributions are heavy-tailed and a mean/std summary is dominated by the junk
we are trying to find.
"""

from __future__ import annotations

import gzip
import math
from typing import List, Sequence

# A scale factor making MAD a consistent estimator of the std under normality.
_MAD_TO_SIGMA = 1.4826


def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def mad(xs: Sequence[float], med: float = None) -> float:
    """Median absolute deviation."""
    s = list(xs)
    if not s:
        return 0.0
    m = median(s) if med is None else med
    return median([abs(x - m) for x in s])


def robust_z(x: float, med: float, mad_val: float) -> float:
    """Robust z-score using median/MAD. Returns 0 when MAD collapses to 0."""
    if mad_val <= 1e-12:
        return 0.0
    return (x - med) / (_MAD_TO_SIGMA * mad_val)


def std(xs: Sequence[float]) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def quantile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile, q in [0, 1]."""
    s = sorted(xs)
    if not s:
        return 0.0
    if q <= 0:
        return s[0]
    if q >= 1:
        return s[-1]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def gzip_ratio(text: str) -> float:
    """Compression ratio = compressed bytes / raw UTF-8 bytes.

    A classic, model-free redundancy signal. Natural prose lands around
    0.30-0.45; highly repetitive boilerplate/spam compresses *better* (lower
    ratio); near-random garble (mojibake, base64 dumps) compresses *worse*
    (ratio toward 1). Both tails flag non-prose, so the ratio is a cheap
    two-sided quality screen.
    """
    raw = text.encode("utf-8", errors="ignore")
    if not raw:
        return 0.0
    comp = gzip.compress(raw, compresslevel=6)
    # gzip adds a small fixed header/footer; subtract a constant so tiny inputs
    # are not reported as ratio > 1 purely from framing overhead.
    overhead = 18
    return max(0.0, (len(comp) - overhead)) / len(raw)


def sha1_short(text: str, n: int = 12) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def ewma(series: Sequence[float], alpha: float) -> List[float]:
    """Exponentially weighted moving average of a time series."""
    out: List[float] = []
    s = None
    for x in series:
        s = x if s is None else alpha * x + (1 - alpha) * s
        out.append(s)
    return out
