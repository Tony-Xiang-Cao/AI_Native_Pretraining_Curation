"""Deterministic synthetic corpora (pure stdlib, seedable, no network).

Everything the experiments consume is generated here so results reproduce
bit-for-bit. We generate *clean* prose that a good extractor would have
produced — natural function-word density, vowel-rich tokens, terminal
punctuation — across a few source cohorts and the three extraction modalities.
The mutation oracle (``verify.corruptions``) then injects known defects into
these clean documents; a correct gate keeps the clean original and flags the
corrupted copy.

This module deliberately generates only the *extraction* corpora (the novel
part). The content/cascade experiment (E2) reuses the sibling ``judgecurate``
``mini_corpus`` and judge directly when available (see ``judge.py``).
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Sequence

from .schema import Document

# --------------------------------------------------------------------------- #
# A small bank of natural-language material. Function words are included at
# realistic density so clean docs have stop-word fractions ~0.3-0.4 and pass
# the gates; content words are vowel-rich so the OCR garbage detector is happy.
# --------------------------------------------------------------------------- #

_SUBJECTS = [
    "the researcher", "a recent study", "the committee", "this method",
    "the enzyme", "the algorithm", "the author", "the river system",
    "the central bank", "the telescope", "the immune response", "the dataset",
    "the medieval city", "the new policy", "the protein", "the population",
]
_VERBS = [
    "demonstrates", "measured", "concluded that", "describes", "increased",
    "analyzed", "reported", "suggests that", "produced", "examined",
    "improved", "observed that", "estimated", "revealed", "supported",
]
_OBJECTS = [
    "a significant effect on the outcome", "the underlying mechanism in detail",
    "several important properties of the material", "the relationship between the variables",
    "a measurable reduction in the error rate", "the historical context of the period",
    "the structure of the observed signal", "consistent results across conditions",
    "the distribution of the sampled values", "a plausible explanation for the trend",
]
_CONNECTORS = [
    "In addition,", "However,", "As a result,", "For example,", "In contrast,",
    "Notably,", "According to the report,", "Over the following years,",
    "Although the evidence was limited,", "Based on these observations,",
]
_SOURCES = ["web", "encyclopedia", "academic", "forum"]


def _sentence(rng: random.Random) -> str:
    parts = []
    if rng.random() < 0.5:
        parts.append(rng.choice(_CONNECTORS))
    parts.append(rng.choice(_SUBJECTS))
    parts.append(rng.choice(_VERBS))
    parts.append(rng.choice(_OBJECTS))
    s = " ".join(parts)
    return s[0].upper() + s[1:] + rng.choice([".", ".", ".", "?"])


_HEADINGS = [
    "Background And Methods", "Overview", "Results", "Historical Context",
    "Materials And Methods", "Discussion", "Key Findings", "Introduction",
]
_LIST_ITEMS = [
    "a higher sampling rate", "the control group", "improved baselines",
    "the revised estimate", "additional measurements", "the second cohort",
]


def make_clean_text(rng: random.Random, n_sentences: Optional[int] = None,
                    realistic: bool = True) -> str:
    n = n_sentences or rng.randint(6, 14)
    sents = [_sentence(rng) for _ in range(n)]
    paras: List[str] = []
    i = 0
    while i < len(sents):
        k = rng.randint(2, 4)
        paras.append(" ".join(sents[i : i + k]))
        i += k
    body = "\n\n".join(paras)
    if not realistic:
        return body

    # Benign imperfections that real *clean* main text legitimately contains —
    # a heading, a short list, an inline figure number. They depress the
    # reference-free quality estimate mildly (never to defect levels), giving a
    # realistic clean-quality spread and a genuine precision/recall trade-off.
    parts: List[str] = []
    if rng.random() < 0.85:
        parts.append(rng.choice(_HEADINGS))                       # short title line
    if rng.random() < 0.45:
        parts.append(rng.choice(_HEADINGS))                       # a subheading too
    parts.append(body)
    if rng.random() < 0.35:
        items = rng.sample(_LIST_ITEMS, rng.randint(2, 3))
        parts.append("\n".join(f"- {it}" for it in items))        # short bullet list
    return "\n\n".join(parts)


def make_json_record(rng: random.Random, text: str, source: str) -> Dict[str, object]:
    return {
        "id": f"rec-{rng.randrange(10**6):06d}",
        "url": f"https://{source}.example.org/p/{rng.randrange(10**5)}",
        "text": text,
        "lang": "en",
        "tokens": len(text.split()),
        "title": " ".join(text.split()[:5]),
    }


def clean_corpus(n: int = 400, seed: int = 7,
                 modalities: Sequence[str] = ("html", "ocr", "json")) -> List[Document]:
    """A balanced corpus of clean documents across modalities and sources."""
    rng = random.Random(seed)
    docs: List[Document] = []
    for i in range(n):
        modality = modalities[i % len(modalities)]
        source = _SOURCES[rng.randrange(len(_SOURCES))]
        text = make_clean_text(rng)
        raw: Optional[str] = None
        if modality == "html":
            raw = f"<html><body><article><p>{text.replace(chr(10)+chr(10), '</p><p>')}</p></article></body></html>"
        elif modality == "json":
            rec = make_json_record(rng, text, source)
            raw = json.dumps(rec, ensure_ascii=False)
        docs.append(Document(id=f"clean-{i:05d}", text=text, source=source,
                             modality=modality, raw=raw))
    return docs


def load_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
