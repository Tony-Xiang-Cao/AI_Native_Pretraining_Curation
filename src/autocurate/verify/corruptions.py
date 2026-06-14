"""Mutation operators: inject *known* defects into clean documents.

Each operator is a deterministic, seedable function that damages a clean
``Document`` and returns the corrupted copy. Operators that draw from a finite
vocabulary take a ``slice`` argument selecting the **train** or **heldout**
partition (``lexicons.py``): scoring on the held-out slice measures
generalization (the conservative floor), scoring on train measures the
read-back upper bound, and the gap between them is the memorization a gate is
not allowed to be rewarded for.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from .. import lexicons as LEX
from ..schema import Document


@dataclass
class DefectRecord:
    """Ground-truth log of a single injected defect."""

    doc_id: str
    modality: str
    operator: str
    slice: str          # "train" | "heldout"


def _slice_list(train: Sequence, heldout: Sequence, which: str) -> Sequence:
    return train if which == "train" else heldout


# --------------------------------------------------------------------------- #
# HTML corruptions
# --------------------------------------------------------------------------- #

def html_tag_inject(doc: Document, rng: random.Random, which: str = "heldout") -> Document:
    words = doc.text.split(" ")
    tags = ["<div class=\"col\">", "<span>", "<p>", "</p>", "</div>",
            "&nbsp;", "&amp;", "<!-- ad -->", "<br/>"]
    for _ in range(max(3, len(words) // 25)):
        i = rng.randrange(max(1, len(words)))
        words.insert(i, rng.choice(tags))
    return _with_text(doc, " ".join(words))


def html_nav_boilerplate(doc: Document, rng: random.Random, which: str = "heldout") -> Document:
    phrases = _slice_list(LEX.BOILERPLATE_TRAIN, LEX.BOILERPLATE_HELDOUT, which)
    # ~30% of injections are "light" (1-2 chrome lines): borderline cases that
    # land near the gate threshold, so catching them trades off against
    # false-positives on clean headings/lists — the realistic gray zone.
    n = rng.randint(1, 2) if rng.random() < 0.3 else rng.randint(4, 8)
    chrome = [rng.choice(phrases).title() for _ in range(n)]
    head = "\n".join(chrome[: max(1, len(chrome) // 2)])
    tail = "\n".join(chrome[len(chrome) // 2:])
    return _with_text(doc, f"{head}\n{doc.text}\n{tail}")


def html_script_style(doc: Document, rng: random.Random, which: str = "heldout") -> Document:
    block = ("<script>function track(){var x=window.location;"
             "document.addEventListener('click',track);}</script>"
             "<style>.nav{display:flex;padding:10px;margin:0px;}@media{}</style>")
    return _with_text(doc, f"{block}\n{doc.text}")


def html_link_dump(doc: Document, rng: random.Random, which: str = "heldout") -> Document:
    links = "\n".join(
        f"[link {k}](https://ads.example.com/{rng.randrange(10**5)})"
        for k in range(rng.randint(6, 12))
    )
    return _with_text(doc, f"{doc.text}\n{links}")


# --------------------------------------------------------------------------- #
# OCR corruptions
# --------------------------------------------------------------------------- #

def ocr_charsub(doc: Document, rng: random.Random, which: str = "heldout") -> Document:
    confs = _slice_list(LEX.OCR_CONFUSIONS_TRAIN, LEX.OCR_CONFUSIONS_HELDOUT, which)
    text = doc.text
    for a, b in confs:
        out = []
        i = 0
        while i < len(text):
            if text[i : i + len(a)] == a and rng.random() < 0.5:
                out.append(b)
                i += len(a)
            else:
                out.append(text[i])
                i += 1
        text = "".join(out)
    return _with_text(doc, text)


def ocr_wordbreak(doc: Document, rng: random.Random, which: str = "heldout") -> Document:
    words = doc.text.split(" ")
    out: List[str] = []
    for w in words:
        if len(w) > 5 and rng.random() < 0.25:
            k = rng.randint(2, len(w) - 2)
            out.append(w[:k] + " " + w[k:])
        elif len(w) > 6 and rng.random() < 0.1:
            k = rng.randint(2, len(w) - 2)
            out.append(w[:k] + "-\n" + w[k:])
        else:
            out.append(w)
    return _with_text(doc, " ".join(out))


def ocr_mojibake(doc: Document, rng: random.Random, which: str = "heldout") -> Document:
    text = doc.text
    n = max(6, len(text) // 80)
    chars = list(text)
    for _ in range(n):
        i = rng.randrange(max(1, len(chars)))
        chars.insert(i, rng.choice(LEX.MOJIBAKE_FRAGMENTS))
    return _with_text(doc, "".join(chars))


def ocr_linenoise(doc: Document, rng: random.Random, which: str = "heldout") -> Document:
    glyphs = list("¬|~^`¦§°±•")
    tokens = doc.text.split(" ")
    for _ in range(max(8, len(tokens) // 8)):
        i = rng.randrange(max(1, len(tokens)))
        tokens.insert(i, rng.choice(glyphs))
    return _with_text(doc, " ".join(tokens))


# --------------------------------------------------------------------------- #
# JSON corruptions (operate on the serialized record in ``raw``)
# --------------------------------------------------------------------------- #

def json_truncate(doc: Document, rng: random.Random, which: str = "heldout") -> Document:
    raw = doc.raw or doc.text
    cut = rng.randint(int(len(raw) * 0.3), int(len(raw) * 0.9))
    return _with_raw(doc, raw[:cut])


def json_schema_break(doc: Document, rng: random.Random, which: str = "heldout") -> Document:
    raw = doc.raw or doc.text
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return _with_raw(doc, raw[: len(raw) // 2])
    if not isinstance(obj, dict) or not obj:
        return doc
    keys = list(obj.keys())
    mode = rng.choice(["delete", "retype", "empty"])
    k = rng.choice(keys)
    if mode == "delete":
        obj.pop(k, None)
    elif mode == "retype":
        obj[k] = {"v": obj[k]} if not isinstance(obj[k], dict) else "string-now"
    else:
        obj[k] = None
    return _with_raw(doc, json.dumps(obj, ensure_ascii=False))


def json_delimiter(doc: Document, rng: random.Random, which: str = "heldout") -> Document:
    raw = doc.raw or doc.text
    if rng.random() < 0.5 and raw.endswith("}"):
        return _with_raw(doc, raw[:-1])                       # drop closing brace
    return _with_raw(doc, raw.replace(",", ";", 1))           # break a delimiter


# --------------------------------------------------------------------------- #
# Content corruptions (semantic; for completeness / risk experiments)
# --------------------------------------------------------------------------- #

def risk_inject(doc: Document, rng: random.Random, which: str = "heldout") -> Document:
    markers = _slice_list(LEX.RISK_TRAIN, LEX.RISK_HELDOUT, which)
    return _with_text(doc, f"{doc.text}\n{rng.choice(markers)}.")


def contra_inject(doc: Document, rng: random.Random, which: str = "heldout") -> Document:
    markers = _slice_list(LEX.CONTRA_TRAIN, LEX.CONTRA_HELDOUT, which)
    return _with_text(doc, f"{doc.text} {rng.choice(markers)}.")


# --------------------------------------------------------------------------- #
# Registry + helpers
# --------------------------------------------------------------------------- #

CORRUPTORS: Dict[str, List[Callable]] = {
    "html": [html_tag_inject, html_nav_boilerplate, html_script_style, html_link_dump],
    "ocr": [ocr_charsub, ocr_wordbreak, ocr_mojibake, ocr_linenoise],
    "json": [json_truncate, json_schema_break, json_delimiter],
    "content": [risk_inject, contra_inject],
}


def _with_text(doc: Document, text: str) -> Document:
    return Document(id=doc.id, text=text, source=doc.source, modality=doc.modality,
                    raw=doc.raw, meta=dict(doc.meta))


def _with_raw(doc: Document, raw: str) -> Document:
    return Document(id=doc.id, text=doc.text, source=doc.source, modality=doc.modality,
                    raw=raw, meta=dict(doc.meta))


def corrupt(doc: Document, modality: str, rng: random.Random,
            which: str = "heldout") -> Tuple[Document, DefectRecord]:
    """Apply one randomly chosen operator for ``modality`` to a clean doc."""
    op = rng.choice(CORRUPTORS[modality])
    corrupted = op(doc, rng, which)
    return corrupted, DefectRecord(doc.id, modality, op.__name__, which)


def build_mutation_set(clean_docs: Sequence[Document], modality: str,
                       rng: random.Random, which: str = "heldout"
                       ) -> List[Tuple[Document, int]]:
    """Balanced labelled set: each clean doc yields a clean (0) and corrupt (1) copy."""
    out: List[Tuple[Document, int]] = []
    for d in clean_docs:
        out.append((d, 0))
        corrupted, _ = corrupt(d, modality, rng, which)
        out.append((corrupted, 1))
    return out
