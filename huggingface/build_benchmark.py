#!/usr/bin/env python3
"""Materialize the extraction-quality mutation benchmark as a JSONL for upload.

Each line is one labelled example: a clean document or a corrupted copy, tagged
with modality, the corruption operator, the vocabulary slice (train/heldout),
and the binary defect label. The set is generated deterministically by
``autocurate.datagen`` + ``autocurate.verify.corruptions`` so it reproduces
bit-for-bit.

    PYTHONPATH=../src python build_benchmark.py        # writes extraction_mutation_bench.jsonl
"""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from autocurate.datagen import clean_corpus            # noqa: E402
from autocurate.verify.corruptions import CORRUPTORS, corrupt  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "extraction_mutation_bench.jsonl")


def build(n_per_modality: int = 150, seed: int = 7):
    docs = clean_corpus(n_per_modality * 4, seed=seed)
    rng = random.Random(seed)
    rows = []
    for mod in ("html", "ocr", "json"):
        dd = [d for d in docs if d.modality == mod][:n_per_modality]
        for d in dd:
            rows.append({"id": d.id, "modality": mod, "text": d.text, "raw": d.raw,
                         "defect": 0, "operator": "", "slice": ""})
            for which in ("train", "heldout"):
                op = rng.choice(CORRUPTORS[mod])
                c = op(d, rng, which)
                rows.append({"id": f"{d.id}-{op.__name__}-{which}", "modality": mod,
                             "text": c.text, "raw": c.raw, "defect": 1,
                             "operator": op.__name__, "slice": which})
    return rows


def main():
    rows = build()
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_def = sum(r["defect"] for r in rows)
    print(f"wrote {len(rows)} examples ({n_def} corrupted, {len(rows)-n_def} clean) -> {OUT}")


if __name__ == "__main__":
    main()
