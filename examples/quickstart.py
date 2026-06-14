#!/usr/bin/env python3
"""Quickstart: profile a document, gate an extraction, verify a gate, hill-climb.

Run with the repo's src on the path:
    PYTHONPATH=src python examples/quickstart.py
"""

from autocurate import (
    CurationLoop, Document, HTMLGate, evaluate_gate, hillclimb, text_features,
)
from autocurate.datagen import clean_corpus

# 1) cheap reference-free heuristics --------------------------------------- #
feats = text_features("The enzyme catalyzed the reaction. According to the 1998 "
                      "study, the rate increased by 40 percent.")
print("gzip_ratio:", round(feats["gzip_ratio"], 3),
      "| stopword_fraction:", round(feats["stopword_fraction"], 3))

# 2) reference-free HTML extraction-quality gate --------------------------- #
leaky = Document(id="x", modality="html", text=(
    "Skip to content\nCookie preferences\nSubscribe to our newsletter\n"
    "<div class=\"ad\">actual article text.</div>\nBack to top\nFollow us"))
gr = HTMLGate().evaluate(leaky)
print(f"\nHTML gate: quality={gr.quality:.2f} passed={gr.passed} flags={gr.flags}")

# 3) verify the gate against the mutation oracle (no human labels) --------- #
html = [d for d in clean_corpus(160, seed=7) if d.modality == "html"][:60]
floor = evaluate_gate(HTMLGate(), html, "html", seed=0, which="heldout").f1
print(f"\nHTML gate held-out F1 (mutation oracle): {floor:.3f}")

# 4) verified self-improvement keeps the clean-guard FPR at ~0 ------------- #
res = hillclimb(HTMLGate(), html, "html", regime="verified")
print(f"verified hill-climb: end guard_fpr={res.final['guard_fpr']:.3f} "
      f"(naive would inflate this).")

# 5) end-to-end cascade decision (cheap heuristic tier) -------------------- #
loop = CurationLoop(mode="heuristic")
dec, rec = loop.curate(Document(id="s", modality="text",
                                text="Buy now! Click here for free money. Act now!"))
print(f"\ncascade decision: {dec.label_name} (stage={dec.stage}) "
      f"-> responsible={rec.responsible_stage}")
