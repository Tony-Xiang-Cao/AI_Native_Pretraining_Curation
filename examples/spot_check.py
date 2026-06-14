#!/usr/bin/env python3
"""Qualitative face-validity check: run the reference-free gates on realistic
samples that mimic real extraction defects (committed, deterministic), and
optionally on a *live* URL.

    PYTHONPATH=src python examples/spot_check.py
    PYTHONPATH=src python examples/spot_check.py --url https://example.com/article

The embedded samples are hand-authored to look like real HTML extractions, real
OCR output, and real JSONL records (clean and defective). This is a sanity check
that the gates behave sensibly on realistic text — not a benchmark; the
verifiable numbers are the mutation-oracle experiments (E1, E7).
"""

import argparse

from autocurate.extract import HTMLGate, JSONGate, OCRGate
from autocurate.schema import Document

SAMPLES = [
    # (modality, label, text, raw)
    ("html", "clean", "Photosynthesis converts light energy into chemical energy stored "
     "in glucose. In plants, the reaction occurs in the chloroplasts, where chlorophyll "
     "absorbs primarily red and blue wavelengths. The 1779 experiments of Jan Ingenhousz "
     "first showed that light is required for the process.", None),
    ("html", "boilerplate-leaked", "Skip to main content\nCookie settings\nWe use cookies "
     "to improve your experience.\nSubscribe to our newsletter\nPhotosynthesis converts "
     "light energy into chemical energy.\nShare on Twitter\nShare on Facebook\n"
     "© 2024 Example Media. All rights reserved.\nRelated stories", None),
    ("ocr", "clean", "The manuscript was completed in the spring of 1923 and circulated "
     "privately among a small group of scholars before its first public printing.", None),
    ("ocr", "ocr-errors", "The rnanuscript was cornpleted in the spri ng of 1923 and "
     "circu-\nlated privately arnong a srnall group of sch0lars before its f1rst public "
     "printi ng.", None),
    ("json", "clean", None, '{"id": "doc-42", "url": "https://ex.org/a", "text": "A short '
     'clean article about rivers.", "lang": "en", "tokens": 7}'),
    ("json", "truncated", None, '{"id": "doc-43", "url": "https://ex.org/b", "text": "A '
     'record cut off mid-w'),
]


def _gate_for(modality):
    return {"html": HTMLGate(), "ocr": OCRGate(), "json": JSONGate()}[modality]


def run_samples():
    print(f"{'modality':6} {'label':18} {'quality':>7}  passed  flags")
    print("-" * 70)
    json_gate = JSONGate().fit_schema([s[3] for s in SAMPLES if s[0] == "json" and s[3]])
    for modality, label, text, raw in SAMPLES:
        gate = json_gate if modality == "json" else _gate_for(modality)
        doc = Document(id=label, text=text or "", modality=modality, raw=raw)
        r = gate.evaluate(doc)
        print(f"{modality:6} {label:18} {r.quality:7.2f}  {str(r.passed):5}  {','.join(r.flags)}")


def run_url(url):
    import re
    import urllib.request
    html = urllib.request.urlopen(url, timeout=20).read().decode("utf-8", "ignore")
    # naive strip-tags "extraction" — exactly the kind of low-quality extraction the gate flags
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    r = HTMLGate().evaluate(Document(id="live", text=text[:4000], modality="html"))
    print(f"\nlive {url}\n  naive strip-tags extraction: quality={r.quality:.2f} "
          f"passed={r.passed} flags={r.flags}")
    print("  (a good extractor — trafilatura/Resiliparse — would score higher; "
          "naive strip-tags leaks boilerplate, which the gate catches.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="optional live URL to spot-check")
    args = ap.parse_args()
    run_samples()
    if args.url:
        run_url(args.url)


if __name__ == "__main__":
    main()
