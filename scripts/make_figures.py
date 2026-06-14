#!/usr/bin/env python3
"""Render paper figures (SVG) from results/*.json — pure stdlib, deterministic."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from autocurate.svgplot import bar_chart, line_chart  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")


def _load(name):
    with open(os.path.join(RES, f"{name}.json")) as fh:
        return json.load(fh)


def _save(name, svg):
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(FIG, name), "w") as fh:
        fh.write(svg)


FRAMEWORK = '''<svg xmlns="http://www.w3.org/2000/svg" width="860" height="300" viewBox="0 0 860 300" font-family="Helvetica,Arial,sans-serif">
<rect width="860" height="300" fill="white"/>
<text x="430" y="26" font-size="17" text-anchor="middle" font-weight="bold">The AutoCurate operating loop (one daily tick)</text>
<g font-size="12.5" text-anchor="middle">
<rect x="20" y="70" width="120" height="56" rx="8" fill="#dbeafe" stroke="#2563eb"/><text x="80" y="94">crawl shards</text><text x="80" y="112">+ provenance</text>
<rect x="165" y="70" width="120" height="56" rx="8" fill="#dbeafe" stroke="#2563eb"/><text x="225" y="94">extraction gates</text><text x="225" y="112">HTML/OCR/JSON</text>
<rect x="310" y="70" width="120" height="56" rx="8" fill="#dbeafe" stroke="#2563eb"/><text x="370" y="90">profile +</text><text x="370" y="106">outlier screen</text>
<rect x="455" y="70" width="120" height="56" rx="8" fill="#dbeafe" stroke="#2563eb"/><text x="515" y="90">cascade route</text><text x="515" y="106">judge the middle</text>
<rect x="600" y="70" width="115" height="56" rx="8" fill="#dcfce7" stroke="#059669"/><text x="657" y="90">decision +</text><text x="657" y="106">quality report</text>
<rect x="310" y="180" width="170" height="56" rx="8" fill="#fee2e2" stroke="#dc2626"/><text x="395" y="202">verify (mutation oracle)</text><text x="395" y="220">+ verified hill-climb</text>
<rect x="520" y="180" width="195" height="56" rx="8" fill="#fef9c3" stroke="#d97706"/><text x="617" y="202">SPC drift monitor</text><text x="617" y="220">throughput &amp; quality</text>
</g>
<g stroke="#555" fill="none" stroke-width="1.6" marker-end="url(#a)">
<defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#555"/></marker></defs>
<path d="M140,98 L162,98"/><path d="M285,98 L307,98"/><path d="M430,98 L452,98"/><path d="M575,98 L597,98"/>
<path d="M657,128 L657,160 L720,160 L720,205 L718,205"/>
<path d="M520,180 L520,150 L516,150"/>
<path d="M395,180 L395,150 L375,150 L375,128"/>
<path d="M310,208 L150,208 L150,128"/>
</g>
<text x="225" y="262" font-size="11.5" text-anchor="middle" fill="#dc2626">drift / quality regression → trigger verified hill-climb → re-extract</text>
</svg>'''


def main():
    _save("framework.svg", FRAMEWORK)

    e1 = _load("e1_gate_oracle")
    gates = ["html", "ocr", "json"]
    _save("e1_gates.svg", bar_chart(
        gates,
        {"held-out floor": [e1[g]["floor_f1"] for g in gates],
         "upper bound (full vocab)": [e1[g]["upper_f1"] for g in gates]},
        "E1 — extraction-gate F1 via the mutation oracle"))

    e2 = _load("e2_cascade")
    pts_f1 = [(c["judge_calls"] / e2["n"], c["macro_f1"]) for c in e2["curve"]]
    pts_ag = [(c["judge_calls"] / e2["n"], c["agreement"]) for c in e2["curve"]]
    _save("e2_cascade.svg", line_chart(
        {"macro-F1": pts_f1, "agreement": pts_ag},
        "E2 — cascade: quality vs judge-call budget",
        "judge-call rate (fraction of documents escalated)", "score",
        ymax=0.8, ymin=0.3, xmax=1.0))

    e3 = _load("e3_hillclimb")
    nr = [(t["iter"], t["guard_fpr"]) for t in e3["naive_recall"]["trajectory"]]
    nf = [(t["iter"], t["guard_fpr"]) for t in e3["naive_f1"]["trajectory"]]
    vf = [(t["iter"], t["guard_fpr"]) for t in e3["verified"]["trajectory"]]
    _save("e3_hillclimb.svg", line_chart(
        {"naive (recall): guard FPR": nr, "naive (F1): guard FPR": nf,
         "verified: guard FPR": vf},
        "E3 — held-out clean-guard FPR during self-improvement",
        "hill-climb iteration", "clean-guard false-positive rate", ymax=0.6, ymin=0.0,
        xmax=max(t["iter"] for t in e3["naive_recall"]["trajectory"])))

    e4 = _load("e4_drift")
    q = [(s["day"], s["quality"]) for s in e4["series"]]
    _save("e4_drift.svg", line_chart(
        {"daily extraction quality": q},
        "E4 — drift detection & recovery on a daily stream",
        "day", "mean gate quality", ymax=1.0, ymin=0.4,
        xmax=max(s["day"] for s in e4["series"]),
        markers=[(e4["shift_day"], "parser regression")]))

    e5 = _load("e5_outliers")
    metr = ["precision", "recall", "f1"]
    _save("e5_outliers.svg", bar_chart(
        metr,
        {"robust-z (MAD)": [e5["robust_z_mad"][m] for m in metr],
         "mean ± kσ": [e5["mean_k_sigma"][m] for m in metr]},
        "E5 — outlier detection: robust-z vs mean ± kσ"))

    print("wrote figures to", FIG)


if __name__ == "__main__":
    main()
