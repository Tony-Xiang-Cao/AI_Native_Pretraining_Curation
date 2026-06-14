#!/usr/bin/env python3
"""Run the harness-agnostic operating loop over a simulated multi-day crawl and
print a quality report — the end-to-end AI-native curation loop.

    PYTHONPATH=src python examples/full_loop.py
"""

import random

from autocurate.agentloop import CrawlMonitorRoutine, LocalCronHarness, ProvenanceLedger, StreamMonitor
from autocurate.datagen import clean_corpus
from autocurate.pipeline import CurationLoop
from autocurate.report import build_report, render_markdown
from autocurate.verify.corruptions import corrupt


def reextract(docs):
    """A 'better extractor' the routine applies on a quality alarm."""
    from autocurate.schema import Document
    out = []
    for d in docs:
        lines = [ln for ln in d.text.split("\n")
                 if not (0 < len(ln.strip()) < 25 and not ln.strip().endswith("."))]
        out.append(Document(id=d.id, text=" ".join(lines), source=d.source, modality="html"))
    return out


def main():
    loop = CurationLoop(mode="heuristic")
    ledger = ProvenanceLedger()
    routine = CrawlMonitorRoutine(
        loop, ledger, warmup=8,
        quality_monitor=StreamMonitor(alpha=0.3, watch="down"),
        remediator=reextract)
    harness = LocalCronHarness(ledger, verbose=True)
    rng = random.Random(1)

    print(f"schedule line: {harness.schedule('0 7 * * *', routine.routine_id)}\n")
    for day in range(18):
        docs = [d for d in clean_corpus(60, seed=200 + day) if d.modality == "html"][:30]
        if day >= 11:                       # an upstream parser regression appears
            docs = [corrupt(d, "html", rng, "heldout")[0] if i % 2 == 0 else d
                    for i, d in enumerate(docs)]
        routine.tick(harness, docs)

    # final-day quality report (one batch)
    docs = [d for d in clean_corpus(60, seed=999) if d.modality == "html"][:30]
    decisions, records = loop.curate_batch(docs)
    report = build_report(decisions, records, judge_calls=loop.judge_calls)
    print("\n" + render_markdown(report))
    print(f"ledger entries: {len(ledger.records())} "
          f"(decisions={len(ledger.records('decision'))}, summaries={len(ledger.records('summary'))})")


if __name__ == "__main__":
    main()
