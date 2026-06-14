import random

from autocurate.agentloop import (
    CrawlMonitorRoutine, LocalCronHarness, ProvenanceLedger, StreamMonitor,
)
from autocurate.datagen import clean_corpus
from autocurate.pipeline import CurationLoop
from autocurate.schema import ProvenanceRecord
from autocurate.verify.corruptions import corrupt


def test_stream_monitor_flags_downward_shift():
    mon = StreamMonitor(alpha=0.3, watch="down")
    mon.fit([0.9, 0.91, 0.89, 0.9, 0.92, 0.88, 0.9, 0.9, 0.91, 0.9])
    pre = [mon.update(0.9, d) for d in range(5)]
    assert not any(s.alarm for s in pre)               # stable -> no alarm
    post = [mon.update(0.5, d) for d in range(5, 10)]
    assert any(s.alarm and s.direction == "down" for s in post)


def test_ledger_append_and_trace():
    led = ProvenanceLedger()
    rec = ProvenanceRecord("d1", "web", "2026-01-01T00:00:00Z", "html",
                           "html-gate@v1", 0.9, "retain", "heuristic", "abc")
    led.append_record(rec)
    led.append_summary({"day": 0, "quality": 0.9})
    assert len(led.records("decision")) == 1
    assert len(led.records("summary")) == 1
    assert led.trace("d1")[0]["decision"] == "retain"


def test_routine_detects_drift_and_recovers():
    def reextract(docs):
        # trivial "better extractor": drop short non-prose lines
        from autocurate.schema import Document
        out = []
        for d in docs:
            lines = [ln for ln in d.text.split("\n")
                     if not (0 < len(ln.strip()) < 25 and not ln.strip().endswith("."))]
            out.append(Document(id=d.id, text=" ".join(lines), source=d.source, modality="html"))
        return out

    loop = CurationLoop(mode="heuristic")
    routine = CrawlMonitorRoutine(loop, warmup=8,
                                  quality_monitor=StreamMonitor(alpha=0.3, watch="down"),
                                  remediator=reextract)
    h = LocalCronHarness()
    rng = random.Random(1)
    alarmed = False
    for day in range(20):
        base = [d for d in clean_corpus(60, seed=200 + day) if d.modality == "html"][:30]
        if day >= 12:
            base = [corrupt(d, "html", rng, "heldout")[0] if i % 2 == 0 else d
                    for i, d in enumerate(base)]
        r = routine.tick(h, base)
        if day >= 12 and "quality_drift_down" in r.alerts:
            alarmed = True
    assert alarmed
