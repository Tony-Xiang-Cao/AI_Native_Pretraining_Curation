"""Command-line interface: ``python -m autocurate <command>``.

Commands
  profile <text>         cheap reference-free heuristics for one document
  gate <modality> <text> reference-free extraction-quality score (html/ocr/json/text)
  curate <text>          end-to-end cascade decision for one document
  experiments            run E1-E5 and print a summary
  report <jsonl>         curate a JSONL corpus and emit a quality report
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .datagen import load_jsonl
from .extract import GATES
from .pipeline import CurationLoop
from .profile import text_features
from .report import build_report, render_markdown
from .schema import Document


def _cmd_profile(args) -> None:
    feats = text_features(args.text)
    print(json.dumps({k: round(v, 4) for k, v in feats.items()}, indent=2))


def _cmd_gate(args) -> None:
    if args.modality not in GATES:
        print(f"no gate for modality '{args.modality}' (choose from {list(GATES)})")
        return
    gate = GATES[args.modality]()
    doc = Document(id="cli", text=args.text, modality=args.modality, raw=args.text)
    res = gate.evaluate(doc)
    print(json.dumps({"gate": res.gate, "quality": round(res.quality, 4),
                      "passed": res.passed, "flags": res.flags,
                      "signals": res.signals}, indent=2))


def _cmd_curate(args) -> None:
    judge = None
    if args.judge:
        from .judge import JudgeAdapter
        judge = JudgeAdapter(args.judge)
    loop = CurationLoop(judge=judge, mode="cascade" if judge else "heuristic")
    dec, rec = loop.curate(Document(id="cli", text=args.text, modality=args.modality))
    print(json.dumps({"label": dec.label_name, "score": round(dec.score, 4),
                      "stage": dec.stage, "reasons": dec.reasons}, indent=2))


def _cmd_experiments(args) -> None:
    from . import experiments
    res = experiments.run_all()
    print(json.dumps(res, indent=2)[:4000])


def _cmd_report(args) -> None:
    rows = load_jsonl(args.path)
    docs = [Document(id=str(r.get("id", i)), text=r.get("text", ""),
                     source=r.get("source", "unknown"),
                     modality=r.get("modality", "text"), raw=r.get("raw"))
            for i, r in enumerate(rows)]
    loop = CurationLoop(mode="heuristic")
    decisions, records = loop.curate_batch(docs)
    rep = build_report(decisions, records, judge_calls=loop.judge_calls)
    print(render_markdown(rep) if args.markdown else json.dumps(rep, indent=2))


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="autocurate", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("profile")
    sp.add_argument("text")
    sp.set_defaults(fn=_cmd_profile)

    sp = sub.add_parser("gate")
    sp.add_argument("modality")
    sp.add_argument("text")
    sp.set_defaults(fn=_cmd_gate)

    sp = sub.add_parser("curate")
    sp.add_argument("text")
    sp.add_argument("--modality", default="text")
    sp.add_argument("--judge", default=None)
    sp.set_defaults(fn=_cmd_curate)

    sp = sub.add_parser("experiments")
    sp.set_defaults(fn=_cmd_experiments)

    sp = sub.add_parser("report")
    sp.add_argument("path")
    sp.add_argument("--markdown", action="store_true")
    sp.set_defaults(fn=_cmd_report)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
