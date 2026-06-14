#!/usr/bin/env python3
"""Run E2 (cascade efficiency) with a REAL LLM judge instead of the offline one.

The committed E2 uses ``judgecurate``'s deterministic heuristic judge so the
numbers reproduce in CI. This script reruns the identical cascade with a real
LLM backend (Claude / GPT / a local model) and writes ``results/e2_cascade_llm.json``.
It is NOT part of the reproducible suite (it needs an API key and is non-
deterministic), but it lets the "recovers ~69% of the judge's gain" claim be
checked against an actual judge.

    pip install -e ".[judge,llm]"
    export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY
    python scripts/run_llm_e2.py --judge anthropic --limit 150
    python scripts/run_llm_e2.py --judge openai --model gpt-4o-mini
    python scripts/run_llm_e2.py --judge vllm --model Qwen/Qwen2.5-7B-Instruct \
        --base-url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
_SIB = os.path.join(os.path.dirname(__file__), "..", "..",
                    "LLM_as_Judge_Pretraining_Data_Curation", "src")
if os.path.isdir(_SIB):
    sys.path.insert(0, _SIB)

from autocurate import experiments  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="anthropic",
                    help="judgecurate backend: anthropic | openai | vllm | ollama")
    ap.add_argument("--model", default=None, help="model id (backend default if omitted)")
    ap.add_argument("--base-url", default=None, help="for vllm/ollama OpenAI-compatible servers")
    ap.add_argument("--limit", type=int, default=150, help="cap documents to control API cost")
    args = ap.parse_args()

    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    if args.base_url:
        kwargs["base_url"] = args.base_url

    print(f"running E2 with real judge backend='{args.judge}' on {args.limit} docs ...")
    res = experiments.e2_cascade(judge_backend=args.judge, limit=args.limit, **kwargs)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "e2_cascade_llm.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    if "skipped" in res:
        print("skipped:", res["skipped"])
    else:
        op = res["operating_point"]
        print(f"router-only {res['heuristic_only_f1']} -> judge-everything {res['judge_only_f1']} "
              f"macro-F1; at {op['judge_call_rate']:.0%} calls -> {op['macro_f1']} "
              f"(recovered {op['recovered_gain']:.0%}). wrote {out}")


if __name__ == "__main__":
    main()
