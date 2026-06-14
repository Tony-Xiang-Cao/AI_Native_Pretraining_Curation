#!/usr/bin/env python3
"""Run E1-E5 and write the committed ``results/*.json`` (asserted by the tests)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
# allow importing the sibling judgecurate without installing it
_SIB = os.path.join(os.path.dirname(__file__), "..", "..",
                    "LLM_as_Judge_Pretraining_Data_Curation", "src")
if os.path.isdir(_SIB):
    sys.path.insert(0, _SIB)

from autocurate import experiments  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    fns = {
        "e1_gate_oracle": experiments.e1_gate_oracle,
        "e2_cascade": experiments.e2_cascade,
        "e3_hillclimb": experiments.e3_hillclimb,
        "e4_drift": experiments.e4_drift,
        "e5_outliers": experiments.e5_outliers,
        "e6_drift_recovery": experiments.e6_drift_recovery,
        "e7_baselines": experiments.e7_baselines,
    }
    summary = {}
    for name, fn in fns.items():
        print(f"running {name} ...", flush=True)
        res = fn()
        with open(os.path.join(RESULTS, f"{name}.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        summary[name] = res
    with open(os.path.join(RESULTS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote results/*.json")


if __name__ == "__main__":
    main()
