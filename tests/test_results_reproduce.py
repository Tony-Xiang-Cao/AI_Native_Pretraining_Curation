"""Recompute E1-E5 and assert they match the committed ``results/*.json``.

Everything is deterministic, so the committed numbers are the contract: if the
mechanism changes, this test fails until the results (and the paper) are
regenerated. Mirrors judgecurate's "tests assert the committed numbers" policy.
"""

import json
import os

import pytest

from autocurate import experiments

RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def _committed(name):
    with open(os.path.join(RES, f"{name}.json")) as fh:
        return json.load(fh)


def test_e1_reproduces():
    assert experiments.e1_gate_oracle() == _committed("e1_gate_oracle")


def test_e3_reproduces():
    assert experiments.e3_hillclimb() == _committed("e3_hillclimb")


def test_e4_reproduces():
    assert experiments.e4_drift() == _committed("e4_drift")


def test_e5_reproduces():
    assert experiments.e5_outliers() == _committed("e5_outliers")


def test_e2_reproduces_if_judge_available():
    committed = _committed("e2_cascade")
    if "skipped" in committed:
        pytest.skip("e2 committed as skipped (no judgecurate)")
    fresh = experiments.e2_cascade()
    if "skipped" in fresh:
        pytest.skip("judgecurate not importable in this environment")
    assert fresh == committed


def test_headline_numbers_are_sane():
    e1 = _committed("e1_gate_oracle")
    assert e1["macro"]["floor_f1"] <= e1["macro"]["upper_f1"]
    e3 = _committed("e3_hillclimb")
    assert e3["reward_hack_fpr_gap"] > 0.1            # naive over-flags, verified does not
    e5 = _committed("e5_outliers")
    assert e5["robust_z_mad"]["f1"] >= e5["mean_k_sigma"]["f1"]
