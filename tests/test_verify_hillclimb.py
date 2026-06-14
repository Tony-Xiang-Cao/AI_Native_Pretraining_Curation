from autocurate.datagen import clean_corpus
from autocurate.extract import HTMLGate
from autocurate.hillclimb import hillclimb
from autocurate.lexicons import BOILERPLATE_HELDOUT, BOILERPLATE_TRAIN
from autocurate.schema import HillclimbConfig
from autocurate.verify import accept_candidate, floor_and_upper, guard_fpr


def test_held_out_vocab_is_disjoint():
    # de-circularization: a gate may only know the train slice
    assert set(BOILERPLATE_TRAIN).isdisjoint(set(BOILERPLATE_HELDOUT))
    assert len(BOILERPLATE_HELDOUT) >= 1


def test_floor_not_above_upper():
    docs = [d for d in clean_corpus(160, seed=7) if d.modality == "html"][:60]
    floor, upper = floor_and_upper(HTMLGate(), docs, "html", seeds=range(3))
    assert floor <= upper + 1e-9
    assert guard_fpr(HTMLGate(), docs) <= 0.1


def _degraded():
    g = HTMLGate()
    for k in g.weights:
        g.weights[k] *= 0.5
    g.threshold = 0.5
    return g


def test_accept_rule_fpr_clause_is_load_bearing():
    """A candidate that *improves* held-out F1 but raises clean-guard FPR must be
    rejected by the FPR clause specifically — not incidentally by the F1 clause."""
    docs = [d for d in clean_corpus(200, seed=7) if d.modality == "html"][:50]
    inc = _degraded()
    cand = _degraded()
    cand.threshold = 0.9                       # catches more defects AND flags more clean
    strict = accept_candidate(cand, inc, docs, "html", docs, eps=0.005)
    loose = accept_candidate(cand, inc, docs, "html", docs, eps=1.0)
    # held-out F1 genuinely improved (so the F1 clause is satisfied) ...
    assert strict.lower_bound > 0.005
    assert strict.delta_fpr > 0.01
    # ... yet the strict guard rejects, and only relaxing eps lets it through:
    assert not strict.accept
    assert "reward-hack lock" in strict.reason
    assert loose.accept


def test_naive_recall_reward_hacks_verified_does_not():
    climb = [d for d in clean_corpus(240, seed=7) if d.modality == "html"][:60]
    held = [d for d in clean_corpus(240, seed=8) if d.modality == "html"][:60]
    cfg = HillclimbConfig(iterations=20, population=8, step=0.12, seed=3)
    naive = hillclimb(_degraded(), climb, "html", regime="naive_recall", config=cfg,
                      eval_docs=held, guard_docs=held)
    verified = hillclimb(_degraded(), climb, "html", regime="verified", config=cfg,
                         eval_docs=held, guard_docs=held)
    # un-guarded recall over-flags held-out clean data; verified keeps FPR at zero
    assert naive.final["guard_fpr"] > verified.final["guard_fpr"] + 0.1
    assert verified.final["guard_fpr"] <= 0.02
