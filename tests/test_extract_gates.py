import random

import pytest

from autocurate.datagen import clean_corpus
from autocurate.extract import HTMLGate, JSONGate, OCRGate
from autocurate.verify.corruptions import CORRUPTORS


@pytest.fixture(scope="module")
def corpus():
    return clean_corpus(120, seed=7)


def _sub(corpus, mod):
    return [d for d in corpus if d.modality == mod][:30]


@pytest.mark.parametrize("name,GateCls,mod", [
    ("html", HTMLGate, "html"), ("ocr", OCRGate, "ocr"), ("json", JSONGate, "json")])
def test_clean_passes_corrupt_flagged(name, GateCls, mod, corpus):
    dd = _sub(corpus, mod)
    gate = GateCls()
    if name == "json":
        gate.fit_schema([d.raw for d in dd])
    # clean documents pass with high quality
    clean_q = [gate.evaluate(d).quality for d in dd]
    assert sum(q >= gate.threshold for q in clean_q) >= 0.9 * len(dd)
    # most corrupted documents are flagged (recall), over all operators
    rng = random.Random(0)
    caught = total = 0
    for op in CORRUPTORS[mod]:
        for d in dd:
            c = op(d, rng, "heldout")
            total += 1
            caught += int(not gate.evaluate(c).passed)
    assert caught / total >= 0.6


def test_param_vector_roundtrip():
    g = HTMLGate()
    v = g.get_params()
    assert len(v) == len(g.SIGNALS) + 1
    g.set_params([x + 0.1 for x in v])
    assert abs(g.threshold - (v[-1] + 0.1)) < 1e-6 or g.threshold in (0.05, 0.95)


def test_noisy_and_single_channel_vetoes():
    # a truncated JSON record trips parse_fail alone and must score near zero
    g = JSONGate()
    from autocurate.schema import Document
    doc = Document(id="t", text="", modality="json", raw='{"a": 1, "b": ')
    assert g.evaluate(doc).quality < 0.2
    assert not g.evaluate(doc).passed
