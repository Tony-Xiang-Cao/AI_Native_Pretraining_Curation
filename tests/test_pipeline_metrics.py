from autocurate.metrics import agreement, binary_prf, macro_f1
from autocurate.pipeline import CurationLoop
from autocurate.report import build_report, render_markdown
from autocurate.schema import Document, FILTER, RETAIN


def test_macro_f1_perfect_and_chance():
    g = [0, 1, 2, 0, 1, 2]
    assert macro_f1(g, g) == 1.0
    assert agreement(g, g) == 1.0
    assert macro_f1([0, 0, 0, 0, 0, 0], g) < 0.5


def test_binary_prf():
    r = binary_prf([1, 1, 0, 0], [1, 0, 0, 1])
    assert 0 <= r["f1"] <= 1


def test_cascade_filters_spam_keeps_knowledge():
    loop = CurationLoop(mode="heuristic")
    spam = Document(id="s", text="Buy now! Click here for free money. Act now! Buy now!",
                    modality="text")
    good = Document(id="g", modality="text", text=(
        "The enzyme catalyzes the reaction. According to the 1998 study by Smith, "
        "the rate increased by 40 percent under standard laboratory conditions, and "
        "the National Institute confirmed the measurement across three trials."))
    sdec, _ = loop.curate(spam)
    gdec, _ = loop.curate(good)
    assert sdec.label == FILTER
    assert gdec.label in (RETAIN, 1)        # retain or review, never filter


def test_report_builds_and_renders():
    loop = CurationLoop(mode="heuristic")
    docs = [Document(id=str(i), text="The study measured the effect. " * 6, modality="text")
            for i in range(10)]
    decisions, records = loop.curate_batch(docs)
    rep = build_report(decisions, records, judge_calls=loop.judge_calls)
    assert rep["n_documents"] == 10
    assert "responsibility" in rep
    md = render_markdown(rep)
    assert "AutoCurate quality report" in md
