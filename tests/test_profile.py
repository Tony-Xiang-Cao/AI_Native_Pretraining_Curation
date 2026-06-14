from autocurate.datagen import clean_corpus
from autocurate.profile import CohortOutlierDetector, profile_corpus, text_features
from autocurate.profile.heuristics import FEATURE_NAMES


def test_features_complete_and_bounded():
    f = text_features("The enzyme catalyzed the reaction. According to the 1998 study, "
                      "the rate increased by forty percent under standard conditions.")
    assert set(f) == set(FEATURE_NAMES)
    for name in ("gzip_ratio", "type_token_ratio", "stopword_fraction", "alpha_fraction"):
        assert 0.0 <= f[name] <= 1.0
    assert f["token_length"] > 0


def test_gzip_ratio_two_sided():
    repetitive = text_features("buy now " * 200)["gzip_ratio"]
    natural = text_features(" ".join(f"word{i}" for i in range(200)))["gzip_ratio"]
    # highly repetitive text compresses far better (lower ratio) than varied text
    assert repetitive < natural


def test_outlier_detector_flags_injected_anomaly():
    docs = clean_corpus(120, seed=3)
    for d in docs:
        d.source = "web"
    docs[5].text = "x"                       # truncated/degenerate
    docs[9].text = "spam " * 300             # extreme repetition
    profs = profile_corpus(docs)
    by_id = {p.doc_id: p for p in profs}
    assert by_id[docs[5].id].is_outlier
    assert by_id[docs[9].id].is_outlier
    # most clean docs are not outliers
    assert sum(p.is_outlier for p in profs) < 0.2 * len(profs)


def test_detector_fit_score_roundtrip():
    docs = clean_corpus(60, seed=1)
    det = CohortOutlierDetector().fit(docs)
    profs = det.score_all(docs)
    assert len(profs) == len(docs)
    assert all(p.outlier_score >= 0 for p in profs)
