"""Reproducible experiment harness (E1–E5).

Everything here is deterministic, CPU-only, and pure-stdlib except E2, which
needs the optional ``judgecurate`` extra for the LLM-as-Judge tier. Each
function returns a plain dict of floats/ints; ``run_all`` assembles them and the
test suite asserts the committed ``results/*.json`` so the numbers cannot drift.

These characterize the *mechanism* on controlled synthetic data with held-out
corruption vocabularies; they are an honest floor, not a real-corpus SOTA claim.
"""

from __future__ import annotations

import os
import random
import re
from typing import Dict

from .agentloop import CrawlMonitorRoutine, LocalCronHarness, ProvenanceLedger, StreamMonitor
from .baselines import fitted_baselines
from .datagen import clean_corpus, load_jsonl, make_clean_text
from .extract import HTMLGate, JSONGate, OCRGate
from .hillclimb import hillclimb
from .metrics import binary_prf, macro_f1, agreement
from .pipeline import CurationLoop
from .profile import CohortOutlierDetector, text_features
from .profile.heuristics import FEATURE_NAMES
from .schema import Document, FILTER, HillclimbConfig, ProfileConfig, RETAIN
from .utils import mean, std
from .verify import evaluate_gate, floor_and_upper, guard_fpr
from .verify.corruptions import corrupt

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")


# --------------------------------------------------------------------------- #
# E1 — extraction-gate F1 via the mutation oracle (floor vs upper bound)
# --------------------------------------------------------------------------- #

def e1_gate_oracle(n_per_modality: int = 80, seed: int = 7, seeds=range(5)) -> Dict:
    docs = clean_corpus(n_per_modality * 4, seed=seed)
    out = {}
    macro_floor = macro_upper = 0.0
    for name, gate, mod in (("html", HTMLGate(), "html"),
                            ("ocr", OCRGate(), "ocr"),
                            ("json", JSONGate(), "json")):
        dd = [d for d in docs if d.modality == mod][:n_per_modality]
        if name == "json":
            gate.fit_schema([d.raw for d in dd])
        floor, upper = floor_and_upper(gate, dd, mod, seeds=seeds)
        fpr = guard_fpr(gate, dd)
        out[name] = {"floor_f1": round(floor, 4), "upper_f1": round(upper, 4),
                     "guard_fpr": round(fpr, 4), "memorization_gap": round(upper - floor, 4)}
        macro_floor += floor
        macro_upper += upper
    out["macro"] = {"floor_f1": round(macro_floor / 3, 4), "upper_f1": round(macro_upper / 3, 4)}
    # The "hard" reference-free gates (HTML, OCR); JSON is a structural parse
    # oracle (floor==upper==1.0) and is reported separately so it does not
    # inflate the headline.
    hard = (out["html"]["floor_f1"] + out["ocr"]["floor_f1"]) / 2
    hard_u = (out["html"]["upper_f1"] + out["ocr"]["upper_f1"]) / 2
    out["hard_macro"] = {"floor_f1": round(hard, 4), "upper_f1": round(hard_u, 4),
                         "note": "HTML+OCR only; JSON excluded as a parse oracle"}
    return out


# --------------------------------------------------------------------------- #
# E2 — cascade efficiency (quality vs judge-call budget)
# --------------------------------------------------------------------------- #

def e2_cascade(fractions=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0),
               operating_point: float = 0.5, judge_backend: str = "heuristic",
               limit: int = None, **judge_kwargs) -> Dict:
    """``judge_backend`` defaults to ``judgecurate``'s deterministic offline judge
    (so the committed numbers reproduce); pass ``"anthropic"`` / ``"openai"`` /
    ``"vllm"`` / ``"ollama"`` (with a key + the ``[llm]`` extra) for a real LLM run,
    optionally ``limit``-ing the document count to control API cost."""
    try:
        from .judge import JudgeAdapter, _HAVE_JC, heuristic_governance_score
    except Exception:
        _HAVE_JC = False
    if not _HAVE_JC:
        return {"skipped": "judgecurate not installed (pip install autocurate[judge])"}

    path = os.path.join(_DATA_DIR, "judge_mini_corpus.jsonl")
    rows = [r for r in load_jsonl(path) if r.get("split") == "test"]
    if limit:
        rows = rows[:limit]
    docs = [Document(id=str(i), text=r["text"], source=r.get("source", "web"), modality="text")
            for i, r in enumerate(rows)]
    gold = [r["gold_label"] for r in rows]
    n = len(docs)

    hs = [heuristic_governance_score(d.text) for d in docs]
    ja = JudgeAdapter(judge_backend, n_rounds=3, **judge_kwargs)
    jlab = [ja.adjudicate(d).label for d in docs]          # judge each once (cache)
    order = sorted(range(n), key=lambda i: hs[i])
    rank = {idx: r for r, idx in enumerate(order)}

    curve = []
    base_f1 = judge_f1 = None
    for e in fractions:
        k = int(round(e * n))
        lo = (n - k) // 2
        hi = lo + k
        esc = set(order[lo:hi])
        preds = [jlab[i] if i in esc else (FILTER if rank[i] < lo else RETAIN) for i in range(n)]
        mf, ag = macro_f1(preds, gold), agreement(preds, gold)
        curve.append({"escalate_frac": e, "judge_calls": len(esc),
                      "macro_f1": round(mf, 4), "agreement": round(ag, 4)})
        if e == 0.0:
            base_f1 = mf
        if e == 1.0:
            judge_f1 = mf

    # recovered fraction of the judge-only gain at the chosen operating point
    op = min(curve, key=lambda c: abs(c["escalate_frac"] - operating_point))
    recovered = ((op["macro_f1"] - base_f1) / (judge_f1 - base_f1)) if judge_f1 != base_f1 else 0.0
    return {
        "n": n,
        "judge_backend": judge_backend,
        "heuristic_only_f1": round(base_f1, 4),
        "judge_only_f1": round(judge_f1, 4),
        "operating_point": {"escalate_frac": op["escalate_frac"],
                            "judge_call_rate": round(op["judge_calls"] / n, 4),
                            "macro_f1": op["macro_f1"],
                            "recovered_gain": round(recovered, 4)},
        "curve": curve,
    }


# --------------------------------------------------------------------------- #
# E3 — verified vs naive hill-climbing (reward-hacking)
# --------------------------------------------------------------------------- #

def _degraded_html() -> HTMLGate:
    g = HTMLGate()
    for k in g.weights:
        g.weights[k] *= 0.5
    g.threshold = 0.5
    return g


def e3_hillclimb(n: int = 60, iterations: int = 30, population: int = 10) -> Dict:
    # disjoint document sets: optimize on one corpus draw, evaluate/guard on another
    climb = [d for d in clean_corpus(n * 4, seed=7) if d.modality == "html"][:n]
    held = [d for d in clean_corpus(n * 4, seed=8) if d.modality == "html"][:n]
    cfg = HillclimbConfig(iterations=iterations, population=population, step=0.12, seed=3)
    out = {"protocol": "optimize on seed-7 docs; eval+guard on disjoint seed-8 docs"}
    for regime in ("naive_recall", "naive_f1", "verified"):
        r = hillclimb(_degraded_html(), climb, "html", regime=regime, config=cfg,
                      eval_docs=held, guard_docs=held)
        a, b = r.trajectory[0], r.trajectory[-1]
        out[regime] = {
            "start": {k: round(a[k], 4) for k in ("recall_train", "f1_heldout", "guard_fpr")},
            "end": {k: round(b[k], 4) for k in ("recall_train", "f1_heldout", "guard_fpr")},
            "trajectory": [{"iter": int(t["iter"]),
                            "recall_train": round(t["recall_train"], 4),
                            "f1_heldout": round(t["f1_heldout"], 4),
                            "guard_fpr": round(t["guard_fpr"], 4)} for t in r.trajectory],
        }
    out["reward_hack_fpr_gap"] = round(
        out["naive_recall"]["end"]["guard_fpr"] - out["verified"]["end"]["guard_fpr"], 4)
    return out


# --------------------------------------------------------------------------- #
# E4 — drift detection + recovery on a simulated daily stream
# --------------------------------------------------------------------------- #

_TAG_RE = re.compile(r"</?[a-zA-Z][^>]{0,80}>")
_URL_RE = re.compile(r"https?://\S+")


def simple_reextract(docs):
    """A 'better extractor' the routine applies on a quality alarm: strip residual
    tags / links / short boilerplate lines (re-extraction)."""
    out = []
    for d in docs:
        t = _TAG_RE.sub(" ", d.text)
        lines = [ln for ln in t.split("\n")
                 if not (0 < len(ln.strip()) < 25 and not ln.strip().endswith((".", "?", "!")))]
        t = " ".join(_URL_RE.sub("", ln) for ln in lines)
        out.append(Document(id=d.id, text=t, source=d.source, modality="html"))
    return out


def _run_drift_stream(stream: int, days: int, shift_day: int, warmup: int,
                      per_day: int, corrupt_frac: float) -> Dict:
    def day_docs(day: int, rng: random.Random):
        base = [d for d in clean_corpus(per_day * 3, seed=100 + day + stream * 1000)
                if d.modality == "html"][:per_day]
        if day < shift_day:
            return base
        return [corrupt(d, "html", rng, "heldout")[0] if (i % int(1 / corrupt_frac)) == 0 else d
                for i, d in enumerate(base)]

    routine = CrawlMonitorRoutine(
        CurationLoop(mode="heuristic"), ProvenanceLedger(), warmup=warmup,
        quality_monitor=StreamMonitor(alpha=0.3, L=3.0, k_sigma=0.5, h_sigma=4.0, watch="down"),
        remediator=simple_reextract)
    h = LocalCronHarness()
    rng = random.Random(7 + stream)

    series, pre_q = [], []
    detect = recover = None
    false_alarms = 0
    shifted_q = None
    for day in range(days):
        r = routine.tick(h, day_docs(day, rng))
        q = r.metrics["quality"]
        series.append({"day": day, "quality": round(q, 4), "alarm": bool(r.alerts)})
        if warmup <= day < shift_day:
            pre_q.append(q)
            if r.alerts:
                false_alarms += 1
        if day == shift_day:
            shifted_q = q
        if day >= shift_day and detect is None and "quality_drift_down" in r.alerts:
            detect = day
        if detect is not None and recover is None and day > detect and "recovered" in r.actions:
            recover = day
    return {
        "detection_latency": None if detect is None else detect - shift_day,
        "false_alarms_pre_shift": false_alarms,
        "pre_shift_quality": round(mean(pre_q), 4) if pre_q else None,
        "shifted_quality": round(shifted_q, 4) if shifted_q is not None else None,
        "recovery_latency": None if (recover is None or detect is None) else recover - detect,
        "series": series,
    }


def e4_drift(days: int = 30, shift_day: int = 15, warmup: int = 10,
             per_day: int = 40, corrupt_frac: float = 0.5, n_streams: int = 8) -> Dict:
    runs = [_run_drift_stream(s, days, shift_day, warmup, per_day, corrupt_frac)
            for s in range(n_streams)]
    lat = [r["detection_latency"] for r in runs if r["detection_latency"] is not None]
    rec = [r["recovery_latency"] for r in runs if r["recovery_latency"] is not None]
    fa = [r["false_alarms_pre_shift"] for r in runs]
    pre = [r["pre_shift_quality"] for r in runs if r["pre_shift_quality"] is not None]
    sh = [r["shifted_quality"] for r in runs if r["shifted_quality"] is not None]
    return {
        "shift_day": shift_day,
        "n_streams": n_streams,
        "detected_streams": len(lat),
        "mean_detection_latency_days": round(mean(lat), 3) if lat else None,
        "max_detection_latency_days": max(lat) if lat else None,
        "false_alarm_rate_per_stream": round(mean(fa), 3),
        "mean_pre_shift_quality": round(mean(pre), 4) if pre else None,
        "mean_shifted_quality": round(mean(sh), 4) if sh else None,
        "mean_recovery_latency_days": round(mean(rec), 3) if rec else None,
        "series": runs[0]["series"],   # one representative stream for the figure
    }


# --------------------------------------------------------------------------- #
# E5 — outlier detection: robust-z (MAD) vs mean ± kσ
# --------------------------------------------------------------------------- #

def _anomaly(rng: random.Random) -> str:
    kind = rng.choice(["short", "repeat", "garble", "long"])
    if kind == "short":
        return "ok."
    if kind == "repeat":
        return "buy now " * 120
    if kind == "garble":
        return " ".join("â€™Ã©" + "".join(rng.choice("qxzwk") for _ in range(8)) for _ in range(60))
    return make_clean_text(rng, n_sentences=120, realistic=False)


def e5_outliers(n: int = 400, anomaly_frac: float = 0.12,
                zs=(3.0, 3.5, 4.0, 4.5), headline_z: float = 3.5, seed: int = 11) -> Dict:
    rng = random.Random(seed)
    docs = clean_corpus(n, seed=5)
    for d in docs:
        d.source = "web"
    labels = [0] * len(docs)
    idx = rng.sample(range(len(docs)), int(anomaly_frac * len(docs)))
    for i in idx:
        docs[i] = Document(id=docs[i].id, text=_anomaly(rng), source="web", modality="text")
        labels[i] = 1
    feats = [text_features(d.text) for d in docs]
    stats = {f: (mean([ff[f] for ff in feats]), std([ff[f] for ff in feats])) for f in FEATURE_NAMES}

    def robust_flags(z):
        det = CohortOutlierDetector(ProfileConfig(z_threshold=z, min_cohort=5)).fit(docs, feats)
        return [1 if p.is_outlier else 0 for p in det.score_all(docs, feats)]

    def nonrobust_flags(z):
        out = []
        for ff in feats:
            hit = 0
            for f in FEATURE_NAMES:
                mu, sd = stats[f]
                if sd > 1e-9 and abs((ff[f] - mu) / sd) >= z:
                    hit = 1
                    break
            out.append(hit)
        return out

    sweep = []
    for z in zs:
        rob = binary_prf(robust_flags(z), labels)
        nonrob = binary_prf(nonrobust_flags(z), labels)
        sweep.append({"z": z, "robust_f1": round(rob["f1"], 4),
                      "nonrobust_f1": round(nonrob["f1"], 4)})
    hz = headline_z
    return {
        "n": len(docs), "anomalies": len(idx), "headline_z": hz,
        "robust_z_mad": {k: round(v, 4) for k, v in binary_prf(robust_flags(hz), labels).items()},
        "mean_k_sigma": {k: round(v, 4) for k, v in binary_prf(nonrobust_flags(hz), labels).items()},
        "z_sweep": sweep,
        "robust_wins_at": [s["z"] for s in sweep if s["robust_f1"] >= s["nonrobust_f1"]],
    }


# --------------------------------------------------------------------------- #
# E6 — drift-triggered self-improvement (the loop adapts to a new defect type)
# --------------------------------------------------------------------------- #

def e6_drift_recovery(n: int = 80, iterations: int = 25, population: int = 12) -> Dict:
    """A parser regression starts leaking raw HTML tags — a defect the deployed
    gate under-weights (it was tuned in a clean-extraction era). The drift monitor
    fires; the loop hill-climbs the gate, *targeting the drifting defect type*, and
    recovers detection. The naive control shows the guard matters even here.
    """
    from .verify.corruptions import html_tag_inject
    climb = [d for d in clean_corpus(n * 4, seed=7) if d.modality == "html"][:n]
    held = [d for d in clean_corpus(n * 4, seed=8) if d.modality == "html"][:n]
    drift_ops = [html_tag_inject]                 # the newly-appearing defect

    blind = HTMLGate()
    blind.weights["markup_leak"] = 0.0            # blind to tag leakage
    before_f1, _ = floor_and_upper(blind, held, "html", seeds=range(5), ops=drift_ops)
    before_fpr = guard_fpr(blind, held)

    cfg = HillclimbConfig(iterations=iterations, population=population, step=0.15, seed=3)
    ver = hillclimb(blind.clone(), climb, "html", regime="verified", config=cfg,
                    eval_docs=held, guard_docs=held, ops=drift_ops)
    nai = hillclimb(blind.clone(), climb, "html", regime="naive_recall", config=cfg,
                    eval_docs=held, guard_docs=held, ops=drift_ops)
    ver_f1, _ = floor_and_upper(ver.gate, held, "html", seeds=range(5), ops=drift_ops)
    nai_f1, _ = floor_and_upper(nai.gate, held, "html", seeds=range(5), ops=drift_ops)

    return {
        "drift_defect": "html_tag_inject (raw markup leaking into extracted text)",
        "blind_channel": "markup_leak",
        "before": {"f1": round(before_f1, 4), "markup_weight": round(blind.weights["markup_leak"], 3),
                   "guard_fpr": round(before_fpr, 4)},
        "verified": {"f1": round(ver_f1, 4), "markup_weight": round(ver.gate.weights["markup_leak"], 3),
                     "guard_fpr": round(guard_fpr(ver.gate, held), 4),
                     "recovered_gain": round(ver_f1 - before_f1, 4)},
        "naive_recall": {"f1": round(nai_f1, 4),
                         "guard_fpr": round(guard_fpr(nai.gate, held), 4)},
    }


# --------------------------------------------------------------------------- #
# E7 — our gates vs external reference-free baselines on the same oracle
# --------------------------------------------------------------------------- #

def e7_baselines(n_per_modality: int = 80, seed: int = 7, seeds=range(5)) -> Dict:
    docs = clean_corpus(n_per_modality * 4, seed=seed)
    out = {}
    for name, gate, mod in (("html", HTMLGate(), "html"),
                            ("ocr", OCRGate(), "ocr"),
                            ("json", JSONGate(), "json")):
        dd = [d for d in docs if d.modality == mod][:n_per_modality]
        if name == "json":
            gate.fit_schema([d.raw for d in dd])
        ours = mean(evaluate_gate(gate, dd, mod, s, "heldout").f1 for s in seeds)
        base = {bn: round(mean(evaluate_gate(bg, dd, mod, s, "heldout").f1 for s in seeds), 4)
                for bn, bg in fitted_baselines(mod, dd).items()}
        out[mod] = {"autocurate_gate": round(ours, 4), "baselines": base,
                    "best_baseline": round(max(base.values()), 4) if base else 0.0}
    return out


def run_all() -> Dict:
    return {
        "e1_gate_oracle": e1_gate_oracle(),
        "e2_cascade": e2_cascade(),
        "e3_hillclimb": e3_hillclimb(),
        "e4_drift": e4_drift(),
        "e5_outliers": e5_outliers(),
        "e6_drift_recovery": e6_drift_recovery(),
        "e7_baselines": e7_baselines(),
    }
