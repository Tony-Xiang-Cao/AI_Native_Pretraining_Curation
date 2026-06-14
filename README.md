# autocurate — an AI-native, self-verifying, self-improving curation loop

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Zero core deps](https://img.shields.io/badge/core-zero%20dependencies-success.svg)](pyproject.toml)

Pretraining-data curation is usually shipped as a one-shot filter. But a real
crawl runs **every day**: sources drift, an HTML extractor silently regresses and
starts leaking nav bars, an OCR engine mangles a font, a JSONL feed truncates
records — and a static content classifier scores the *content* of a document
whose *extraction* already destroyed it. `autocurate` treats curation as a
standing **operating loop** and turns a four-direction data-governance theory —
*traceable sources, assessable quality, graded risk, divisible responsibility* —
into runnable, **self-verifying, self-improving** code.

It is the sequel to, and reuses, [`judgecurate`](https://github.com/Tony-Xiang-Cao/LLM_as_Judge_Pretraining_Data_Curation)
(LLM-as-Judge) as the semantic adjudicator for the expensive tier of a
cheap→expensive cascade.

<p align="center"><img src="paper/figures/framework.svg" alt="The AutoCurate operating loop" width="840"></p>

## What it does that a static filter can't

- **Gates extraction quality reference-free.** Reference-free HTML / OCR / JSON
  gates estimate whether the text was faithfully *extracted* — boilerplate leak,
  OCR garble, truncated records — the upstream defect class no content classifier
  sees. ([`extract/`](src/autocurate/extract))
- **Verifies every claim against a mutation oracle.** It corrupts clean documents
  with *known* defects and measures recovery, so every gate, threshold, and
  capability is scored **without human labels**. Held-out corruption vocabularies
  force generalization over read-back. ([`verify/`](src/autocurate/verify))
- **Improves its own capabilities — safely.** A verifier-gated hill-climber adopts
  a change only if it improves F1 on *disjoint held-out documents* and doesn't
  regress a clean guard set. We show the failure mode it prevents: an un-guarded
  recall optimizer reward-hacks by **destroying 51.7% of held-out clean data**,
  while the guard gives a hard zero false-positive guarantee.
  ([`hillclimb/`](src/autocurate/hillclimb))
- **Runs as a harness-agnostic daily routine.** One `Routine.tick()`, driven
  identically by a plain cron, a cloud routine, or any agent runner;
  EWMA+CUSUM drift detection on throughput & quality triggers self-improvement on
  regression; an append-only provenance ledger records who decided what.
  ([`agentloop/`](src/autocurate/agentloop))

## The four governance directions → concrete modules

| Direction | Module | Artifact |
|---|---|---|
| **Traceable sources** | `agentloop.ledger` | append-only `ProvenanceRecord` (source, fetch time, extractor + version) |
| **Assessable quality** | `profile`, `extract`, `judge` | heuristic profile + gate quality + cascade decision + quality report |
| **Graded risk** | `judge` (reused risk penalty), `report` | 5-level risk grade, not a binary flag |
| **Divisible responsibility** | `Decision.stage`, ledger | every decision names the accountable stage (heuristic / gate / judge) |

## Install

```bash
git clone https://github.com/Tony-Xiang-Cao/AI_Native_Pretraining_Curation
cd AI_Native_Pretraining_Curation
pip install -e .                # pure-stdlib core, no third-party deps
pip install -e ".[judge]"       # + judgecurate, for the LLM-as-Judge cascade tier
pip install -e ".[llm,dev]"     # + real LLM backends, pytest, ruff
```

The core engine is **pure standard library**; `judgecurate` and the LLM SDKs are
optional extras.

## Quickstart

```bash
# cheap reference-free heuristics for one document
python -m autocurate profile "The enzyme catalyzed the reaction (1998 study, +40%)."

# reference-free extraction-quality score for an HTML extraction
python -m autocurate gate html "Skip to content
Cookie preferences
<div>actual article text here.</div>
Subscribe to our newsletter"

# end-to-end cascade decision
python -m autocurate curate "Buy now! Click here for free money. Act now!"
```

```python
from autocurate import CurationLoop, Document, hillclimb, HTMLGate, evaluate_gate
from autocurate.datagen import clean_corpus

# 1) curate a batch (cheap heuristics; add a judge for the cascade middle)
loop = CurationLoop(mode="heuristic")
docs = [Document(id="1", text="...", source="web", modality="html")]
decisions, ledger = loop.curate_batch(docs)

# 2) verify an extraction gate against the mutation oracle
html = [d for d in clean_corpus(160, seed=7) if d.modality == "html"][:60]
print(evaluate_gate(HTMLGate(), html, "html", which="heldout").f1)   # held-out floor

# 3) let the gate improve itself — safely (verified, guard-protected)
res = hillclimb(HTMLGate(), html, "html", regime="verified")
print(res.final["guard_fpr"])    # stays ~0; the naive regime would blow this up
```

Use a **real LLM judge** for the cascade middle band — any provider, via
`judgecurate`'s interface:

```python
from autocurate.judge import JudgeAdapter
from autocurate.pipeline import CurationLoop
loop = CurationLoop(judge=JudgeAdapter("anthropic"), mode="cascade")   # or "openai", "ollama", "vllm"
```

## Results (reproducible, controlled benchmark)

> **What this is.** Numbers are a mechanism characterization on deterministic,
> CPU-only synthetic corpora with **held-out corruption vocabularies** — an honest
> floor, not a real-corpus SOTA claim. The committed `results/*.json` are asserted
> by the test suite, which fails if a number drifts.

- **E1 — extraction gates (mutation oracle):** the two *hard* reference-free gates
  reach **0.851 held-out-floor F1 / 0.977 upper bound** (HTML 0.887/0.972, OCR
  0.816/0.982); the JSON gate is a structural parse oracle (1.0/1.0, reported
  separately so it doesn't inflate the headline). Clean-guard FPR ≤ 0.025.
- **E2 — cascade efficiency:** a cheap router escalating the uncertain middle to
  `judgecurate`'s *offline heuristic* judge (not an LLM) takes the macro-F1 from
  0.433 (router-only) toward 0.653 (judge-everything), recovering **~69% of the
  gain at half the judge calls** — the value is cost, not a "beats the judge" claim.
- **E3 — reward hacking:** an un-guarded **recall** objective reward-hacks
  catastrophically — **51.7% of held-out clean data destroyed** — while the
  explicit guard delivers a hard **0.0** false-positive guarantee (an un-guarded
  *balanced-F1* objective is empirically safer at 0.017 but carries no guarantee).
  Optimized and evaluated on **disjoint** document draws.
- **E4 — drift:** an injected parser regression is detected with **0-day latency
  across 8 streams** (0.125 false alarms/stream pre-shift); quality 0.885 → 0.582
  on shift, ~5.5-day recovery after re-extraction.
- **E5 — outliers:** robust-z (MAD) beats mean ± kσ at 3 of 4 cutoffs (F1 **0.932
  vs. 0.889** at z = 3.5).
- **E6 — drift-triggered self-improvement:** a parser regression starts leaking raw
  HTML tags; a gate blind to it detects at F1 **0.07**, and the loop's verified
  hill-climb **adapts it to 0.98** (guard FPR 0) — the naive control "recovers" to
  only 0.89 while over-flagging (FPR 0.25).
- **E7 — vs. external baselines:** the gates beat every reference-free prior-art
  baseline on the same oracle (HTML **0.887** vs. gzip 0.19 / Gopher 0.02; OCR
  **0.816** vs. Alex&Burns dict 0.74; JSON **1.0** vs. parse-only 0.82).

<p align="center"><img src="paper/figures/e3_hillclimb.svg" alt="Verified vs naive hill-climbing" width="640"></p>

```bash
python scripts/run_experiments.py     # writes results/*.json (E1-E7, ~45s)
python scripts/make_figures.py        # renders the 8 SVG figures
pytest                                # 31 tests, ~90s; asserts the committed numbers
# optional: rerun E2 against a REAL LLM judge (needs an API key + the [llm] extra)
python scripts/run_llm_e2.py --judge anthropic --limit 150
```

See [`paper/paper.md`](paper/paper.md) for the full write-up and references, and
[`docs/architecture.md`](docs/architecture.md) for the design.

## Repository layout

```
src/autocurate/
  schema.py           shared dataclasses + AutoCurateConfig
  profile/            reference-free heuristics + robust-z (MAD) cohort outliers
  extract/            HTML / OCR / JSON reference-free quality gates (noisy-AND, tunable)
  judge.py            thin adapter reusing judgecurate for the cascade middle
  verify/             mutation oracle: corruptions (held-out vocab) + accept rule  <- the heart
  hillclimb/          verifier-gated self-improvement (offline + agentic proposers)
  agentloop/          harness-agnostic Routine + AgentHarness + ledger + EWMA/CUSUM SPC
  report.py           quality report (markdown + json), by governance direction
  pipeline.py         the cheap->expensive cascade
  baselines.py        external reference-free baselines (gzip / Gopher / Alex&Burns / parse-only)
  datagen.py          deterministic synthetic corpora
  metrics.py · svgplot.py · cli.py · experiments.py (E1-E7)
scripts/              run_experiments, make_figures, run_llm_e2 (optional real LLM)
examples/             quickstart, full_loop, spot_check (real-corpus face validity)
tests/                31 tests, no network; assert the committed results
paper/                paper.md + figures (SVG)
results/              committed deterministic E1-E7 numbers
```

## Limitations

The corpora and corruptions are synthetic and transparent; this exercises and
*verifies the loop*, it does not estimate real-corpus filtering quality. The OCR
gate's dictionary-free garble detector is a deliberately weak stand-in (its
0.166 floor↔upper gap is the honest cost), and the cheap governance score is
minimal by design, so E2 is a lower bound on a stronger router. We train no
language model and measure no downstream accuracy — the loop emits a curated,
provenance-tagged corpus ready for that study. See [paper §10](paper/paper.md).

## License

Apache-2.0 — see [LICENSE](LICENSE).
