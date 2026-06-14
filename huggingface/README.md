---
license: apache-2.0
task_categories:
  - text-classification
  - other
language:
  - en
tags:
  - pretraining-data
  - data-curation
  - ocr-quality
  - html-extraction
  - data-governance
  - synthetic
  - mutation-testing
  - verifiable-rewards
  - benchmark
  - extraction-quality
pretty_name: "autocurate Extraction-Quality Mutation Benchmark"
size_categories:
  - n<1K
---

# autocurate Extraction-Quality Mutation Benchmark

A **mechanism-characterization** synthetic benchmark for *reference-free
extraction-quality gates* — the gates that decide whether a document survived
HTML extraction, OCR, or JSON parsing cleanly enough to belong in a pretraining
corpus. It pairs deterministic **clean** synthetic corpora with parametric
**corruption operators** that inject *known* extraction defects, so a gate can
be scored against a ground-truth oracle (it should keep the clean original and
flag the corrupted copy) rather than against a noisy human label.

> **This is not a real corpus.** It is synthetic data generated reproducibly,
> on CPU, with no network access, by `autocurate.datagen` (clean documents)
> and `autocurate.verify.corruptions` (the mutation oracle). Its purpose is to
> *characterize a mechanism* — how well a given extraction gate separates clean
> main text from extraction garbage — under controlled, verifiable conditions.
> Do not pretrain on it.

- **Package:** [`autocurate`](https://github.com/Tony-Xiang-Cao/AI_Native_Pretraining_Curation) (v0.1.0)
- **Author:** Xiang "Tony" Cao &lt;caoxiang828@gmail.com&gt;
- **License:** Apache-2.0
- **Core:** pure Python standard library, CPU-only, fully deterministic/seedable

## Why this exists

`autocurate` operationalizes a data-**governance** theory — four directions:
*traceable sources*, *assessable quality*, *graded risk*, and *divisible
responsibility* — into an **AI-native, self-verifying, self-improving** curation
operating loop. A curation loop that improves itself needs a *verifier* it
cannot cheat. This dataset is the substrate for that verifier: a mutation oracle
where every defect is logged, so an extraction gate's improvement can be
**rewarded only for genuine detection**, never for memorizing the test.

The modules around it: `profile` (heuristics + robust-z outliers), `extract`
(the reference-free HTML/OCR/JSON gates under test), `judge` (reuses the sibling
[`judgecurate`](https://pypi.org/project/judgecurate/) for the cascade middle),
`verify` (this **mutation-oracle harness** — the heart of the system),
`hillclimb` (verifier-gated self-improvement), `agentloop` (a harness-agnostic
`Routine` + `AgentHarness` + provenance ledger + EWMA/CUSUM drift control), and
`report`.

## What is in the data

### Clean corpora (`autocurate.datagen`)
Deterministically generated clean prose that a *good* extractor would have
produced: natural function-word density (stop-word fraction ~0.3-0.4),
vowel-rich content tokens, terminal punctuation, and benign imperfections that
real clean main text legitimately contains (a heading, a short bullet list, an
inline figure number). These benign imperfections depress the reference-free
quality estimate *mildly* — never to defect levels — giving a realistic
clean-quality spread and a genuine precision/recall trade-off, so a gate cannot
score well by being trivially strict. Clean documents are emitted across source
cohorts (`web`, `encyclopedia`, `academic`, `forum`) and three extraction
modalities:

- **HTML** — clean text wrapped in well-formed `<html><body><article>` markup.
- **OCR** — clean plain text (the "perfect scan").
- **JSON** — clean text inside a well-formed record (`id`, `url`, `text`,
  `lang`, `tokens`, `title`).

### Corruption operators (`autocurate.verify.corruptions`)
Each operator is a deterministic, seedable function that damages a clean
document and logs a `DefectRecord` (`doc_id`, `modality`, `operator`, `slice`):

| Modality | Operators |
|---|---|
| **HTML** | boilerplate/navigation-chrome injection, raw tag injection, `<script>`/`<style>` leakage, ad/link dumps |
| **OCR**  | character-confusion substitution (`rn`↔`m`, `l`↔`1`, `O`↔`0`, …), word-break / hyphenation splitting, mojibake (UTF-8-as-Latin-1 fragments), line-noise glyphs |
| **JSON** | truncation, schema break (key delete / retype / null), delimiter break (dropped brace, swapped separator) |

(Content-level `risk`/`contradiction` injectors also exist for the governance
risk/cascade experiments, but the headline benchmark is the three *extraction*
modalities above.)

## The de-circularization guarantee (held-out 25% vocabulary)

The central design choice — inherited as a lesson from the sibling
`judgecurate` work — is that **a gate must never be scored on strings it was
allowed to memorize.** Every finite corruption vocabulary (HTML boilerplate
phrases, OCR confusion pairs, risk/contradiction markers) is deterministically
partitioned by `lexicons.split_vocab`:

```python
HELD_OUT_RATE = 0.25   # 25% held out, fixed split seed → reproducible partition
```

- The **train slice (75%)** is the only vocabulary a gate is permitted to
  pattern-match against.
- The **held-out slice (25%)** is what the oracle injects to measure
  **generalization**.

Scoring on the held-out slice gives the conservative **floor F1**; scoring on
the train slice gives the read-back **upper-bound F1**; the gap between them is
the **memorization** a gate is *not* allowed to be rewarded for. Reporting the
`(floor, upper)` pair makes that gap explicit and keeps the verifiable-reward
loop honest.

## Reproduce

```bash
pip install autocurate                 # core: zero third-party deps
python scripts/run_experiments.py      # regenerates results/*.json
python scripts/make_figures.py
pytest                                 # 29 tests, ~50s
```

Optional extras: `autocurate[judge]` (`judgecurate>=0.3.0`),
`autocurate[llm]` (`anthropic`, `openai`), `autocurate[viz]` (`matplotlib`),
`autocurate[dev]` (`pytest`, `ruff`).

## Committed headline results (`results/*.json`)

- **E1 — extraction-gate oracle.** Macro F1: **floor 0.902 / upper 0.985**.
  Per modality (floor / upper): HTML **0.887 / 0.972**, OCR **0.820 / 0.983**,
  JSON **1.000 / 1.000**. Clean-guard false-positive rates stay low
  (HTML 0.025, OCR/JSON 0.0).
- **E2 — judge cascade.** Heuristic-only macro-F1 **0.433**; judge-only
  **0.653**; at **50% judge calls** the cascade reaches **0.584**, recovering
  ~**69%** of the heuristic→judge gain at half the adjudication cost.
- **E3 — self-improvement safety.** A naive (unverified) hill-climb drives the
  clean-guard FPR to **0.275** — it destroys **27.5%** of clean data chasing
  recall — while the **verifier-gated** climb holds FPR at **0.0**.
- **E4 — drift control.** On an injected distribution shift: detection latency
  **0 days**, **0 false alarms**, quality drops **0.889 → 0.557** at the shift
  and **recovers in 3 days**.
- **E5 — outlier detection.** Robust-z (median/MAD) outlier F1 **0.932** vs
  mean±kσ **0.889**.

## Intended uses & limitations

- **Intended:** benchmarking reference-free extraction/quality gates;
  studying verifier-gated self-improvement; demonstrating de-circularized,
  verifiable evaluation; teaching/ablating data-governance mechanisms.
- **Not intended:** as pretraining material, as a model of real-web text
  distributions, or as evidence about absolute quality of any production
  extractor. It is *synthetic mechanism characterization* — the operators are
  stylized abstractions of real defects, and absolute scores reflect the
  operator design, not field prevalence.

## Citation

```bibtex
@software{cao_autocurate_2025,
  author  = {Cao, Xiang},
  title   = {autocurate: an AI-native, self-verifying, self-improving
             pretraining data curation operating loop},
  version = {0.1.0},
  license = {Apache-2.0},
  url     = {https://github.com/Tony-Xiang-Cao/AI_Native_Pretraining_Curation}
}
```

### Selected references

Quality-filtered pretraining and extraction: Gopher
([arXiv:2112.11446](https://arxiv.org/abs/2112.11446)),
C4 ([1910.10683](https://arxiv.org/abs/1910.10683)),
RefinedWeb ([2306.01116](https://arxiv.org/abs/2306.01116)),
FineWeb ([2406.17557](https://arxiv.org/abs/2406.17557)),
DCLM ([2406.11794](https://arxiv.org/abs/2406.11794)),
Trafilatura (ACL 2021), Bevendorff et al. (SIGIR 2023),
Rerunning-OCR ([2110.01661](https://arxiv.org/abs/2110.01661)).
Data-as-compression / selection theory: compression≈intelligence
([2404.09937](https://arxiv.org/abs/2404.09937)),
Entropy Law ([2407.06645](https://arxiv.org/abs/2407.06645)).
Verifiable self-improvement: STaR
([2203.14465](https://arxiv.org/abs/2203.14465)),
RLVR / Tulu 3 ([2411.15124](https://arxiv.org/abs/2411.15124)),
FunSearch (Nature 2024), AlphaEvolve
([2506.13131](https://arxiv.org/abs/2506.13131)),
reward over-optimization (Gao et al.,
[2210.10760](https://arxiv.org/abs/2210.10760)).
Data validation / drift: TFDV (SIGMOD 2020).
