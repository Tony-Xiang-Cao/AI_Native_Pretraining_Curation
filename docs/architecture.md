# AutoCurate Architecture

`autocurate` (v0.1.0, Apache-2.0, Xiang "Tony" Cao) is an AI-native,
self-verifying, self-improving operating loop for pretraining-data curation. The
core engine is **pure standard library**, CPU-only, and deterministic; semantic
adjudication and real model backends arrive only as optional extras
(`[judge]=judgecurate>=0.3.0`, `[llm]=anthropic+openai`, `[viz]=matplotlib`,
`[dev]=pytest+ruff`). The package operationalizes a four-direction data-governance
theory — **traceable sources, assessable quality, graded risk, divisible
responsibility** — into runnable code rather than prose.

## Data flow

A document moves through one cheap→expensive cascade, leaving with a `Decision`
and an append-only `ProvenanceRecord`:

```
crawl shard
   │  Document{id, text, raw, source, modality, meta}
   ▼
extraction gate ───────────────► FILTER  (extraction_defect)   [stage="gate"]
   │  is the *text* faithfully extracted? (reference-free)
   ▼
profile / outlier screen
   │  per-cohort robust-z anomaly signal over heuristic features
   ▼
surface governance score  ┌──► RETAIN  (score ≥ retain_above)  [stage="heuristic"]
   │  cheap content value  ├──► FILTER  (score ≤ filter_below)  [stage="heuristic"]
   ▼                       └──► middle band ↓
cascade route → judge middle ──► RETAIN / REVIEW / FILTER       [stage="judge"]
   │  only the ambiguous band escalates to judgecurate
   ▼
decision + provenance  (ProvenanceLedger: one JSONL line per doc per stage)
```

The pipeline is implemented in `CurationLoop.curate()` (`pipeline.py`). The same
object serves three `mode`s the experiments compare directly: `"cascade"` (the
proposed system — gates + score route, judge only the middle), `"heuristic"`
(middle → REVIEW, no judge call), and `"judge"` (escalate every non-gate-filtered
document). Labels `FILTER=0 / REVIEW=1 / RETAIN=2` are shared with `judgecurate`,
so the two pipelines compose without translation.

## Key interfaces

- **`Gate`** (`extract/base.py`) — a reference-free extraction-quality gate.
  Subclasses declare `SIGNALS` + `DEFAULTS` and implement `_signals(doc)`; the
  base combines defect signals as a **noisy-AND** (independent veto),
  `quality = Π_k (1 − clamp(wₖ·sₖ))`, and `evaluate(doc)` returns a `GateResult`
  with `quality`, `passed = quality ≥ threshold`, and fired `flags`. The
  per-signal weights `w` and the accept `threshold` form the gate's
  hill-climbable **parameter vector** (`get_params`/`set_params`/`bounds`/`clone`).
  `HTMLGate`, `OCRGate`, `JSONGate` ship as `GATES`.
- **`Proposer`** (`OfflineProposer`, `hillclimb/offline.py`) — `propose(gate)` /
  `propose_many(gate, k)` mutate a gate's parameter vector with bounded, seeded
  Gaussian steps on a random coordinate subset. Default, deterministic, runs in CI.
- **Verifier / accept rule** (`verify/harness.py`) — `build_mutation_set` corrupts
  clean docs into a verifiable answer key; `evaluate_gate` scores precision /
  recall / F1 / FPR; `floor_and_upper` brackets held-out vs. train; and
  `accept_candidate(candidate, incumbent, eval_docs, …, guard_docs, delta, eps)`
  is the reward-hack lock: adopt only if the **small-sample t lower bound**
  (`t_{.975,n-1}`) of the held-out F1 gain on *disjoint* eval docs clears `delta`
  **and** clean-guard FPR does not rise by more than `eps`.
- **`Routine`** (`CrawlMonitorRoutine`, `agentloop/routine.py`) — owns *logic*;
  one `tick(harness, day_docs, …)` is a full Observe→Orient→Decide→Act pass that
  curates a shard, updates drift state, and emits ledger records.
- **`AgentHarness`** (Protocol) — owns *scheduling and side-effects*:
  `schedule(cron, routine_id)`, `run_step(prompt, headless)`, `report(result)`,
  `now()`. Adapters `LocalCronHarness` (zero-dep crontab + JSONL + stdout) and
  `AgentCliHarness` (cloud routines / headless agent CLI) leave the `Routine`
  unchanged.

## Governance directions → modules → artifacts

| Direction | Module | Artifact |
|---|---|---|
| Traceable sources | `agentloop/ledger.py` | `ProvenanceRecord` (source + fetch time + extractor version) in append-only JSONL |
| Assessable quality | `profile`, `extract`, `judge` | robust-z profile, gate `quality`, judge decision |
| Graded risk | `judge.py` (+ `judgecurate`) | risk-penalized adjudication of the ambiguous middle |
| Divisible responsibility | `pipeline.py` → ledger | `responsible_stage` names the one stage answerable per decision |

Corrections are appended as new records that `supersede` an earlier one — never
edits — so `ledger.trace(doc_id)` recovers a full, auditable lineage. The
self-improvement modules `verify` (the mutation-oracle harness, the heart) and
`hillclimb` (verifier-gated search) close the loop; `agentloop/spc.py` runs
paired **EWMA + CUSUM** charts (robust MAD baseline) for drift detection. `report`
renders results to Markdown/figures.

## Module tree

```
autocurate/
├── schema.py        Document, Decision, GateResult, QualityProfile,
│                    ProvenanceRecord, AutoCurateConfig, FILTER/REVIEW/RETAIN
├── pipeline.py      CurationLoop — the cascade (cascade/heuristic/judge modes)
├── profile/         heuristics.py (text_features) · outliers.py (CohortOutlierDetector, robust-z)
├── extract/         base.py (Gate, noisy-AND) · html_gate · ocr_gate · json_gate (GATES)
├── judge.py         JudgeAdapter — thin adapter reusing judgecurate for the middle
├── verify/          corruptions.py (build_mutation_set) · harness.py (evaluate_gate,
│                    floor_and_upper, verified_objective, accept_candidate)  ← the heart
├── hillclimb/       base.py (verified vs naive regimes) · offline.py (OfflineProposer) · agentic.py
├── agentloop/       routine.py (Routine + AgentHarness) · ledger.py (ProvenanceLedger) · spc.py (EWMA/CUSUM)
└── report.py / cli.py / metrics.py / utils.py
```

## Cheap → expensive cascade

The ordering is an economic argument: spend compute only where signal is scarce.
The extraction gate and surface score are microsecond-cheap and resolve the
easy ends; the LLM judge — the only expensive tier — sees only the contested
middle. Empirically (`results/*.json`), the hard reference-free gates (HTML+OCR)
reach F1 floor 0.851 / upper 0.977 (a JSON parse oracle is perfect and reported
separately); on routing, router-only macro-F1 is 0.433 and judge-everything
0.653, yet the cascade at **50% judge calls recovers ~69%** of that gain (0.584)
— halving cost for most of the quality. The same discipline guards
self-improvement: an un-guarded recall hill-climb drives the held-out clean-guard
FPR to 0.517 (destroys 51.7% of clean data), while the verifier-gated regime
holds FPR at 0.0.
