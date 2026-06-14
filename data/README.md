# data/

- **`judge_mini_corpus.jsonl`** — a vendored copy of the `judgecurate`
  `mini-curation-bench` (3000 gold-labelled documents across web / encyclopedia /
  academic / noise sources, with `split`, `gold_label`, `gold_high_value`). Used
  **only** by experiment E2 (cascade efficiency), where the cheap surface tier
  routes and `judgecurate` adjudicates the ambiguous middle band. It is reused
  verbatim from the sibling project so the cascade is measured against the same
  content-quality labels the judge was designed for; see that repo for how it is
  generated and de-circularized.

Everything else the experiments consume — the clean extraction corpora and the
parametric corruptions for E1/E3/E4/E5 — is **generated deterministically at run
time** by `autocurate.datagen` and `autocurate.verify.corruptions` (seeded, pure
stdlib), so there is nothing else to store here.
