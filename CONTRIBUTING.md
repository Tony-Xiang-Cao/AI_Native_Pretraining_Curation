# Contributing to autocurate

Thanks for your interest in improving `autocurate`. This guide covers local
setup and the conventions the project depends on.

## Setup

Install the package in editable mode with the judge and dev extras:

```bash
pip install -e ".[judge,dev]"
```

## Tests

Run the full suite with:

```bash
pytest
```

## Results are the contract

The JSON files under `results/*.json` are not just artifacts — they are the
**contract asserted by the test suite**. Tests load these files and check the
mechanism still produces the documented numbers. If you change the mechanism in
a way that legitimately changes the results, regenerate them:

```bash
python scripts/run_experiments.py
```

Then review the diff to `results/*.json` carefully before committing: a change
here means a change in the paper's headline claims.

## Linting

Lint with [ruff](https://docs.astral.sh/ruff/):

```bash
ruff check .
```

## Pure-stdlib core

The core of `autocurate` must stay **pure-stdlib, CPU-only, and deterministic**.
Anything requiring third-party libraries (LLM clients, matplotlib, the
`judgecurate` cascade, etc.) belongs behind an optional extra (`[judge]`,
`[llm]`, `[viz]`, `[dev]`) and must not be imported by core modules at import
time. Keep the core importable and runnable with nothing but the standard
library.
