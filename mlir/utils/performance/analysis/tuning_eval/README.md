# Tuning Evaluation Harness

A standalone, plug-n-play framework for comparing perfConfig **proposers** (set cover, nearest-problem, learned model, random) against the absolute-best TFlops recorded in exhaustive tuning corpora, under a fixed compile-time budget.

The harness is intentionally decoupled from the compiler: proposers communicate via plain problem signatures and perfConfig strings, never MLIR ops. It reuses `quickTuningGen` (loading, problem keys, set-cover, hashing), `tuningRunner` (config-file expansion), and `rocmlir-gen --emit-tuning-space` (the candidate pool), so the harness never drifts from the shipped pipeline.

## What it does

Given a **configs file** (the problem set) and one or more `tuningRunner` `.debug` files (the tuning-db), the harness:

1. Treats the `--configs-file` as the **source of truth** for the problem set:
   it is expanded per arch through the same `tuningRunner` path that generated
   the data, and any expected problem missing from the tuning-db is logged (a
   data-collection gap) and excluded from scoring.
2. Loads the `.debug` tuning-db into an in-memory **oracle**:
   `(arch, op, dtype) -> problem -> {perfConfig: best_tflops}`.
3. **Groups the corpus by arch and processes each arch independently** — one
   model / cover scope per arch, equivalent to invoking the tool once per arch
   (mirrors `quickTuningGen`'s per-`(arch, op)` grouping). Data is never mixed
   across architectures. Per-arch results are written to `<output>/<arch>/`.
4. Within each arch, splits problems into train/test (grouped k-fold, or
   held-out `dtype`) and fits each **proposer** on the train problems.
5. For each test problem, asks the proposer for an ordered list of perfConfigs
   under a **budget** (how many configs you can afford to compile/run).
6. Scores the proposal against the recorded oracle and reports:
   - **regret@k** — `1 - (best achieved in top-k) / (absolute best)`. 0 is perfect.
   - **coverage@k** — fraction of problems where some top-k config is within `threshold` (default 0.93) of the per-problem best.
   - **proposal_coverage** — fraction of proposed configs that actually have a valid measurement (i.e. budget not wasted on inapplicable configs).

### The proposer abstraction

A **`ConfigProposer`** (`proposers/`) is *what* is being tested: given a problem signature and a budget, it emits an ordered list of perfConfigs. Proposals are scored directly against the recorded corpus oracle (`metrics.py`); no GPU is involved.

- `random` — sanity floor every other method must beat.
- `set_cover` — today's quick-tuning behavior (the bar to beat).
- `nearest` — the deployed per-problem lookup, relaxed to the nearest known training problem on a hash miss.
- `model` — offline-trained two-stage **LightGBM** classifiers (applicability + optimality) that rank the candidate pool. The harness fits **one model per `(arch, op)`** by processing each arch's data on its own (the model itself is arch-agnostic; the per-arch grouping happens at the orchestration level, like quickTuningGen).

The op (gemm / conv / attention) is inferred from the `.debug` files (pass `--op` to assert it for a mixed-op corpus).

## Requirements

- Python deps from the repo-root `pip_requirements.txt` (pandas, scikit-learn, pulp, xxhash, lightgbm, ...). The deployment export reads the fitted LightGBM trees directly (`booster.dump_model()`), so no extra inference/codegen dependency is needed.
- A built rocMLIR tree (provides `rocmlir-gen` / `perfRunner` / `tuningRunner`) for the candidate pool (`rocmlir-gen --emit-tuning-space`) and the `--configs-file` expansion.

The package locates `quickTuningGen` and the build dir relative to its own path, so run it from a checkout.

## Quick start

Run as a module (`python -m tuning_eval`) or directly (`python __main__.py`). Globbed tuning-db paths are allowed. `--split` selects how problems are held out.

```bash
cd mlir/utils/performance/analysis

# conv, grouped 3-fold CV
python -m tuning_eval \
    --configs-file configs/tier1-conv-configs \
    --tuning-db 'tier1-conv-*.tsv.debug' \
    --split kfold -o conv_out

# gemm, hold out a dtype to test transfer to an unseen precision
python -m tuning_eval \
    --configs-file configs/tier1-gemm-configs \
    --tuning-db gemm.debug \
    --split dtype --test-dtype f16 -o gemm_out
```

Writes `rows.csv` (per problem/budget), `summary.csv` (pooled), `summary_by_dtype.csv`, and `diagnostics.csv` **per arch** to `<output>/<arch>/`, and prints the pooled and per-dtype tables for each arch.

## Deployment (fit + emit the embeddable tree table)

`python -m tuning_eval` *splits* the data to measure proposers. To produce models for deployment, `python -m tuning_eval.train` groups the tuning-db by arch and fits **one `model` per arch** on the **entire** tuning-db (no held-out test set), then emits each one's decision trees as an embeddable `.inc` of plain tree *data* — flattened node arrays, mirroring `QuickTuningDb`'s per-key `.inc` files. The build glues every model's `.inc` into one table that a single generic evaluator (`SmartTuningDb.cpp`) walks; there is no link against LightGBM/Treelite at deploy time and many `(arch, op)` models embed into one library with no symbol collisions. Training is always from scratch on all the data given, and each arch is processed independently (data is never mixed across archs).

For each arch's fitted stages, the exporter writes under `--output` (default `<repo>/mlir/lib/Dialect/Rock/Tuning/Models`, mirroring `quickTuningGen.py`):

```
<Arch><Op>.inc                 embeddable tree table (two-phase include, like QuickTuningDb)
<arch>_<op>_features.txt       input contract: feature order, one per line
```

One directory can hold several `(arch, op)` models side by side. No GPU is required for the fit, but a built `rocmlir-gen` is: features come from `rocmlir-gen --emit-features` (see below). Pass `--mlir-build-dir` or let it auto-discover.

### Consuming the model in the compiler

The C++ side (`mlir/lib/Dialect/Rock/Tuning/SmartTuningDb.cpp`, `SmartTuningFeatures.cpp`) resolves the per-`(arch, op)` model, builds each candidate's feature vector, scores it through the two stages, and returns the top-K. `rocmlir-gen --emit-tuning-space=smart` (and `rocmlir-tuning-driver -tuning-space smart`) drive it, capped by `ROCMLIR_SMART_TUNING_LIST_MAX` (default 30). There is **no fallback**: requesting `smart` for an `(arch, op)` with no embedded model is a hard error. GEMM, conv (fwd/bwd/wrw share one model), and attention are all wired end-to-end.

Both the smart-tuning ranker and `rocmlir-gen --emit-features` extract features through one shared entry point (`rock::getSmartFeatureExtractor`), so the op→signature→features logic exists in exactly one place — and `features.py` consumes that same C++ output, so there is no second feature implementation to keep in parity.

### `--emit-features`: the single source of truth

`rocmlir-gen --emit-features` prints the C++ feature vector(s) for a kernel as CSV (a header row of feature names — matching the committed `<arch>_<op>_features.txt` — followed by one row per perfConfig, the config quoted in the first column). `features.py` shells out to it (reusing `tuning_space`'s command builders), so training and inference share one feature implementation; the Python side never recomputes features.

```bash
# every config in the exhaustive applicable space
rocmlir-gen --arch gfx942 --operation=gemm -t f16 -m 1024 -k 768 -n 512 --emit-features

# a single config
rocmlir-gen ... -perf_config='v4:128,128,8,...' --emit-features

# batch: one config per line on stdin (one process per problem)
printf 'cfg1\ncfg2\n' | rocmlir-gen ... -perf_config=- --emit-features
```

To match a tuned `.debug` row exactly, pass the row's `-num_cu` / `-num_chiplets` (otherwise the op defaults to the arch minimum/maximum); `tuning_space` does this automatically from the `ProblemSig`.

```bash
# train from scratch and compile to the default model dir
python -m tuning_eval.train --tuning-db 'gemm-*.debug'

# custom location
python -m tuning_eval.train --tuning-db 'gemm-*.debug' -o model_gemm
```

## Splits

Splits are applied **within each arch** (the harness already processes archs independently, so there is no cross-arch split).

| `--split` | Tests generalization to... |
|---|---|
| `kfold` (`--folds N`) | unseen problem **shapes**, same arch/dtype (the production case). Grouped by problem key so a shape never straddles the train/test boundary. |
| `dtype` (`--test-dtype`) | an unseen **dtype** of the same arch. |

## Candidate pool

Proposers that rank a pool (`random`, `model`) draw candidates from the per-problem *applicable* tuning space from `rocmlir-gen --emit-tuning-space=<kind>` (`quick` / `full` / `exhaustive`). This is the single source of truth: configs that compile/run-fail show up as wasted budget, and an expected problem that is missing from the tuning-db is flagged as a data gap. Needs `--mlir-build-dir`.

## Key options

| Flag | Meaning |
|---|---|
| `--configs-file PATH` | Config file defining the problem set (source of truth). Required. |
| `--tuning-db GLOB...` | tuningRunner `.debug` files (globs allowed). Required. |
| `--op` | `gemm` / `conv` / `attention`; inferred if omitted. |
| `--split` | `kfold` / `dtype` (applied within each arch). Refits proposers per fold. Required. |
| `--folds` | Number of CV folds for `kfold` (default 3). |
| `--test-dtype` | Required for the `dtype` split. |
| `--proposers ...` | Subset of `random set_cover nearest model`. |
| `--budgets 1,2,4,...` | Comma-separated budgets to report. |
| `--threshold` | "Good enough" fraction of best (default 0.93). |
| `--emit-tuning-space-kind` | `quick` / `full` / `exhaustive` pool kind (default `exhaustive`). |
| `--mlir-build-dir` | Build dir for the candidate pool. |
| `--no-splitk` | Drop Split-K configs from the corpus. |
| `--seed` | Deterministic splits/subsampling. |
| `-o` / `--output` | Output directory. Eval CSVs are written to a per-arch subdir `<output>/<arch>` (default `.`); for `train` it is the model dir (default `<repo>/mlir/lib/Dialect/Rock/Tuning/Models`). |
| `--n-estimators` / `--max-depth` / `--learning-rate` | Model (LightGBM) hyperparameters. |
| `--no-group-subsample` / `--no-class-weight` | Model training ablations. |
| `--max-train-pairs` | Cap on (problem, config) training rows. |

Run with `--help` for the authoritative list.

## Tests

No GPU required, but a built `rocmlir-gen` is (features are extracted by it). Point the tests at the build via `ROCMLIR_BUILD_DIR` or let it auto-discover:

```bash
cd mlir/utils/performance
ROCMLIR_BUILD_DIR=/path/to/build python -m pytest tests/test_tuning_eval.py -q
```

## Layout

```
tuning_eval/
  __main__.py       eval CLI: python -m tuning_eval (per-arch loop -> <output>/<arch>)
  train.py          deployment CLI: python -m tuning_eval.train (per-arch fit + emit .inc)
  cli.py            shared CLI helpers (corpus loading, model hyperparameter flags)
  export.py         LightGBM trees -> embeddable .inc tree table (consumed by SmartTuningDb.cpp)
  corpus.py         .debug -> in-memory oracle + ProblemSig
  config_specs.py   expand the --configs-file spec per arch; report missing problems
  features.py       feature extraction via rocmlir-gen --emit-features + labeling
  metrics.py        regret@k, coverage@k, the eval loop (scores against the oracle)
  splits.py         train/test splits (k-fold, held-out dtype), within an arch
  tuning_space.py   per-problem candidate pool via rocmlir-gen --emit-tuning-space
  proposers/        random, set_cover, nearest, model (the methods under test)
```
