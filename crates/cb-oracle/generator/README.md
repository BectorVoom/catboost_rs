# cb-oracle build-time generator

Python-first oracle generator for the catboost-rs parity harness. Produces the
**frozen** fixtures that the Rust `cb-oracle` crate reads and compares against at
absolute error `<= 1e-5` (INFRA-03/INFRA-04).

## Hard rules (D-12)

- **This generator NEVER runs in CI.** The GitHub Actions CPU lane only *reads*
  the committed `.npy` / `.json` artifacts — it never installs `catboost`/`numpy`
  and never runs these scripts.
- **Fixtures are committed frozen.** Everything under
  `crates/cb-oracle/fixtures/` is tracked in git and reviewed in PRs. The `.venv`
  is gitignored; the fixtures are not.
- **Determinism is mandatory.** All training pins `thread_count=1` and a fixed
  `random_seed` (Pitfall 2 / threat T-01-07). All numeric arrays are written as
  `np.float64` with a dtype assert (Pitfall 3 / threat T-01-08).

## Pinned environment (D-07)

`requirements.txt` pins `catboost==1.2.10` and `numpy==1.26.4` — the exact oracle
baseline. Do not float these versions; a different CatBoost build can shift the
`1e-5` reference.

## (Re)generating the fixtures

```bash
cd crates/cb-oracle/generator
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python gen_inputs.py     # writes the frozen INPUT corpus
.venv/bin/python gen_fixtures.py   # writes the per-stage expected-OUTPUT fixtures
```

Then `git add crates/cb-oracle/fixtures` and commit. Review the diff: a changed
fixture means the oracle baseline moved and must be justified.

## Fixture layout (D-09 / D-10 hybrid `.npy` + `config.json`)

### Frozen INPUT corpus — `fixtures/inputs/<dataset>/`

| dataset               | shape                              | files                                  |
| --------------------- | ---------------------------------- | -------------------------------------- |
| `numeric_tiny`        | 50 x 4, pure numeric               | `X.npy`, `y.npy`, `config.json`        |
| `numeric_categorical` | 50 rows, 3 numeric + 2 categorical | `X.npy`, `cat.npy`, `y.npy`, `config.json` |
| `grouped_ranking`     | 60 rows, 12 groups of 5            | `X.npy`, `group_id.npy`, `y.npy`, `config.json` |

Each `config.json` records the seed, row/column counts, column kinds, and target
type (the metadata half of the hybrid format).

### Per-stage expected-OUTPUT — `fixtures/regression_skeleton/`

Trained from the `numeric_tiny` input with the pinned `CatBoostRegressor` params.

| file                       | stage                | layout                                              |
| -------------------------- | -------------------- | --------------------------------------------------- |
| `borders.npy`              | `Borders`            | flat f64: feature 0 borders, then feature 1, ...    |
| `borders_per_feature.npy`  | `Borders` (split)    | f64 counts; split `borders.npy` back per feature    |
| `model.json`               | `Splits`/`LeafValues`| CatBoost JSON model (`oblivious_trees`, leaf values)|
| `staged.npy`               | `StagedApprox`       | flat f64: stage 0 (n_rows), then stage 1, ...       |
| `predictions.npy`          | `Predictions`        | f64, length n_rows                                  |
| `config.json`              | metadata             | seed, version, thread_count, params, layouts        |

The Rust side loads the `.npy` arrays via `cb_oracle::load_f64_vec` and gates
each stage with `cb_oracle::compare_stage` at `1e-5`.

## Phase-5 per-object oracle — `ordered_oracle.cpp` (transcribe-then-self-oracle)

`ordered_oracle.cpp` is a **standalone, dependency-free transcription** (ZERO
catboost includes) of the four small upstream algorithms the per-object oracle
needs. It mirrors `cityhash_oracle.cpp` exactly: the research-flagged TUs
(`online_ctr.cpp`, the ordered path in `approx_calcer.cpp`, `fold.cpp`
permutation/prefix generation) **cannot be linked in isolation** — they
transitively pull in the whole `TLearnContext` / options / metrics graph
(05-RESEARCH § "Per-Object Oracle Strategy" ESCALATION). So we transcribe the
leaf math verbatim with file:line citations and **self-oracle** it.

### What is transcribed (verbatim, cited in the source)

| Algorithm | Upstream source |
| --------- | --------------- |
| `TFastRng64` (permutation seed) | `util/random/fast.h`, `lcg_engine.h`, `common_ops.h` — identical to the Rust port at `crates/cb-core/src/rng.rs` |
| `Shuffle` (Fisher-Yates) | `util/random/shuffle.h:24-32` (block size 1 for N<1000) |
| Online CTR read-before-increment + `CalcCTR` | `online_ctr.cpp:168-184`/`300-307`, `online_ctr.h:128-131` (online denom is hard `+1`), `online_ctr.cpp:102-111` (`CalcNormalization`) |
| Body/tail prefix | `fold.cpp:35-41` (`SelectMinBatchSize`/`SelectTailSize`) + `156-198` (`BuildDynamicFold`), `fold_len_multiplier` default `2.0` |
| Ordered approx prefix update | `approx_calcer.cpp:566-600` (`UpdateApproxDeltasHistoricallyImpl`) |

### Self-oracle anchors (the transcription is a CROSS-CHECK, not the sole truth)

- **Permutation (D-03 linchpin):** cross-checked against the already-oracle-locked
  `cb-core::TFastRng64` Rust reproduction of the SAME Fisher-Yates draw
  (`rng_test.rs` is bitstream-verified). For `seed=42, N=5` both produce
  `[4, 2, 0, 3, 1]` — the harness MUST agree (verified during 05-01 execution).
  Compared integer-exact via `Stage::Permutation`.
- **Final whole-set CTR counts:** cross-checked against the trained upstream
  model's `ctr_data` `TCtrValueTable` blobs (`model.json` from offline
  `catboost==1.2.10`, parsed by `cb_oracle::ModelJson::ctr_data()`), interpreted
  per `static_ctr_provider.cpp:14-126`.
- **Ordered approx / per-object running CTR:** no direct external dump; anchored
  indirectly via final-prediction parity + internal consistency (identity
  permutation ⇒ prefix == final). Residual risk flagged in 05-RESEARCH.

### Build (offline only — NEVER in CI, D-09)

```bash
g++ -O2 -std=c++17 ordered_oracle.cpp -o ordered_oracle
# stdin: N fold_count fold_len_multiplier prior border_count
#        cat_bin[0..N)  target_class[0..N)  der[0..N)  seed
./ordered_oracle <fixture-dir>   # writes the D-02 .npy stack
```

### Frozen per-object fixtures — `fixtures/{one_hot_cat,plain_ctr,ordered_ctr,ordered_boost,tensor_ctr}/`

Each is a purpose-built categorical fixture (D-08) with a `config.json` pinning
**every** knob explicitly (RESEARCH Pitfall 6: `boosting_type` Plain/Ordered —
never auto, `simple_ctr`/`combinations_ctr` + explicit prior, `one_hot_max_size`,
`max_ctr_complexity`, `permutation_count`/`fold_len_multiplier`,
`counter_calc_method`, `thread_count=1`, `catboost_version=1.2.10`, `seed`) plus
the frozen `.npy` stack:

| dir             | requirement | isolates |
| --------------- | ----------- | -------- |
| `one_hot_cat`   | ORD-04 | one-hot vs CTR boundary: `cat0` cardinality == `one_hot_max_size` (one-hot), `cat1` == `one_hot_max_size + 1` (CTR) |
| `plain_ctr`     | ORD-03 | Plain-mode online CTR (permutation needed because a cat feature exceeds `one_hot_max_size`) |
| `ordered_ctr`   | ORD-03 | Ordered-mode ordered CTR, two per-fold permutations |
| `ordered_boost` | ORD-02 | ordered-boosting approximant body/tail prefix (pure numeric) |
| `tensor_ctr`    | ORD-05 | combination/tensor CTR (`max_ctr_complexity=2`, explicit `combinations_ctr`) |

`.npy` schema per dir (D-02): `permutation_fold{k}.npy [N] int32`,
`ctr_good_count.npy`/`ctr_total_count.npy [N] int32` (exact integers),
`ctr_value.npy [N] f64`, `body_tail_boundaries.npy [*] int32`,
`ordered_approx_iter{t}.npy [N] f64`. **Only the frozen `.npy`/`config.json` are
committed; the generator never runs in CI.**
