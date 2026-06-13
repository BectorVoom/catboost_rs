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
