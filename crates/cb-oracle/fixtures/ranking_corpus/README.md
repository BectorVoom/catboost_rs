# Ranking Corpus (Phase 6.3, LOSS-04 / LOSS-05)

The **shared, frozen ranking dataset** every Wave A–D oracle consumes. Produced
**OFFLINE** by `crates/cb-oracle/generator/gen_ranking_fixtures.py` against
**catboost 1.2.10** (RUN-ONCE / COMMIT, D-08). CI only **reads** the committed
`.npy` artifacts under this directory; the generator is **never** invoked from
CI.

## Corpus shape (FROZEN)

A small deterministic ranking dataset over **12 objects** split into **5
contiguous groups of varied size**:

| Group | `group_id` | size | global object indices | subgroup_id (group-local) |
|-------|-----------|------|-----------------------|---------------------------|
| 0     | 0         | 3    | 0, 1, 2               | 0, 1, 2                   |
| 1     | 1         | 2    | 3, 4                  | 0, 1                      |
| 2     | 2         | 4    | 5, 6, 7, 8            | 0, 1, 2, 3                |
| 3     | 3         | 1    | 9                     | 0                         |
| 4     | 4         | 2    | 10, 11                | 0, 1                      |

`group_id` is **contiguous and unique** (each id appears in exactly one
contiguous run) — the shape upstream `GroupSamples` (`query.h:48-67`) and
`cb-train::build_query_info` assert.

### Explicit pairs (`pairs.npy`, global `(winner_id, loser_id)`)

Every pair's endpoints fall in the **same** group (a cross-group pair is rejected
by the Rust builder). Group 3 is a singleton and has no pairs.

```
(0, 2)  (1, 2)          # group 0
(3, 4)                  # group 1
(5, 8)  (6, 7)  (5, 7)  # group 2
(10, 11)                # group 4
```

### Features & target

- `X.npy` — `(12, 4)` `f64` numeric feature matrix, drawn from a fixed RNG
  (`random_seed = 20260617`).
- `y.npy` — graded relevance target in `{0, 1, 2, 3}`, rank-quantized from the
  features so the model has learnable signal.

## Pinned params (uniform across the whole corpus)

| Param | Value | Why |
|-------|-------|-----|
| `catboost` | **1.2.10** | matches the vendored source + `.venv` |
| `thread_count` | 1 | deterministic summation order (Pitfall 4) |
| `depth` | 2 | small trees, mirrors prior phases |
| `iterations` | 5 | short staged-approx sequence |
| `leaf_estimation_iterations` | 1 | single Newton/Gradient step |
| `boosting_type` | **Plain** | the `*Pairwise` variants force Plain; pinning Plain for the whole corpus keeps fixtures uniform |
| `bootstrap_type` | No | no sampling noise |
| `random_strength` | 0 | no score randomization |
| `learning_rate` | 0.3 | fixed |
| `random_seed` | 20260617 | fixed |

## Per-fixture layout

Each `--loss <name>` invocation writes `ranking_corpus/<name>/`:

| File | Stage | Content |
|------|-------|---------|
| `model.json` | `Stage::Splits` / `Stage::LeafValues` | tree splits + leaf values |
| `staged.npy` | `Stage::StagedApprox` | per-iteration staged approx (flat `f64`) |
| `predictions.npy` | `Stage::Predictions` | final predictions (flat `f64`) |
| `config.json` | — | the exact pinned params (auditable baseline) |

Each `--metric <name>` invocation writes `ranking_corpus/<name>/`:

| File | Content |
|------|---------|
| `metric_value.npy` | per-iteration metric value(s) — the LOSS-05 reference |
| `config.json` | pinned params + `eval_metric` spec |

The shared corpus inputs live under `ranking_corpus/inputs/`
(`X.npy`, `y.npy`, `group_id.npy`, `subgroup_id.npy`, `pairs.npy`, `meta.json`).

## OFFLINE run command

Run by hand on a machine where `.venv` has **catboost 1.2.10** importable, then
**COMMIT** the produced `.npy` files. Never run from CI.

```bash
# Write the frozen corpus inputs:
.venv/bin/python crates/cb-oracle/generator/gen_ranking_fixtures.py --inputs

# One deterministic loss fixture (Plans 02–05 call these per loss):
.venv/bin/python crates/cb-oracle/generator/gen_ranking_fixtures.py --loss QueryRMSE
.venv/bin/python crates/cb-oracle/generator/gen_ranking_fixtures.py --loss QuerySoftMax
.venv/bin/python crates/cb-oracle/generator/gen_ranking_fixtures.py --loss PairLogit
.venv/bin/python crates/cb-oracle/generator/gen_ranking_fixtures.py --loss LambdaMart

# One eval-only metric fixture:
.venv/bin/python crates/cb-oracle/generator/gen_ranking_fixtures.py --metric NDCG
```

> **Randomized losses (YetiRank / StochasticRank) are NOT produced here.** Their
> RNG-stream ground truth has no clean Python-reachable form (D-6.3-02) and comes
> from the separate **instrumented C++** generators in Wave C
> (`yetirank_oracle.cpp` / `stochasticrank_oracle.cpp`), also OFFLINE / frozen.
