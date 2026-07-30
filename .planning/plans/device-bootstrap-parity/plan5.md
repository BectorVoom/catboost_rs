---
plan: 5
task_id: TASK-05
phase: device-bootstrap-parity
status: pending
order: 5
wave: A
hardware: none required (pure host; runs in ordinary `cargo test`)
depends_on: [none]
specifications: [WR01-S7, WR01-S8]
---

# Task 5: Host RNG draw replay + `prev_leaf_mean_l2` carry on the device branch

## Objective

After this task the device branch of `train_inner` advances the persistent training
RNG to **exactly** the position a CPU-grown tree would have left it in, and carries
`prev_leaf_mean_l2` forward so MVS λ is per-tree. Both changes are INERT while
`bootstrap_type = No` (the current gate), so the whole CPU suite must stay
byte-identical — which makes this the cheapest early warning in the phase and the
reason it sits in Wave A.

Research called the observability of `train_inner`'s local `rng` an open question.
**Resolution:** extract the replay into a named, pure, public-in-crate function and
unit-test *that* against the real grower's consumption. No test seam into
`train_inner` is required.

## Specification references

- `WR01-S7` — the device branch replays the CPU grow's RNG draws exactly.
  Primary failure reason: the RNG stream position drifts, so tree ≥ 1 samples differ.
- `WR01-S8` — the device branch carries `prev_leaf_mean_l2` for MVS λ.
  Primary failure reason: MVS λ frozen across trees.

## Context and evidence

The exact per-tree draw shape, read from source this session:

| Where | Draws on the MAIN rng | Evidence |
|---|---|---|
| before `bootstrap()` | `PRE_TREE_DRAWS = 2` × `gen_rand` | `boosting.rs:58-59`, `:3665-3671` |
| `bootstrap()` | type-dependent (Bayesian 1; Bernoulli `n`; MVS 1 + 2 compensation) | `bootstrap.rs:383-439` |
| per level, step 1 | `n_features` × `gen_rand_real1` (RSM, unconditional, incl. border-less features) | `tree.rs:596-614` |
| per level, step 2 | 1 × `gen_rand` (`CalcScores` randSeed) | `tree.rs:1141-1142` |
| per level, step 3 | **`n_features` × `std_normal`** (`SelectBestCandidate` `GetInstance`, ONE per LISTED feature including border-less) | `tree.rs:1189-1218` |
| per level, SetBestScore | **none on the main rng** — a fresh `TFastRng64::from_seed(rand_seed + task_idx).advance(10)` supplies the per-border normals | `tree.rs:1157-1180` |
| after the tree | `POST_TREE_EXTRA_DRAWS = 2` × `gen_rand` | `boosting.rs:68-69`, `:4580-4584` |
| after the tree | `prev_leaf_mean_l2 = Some(last_iter_mean_leaf_value(&leaf_values))` over the **lr-scaled** leaf values | `boosting.rs:4586-4589` |

Two traps:

- `std_normal`'s internal `gen_rand` count is **data-dependent** (Marsaglia polar
  rejection); GROUND_TRUTH.md:60-71 records 8/10/12/14 observed at `n_features = 4`.
  The replay must **call `cb_core::std_normal` literally**, never a formula.
- The per-feature `SetBestScore` normals come from a *reseeded* RNG, not the main
  stream — do NOT replay them.

Ordering: the CPU interleaves these draws with the grow, but the device grow
consumes **zero** draws, so replaying the whole block after the grow lands on the
identical next-tree position `[RESEARCH MEDIUM 19 — this task upgrades it to
VERIFIED via the unit test]`.

The device oblivious fold arm already produces lr-scaled leaf values via
`normalize_leaf_values` (`boosting.rs:3376-3383`), matching what the CPU arm feeds
into `last_iter_mean_leaf_value`.

## Files

- Create: `crates/cb-train/src/device_draw_replay.rs`
  — `pub(crate) fn replay_grow_draws(rng: &mut TFastRng64, depth: usize, n_features: usize)`
  with a doc comment reproducing the table above (with file:line), plus
  `#[cfg(test)] #[path = "device_draw_replay_test.rs"] mod tests;` as the LAST item.
- Create: `crates/cb-train/src/device_draw_replay_test.rs`
- Modify: `crates/cb-train/src/lib.rs` — `mod device_draw_replay;` (private module;
  no public re-export needed).
- Modify: `crates/cb-train/src/boosting.rs`
  - device branch: `PRE_TREE_DRAWS` loop before the (TASK-06) `bootstrap()` call —
    land the loop now, guarded by `if draws_active`, which is `false` today.
  - after the oblivious fold, before `continue` (`:3486`):
    `replay_grow_draws(&mut rng, params.depth, matrix.n_features())`,
    `POST_TREE_EXTRA_DRAWS` loop, and the `prev_leaf_mean_l2` assignment — all under
    `if draws_active` except the `prev_leaf_mean_l2` carry, which is harmless and
    should be unconditional (mirroring the CPU arm, which assigns it
    unconditionally at `:4589`).

## TDD sequence

### 1. Red

In `device_draw_replay_test.rs`:

- `replay_matches_real_grow_rng_state` — the load-bearing test. Build a small
  `FeatureMatrix` with `n_features = 4` where **one feature has no borders** (the
  border-less case that still draws). Clone one `TFastRng64::from_seed(S)` into
  `rng_a` and `rng_b`. Run
  `greedy_tensor_search_oblivious_perturbed(..., Some(Perturbation { rng: &mut rng_a, score_st_dev: 0.0 }), ...)`
  and `replay_grow_draws(&mut rng_b, depth, 4)`. Assert
  `rng_a.call_count() == rng_b.call_count()` **and**
  `rng_a.raw_state() == rng_b.raw_state()`
  (`crates/cb-core/src/rng.rs:204` and `:221`). Principal failure reason: the replay
  shape is wrong.
- `replay_is_seed_and_shape_parameterised` — repeat the above over a table of
  `(seed, depth, n_features) ∈ {(1,1,1), (7,3,4), (42,6,16), (2024,2,7)}`. This is
  where a data-dependent `std_normal` count is actually caught (a formula-based
  replay passes one seed and fails another).
- `formula_based_replay_would_fail` — a deliberately wrong local reimplementation
  that advances by a FIXED count per `std_normal` must produce a different
  `raw_state()` for at least one table row. This proves the suite is not vacuous and
  documents the trap in executable form.
- `zero_features_or_zero_depth_is_a_no_op` — `replay_grow_draws(rng, 0, k)` and
  `(rng, k, 0)` leave `raw_state()` unchanged.

- Run: `cargo test -p cb-train --lib device_draw_replay -- --nocapture`

### 2. Green

- Implement `replay_grow_draws` as three literal loops per level, in order:
  `for _ in 0..n_features { rng.gen_rand_real1(); }`, then `rng.gen_rand();`, then
  `for _ in 0..n_features { let _ = cb_core::std_normal(rng); }`.
- Wire the device-branch call sites in `boosting.rs`. Because `draws_active` is
  `false` for every fit that currently reaches the device branch
  (`boosting.rs:3021` — it is `bootstrap_type != No || random_strength != 0`, and
  both are excluded by the gate today), this must be a **no-op** for every existing
  test.
- Run: `cargo test -p cb-train --lib device_draw_replay`, then the full CPU suite.

### 3. Refactor

- Give `replay_grow_draws` a `#[must_use]`-free but explicitly documented contract
  and move the evidence table into the doc comment (it is the spec of the function).
- Add a one-line comment at each `boosting.rs` call site naming WR01-S7 / WR01-S8 so
  a future reader does not "simplify" the block away.
- Run: `cargo test -p cb-train --lib device_draw_replay`.

### 4. Verify

- Run: `cargo test -p cb-train --lib device_draw_replay -- --nocapture`
- Run: `cargo test -p cb-train --test bootstrap_oracle_test` (**blocking gate**)
- Run: `cargo test -p cb-train --test regularization_oracle_test`
- Run: `cargo test -p cb-train --test multidim_sampling_regression_test`
- Run: `cargo test -p cb-train` (whole CPU suite; expect exactly the one known
  pre-existing failure `monotone_non_symmetric_and_region_are_typed_errors`)
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_oblivious_parity_test -- --nocapture`
  (device branch touched → re-confirm the TASK-01 numbers)
- Run: `cargo clippy -p cb-train --lib --no-deps`
- Confirm: no `unwrap`/`expect`/`panic`/raw index in the new production module.

## Implementation steps

1. Write the four unit tests against a not-yet-existing function (they will not
   compile — that is the RED state; make it compile with an empty body first so the
   failure is an assertion failure, not a build error).
2. Implement the three loops.
3. Add the `boosting.rs` call sites.
4. Run the blocking CPU gate and the whole suite.
5. Record in `progress.md`: the `call_count()` values observed per table row, since
   they are the ground truth a later regression would be compared against.

## Completion criteria

- [ ] `raw_state()` equality holds for every `(seed, depth, n_features)` row.
- [ ] The deliberately-wrong formula variant fails at least one row.
- [ ] `bootstrap_oracle_test` and `regularization_oracle_test` green and unchanged.
- [ ] The whole CPU suite shows exactly the one known pre-existing failure.
- [ ] `device_oblivious_parity_test` reports the same numbers as TASK-01.
- [ ] `prev_leaf_mean_l2` is assigned on the device oblivious arm from the
      lr-scaled leaf values.
- [ ] `crates/cb-train/src/device_draw_replay.rs` contains no test body
      (source/test separation).

## Risks and guardrails

- **R3 RNG phase drift** — the whole point of this task; the `raw_state()` assertion
  is stronger than `call_count()` alone and catches an equal-count-but-wrong-order
  replay.
- **R4 `std_normal` treated as constant** — caught by the multi-seed table plus the
  explicit `formula_based_replay_would_fail` test.
- **Accidentally activating draws today** — `draws_active` is false under the
  current gate; if the whole CPU suite shifts by even one value, the guard is wrong.
  Guard: the suite comparison in Verify.
- **`prev_leaf_mean_l2` from UNSCALED leaves** — the device tree returns unscaled
  leaf values; the carry must use the values AFTER `normalize_leaf_values`
  (`boosting.rs:3376-3383`), matching the CPU arm. Guard: assert in TASK-08's MVS
  scenario, and note the ordering explicitly in the call-site comment.
