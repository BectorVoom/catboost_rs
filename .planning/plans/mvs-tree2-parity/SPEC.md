---
title: "MVS sampler upstream-parity — RNG draw contract and f32 transcription"
status: draft
format: markdown
spec_version: 3
updated_at: 2026-07-31T00:00:00Z
revision_note: >
  v2 (plan-check response, 2026-07-31): specification DEFECTS corrected, no behaviour
  rescoped. (1) MINOR-5 — `algo_helpers/fold.h` does not exist; every upstream citation
  re-read and collected in a new §10.1 "verified citation set", because MVS-S5/S6/S7
  mandate these strings into production doc comments. (2) MINOR-2 — MVS-S7's postcondition
  and AC-7 were literally unsatisfiable (23 of 26 repo-wide hits are this phase's own
  refuted quotations); scoped to exclude `.planning/plans/mvs-tree2-parity/`.
  (3) MAJOR-3 — MVS-S6 had no falsifiable Red: 4 of 5 mirrored call sites are
  bit-identical under the f32 change, so a device-only criterion is satisfiable while the
  transcriptions disagree; a non-device-gated mirror test is now required.
  (4) MAJOR-2 — MVS-S8 gains device/CPU split-argmax disagreement as a second failure
  mode and a pre-defined device-only-residual escalation; AC-8's 3/3 marked projected.
  (5) MAJOR-5 — MVS-S3's pre-fix Red relaxed from an exact 7/10 set to a >=5/10 threshold
  with >=1 per bias setting, because the committed family's own quantization may differ.
  (6) MINOR-1 — a doc-only note at `mvs_device.rs:145-146` is now required, and the gate
  is "no executable change" not "byte-unchanged". (7) MINOR-6/MINOR-7 — the 4 ignored
  tests are stated, and "ALL trees" is bounded to the fixtures' 3 boosting iterations.
  (8) CRITICAL-1/CRITICAL-2 — R5 now carries the MEASURED HEAD baselines of every
  lint/script gate (clippy is red, two of three scripts exit 1), and every gate in this
  phase is differential rather than absolute.
base_commit: 2c14d7f
branch: fix/bootstrap-rng-draw-accounting
source_requirements:
  - "User request 2026-07-30: root-cause and fix the CPU MVS sampler's divergence from upstream CatBoost 1.2.10 so MVS matches on ALL trees at <=1e-5 for BOTH boost_from_average settings, and MVS_GATED_TREES can be raised."
  - "User decision 2026-07-30: include BOTH f32 transcription fixes (sampleSize and SampleWeights narrowing)."
  - "User decision 2026-07-30: add a committed multi-seed MVS fixture family."
source_research: .planning/plans/mvs-tree2-parity/research.md
---

# MVS sampler upstream-parity — RNG draw contract and f32 transcription

## 1. Context

`cb_train::bootstrap::bootstrap`'s `Mvs` arm consumes **three** draws on the persistent
training RNG per tree — one real `rand_seed` plus two fabricated "compensation" draws
at `crates/cb-train/src/bootstrap.rs:413-423`. Instrumented upstream CatBoost 1.2.10
consumes **exactly one**
`[VERIFIED: LOCAL research.md §1.7, §2.4; instrumented trace §4.2]`.

The cited justification in that code block is wrong on its own source: with
`performRandomChoice == false`, `TCalcScoreFold::Sample`
(`calc_score_cache.cpp:730-748`) takes the `SetControlNoZeroWeighted` branch and never
touches `rand`; `CalcWeightedData` (`tensor_search_helpers.cpp:442-485`) is draw-free
`[VERIFIED: LOCAL research.md §1.7, §2.5; line ranges re-read in the upstream tree
during the plan-check revision — `Sample` is `:730-…` with the branch at `:742-748`,
`SetControlNoZeroWeighted` is `:1196-1204` with the mask at `:1202`, `CalcWeightedData`
is `:442-485` and `Bootstrap` begins at `:487`]`.

The consequence is a **+2-draws-per-tree phase drift**. Tree 0 samples correctly (its
`rand_seed` lands at call-count 2 on both sides); from tree 1 onward the sampler draws a
different 80 % subset, and a wrong subset eventually flips a split argmax. Because
flipping requires the wrong subset to actually change an argmax, the failure is
seed- and bias-dependent, which is exactly the observed pattern.

### 1.1 A correction to the originating bug report

The originating report stated the divergence begins at **tree 2, never trees 0 or 1**.
That is **false as a general claim**: with `boost_from_average=true, seed=4` the first
divergent split is flat index **3** — tree 1, split 1. The measured first-bad-split set
across the 7 failing configurations is `{3, 4, 4, 4, 5, 5, 4}`
`[VERIFIED: LOCAL research.md §8.2]`. Drift begins at tree 1, and the tree-2 appearance
was an artifact of the one configuration first examined. Any specification, test name, or
comment asserting a tree-2-specific mechanism is superseded and must be rewritten rather
than merely re-tuned.

### 1.2 Why the existing oracle did not catch it

`crates/cb-oracle/fixtures/bootstrap/mvs` is a single configuration —
`random_seed=0`, `boost_from_average=True` — and it is one of the 3-of-5 configurations
that pass **despite** the defect. Seven of ten seed/bias configurations were silently
broken while the committed oracle stayed green
`[VERIFIED: LOCAL research.md §8.2]`.

### 1.3 Proof that the fix is a deletion

Deleting `bootstrap.rs:413-423` (no algorithmic change) was spiked and reverted
`[VERIFIED: LOCAL research.md §0, §8.2, §8.3]`:

| configuration | before | after |
|---|---|---|
| `boost_from_average=true`, seeds 0–4, 3 trees, ≤1e-5 | 3 / 5 | **5 / 5** |
| `boost_from_average=false`, seeds 0–4, 3 trees, ≤1e-5 | 0 / 5 | **5 / 5** |
| frozen `bootstrap/mvs` oracle | pass | pass |
| `bootstrap_dev/mvs` at `MVS_GATED_TREES = 3` | fail @ split 5 | pass |
| whole `cb-train` suite | 503 / 1 known red | 503 / 1 known red |

Post-fix residuals, every configuration: `max|Δleaf| ∈ [5.9e-9, 6.9e-9]`,
`max|Δstaged| ∈ [1.6e-8, 2.4e-8]` — three orders inside the ≤1e-5 bar.

---

## 2. Scope and non-goals

### In scope

- Delete the two fabricated MVS draws and replace the surrounding documentation with the
  trace-verified contract (`MVS-S1`, `MVS-S2`).
- A unit-level MVS **draw-count** contract test, which does not exist today (`MVS-S1`).
- Remove the `MVS_GATED_TREES` carve-out so MVS is gated over all trees like every other
  scenario, and delete the superseded diagnosis comment (`MVS-S2`, `MVS-S7`).
- A committed **multi-seed × bias** MVS upstream fixture family and its oracle
  (`MVS-S3`) — the coverage whose absence allowed the defect to survive.
- Two f32 transcription fixes: the block sample-size target (`MVS-S4`) and the stored
  sample weight (`MVS-S5`), each mirrored into `cb-backend`'s inline CPU copies
  (`MVS-S6`).
- Re-verification that the device path — which shares this exact host sampler — still
  agrees with the CPU path at ≤1e-5 (`MVS-S8`).
- Frozen-fixture invariance (`MVS-S9`).

### Out of scope (explicitly, not silently)

- Wiring the device-**resident** MVS sampler (`DeviceTrainConfig::mvs_lambda` is never
  `Some` from `cb-train`, so `MvsState` / `launch_mvs_weights_resident` are dead on the
  live path) — Design B′, a separate perf phase
  `[VERIFIED: LOCAL research.md §6.2]`.
- `mvs_reg` / upstream's `TMaybe<float> Lambda` override.
- Ordered-boosting MVS (`mvs.cpp:136-168` tail-derivative copy), group-unit sampling,
  pairwise MVS, Poisson on CPU.
- `CalculateMeanGradValue`'s 125-block reduction order (§`MVS-S10` records it as a
  documented deviation; no oracle can observe it — tree-0 λ only).
- Learn-weight multiplication into `SampleWeights` (no weighted-MVS fixture exists).
- Multi-dimension MVS exactness.
- Re-tuning `PRE_TREE_DRAWS`, `POST_TREE_EXTRA_DRAWS`, `replay_grow_draws`, or
  `select_level_perturbed`'s draw shape. All four are independently verified; changing
  any of them re-breaks Bayesian.

### Scope limit behind the phrase "ALL trees"

Every in-scope fixture pins **`iterations = 3`** (`gen_fixtures.py:899` for
`bootstrap_dev`; `bootstrap_oracle_test.rs:66`; the new `mvs_seeds` family in
`MVS-S3`). "MVS matches upstream over ALL trees" therefore means **all 3 boosting
iterations of the committed fixtures** — it is not a proof for arbitrary tree counts
`[VERIFIED: LOCAL — read at HEAD 2c14d7f]`.

What generalises past 3 trees is `MVS-S1`, not the oracles: the per-call draw count is
pinned directly at unit level (including a three-consecutive-call accumulation leg), so
a phase drift at tree `k` for any `k` is impossible while that contract holds. Raising
the fixture iteration count is deliberately NOT in scope — it would require
regenerating a frozen family (`MVS-S9`).

---

## 3. Dependencies

| Dependency | Why it constrains this phase |
|---|---|
| `cb_core::TFastRng64` (`gen_rand`, `gen_rand_real1`, `advance`, `raw_state`, `call_count`) | the only sanctioned draw source; `raw_state`/`call_count` are how a draw contract is asserted |
| `cb_train::bootstrap::{mvs_lambda, mvs_sample_weights, calculate_threshold, single_probability, mean_grad_value}` | the sampler under repair; `calculate_threshold` is upstream-faithful and must NOT be reimplemented |
| `cb_train::bootstrap::last_iter_mean_leaf_value` | the λ carry; already proven correct to ~3e-9 — do not touch |
| `MVS_BLOCK_SIZE = 8192` (`bootstrap.rs`) | upstream `mvs.h:48 BlockSize`; the per-block reseed unit |
| upstream `mvs.cpp` / `mvs.h` / `calc_score_cache.cpp` / `tensor_search_helpers.cpp` at `/home/user/cb_instrumented_build/catboost-src` | source of record; `SampleRate` is `float` (`mvs.h:47`), `SampleWeights` is `TVector<float>` (**`catboost/private/libs/algo/fold.h:217`** — NOT `algo_helpers/`, which has no `fold.h`; re-verified in the upstream tree) |
| instrumented 1.2.10 build + `CB_INSTRUMENT_LOG` | already compiled and reproduces its committed traces byte-for-byte; the draw-count ground truth |
| `crates/cb-oracle` `compare_stage` / `Stage` / `load_model_json` / `load_f64_vec` | the ≤1e-5 comparison harness |
| `crates/cb-oracle/generator/gen_fixtures.py` + `catboost==1.2.10` | the only upstream fixture source; the `--*-only` entrypoint convention protects committed fixtures |
| `crates/cb-backend/src/kernels/mvs_device_test.rs:55-172` | **verbatim inline copies** of the CPU sampler (deliberate, to avoid a `cb-train` dep); any numeric change here must be mirrored or the device self-oracle diverges |
| `crates/cb-train/src/boosting.rs:3262` (device branch) and `:3833` (CPU branch) | both call the SAME `bootstrap()`, so the fix moves device and CPU numbers identically |
| CatBoost quantization borders are **not** stable across configurations | every comparison must use that configuration's own `float_feature_borders()`; a shared-border comparison yields bogus conclusions |

---

## 4. Typed contracts

No public signature changes. `bootstrap` keeps:

```rust
pub fn bootstrap(
    bootstrap_type: EBootstrapType,
    derivatives: &[f64],   // length n; MVS reads the VALUES, other arms only the LENGTH
    subsample: f64,
    bagging_temperature: f32,
    prev_leaf_mean_l2: Option<f64>,
    rng: &mut TFastRng64,
) -> CbResult<BootstrapResult>;
```

`BootstrapResult { sample_weights: Vec<f64>, control: Vec<bool> }` is unchanged —
`MVS-S5` narrows the stored VALUE through `f32` but the container stays `Vec<f64>`, so no
caller and no re-export changes.

Internal, changed:

```rust
// crates/cb-train/src/bootstrap.rs — mvs_sample_weights
// MVS-S4: the per-block threshold-search target must reproduce upstream's FLOAT
// expression `SampleRate * blockSize` (SampleRate is `float`, mvs.h:47).
let sample_size: f64 = f64::from((sample_rate as f32) * block_size as f32);

// MVS-S5: the stored weight must be narrowed like upstream's TVector<float>.
*slot = f64::from(((weight * f64::from(r < probability)) as f32));
```

---

## 5. Specifications

Each specification is sized so one focused test demonstrates it and a failure has one
principal cause.

---

### SPEC MVS-S1: An MVS `bootstrap()` call consumes exactly ONE main-stream draw

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** the `Mvs` arm advances the persistent RNG by a number of
  `gen_rand` calls other than one.
- **Scope:** `crates/cb-train/src/bootstrap.rs`, the `EBootstrapType::Mvs` arm of
  `bootstrap` (currently `:410-434`, including the `if subsample < 1.0 { … }` block at
  `:413-423`).
- **Dependencies:** `cb_core::TFastRng64`; `mvs_sample_weights` (which draws only on its
  own per-block `TFastRng64::from_seed(rand_seed + block_idx)` sub-streams, never the main
  stream).
- **Input type:** `bootstrap(EBootstrapType::Mvs, &[f64] /* len n */, subsample: f64,
  _bagging: f32, prev_leaf_mean_l2: Option<f64>, rng: &mut TFastRng64)`.
- **Output type:** `CbResult<BootstrapResult>` — unchanged.
- **Preconditions:** `subsample < 1.0` (the sampling regime; `subsample >= 1.0`
  short-circuits to the zero-draw identity path and is already covered).
- **Behavior:** Given a fresh `TFastRng64::from_seed(s)`, when `bootstrap` is called with
  `EBootstrapType::Mvs` and `subsample < 1.0`, then the RNG is left in exactly the state a
  probe that consumed exactly one `gen_rand()` would be in.
- **Postconditions:** `rng.call_count()` increases by exactly 1 relative to entry;
  `rng.raw_state()` equals that of `TFastRng64::from_seed(s)` after one `gen_rand()`.
- **Errors:** unchanged (`Poisson` remains the only erroring arm).
- **Side effects:** none beyond the RNG advance.
- **Acceptance examples:**
  - Given `seed = 0`, `n = 1500`, `subsample = 0.8`, when `bootstrap(Mvs, …)` runs, then
    `rng.raw_state()` equals a probe's state after one `gen_rand()`.
  - Given `subsample = 1.0`, then ZERO draws are consumed (the existing contract, which
    must not regress).
  - Given three consecutive `bootstrap(Mvs, …)` calls on one stream, then `call_count()`
    increased by exactly 3.
- **Evidence:** `[VERIFIED: LOCAL research.md §1.7, §2.4, §2.5, §4.2]`; upstream
  `mvs.cpp:174 randSeed = rand->GenRand()`; trace arithmetic
  `tree_rng_pre_gts.cc = 2 → level-0 cc_start = 7 = 1 bootstrap + 4 RSM`.
- **Non-goals:** the per-block sub-stream draw shape (unchanged and already correct);
  `PRE_TREE_DRAWS` / `POST_TREE_EXTRA_DRAWS`.

---

### SPEC MVS-S2: MVS matches upstream over ALL trees on the bias-0 family

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** the MVS sample for trees ≥ 1 is drawn from the wrong RNG
  phase, so a split argmax flips.
- **Scope:** `crates/cb-train/tests/bootstrap_dev_oracle_test.rs` — remove the
  `MVS_GATED_TREES` / `MVS_SCENARIO` carve-out (currently `:118-155`), the `gated_trees`
  parameter of `gate_against_upstream` (`:163-220`), and the
  `.chain(std::iter::once((MVS_SCENARIO, MVS_GATED_TREES)))` at `:234-237` and `:344-347`;
  fold `mvs` back into `SCENARIOS` (`:112-116`).
- **Dependencies:** `MVS-S1` (the fix); `crates/cb-oracle/fixtures/bootstrap_dev/mvs`
  (already committed at `2c14d7f`, already the bias-0 worst case — no fixture work);
  `compare_stage` at ≤1e-5.
- **Input type:** the committed `bootstrap_dev/mvs` fixture + its own
  `float_feature_borders()`.
- **Output type:** test pass/fail.
- **Preconditions:** none beyond a `cpu`-feature build.
- **Behavior:** Given the `bootstrap_dev/mvs` scenario, when the CPU path trains 3 trees,
  then splits, leaf values and staged approximants all agree with upstream to ≤1e-5 over
  **3/3** trees.
- **Postconditions:** no scenario in this test carries a reduced tree range; the
  `gated_trees` concept is gone from the file.
- **Errors:** n/a.
- **Side effects:** none.
- **Acceptance examples:**
  - Before `MVS-S1`, gating MVS over 3 trees fails with
    `StageDiverged { stage: Splits, index: 5, expected: -0.025514747947454453,
    actual: -0.2692405581474304 }`.
  - After `MVS-S1`, all four `bootstrap_dev` scenarios report 3/3 trees within 1e-5.
- **Evidence:** `[VERIFIED: LOCAL research.md §8.1, §8.2]`.
- **Non-goals:** adding scenarios (that is `MVS-S3`).

---

### SPEC MVS-S3: A multi-seed × bias MVS fixture family gates the sampler

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** an MVS regression passes because the only committed
  configuration happens to be argmax-insensitive to the defect.
- **Scope:** `crates/cb-oracle/generator/gen_fixtures.py` — a new generator function plus
  a new targeted `--*-only` entrypoint, writing a NEW fixture root (e.g.
  `crates/cb-oracle/fixtures/mvs_seeds/s{seed}_bfa{0,1}/`); and a new integration test
  under `crates/cb-train/tests/`.
- **Dependencies:** `catboost==1.2.10`; the frozen `inputs/bootstrap_multiblock` dataset
  (loaded, never rewritten); `compare_stage`; **each scenario's own**
  `float_feature_borders()` — borders are not stable across configurations, so a shared
  border set would invalidate the whole family.
- **Input type:** per scenario, `(random_seed, boost_from_average)` over ≥ 5 seeds × both
  bias settings, all other knobs pinned to the `bootstrap_dev` values
  (`subsample = 0.8`, RMSE, depth 2, 3 iterations, `l2_leaf_reg = 3.0`,
  `learning_rate = 0.1`, `random_strength = 0`, `score_function = L2`,
  `leaf_estimation_method = Gradient`, `thread_count = 1`).
- **Output type:** test pass/fail per scenario.
- **Preconditions:** the generator entrypoint must be reachable WITHOUT running `main()`.
- **Behavior:** Given each `(seed, bias)` scenario, when the CPU path trains, then splits,
  leaf values and staged approximants agree with upstream to ≤1e-5 over all 3 trees.
- **Postconditions:** every knob that `catboost`'s raw dict API defaults differently from
  `BoostParams` is pinned EXPLICITLY on both sides — `random_strength = 0` in particular.
  The family must have **demonstrated** discriminating power, measured as an executable
  gate rather than assumed (see the acceptance examples).
- **Errors:** a missing fixture is a hard test failure, never a skip.
- **Side effects:** new committed fixture files under a NEW root.
- **Acceptance examples:**
  - **Binding:** before `MVS-S1`, **≥ 5 of the 10** scenarios fail, including **≥ 1 with
    `boost_from_average = true`** and **≥ 1 with `boost_from_average = false`**, so both
    bias settings are demonstrably gated.
  - **Expected-and-recorded (NOT binding):** 7 of 10 fail —
    `bias=true` seeds 1 and 4 plus `bias=false` seeds 0–4 — with first-divergent-split
    indices `{3, 4, 4, 4, 5, 5, 4}`. These were measured against research §8.7's
    THROWAWAY scratch fixtures `[VERIFIED: LOCAL research.md §8.2]`; the newly committed
    family is quantized by its own CatBoost run, and CatBoost quantization is known to
    be run-to-run nondeterministic in some configurations
    `[VERIFIED: PROJECT memory ctr-model-loading]`, so a different fit — and hence a
    different failing set — is a benign outcome to RECORD, not a blocker. A 10/10
    pre-fix pass, by contrast, fails the specification.
  - After `MVS-S1`, all 10 pass.
  - Running the new generator entrypoint leaves `fixtures/bootstrap/**`,
    `fixtures/bootstrap_dev/**` and `fixtures/inputs/**` byte-unchanged.
- **Evidence:** `[VERIFIED: LOCAL research.md §8.2, §8.7]`; the trap
  `[VERIFIED: PROJECT memory cv-orch01-random-strength-fixture]`.
- **Non-goals:** weighted MVS; other bootstrap types; GPU scenarios.

---

### SPEC MVS-S4: The block threshold target reproduces upstream's `float` expression

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** the threshold-search target is computed in `f64` where
  upstream computes `float SampleRate * ui32 blockSize` in `f32`.
- **Scope:** `crates/cb-train/src/bootstrap.rs`, the `sample_rate * block_size as f64`
  argument to `calculate_threshold` (currently `:312`).
- **Dependencies:** `calculate_threshold` (unchanged); upstream `mvs.h:47`
  (`float SampleRate`), `mvs.cpp` `CalculateThreshold(…, SampleRate * blockSize)`.
- **Input type:** `sample_rate: f64`, `block_size: usize` (`≤ MVS_BLOCK_SIZE = 8192`).
- **Output type:** `f64` — the `sampleSize` target.
- **Preconditions:** `block_size ≤ 8192 < 2^24`, so the `f32` cast of the count is exact
  and the expression is a faithful `float * ui32` transcription.
- **Behavior:** the target equals `f64::from((sample_rate as f32) * block_size as f32)`.
- **Postconditions:** at `block_size = 1500, sample_rate = 0.8` the target is exactly
  `1200.0`, not `1200.0000178813934` (a `+1.788e-5` error today).
- **Errors:** none.
- **Side effects:** none.
- **Acceptance examples:**
  - `(0.8, 1500) → 1200.0` exactly.
  - `(0.8, 8192) → 6553.60009765625` (unchanged — power-of-two scaling was already exact,
    so this case must not move).
  - `(0.8, 3616) → 2892.800048828125`.
- **Evidence:** `[VERIFIED: LOCAL research.md §2.6, §5.3, §8.4 (numpy table)]`; spiked
  green, all oracles + 10/10 seeds, residuals unchanged.
- **Non-goals:** changing `calculate_threshold`'s algorithm.

---

### SPEC MVS-S5: Stored MVS sample weights are narrowed through `f32`

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** the stored weight keeps `f64` precision where upstream
  stores `TVector<float>`.
- **Scope:** `crates/cb-train/src/bootstrap.rs`, the weight store in
  `mvs_sample_weights` — the single line **`:323`**
  (`*slot = weight * f64::from(r < probability);`). `:321` is the CONDITIONAL DRAW and
  `:325-326` the zero arm; neither changes. *(v2 said `:321-326`, which wrongly implied the
  draw and the zero arm were in scope — corrected per C2-10.)*
- **Dependencies:** upstream **`catboost/private/libs/algo/fold.h:217`**
  `TVector<float> SampleWeights;` (narrowed at `mvs.cpp:213`, read back as a
  `const float*` at `tensor_search_helpers.cpp:456`) — this is the citation `MVS-S5`
  mandates into the production doc comment, so the path must be exactly this;
  `single_probability`; `BootstrapResult.sample_weights` (container stays `Vec<f64>`).
- **Input type:** `weight = 1.0 / probability`, `r: f64`, `probability: f64`.
- **Output type:** the stored `f64`, equal to a value representable in `f32`.
- **Preconditions:** none.
- **Behavior:** the stored value is `f64::from((weight * f64::from(r < probability)) as f32)`.
- **Postconditions:** every element of `BootstrapResult.sample_weights` for the MVS arm
  round-trips through `f32` losslessly. The `control` mask
  (`w > f64::from(f32::EPSILON)`) is unchanged and remains correct.
- **Errors:** none.
- **Side effects:** none.
- **Acceptance examples:**
  - Every returned MVS weight `w` satisfies `w == f64::from(w as f32)`.
  - The zero-probability arm still stores exactly `0.0`.
  - Oracle deltas do not regress (spiked: **zero** measurable effect).
- **Evidence:** `[VERIFIED: LOCAL research.md §2.6, §5.4, §8.6]`.
- **Non-goals:** narrowing the Bayesian/Bernoulli arms' weights; changing
  `BootstrapResult`'s type.

---

### SPEC MVS-S6: The `cb-backend` inline CPU transcription stays consistent

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** `MVS-S4`/`MVS-S5` change the CPU sampler's numerics while
  `cb-backend`'s deliberately duplicated CPU reference still models the old ones, so the
  device MVS self-oracle diverges — and only on rocm/cuda, invisibly to the default CI.
- **Scope:** `crates/cb-backend/src/kernels/mvs_device_test.rs` — `cpu_block_threshold`
  (currently `:130-133`, carrying the same `sample_rate * block.len() as f64`) and
  `cpu_mvs_sample` (currently `:138-172`, carrying the same full-`f64` weight store).
- **Dependencies:** `MVS-S4`, `MVS-S5`; `launch_mvs_weights_resident`
  (`crates/cb-backend/src/kernels/mvs_device.rs`); the `cb-backend` → `cb-train`
  dependency ban, which is WHY these are copies rather than imports.
- **Input type / output type:** unchanged test-local helpers.
- **Preconditions:** the mirrored-arithmetic assertions require NO device (they are pure
  host `f64`/`f32` arithmetic); the device-vs-CPU weight comparison requires a rocm or
  cuda build (`cb-backend --lib` device tests).
- **Behavior:** the inline copies compute the same `sampleSize` and store the same
  `f32`-narrowed weight as the post-fix CPU sampler, **and a test that runs on the
  DEFAULT `cpu` feature enforces it**.
- **Postconditions:** the device MVS self-oracle passes on the device backend with the
  same tolerance as before, and the agreement between the two transcriptions is
  protected by a regression test rather than by a prose "keep in sync" note.
- **Errors:** n/a.
- **Side effects:** none.
- **Acceptance examples:**
  - `cargo test -p cb-backend --no-default-features --features rocm --lib mvs` is green.
  - `grep` shows no remaining `sample_rate * block.len() as f64` or un-narrowed weight
    store in `mvs_device_test.rs`.
  - **A non-device-gated test in `mvs_device_test.rs` pins the mirrored target values
    (`(0.8, 1500) → 1200.0`, `(0.8, 8192) → 6553.60009765625`,
    `(0.8, 3616) → 2892.800048828125` — the same three `MVS-S4` pins) and asserts every
    weight `cpu_mvs_sample` returns satisfies `w == f64::from(w as f32)`; it FAILS if
    either mirrored expression regresses.** Without it the mirror has no falsifiable
    Red: measured at plan-check time, **4 of the 5** existing `cpu_block_threshold` call
    sites are BIT-IDENTICAL under the `MVS-S4` change (only `(rate 0.3, n 200)` moves,
    by 1.431e-6 absolute / **2.384e-8 relative**), and the self-oracle's bar is
    `TOL = 1e-4` — four orders looser
    `[VERIFIED: RUN — numpy f32-vs-f64 target table over all five call sites]`. So a
    device-only criterion is satisfiable while the transcriptions silently disagree,
    which is precisely risk `R2`.
- **Evidence:** `[VERIFIED: LOCAL research.md §6.3, R3 in §7; the 5-call-site delta
  table measured during the plan-check revision]`.
- **Non-goals:** eliminating the duplication (the dependency ban makes it deliberate);
  changing `mvs_sample_kernel`'s **executable** behaviour. A **doc-only** comment at
  `crates/cb-backend/src/kernels/mvs_device.rs:145-146` recording that the kernel
  deliberately keeps an `f64` `sample_size` target — a known deviation from upstream's
  `float` expression — and pointing at `bootstrap.rs`'s helper IS required: a future
  Design-B′ implementer reads the kernel, not the sibling test file. The gate is
  therefore "no executable change to `mvs_device.rs`" (`git diff` shows comment lines
  only), not "byte-unchanged".

---

### SPEC MVS-S7: The superseded tree-2 diagnosis is removed from the tree

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** documentation asserts a tree-2-specific mechanism and a
  λ-derived cause that the evidence refutes, misleading the next reader.
- **Scope:** the `MVS_GATED_TREES` doc comment in
  `crates/cb-train/tests/bootstrap_dev_oracle_test.rs` (currently `:121-154`); the MVS
  bullet in `crates/cb-train/src/bootstrap.rs` module docs (currently `:35-39`); the
  deleted block's own comment (`:414-420`); and
  `.planning/plans/device-bootstrap-parity/progress.md` residual **R-1**.
- **Dependencies:** `MVS-S1`, `MVS-S2`.
- **Input type / output type:** documentation only.
- **Preconditions:** none.
- **Behavior:** every surviving statement about this defect says: the cause was **two
  fabricated draws** in the MVS arm producing a per-tree phase drift; drift begins at
  **tree 1**; the observed first-divergent-split set is `{3, 4, 4, 4, 5, 5, 4}`; upstream
  takes exactly one draw, with the trace citation.
- **Postconditions:** no occurrence of the claims "never trees 0 or 1", "the divergence
  enters when tree 2's sample is drawn from that λ", or "Raise this to 3 once the MVS
  tree-2 sampling gap is fixed" remains anywhere in the repo **outside
  `.planning/plans/mvs-tree2-parity/`**, where those exact phrasings appear only as
  explicitly-refuted quotations (in `research.md`, `SPEC.md` §1.1 and the plan files
  that mandate their removal).
  This exclusion is a correction, not a relaxation: as originally written the
  postcondition and `AC-7` were **literally unsatisfiable** — a measured 23 of the 26
  repo-wide hits are inside this phase's own artifacts
  `[VERIFIED: RUN — grep -rn over crates/ and .planning/ at HEAD 2c14d7f: 14 hits in
  plan7.md, 4 in research.md, 2 in plan9.md, 2 in SPEC.md, 1 in progress.md; only 1 in
  crates/ (bootstrap_dev_oracle_test.rs) and 2 in device-bootstrap-parity/progress.md]`.
- **Errors:** n/a.
- **Side effects:** none.
- **Acceptance examples:**
  - `grep -rn "tree-2\|tree 2" crates/ .planning/plans/device-bootstrap-parity/` returns
    no surviving MVS causal claim.
  - The three-phrase grep over `crates/` and `.planning/` returns hits ONLY under
    `.planning/plans/mvs-tree2-parity/`, and each such hit is verified by reading to sit
    in an explicit "this claim is false / superseded" context.
  - `progress.md` R-1 is marked RESOLVED with the real root cause and a pointer to this
    phase.
- **Evidence:** `[VERIFIED: LOCAL research.md §0 finding 3, §5.2, §8.2]`.
- **Non-goals:** rewriting unrelated documentation.

---

### SPEC MVS-S8: Device-vs-CPU MVS parity survives the fix

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** the device branch shares this exact host sampler
  (`boosting.rs:3262`), so the fix moves the device numbers; if any device-side draw
  accounting silently depended on the old 3-draw count, device and CPU would diverge.
- **Second, independent failure mode — device/CPU split-argmax disagreement on the MVS
  sample.** The completed sibling phase measured `split_mismatched_trees = 4/20` on the
  BASE grower — the device *does* pick a different split from the CPU on near-ties
  `[VERIFIED: PROJECT .planning/plans/device-bootstrap-parity/progress.md:162]`. That is
  tolerated by `device_bootstrap_parity_test.rs:411-417` only because it bounds each
  divergent tree's *contribution* to ≤1e-5. But the device arm of
  `bootstrap_dev_oracle_test` compares raw split borders through
  `compare_stage(Stage::Splits, …)`, where a different split is a **hard failure, not a
  small delta**. MVS's heterogeneous keep-probabilities make near-ties more likely than
  on the base grower, so this mode is live, not hypothetical, and it is *not* a draw-count
  problem — the two failure modes are distinguishable by whether the CPU arm of the same
  test still passes.
- **Pre-defined escalation outcome** (so a failure has a planned remedy rather than only
  "escalate"): if the device arm cannot hold 3/3 for MVS while the CPU arm does, the
  accepted resolution is a **device-only documented residual** — a new specification entry
  recording that the device MVS fit diverges from upstream at tree `k` through a
  fixed-point tie-break, with the divergent tree's contribution bounded and measured.
  It is **NEVER** a tolerance loosening, **NEVER** a `replay_grow_draws` /
  `PRE_TREE_DRAWS` / `POST_TREE_EXTRA_DRAWS` change (SPEC `R1`), and **NEVER** a
  re-introduction of a reduced-tree carve-out on the CPU arm.
- **Scope:** verification only — no edit expected in
  `crates/cb-train/tests/device_bootstrap_parity_test.rs`,
  `bootstrap_dev_oracle_test.rs` (device arm), `device_bootstrap_speed_test.rs`,
  `device_oblivious_parity_probe_test.rs`, `device_seam_test.rs`,
  `device_nonsym_fit_test.rs`, `device_region_fit_test.rs`.
- **Dependencies:** `MVS-S1`, `MVS-S4`, `MVS-S5`, `MVS-S6`;
  `cb_train::device_draw_replay::replay_grow_draws` (must NOT change).
- **Input type / output type:** test pass/fail.
- **Preconditions:** a rocm (or cuda) device build.
- **Behavior:** after the fix, device-vs-CPU `max|Δpred|` for MVS stays ≤1e-5, and the
  device arm of `bootstrap_dev_oracle_test` reports 3/3 trees for MVS.
- **Postconditions:** the recorded device-vs-CPU MVS figure is refreshed (was
  `4.703e-11` at `2c14d7f`); no device test regresses.
- **Errors:** n/a.
- **Side effects:** none.
- **Acceptance examples:**
  - `cargo test -p cb-train --no-default-features --features rocm --test
    device_bootstrap_parity_test` is green and the MVS row remains ≤1e-5.
  - The device arm reports `bootstrap_dev/mvs … 3/3 trees`. **This is a PROJECTED
    outcome, not a measured one**: no GPU run was performed during research
    `[VERIFIED: LOCAL research.md §12 MEDIUM #4 — "Not run — no GPU run was performed in
    this session"]`, and it is made binding by the `MVS-S2` carve-out removal well before
    any device task executes. It must be probed as early as the carve-out change itself
    (the device arm's edit is in the same file), so a failure is attributable while the
    change is still isolated.
- **Evidence:** `[VERIFIED: LOCAL research.md §6.2, §8.8]`; the 3/3 device claim itself
  is `[UNVERIFIED — projected; first measured by the ROCm task]`.
- **Non-goals:** changing any device kernel; wiring the device-resident sampler.

---

### SPEC MVS-S9: Frozen fixtures and the shared draw accounting stay invariant

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** the fix is "achieved" by regenerating a frozen fixture or
  by re-tuning shared draw constants, destroying the independent evidence that made this
  root-cause provable.
- **Scope:** `crates/cb-oracle/fixtures/bootstrap/**`,
  `crates/cb-oracle/fixtures/bootstrap_dev/**`,
  `crates/cb-oracle/fixtures/inputs/**`; `PRE_TREE_DRAWS`, `POST_TREE_EXTRA_DRAWS`
  (`crates/cb-train/src/boosting.rs`); `replay_grow_draws`
  (`crates/cb-train/src/device_draw_replay.rs`); `select_level_perturbed`'s draw shape
  (`crates/cb-train/src/tree.rs`).
- **Dependencies:** the whole phase.
- **Input type / output type:** invariant check.
- **Preconditions:** none.
- **Behavior:** those paths are byte-unchanged, and every oracle that depends on the
  shared accounting stays green.
- **Postconditions:** `git status` shows no modification under the three frozen fixture
  roots; `bootstrap_oracle_test` (5 tests, incl. the value-sensitive 3-tree
  `bootstrap_oracle_bayesian`) and `regularization_oracle_test` are green.
- **Errors:** n/a.
- **Side effects:** none.
- **Acceptance examples:**
  - `git status --short crates/cb-oracle/fixtures/bootstrap crates/cb-oracle/fixtures/bootstrap_dev crates/cb-oracle/fixtures/inputs` is empty.
  - `cargo test -p cb-train --no-fail-fast` → 503 passed / 1 failed, the single failure
    being the pre-existing `monotone_non_symmetric_and_region_are_typed_errors`, plus the
    new tests from `MVS-S1`/`MVS-S3`.
- **Evidence:** `[VERIFIED: LOCAL research.md §5.5, §8.3, R2 in §7]`.
- **Non-goals:** n/a.

---

### SPEC MVS-S10: Remaining known deviations are documented, not silently carried

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** a future reader assumes the MVS port is now a complete
  transcription and is surprised by an unmodelled difference.
- **Scope:** doc comments in `crates/cb-train/src/bootstrap.rs` (`mean_grad_value`, the
  module MVS bullet).
- **Dependencies:** none.
- **Input type / output type:** documentation only.
- **Preconditions:** none.
- **Behavior:** each of the following is recorded as a KNOWN, deliberate deviation with
  its reason and its observability: (a) `CalculateMeanGradValue`'s upstream
  `CB_THREAD_LIMIT`-blocked reduction order (ours is a flat ordered `sum_f64`;
  observable only through tree-0 λ, and no oracle currently sees it); (b) `mvs_reg` /
  `TMaybe<float> Lambda` is unmodelled; (c) learn-weight multiplication into
  `SampleWeights` is unimplemented and no weighted-MVS fixture exists; (d) Ordered-boosting
  MVS tail-derivative handling is out of scope.
- **Postconditions:** none of these is presented as parity.
- **Errors:** n/a.
- **Side effects:** none.
- **Acceptance examples:** the four deviations appear in the source docs with the reason
  and the observability claim.
- **Evidence:** `[VERIFIED: LOCAL research.md §1.5, §2.7, §10]`.
- **Non-goals:** implementing any of them.

---

## 6. Acceptance scenarios (phase level)

- **AC-1** An MVS `bootstrap()` call advances the persistent RNG by exactly one
  `gen_rand`, asserted by `raw_state()` against a one-draw probe (`MVS-S1`).
- **AC-2** `bootstrap_dev/mvs` matches upstream over **3/3** trees at ≤1e-5, with no
  reduced-tree carve-out anywhere in the test (`MVS-S2`).
- **AC-3** All 10 `(seed, bias)` scenarios of the new MVS family match upstream at ≤1e-5,
  and — binding — **≥ 5 of them (including ≥ 1 per `boost_from_average` setting)**
  demonstrably fail without the fix; 7/10 with first-bad-split set `{3,4,4,4,5,5,4}` is
  the expected-and-recorded value, not a gate (`MVS-S3`).
- **AC-4** The block threshold target is exactly `1200.0` at `(0.8, 1500)` (`MVS-S4`).
- **AC-5** Every MVS sample weight round-trips through `f32` losslessly (`MVS-S5`).
- **AC-6** The device MVS self-oracle is green on the device backend, **and a
  non-device-gated test enforces that the `cb-backend` inline copies match the CPU
  sampler** (`MVS-S6`).
- **AC-7** No superseded tree-2 causal claim survives in the repo **outside
  `.planning/plans/mvs-tree2-parity/`**, where the phrasings appear only as
  explicitly-refuted quotations (`MVS-S7`).
- **AC-8** Device-vs-CPU MVS parity stays ≤1e-5 and the device arm reports 3/3 trees
  (`MVS-S8`). The 3/3 device figure is a **projected** outcome to be confirmed on the
  ROCm rig; if it cannot hold while the CPU arm does, `MVS-S8`'s pre-defined
  device-only-residual escalation applies.
- **AC-9** The three frozen fixture roots are byte-unchanged; the shared draw constants
  and the replay are untouched; `cb-train` is **503 passed / 1 failed / 4 ignored** plus
  the new tests, and every lint/script gate shows **no NEW entry** relative to its
  recorded HEAD baseline (several are red at HEAD — see `R5`) (`MVS-S9`).
- **AC-10** The four remaining deviations are documented as deviations (`MVS-S10`).

---

## 7. Impact scope

| Area | Classification | Notes |
|---|---|---|
| `cb_train::bootstrap` MVS arm | **local** | no signature change; `bootstrap`/`BootstrapResult`/`EBootstrapType` re-exports untouched |
| CPU boosting (`boosting.rs:3833`) | cross-module | numerics move for MVS fits only |
| DEVICE boosting (`boosting.rs:3262`) | cross-module | shares the SAME host sampler ⇒ device numbers move identically (desired) |
| `cb-backend` inline CPU copies | cross-module | `MVS-S6` mirror required, rocm/cuda-only visibility |
| Python surface (`catboost-rs-py/src/params.rs`) | external/public | `bootstrap_type="MVS"` behaviour changes numerically; no API change |
| Committed fixtures | operational | NEW root only; three existing roots frozen |
| Device-resident MVS sampler | none | dead on the live path (`config.mvs_lambda` never `Some`) |

---

## 8. Compatibility and migration

An MVS fit's output CHANGES for the same seed — that is the point (it becomes
upstream-correct). This is a behavioural change to a training path, not an API change:
no signature, type, re-export or Python parameter moves. Any consumer that pinned golden
values from a `bootstrap_type="MVS"` fit must re-baseline; nothing in this repo does
except the fixtures explicitly covered here. No migration or feature flag is warranted —
carrying both behaviours would mean carrying a known-wrong sampler.

---

## 9. Risks and open questions

| # | risk | mitigation | verified by |
|---|---|---|---|
| R1 | "Fixing" the count by re-tuning `POST_TREE_EXTRA_DRAWS` / the replay instead of deleting the MVS block | re-breaks the value-sensitive `bootstrap_oracle_bayesian` and the YetiRank RNG oracles; the MVS arm is the ONLY wrong count | `MVS-S9` |
| R2 | Editing the sampler without mirroring `mvs_device_test.rs:55-172` | device self-oracle fails on rocm only — invisible on default CI | `MVS-S6` |
| R3 | Comparing across configurations with one shared border set | bogus divergence/agreement; borders shift with unrelated params | `MVS-S3` postcondition |
| R4 | Regenerating a frozen fixture to make a test pass | destroys the independent evidence | `MVS-S9` |
| R5 | Chasing pre-existing reds | recorded as baseline, do not chase; **every** lint/script gate in this phase is DIFFERENTIAL ("no new entry vs the recorded HEAD baseline"), never "clean"/"passes" | research §8.8 + the measured baselines below |

**R5's measured HEAD baselines (`2c14d7f`) — a "clean gate" criterion is unsatisfiable
without editing files this phase freezes, so it must never be written.**
`[all VERIFIED: RUN during the plan-check revision]`

| gate | HEAD result | note |
|---|---|---|
| `cargo test -p cb-train --no-fail-fast` | **503 passed / 1 failed / 4 ignored** | the failure is `monotone_non_symmetric_and_region_are_typed_errors` (`monotone_oracle_test.rs:286`) |
| `cargo clippy -p cb-train --all-targets` | **RED — build aborts** at `crates/cb-oracle/src/model_json.rs:161:17` ("indexing may panic") before `cb-train`'s own targets are reached | clippy lints path dependencies; the command as written is unusable |
| `cargo clippy -p cb-train --all-targets --no-deps --keep-going` | **RED — 100 errors** across **14 pre-existing `cb-train` integration test files** missing the file-level `#![allow(...)]` (`tensor_ctr_oracle_test.rs` 31, `device_seam_test.rs` 22, `yetirank_pairwise_tree_rng_oracle_test.rs` 11, `ordered_ctr_oracle_test.rs` 11, `plain_ctr_oracle_test.rs` 8, `ordered_boost_oracle_test.rs` 8, `permutation_oracle_test.rs` 3, `structure_fold_cycle_oracle_test.rs` 2, `s_order_ctr_bins_oracle_test.rs` 2, `learn_set_shuffle_oracle_test.rs` 2, + 1 each in `tensor_ctr_e2e_oracle_test.rs`, `multilabel_oracle_test.rs`, `multiclass_oracle_test.rs`, `ctr_split_scoring_test.rs`) | **`--keep-going` is mandatory**: without it cargo aborts targets in parallel and the surfaced subset varies run to run, so the baseline is not comparable |
| `cargo clippy -p cb-backend --lib --no-deps --keep-going` | **RED — exactly 4**: `kernels/exact_quantile.rs:178:8`, `kernels/bootstrap_device.rs:230:28`, `cpu_runtime.rs:696:13`, `cpu_runtime.rs:1025:29` | matches research §8.8 |
| `bash scripts/check-source-test-separation.sh` | **exit 0** — `OK: no inline #[cfg(test)] module bodies in production source` | the one script that IS a real absolute gate |
| `bash scripts/check-no-raw-float-sum.sh` | **exit 1** — 15 files / 36 output lines | mostly the script matching `.sum()` inside doc comments that *describe the ban*, plus genuine integer sums (`sizes.iter().sum()` on `Vec<usize>`). Includes `cb-train/src/boosting.rs:1649`, a file `MVS-S9` requires byte-unchanged ⇒ "make this script pass" directly contradicts `MVS-S9` |
| `bash scripts/check-no-anyhow.sh` | **exit 1** — 12 files / 25 output lines | every hit is a doc comment reading "no `anyhow`" |
| R6 | An all-zero-gradient block yields `p = 0` ⇒ all objects dropped (upstream would produce `inf`/`NaN`) | out of scope; do NOT harmonise to upstream UB | no fixture exercises it |

**Open questions (none blocking):**

1. Whether `CalculateMeanGradValue`'s 125-block reduction order is observable on any
   realistic dataset. `[UNVERIFIED]` — documented as a deviation by `MVS-S10` instead.
2. Whether the new MVS family should also cover `subsample` values other than `0.8`.
   Deferred; the 10 `(seed, bias)` scenarios already have proven discriminating power.

---

## 10. Traceability and sources

| Spec | Primary evidence |
|---|---|
| MVS-S1 | research.md §1.7, §2.4, §2.5, §4.2; upstream `mvs.cpp:174` |
| MVS-S2 | research.md §8.1, §8.2 |
| MVS-S3 | research.md §8.2, §8.7; memory `cv-orch01-random-strength-fixture` |
| MVS-S4 | research.md §2.6, §5.3, §8.4; upstream `mvs.h:47`, `mvs.cpp:197-202` |
| MVS-S5 | research.md §2.6, §5.4, §8.6; upstream `catboost/private/libs/algo/fold.h:217`, `mvs.cpp:213` |
| MVS-S6 | research.md §6.3, §7 R3 |
| MVS-S7 | research.md §0, §5.2, §8.2 |
| MVS-S8 | research.md §6.2, §8.8 |
| MVS-S9 | research.md §5.5, §8.3, §7 R2 |
| MVS-S10 | research.md §1.5, §2.7, §10 |

Upstream source of record: `/home/user/cb_instrumented_build/catboost-src`
(tag `v1.2.10`, commit `b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084`), files
`catboost/private/libs/algo/{mvs.cpp,mvs.h,calc_score_cache.cpp,tensor_search_helpers.cpp,fold.h}`.
Instrumented ground truth:
`.planning/plans/bayesian-rng-draw-accounting/instrumented-ground-truth/`.

### 10.1 Verified citation set (the exact strings to transcribe into production docs)

`MVS-S5` and `MVS-S7` mandate these citations into `crates/cb-train/src/bootstrap.rs`
(and `MVS-S6` into `crates/cb-backend/src/kernels/mvs_device_test.rs`), so a wrong path
would be committed into the source. Every entry below was re-read in the upstream tree
during the plan-check revision `[VERIFIED: RUN]`:

| fact | citation |
|---|---|
| the ONE main-stream draw | `mvs.cpp:174` `const ui64 randSeed = rand->GenRand();` |
| `float SampleRate` | `mvs.h:47` |
| `const ui32 BlockSize = 8192` | `mvs.h:48` |
| `TMaybe<float> Lambda` (the `mvs_reg` override, unmodelled) | `mvs.h:49` |
| `CalculateThreshold(..., SampleRate * blockSize)` | `mvs.cpp:197-202` (the product is at `:202`) |
| the CONDITIONAL reweight draw | `mvs.cpp:210-212` (`probability > double-epsilon`, then `prng.GenRandReal1()`) |
| the narrowing store into `TVector<float>` | `mvs.cpp:213` |
| `TVector<float> SampleWeights;` | **`catboost/private/libs/algo/fold.h:217`** — `algo_helpers/fold.h` **does not exist** |
| `performRandomChoice == false` ⇒ draw-free branch | `calc_score_cache.cpp:742-748` (inside `Sample`, which begins at `:730`) |
| `SetControlNoZeroWeighted` | `calc_score_cache.cpp:1196-1204`; the mask line is `:1202` |
| `CalcWeightedData` is draw-free | `tensor_search_helpers.cpp:442-485` |
| `SampleWeights` read as `const float*` | `tensor_search_helpers.cpp:456` (**not** `:457`) |
| learn-weight multiply (unimplemented here) | `tensor_search_helpers.cpp:479-484`, statement at `:482` (**not** `:481-485`) |
| `Bootstrap` entry | `tensor_search_helpers.cpp:487-566` |
| `CB_THREAD_LIMIT = 128` | `catboost/private/libs/options/restrictions.h:59` |

For contrast, the **existing in-repo comment being deleted** cites
`calc_score_cache.cpp:752` and `:1203-1211` — both wrong. Do not carry either forward.

## 11. Specification-store synchronization

TreeFinder MCP is available, but this repository keeps its specification store as plain
`.planning/plans/<slug>/SPEC.md` files (thirteen sibling phases, none registered in
TreeFinder). No TreeFinder document was added, updated, or left stale; this file is the
draft spec of record. `[VERIFIED: LOCAL — `ls .planning/plans/`]`

| Document | Action | State |
|---|---|---|
| `.planning/plans/mvs-tree2-parity/research.md` | added (research agent) | final |
| `.planning/plans/mvs-tree2-parity/SPEC.md` | added | draft |
| TreeFinder corpus | **not synchronized** (by design) | n/a |
