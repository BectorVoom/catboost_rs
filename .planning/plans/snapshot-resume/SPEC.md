---
title: "ORCH-03 — Training snapshot / resume (slice 1: float numeric, no-draw)"
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-19T00:00:00Z
phase: 20-orchestration
slice: snapshot-resume
source_requirements:
  - "User: Implement CatBoost's snapshot_file / snapshot_interval / save_snapshot boosting params (training checkpoints; fit() auto-resumes from the last checkpoint if the snapshot file exists)."
  - "Research: Phase-20 three-gap research pass (snapshot/resume flagged the HIGHEST-RISK of the three gaps; a design spike was requested before an approach is locked)."
  - "Sibling precedent (house style): .planning/phases/20-orchestration/calc-metrics/{SPEC,PLAN}.md."
pageindex_pending:
  reason: "No TreeFinder/PageIndex write target confirmed for the catboost-rs planning corpus in-session; the SPEC is authored locally under .planning/plans/ (the effective spec store, mirroring the calc-metrics sibling)."
  intended_identifier: "catboost-rs / .planning/plans/snapshot-resume/SPEC.md"
---

# ORCH-03 — Training Snapshot / Resume (Slice 1)

> Draft specification. NOT approved, accepted, final, or implemented.
> Evidence tags: `[VERIFIED: CODEGRAPH …]`, `[VERIFIED: LOCAL <path>]`,
> `[INFERRED: …]`, `[UNVERIFIED: …]`.
>
> **Risk banner.** This is the highest-risk of the Phase-20 gaps. The training
> surface is a single ~2500-line function whose entire mutable state lives in
> local variables (`crates/cb-train/src/boosting.rs:2259-4757`
> `[VERIFIED: CODEGRAPH/LOCAL]`). This SPEC does NOT attempt full snapshot/resume
> parity. It scopes a deliberately narrow, provably-resumable first slice and
> makes the go/no-go on the approach an explicit **gating verification spike**
> (see §9 and PLAN TASK-01). If that spike returns NO-GO, the feature escalates
> to a standalone design-spike phase and slice 1 does not proceed on this plan.

---

## 1. Context

`catboost_rs` is a Rust rewrite of CatBoost, oracle-tested ≤10⁻⁵. Upstream
CatBoost persists training checkpoints via the boosting params `save_snapshot` /
`snapshot_file` / `snapshot_interval`; a `fit()` whose `snapshot_file` already
exists resumes from the last checkpoint instead of retraining from tree 0. That
surface does **not** exist in `catboost_rs`.

The training entry points are purely functional: `train` (`boosting.rs:1946`) →
`train_with_eval_sets` (`:2048`) → private `train_inner` (`:2259`); `train_cat`
(`:2145`) and `train_ranking` (`:2092`) also route through `train_inner`
`[VERIFIED: CODEGRAPH train/train_with_eval_sets/train_cat/train_ranking/train_inner]`.
There is **no `Trainer` object** holding cross-call state. Every piece of mutable
training state is a local variable inside `train_inner`:

| Local (train_inner) | Line | Carried per tree? | In scope for slice 1? |
|---|---|---|---|
| `approx: Vec<f64>` (dim-major, `approx_dimension*n`) | `:2599`/`:2612` | YES (accumulates each tree) | YES |
| `trees: Vec<ObliviousTree>` | `:2952` | YES (grows) | YES |
| `rng: TFastRng64` (persistent sampling RNG) | `:3011` | YES (advances iff `draws_active`) | YES (captured; unused in the no-draw regime) |
| `non_symmetric_trees` / `region_trees` | `:2955`/`:2959` | grows (grow_policy≠Symmetric / device) | NO (out-of-scope grow policies) |
| `detector`, `best_model`, `eval_approx`, `history` | `:2979`-`:2990` | YES (only with eval sets) | NO (no eval sets) |
| `prev_leaf_mean_l2: Option<f64>` | `:3021` | YES (only MVS bootstrap) | NO (bootstrap=No) |
| `used_features: Vec<bool>` | `:2685` | YES (only with penalties) | NO (no penalties) |
| `group_spans`, `yetirank_seeder`, `stochasticrank_seeder`, `learn_approx` | `:2383`-`:2624` | YES (only ranking) | NO (non-ranking) |
| `ordered_learning_perm` | `:2643` | fixed-once (only Ordered) | NO (Plain) |
| CTR fold columns / `BakedCtrData` | `:2863`-`:4744` | fixed-once / post-loop (only cat) | NO (empty `cat_columns`) |

The design insight that makes slice 1 responsibly plannable **without** the
giant-loop refactor the research feared: in the **float-only, `Plain`,
non-ranking, no-CTR, no-ordered, no-eval, `bootstrap_type=No`,
`random_strength=0`, single-dimension** regime, the loop-carried mutable state
set collapses to exactly **`{approx, trees, rng}`** — every other per-tree
mutable local is gated behind a feature that is out of scope and is provably
inert on this path (the right-hand column above). Confirming that collapse on the
actual code — that NO other per-tree mutable state leaks into the scoped path —
is the sole gating spike (TASK-01).

Two further facts remove the two risks the research flagged as unverified:

1. **No summation-equivalence assumption.** The snapshot serializes `approx`
   **directly** (it is just a `Vec<f64>` of length `n`), exactly as upstream's
   `LearnProgress` persists `AvrgApprox` `[INFERRED: upstream LearnProgress model
   knowledge]`. We do NOT reconstruct `approx` by re-applying a partial model, so
   there is no dependence on the loop-accumulation order matching the predict
   path. (That equivalence is used only as an optional cross-check, never as a
   correctness contract.)
2. **RNG restore is a tiny, isolated cb-core addition.** `raw_state()`
   (`crates/cb-core/src/rng.rs:221`) already exposes the full `[r1.x, r1.c, r2.x,
   r2.c]` state and `call_count()` (`:204`); the only missing piece is a
   `from_raw_state` restore constructor `[VERIFIED: CODEGRAPH rng.rs — no restore
   ctor exists; only `new`/`from_seed`]`. In the slice-1 no-draw regime
   (`draws_active == false`, `boosting.rs:3018`) the RNG is never consumed, so
   exact resume holds even without the restore; the restore is included anyway so
   the snapshot format is correct and forward-compatible.

**Hard architectural constraint (research-corrected).** `cb-train` does **not**
depend on `cb-model` in the build graph — the direction is `cb-model → cb-train`
(`cb-model` is a `cb-train` **dev**-dependency only) `[VERIFIED: LOCAL
crates/cb-train/Cargo.toml:38-45 "cb-model depends on cb-train in the build
graph; dev-deps are exempt"]`. Therefore the snapshot (de)serializer MUST live in
`cb-train` and serialize `cb-train`'s OWN types; it MUST NOT reuse `cb-model`'s
`save_cbm`/`save_json` (`crates/cb-model/src/{cbm,json}.rs`) — that would invert
the dependency edge. **CORRECTION (PLAN-CHECK pass 3): `cb-train` currently
has `serde_json` ONLY as a dev-dependency (`Cargo.toml:51`, under
`[dev-dependencies]`) and no `serde` dependency at all — neither is usable
from production code. PLAN TASK-04 promotes both to normal `[dependencies]`
(mirroring `crates/cb-model/Cargo.toml:37-38`'s already-correct pattern) as
its first, blocking step, before any `snapshot.rs` code is written.**
`[VERIFIED: LOCAL crates/cb-train/Cargo.toml:51 — dev-dependencies section]`.
`cb_train::ObliviousTree` (`boosting.rs:771`) is a DISTINCT type from
`cb_model::ObliviousTree` (`model.rs:254`) `[VERIFIED: LOCAL]`.

---

## 2. Scope and Non-Goals

### In scope (slice 1)

- A `SnapshotConfig` value (upstream `snapshot_file` / `snapshot_interval`;
  `save_snapshot` maps to `Some`/`None`) threaded as an ADDITIVE optional
  parameter into `train_inner`; the four existing callers pass `None` and stay
  byte-identical (D-04 no-regression).
- A `TFastRng64::from_raw_state([u64;4], call_count)` restore constructor in
  `cb-core`, round-trip-oracled bit-for-bit against `gen_rand`.
- A `cb-train`-internal snapshot state model + `serde_json` (de)serialization
  (a dedicated DTO over the public fields of `approx`, `trees`, and the RNG state
  — NOT serde derives on `Model`/`ObliviousTree`, keeping those types untouched,
  D-04). Captures: format version, a compat fingerprint, completed-iteration
  count `K`, `approx`, the `K` partial oblivious trees, and RNG `raw_state` +
  `call_count`.
- A training-config/data **fingerprint** (compat key) written into the snapshot;
  a resume against a mismatched fingerprint is a typed error (never a silently
  wrong model).
- Periodic snapshot **write** inside the loop at a **per-tree** boundary,
  honoring `snapshot_interval` (wall-clock), written atomically (temp + rename).
- **Resume** on `train_inner` entry: if the snapshot file exists and its
  fingerprint matches, load it, pre-populate `trees`/`approx`, restore `rng`, and
  start the loop at `K` (`for iter in K..iterations`).
- A **scope guard**: requesting a snapshot on an out-of-scope regime (ranking /
  CTR / ordered / eval sets / `bootstrap_type != No` / `random_strength != 0` /
  multi-dim / penalties / non-symmetric grow policy / device path / a
  `Loss::Custom(_)` objective or `EvalMetric::Custom(_)` metric) is a typed
  error at entry — no snapshot of an unresumable regime is ever written. (The
  `Custom` exclusions are mandatory: an opaque `Arc<dyn>` objective/metric has no
  cross-process-stable identity, so it can never be soundly fingerprinted — §S4/§S5.)
- A new public Rust entry `train_with_snapshot(...)` (numeric path) wiring the
  write + resume, returning the model plus the resume point (so a test can prove
  a genuine resume, not a silent retrain).
- **Oracle: self-consistency, exact.** Train `N` straight-through vs. train `K`
  with a snapshot then resume the remaining `N-K` — the two resulting models MUST
  be **byte-identical** (serialized snapshot state) and predict-identical (exact;
  the no-draw scalar CPU path is fully deterministic — `sum_f64` reductions, no
  RNG draws, no threaded nondeterminism), NOT merely ≤1e-5.

### Non-goals (explicit — documented, not silently dropped)

- **Ranking resume** (YetiRank / YetiRankPairwise / StochasticRank): `group_spans`
  is mutated in place per tree (`boosting.rs:3522`), and the per-tree
  `YetiRankTreeSeeder` / `stochasticrank_seeder` carry advancing RNG-draw state
  (`:2460`/`:2493`), plus a separate `learn_approx` learning-fold buffer
  (`:2624`). Resuming mid-stream needs all of that captured — a later slice.
- **CTR / categorical resume**: fold permutations and `BakedCtrData` are built
  once / post-loop from the seed (`:2863`, `:4708`); mid-CTR resume is out of
  scope.
- **Ordered boosting resume**: `ordered_learning_perm` + dynamic body/tail folds
  (`:2643`) — out of scope (Plain only).
- **Sampled-training resume** (`bootstrap_type != No`, `random_strength != 0`):
  the RNG IS consumed per tree, and MVS additionally carries `prev_leaf_mean_l2`.
  The format captures RNG state for forward-compat, but the exact oracle and the
  scope guard restrict slice 1 to the no-draw regime.
- **Overfitting detector / `use_best_model` / eval-set history resume**
  (`detector`/`best_model`/`eval_approx`/`history`, `:2979`-`:2990`).
- **Multi-dimensional** (multiclass / multilabel / `RmseWithUncertainty`) resume
  (`approx_dimension > 1`) and **non-symmetric / Region grow policies**.
- **GPU / device grow path** resume (the per-fit device branch, `:3193`).
- **Custom-objective / custom-metric resume** (`Loss::Custom(_)` /
  `EvalMetric::Custom(_)`, `runtime.rs:494` / `metrics.rs:167`): the handle is an
  opaque `Arc<dyn>` trait object whose only equality is process-local `Arc::ptr_eq`
  (`custom.rs:134-142,165-169`) with no `Serialize`/`Deserialize`, so no
  cross-process-stable fingerprint can distinguish two custom instances. Rejected
  at entry by the §S5 scope guard (never written), NOT silently accepted — a
  follow-on if ever supported.
- **Mid-tree checkpoint granularity.** The checkpoint boundary is per-tree ONLY;
  `K` is a completed-tree count.
- **Python surface** (`save_snapshot` / `snapshot_file` on the estimator). A
  follow-on; slice 1 is Rust-only to contain risk.
- **`iterations` change on resume** (upstream allows growing `iterations`). Slice
  1 requires the resumed `iterations` to equal the snapshot's target (part of the
  fingerprint); changing it is a follow-on.

---

## 3. Dependencies

| Dependency | Kind | Evidence |
|-----------|------|----------|
| `cb_train::boosting::train_inner` (the single training loop) | modify (additive optional param + resume/write hooks) | `[VERIFIED: CODEGRAPH boosting.rs:2259; loop :3184; assembly :4746]` |
| `cb_train::boosting::{train, train_with_eval_sets, train_cat, train_ranking}` | modify (pass `None` — byte-identical) | `[VERIFIED: CODEGRAPH callers of train_inner]` |
| `cb_train::BoostParams` (loss/iterations/depth/lr/l2/random_seed/boosting_type/…) | reuse (fingerprint inputs; NOT extended) | `[VERIFIED: CODEGRAPH boosting.rs:165]` |
| `cb_train::Model` + `cb_train::ObliviousTree` (`splits`/`leaf_values`/`leaf_weights`/`ctr_splits`, all `pub`) | reuse (snapshot DTO source; NOT serde-derived) | `[VERIFIED: LOCAL boosting.rs:771,852]` |
| `cb_core::rng::TFastRng64` (`raw_state`, `call_count`, `from_seed`) + new `from_raw_state` | reuse + tiny addition | `[VERIFIED: CODEGRAPH rng.rs:171,204,221 — no restore ctor]` |
| `cb_core::{CbError, CbResult}` | reuse (typed errors: mismatch / out-of-scope / IO) | `[VERIFIED: CODEGRAPH boosting.rs uses CbError::{Degenerate,OutOfRange,…}]` |
| `draws_active` gate (`bootstrap_type != No || random_strength != 0`) | reuse (no-draw invariant + scope guard) | `[VERIFIED: CODEGRAPH boosting.rs:3017-3018]` |
| `serde` / `serde_json` | **NEW normal dependency** (currently `serde_json` is dev-only, `serde` absent — TASK-04 promotes both, see PLAN TASK-04 Cargo fix) | `[VERIFIED: LOCAL crates/cb-train/Cargo.toml:51 dev-dependencies; crates/cb-model/Cargo.toml:37-38 correct-pattern precedent]` |
| `std::fs` + atomic temp-then-rename; `std::time::Instant` (interval) | reuse (write) | `[INFERRED: std]` |
| `cb_oracle` harness (≤1e-5 or exact compare) | reuse (self-consistency test) | `[VERIFIED: LOCAL crates/cb-train/tests/*_oracle_test.rs pattern]` |

**Explicitly NOT a dependency:** `cb-model`'s `save_cbm`/`save_json` (dep-direction
violation — see §1) `[VERIFIED: LOCAL crates/cb-train/Cargo.toml:38-45]`.

No new external crate is required.

---

## 4. Typed Contracts

New `cb-train` module `crates/cb-train/src/snapshot.rs` (prod) with tests in
`crates/cb-train/src/snapshot_test.rs`, mounted via the project idiom
`#[cfg(test)] #[path = "snapshot_test.rs"] mod tests;`
`[VERIFIED: LOCAL in-crate cb-train precedent crates/cb-train/src/metrics.rs:714
(`#[path = "metrics_test.rs"]`). A cross-crate (cb-model) precedent is
crates/cb-model/src/ctr_data.rs:65 — noted as cross-crate, not an in-crate
example]`. Integration/self-consistency tests under
`crates/cb-train/tests/`.

```rust
// crates/cb-train/src/snapshot.rs

/// Caller-facing snapshot configuration (mirrors upstream `snapshot_file` /
/// `snapshot_interval`; upstream `save_snapshot=true` == passing `Some(_)`).
pub struct SnapshotConfig {
    /// Destination/source path. Resume triggers automatically iff this file
    /// already exists AND its fingerprint matches (upstream semantics).
    pub snapshot_file: std::path::PathBuf,
    /// Minimum wall-clock between checkpoint writes (upstream default 600s).
    pub snapshot_interval: std::time::Duration,
}

/// The on-disk snapshot state (serde DTO — mirrors cb-train's live loop state for
/// the scoped regime; converted to/from the live types so `Model`/`ObliviousTree`
/// stay serde-free, D-04).
///
/// `format_version` guards forward/backward changes; `fingerprint` is the compat
/// key (§ ORCH-03-S4); `completed_iters` is `K`; `approx`/`trees`/`rng_*` are the
/// loop-carried state for the scoped regime (§ ORCH-03-S1 audit).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct TrainSnapshot {
    pub format_version: u32,
    pub fingerprint: u64,
    pub completed_iters: usize,
    pub bias: f64,
    pub approx_dimension: usize,
    pub approx: Vec<f64>,
    pub trees: Vec<OblivousTreeDto>, // (splits/leaf_values/leaf_weights) — cb-train tree, DTO'd
    pub rng_raw_state: [u64; 4],
    pub rng_call_count: u64,
}

/// Compute the compat fingerprint from the resume-relevant inputs (loss kind,
/// hyperparameters, data shape + content hash). A resume against a snapshot whose
/// fingerprint differs is rejected (§ ORCH-03-S4).
///
/// COMPLETENESS CONTRACT (ORCH-03-S4). The fingerprint MUST fold EVERY
/// `BoostParams` field the scoped CPU tree-building path actually reads, so that
/// any change that would alter the resumed computation yields a different
/// fingerprint and is rejected — never silently accepted. The hashed field set
/// is (documented order is part of the contract):
///   1. the loss discriminant tag AND — for the in-scope parametric variants —
///      each numeric payload field via `.to_bits()` (see the LOSS PAYLOAD
///      SUB-CONTRACT below); the discriminant tag ALONE is NOT sufficient
///   2. `iterations`
///   3. `depth`
///   4. `learning_rate.to_bits()`
///   5. `l2_leaf_reg.to_bits()`
///   6. `random_seed`
///   7. a `boosting_type` tag
///   8. a `leaf_method` discriminant tag        (leaf-estimation branch, boosting.rs:4308-4338)
///   9. a `score_function` discriminant tag     (split scoring, boosting.rs:3850-3989)
///  10. `min_data_in_leaf`                       (included DEFENSIVELY / forward-compat;
///      NOT read on the scoped SymmetricTree path — see the CORRECTED note below)
///  11. each `monotone_constraints[i]` (`Vec<i8>`, element-wise; PAVA leaf projection)
///  12. `boost_from_average` (as `u8`)           (bias = starting_approx, boosting.rs:2591/1144)
///  13. `auto_learning_rate` (as `u8`)           (effective LR = autolr::guess(.., iterations), boosting.rs:2348-2365)
///  14. `n`
///  15. each `feature_borders[f][b].to_bits()`
///  16. each `target[i].to_bits()`
///
/// LOSS PAYLOAD SUB-CONTRACT (ORCH-03-S4, CRITICAL-3 fix). Field 1 folds in the
/// loss discriminant AND every numeric payload field of the in-scope,
/// single-dimension, non-ranking parametric `Loss` variants — because those `f64`
/// params are load-bearing for the per-object gradient/leaf math yet SHARE a
/// discriminant with a differently-parameterized sibling, so a discriminant-only
/// hash would accept a changed parameter silently. Verified against
/// `crates/cb-compute/src/runtime.rs` (exact variant + field names):
///   - `Loss::Quantile { alpha, delta }`   => hash(disc, alpha.to_bits(), delta.to_bits())  (runtime.rs:96-103; `quantile_der1` branches on `alpha`, cb-compute/src/loss.rs:166-175)
///   - `Loss::Lq { q }`                     => hash(disc, q.to_bits())                         (runtime.rs:116-119; `validate_leaf_method` branches on `q`, boosting.rs:1302-1310)
///   - `Loss::Huber { delta }`              => hash(disc, delta.to_bits())                     (runtime.rs:126-129)
///   - `Loss::Expectile { alpha }`          => hash(disc, alpha.to_bits())                     (runtime.rs:136-139)
///   - `Loss::Focal { alpha, gamma }`       => hash(disc, alpha.to_bits(), gamma.to_bits())    (runtime.rs:74-79)
///   - `Loss::Tweedie { variance_power }`   => hash(disc, variance_power.to_bits())            (runtime.rs:159-162)
///   - `Loss::MultiQuantile { alpha, delta }` (WHEN `alpha.len() == 1`, the only
///     single-dimension case, per `loss_approx_dimension`'s literal `alpha.len()`
///     for this variant — `alpha.len() > 1` is multi-dimensional and ALREADY
///     scope-guard-rejected by `approx_dimension != 1`, so it never reaches this
///     match) => hash(disc, alpha.len() as u64, alpha[0].to_bits(), delta.to_bits()).
///     **Added at PLAN-CHECK pass 3 — a from-scratch, complete re-enumeration of
///     ALL 28 `Loss` variants found this one was missed by the original six-variant
///     list (itself already a correction of the reviewer's initial list, which had
///     omitted `Tweedie`): `MultiQuantile` at `alpha.len()==1` is real,
///     constructible (`Loss::validate` only rejects an EMPTY `alpha`, not a
///     length-1 one), in-scope, and its numeric payload is load-bearing — it
///     reuses `quantile_der1` with `alpha[d]`, mathematically identical to the
///     already-fixed `Loss::Quantile`. Omitting it would have been the exact same
///     silent-corruption gap a THIRD time.**
///   - non-parametric single-dim variants (`Rmse`/`Logloss`/`CrossEntropy`/`Mae`/
///     `LogCosh`/`Poisson`/`Mape`)          => hash(disc) only
///   - `Loss::Custom(_)` NEVER reaches the fingerprint — it is rejected at entry by
///     the §S5 scope guard (no cross-process-stable identity; see design note).
/// ALL parametric variants above are confirmed single-dimension via
/// `loss_approx_dimension`'s `_ => 1` default arm (`boosting.rs:1074-1096`, or the
/// literal `alpha.len()` arm for `MultiQuantile` when that length is 1) and
/// non-grouped via `is_grouped_loss` returning `false` (`boosting.rs:989-1001`),
/// hence IN-SCOPE for slice 1 (which scopes on "non-ranking, single-dimension", NOT
/// on loss kind). The remaining multi-dimensional / ranking parametric variants
/// (`MultiQuantile` at `alpha.len() > 1`, `QuerySoftMax`/`LambdaMart`/`YetiRank*`/
/// `StochasticRank`) are scope-guard-rejected (`approx_dimension != 1` /
/// `is_grouped_loss`) BEFORE the fingerprint is computed, so they need no payload
/// hashing here. **Implementer note: before this task's Green step ships, do ONE
/// more full pass over the CURRENT `Loss` enum in `runtime.rs` (it may have grown
/// variants since this SPEC was written) and confirm no single-dimension,
/// non-grouped, parametric variant is still missing from this list — this
/// silent-corruption shape has now been found three times by three separate
/// review passes, and the enum is not `#[non_exhaustive]`-guarded against future
/// additions here, so an exhaustive (non-wildcard-catchall-for-payload) match is
/// the only durable fix.**
///
/// Fields 8–9 and 11–13 (five of the six CRITICAL-1 additions) each feed directly
/// into the scoped (Plain, float-only, no-CTR) tree-grow/leaf-estimation
/// computation but were previously unfingerprinted AND unguarded, so a resume that
/// changed one alone would have been silently accepted and silently wrong. NOTE
/// `auto_learning_rate`'s effective LR also depends on `iterations` (field 2,
/// already hashed).
///
/// NOTE on field 10 (`min_data_in_leaf`) — CORRECTED CITATION (was CRITICAL-3
/// MINOR). This field is included DEFENSIVELY / for forward-compat with a future
/// non-symmetric-in-scope slice; it is NOT read on the scoped SymmetricTree path.
/// The in-scope dispatch `greedy_tensor_search_oblivious_perturbed`
/// (`crates/cb-train/src/tree.rs:574-584`) takes NO `min_data_in_leaf` parameter;
/// the earlier `boosting.rs:3139`/`:3850`/`:3874` citations are all inside
/// device / Region / Lossguide|Depthwise arms that the §S5 scope guard ALREADY
/// excludes. Keeping the field in the fingerprint is harmless over-fingerprinting
/// (at worst a spurious-but-SAFE rejection; never a silent-corruption risk), so it
/// is retained defensively rather than removed
/// `[VERIFIED: LOCAL tree.rs:574-584 — signature has no min_data_in_leaf]`.
///
/// DESIGN NOTE — why fingerprint-expansion and NOT a whole-`BoostParams`
/// `PartialEq` check (the review's alternative option b): a cross-run compat
/// check must persist the snapshot-time params, but `BoostParams` is NOT
/// serde-storable — `loss: Loss` and `eval_metric: Option<EvalMetric>` carry
/// `Custom` variants holding `Arc<dyn CustomObjective>` / `Arc<dyn CustomMetric>`
/// trait objects `[VERIFIED: LOCAL cb-compute/src/custom.rs:117-175 — only
/// `Arc::ptr_eq` PartialEq, NO Serialize/Deserialize; Loss::Custom runtime.rs:494;
/// EvalMetric::Custom metrics.rs:167]`, which cannot derive `Serialize`/`Deserialize`.
/// So no whole-struct `PartialEq` across a process boundary is buildable; folding
/// the complete scoped-read field set into the persisted fingerprint (recomputed
/// from the current params at resume and compared) is the equivalent, buildable
/// guarantee. And because a `Custom` objective/metric has NO content-based,
/// cross-process-stable identity at all (`Arc::ptr_eq` is process-local, meaningless
/// across two separate process invocations), it CANNOT be soundly fingerprinted —
/// so `Loss::Custom(_)` / `EvalMetric::Custom(_)` are scope-guard-rejected at entry
/// (§S5), never fingerprinted.
pub fn fingerprint(
    params: &crate::BoostParams,
    n: usize,
    feature_borders: &[Vec<f64>],
    target: &[f64],
) -> u64;

/// Serialize the snapshot to a byte buffer (serde_json). Total function.
///
/// # Errors
/// `CbError::Degenerate` on a serialization failure (never panics).
pub fn encode(snap: &TrainSnapshot) -> CbResult<Vec<u8>>;

/// Deserialize + validate `format_version`. Total function.
///
/// # Errors
/// `CbError::Degenerate` on malformed bytes or an unknown `format_version`.
pub fn decode(bytes: &[u8]) -> CbResult<TrainSnapshot>;

/// Atomically write `snap` to `path` (temp file in the same dir + rename).
///
/// # Errors
/// `CbError::Degenerate` wrapping an IO error (never panics; no `unwrap`).
pub fn write_atomic(path: &std::path::Path, snap: &TrainSnapshot) -> CbResult<()>;
```

```rust
// crates/cb-core/src/rng.rs  (tiny addition)

impl TFastRng64 {
    /// Restore a generator from a `raw_state()` snapshot + `call_count`,
    /// reproducing the `gen_rand` stream bit-for-bit from that position.
    #[must_use]
    pub fn from_raw_state(raw_state: [u64; 4], call_count: u64) -> Self;
}
```

```rust
// crates/cb-train/src/boosting.rs  (additive signature; None ⇒ byte-identical)

fn train_inner<R: Runtime>(
    /* …existing params… */,
    snapshot: Option<&SnapshotConfig>,   // NEW, trailing — None on all 4 existing callers
) -> CbResult<(Model, BakedCtrData)>;

/// New public numeric entry: like `train_with_eval_sets` (no eval sets), but with
/// snapshot save/resume. Returns the model and the iteration the run RESUMED from
/// (`0` = fresh; `K>0` = resumed after `K` snapshotted trees), so callers/tests
/// can distinguish a genuine resume from a retrain.
///
/// # Errors
/// `CbError::Degenerate` for an out-of-scope regime (scope guard), a fingerprint
/// mismatch on resume, a malformed snapshot, or an IO failure.
pub fn train_with_snapshot<R: Runtime>(
    runtime: &R,
    feature_values: &[Vec<f32>],
    feature_borders: &[Vec<f64>],
    target: &[f64],
    weights: &[f64],
    params: &BoostParams,
    snapshot: &SnapshotConfig,
) -> CbResult<(Model, usize)>;
```

---

## 5. Failure-Isolated Behavioral Specifications

Each spec has ONE primary reason a failing acceptance test would fail. All specs
are **unimplemented** (spec+plan only).

### ORCH-03-S1 — Additive snapshot param + scoped loop-state audit (D-04 anchor)
- **Status:** unimplemented
- **Primary failure reason:** threading the optional `snapshot` param changes the
  behavior of an existing (`None`) caller, OR the resumable loop-state set for the
  scoped regime is not exactly `{approx, trees, rng}`.
- **Responsibility:** add `snapshot: Option<&SnapshotConfig>` to `train_inner`;
  the four existing callers pass `None`; and CONFIRM (audit, TASK-01) that on the
  scoped path every per-tree mutable local other than `approx`/`trees`/`rng` is
  inert (per the §1 table). Nothing else.
- **Input/Output:** unchanged signatures for the 4 callers; `None` path unchanged.
- **Behavior:** with `snapshot = None`, `train`/`train_with_eval_sets`/`train_cat`/
  `train_ranking` produce byte-identical models to the current code.
- **Invariants:** D-04 no-regression; the existing `cb-train` oracle suite stays
  green unchanged.
- **Acceptance:** the full `cargo test -p cb-train` suite passes with no fixture
  or expected-value change; a focused equality test (fresh vs fresh) is exact.
- **Traceability:** `boosting.rs:2259` (signature), `:3184` (loop), §1 state table.

### ORCH-03-S2 — `TFastRng64::from_raw_state` restore constructor
- **Status:** unimplemented
- **Primary failure reason:** a generator restored from `raw_state`+`call_count`
  does not reproduce the `gen_rand` stream bit-for-bit.
- **Responsibility:** a pure `cb-core` restore ctor; nothing else.
- **Input:** `[u64;4]` raw state + `u64` call_count. **Output:** `TFastRng64`.
- **Behavior (Given/When/Then):** Given a generator advanced by `M` draws, When
  `from_raw_state(g.raw_state(), g.call_count())` is constructed, Then the next
  `K` `gen_rand()` outputs (and the running `call_count`) equal the original's.
- **Invariants:** total; no panic; does not itself advance the stream.
- **Acceptance:** a `cb-core` unit round-trip oracle over several seeds/offsets.
- **Traceability:** `rng.rs:221` (`raw_state`), `:204` (`call_count`), `:171`
  (`from_seed`), `:158` (`new`).

### ORCH-03-S3 — Snapshot state model + serde round-trip
- **Status:** unimplemented
- **Primary failure reason:** `decode(encode(s))` is not equal to `s` (lossy /
  wrong field), or serde derives leak onto `Model`/`ObliviousTree` (D-04).
- **Responsibility:** the `TrainSnapshot` DTO + `encode`/`decode` (serde_json) +
  the DTO↔live-type conversions, confined to `snapshot.rs`.
- **Input/Output:** `TrainSnapshot` ⇄ `Vec<u8>`.
- **Behavior:** `decode(encode(&s))? == s` for a representative snapshot
  (multi-tree, non-trivial `approx`, non-zero RNG state); an unknown
  `format_version` ⇒ `Err(CbError::Degenerate)`; `Model`/`ObliviousTree` carry NO
  serde derive (conversion happens via public fields only).
- **Invariants:** total functions; no panic; version-guarded.
- **Acceptance:** `snapshot_test.rs` round-trip + bad-version unit tests.
- **Traceability:** `boosting.rs:771` (tree fields), `Cargo.toml:51`
  (serde_json — currently dev-only, promoted to normal by TASK-04's Cargo fix).

### ORCH-03-S4 — Compat fingerprint + mismatch rejection
- **Status:** unimplemented
- **Primary failure reason:** an incompatible resume (different loss / hyperparams
  / data) is NOT rejected, yielding a silently wrong model; or the fingerprint is
  non-deterministic.
- **Responsibility:** `fingerprint(params, n, feature_borders, target)` and the
  resume-time equality check.
- **Input:** the resume-relevant inputs. **Output:** `u64`.
- **Behavior:** identical inputs ⇒ identical fingerprint (deterministic across
  runs); any change to loss kind / **a parametric loss variant's numeric payload
  (`Quantile{alpha,delta}` / `Lq{q}` / `Huber{delta}` / `Expectile{alpha}` /
  `Focal{alpha,gamma}` / `Tweedie{variance_power}`)** / `iterations` / `depth` /
  `learning_rate` / `l2_leaf_reg` / `random_seed` / `boosting_type` / **`leaf_method` /
  `score_function` / `min_data_in_leaf` / `monotone_constraints` /
  `boost_from_average` / `auto_learning_rate`** / `n` / `feature_borders` /
  `target` ⇒ a different fingerprint with high probability; a resume whose stored
  fingerprint differs from the current one ⇒ `Err(CbError::Degenerate)`.
- **Completeness (CRITICAL-1 + CRITICAL-3).** (a) The six bolded hyperparameter
  fields all feed the scoped CPU tree-build/leaf-estimation computation
  (`leaf_method` `boosting.rs:4308-4338`, `score_function` `:3850-3989`,
  `monotone_constraints` PAVA leaf projection, `boost_from_average` bias
  `:2591`/`:1144`, `auto_learning_rate` effective-LR `:2348-2365`;
  `min_data_in_leaf` is included DEFENSIVELY — see §4's corrected field-10 note, it
  is NOT read on the scoped SymmetricTree path but is harmless over-fingerprinting)
  yet are NOT restricted by the §S5 scope guard. (b) The **loss discriminant tag
  alone is insufficient**: the in-scope single-dim parametric variants
  (`Quantile`/`Lq`/`Huber`/`Expectile`/`Focal`/`Tweedie`) carry load-bearing `f64`
  payloads (`quantile_der1` branches on `alpha`, `cb-compute/src/loss.rs:166-175`;
  `validate_leaf_method` branches on `Lq{q}`, `boosting.rs:1302-1310`) that a
  discriminant-only hash would miss, so a resume changing e.g. `Quantile{alpha:0.1}`
  → `Quantile{alpha:0.9}` would be silently accepted and silently wrong. Field 1
  therefore folds in each parametric payload via `.to_bits()` (§4 LOSS PAYLOAD
  SUB-CONTRACT). A whole-struct `BoostParams` `PartialEq` cross-run check is NOT
  buildable (`loss`/`eval_metric` carry `Arc<dyn>` `Custom` trait objects → not
  serde-storable, only `Arc::ptr_eq` identity), so completeness is enforced by
  folding every scoped-read field (incl. parametric-loss payloads) into the
  fingerprint (see §4). `Loss::Custom` / `EvalMetric::Custom` cannot be
  fingerprinted at all (no cross-process-stable identity) and are scope-guard
  rejected at entry (§S5), never fingerprinted.
- **Invariants:** total; no panic; stable hashing (documented algorithm).
- **Acceptance:** unit tests in `snapshot_test.rs`: same-inputs-equal;
  each-changed-input-differs with a row for EACH of the six added hyperparameter
  fields AND a row for at least one same-discriminant/changed-parameter case
  (`Quantile{alpha:0.1,delta:1e-6}` vs `Quantile{alpha:0.9,delta:1e-6}` ⇒ different
  fingerprint) plus `Lq{q}` (`Lq{q:2.0}` vs `Lq{q:3.0}`); mismatch-errs.
- **Traceability:** `BoostParams` fields `boosting.rs:165`; `monotone_constraints:
  Vec<i8>` `metrics.rs:368`; parametric `Loss` variants `runtime.rs:74-162`;
  `loss_approx_dimension` `_ => 1` `boosting.rs:1074-1096`; `Loss::Custom`
  `runtime.rs:494`; `EvalMetric::Custom` `metrics.rs:167`.

### ORCH-03-S5 — Periodic per-tree write + scope guard
- **Status:** unimplemented
- **Primary failure reason:** the snapshot is not written at a per-tree boundary /
  ignores `snapshot_interval`, is not crash-safe (torn file), OR an out-of-scope
  regime is allowed to write an unresumable snapshot.
- **Responsibility:** the in-loop write hook (after each tree, gated on interval)
  + the entry scope guard.
- **Input:** `snapshot: Option<&SnapshotConfig>` + the scoped params.
- **Behavior:**
  - When `Some(cfg)` and ≥ `cfg.snapshot_interval` has elapsed since the last
    write, after appending tree `iter`, write a `TrainSnapshot` with
    `completed_iters = iter + 1` atomically (temp + rename) to `cfg.snapshot_file`.
  - The file, when present, always reflects a whole-tree boundary (`K` trees, an
    `approx` consistent with those `K` trees).
  - When `Some(cfg)` but the regime is out of scope (ranking / CTR / ordered /
    eval sets / `bootstrap_type != No` / `random_strength != 0` /
    `approx_dimension != 1` / penalties / non-symmetric or Region grow policy /
    device-eligible / **a `Loss::Custom(_)` objective or an
    `EvalMetric::Custom(_)` metric**), `train_with_snapshot` returns
    `Err(CbError::Degenerate)` BEFORE any tree is grown — no snapshot is written.
    The `Custom` exclusions (CRITICAL-3) are mandatory: a custom objective/metric
    is an opaque `Arc<dyn>` trait object whose only equality is process-local
    `Arc::ptr_eq` (`cb-compute/src/custom.rs:134-142,165-169`), so NO
    cross-process-stable fingerprint can distinguish two custom instances — a
    snapshot under a custom objective/metric can never be soundly resumed and must
    be rejected at entry, not fingerprinted. (`Loss::Custom` is single-dimension
    and NOT grouped, so it is caught by neither `approx_dimension != 1` nor
    `is_grouped_loss` — it needs its OWN explicit predicate.)
  - `None` ⇒ no write, no guard (byte-identical, D-04).
- **Invariants:** atomic replace (no torn reader); no `unwrap`/panic on the IO
  path.
- **Acceptance:** integration test asserting a file appears with the expected `K`
  after a bounded run (interval set to 0 to force per-tree writes) + scope-guard
  tests: (i) `bootstrap_type != No` ⇒ `Err`, no file; (ii) `Loss::Custom(_)` +
  `SnapshotConfig` ⇒ typed `Err`, no file; (iii) `EvalMetric::Custom(_)` +
  `SnapshotConfig` ⇒ typed `Err`, no file.
- **Traceability:** loop `:3184`; `draws_active` `:3018`; device gate `:3051`.

### ORCH-03-S6 — Resume: load, pre-populate, start at K (self-consistency oracle)
- **Status:** unimplemented
- **Primary failure reason:** the resumed model is NOT identical to the
  straight-through model (wrong `approx`/`trees`/`rng` restore, or the loop does
  not start at `K`).
- **Responsibility:** the entry resume path — if `snapshot_file` exists and the
  fingerprint matches, load, set `trees`/`approx`/`bias`/`approx_dimension`,
  restore `rng` via `from_raw_state`, and run `for iter in K..iterations`.
- **Input:** an existing, fingerprint-matching snapshot.
- **Behavior (Given/When/Then):**
  - Given a straight-through model `M_full = train_with_snapshot(N)` and a fresh
    file, When a first run is stopped after `K` trees (snapshot written) and a
    second `train_with_snapshot(N)` finds that file, Then the second run RESUMES
    (`resumed_from == K`) and its model is **byte-identical** (serialized
    `TrainSnapshot` state) and predict-identical (exact) to `M_full`.
  - Given no snapshot file, Then `resumed_from == 0` (fresh) and the model equals
    `M_full`.
  - Given a fingerprint mismatch, Then `Err(CbError::Degenerate)` (from S4).
- **Invariants:** exact (deterministic no-draw scalar CPU path); no partial-tree
  application (`approx` is restored verbatim, not recomputed).
- **Intermediate-fixture construction (CRITICAL-2 — locked, not deferred).** The
  "trained K trees, snapshot written as if interrupted mid-`iterations=N`-run"
  fixture is built deterministically from PUBLIC surfaces only, with NO new
  prod-code interruption primitive: run `train_with_snapshot` with `iterations=K`
  and `snapshot_interval=Duration::ZERO` into a temp file (its final per-tree
  write, from S5, is a genuine K-tree `TrainSnapshot`); `decode` it (S3), overwrite
  its `fingerprint` field with `fingerprint(&params_N, n, feature_borders, target)`
  (S4) so the fixture carries `iterations=N`, and `write_atomic` it to the resume
  path (S5). This K-run snapshot is byte-identical to a genuine mid-`N`-run
  interruption at K **because** the scoped path's first K trees are
  `iterations`-independent when `auto_learning_rate=false` (each tree depends only
  on the running `approx` + fixed params, not on `params.iterations`). The keystone
  fixtures therefore pin `auto_learning_rate=false`: with `auto_learning_rate=true`
  the effective LR is `autolr::guess(.., params.iterations)` (`boosting.rs:2348-2365`),
  so a K-run and an N-run's first K trees would use DIFFERENT learning rates and
  the equivalence would not hold. An `auto_learning_rate=true` resume is not
  silently wrong — it is caught by the fingerprint (field 13, §4) and errors; its
  self-consistency is a later slice.
- **Acceptance:** the keystone self-consistency integration test
  (`snapshot_resume_selfconsistency_oracle_test.rs`), several `(N, K)` pairs, all
  with `auto_learning_rate=false`.
- **Traceability:** `approx` `:2599`/`:2612`, `trees` `:2952`, `rng` `:3011`,
  loop `:3184`, assembly `:4746`.

### ORCH-03-S7 — Rust facade `train_with_snapshot`
- **Status:** unimplemented
- **Primary failure reason:** the public entry does not wire write+resume, or its
  `resumed_from` return does not reflect a genuine resume.
- **Responsibility:** the numeric public entry composing S5 (write) + S6 (resume);
  passes empty eval sets / no ranking / no cat.
- **Input:** `(runtime, feature_values, feature_borders, target, weights, params,
  snapshot: &SnapshotConfig)`. **Output:** `CbResult<(Model, usize)>`.
- **Behavior:** end-to-end — first call fresh (`resumed_from==0`, snapshot
  written); a second call on the same file resumes (`resumed_from==K`) to an
  identical model; re-exported from `cb_train` (`lib.rs`).
- **Acceptance:** covered by the S6 keystone test driving this entry.
- **Traceability:** `train_with_eval_sets` `:2048` (entry precedent), `lib.rs`
  re-export site.

---

## 6. Acceptance Scenarios

| ID | Scenario | Oracle | Tolerance |
|----|----------|--------|-----------|
| AT-S1 | `snapshot=None` ⇒ full `cb-train` suite unchanged; fresh==fresh | existing suite + unit | exact |
| AT-S2 | `from_raw_state(raw_state, call_count)` reproduces the `gen_rand` stream | cb-core unit round-trip | exact (bitwise) |
| AT-S3 | `decode(encode(s))==s`; bad `format_version` ⇒ `Err`; no serde on `Model` | unit | exact |
| AT-S4 | fingerprint: same-inputs-equal, each-changed-input-differs (incl. a row per each of `leaf_method`/`score_function`/`min_data_in_leaf`/`monotone_constraints`/`boost_from_average`/`auto_learning_rate`, AND a same-discriminant/changed-payload row for `Quantile{alpha}` and `Lq{q}`), mismatch ⇒ `Err` | unit | — |
| AT-S5a | with interval 0, a snapshot file with the expected `K` appears per tree | integration | — |
| AT-S5b | out-of-scope regime (e.g. `bootstrap_type != No`) ⇒ `Err`, no file written | integration | — |
| AT-S5c | `Loss::Custom(_)` + `SnapshotConfig` ⇒ typed `Err`, no file; `EvalMetric::Custom(_)` + `SnapshotConfig` ⇒ typed `Err`, no file | integration | — |
| AT-S6 | straight-through `N` == `K` + resume `N-K` (fixture re-fingerprinted from a `K`-run, `auto_learning_rate=false`); `resumed_from==K` | self-consistency | exact (bytes + predict) |
| AT-S7 | `train_with_snapshot` fresh-then-resume end-to-end identical model | self-consistency | exact |

---

## 7. Impact Scope

- **`crates/cb-core/src/rng.rs`** — `local` (additive `from_raw_state`; no change
  to `new`/`from_seed`/`gen_rand`; blast radius of `TFastRng64` unaffected)
  `[VERIFIED: CODEGRAPH rng.rs]`.
- **`crates/cb-train/src/snapshot.rs`** (NEW) — `local`. New leaf module
  (DTO/serde/fingerprint/atomic-write); no existing symbol changes semantics.
- **`crates/cb-train/src/boosting.rs`** — `cross-module`. `train_inner` gains a
  TRAILING optional param (the 4 callers pass `None`); the loop gains a resume
  pre-populate before it and a write hook at the per-tree boundary inside the CPU
  arm; a new public `train_with_snapshot`. The `None`/device/ranking/CTR/ordered
  paths stay byte-identical (D-04). **Blast radius to verify** (TASK-01): callers
  of `train_inner` (3 internal) + `train`/`train_with_eval_sets`/`train_cat`/
  `train_ranking` (many oracle-test callers) `[VERIFIED: CODEGRAPH blast radius:
  train_ranking 9 callers, train_cat 3, train_with_eval_sets 5, train_inner 3]`.
- **`crates/cb-train/src/lib.rs`** — `local`. `pub mod snapshot;` +
  `pub use snapshot::SnapshotConfig;` + re-export `train_with_snapshot`.
- **Tests** — new `snapshot_test.rs` (unit: S2 lives in cb-core, S3/S4 in
  cb-train), `snapshot_write_oracle_test.rs` (S5), and
  `snapshot_resume_selfconsistency_oracle_test.rs` (S6/S7, the keystone).

No persistence-schema, event, cache, config-flag, or `BoostParams`-shape change
(the `SnapshotConfig` is a SEPARATE param, deliberately NOT a `BoostParams` field
— see §9). No public contract of an existing symbol changes.

---

## 8. Compatibility and Migration

- **Additive.** No existing signature, serialization format, or behavior changes;
  the four training entries keep their exact signatures and outputs (they pass
  `None` internally). `TFastRng64` gains a ctor; nothing existing changes.
- **Snapshot format is versioned** (`format_version`) and cb-train-internal (NOT
  the `.cbm`/JSON model format) — no interop promise beyond this repo, and no
  migration of existing models. Rollback = remove the `snapshot` module + param +
  the rng ctor.
- **Naming parity:** `snapshot_file` / `snapshot_interval` mirror upstream
  boosting params; `save_snapshot=true` maps to passing `Some(SnapshotConfig)`.

---

## 9. Risks and Open Questions

| Risk | Consequence | Mitigation |
|------|-------------|------------|
| The scoped loop-state set is NOT just `{approx,trees,rng}` (some per-tree mutable local leaks in) | resumed model diverges silently | **Gating spike TASK-01** audits every `mut` local on the scoped path against the §1 table BEFORE any injection; NO-GO ⇒ escalate to a design-spike phase `[VERIFIED: CODEGRAPH §1 table lines]` |
| `train_inner` is ~2500 lines; the injection is delicate | regression on existing dependents | Additive TRAILING `Option` param (`None` byte-identical); resume pre-populate is a single pre-loop block; write hook is a single per-tree-boundary block in the CPU arm; device/ranking/CTR/ordered arms untouched; full `cargo test -p cb-train` is the D-04 anchor (AT-S1) |
| RNG not actually restorable exactly | wrong resume once draws are enabled | `raw_state` already exposes full state (`rng.rs:221`); `from_raw_state` round-trip oracle (AT-S2). In slice 1 the no-draw regime makes RNG inert anyway (`draws_active==false`, `:3018`) — the restore is belt-and-suspenders + forward-compat |
| `approx` reconstruction assumption | broken resume | AVOIDED — `approx` is serialized verbatim (§1), not recomputed from a partial model; no summation-order dependence |
| Fingerprint omits a scoped-read hyperparameter (`leaf_method`/`score_function`/`monotone_constraints`/`boost_from_average`/`auto_learning_rate`; `min_data_in_leaf` is included DEFENSIVELY only — NOT scoped-read, see §4 corrected field-10 note) | resume with a changed field silently accepted + silently WRONG (CRITICAL-1) | Fingerprint folds EVERY scoped-CPU-read `BoostParams` field plus the defensive `min_data_in_leaf` (§4 completeness contract, 16 fields); a per-field `fingerprint_sensitive` test row proves each of the six is caught. Whole-struct `PartialEq` is not buildable (`Arc<dyn>` `Custom` variants aren't serde-storable), so completeness lives in the hash |
| Fingerprint hashes only the loss DISCRIMINANT, missing a parametric variant's `f64` payload (`Quantile{alpha,delta}`/`Lq{q}`/`Huber{delta}`/`Expectile{alpha}`/`Focal{alpha,gamma}`/`Tweedie{variance_power}`) | resume that changes only a payload (e.g. `Quantile{0.1}`→`Quantile{0.9}`) silently accepted + silently WRONG (CRITICAL-3) | Field 1 folds each in-scope parametric variant's payload via `.to_bits()` (§4 LOSS PAYLOAD SUB-CONTRACT); `fingerprint_sensitive` rows for `Quantile{alpha}` and `Lq{q}` same-discriminant/changed-payload prove it. Verified in-scope single-dim (`loss_approx_dimension` `_ => 1` `boosting.rs:1074-1096`) + non-grouped (`is_grouped_loss` `:989-1001`), payloads load-bearing (`quantile_der1` on `alpha` `loss.rs:166-175`; `validate_leaf_method` on `Lq{q}` `:1302-1310`) |
| A `Loss::Custom(_)` / `EvalMetric::Custom(_)` snapshot can't be fingerprinted (opaque `Arc<dyn>`, only `Arc::ptr_eq` identity, process-local) | resume under a swapped custom objective/metric silently accepted + silently WRONG (CRITICAL-3) | Scope guard REJECTS `Loss::Custom(_)` / `EvalMetric::Custom(_)` at entry with a typed `Err` BEFORE any file is written (§S5) — not caught by `approx_dimension != 1` (Custom is single-dim) nor `is_grouped_loss` (not grouped), so an explicit predicate is required; integration test AT-S5c proves it. `[VERIFIED: LOCAL custom.rs:117-175; runtime.rs:494; metrics.rs:167]` |
| Reusing `cb-model` serialization would invert the dep graph | build breakage / illegal edge | Snapshot serializer lives in `cb-train`, serializes cb-train's own types via `serde_json`; `cb-model` is NEVER referenced `[VERIFIED: LOCAL Cargo.toml:38-45,51]` |
| Torn snapshot file on crash mid-write | corrupt resume | Atomic temp-file-then-rename; readers only ever see a complete file |
| Lint gate is CLIPPY not build (`unwrap`/`expect`/`panic`/`indexing_slicing` denied) | CI red despite `cargo build` green | All new prod code returns typed `CbError`; gate each change with `cargo clippy -p <crate> --lib --no-deps` `[VERIFIED: LOCAL MEMORY fstr03-plan gotchas]` |
| Test-mount omission runs 0 tests silently | false green | Mount `snapshot_test.rs` via `#[cfg(test)] #[path=…] mod tests;` `[VERIFIED: LOCAL crates/cb-train/src/metrics.rs:714]` |
| Non-determinism would break the EXACT oracle | flaky test | Slice 1 is the no-draw scalar CPU path: no RNG draws, `sum_f64` deterministic reductions, single-threaded scalar accumulation ⇒ exact bit-equality is legitimate `[VERIFIED: CODEGRAPH draws_active :3018]` |

**Open questions**

1. **`SnapshotConfig` as a separate param vs a `BoostParams` field.** RESOLVED
   (design decision, this SPEC): a SEPARATE trailing param, NOT three new
   `BoostParams` fields — adding fields to `BoostParams` would ripple through
   every oracle test that builds it by literal (large blast radius / D-04 risk).
   The separate-param form leaves `BoostParams` untouched. `[VERIFIED: CODEGRAPH
   BoostParams :165; broad test construction]`
2. **Snapshot tree serialization mechanism.** RESOLVED (design decision): a
   `serde`-derived DTO in `snapshot.rs` mirroring cb-train's public tree fields
   (NOT serde derives on `Model`/`ObliviousTree`), (de)serialized with
   `serde_json`. Confirm in TASK-01/TASK-03 that the cb-train `ObliviousTree`
   fields (`splits`/`leaf_values`/`leaf_weights`/`ctr_splits`) are all `pub` and
   plain-data. `[VERIFIED: LOCAL boosting.rs:771 fields pub]`
3. **Fingerprint hashing algorithm.** TBD at TASK-05 Green: a documented,
   deterministic content hash over the §4 completeness-contract inputs (all 16
   fields, incl. the six CRITICAL-1 additions AND field 1's CRITICAL-3 parametric-
   loss payload folding — `Quantile`/`Lq`/`Huber`/`Expectile`/`Focal`/`Tweedie`
   `.to_bits()`) (e.g. `std` `DefaultHasher` is
   NOT guaranteed stable across toolchains — prefer an explicit stable hash, e.g.
   FNV/xxhash-style folded over the byte reprs, or pin the algorithm). Resolve to
   a run-to-run stable choice; it need not match upstream's hash. `[UNVERIFIED —
   resolve at TASK-05]`
4. **Does the scoped no-draw regime ever engage the device grow path?** On the
   default `CpuBackend`, `begin_device_training` always returns `Ok(false)` so
   `device_active==false` `[VERIFIED: CODEGRAPH boosting.rs:3158-3176 + CLAUDE.md
   CpuBackend]`; the scope guard additionally rejects device-eligible configs.
   Confirm in TASK-01.
5. **S6 keystone intermediate-fixture construction.** RESOLVED (§5 ORCH-03-S6,
   CRITICAL-2): a `K`-run snapshot re-fingerprinted to the `N`-run's fingerprint
   (deterministic; PUBLIC surfaces only; no new prod interruption primitive),
   valid because the scoped path's first K trees are `iterations`-independent when
   `auto_learning_rate=false` — which the keystone fixtures pin.

**Gating decision (must be answered before implementation).** Slice 1 proceeds
ONLY if TASK-01 confirms the scoped loop-state set is exactly `{approx, trees,
rng}` and the additive injection points are clean. This SPEC judges — from the §1
code audit already performed — that GO is **likely**, so a standalone
design-spike *phase* is NOT mandated; the spike is folded into TASK-01 as a
gating in-plan task. If TASK-01 returns NO-GO, STOP and escalate.

---

## 10. Traceability and Sources

- **Training loop:** `crates/cb-train/src/boosting.rs` — `train_inner` (2259),
  loop (3184), `approx` (2599/2612), `trees` (2952), `rng` (3011), `draws_active`
  (3018), device gate (3051/3158), assembly (4746); `train`/`train_with_eval_sets`/
  `train_cat`/`train_ranking` (1946/2048/2145/2092); `BoostParams` (165);
  `ObliviousTree` (771); `Model` (852) `[VERIFIED: CODEGRAPH/LOCAL]`.
- **RNG:** `crates/cb-core/src/rng.rs` — `TFastRng64` (137), `from_seed` (171),
  `new` (158), `gen_rand` (183), `advance` (192), `call_count` (204), `raw_state`
  (221) `[VERIFIED: CODEGRAPH/LOCAL]`.
- **Dep direction / serde:** `crates/cb-train/Cargo.toml:38-45` (cb-model is a
  DEV-dep; cb-model→cb-train build edge), `:51` (serde_json) `[VERIFIED: LOCAL]`.
- **cb-model serialization (NOT reused):** `crates/cb-model/src/cbm.rs:495,817,827`,
  `json.rs:625,803`; `from_trained` `model.rs:326` (takes `&cb_train::Model`)
  `[VERIFIED: LOCAL]`.
- **House style:** `.planning/phases/20-orchestration/calc-metrics/{SPEC,PLAN,
  PLAN-CHECK}.md` `[VERIFIED: LOCAL]`.
- **Upstream:** CatBoost `snapshot_file`/`snapshot_interval`/`save_snapshot`
  boosting params; `LearnProgress` persists RNG state + trees + approx
  `[INFERRED/UNVERIFIED — model knowledge; catboost-master is a sparse checkout
  here]`.

---

## 11. Implementation Evidence

> NONE. This is a spec+plan artifact only. Every ORCH-03-S* is `unimplemented`.
> No production code was authored. The document lifecycle is `status: draft`.
