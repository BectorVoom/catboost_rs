---
title: "ORCH-03 — Training snapshot / resume (slice 1) — TDD Implementation Plan"
phase: 20-orchestration
slice: snapshot-resume
plan_version: 1
status: implemented
updated_at: 2026-08-03T00:00:00Z
result: ./IMPLEMENTATION-RESULT.md
spike_findings: ./TASK-01-findings.md
source_spec: .planning/plans/snapshot-resume/SPEC.md
source_research: "Phase-20 three-gap research pass (snapshot/resume = highest-risk gap)"
gsd_used: false
---

# ORCH-03 — TDD Implementation Plan (Slice 1)

Plan-only artifact. No production code authored here. Every file/symbol/line
below is verified against the on-disk source via CodeGraph + Read (evidence
inline). The training arithmetic (`train_inner`'s tree grower, `EvalMetric`, the
apply path) is **reused, never modified**; the only edits are ADDITIVE (an
optional trailing param, a pre-loop resume block, a per-tree write hook, a new
public entry, a new leaf module, and one tiny `cb-core` ctor).

> **Risk posture (read first).** This is the highest-risk Phase-20 gap.
> `train_inner` is a single ~2500-line function whose entire mutable state is
> local variables (`crates/cb-train/src/boosting.rs:2259-4757`). **TASK-01 is an
> explicit GATING VERIFICATION SPIKE, not a Red/Green increment.** It must return
> GO before any code-writing task starts. If TASK-01 returns NO-GO (some per-tree
> mutable local leaks into the scoped path, or the additive injection is not
> clean), STOP: escalate to a standalone design-spike phase and do NOT proceed
> with TASK-02+ on this plan. This is called out again in §4.

## 0. Goal-backward derivation

Acceptance outcomes (SPEC §6) drive the task set:

| Acceptance | Observable success | Task |
|---|---|---|
| AT-S1 | `snapshot=None` ⇒ full `cargo test -p cb-train` unchanged; fresh==fresh exact | TASK-03 (+ gated by TASK-01) |
| AT-S2 | `from_raw_state(raw_state, call_count)` reproduces the `gen_rand` stream bit-for-bit | TASK-02 |
| AT-S3 | `decode(encode(s))==s`; bad `format_version` ⇒ `Err`; no serde on `Model`/`ObliviousTree` | TASK-04 |
| AT-S4 | fingerprint deterministic; each-changed-input-differs (incl. same-discriminant/changed-payload `Quantile{alpha}`/`Lq{q}`); mismatch ⇒ `Err` | TASK-05 |
| AT-S5a/b/c | per-tree write with expected `K` (interval 0); out-of-scope regime ⇒ `Err`, no file; `Loss::Custom`/`EvalMetric::Custom` ⇒ typed `Err`, no file | TASK-06 |
| AT-S6 | straight-through `N` == `K`+resume `N-K`; `resumed_from==K`; exact model | TASK-07 |
| AT-S7 | `train_with_snapshot` fresh-then-resume end-to-end identical model | TASK-08 |

Reused seams (verified — do NOT modify):

- `train_inner` — `crates/cb-train/src/boosting.rs:2259`; the per-tree loop
  `for iter in 0..params.iterations` at `:3184`; model assembly
  `Ok((Model { oblivious_trees: trees, non_symmetric_trees, region_trees, bias,
  approx_dimension, class_to_label }, baked))` at `:4746-4756`. `[VERIFIED: Read]`
- Loop-carried mutable locals for the scoped regime: `approx` (`:2599`/`:2612`),
  `trees: Vec<ObliviousTree>` (`:2952`), `rng = TFastRng64::from_seed(random_seed)`
  (`:3011`). `draws_active` (`:3018`) is FALSE for `bootstrap_type=No` +
  `random_strength=0`, so `rng` is never consumed on the scoped path. `[VERIFIED: Read]`
- Out-of-scope per-tree locals, each gated behind an out-of-scope feature (SPEC §1
  table): `detector`/`best_model`/`eval_approx`/`history` (`:2979`-`:2990`, eval
  sets), `prev_leaf_mean_l2` (`:3021`, MVS), `used_features` (`:2685`, penalties),
  `group_spans`/`yetirank_seeder`/`stochasticrank_seeder`/`learn_approx`
  (`:2383`-`:2624`, ranking), `ordered_learning_perm` (`:2643`, Ordered), CTR fold
  columns / `BakedCtrData` (`:2863`/`:4708`, cat). `[VERIFIED: Read]`
- The 4 public callers of `train_inner`: `train` (`:1946`)→`train_with_eval_sets`
  (`:2048`), `train_ranking` (`:2092`), `train_cat` (`:2145`) — each constructs
  the full `train_inner(...)` argument list explicitly. A new TRAILING param means
  each gains a `None` argument. `[VERIFIED: CODEGRAPH/Read boosting.rs:2062-2167]`
- `cb_train::ObliviousTree` (`boosting.rs:771`) fields `splits`/`ctr_splits`/
  `leaf_values`/`leaf_weights` — the DTO source; DISTINCT from
  `cb_model::ObliviousTree` (`model.rs:254`). `[VERIFIED: LOCAL]`
- `cb_core::rng::TFastRng64` — `from_seed` (`rng.rs:171`), `new(seed1,seq1,seed2,
  seq2)` (`:158`), `gen_rand` (`:183`), `advance` (`:192`), `call_count` (`:204`),
  `raw_state -> [r1.x, r1.c, r2.x, r2.c]` (`:221`). NO restore ctor exists —
  TASK-02 adds one. `Lcg32` fields `x`/`c` are set inside `cb-core`
  (`raw_state` reads `self.r1.x` etc.). `[VERIFIED: Read]`
- Dependency direction: `cb-train` build graph does NOT include `cb-model`
  (`cb-model` is a `cb-train` DEV-dep only; `cb-model → cb-train` at build time) —
  the snapshot serializer must be cb-train-internal. **CORRECTION (PLAN-CHECK
  pass 3): `serde_json` is currently a DEV-ONLY dependency of `cb-train`
  (`Cargo.toml:51`, under `[dev-dependencies]`), and `serde` is not a
  dependency at all — NEITHER is usable from production code as-is. TASK-04
  adds both as normal `[dependencies]` (see TASK-04's "CARGO DEPENDENCY FIX").**
  `[VERIFIED: Read crates/cb-train/Cargo.toml:38-45,51]`
- Test-mount idiom `#[cfg(test)] #[path="X_test.rs"] mod tests;` — confirmed
  IN-CRATE at `crates/cb-train/src/metrics.rs:714` (`#[path = "metrics_test.rs"]`).
  (`crates/cb-model/src/ctr_data.rs:65` shows the SAME idiom but is a cross-crate
  cb-model precedent, not cb-train.) `[VERIFIED: LOCAL]`
- Oracle harness pattern (integration tests carry the clippy allow-block; drive
  a public train entry, compare) — `crates/cb-train/tests/*_oracle_test.rs`.
  `[VERIFIED: LOCAL]`

Design invariants locked by the SPEC (do NOT re-litigate at implementation):

- `approx` is serialized VERBATIM (no partial-model reconstruction) ⇒ no
  summation-order dependence (SPEC §1). 
- `SnapshotConfig` is a SEPARATE trailing param, NOT `BoostParams` fields (SPEC §9
  Q1) ⇒ `BoostParams` and every oracle test that builds it stay untouched.
- The snapshot serializer lives in `cb-train`; `cb-model` is NEVER referenced
  (SPEC §9, dep-direction).

## 1. Execution order & waves

```
Wave A (parallel):  TASK-01 (GATING SPIKE, cb-train read-only audit)
                    ∥ TASK-02 (cb-core rng from_raw_state)        [disjoint crate/file]
        ── TASK-01 must return GO before Wave B ──
Wave B:  TASK-03 (additive `snapshot` param + None on 4 callers; D-04 anchor)   depends: TASK-01
Wave C:  TASK-04 (snapshot DTO + serde encode/decode round-trip)               depends: TASK-03
Wave D:  TASK-05 (fingerprint + mismatch rejection)                            depends: TASK-04
Wave E:  TASK-06 (per-tree write hook + scope guard)                           depends: TASK-03, TASK-05
Wave F:  TASK-07 (resume load/pre-populate + loop-start-at-K + self-consistency) depends: TASK-06, TASK-02
Wave G:  TASK-08 (facade `train_with_snapshot` end-to-end)                      depends: TASK-07
```

Dependency graph:

```
TASK-01 (spike) ─GO─> TASK-03 ─> TASK-04 ─> TASK-05 ─> TASK-06 ─> TASK-07 ─> TASK-08
TASK-02 ──────────────────────────────────────────────────────────┘
             (TASK-02 feeds TASK-07's rng restore; otherwise independent)
```

Acyclic. Parallelism: TASK-01 ∥ TASK-02 ONLY (disjoint: cb-train read-only audit
vs cb-core `rng.rs`). TASK-03..08 form a SINGLE sequential chain — all edit
`crates/cb-train/src/{boosting.rs, snapshot.rs, lib.rs}` (TASK-03 and TASK-04 both
touch `lib.rs`: `pub mod snapshot;` / the re-exports), so NONE of them may run in
parallel with each other. The diagram above shows them as one chain to match this
constraint (an earlier draft drew TASK-03/TASK-04 as sibling branches off TASK-01,
which was misleading — they share `lib.rs`/`snapshot.rs` and must be sequential).

## 2. Spec-ID → task coverage

| Spec | Behavior | Task(s) |
|---|---|---|
| ORCH-03-S1 | additive `snapshot` param + scoped loop-state audit (D-04 anchor) | TASK-01 (audit) + TASK-03 (param) |
| ORCH-03-S2 | `TFastRng64::from_raw_state` restore ctor | TASK-02 |
| ORCH-03-S3 | snapshot DTO + serde round-trip | TASK-04 |
| ORCH-03-S4 | compat fingerprint + mismatch rejection | TASK-05 |
| ORCH-03-S5 | per-tree write + scope guard | TASK-06 |
| ORCH-03-S6 | resume load/pre-populate + start-at-K (self-consistency oracle) | TASK-07 |
| ORCH-03-S7 | facade `train_with_snapshot` | TASK-08 |

Every S1..S7 covered; TASK-01 is a gating prerequisite (audits S1's core claim and
GO/NO-GO gates the whole slice).

---

## TASK-01 — GATING VERIFICATION SPIKE (audit; NOT a Red/Green task) (ORCH-03-S1)

- **Type:** investigation / verification spike. Produces a GO/NO-GO decision + a
  confirmed "resume state set", NOT production code.
- **Spec refs:** ORCH-03-S1 (its core claim: the scoped loop-carried mutable state
  set is exactly `{approx, trees, rng}`). Primary risk being retired: a per-tree
  mutable local leaks into the scoped path and would make a restored resume
  diverge silently.
- **Goal / completion:** a written GO/NO-GO with, for GO, (a) the confirmed
  loop-carried mutable-state set for the scoped regime, (b) the exact injection
  points (pre-loop resume block location; per-tree write-hook location(s); loop
  bound edit), and (c) confirmation the 4 callers construct `train_inner(...)`
  positionally so a trailing `None` is mechanical.
- **Prerequisites:** none (parallel with TASK-02).
- **Method (read-only; CodeGraph + Read):**
  1. Re-walk `train_inner` `:2259`→`:4757`. For EACH `let mut` binding, classify:
     loop-carried vs computed-once-before-loop vs written-only-inside-an-out-of-
     scope branch. Confirm the SPEC §1 table: on the scoped path (Plain, non-
     ranking, empty `cat_columns`, no eval sets, `bootstrap_type=No`,
     `random_strength=0`, `approx_dimension==1`, no penalties, SymmetricTree,
     `device_active==false`) the ONLY loop-carried mutable locals read at the top
     of iteration `k+1` and written at the bottom of iteration `k` are `approx`,
     `trees`, `rng`.
  2. Confirm `draws_active` (`:3018`) is provably FALSE for the scoped regime ⇒
     `rng` is untouched across the loop (so restore is a no-op for the exact
     oracle but still captured for forward-compat).
  3. Confirm `device_active` (`:3158`) is FALSE on the default `CpuBackend`
     (`begin_device_training` returns `Ok(false)`), so the device `continue`
     branch (`:3193`) is never taken; the write hook lives at the END of the CPU
     iteration body (after the tree is pushed + `approx` updated).
  4. Confirm the pre-loop resume injection point: the block just BEFORE `:3184`
     (after all once-only setup) where `trees`/`approx`/`rng` can be replaced and
     the loop bound switched to `resume_from..iterations`.
  5. Confirm the 4 callers (`:2062`, `:2102`, `:2155`, `:2009`→`:2062`) build the
     `train_inner` arg list positionally (so a trailing `None` compiles with a
     one-line edit each).
  6. Enumerate the exact scope-guard predicate set (the negation of the scoped
     regime) from the real gate expressions already present
     (`is_grouped_loss`, `ctr_candidates.is_empty()`, `boosting_type`,
     `eval_sets.is_empty()`, `bootstrap_type`, `random_strength`,
     `approx_dimension`, `penalties_active`, `grow_policy`, `device_host_eligible`)
     PLUS the two CRITICAL-3 `Custom` predicates
     (`matches!(params.loss, Loss::Custom(_))`,
     `matches!(params.eval_metric, Some(EvalMetric::Custom(_)))`) — these are NOT
     covered by `approx_dimension`/`is_grouped_loss` (Custom is single-dim &
     non-grouped) and MUST be added explicitly (TASK-06).
- **GO criteria:** items 1–6 all confirmed AND no additional loop-carried mutable
  state found on the scoped path.
- **NO-GO handling (must state explicitly):** if item 1 finds ANY extra
  loop-carried mutable local on the scoped path, or the injection points are not
  cleanly additive, STOP. Do NOT start TASK-03+. Escalate to a standalone design-
  spike phase (trainer-state extraction / struct-ification of `train_inner`),
  and revise the SPEC scope. Record the offending local + line.
- **Deliverable:** the GO/NO-GO note appended to this plan's progress (or a short
  `TASK-01-findings` section), including the confirmed state set and injection
  points, feeding TASK-03/06/07.
- **Validation:** none executable (read-only). The "test" is the audit itself;
  its output gates every downstream task.
- **Parallelization:** parallel with TASK-02. Blocks TASK-03..08.

---

## TASK-02 — `TFastRng64::from_raw_state` restore constructor (ORCH-03-S2)

- **Spec refs:** ORCH-03-S2. Primary failure reason: a restored generator does not
  reproduce the `gen_rand` stream bit-for-bit.
- **Goal / completion:** `cb_core::rng::TFastRng64::from_raw_state([u64;4], u64)`
  exists; a `cb-core` round-trip unit test passes; `cargo clippy -p cb-core --lib
  --no-deps` clean.
- **Prerequisites:** none (parallel with TASK-01; disjoint file).
- **Files:**
  - Modify: `crates/cb-core/src/rng.rs` — add `pub fn from_raw_state(raw_state:
    [u64;4], call_count: u64) -> Self` that sets `r1 = Lcg32 { x: raw_state[0],
    c: raw_state[1] }`, `r2 = Lcg32 { x: raw_state[2], c: raw_state[3] }`,
    `call_count`. (Confirm the `Lcg32` field names/visibility at TASK-02 Green —
    `raw_state` reads `self.r1.x`/`self.r1.c` at `:222`, so `x`/`c` exist and are
    at least crate-visible; a struct literal in the same module is valid.)
  - Modify (or create): the `cb-core` rng test file (follow the crate's existing
    test-file convention — a sibling `rng_test.rs` mounted via
    `#[cfg(test)] #[path="rng_test.rs"] mod tests;` if `rng.rs` uses that idiom,
    else the existing rng test module). Confirm the convention before writing.
- **CodeGraph/Read evidence:** `raw_state` `:221`, `call_count` `:204`, `gen_rand`
  `:183`, `new` `:158`, `Lcg32` field access `:222`. `[VERIFIED: Read]`
- **Red:** a round-trip test: `let mut a = TFastRng64::from_seed(seed); for _ in
  0..M { a.gen_rand(); } let mut b = TFastRng64::from_raw_state(a.raw_state(),
  a.call_count()); for _ in 0..K { assert_eq!(a.gen_rand(), b.gen_rand()); }
  assert_eq!(a.call_count(), b.call_count());` over several `(seed, M, K)`.
  Expected INITIAL failure: `from_raw_state` unresolved ⇒ the test file fails to
  build.
- **Green:** implement the ctor (direct field set). The round-trip test passes.
- **Refactor:** none (a 4-field constructor). Regression scope: `cargo test -p
  cb-core` — `new`/`from_seed`/`gen_rand`/`advance`/`call_count` untouched.
- **Validation:**
  - `cargo test -p cb-core` (rng round-trip + existing rng tests green)
  - `cargo clippy -p cb-core --lib --no-deps`
- **Completion evidence:** round-trip green over ≥3 offsets; clippy clean; no
  change to any existing rng symbol.
- **Compat/rollback:** additive ctor; rollback = remove `from_raw_state` + its test.
- **Parallelization:** parallel with TASK-01. Independent of TASK-03/04.

---

## TASK-03 — Additive `snapshot` param + `None` on the 4 callers (ORCH-03-S1)

- **Spec refs:** ORCH-03-S1 (the additive-param half). Primary failure reason:
  threading the param changes an existing (`None`) caller's behavior.
- **Goal / completion:** `train_inner` gains a TRAILING `snapshot:
  Option<&SnapshotConfig>`; the 4 callers pass `None`; a new `snapshot` module
  scaffold exists (`SnapshotConfig` struct only for now); the FULL `cargo test -p
  cb-train` suite passes UNCHANGED (the D-04 anchor); `cargo clippy -p cb-train
  --lib --no-deps` clean.
- **Prerequisites:** TASK-01 GO.
- **Files:**
  - Create: `crates/cb-train/src/snapshot.rs` — module doc + `pub struct
    SnapshotConfig { pub snapshot_file: std::path::PathBuf, pub snapshot_interval:
    std::time::Duration }`. Mount tests: `#[cfg(test)] #[path="snapshot_test.rs"]
    mod tests;`.
  - Create: `crates/cb-train/src/snapshot_test.rs` — (empty scaffold now; S3/S4
    tests land in TASK-04/05).
  - Modify: `crates/cb-train/src/lib.rs` — `pub mod snapshot;` +
    `pub use snapshot::SnapshotConfig;`.
  - Modify: `crates/cb-train/src/boosting.rs` — add `snapshot: Option<&SnapshotConfig>`
    as the LAST param of `train_inner` (`:2259`); pass `None` at each of the 4
    call sites (`train_with_eval_sets` `:2062`, `train_ranking` `:2102`,
    `train_cat` `:2155`, and `train` via `train_with_eval_sets`). Body otherwise
    untouched (the param is unused this task — `let _ = snapshot;` if clippy flags
    it, or gate its first read in TASK-06/07).
- **CodeGraph/Read evidence:** the 4 call sites + arg lists `:2009-2167`; loop
  `:3184`; `BoostParams` NOT touched (`:165`). `[VERIFIED: Read]`
- **Red:** a focused equality test in a new
  `crates/cb-train/tests/snapshot_none_regression_test.rs` (or reuse an existing
  train harness): train the same tiny numeric pool twice with `train`
  (`snapshot=None` path) and assert the two `Model`s are equal (exact) — plus the
  keystone anchor is the EXISTING suite. Expected INITIAL failure: BEFORE the
  param is threaded, the test file referencing the (future) `None`-passing entry
  need not exist; the real "red" here is the compile break from adding a param
  without updating a caller (caught by `cargo build`). This task's success signal
  is that the WHOLE existing suite still passes after the edit.
- **Green:** thread the param; pass `None` everywhere; existing suite unchanged.
- **Refactor:** none. Regression scope: `cargo test -p cb-train` (the entire
  suite is the D-04 anchor — every oracle test drives one of the 4 callers).
- **Validation:**
  - `cargo build -p cb-train` (the 4 callers compile with the new arg)
  - `cargo test -p cb-train` (FULL suite unchanged — AT-S1)
  - `cargo clippy -p cb-train --lib --no-deps`
- **Completion evidence:** full suite green with no fixture/expected change;
  `SnapshotConfig` compiles; clippy clean.
- **Compat/rollback:** additive; rollback = remove the param + the 4 `None`s +
  the module + the 2 `lib.rs` lines.
- **Parallelization:** sequential (edits `boosting.rs` + `lib.rs`; blocks TASK-04+
  on `lib.rs`/`snapshot.rs`).

---

## TASK-04 — Snapshot DTO + serde round-trip (ORCH-03-S3)

- **Spec refs:** ORCH-03-S3. Primary failure reason: `decode(encode(s)) != s`
  (lossy/wrong field) OR serde derives leak onto `Model`/`ObliviousTree`.
- **Goal / completion:** `snapshot.rs` defines `TrainSnapshot` (+ an
  `ObliviousTreeDto` mirroring cb-train's `ObliviousTree` public fields), `encode`
  / `decode` (serde_json), and DTO↔live conversions; round-trip + bad-version unit
  tests pass; clippy clean; `Model`/`ObliviousTree` carry NO serde derive.
- **Prerequisites:** TASK-01 GO (and TASK-03's module scaffold).
- **CARGO DEPENDENCY FIX (blocking, found at PLAN-CHECK pass 3 — do this FIRST,
  before any `snapshot.rs` code):** `crates/cb-train/Cargo.toml` does NOT
  currently have `serde` as a dependency at all, and has `serde_json` ONLY
  under `[dev-dependencies]` (`Cargo.toml:51`, confirmed via direct read — the
  earlier "`Cargo.toml:51` proves this is a normal dependency" citation
  elsewhere in this plan/SPEC was WRONG). A dev-only dependency is invisible to
  the crate's own library build (`cargo build -p cb-train`) and to any
  downstream crate, so `#[derive(serde::Serialize, serde::Deserialize)]` and
  `serde_json::to_vec`/`from_slice` in PRODUCTION `snapshot.rs` code would fail
  to compile outside `cargo test`. **Required fix:** add BOTH
  `serde.workspace = true` and `serde_json.workspace = true` under
  `crates/cb-train/Cargo.toml`'s `[dependencies]` section (promoting
  `serde_json` out of `[dev-dependencies]` if the existing dev-only entry would
  otherwise conflict — check at edit time), mirroring the already-correct
  pattern in `crates/cb-model/Cargo.toml:37-38` (`serde.workspace = true` /
  `serde_json.workspace = true`, both normal deps). Both `serde`/`serde_json`
  are already pinned at the workspace root (`Cargo.toml:23-24`), so this is a
  one-line-per-crate addition, no new external dependency.
- **Files:**
  - Modify: `crates/cb-train/Cargo.toml` — add `serde.workspace = true` and
    `serde_json.workspace = true` under `[dependencies]` (see fix above).
  - Modify: `crates/cb-train/src/snapshot.rs` — add `#[derive(serde::Serialize,
    serde::Deserialize, PartialEq, Debug)] struct ObliviousTreeDto { splits:
    Vec<SplitDto>, leaf_values: Vec<f64>, leaf_weights: Vec<f64> }` (+ a
    `SplitDto { feature: usize, border: f64 }` mirroring `cb_train::Split`; NOTE
    `ctr_splits` is EMPTY on the scoped float path — carry it as an assert-empty
    or omit-with-guard, confirm the `Split`/`ctr_splits` shapes at Green),
    `TrainSnapshot` (SPEC §4), `encode`/`decode`, and
    `fn dto_from_tree(&ObliviousTree)->ObliviousTreeDto` /
    `fn tree_from_dto(&ObliviousTreeDto)->ObliviousTree` conversions over PUBLIC
    fields only.
  - Modify: `crates/cb-train/src/snapshot_test.rs` — S3 unit tests.
- **CodeGraph/Read evidence:** `cb_train::ObliviousTree` fields `boosting.rs:771`
  (`splits`, `ctr_splits`, `leaf_values`, `leaf_weights`); `serde_json` currently
  DEV-ONLY at `Cargo.toml:51` — promoted to normal by this task's Cargo fix
  above; `cb-model` NOT referenced (dep direction). Confirm the `Split`
  type + its fields at Green. `[VERIFIED: Read boosting.rs:771; Cargo.toml:51]`
- **Red:** in `snapshot_test.rs`:
  - `snapshot_roundtrip` — build a `TrainSnapshot` with 2 non-trivial trees, a
    length-`n` `approx`, non-zero `rng_raw_state`/`rng_call_count`; assert
    `decode(&encode(&s)?)? == s`.
  - `snapshot_bad_version_errs` — hand-craft bytes with an unknown
    `format_version`; assert `decode(...).is_err()`.
  Expected INITIAL failure: `TrainSnapshot`/`encode`/`decode` unresolved ⇒ build
  fails.
- **Green:** implement the DTO + `encode` (`serde_json::to_vec(...).map_err(|e|
  CbError::Degenerate(e.to_string()))`) + `decode` (`serde_json::from_slice`
  then version check) + conversions. No serde on `Model`/`ObliviousTree`.
- **Refactor:** keep the conversions private + total. Regression scope:
  `snapshot_test.rs` only; `boosting.rs` tree types untouched.
- **Validation:**
  - `cargo build -p cb-train` **(MANDATORY, run FIRST — added at the post-cap
    extra verification pass, pass 4, closing CRITICAL-6: this task's own
    Cargo dependency fix, above, was found to have been specified without a
    non-test build check to actually catch a misplaced entry. `cargo test -p
    cb-train` CANNOT detect `serde`/`serde_json` being left under
    `[dev-dependencies]` instead of `[dependencies]`, since dev-dependencies
    ARE visible when compiling a crate's own test targets — only a plain,
    non-test `cargo build -p cb-train` exercises the crate's LIBRARY build,
    which is where a dev-only dependency would actually fail to resolve. This
    is the same class of "cargo test passes, cargo build doesn't" trap the
    Cargo fix itself was raised to close — verify the fix is placed correctly,
    don't just verify the fix was mentioned.)**
  - `cargo test -p cb-train --lib snapshot`
  - `cargo clippy -p cb-train --lib --no-deps` (no `unwrap`/`panic` on encode/decode)
  - `cargo build -p cb-model` (downstream sanity: `cb-model` depends on
    `cb-train` normally — confirms the newly-normal `serde`/`serde_json` deps
    don't break the downstream build either)
- **Completion evidence:** `cargo build -p cb-train` green BEFORE the test
  suite is trusted; round-trip + bad-version tests green; `grep` confirms no
  `#[derive(...Serialize...)]` on `Model`/`ObliviousTree`.
- **Compat/rollback:** additive; rollback = remove the DTO/serde fns + tests.
- **Parallelization:** sequential after TASK-03 (same file / `lib.rs`).

---

## TASK-05 — Compat fingerprint + mismatch rejection (ORCH-03-S4)

- **Spec refs:** ORCH-03-S4. Primary failure reason: an incompatible resume is not
  rejected (silent wrong model) OR the fingerprint is non-deterministic.
- **Goal / completion:** `snapshot::fingerprint(params, n, feature_borders,
  target) -> u64` (deterministic, run-to-run stable) + the resume-time equality
  check surface; unit tests pass; clippy clean.
- **Prerequisites:** TASK-04 (fingerprint is a `TrainSnapshot` field).
- **Files:**
  - Modify: `crates/cb-train/src/snapshot.rs` — add `pub fn fingerprint(...)`.
    **Resolve SPEC §9 Q3 here:** use a documented STABLE hash (NOT `std`
    `DefaultHasher`, whose output is not guaranteed stable across toolchains) —
    e.g. a hand-rolled FNV-1a folded over the little-endian byte reprs, in this
    documented order (order is part of the contract; mirrors the §4 completeness
    contract exactly — 16 fields):
      1. the loss discriminant tag AND (CRITICAL-3) each in-scope PARAMETRIC
         variant's numeric payload via `.to_bits()` — see the LOSS PAYLOAD match
         below; the discriminant ALONE is NOT sufficient
      2. `iterations`
      3. `depth`
      4. `learning_rate.to_bits()`
      5. `l2_leaf_reg.to_bits()`
      6. `random_seed`
      7. a `boosting_type` tag
      8. a `leaf_method` discriminant tag        (`LeafMethod`, cb-compute; `:4308-4338`)
      9. a `score_function` discriminant tag     (`EScoreFunction`, cb-compute; `:3850-3989`)
     10. `min_data_in_leaf`                       (DEFENSIVE / forward-compat only —
         NOT read on the scoped SymmetricTree path; see the corrected-citation note below)
     11. each `monotone_constraints[i]` (`Vec<i8>`, element-wise byte; PAVA projection)
     12. `boost_from_average` as `u8`            (bias, `:2591`/`:1144`)
     13. `auto_learning_rate` as `u8`            (effective-LR, `:2348-2365`)
     14. `n`
     15. each `feature_borders[f][b].to_bits()`
     16. each `target[i].to_bits()`
    **LOSS PAYLOAD match (field 1, CRITICAL-3 — verified against
    `crates/cb-compute/src/runtime.rs`; confirm the exact variant/field names at
    Green):**
    ```rust
    match &params.loss {
        // non-parametric single-dim variants: discriminant tag only
        Loss::Rmse | Loss::Logloss | Loss::CrossEntropy | Loss::Mae
            | Loss::LogCosh | Loss::Poisson | Loss::Mape => feed_disc(&mut h, &params.loss),
        Loss::Quantile { alpha, delta } => { feed_disc(&mut h, &params.loss); feed(&mut h, &alpha.to_bits().to_le_bytes()); feed(&mut h, &delta.to_bits().to_le_bytes()); } // runtime.rs:96-103
        Loss::Lq { q }                  => { feed_disc(&mut h, &params.loss); feed(&mut h, &q.to_bits().to_le_bytes()); }                                                        // runtime.rs:116-119
        Loss::Huber { delta }           => { feed_disc(&mut h, &params.loss); feed(&mut h, &delta.to_bits().to_le_bytes()); }                                                    // runtime.rs:126-129
        Loss::Expectile { alpha }       => { feed_disc(&mut h, &params.loss); feed(&mut h, &alpha.to_bits().to_le_bytes()); }                                                    // runtime.rs:136-139
        Loss::Focal { alpha, gamma }    => { feed_disc(&mut h, &params.loss); feed(&mut h, &alpha.to_bits().to_le_bytes()); feed(&mut h, &gamma.to_bits().to_le_bytes()); }      // runtime.rs:74-79
        Loss::Tweedie { variance_power } => { feed_disc(&mut h, &params.loss); feed(&mut h, &variance_power.to_bits().to_le_bytes()); }                                          // runtime.rs:159-162
        // MultiQuantile is multi-dimensional (approx_dimension = alpha.len()) at
        // alpha.len() > 1, so the ONLY way this arm is reached is alpha.len() == 1
        // (single-dim, in-scope) — TASK-06's scope guard already rejects the
        // multi-dim case before fingerprint() is called. Hash alpha's LENGTH too
        // (not just alpha[0]) as defense-in-depth in case the scope guard's
        // approx_dimension check is ever bypassed or reordered.
        Loss::MultiQuantile { alpha, delta } => {
            feed_disc(&mut h, &params.loss);
            feed(&mut h, &(alpha.len() as u64).to_le_bytes());
            if let Some(a0) = alpha.first() { feed(&mut h, &a0.to_bits().to_le_bytes()); }
            feed(&mut h, &delta.to_bits().to_le_bytes());
        }                                                                                                                                                                          // runtime.rs:281-286; added at PLAN-CHECK pass 3 (missed by the original six-variant list)
        // Loss::Custom(_) and all remaining multi-dim/ranking variants are
        // scope-guard-rejected at entry (TASK-06) BEFORE fingerprint() is ever
        // called, so they are unreachable here; if a future refactor changes
        // that, this arm must reject rather than silently discriminant-hash.
        _ => feed_disc(&mut h, &params.loss),
    }
    ```
All parametric variants above (six discriminant-plus-payload arms, plus
    `MultiQuantile` at `alpha.len()==1`) are confirmed single-dimension via
    `loss_approx_dimension`'s `_ => 1` default arm (`boosting.rs:1074-1096`, or
    its literal `alpha.len()` arm for `MultiQuantile`) and non-grouped via
    `is_grouped_loss` returning `false` (`boosting.rs:989-1001`), hence
    IN-SCOPE. NOTE `Tweedie{variance_power}` and `MultiQuantile{alpha,delta}`
    (at length 1) were each added BEYOND an earlier reviewer's illustrative
    list, across two separate PLAN-CHECK passes — this silent-corruption shape
    has now been found three times; **before Green, re-scan the CURRENT `Loss`
    enum in `runtime.rs` one more time for any variant added since this SPEC
    was written and not yet covered here.** Each payload is
    load-bearing: `quantile_der1(approx, target, alpha, delta)` branches on `alpha`
    (`cb-compute/src/loss.rs:166-175`); `validate_leaf_method` branches on `Lq{q}`
    (`boosting.rs:1302-1310`). `[VERIFIED: Read runtime.rs:74-162; loss.rs:166-175;
    boosting.rs:1074-1096,1302-1310,989-1001]`
    **CRITICAL-1 (blocking) resolution.** Fields 8–9, 11–13 (five of the six the
    Plan Checker found MISSING) each feed the scoped (Plain, float-only, no-CTR)
    tree-build / leaf-estimation path yet is NOT restricted by TASK-06's scope
    guard, so a resume that changed one alone would be silently accepted + silently
    WRONG. They are folded into the fingerprint (option a). (`min_data_in_leaf`,
    field 10, is the sixth — retained DEFENSIVELY; see the corrected-citation note.)
    The reviewer's alternative (option b — a whole-`BoostParams` `PartialEq`
    cross-run check) is NOT buildable here: `BoostParams.loss: Loss` and
    `.eval_metric: Option<EvalMetric>` carry `Custom` variants holding
    `Arc<dyn CustomObjective>` / `Arc<dyn CustomMetric>` trait objects, whose only
    equality is process-local `Arc::ptr_eq` and which cannot derive
    `Serialize`/`Deserialize` `[VERIFIED: Read cb-compute/src/custom.rs:117-175;
    Loss::Custom runtime.rs:494; EvalMetric::Custom metrics.rs:167]`, so the
    snapshot-time params cannot be persisted for a cross-run struct comparison.
    Folding the complete scoped-read field set (incl. parametric-loss payloads)
    into the persisted fingerprint is the equivalent, buildable guarantee.
    **CRITICAL-3 field-10 citation correction (MINOR).** `min_data_in_leaf` is
    kept in the fingerprint DEFENSIVELY / for forward-compat, NOT because the scoped
    path reads it — it does not. The in-scope SymmetricTree dispatch
    `greedy_tensor_search_oblivious_perturbed` (`crates/cb-train/src/tree.rs:574-584`)
    takes NO `min_data_in_leaf` parameter; the earlier `boosting.rs:3139`/`:3850`/
    `:3874` citations are all inside device / Region / Lossguide|Depthwise arms
    that TASK-06's scope guard ALREADY excludes. Keeping the field is harmless
    over-fingerprinting (worst case: a spurious-but-SAFE rejection; never a
    silent-corruption risk), so it is retained, not removed. `[VERIFIED: Read
    tree.rs:574-584 — signature has no min_data_in_leaf; boosting.rs BoostParams
    :165; monotone_constraints Vec<i8> metrics.rs:368]`
  - Modify: `crates/cb-train/src/snapshot_test.rs` — S4 unit tests.
- **CodeGraph/Read evidence:** `BoostParams` fields `boosting.rs:165`; `f64::to_bits`
  for stable float hashing. `[VERIFIED: Read]`
- **Red:** in `snapshot_test.rs`:
  - `fingerprint_deterministic` — same inputs twice ⇒ equal.
  - `fingerprint_sensitive` — a table of single-field mutations, each ⇒ a
    different fingerprint. The table MUST enumerate EVERY hashed field (complete
    coverage — this is the CRITICAL-1 + CRITICAL-3 test): loss DISCRIMINANT
    (e.g. Rmse vs Logloss) / iterations / depth / lr / l2 / seed / boosting_type /
    **leaf_method / score_function / min_data_in_leaf / monotone_constraints /
    boost_from_average / auto_learning_rate** / n / a border / a target value.
    The six bolded rows are the CRITICAL-1 additions. In ADDITION (CRITICAL-3),
    the table MUST include same-discriminant / changed-PAYLOAD rows that a
    discriminant-only hash would miss:
      - `Quantile{alpha:0.1, delta:1e-6}` vs `Quantile{alpha:0.9, delta:1e-6}`
        ⇒ different fingerprint (changed `alpha`, same discriminant);
      - `Lq{q:2.0}` vs `Lq{q:3.0}` ⇒ different fingerprint (changed `q`, same
        discriminant);
      - `MultiQuantile{alpha:vec![0.1], delta:1e-6}` vs
        `MultiQuantile{alpha:vec![0.9], delta:1e-6}` ⇒ different fingerprint
        (changed `alpha[0]`, same discriminant, both length-1/single-dim —
        added at PLAN-CHECK pass 3, closing the third instance of this gap).
    (Recommended, not required: also a `Quantile` `delta`-only row and a
    `Focal`/`Huber`/`Expectile`/`Tweedie` payload row for full parametric
    coverage.) Each flips exactly that one field/payload on an otherwise-identical
    `BoostParams` and asserts `fingerprint(&a, ..) != fingerprint(&b, ..)`.
  - `fingerprint_mismatch_errs` — a helper `check_resume(stored, current)` returns
    `Err(CbError::Degenerate)` when the two differ, `Ok(())` when equal.
  Expected INITIAL failure: `fingerprint` unresolved ⇒ build fails.
- **Green:** implement the stable hash + the equality check helper.
- **Refactor:** factor the byte-folding into a small local `feed(&mut u64, &[u8])`;
  no behavior change. Regression scope: `snapshot_test.rs`.
- **Validation:**
  - `cargo test -p cb-train --lib snapshot`
  - `cargo clippy -p cb-train --lib --no-deps`
- **Completion evidence:** deterministic + sensitive (a row per EACH of the 16
  hashed fields, incl. the six CRITICAL-1 additions AND the CRITICAL-3
  same-discriminant/changed-payload rows for `Quantile{alpha}` and `Lq{q}`) +
  mismatch tests green; documented field order + the LOSS PAYLOAD match match the
  §4 completeness contract.
- **Compat/rollback:** additive; rollback = remove `fingerprint` + tests.
- **Parallelization:** sequential after TASK-04 (same file).

---

## TASK-06 — Periodic per-tree write hook + scope guard (ORCH-03-S5)

- **Spec refs:** ORCH-03-S5. Primary failure reason: no per-tree/interval-honoring
  write, non-atomic write, OR an out-of-scope regime is allowed to write.
- **Goal / completion:** with `snapshot=Some(cfg)` on the scoped path, a snapshot
  file appears at per-tree boundaries honoring `snapshot_interval` (atomic replace)
  carrying the expected `K`; an out-of-scope regime returns `Err(CbError::Degenerate)`
  BEFORE any tree grows (no file). `None` ⇒ byte-identical (D-04).
- **Prerequisites:** TASK-03 (param), TASK-05 (fingerprint + DTO from TASK-04).
- **Files:**
  - Modify: `crates/cb-train/src/snapshot.rs` — add `pub fn write_atomic(path,
    &TrainSnapshot) -> CbResult<()>` (write to `path.with_extension("tmp")` in the
    same dir, `fs::rename` into place; map IO errors to `CbError::Degenerate`; no
    `unwrap`).
  - Modify: `crates/cb-train/src/boosting.rs` —
    (a) A scope-guard helper `fn snapshot_scope_ok(params, cat_columns, eval_sets,
        approx_dimension, penalties_active, device_active) -> CbResult<()>` invoked
        at `train_inner` entry ONLY when `snapshot.is_some()`; returns
        `Err(Degenerate)` for any out-of-scope predicate (the negation set from
        TASK-01 item 6). Confirm each predicate reads an already-computed local
        (e.g. `is_grouped_loss(&params.loss)`, `!ctr_candidates.is_empty()`,
        `params.boosting_type != Plain`, `!eval_sets.is_empty()`, `bootstrap_type
        != No`, `random_strength != 0.0`, `approx_dimension != 1`,
        `penalties_active`, non-Symmetric `grow_policy`, `device_active`,
        **`matches!(params.loss, Loss::Custom(_))`**, **`matches!(params.eval_metric,
        Some(EvalMetric::Custom(_)))`**). NOTE: some of these locals are computed
        AFTER entry — place the guard right after the last of them is available but
        BEFORE the loop (per TASK-01 injection map), so it still fires before any
        tree is grown.
        **CRITICAL-3 (blocking) — the two `Custom` predicates are mandatory.** A
        `Loss::Custom(_)` objective / `EvalMetric::Custom(_)` metric is an opaque
        `Arc<dyn>` trait object whose only equality is process-local `Arc::ptr_eq`
        (`cb-compute/src/custom.rs:134-142,165-169`) with no `Serialize`/
        `Deserialize`, so NO cross-process-stable fingerprint can distinguish two
        custom instances — a snapshot under a custom objective/metric can never be
        soundly resumed and MUST be rejected at entry, never written. Neither is
        caught by the other predicates: `Loss::Custom` is single-dimension
        (`loss_approx_dimension` `_ => 1`, `boosting.rs:1074-1096`) so
        `approx_dimension != 1` misses it, and it is NOT in `is_grouped_loss`
        (`boosting.rs:989-1001`) — so each needs its OWN explicit predicate.
        `[VERIFIED: Read custom.rs:117-175; Loss::Custom runtime.rs:494;
        EvalMetric::Custom metrics.rs:167; BoostParams.eval_metric boosting.rs:225]`
    (b) A `let mut last_snapshot_write = std::time::Instant::now();` alongside the
        loop-carried state, and at the END of the CPU iteration body (after the
        tree push + `approx` update, before the iteration ends), when
        `snapshot.is_some()` and `last_snapshot_write.elapsed() >= cfg.snapshot_interval`,
        build a `TrainSnapshot { completed_iters: iter+1, approx, trees→DTO, rng
        raw_state/call_count, bias, approx_dimension, fingerprint }` and
        `write_atomic(...)`; reset `last_snapshot_write`. (Device arm out of scope;
        it `continue`s before this point and is excluded by the guard.)
  - Create: `crates/cb-train/tests/snapshot_write_oracle_test.rs`.
- **CodeGraph/Read evidence:** loop body end (CPU arm) after the tree push +
  `approx` update (the CPU grower's push + `staged_out` extend pattern mirrors the
  device arm at `:3392`/`:3388`); `draws_active` `:3018`; `device_active` `:3158`;
  the gate predicates `is_grouped_loss`/`ctr_candidates`/`boosting_type`/
  `eval_sets`/`bootstrap_type`/`random_strength`/`approx_dimension`/
  `penalties_active`/`grow_policy` all already present. `[VERIFIED: Read]`
- **Red:** `snapshot_write_oracle_test.rs`:
  - `writes_per_tree_with_expected_k` — train a tiny numeric pool for `N=4` with
    `SnapshotConfig { snapshot_file: <tempdir>/s.snap, snapshot_interval:
    Duration::ZERO }` via `train_with_snapshot` (TASK-08 entry) OR — since TASK-08
    is later — via a thin test-only call path threading `Some(cfg)` into
    `train_inner`. To avoid ordering coupling, this test drives the eventual
    `train_with_snapshot`; if run before TASK-08, mark it `#[ignore]` and land the
    live assertion in TASK-08's Verify. (Prefer: land the WRITE test here using a
    minimal internal `train_with_snapshot` stub that TASK-08 completes; keep this
    task's assertion on `write_atomic` + `decode` of the produced file: after the
    run, `decode(fs::read(path))` has `completed_iters == 4`.)
  - `out_of_scope_regime_errs` — call the snapshot entry with `bootstrap_type !=
    No`; assert `Err` AND the file does NOT exist.
  - `custom_objective_regime_errs` (CRITICAL-3) — call the snapshot entry with
    `params.loss = Loss::Custom(<a test CustomObjectiveHandle>)` (single-dim,
    Plain, otherwise in-scope); assert a typed `Err(CbError::Degenerate)` AND the
    snapshot file does NOT exist (rejected at entry, before any tree).
  - `custom_metric_regime_errs` (CRITICAL-3) — call the snapshot entry with an
    otherwise-in-scope config but `params.eval_metric =
    Some(EvalMetric::Custom(<a test CustomMetricHandle>))`; assert a typed
    `Err(CbError::Degenerate)` AND no snapshot file exists.
  Expected INITIAL failure: no write occurs / wrong `K` / guard missing (and,
  before the `Custom` predicates are added, the `Custom` runs would NOT error and
  would write a file — the CRITICAL-3 red).
- **Green:** implement `write_atomic` + the guard + the in-loop hook.
- **Refactor:** extract the `TrainSnapshot`-build into a `fn capture_snapshot(iter,
  &approx, &trees, &rng, bias, approx_dimension, fingerprint) -> TrainSnapshot`
  reused by TASK-07's final write. Regression scope: `None` path unchanged (D-04
  anchor `cargo test -p cb-train`).
- **Validation:**
  - `cargo test -p cb-train --test snapshot_write_oracle_test`
  - `cargo test -p cb-train` (D-04 anchor: `None` path unchanged)
  - `cargo clippy -p cb-train --lib --no-deps`
- **Completion evidence:** file with `completed_iters==K` appears; interval honored
  (ZERO ⇒ every tree; a large interval ⇒ at most the final write); out-of-scope ⇒
  `Err` + no file (incl. the CRITICAL-3 `Loss::Custom(_)` and `EvalMetric::Custom(_)`
  rejection cases — typed `Err`, no file); full suite green.
- **Compat/rollback:** additive; rollback = remove the hook + guard + `write_atomic`
  + the test.
- **Parallelization:** sequential after TASK-03/05 (same `boosting.rs`/`snapshot.rs`).

---

## TASK-07 — Resume: load, pre-populate, start at K + self-consistency oracle (ORCH-03-S6)

- **Spec refs:** ORCH-03-S6 (the keystone). Primary failure reason: the resumed
  model is NOT identical to the straight-through model (wrong restore, or the loop
  does not start at `K`).
- **Goal / completion:** on `train_inner` entry, when `snapshot=Some(cfg)` and
  `cfg.snapshot_file` exists with a matching fingerprint, load it, set
  `trees`/`approx`/`bias`/`approx_dimension`, restore `rng` via `from_raw_state`,
  and run `for iter in K..iterations`; the self-consistency oracle passes EXACTLY.
- **Prerequisites:** TASK-06 (write + guard + capture helper), TASK-02 (rng restore).
- **Files:**
  - Modify: `crates/cb-train/src/boosting.rs` — at the pre-loop injection point
    (TASK-01 item 4): if `snapshot.is_some()` and the file exists, `decode` it,
    `snapshot::check_resume(stored.fingerprint, current_fingerprint)?` (S4), then
    `trees = stored.trees.map(tree_from_dto)`, `approx = stored.approx`,
    `bias`/`approx_dimension` from the snapshot (assert they match the fresh
    compute or take the stored — document which; they are deterministic from
    inputs so they MUST match — assert-equal as a guard), `rng =
    TFastRng64::from_raw_state(stored.rng_raw_state, stored.rng_call_count)`, and
    set `resume_from = stored.completed_iters`. Change the loop header `for iter in
    0..params.iterations` (`:3184`) to `for iter in resume_from..params.iterations`
    (`resume_from` defaults to `0` ⇒ byte-identical when not resuming, D-04).
    Surface `resume_from` to the caller via the new entry (TASK-08).
  - Create: `crates/cb-train/tests/snapshot_resume_selfconsistency_oracle_test.rs`.
- **CodeGraph/Read evidence:** loop header `:3184` (bound edit is the only loop
  change); `approx`/`trees`/`rng` init sites `:2599`/`:2612`/`:2952`/`:3011`
  (the pre-populate REPLACES these values after their init); model assembly
  consumes exactly `trees`/`bias`/`approx_dimension` `:4746`. `[VERIFIED: Read]`
- **Red:** `snapshot_resume_selfconsistency_oracle_test.rs` (drives
  `train_with_snapshot`, TASK-08 — if landing before TASK-08, thread `Some(cfg)`
  via the internal path):
  - `resume_equals_straight_through` — for `(N,K)` in `{(4,1),(4,2),(6,3)}`, ALL
    with `auto_learning_rate=false` (see the locked mechanism below):
    (i) `M_full = train_with_snapshot(params_N)` on a FRESH file (deleted first);
    (ii) build the intermediate resume fixture via the LOCKED
    `refingerprinted_k_run` mechanism (below) → `P_resume`;
    (iii) `(M_resumed, resumed_from) = train_with_snapshot(params_N)` pointed at
    `P_resume`; assert `resumed_from == K` AND `encode(capture(M_resumed)) ==
    encode(capture(M_full))` (byte-identical) AND `predict(M_resumed) ==
    predict(M_full)` exactly.
  - `no_file_is_fresh` — no snapshot file ⇒ `resumed_from == 0`, model == `M_full`.
  - `fingerprint_mismatch_errs` — pre-write a snapshot with a mismatched
    fingerprint ⇒ `Err`.
  Expected INITIAL failure: without the resume block, the second run retrains from
  0 (`resumed_from==0`, or a wrong/torn state) ⇒ the assertion fails.
  - **LOCKED intermediate-fixture mechanism `refingerprinted_k_run` (CRITICAL-2 —
    decided now, NOT deferred to Green).** No new prod-code interruption primitive
    is introduced; the fixture is built deterministically from PUBLIC surfaces the
    earlier tasks already ship:
    1. Run `train_with_snapshot` with a params clone whose `iterations = K` and
       `snapshot_interval = Duration::ZERO`, into a temp path `P_k`. TASK-06's
       per-tree write (interval ZERO) fires at every tree, so the FINAL write of
       this run is a genuine K-tree `TrainSnapshot` with `completed_iters == K`.
    2. `let mut snap_k = snapshot::decode(&fs::read(P_k)?)?;` (TASK-04 `decode`,
       `pub`). This carries the true K-tree `approx`/`trees`/`rng`/`bias`, but its
       `fingerprint` was computed for `iterations = K`.
    3. Overwrite the one field: `snap_k.fingerprint =
       snapshot::fingerprint(&params_N, n, &feature_borders, &target);` (TASK-05
       `fingerprint`, `pub`) so the fixture carries `iterations = N`, then
       `snapshot::write_atomic(P_resume, &snap_k)?;` (TASK-06 `write_atomic`,
       `pub`).
    4. `P_resume` is now byte-for-byte what a REAL `iterations=N` run interrupted
       exactly after its K-th tree would have written — VALID because the scoped
       path's first K trees are `iterations`-INDEPENDENT when
       `auto_learning_rate=false` (each tree depends only on the running `approx`
       + fixed params, not on `params.iterations`), so the K-run's trees/approx/rng
       equal an N-run's first K. Re-fingerprinting corrects the ONLY field that
       legitimately differs between a K-run and a mid-N-run snapshot.
    **Why `auto_learning_rate=false` is pinned (and why this is not the "naive
    K-run" the checker rejected).** With `auto_learning_rate=true` the effective LR
    is `autolr::guess(.., params.iterations)` (`boosting.rs:2348-2365`), so a K-run
    and an N-run's first K trees would use DIFFERENT learning rates and step 4's
    equivalence would break. The keystone fixtures therefore pin
    `auto_learning_rate=false`; an `auto_learning_rate=true` resume is not silently
    wrong (the fingerprint's field 13 catches an auto-LR change and errors) — its
    self-consistency is a later slice. This mechanism requires NO prod-code change
    beyond TASK-06's write hook + TASK-04/05's `pub` `decode`/`fingerprint`/
    `write_atomic`; it adds NO interruption escape-hatch to `train_inner`, so
    SPEC §7's additive-only Impact Scope is preserved. `[VERIFIED: Read
    boosting.rs:2348-2365 auto-LR depends on params.iterations]`
- **Green:** implement the pre-loop resume block + the loop-bound edit; the oracle
  passes exactly for all `(N,K)`.
- **Refactor:** unify the "capture current state → TrainSnapshot" with TASK-06's
  `capture_snapshot`; keep the resume block a single cohesive `if let`. Regression
  scope: `None` path byte-identical; full `cargo test -p cb-train`.
- **Validation:**
  - `cargo test -p cb-train --test snapshot_resume_selfconsistency_oracle_test`
  - `cargo test -p cb-train` (D-04 anchor)
  - `cargo clippy -p cb-train --lib --no-deps`
- **Completion evidence:** exact resume equality for `(4,1)/(4,2)/(6,3)` (all
  `auto_learning_rate=false`, fixture via the locked `refingerprinted_k_run`
  mechanism — no prod interruption hook added); `resumed_from==K`; no-file fresh;
  mismatch errs; full suite green.
- **Compat/rollback:** additive (resume block + `0`→`resume_from` bound); rollback
  = remove the resume block + restore the `0..iterations` bound.
- **Parallelization:** sequential after TASK-06/02.

---

## TASK-08 — Facade `train_with_snapshot` end-to-end (ORCH-03-S7)

- **Spec refs:** ORCH-03-S7. Primary failure reason: the public entry does not wire
  write+resume, or `resumed_from` does not reflect a genuine resume.
- **Goal / completion:** `pub fn train_with_snapshot<R: Runtime>(...) ->
  CbResult<(Model, usize)>` (SPEC §4) exists, re-exported from `cb_train`; it
  routes through `train_inner` with `Some(snapshot)`, empty eval sets / no ranking
  / empty `cat_columns`, and returns `(model, resume_from)`; the end-to-end test
  green.
- **Prerequisites:** TASK-07 (resume + `resume_from` surfaced).
- **Files:**
  - Modify: `crates/cb-train/src/boosting.rs` — add `train_with_snapshot`
    (mirror `train_with_eval_sets` `:2048`), passing `Some(snapshot)` into
    `train_inner` and returning `(Model, resume_from)`.
    **LOCKED design — PEEK-BASED `resume_from` (MAJOR resolution; decided now,
    NOT at Green).** `train_with_snapshot` computes `resume_from` ITSELF, BEFORE
    delegating, by peeking the snapshot file — it does NOT grow `train_inner`'s
    return type and does NOT touch `train_inner`'s 3 existing call sites:
      1. If `snapshot.snapshot_file` exists: `let stored =
         snapshot::decode(&fs::read(&snapshot.snapshot_file)?)?;`
         `snapshot::check_resume(stored.fingerprint,
         fingerprint(params, n, feature_borders, target))?;` (S4 mismatch ⇒ `Err`
         here, before any training) `let resume_from = stored.completed_iters;`
      2. Else `let resume_from = 0;`
      3. `let (model, _baked) = train_inner(.., Some(snapshot))?;` — `train_inner`
         independently re-decodes + restores internally (TASK-07); its return type
         stays `CbResult<(Model, BakedCtrData)>` UNCHANGED.
      4. `Ok((model, resume_from))`.
    The decode happens twice (facade peek + `train_inner` restore); on the scoped
    deterministic path that is a cheap, side-effect-free re-read — the chosen
    trade for a ZERO-blast-radius `train_inner` signature (the return-type-growth
    alternative would force a 3-tuple destructure at all 3 internal call sites, a
    larger, non-additive edit contradicting SPEC §7). `[VERIFIED: CODEGRAPH
    train_inner 3 call sites :2062/:2102/:2155; return CbResult<(Model, BakedCtrData)>]`
  - Modify: `crates/cb-train/src/lib.rs` — `pub use boosting::train_with_snapshot;`
    near the existing training re-exports.
  - (Tests: the S6 keystone test in TASK-07 already drives this entry; add a
    focused `train_with_snapshot` smoke test here if not covered.)
- **CodeGraph/Read evidence:** `train_with_eval_sets` entry shape `:2048-2076`;
  `lib.rs` training re-export site. `[VERIFIED: Read]`
- **Red:** a smoke test: `train_with_snapshot` on a fresh file returns
  `resumed_from==0` and writes a snapshot; a second call returns `resumed_from>0`
  and an identical model (this is the S7 end-to-end; may be the same test as the
  S6 keystone). Expected INITIAL failure: `train_with_snapshot` unresolved ⇒ build
  fails.
- **Green:** implement the entry + re-export; `resumed_from` reflects the resume.
- **Refactor:** `train_with_snapshot` is a thin wrapper over `train_inner` +
  the resume-point surfacing. Regression scope: full `cargo test -p cb-train`.
- **Validation:**
  - `cargo test -p cb-train --test snapshot_resume_selfconsistency_oracle_test`
  - `cargo test -p cb-train`
  - `cargo clippy -p cb-train --lib --no-deps`
- **Completion evidence:** end-to-end fresh-then-resume identical model;
  `resumed_from` correct; re-export compiles; full suite green.
- **Compat/rollback:** additive public entry; rollback = remove it + the re-export.
- **Parallelization:** sequential after TASK-07.

---

## 3. Cross-cutting guardrails (apply to every code task)

- **Clippy gate, not build:** `unwrap`/`expect`/`panic`/`indexing_slicing` are
  DENY in prod. Gate each prod change with `cargo clippy -p <crate> --lib
  --no-deps`. Integration tests carry `#![allow(clippy::unwrap_used,
  clippy::expect_used, clippy::panic, clippy::indexing_slicing)]` (as the existing
  `*_oracle_test.rs` do). `[VERIFIED: LOCAL MEMORY fstr03-plan]`
- **Test mount:** `snapshot_test.rs` MUST be mounted
  (`#[cfg(test)] #[path="snapshot_test.rs"] mod tests;`) or `cargo test` silently
  runs 0 unit tests. `[VERIFIED: LOCAL crates/cb-train/src/metrics.rs:714]`
- **D-04 no-regression:** the `None` snapshot path + the device/ranking/CTR/ordered
  arms of `train_inner` stay byte-identical; the FULL `cargo test -p cb-train`
  suite is the anchor (every oracle test drives one of the 4 callers). `metrics.rs`/
  the tree grower/apply path are read-only.
- **Dep direction:** `cb-model` is NEVER imported from `cb-train` (it is a dev-dep
  only; the build edge is `cb-model → cb-train`). The snapshot serializer is
  cb-train-internal via `serde_json`. `[VERIFIED: LOCAL Cargo.toml:38-45,51]`
- **Determinism for the EXACT oracle:** slice 1 is the no-draw scalar CPU path
  (`draws_active==false`, `device_active==false`); its output is deterministic, so
  byte-exact self-consistency is a legitimate oracle (stronger than ≤1e-5).

## 4. Unresolved blockers / assumptions

1. **GATING SPIKE (TASK-01) must return GO before any code task.** The whole slice
   rests on the claim that the scoped loop-carried mutable-state set is exactly
   `{approx, trees, rng}`. The SPEC's §1 audit (performed from the actual source)
   judges GO **likely**, but TASK-01 must confirm it on the live code. **If TASK-01
   is NO-GO, STOP and escalate to a standalone design-spike phase (trainer-state
   extraction) — do NOT run TASK-02+.** This is the single most important gate.
2. **`resume_from` threading out of `train_inner` (TASK-08).** RESOLVED / LOCKED
   (MAJOR review finding): the PEEK-BASED design — `train_with_snapshot` decodes
   the snapshot file's `completed_iters` ITSELF before delegating and returns it;
   `train_inner`'s return type is UNCHANGED and its 3 internal call sites are
   untouched. The return-type-growth alternative is explicitly rejected (it would
   force a 3-tuple destructure at all 3 call sites — a larger, non-additive edit).
   No longer open.
3. **Producing the intermediate K-tree file for the S6 oracle (TASK-07).**
   RESOLVED / LOCKED (CRITICAL-2 review finding): the `refingerprinted_k_run`
   mechanism — run `iterations=K` with interval ZERO into a temp file, `decode` it,
   overwrite ONLY its `fingerprint` field with `fingerprint(&params_N, ..)`, and
   `write_atomic` it to the resume path. Valid because the scoped path's first K
   trees are `iterations`-independent when `auto_learning_rate=false` (the keystone
   fixtures pin it). Uses PUBLIC `decode`/`fingerprint`/`write_atomic` only; adds
   NO prod-code interruption primitive beyond TASK-06's write hook. No longer open.
4. **`Lcg32` field visibility for `from_raw_state` (TASK-02).** `raw_state` reads
   `self.r1.x`/`self.r1.c` (`rng.rs:222`), so the fields exist and are usable in a
   same-module struct literal; confirm exact names at Green. Not a blocker.
5. **Fingerprint hash stability (TASK-05).** Use an explicit stable hash over
   `to_bits()` byte reprs, NOT `std::DefaultHasher` (toolchain-unstable). Documented
   field order is part of the contract. Resolved at TASK-05 Green.
6. No PageIndex write target confirmed for this corpus (SPEC frontmatter
   `pageindex_pending`); the SPEC under `.planning/plans/snapshot-resume/SPEC.md`
   is the effective spec store. Not a planning blocker.

No requirement conflicts detected. No production code was authored.
