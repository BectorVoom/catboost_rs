# Plan Check Result

**Verdict:** ISSUES_FOUND
**Goal:** Implement `staged_predict` (per-tree-prefix cumulative raw prediction) for scalar oblivious float-only models, ≤1e-5 oracle vs `model.staged_predict`; acceptance = SP-01..SP-04 + §6 scenarios.
**Plan:** `.planning/plans/staged-predict/PLAN.md` (SPEC `.planning/plans/staged-predict/SPEC.md`)

## Summary

The plan is well-structured, TDD-shaped, and correct on almost every structural claim I re-verified through CodeGraph: the additive `predict_raw_staged` reusing private `leaf_index_for` is sound and keeps the existing scalar apply byte-identical; the apply.rs test-mount claim is accurate; the facade guard fields are all `pub`; the SP-02 schedule matches the SPEC's stated rule; and R1 (empirical upstream stage-index confirmation) is an explicit MANDATORY T4 step with a STOP-and-correct-T2 gate rather than an assumption. **One CRITICAL blocker sinks the "additive only" claim:** the new `CatBoostError::UnsupportedModel` variant is NOT additive — it breaks an exhaustive `match` in the `catboost-rs-py` crate (`to_pyerr`), and none of the plan's validation commands compile that crate, so the plan would mark T3 "done" with the workspace not building. Two MINORs noted.

## Specification Coverage

- [x] SP-01 single-stage prefix == truncated apply → T1-STAGED-CORE. `predict_raw_one` (`crates/cb-model/src/apply.rs:318`) sums `bias + Σ tree.leaf_values[leaf_index_for(...)]`; T1 asserts prefix sum and `k==T` equals `predict_raw`. Sound.
- [x] SP-02 stage schedule → T2-STAGED-SCHEDULE. `end==0⇒len`, `period==0⇒1`, step from `start+step` while `<end` then always push `end`, `start>=end⇒empty`. `{3,6,9,10}` for T=10/period=3 is correct under the stated rule.
- [x] SP-03 facade wiring + feature-count guard + scalar-oblivious guard → T3-STAGED-FACADE. Guard reads `approx_dimension`, `non_symmetric_trees`, `region_trees`, `ctr_data` (all `pub`, verified) via `as_canonical()` BEFORE `feature_columns`. Correct.
- [x] SP-04 oracle parity → T4-STAGED-ORACLE, with the R1 empirical-confirmation step made explicit and mandatory.
- [~] §6 scenario 3 (reject non-scalar/non-oblivious/CTR) → guarded at facade in T3. Correct at the facade, but the pub free fn `predict_raw_staged` itself is unguarded (see MINOR-1).

## CodeGraph Evidence

- `predict_raw` `crates/cb-model/src/apply.rs:370` → `predict_raw_cat:386` → (`approx_dimension>1`) `predict_raw_multi_cat:515` → `leaf_index_for:208`.
  - `leaf_index_for:208` is a private in-module fn; 4 callers all in apply.rs. A new sibling `predict_raw_staged` can call it with no visibility change. **Additive-reuse claim: CONFIRMED.**
  - `predict_raw_cat:394` routes scalar (`approx_dimension<=1`) through byte-identical `predict_raw_one`; multi through `predict_raw_multi_cat`. Adding a new fn does not touch these. **Byte-identical claim for predict_raw/predict_raw_cat/predict_raw_one: CONFIRMED (additive only).**
  - NaN-pad reference (`col.get(obj).copied().unwrap_or(f32::NAN)`) is real at apply.rs:404-407. Plan's T1 Green cite is accurate.
- `sum_f64` `crates/cb-core/src/reduction.rs:32` is the order-locked accumulator (130 callers). Reuse is correct.
- apply.rs test mount tail: `#[cfg(test)] #[path = "region_apply_test.rs"] mod region_apply_test;` at **:812-814** exactly; there is no `apply_test.rs`/`staged_predict_test.rs` sibling. **New mount block IS required or 0 tests run — CONFIRMED.**
- `cb_model::Model` fields (`crates/cb-model/src/model.rs`): `oblivious_trees:274`, `non_symmetric_trees:281`, `region_trees:287`, `bias:289`, `float_feature_borders:293`, `ctr_data:298`, `approx_dimension:306` — **all `pub`. Facade guard via `as_canonical()` (`crates/catboost-rs/src/model.rs:45`) is sound. CONFIRMED.**
- Facade `feature_columns` (`crates/catboost-rs/src/model.rs:60`) checks float count → `FeatureMismatch`, narrows to `Vec<Vec<f32>>`. `use cb_model::{...}` block at :17-21 — adding `predict_raw_staged` is additive. CONFIRMED.
- `CatBoostError` `crates/catboost-rs/src/error.rs:33` — variants Train/Model/Io/Deserialize/SchemaVersion/FeatureMismatch/PartialDependence/Export. **NO `#[non_exhaustive]` attribute (line 32 is a bare `#[derive(...)]`).** The doc at :30-31 ("downstream matches should remain robust") is prose, not a compiler guarantee.
- **Blast radius of adding a variant — the CRITICAL:** `to_pyerr` at `crates/catboost-rs-py/src/errors.rs:113-135` is `match err { ... }` enumerating every variant with **NO `_` wildcard arm**. `PyCbError`/`to_pyerr` have 9 call sites across the py crate. `error_test.rs:14` DOES have an `other => panic!` catch-all (safe); `errors_test.rs` only constructs variants (safe). So the sole break site is `to_pyerr`.

## Issues

### [CRITICAL] New `CatBoostError::UnsupportedModel` variant breaks the exhaustive `to_pyerr` match in `catboost-rs-py`; plan validation cannot detect it

- **Plan location:** T3-STAGED-FACADE — "Modify `error.rs` — add one additive variant `CatBoostError::UnsupportedModel(String)` … Downstream `match`es are already documented as non-exhaustive-tolerant (`error.rs:30-31`)." Also the Consistency-check line "additive fn + additive re-export + additive facade method + additive error variant only."
- **Requirement:** SP-03 / §6 scenario 3 (return a typed error for non-scalar-oblivious) without breaking the existing system.
- **Evidence:** `CatBoostError` (`crates/catboost-rs/src/error.rs:32-33`) is NOT `#[non_exhaustive]`. `to_pyerr` (`crates/catboost-rs-py/src/errors.rs:113-135`) is an exhaustive `match` over the enum with no wildcard. The "documented as non-exhaustive-tolerant" claim rests on a prose doc comment (error.rs:30-31), which the Rust compiler does not honor. CodeGraph confirms `PyCbError`/`to_pyerr` are the py crate's single error chokepoint (9 callers).
- **Failure scenario:** Adding the variant makes `to_pyerr`'s match non-exhaustive → rustc error E0004 in `catboost-rs-py`. The plan's T3 validation commands are `cargo test -p catboost-rs` and `cargo clippy -p catboost-rs --lib --no-deps` — neither compiles `catboost-rs-py`. T3 would be marked "done" while `cargo build`/`cargo test` on the workspace (or the py crate) fails to compile.
- **Impact:** Workspace / Python-extension build broken; a real regression masked by insufficient validation. Contradicts the plan's own "additive only / no existing path breaks" guarantee.
- **Required revision:**
  1. T3 file list MUST add `crates/catboost-rs-py/src/errors.rs`: add a `FacadeError::UnsupportedModel(m) => CatBoostValueError::new_err(m.clone())` arm to `to_pyerr` (semantically a bad-input-model value error, mirroring `FeatureMismatch`/`PartialDependence`).
  2. Add a matching assertion in `crates/catboost-rs-py/src/errors_test.rs` (variant → `CatBoostValueError`).
  3. T3 validation commands MUST include a build/clippy of the py crate, e.g. `cargo build -p catboost-rs-py` (or `cargo clippy -p catboost-rs-py --no-deps`), so the break is caught.
  4. Update the SPEC/PLAN "additive only" wording: the error variant touches the py error taxonomy, not purely additive. (This edit stays disjoint from T4's files, so Wave-3 parallelization remains valid.)

### [MINOR] Pub `predict_raw_staged` free fn is unguarded — silent wrong output for a direct cb-model caller on a non-oblivious / multi-dim model

- **Plan location:** T1/T2 (the pub fn is re-exported in `lib.rs`); guard lives only at the facade (T3).
- **Evidence:** `predict_raw_staged` iterates only `oblivious_trees` and adds `bias` once; it ignores `non_symmetric_trees`/`region_trees` and does not multiplex `approx_dimension`. Unlike the facade, a direct `cb_model::predict_raw_staged` caller has no guard.
- **Failure scenario:** A non-facade caller invokes it on a non-symmetric model → all stages return just `bias` (empty `oblivious_trees`); on a multi-dim model → dimensions silently dropped.
- **Impact:** Latent footgun on a public surface; no build break. The SPEC §4 doc says "SCALAR oblivious models" but does not enforce it.
- **Required revision:** Either keep the guard at the facade (accepted, matching the `predict_raw` vs `predict_raw_multi` split) AND strengthen the SPEC §4 doc-comment to a `# Panics`/`# Warning`-style contract note that the fn assumes a scalar-oblivious model and yields undefined values otherwise; or return `Vec::new()`/document behavior for `approx_dimension>1`. Low priority.

### [MINOR] R1-driven T2 correction can invalidate T3 built in the same wave

- **Plan location:** Waves — T3 ∥ T4; T4-R1 may STOP-and-correct T2's `stage_counts`.
- **Evidence:** T3 and T4 both depend on T2. If T4's empirical R1 step forces a T2 `stage_counts` change, T3 (built against the pre-correction T2) is not re-validated in the stated flow.
- **Failure scenario:** Low: T3's principal assertion is `stages.last() == predict(pool)` (the full-tree stage is invariant under first-stage/inclusion changes), so most schedule corrections won't break T3. A change to the last-stage inclusion rule could, though.
- **Impact:** Minor re-work risk, not a build break.
- **Required revision:** Add a note to T3/T4: if R1 corrects T2, re-run `cargo test -p catboost-rs` (T3) after the correction lands.

## Implementation Order Review

1. T1-STAGED-CORE (add `predict_raw_staged` single-stage + test mount + lib re-export) — prerequisites: none. Valid.
2. T2-STAGED-SCHEDULE (generalize same fn to full schedule) — after T1 (same fn/file). Valid.
3. Wave 3, parallel, both after T2:
   - T3-STAGED-FACADE (facade method + guard + error variant) — **file list must expand to include `crates/catboost-rs-py/src/errors.rs` + `errors_test.rs`** (CRITICAL). Still disjoint from T4's files, so parallelism holds.
   - T4-STAGED-ORACLE (fixtures + oracle test + R1 confirmation, may correct T2).
   Dependency graph T1→T2→{T3,T4} is acyclic. Only order-related change required is the CRITICAL's expanded T3 scope and the MINOR re-run note.

## Potential Bugs

- Exhaustive-match break in `to_pyerr` on variant addition (CRITICAL, above) — trigger: compile; mitigation: add arm + py-crate build in validation.
- Unguarded pub `predict_raw_staged` (MINOR-1) — trigger: direct cb-model caller on non-oblivious/multi-dim model; mitigation: doc contract.
- Stage-schedule edge cases the plan already handles: `period==0⇒1` (avoids zero-step infinite loop), `start>=end⇒empty`, `end==0⇒len`, saturating arithmetic (no overflow / no raw indexing under deny-lints). Verified against SP-02 Green text — adequate.

## Required Plan Revisions

1. (CRITICAL) T3: add `crates/catboost-rs-py/src/errors.rs` to the file list; add a `to_pyerr` arm for `UnsupportedModel` (→ `CatBoostValueError`) plus an `errors_test.rs` assertion; add `cargo build -p catboost-rs-py` (or clippy) to T3 validation. Drop/qualify the "downstream matches are non-exhaustive-tolerant" claim and the unqualified "additive error variant only" statement.
2. (MINOR) Strengthen SPEC §4 `predict_raw_staged` doc to state the scalar-oblivious precondition and its undefined behavior otherwise.
3. (MINOR) Add a T3/T4 note: re-run T3 facade tests if T4-R1 corrects T2's `stage_counts`.

## Unverified Items

- Upstream `model.staged_predict` exact first-stage/inclusion indexing (SPEC R1) is UNVERIFIED at plan time (sparse checkout / TreeFinder unavailable). This is acceptable: the plan routes it through the MANDATORY T4-R1 empirical-confirmation step (fit → call `staged_predict` → compare stage j to `predict(ntree_end=c_j)` → record `stage_tree_counts` in config.json → STOP-and-correct T2 if different) rather than assuming it. No further action required beyond executing that step.
- The exact in-test `Model` constructor pattern (T1) and non-scalar fixture source (T3) are deferred to "confirm via CodeGraph before writing" — acceptable, non-blocking.

---

# Plan Check Result — PASS 2 (re-review after revision)

**Verdict:** PASS
**Goal:** Implement `staged_predict` (per-tree-prefix cumulative raw prediction) for scalar oblivious float-only models, ≤1e-5 oracle vs `model.staged_predict`; acceptance = SP-01..SP-04 + §6 scenarios.
**Plan:** `.planning/plans/staged-predict/PLAN.md` (SPEC `.planning/plans/staged-predict/SPEC.md`)

## Summary

The revision fully resolves the PASS-1 CRITICAL and both MINORs, and introduced no new break. The formerly-hidden exhaustive-match hazard is now an explicit, tested part of T3, gated by a py-crate build command; the unguarded `predict_raw_staged` contract is documented in SPEC §4; and the R1→T2→T3 re-run coupling is captured. All structural claims re-verified through CodeGraph against current on-disk source.

## CRITICAL resolution — CONFIRMED

`[CRITICAL] New CatBoostError::UnsupportedModel breaks the exhaustive to_pyerr match` is resolved on all three required points, each re-verified:

- (a) **T3 file list now includes the py edits.** PLAN T3 lists `crates/catboost-rs-py/src/errors.rs` (add the `to_pyerr` arm) and `crates/catboost-rs-py/src/errors_test.rs` (add the mapping assertion) as mandatory edits, not optional.
- (b) **The `to_pyerr` arm is specified exactly.** PLAN T3: `FacadeError::UnsupportedModel(m) => CatBoostValueError::new_err(m.clone())`.
  - Re-verified via CodeGraph: `to_pyerr` (`crates/catboost-rs-py/src/errors.rs:113-136`) is `match err { ... }` over `FacadeError` with **no `_` wildcard** — every arm is an explicit variant; the only nested match is `Export(e) => match e { OnnxExportError… }`. Adding an enum variant WOULD make it non-exhaustive (E0004). Break site confirmed real.
  - `FacadeError` is `use catboost_rs::CatBoostError as FacadeError;` (`errors.rs:24`, and identically in `errors_test.rs:7`) — the plan's `FacadeError::UnsupportedModel` names the same enum the plan adds the variant to. Alias CONFIRMED.
  - Variant **shape** matches: the plan adds `CatBoostError::UnsupportedModel(String)` with `#[error("unsupported model: {0}")]`. The existing String-tuple variants `Deserialize(String)`/`SchemaVersion(String)`/`FeatureMismatch(String)` (`crates/catboost-rs/src/error.rs:58,63,70`) are the exact precedent; the planned arm's `m.clone()` is valid for a `String` payload and mirrors `FeatureMismatch(m) => CatBoostValueError::new_err(m.clone())` (`errors.rs:115`) byte-for-byte in form.
  - **`CatBoostError` is NOT `#[non_exhaustive]`** — re-verified: `crates/catboost-rs/src/error.rs:32` is a bare `#[derive(Debug, thiserror::Error)]`; the "downstream matches should remain robust" text (`:30-31`) is prose only. So the compiler break is genuine and the py-crate edit is mandatory, exactly as the revised plan now states.
  - Taxonomy choice sound: `UnsupportedModel → CatBoostValueError` mirrors the guard-rejection variants `PartialDependence` (`errors.rs:124`) and the four `Export` sub-variants (`:125-131`), all of which map to `CatBoostValueError`. The planned `errors_test.rs` assertion mirrors the existing `feature_mismatch_maps_to_value_error` shape (`errors_test.rs:16-23`), and `CatBoostValueError` is already imported there (`:12`).
- (c) **T3 validation now compiles the py crate.** PLAN adds `cargo build -p catboost-rs-py` (and `cargo test -p catboost-rs-py --lib`) to T3 and to the Validation-commands section, with an explicit note that `cargo test -p catboost-rs` never compiles the py crate. The E0004 break would now be caught before T3 is marked done.

**No missed second break site.** Swept every `match` over the error type in the facade + py crates: `to_pyerr` (`errors.rs:114`) is the ONLY exhaustive match. Facade `error_test.rs` uses `other => panic!` (`:18`) plus `matches!` (safe); `errors_test.rs` only constructs variants (safe). CodeGraph blast radius for `CatBoostError` (23 callers) and `PyCbError`/`to_pyerr` (9/14 callers) surfaced no other exhaustive consumer. No pre-existing `UnsupportedModel`/`staged_predict`/`predict_raw_staged` symbol exists anywhere (clean additive surface).

## MINOR resolutions — CONFIRMED

- **MINOR-1 (unguarded pub fn).** SPEC §4 now carries a `# Contract (UNGUARDED — caller's responsibility)` block stating `predict_raw_staged` does NOT validate model shape, returns SILENTLY WRONG output on multi-dim/non-symmetric/Region/CTR models, and that the scalar-oblivious guard lives at the facade (SP-03). The footgun is documented; guard placement matches the existing `predict_raw` vs `predict_raw_multi` split. Resolved.
- **MINOR-2 (R1→T2 may invalidate T3).** PLAN T3 now has a "Re-run coupling" note (re-run `cargo test -p catboost-rs` if T4-R1 corrects T2's `stage_counts`), echoed in the Wave-3 description. Resolved.

## No-regression checks — CONFIRMED

- **Wave-3 parallelism still valid.** T3 touches `catboost-rs/src/{model.rs,error.rs}`, `catboost-rs-py/src/errors{,_test}.rs`, and a new `catboost-rs/tests/` file; T4 touches only `cb-oracle/fixtures/` + a new `cb-model/tests/` file. The newly-added py error files are disjoint from every T4 path — no write conflict; T1→T2→{T3,T4} remains acyclic.
- **predict_raw / predict_raw_cat / predict_raw_one byte-identical.** The change is a new sibling fn + additive `lib.rs` re-export + additive facade method + additive error variant (now with its paired py arm). CodeGraph confirms the existing apply chain (`predict_raw:370 → predict_raw_cat:386 → predict_raw_one / predict_raw_multi_cat → leaf_index_for:208`) is untouched; `leaf_index_for` stays private and in-module.
- **SP-02 schedule + R1 empirical-confirm intact.** SPEC SP-02 (`{3,6,9,10}` for T=10/period=3, `end==0⇒len`, `period==0⇒1`, `start>=end⇒empty`) and the MANDATORY T4-R1 step (fit → `model.staged_predict` → compare stage j to `predict(ntree_end=c_j)` → record `stage_tree_counts` → STOP-and-correct T2 if different) are both present and unchanged by the revision.

## Residual non-blocking observations (no verdict impact)

- The `to_pyerr` doc-comment enumeration (`errors.rs:96-112`) and the `errors_test.rs` module-doc "each of the six … variants" (`:1`) are prose that is already stale relative to the current 8 variants; adding a 9th makes them slightly staler. Pre-existing drift, cosmetic, not introduced by this plan — optionally add a one-line doc mention when writing the arm.

## Unverified Items

- Upstream `model.staged_predict` exact first-stage/inclusion indexing (SPEC R1) remains UNVERIFIED at plan time (sparse checkout / TreeFinder unavailable) — acceptably routed through the MANDATORY T4-R1 empirical-confirmation step with a STOP-and-correct-T2 gate. Unchanged from PASS 1; not a plan-time blocker.
- In-test scalar-oblivious `Model` constructor (T1) and non-scalar fixture source (T3) are deferred to "confirm via CodeGraph before writing" — acceptable, non-blocking.

**Final verdict: PASS.**
