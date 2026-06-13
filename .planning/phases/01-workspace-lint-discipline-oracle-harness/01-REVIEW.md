---
phase: 01-workspace-lint-discipline-oracle-harness
reviewed: 2026-06-13T00:00:00Z
depth: standard
files_reviewed: 38
files_reviewed_list:
  - .github/workflows/ci.yml
  - crates/catboost-rs/Cargo.toml
  - crates/catboost-rs/src/lib.rs
  - crates/cb-backend/Cargo.toml
  - crates/cb-backend/src/lib.rs
  - crates/cb-compute/Cargo.toml
  - crates/cb-compute/src/lib.rs
  - crates/cb-core/Cargo.toml
  - crates/cb-core/src/error.rs
  - crates/cb-core/src/error_test.rs
  - crates/cb-core/src/lib.rs
  - crates/cb-core/src/rng.rs
  - crates/cb-core/src/rng_test.rs
  - crates/cb-data/Cargo.toml
  - crates/cb-data/src/lib.rs
  - crates/cb-model/Cargo.toml
  - crates/cb-model/src/lib.rs
  - crates/cb-oracle/Cargo.toml
  - crates/cb-oracle/fixtures/skeleton/config.json
  - crates/cb-oracle/generator/README.md
  - crates/cb-oracle/generator/gen_fixtures.py
  - crates/cb-oracle/generator/gen_inputs.py
  - crates/cb-oracle/generator/requirements.txt
  - crates/cb-oracle/src/bin/write_skeleton.rs
  - crates/cb-oracle/src/compare.rs
  - crates/cb-oracle/src/compare_test.rs
  - crates/cb-oracle/src/error.rs
  - crates/cb-oracle/src/fixture.rs
  - crates/cb-oracle/src/fixture_test.rs
  - crates/cb-oracle/src/lib.rs
  - crates/cb-oracle/tests/per_stage_oracle_test.rs
  - crates/cb-oracle/tests/skeleton_oracle_test.rs
  - crates/cb-train/Cargo.toml
  - crates/cb-train/src/lib.rs
  - scripts/check-no-anyhow.sh
  - scripts/check-source-test-separation.sh
findings:
  critical: 0
  warning: 5
  info: 6
  total: 11
status: issues-found
---

# Phase 1: Code Review Report

**Reviewed:** 2026-06-13
**Depth:** standard
**Files Reviewed:** 38
**Status:** issues_found

## Summary

Phase 1 stands up the 8-crate workspace, the clippy deny table, the anyhow gate, and the oracle parity harness. The correctness-critical code is in good shape: the `TFastRng64` PRNG port in `crates/cb-core/src/rng.rs` was traced line-by-line against the vendored C++ sources (`fast.h`, `fast.cpp`, `lcg_engine.cpp`, `common_ops.h`) and is **bit-for-bit faithful** — the PCG mixer, LCG iterate, `from_seed` derivation order, `FixSeq` rule, `GenRand` high/low ordering, `LcgAdvance` binary exponentiation, and the rejection-sampling accept condition (`rand < randmax`) all match upstream, and every arithmetic op uses `wrapping_*` so no debug-overflow panic is possible. `try_uniform` cannot panic and cannot livelock for any `bound > 0`.

The defects found are not in the PRNG. The most material is a **NaN/inf hole in the parity comparator** (`compare.rs`): a NaN actual silently *passes* the 1e-5 gate, which directly undermines the harness's entire reason to exist in later phases. The remaining issues are gate-coverage gaps (the anyhow scanner skips the published `catboost-rs` crate; CI never runs the source/test-separation script's sibling clippy variants for non-default backend features), an unused generator variable, and a fixture-schema inconsistency. No Critical (security/data-loss/crash) findings.

## Warnings

### WR-01: Parity comparator silently passes NaN (and matching infinities)

**File:** `crates/cb-oracle/src/compare.rs:40-41`
**Issue:** The divergence test is `let diff = (e - a).abs(); if diff > tol`. When either `e` or `a` is `NaN`, `(e - a).abs()` is `NaN`, and `NaN > tol` evaluates to `false` — so the pair is treated as *within tolerance* and the comparator returns `Ok(())`. This is the core parity primitive every later phase (cb-train P3 staged-approx/predictions, cb-model P4 splits/leaf-values) reuses to detect drift against the CatBoost oracle. A Rust-computed `actual` that has gone NaN (the classic symptom of a diverged gradient, a log-of-zero, or an uninitialized leaf) would be reported as a **PASS**, defeating the harness. The same hole admits `+inf`/`-inf`: if expected and actual are both `+inf`, `diff` is `NaN` and passes; if they differ in sign, `diff` is `+inf` and is correctly caught — so the behavior is inconsistent. The module doc claims "the single audited parity primitive," so this is the one place non-finite handling must be explicit. There is currently no test exercising a NaN/inf input (see `compare_test.rs`), so the hole is unguarded.
**Fix:** Treat any non-finite participation as a divergence rather than relying on the `>` comparison. For example:
```rust
let diff = (e - a).abs();
// A NaN diff (NaN input, or inf - inf) must never silently pass; a finite
// pair only passes when |diff| <= tol.
if diff.is_nan() || diff > tol {
    return Err(OracleError::Diverged {
        index,
        expected: *e,
        actual: *a,
        diff,
    });
}
```
Add a `compare_test.rs` case asserting `assert_abs_close(&[f64::NAN], &[f64::NAN], 1e-5)` and `(&[1.0], &[f64::NAN], 1e-5)` both return `Err(Diverged)`.

### WR-02: anyhow ban gate does not scan the published `catboost-rs` facade crate

**File:** `scripts/check-no-anyhow.sh:13-20`
**Issue:** `CORE_DIRS` enumerates exactly six crates (`cb-core`, `cb-data`, `cb-compute`, `cb-backend`, `cb-train`, `cb-model`) and omits `crates/catboost-rs` — the single published Builder facade (D-04). The facade is a production library crate (CLAUDE.md: `thiserror` for libraries, no `anyhow` in core libs), and it is the crate most likely to be tempted toward `anyhow` for ergonomic top-level error glue in Phase 4. As written, an `anyhow` dependency or `use anyhow::...` added to `catboost-rs/src/` would pass the D-14 backstop undetected. The structural ban (absence from `[dependencies]`) is the primary defense, but this script is explicitly the belt-and-suspenders gate and currently has a hole exactly where it matters most.
**Fix:** Add `"crates/catboost-rs"` to `CORE_DIRS`. The existing `[ -d "$dir" ] || continue` guard already tolerates the crate's absence in earlier waves, so this is safe to add now.

### WR-03: CI compiles non-default backends but never lint-gates or tests them

**File:** `.github/workflows/ci.yml:30-43`
**Issue:** The workflow runs `cargo build -p cb-backend --no-default-features --features {wgpu,cuda,rocm}` (build only), then runs `cargo clippy --workspace --lib` and `cargo test --workspace` using **default features only** (`cpu`). The clippy deny table (`unwrap_used`/`panic`/`indexing_slicing`) and the test suite therefore never see the `wgpu`/`cuda`/`rocm` `cfg` arms of `cb-backend/src/lib.rs`. Today those arms are trivial (`pub type SelectedRuntime = ()`), so the exposure is low — but the CI shape established here is the template later phases inherit, and once Phase 7 puts real runtime code behind those `cfg` arms, lint/test coverage will silently exclude it. A `clippy::unwrap_used` violation inside a `#[cfg(feature = "rocm")]` block would pass CI green.
**Fix:** At minimum add a clippy pass per non-default backend, e.g. `cargo clippy -p cb-backend --no-default-features --features wgpu --lib -- -D warnings` (and cuda/rocm). Note that the per-backend lint must remain `cb-backend`-scoped, not `--workspace`, because `--no-default-features` at the workspace level would disable features other crates rely on.

### WR-04: `compare_stage` `Err(other)` arm is structurally unreachable yet silently forwards mis-tagged errors

**File:** `crates/cb-oracle/src/compare.rs:89-91`
**Issue:** `compare_stage` calls `assert_abs_close`, which by construction only ever returns `Ok`, `LengthMismatch`, or `Diverged`. The `Err(other) => Err(other)` catch-all is dead today. The comment frames it as forward-compatibility, but its behavior is wrong for that purpose: if a *future* `assert_abs_close` grows a new untagged variant, `compare_stage` would forward it **without** stage-tagging — the exact failure mode (an untagged error escaping the per-stage API) the `StageDiverged`/`StageLengthMismatch` design exists to prevent (INFRA-04). A silent passthrough is strictly worse than a compile error here.
**Fix:** Make the exhaustiveness explicit so adding a variant to the upstream primitive forces a deliberate decision at this call site rather than silently leaking. Either match `Ok(()) => Ok(())` and the two known `Err` variants and let the compiler flag a non-exhaustive match when a new variant lands, or replace the catch-all with `Err(other) => unreachable!()` guarded by a `// LINT-EXEMPT` justification — but the preferred fix is to drop the catch-all and rely on exhaustiveness so this stays a compile-time gate.

### WR-05: `write_skeleton` and `compare_test` hardcode the same skeleton values with no shared source of truth

**File:** `crates/cb-oracle/src/bin/write_skeleton.rs:25`, `crates/cb-oracle/src/fixture_test.rs:10`, `crates/cb-oracle/tests/skeleton_oracle_test.rs:17`
**Issue:** `SKELETON_VALUES = [0.0, 0.25, -1.5, 3.14159, 2.71828]` is duplicated verbatim across three files, and the generator's own doc-comment (`write_skeleton.rs:23-24`) admits they "MUST stay in sync" by convention only. The committed `predictions.npy` was produced from the generator copy; the two test copies are independent literals. If anyone edits the generator constant and regenerates the fixture without updating both test copies (or vice-versa), the fixture-vs-reference test passes against a stale literal or fails confusingly. This is exactly the manual-sync drift the source/test-separation discipline elsewhere in this phase is designed to avoid.
**Fix:** Hoist the canonical values into one location the binary, the unit test, and the integration test all reference. A `pub(crate) const SKELETON_VALUES` in `fixture.rs` (or a small `skeleton` module) consumed by `write_skeleton.rs`, `fixture_test.rs`, and re-exported for the integration test removes the three-way duplication. (Note: the integration test in `tests/` is a separate compilation unit and can only see the public API, so the shared constant must be `pub` or the integration test must load-and-compare without a hardcoded literal.)

## Info

### IN-01: Unused `weights` symbol shadowed across generator functions

**File:** `crates/cb-oracle/generator/gen_inputs.py:57,85,123`
**Issue:** Each generator function rebinds a local `weights` array used only on the next line. This is fine functionally, but `numeric_tiny` and `grouped_ranking` reuse the identical name `weights` for different vectors while `numeric_categorical` also introduces `cat_effect`; the repeated single-purpose locals could be inlined or named per-dataset (`numeric_tiny_weights`) for auditability, since these constants define the frozen oracle baseline and reviewers diff them in PRs.
**Fix:** Optional. Either inline the `x @ np.array([...])` expression or give each weight vector a dataset-qualified name so a fixture-baseline diff is unambiguous about which dataset moved.

### IN-02: Input-corpus `config.json` schema is incompatible with `FixtureConfig`

**File:** `crates/cb-oracle/generator/gen_inputs.py:62-72`, `crates/cb-oracle/src/fixture.rs:28-36`
**Issue:** `FixtureConfig` requires `seed`, `catboost_version`, and `thread_count`. The input-corpus `config.json` files written by `gen_inputs.py` contain `seed` but **not** `catboost_version` or `thread_count` (those datasets are pre-training inputs, so the fields are absent). Calling `load_config` on an input-corpus config would fail with a missing-field `serde_json::Error`. No current test does this — `load_config` is only exercised against the skeleton config — so it is latent, but the single `FixtureConfig` type is presented as *the* config loader and will mislead a Phase 2 author who points it at an input config.
**Fix:** Either make `catboost_version`/`thread_count` `Option<...>` in `FixtureConfig`, or document that `load_config` targets the regression/skeleton output configs only and introduce a separate input-corpus config type when Phase 2 needs it.

### IN-03: `cb_result_ok_path_round_trips` uses `.unwrap()` where `assert_eq!` on the Result is cleaner

**File:** `crates/cb-core/src/error_test.rs:21`
**Issue:** `assert_eq!(ok.unwrap(), 42);` is allowed (test-scope lint exemption), but `assert_eq!(ok, Ok(42));` reads better and avoids the `unwrap()` entirely now that `CbError: PartialEq` is derived. Minor consistency nit only.
**Fix:** `assert_eq!(ok, Ok(42));`.

### IN-04: `RandMax()` ported as a literal `u64::MAX` rather than a named constant

**File:** `crates/cb-core/src/rng.rs:200-201`
**Issue:** The rejection-sampling bound uses `u64::MAX` directly with an inline comment tying it to upstream `RandMax()`. This is correct for `TFastRng64` (whose `RandMax()` is indeed `ui64` max), but the magic value is the kind of thing that silently breaks if a 32-bit variant is ever ported into the same module. Cosmetic; the comment mitigates it.
**Fix:** Optional — a `const RAND_MAX: u64 = u64::MAX;` documents intent at the type level.

### IN-05: `pcg_mix` rotation relies on `rotate_right` truncating `rot` mod 32

**File:** `crates/cb-core/src/rng.rs:35-37`
**Issue:** `rot = (x >> 59) as u32` yields a value in `0..=31`, and `u32::rotate_right` already takes its argument mod 32, so this is correct and matches C++ `RotateBitsRight`. Flagging only because the correctness depends on the `>> 59` guaranteeing `rot < 32`; a future edit to the shift constant would silently change semantics without a panic. No fix required — documented here for traceability.
**Fix:** None required; the existing C++-cross-referencing comment is adequate.

### IN-06: `.expect()` in `write_skeleton.rs` is intentional but the file is shipped, not gitignored

**File:** `crates/cb-oracle/src/bin/write_skeleton.rs:29,33`
**Issue:** The one-off generator binary legitimately allows `expect_used`/`panic` at file scope, which is appropriate for a developer tool. It lives under `src/bin/` and is therefore a permanent compiled target of `cb-oracle`, not a throwaway. That is acceptable, but it means the panic-on-error paths are part of the shipped crate's build surface. Low concern given it is never invoked at runtime by library consumers.
**Fix:** None required; optionally relocate to an `xtask`-style helper if the project later adopts one, to keep `src/bin` free of panic-allowed code.

---

_Reviewed: 2026-06-13_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
