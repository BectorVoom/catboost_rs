---
phase: 01-workspace-lint-discipline-oracle-harness
verified: 2026-06-13T12:00:00Z
status: human_needed
score: 5/5 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Trigger a GitHub Actions run by pushing to the remote and confirm the CPU lane goes green end-to-end (all steps: build, cb-backend feature-gate builds, clippy, test, anyhow gate, source-test-separation gate)"
    expected: "All steps pass with green checkmarks; no GPU/ROCm/CUDA job appears"
    why_human: "GitHub Actions requires a live push to the remote repository; cannot be verified locally"
---

# Phase 1: Workspace, Lint Discipline & Oracle Harness — Verification Report

**Phase Goal:** The entire project scaffolding, parity-testing infrastructure, and the exact PRNG port exist so that every subsequent algorithm is born oracle-gated and lint-clean.
**Verified:** 2026-06-13T12:00:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                                                    | Status     | Evidence                                                                                                                               |
|----|--------------------------------------------------------------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------------------------------------------------------|
| 1  | Modular Cargo workspace builds with all backend crates stubbed and feature-gated (cpu/wgpu/cuda/rocm); cargo build and cargo clippy pass on the skeleton | ✓ VERIFIED | `cargo build --workspace` exits 0; `cargo clippy --workspace --lib -- -D warnings` exits 0; `cargo build -p cb-backend --no-default-features --features wgpu/cuda/rocm` all exit 0; 8-crate workspace confirmed |
| 2  | Library crates deny unwrap/expect/panic/indexing_slicing; CI check fails if anyhow appears in core library non-test code | ✓ VERIFIED | `Cargo.toml` `[workspace.lints.clippy]` sets all four restrictions to `"deny"`; `bash scripts/check-no-anyhow.sh` exits 0; `cb-oracle` Cargo.toml lists `anyhow` only under `[dev-dependencies]`; `catboost-rs` has no anyhow in `[dependencies]` (structural ban holds; script gap tracked as WARNING below) |
| 3  | Oracle harness runs against frozen committed upstream-CatBoost fixtures (pinned seed/version, thread_count=1) and can assert per-stage (borders, splits, leaf values, per-iteration approximants) to <= 1e-5 | ✓ VERIFIED | `regression_skeleton/` fixtures exist (borders.npy 33 elements, predictions.npy 50 elements, staged.npy 500 elements, model.json with oblivious_trees/leaf_values/splits/borders); `config.json` records seed=0, catboost_version=1.2.10, thread_count=1; `compare_stage` + `assert_abs_close` proven falsifiable (9e-6 -> Ok, 2e-5 -> Err) on REAL committed fixtures via `per_stage_oracle_test.rs`; 3 integration tests pass. Per-stage comparison against Rust-computed actuals is explicitly and intentionally deferred to P3/P4 per the plan scope note |
| 4  | TFastRng64 port reproduces C++ generator's raw bitstream exactly for a fixed seed (bitstream-oracle-validated)           | ✓ VERIFIED | `from_seed(17).gen_rand() == 14895365814383052362` passes (test3); Uniform(100) 20-value sequence `[37,43,76,17,12,87,60,4,83,47,57,81,28,45,66,74,18,17,18,75]` passes (test2); advance(100) parity passes; advance-boundaries tests pass; all 6 rng_test vectors green; all wrapping_* arithmetic confirmed in rng.rs |
| 5  | Source and test code are strictly separated (no inline #[cfg(test)] in production modules), enforced as a convention from the first commit | ✓ VERIFIED | `bash scripts/check-source-test-separation.sh` exits 0; production modules (`rng.rs`, `compare.rs`, `fixture.rs`, `error.rs`) contain no `#[cfg(test)] mod ... {` bodies; `lib.rs` files use only the allowed `#[cfg(test)] mod name_test;` declaration form; `scripts/check-source-test-separation.sh` wired into `ci.yml` |

**Score: 5/5 truths verified**

---

### Required Artifacts

| Artifact                                           | Expected                                              | Status     | Details                                                                          |
|----------------------------------------------------|-------------------------------------------------------|------------|----------------------------------------------------------------------------------|
| `Cargo.toml`                                       | Workspace root with `[workspace.lints.clippy]` deny table | ✓ VERIFIED | Contains unwrap_used/expect_used/panic/indexing_slicing = "deny"; `[workspace.dependencies]` present |
| `rust-toolchain.toml`                              | Pins stable channel + clippy/rustfmt                  | ✓ VERIFIED | channel = "stable", components = ["clippy", "rustfmt"]                           |
| `crates/cb-core/src/error.rs`                      | thiserror-derived CbError + CbResult alias            | ✓ VERIFIED | `pub type CbResult<T>`, `pub enum CbError` with InvalidBound + OutOfRange; derives Debug, Clone, PartialEq, Eq, thiserror::Error |
| `crates/cb-core/src/rng.rs`                        | TFastRng64 port with wrapping arithmetic, non-crypto doc | ✓ VERIFIED | PCG mixer, LcgAdvance, Lcg32 struct, from_seed, gen_rand, advance, try_uniform; all wrapping_*; non-crypto doc present |
| `crates/cb-core/src/rng_test.rs`                   | Oracle vectors from fast_ut.cpp including 14895365814383052362 | ✓ VERIFIED | All 6 vectors transcribed and passing |
| `crates/cb-oracle/src/compare.rs`                  | assert_abs_close + compare_stage + Stage enum         | ✓ VERIFIED | Both functions exported; Stage enum with 5 variants; zip/enumerate (no indexing); returns Result |
| `crates/cb-oracle/src/fixture.rs`                  | load_f64_vec + FixtureConfig + load_config            | ✓ VERIFIED | All three exported; ndarray-npy pipeline; serde Deserialize on FixtureConfig     |
| `crates/cb-oracle/src/bin/write_skeleton.rs`        | Rust-only fixture generator (no Python)               | ✓ VERIFIED | File exists; writes via ndarray_npy::write_npy; file-scope allow on restriction lints |
| `crates/cb-oracle/fixtures/skeleton/predictions.npy` | Committed f64 .npy fixture                           | ✓ VERIFIED | File exists; loaded and compared at 1e-5 in skeleton_oracle_test.rs              |
| `crates/cb-oracle/fixtures/regression_skeleton/`   | borders.npy, predictions.npy, staged.npy, model.json, config.json | ✓ VERIFIED | All present; borders=33 elements, predictions=50, staged=500, model.json has oblivious_trees/leaf_values/splits |
| `crates/cb-oracle/fixtures/inputs/numeric_tiny/`   | Frozen input corpus                                   | ✓ VERIFIED | X.npy, y.npy, config.json present                                                |
| `crates/cb-oracle/fixtures/inputs/numeric_categorical/` | Frozen input corpus with cat features            | ✓ VERIFIED | X.npy, y.npy, cat.npy, config.json present                                       |
| `crates/cb-oracle/fixtures/inputs/grouped_ranking/` | Frozen input corpus with group_id                    | ✓ VERIFIED | X.npy, y.npy, group_id.npy, config.json present                                  |
| `crates/cb-backend/src/lib.rs`                     | cfg-gated SelectedRuntime alias for cpu/wgpu/cuda/rocm | ✓ VERIFIED | Four cfg arms present; each defines `pub type SelectedRuntime = ()`; guarded with not(...) to prevent double-definition |
| `crates/cb-compute/Cargo.toml`                     | No cubecl dependency                                  | ✓ VERIFIED | No cubecl in [dependencies]; comment confirms D-03 prohibition                   |
| `crates/cb-oracle/generator/gen_fixtures.py`       | Python oracle generator with thread_count=1           | ✓ VERIFIED | File exists; thread_count=1 present on line 47 and 102; catboost==1.2.10 in requirements.txt |
| `scripts/check-no-anyhow.sh`                       | Executable D-14 backstop                              | ✓ VERIFIED | Exits 0; portable fixed-string scan; tolerates missing dirs; structural ban doc comment present |
| `scripts/check-source-test-separation.sh`          | INFRA-06 grep gate for inline #[cfg(test)]            | ✓ VERIFIED | Exits 0; distinguishes mod x_test; (allowed) from mod x { (forbidden); wired into ci.yml |
| `.github/workflows/ci.yml`                         | CPU CI lane with all gates, no GPU job                | ✓ VERIFIED | build + cb-backend feature-gate builds + --lib clippy + test + anyhow gate + separation gate; no GPU/ROCm/CUDA job |

---

### Key Link Verification

| From                                             | To                                                  | Via                               | Status     | Details                                                                                      |
|--------------------------------------------------|-----------------------------------------------------|-----------------------------------|------------|----------------------------------------------------------------------------------------------|
| `crates/cb-oracle/tests/skeleton_oracle_test.rs` | `cb_oracle::load_f64_vec`                           | reads predictions.npy then calls assert_abs_close | ✓ WIRED    | Test imports `cb_oracle::load_f64_vec` and `cb_oracle::assert_abs_close`; passes at 1e-5    |
| `Cargo.toml`                                     | `crates/cb-core/Cargo.toml`                         | members glob + workspace = true   | ✓ WIRED    | `members = ["crates/*"]` picks up all 8 crates; each Cargo.toml has `[lints] workspace = true` |
| `crates/cb-oracle/tests/per_stage_oracle_test.rs` | `regression_skeleton/borders.npy` + `predictions.npy` | load_f64_vec through ndarray-npy | ✓ WIRED    | Integration test loads both files; asserts non-empty; exercises compare_stage gate on real data |
| `.github/workflows/ci.yml`                       | `scripts/check-no-anyhow.sh`                        | bash step in CI lane              | ✓ WIRED    | Step present in ci.yml                                                                       |
| `.github/workflows/ci.yml`                       | `scripts/check-source-test-separation.sh`           | bash step in CI lane              | ✓ WIRED    | Step present in ci.yml                                                                       |
| `.github/workflows/ci.yml`                       | `cb-backend` feature-gate builds                    | cargo build -p cb-backend --features | ✓ WIRED | Three steps (wgpu, cuda, rocm) present in ci.yml                                             |
| `crates/cb-core/src/rng_test.rs`                 | `crates/cb-core/src/rng.rs::TFastRng64`             | asserts gen_rand/uniform/advance  | ✓ WIRED    | All 6 oracle vectors from fast_ut.cpp present and passing                                    |

---

### Data-Flow Trace (Level 4)

| Artifact                              | Data Variable         | Source                                      | Produces Real Data | Status      |
|---------------------------------------|-----------------------|---------------------------------------------|--------------------|-------------|
| `per_stage_oracle_test.rs`            | `borders`, `predictions` | `load_f64_vec` -> `ndarray_npy::read_npy` -> committed .npy | Yes (33 and 50 real f64 values from catboost==1.2.10) | ✓ FLOWING |
| `skeleton_oracle_test.rs`             | `actual`              | `load_f64_vec` -> committed `predictions.npy` (Rust-generated) | Yes (5 values bit-exactly matching write_skeleton) | ✓ FLOWING |
| `rng_test.rs`                         | oracle vectors        | Transcribed verbatim from `fast_ut.cpp`     | Yes (bitstream-exact) | ✓ FLOWING |

---

### Behavioral Spot-Checks

| Behavior                                           | Command                                                      | Result                                                                             | Status  |
|----------------------------------------------------|--------------------------------------------------------------|------------------------------------------------------------------------------------|---------|
| Workspace builds clean                             | `cargo build --workspace`                                    | Finished dev profile, 0 errors                                                     | ✓ PASS  |
| Clippy workspace-wide lib pass                     | `cargo clippy --workspace --lib -- -D warnings`              | Finished, 0 warnings/errors                                                        | ✓ PASS  |
| Full test suite                                    | `cargo test --workspace`                                     | 29 tests total (10 cb-core, 15 cb-oracle, 3 per-stage integration, 1 skeleton integration), all ok | ✓ PASS  |
| PRNG Test3 oracle vector                           | `cargo test -p cb-core test3`                                | `test3_from_seed_17_first_gen_rand ... ok`; value == 14895365814383052362          | ✓ PASS  |
| anyhow ban gate                                    | `bash scripts/check-no-anyhow.sh`                            | "OK: no anyhow in core library code"; exit 0                                       | ✓ PASS  |
| Source/test separation gate                        | `bash scripts/check-source-test-separation.sh`               | "OK: no inline #[cfg(test)] module bodies in production source"; exit 0            | ✓ PASS  |
| cb-backend wgpu/cuda/rocm feature builds           | `cargo build -p cb-backend --no-default-features --features {wgpu,cuda,rocm}` | All 3 exit 0                                                          | ✓ PASS  |

---

### Requirements Coverage

| Requirement | Source Plan | Description                                              | Status      | Evidence                                                                                         |
|-------------|-------------|----------------------------------------------------------|-------------|--------------------------------------------------------------------------------------------------|
| INFRA-01    | 01-01, 01-03 | Modular Cargo workspace with feature-gated backend crates | ✓ SATISFIED | 8-crate workspace; cb-backend with cpu/wgpu/cuda/rocm features; all build; cb-compute cubecl-free |
| INFRA-02    | 01-01        | Lint discipline enforced in library crates; anyhow CI check | ✓ SATISFIED | `[workspace.lints.clippy]` deny table; check-no-anyhow.sh gates in CI; no anyhow in [dependencies] of core libs |
| INFRA-03    | 01-01, 01-03 | Oracle harness with frozen fixtures, pinned seed/version, thread_count=1, <= 1e-5 | ✓ SATISFIED | regression_skeleton fixtures generated by catboost==1.2.10, thread_count=1, seed=0; read pipeline proven; per_stage_oracle_test passes |
| INFRA-04    | 01-03        | Per-stage oracle tooling: borders, splits, leaf values, approximants, predictions | ✓ SATISFIED | Stage enum with all 5 variants; compare_stage proven falsifiable on real fixtures; model.json contains oblivious_trees/splits/leaf_values; per-stage comparison of Rust-computed actuals deferred to P3/P4 per explicit plan scope note |
| INFRA-05    | 01-02        | Exact TFastRng64 port, bitstream-oracle-validated         | ✓ SATISFIED | All 6 oracle vectors from fast_ut.cpp pass; wrapping_* arithmetic throughout; non-crypto doc comment present |
| INFRA-06    | 01-03        | Source and test code strictly separated                  | ✓ SATISFIED | check-source-test-separation.sh exits 0; no inline #[cfg(test)] mod bodies in production files; separation script wired into CI |

**All 6 phase requirements satisfied.**

---

### Anti-Patterns Found

| File                                      | Line  | Pattern                                                        | Severity  | Impact                                                                               |
|-------------------------------------------|-------|----------------------------------------------------------------|-----------|--------------------------------------------------------------------------------------|
| `crates/cb-oracle/src/compare.rs`         | 40-41 | NaN silently passes the `diff > tol` comparator gate (WR-01)  | WARNING   | A NaN actual returns `Ok(())` instead of `Err(Diverged)` — confirmed by test; no NaN test case guards this hole; undermines harness integrity for later phases that may produce NaN from divergent gradients. Tracked as WR-01 in 01-REVIEW.md |
| `scripts/check-no-anyhow.sh`              | 13-20 | `catboost-rs` crate omitted from CORE_DIRS (WR-02)            | WARNING   | The published facade is not scanned by the belt-and-suspenders gate; structural ban (no anyhow in Cargo.toml [dependencies]) currently holds — no violation exists today. Tracked as WR-02 in 01-REVIEW.md |
| `crates/cb-oracle/src/compare.rs`         | 89-91 | Dead `Err(other) => Err(other)` catch-all in compare_stage (WR-04) | INFO   | Forward-compat framing is wrong: new upstream variants would forward untagged, defeating stage-tagging design. Currently unreachable. Tracked as WR-04 in 01-REVIEW.md |
| `crates/cb-oracle/src/bin/write_skeleton.rs` + fixture_test.rs + skeleton_oracle_test.rs | multiple | SKELETON_VALUES duplicated across 3 files with no shared source (WR-05) | INFO | Manual-sync risk; "MUST stay in sync" by convention only. Tracked as WR-05 in 01-REVIEW.md |

**WR-01 Classification:** The NaN hole in `compare.rs` is a known tracked finding (01-REVIEW.md WR-01). It does NOT block the phase 1 goal: the phase goal is to establish the infrastructure and harness; NaN-producing Rust algorithms do not yet exist (they arrive in P3/P4). The fix should be applied before P3 ships its first Rust-computed actuals. Classified as WARNING here, not BLOCKER.

**WR-02 Classification:** The structural ban (no anyhow in `[dependencies]`) is the primary defense and currently effective — `catboost-rs/src/` has zero anyhow references. The grep gap is a belt-and-suspenders weakness. Classified as WARNING.

No `TBD`, `FIXME`, or `XXX` markers found in any phase-modified files. Debt-marker gate: PASS.

---

### Human Verification Required

#### 1. GitHub Actions CPU Lane — Live Green Run

**Test:** Push the current branch to the remote repository and observe the Actions run at `https://github.com/BectorVoom/catboost_rs/actions` (or equivalent).
**Expected:** A single job named "CPU lane (build + clippy + test + anyhow gate)" runs and all steps pass. No GPU/ROCm/CUDA job appears. Steps verified: Checkout, Install Rust toolchain, Build workspace, cb-backend feature-gate build (wgpu), cb-backend feature-gate build (cuda), cb-backend feature-gate build (rocm), Clippy lint gate, Test workspace, anyhow ban backstop, source/test separation gate.
**Why human:** GitHub Actions requires a live push to the remote repository. The workflow file content and all local gates have been verified, but the actual CI execution on a hosted runner cannot be simulated locally.

---

### Gaps Summary

No must-have truths failed. All 5 success criteria and all 6 requirement IDs (INFRA-01 through INFRA-06) are verified in the codebase. The two WARNINGs (WR-01 NaN hole, WR-02 anyhow script gap) are tracked in 01-REVIEW.md and do not block the phase goal — they should be addressed before P3 delivers Rust-computed actuals. The single human-verification item is the live GitHub Actions green run; all local gates pass.

---

_Verified: 2026-06-13T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
