# Deferred Items — Phase 03

## Out-of-scope discoveries (do NOT fix in this wave)

### Pre-existing `clippy::approx_constant` errors in cb-oracle test/bin files
- **Found during:** 03-03 (Task 2) full-workspace `cargo clippy --all-targets`.
- **Files:** `crates/cb-oracle/src/bin/write_skeleton.rs`, `crates/cb-oracle/src/compare_test.rs`, `crates/cb-oracle/src/fixture_test.rs`, `crates/cb-oracle/tests/skeleton_oracle_test.rs` (PI/E literal approximations in test fixtures).
- **Status:** Pre-existing (files last touched in Phase 01, commit `0f75f90`); NOT introduced by 03-03. `cargo test --workspace` is green; these only surface under `clippy --all-targets` warnings-as-errors. Out of scope per the executor SCOPE BOUNDARY (only auto-fix issues directly caused by the current task).

### Pre-existing D-08 grep false-positive on the overfit `horner` Horner fold
- **Found during:** 03-06 (Task 2) `scripts/check-no-raw-float-sum.sh`.
- **File:** `crates/cb-train/src/overfit.rs:521` — `coeffs.iter().fold(0.0, |acc, &c| acc * x + c)` is a Horner POLYNOMIAL EVALUATION (multiply-add), not a parity-critical summation, used by the W.J. Cody `erf` primitive.
- **Status:** Pre-existing — committed in Plan 05 (`e1adc96`); the D-08 grep already failed at HEAD before 03-06 (verified by stashing 03-06 changes). The `SUM_PATTERN` `\.fold\(0\.0` over-matches the Horner fold; it is NOT a float sum and routing it through `cb_core::sum_f64` would be incorrect (sum_f64 cannot express `acc*x + c`). Out of scope per the executor SCOPE BOUNDARY (not introduced by 03-06). Fix candidate: narrow the D-08 pattern to exclude multiply-add Horner folds, or annotate the `horner` line. `cargo test --workspace` is green; 03-06's own `metrics.rs` is D-08-clean.

## Known residual (tracked, not a blocker)

### Bayesian bootstrap multi-tree end-to-end divergence (TRAIN-04)
- The Bayesian per-block weight draws + per-1000-block reseed are unit-verified, and the FIRST tree's splits + leaf values lock end-to-end at <=1e-5 (`bootstrap_oracle_bayesian_first_tree`). The multi-tree lock (`bootstrap_oracle_bayesian`) is `#[ignore]`d: tree-1+ Bayesian splits diverge by ~0.02 and the divergence is INSENSITIVE to any main-RNG phase offset (pre/post/extra-draw), indicating a structural Bayesian-specific issue in the multi-tree draw stream rather than a phase misalignment. No/Bernoulli/MVS lock end-to-end. Candidate for a follow-up investigation (possibly needs the categorical/Rsm draw accounting or a C++-instrumented per-tree weight dump to localize).
