# Phase 04 — Deferred Items

Out-of-scope discoveries logged during execution (NOT fixed; tracked here per the
SCOPE BOUNDARY rule).

| Source plan | Item | Detail | Status |
|-------------|------|--------|--------|
| 04-01 | Pre-existing clippy `excessive_precision` warning | `crates/cb-train/src/bootstrap.rs:134` — `0.693_147_18_f32` in `fast_logf` (Phase-3 code, unrelated to Plan 04-01 changes). Clippy exits 0 (warning only). | Deferred — pre-existing, out of scope |
| 04-01 | flatc multi-file per-namespace cross-reference bug | Per-file `flatc --rust` of model/features/ctr_data (sharing `NCatBoostFbs`) emits `use crate::features_generated::*` that fails to resolve bare cross-file types (`TEstimatedFeature`). Worked around by generating each file SELF-CONTAINED via `flatc --rust --gen-all`. The committed files are still pure flatc output (D-01 honored). | Resolved (deviation Rule 3) |
| 04-02 | `cargo test -p cb-compute loss` could not run (disk full) | The disk has <1 GB free; cb-compute's test profile must recompile `polars-core` (a transitive dev-dep via `cb-data`, ~1.3 GB rlib) and fails with `No space left on device`. The new `cross_entropy`/`focal` der1/der2 unit tests were ADDED to `cb-compute/src/loss_test.rs`, but the SAME functions are fully exercised and PASSING via `cb-train/tests/loss_oracle_test.rs` (`cross_entropy_der_values`, `focal_der_values`, plus the CrossEntropy/Focal training oracle locks ≤1e-5) which compiled and ran green. | Deferred — environment disk limit, not a code defect; equivalent coverage passes via cb-train |
