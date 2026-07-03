---
phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit
plan: 02
subsystem: cb-compute
tags: [gpu, runtime-seam, grow-tree, GPUT-01, D-04]
requires:
  - "Runtime trait (cb-compute/src/runtime.rs) with default-impl precedent (compute_gradients_grouped)"
  - "EScoreFunction + Loss host types (cb-compute)"
provides:
  - "DeviceGrownTree host struct (splits, leaf_values, leaf_of)"
  - "Runtime::begin_device_training (default Ok(false))"
  - "Runtime::grow_tree_on_device (default Ok(None))"
  - "Runtime::end_device_training (default Ok(()))"
affects:
  - "crates/cb-compute/src/runtime.rs"
tech-stack:
  added: []
  patterns:
    - "default-impl trait method returning Ok(false)/Ok(None)/Ok(()) as a CubeCL-free CPU-fallback seam (mirrors compute_gradients_grouped)"
key-files:
  created: []
  modified:
    - "crates/cb-compute/src/runtime.rs"
decisions:
  - "Seam carries only host types (slices/Vecs/Loss/EScoreFunction/CbResult); no cubecl/cb-backend dependency added to cb-compute (T-10-04)."
  - "Defaults return Ok(false)/Ok(None)/Ok(()) so every existing Runtime impl inherits transparent CPU fallback byte-unchanged (D-04 / T-10-05)."
  - "DeviceGrownTree.leaf_of is empty in the hot path; length n only for the oracle structure check so the n-length buffer never crosses the seam per tree (D-05)."
metrics:
  duration: "~50m (dominated by environment disk-exhaustion recovery, not implementation)"
  completed: 2026-07-03
status: complete
---

# Phase 10 Plan 02: GPUT-01 Grow-Tree Runtime Seam Summary

Landed the GPUT-01 grow-tree device seam on the `Runtime` trait: a host-typed
`DeviceGrownTree` struct plus three default-impl methods
(`begin_device_training`, `grow_tree_on_device`, `end_device_training`) that keep
the trait CubeCL-free and preserve the CPU path byte-unchanged (D-04) for every
existing `Runtime` impl.

## What Was Built

- `pub struct DeviceGrownTree { splits: Vec<(u32,u32)>, leaf_values: Vec<f64>, leaf_of: Vec<u32> }`
  deriving `Debug`/`Clone`/`PartialEq`, added to `crates/cb-compute/src/runtime.rs`.
  - `splits`: per-level chosen oblivious split as `(feature_index, bin_id)`; pass
    test `quantized_bin[feature] > bin_id`; caller resolves
    `border = feature_borders[feature][bin_id]` (length = depth).
  - `leaf_values`: per-leaf, length `2^depth`, UN-scaled by `learning_rate`
    (cb-train applies the shrinkage downstream).
  - `leaf_of`: per-object leaf index; EMPTY in the production hot path (D-05),
    length `n` only for the oracle structure check.
- Three default-impl methods added to the `Runtime` trait, mirroring the
  `compute_gradients_grouped` default-impl precedent:
  - `begin_device_training(&self, loss: &Loss, depth, boosting_type_is_plain, fold_count, score_function: EScoreFunction, bins_feature_major: &[u32], weight: &[f64], n, n_features, n_bins, learning_rate, scaled_l2) -> CbResult<bool>` — default `Ok(false)`.
  - `grow_tree_on_device(&self, approx: &[f64], target: &[f64]) -> CbResult<Option<DeviceGrownTree>>` — default `Ok(None)`.
  - `end_device_training(&self) -> CbResult<()>` — default `Ok(())`.
  - Unused default params bound with `let _ = (...)` (the compute_gradients_grouped
    precedent); `#[allow(clippy::too_many_arguments)]` on `begin_device_training`.

## Task Commits

| Task | Name | Commit | Files |
| ---- | ---- | ------ | ----- |
| 1 | DeviceGrownTree host struct + 3 default-impl seam methods | 4ba92ac | crates/cb-compute/src/runtime.rs |

## Verification

- `cargo build -p cb-compute`: **PASS** (clean `Finished`). All changes for this
  plan live in cb-compute; the crate compiles with the new struct and methods.
- cb-train library crate compiles against the modified cb-compute: **PASS** —
  `libcb_train-*.rlib` was produced and appears in the test-binary linker inputs,
  proving the additive trait change is compile-compatible with cb-train (D-04).
- `grep cubecl crates/cb-compute/Cargo.toml`: the only match is the pre-existing
  D-03 comment ("MUST NOT depend on cubecl"); **no cubecl/cb-backend dependency
  was added** (only `runtime.rs` was edited, never `Cargo.toml`). Trait stays
  CubeCL-free (T-10-04 mitigated).

### Verification Limitation (environmental, not a code defect)

The full `cargo test -p cb-train` suite could **not link its test binaries** in
this session due to root-disk exhaustion. The captured errors are exclusively
`rustc-LLVM ERROR: IO failure on output stream: No space left on device`,
`No space left on device (os error 28)`, and `mold: failed to write to an output
file. Disk full?` — with **zero** trait/type/borrow errors. This matches the
documented environment constraint (project memory: "root disk ~100% full;
cb-compute test profile can't link; verify per-crate"). Concurrent phase-10
executors sharing `target/` filled the disk.

Why D-04 no-regression holds despite the unrun suite: the change is purely
additive default-impl trait methods plus a new struct. cb-train consumes the
`Runtime` trait and overrides **none** of the three new methods, so it inherits
the `Ok(false)`/`Ok(None)`/`Ok(())` bodies and its compiled behavior is
byte-unchanged. The successful production of `libcb_train-*.rlib` confirms the
cb-train library compiled against the modified trait; only test-executable
linking failed, and only for lack of disk.

## Deviations from Plan

None functionally. The implementation matches the plan's artifact spec exactly.

Environmental note (not a plan deviation): substantial effort went into working
around root-disk and tmpfs exhaustion caused by concurrent executors — including
reclaiming the regenerable `target/debug/incremental` cache (non-destructive;
cargo regenerates it) to obtain enough disk headroom for git to write the commit
object. No source or shared build-correctness state was altered.

## Known Stubs

None. The three methods are intentional default-impl fallbacks (Ok(false)/
Ok(None)/Ok(())), not stubs — they are the CPU-fallback contract (D-04/T-10-05)
that later plans override on GPU backends.

## Self-Check

- `crates/cb-compute/src/runtime.rs` contains `DeviceGrownTree` and the three
  seam methods: FOUND (Edit applied; cb-compute builds).
- Commit `4ba92ac` (feat(10-02): GPUT-01 grow-tree device seam on Runtime trait):
  FOUND in `.git/logs/HEAD` line 765.

## Self-Check: PASSED
