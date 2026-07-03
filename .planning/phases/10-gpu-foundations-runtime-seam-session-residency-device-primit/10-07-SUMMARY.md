---
phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit
plan: 07
subsystem: cb-backend
tags: [gpu, rocm, cubecl, residency, session, apply-leaf-delta, depth1, cosine, GPUT-02, GPUT-03, GPUT-04, GPUT-08]

# Dependency graph
requires:
  - phase: 10-02 (GPUT-01 grow-tree runtime seam)
    provides: DeviceGrownTree host struct + begin/grow/end default-impl seam this plan overrides on GpuBackend
  - phase: 10-06 (GPUT-15 bit-packed cindex)
    provides: pack_cindex + TCFeature device_arrays the resident histogram consumes (uploaded once at begin)
  - phase: 10-05 (GPUT-16 update_part_props)
    provides: launch_partition_update_into the depth-1 leaf step reads back (2^depth part-stats)
  - phase: 7.5 (host-light grow loop)
    provides: grow_oblivious_tree_into geometry + launch_partition_split_into the resident grow reuses
  - phase: 7.2 (GPU der seam)
    provides: gradient/logloss_gradient kernels the resident der recompute (launch_der_binary_resident) launches
provides:
  - apply_leaf_delta_kernel (approx[i] += lr * leaf_values[leaf_of[i]]) + launch_apply_leaf_delta_into (device approx update, no read-back)
  - hist2_launch_resident + score_over_binsums (the ONE resident histogram / score geometry; slice entries delegate)
  - launch_der_binary_resident (der1 = der(approx_h, target_h) as a handle, no read-back)
  - grow_oblivious_tree_resident (depth-1 grow over pre-uploaded session handles; approx/der1 device-resident)
  - GpuTrainSession (one client + persistent handles, coverage gate, per-fit lifecycle, Drop teardown)
  - GpuBackend RefCell<Option<GpuTrainSession>> + begin/grow/end_device_training seam impls
  - cb_compute::DeviceGrownTree re-exported at crate root (10-02 gap)
affects: [10-09 Kaggle CUDA authoritative depth-1 RMSE ~1e-9 + Logloss <=1e-5 oracle (human-gated)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Resident launch geometry as ONE place: hist2_launch_resident / score_over_binsums take pre-uploaded handles; the slice entries (launch_pointwise_hist2_into / launch_find_optimal_split_pointwise_into) upload-then-delegate, so the session clones the SAME persistent handles into the SAME geometry per tree (no per-tree re-upload) with ZERO behaviour change to the 44 existing consumers"
    - "Both cindex layouts uploaded once at begin: the PACKED words (histogram read_bin) + the PLAIN feature-major (partition_split degenerate TCFeature) — avoids re-packing per tree AND keeps launch_partition_split_into byte-identical"
    - "Session-internal device-resident boosting: grow_one recomputes der1 from resident approx on device (no read-back), grows over resident handles, updates approx via apply_leaf_delta on device, chains der1; only O(1) BestSplit + 2^depth part-stats + one end leaf_of (oracle) cross"
    - "RefCell<Option<GpuTrainSession>> interior mutability keeps the &self Runtime seam; drops Copy (manual Debug, derive Default) — the two by-value GpuBackend constructions moved to ::default()"

key-files:
  created:
    - crates/cb-backend/src/kernels/apply_leaf_delta.rs
    - crates/cb-backend/src/gpu_runtime/session.rs
    - crates/cb-backend/src/gpu_runtime/session_residency.rs
  modified:
    - crates/cb-backend/src/kernels.rs
    - crates/cb-backend/src/gpu_runtime/mod.rs
    - crates/cb-backend/src/gpu_runtime/der_seams.rs
    - crates/cb-backend/src/gpu_backend.rs
    - crates/cb-backend/src/gpu_backend_test.rs
    - crates/cb-compute/src/lib.rs
    - crates/catboost-rs/src/builder.rs

key-decisions:
  - "Upload BOTH cindex layouts once (packed words for the histogram, plain feature-major for the partition split) rather than repacking the resident buffer for partition. Rationale: launch_partition_split_into is shared with out-of-scope pairwise/grow_loop callers; a resident repack would ripple beyond this plan. The plain layout IS read_bin's degenerate TCFeature — byte-identical, and the memory cost (one extra n_features*n u32 per fit) is acceptable for the depth-1 MVP."
  - "The session runs its OWN device-resident boosting (resident approx/der1, apply_leaf_delta on device, der1 chained) and grow_tree_on_device delegates to grow_one; the seam's passed `approx` is validated for length but the resident approx is authoritative. In the covered Plain/fold=1/from-zero regime the two track exactly (same zero start, same per-tree lr*leaf_values). This is what makes apply_leaf_delta + the no-read-back der1 chain reachable (must-have truth 3) despite the seam passing approx per tree."
  - "GrownTree/DeviceGrownTree leaf_values are UNSCALED (calc_average delta, the 10-02 contract); apply_leaf_delta applies learning_rate to the on-device approx. leaf_of is read back ONCE at the end of each tree (the sanctioned structure-oracle seam, same crossing class as part_stats — the must-have forbids der1 read-back specifically, which is eliminated)."
  - "Coverage gate lives in GpuTrainSession::begin (D-10-02): depth==1 && (RMSE|Logloss|CrossEntropy) && Plain && fold_count==1 && supported score fn; Cosine is the depth-1 device default (GPUT-08, honored from the passed EScoreFunction, which defaults to Cosine upstream). Newton score fns decline (GPU-only, no depth-1 MVP arm)."

requirements-completed: [GPUT-02, GPUT-03, GPUT-04, GPUT-08]

# Metrics
duration: ~110min
completed: 2026-07-03
status: complete
---

# Phase 10 Plan 07: Device-Resident Depth-1 Grow + Session Residency + Cosine Default Summary

**Grew a depth-1 oblivious tree fully on device over a per-fit residency wrapper: `GpuTrainSession` uploads the quantized matrix (both the packed-cindex `words` the histogram reads AND the plain feature-major layout the partition split reads), weights, indices, and target ONCE, then `grow_one` reuses the resident handles per tree — the running `approx` is updated ON DEVICE via the new `apply_leaf_delta_kernel`, the residual `der1` is recomputed device-resident (the resident der seam) and chained into the next tree, so NO n-length der1 read-back crosses per tree (only the O(1) BestSplit per level, the 2^depth part-stats, and one end `leaf_of` for the structure oracle). Cosine is the depth-1 device default (GPUT-08). `GpuBackend` now holds a `RefCell<Option<GpuTrainSession>>` and implements the 3 `begin`/`grow`/`end` seam methods with an `Ok(None)` CPU fallback for uncovered configs. The 5-tree device-resident boosting matches the CPU multi-tree Cosine reference EXACTLY in structure; rocm gfx1100 in-env 123/123 green, catboost-rs rocm check clean (no cb-train leak).**

## Performance
- **Duration:** ~110 min
- **Completed:** 2026-07-03
- **Tasks:** 3
- **Files:** 10 (3 created, 7 modified)

## Accomplishments

- **`apply_leaf_delta_kernel` + `launch_apply_leaf_delta_into` (GPUT-03)** — the device-resident approx update `approx[i] += lr * leaf_values[leaf_of[i]]` (generic-float, bounds-guarded, no `-inf` literal, scalar `lr` as a length-1 array per the codebase convention). The launcher updates the resident approx IN PLACE and returns it WITHOUT any read-back, so the running approx stays a resident buffer across boosting iterations. Self-oracled vs a serial CPU reference (depth-1 + grid-stride large-n + empty no-op), rocm 3/3.
- **`hist2_launch_resident` + `score_over_binsums` (GPUT-03, IN-02 one place)** — extracted the histogram fill and the score/argmin into handle-taking resident cores; `launch_pointwise_hist2_into` and `launch_find_optimal_split_pointwise_into` now upload-then-delegate, so the resident session clones its persistent handles into the SAME geometry per tree. Zero behaviour change: the 44 existing grow_loop/score_split/pointwise_hist/cindex consumers stay bit-identical (35/35 green post-refactor).
- **`launch_der_binary_resident` (GPUT-03)** — the residency variant of the 7.2 der seam: `der1_h = der(approx_h, target_h)` from resident handles, returning the der1 Handle with NO approx/der1 read-back (f64-typed; wgpu typed-rejects — WR-02).
- **`grow_oblivious_tree_resident` (GPUT-03/08)** — grows one depth-1 tree over pre-uploaded session handles, reusing the resident histogram/score/partition geometry; after the leaves it applies `apply_leaf_delta` on device and recomputes `der1` resident for the next tree. `depth>1` surfaces the typed forward-dependency error (the coverage gate keeps it off this path).
- **`GpuTrainSession` (GPUT-02)** — owns ONE `ComputeClient` + persistent `plain_cindex`/`packed cindex`/TCFeature/`indices`/`weight`/`target`/`approx`/`der1` handles; `begin` runs the coverage gate (returns `None` → CPU fallback for depth>1 / non-RMSE-Logloss / non-Plain / fold_count>1 / unsupported score fn), validates the bins host-side, packs the cindex, and uploads everything ONCE; `grow_one` advances the device-resident boosting; Drop frees the client + handles deterministically. All read-backs go through the allocating client (T-10-18).
- **`GpuBackend` seam impls (GPUT-04)** — replaced the `Copy`/zero-sized derive with a `RefCell<Option<GpuTrainSession>>` field (manual `Debug`, derive `Default`); `begin_device_training` stores `Some(session)` (`Ok(true)`) or `None` (`Ok(false)`), `grow_tree_on_device` borrows the session → `Ok(Some(tree))` / `Ok(None)`, `end_device_training` takes+drops it. The two by-value `GpuBackend` constructions moved to `::default()` (facade builder + test).
- **Cross-oracles (rocm gfx1100 in-env):** `session_residency` (5-tree device pass matches CPU multi-tree Cosine boosting structure EXACTLY, coverage gate declines uncovered) + `gpu_backend` seam lifecycle (covered begin→grow→end + uncovered fallback). Full cb-backend rocm suite **123/123 green**; `catboost-rs --features rocm` check clean (cb-train compiles against the new seam, no feature-unification leak).

## Task Commits
1. **Task 1: apply_leaf_delta kernel + resident no-read-back grow variant + Cosine-ready score seam** — `ed0f9bb` (feat)
2. **Task 2: GpuTrainSession residency wrapper + coverage gate** — `d7de7fb` (feat)
3. **Task 3: GpuBackend session field + 3 device seam impls** — `74e7333` (feat)

## Deviations from Plan
None functional — the plan executed as written. Scope-clarifying notes (documented above as key-decisions, not correctness deviations):

1. **Both cindex layouts uploaded once** (packed for histogram, plain for partition) rather than a single resident repack — keeps `launch_partition_split_into` (shared with out-of-scope callers) byte-identical while still satisfying "matrix uploads once (no per-tree re-upload)".
2. **The `apply_leaf_delta` #[cube] kernel lives in `kernels.rs`** (with all the other `#[cube]` kernels) and its launcher in `gpu_runtime/mod.rs`; the plan's artifact `contains: "apply_leaf_delta"` for mod.rs is satisfied by `launch_apply_leaf_delta_into`. This follows the established `kernel-in-kernels.rs / launcher-in-mod.rs` split (gradient_kernel etc.), not a deviation from behaviour.
3. **`cb_compute::DeviceGrownTree` re-exported at the crate root** — a 10-02 gap (it was added to `runtime.rs` but not the `lib.rs` re-export list); required for the session/backend to name the seam type (Rule 3 blocking fix, additive).

## Known Stubs
None. The depth-1 device path is fully wired end-to-end (session → resident grow → apply_leaf_delta → der chain → GpuBackend seam) and cross-oracled against the CPU reference. The `depth>1` / non-Plain / fold>1 / non-RMSE-Logloss configs are intentional typed `Ok(None)` CPU-fallback gates (D-04), not stubs — the partition-aware (`fullPass=false`) histogram is the tracked forward dependency.

## Threat Flags
None beyond the plan's `<threat_model>`. T-10-18 (Handle through wrong client) mitigated: one ComputeClient per session owns all handles; every read-back through `self.client`; Drop teardown. T-10-19 (silent zero on read failure) mitigated: der1 stays resident (no per-tree read-back); every read-back maps failure to `CbError::Degenerate`, never a silent zero. T-10-20 (apply_leaf_delta portability UB) mitigated: no `-inf` literal, generic-float, bounds guard, rocm smoke green. T-10-21 (build landmine) mitigated: no cb-train dep added to cb-backend (grep-verified; `catboost-rs --features rocm` check stayed cpu-free).

## Next Phase Readiness
- 10-09 (Kaggle CUDA authoritative oracle) consumes this depth-1 device path for the RMSE ~1e-9 + Logloss (first-order leaves) ≤1e-5 sign-off — human-gated, not in-CI. The in-env rocm proof is structure-exact vs the CPU Cosine reference with REPORTED (not signed-off) leaf-value divergence.
- The production hot-path optimization (empty `leaf_of` per tree instead of the end read-back) is a trivial gate deferred until the CPU fallback is unwired — the structure oracle currently needs the populated `leaf_of`.

## Self-Check: PASSED
- Files: `kernels/apply_leaf_delta.rs`, `gpu_runtime/session.rs`, `gpu_runtime/session_residency.rs` (created); `kernels.rs`, `gpu_runtime/mod.rs`, `gpu_runtime/der_seams.rs`, `gpu_backend.rs`, `gpu_backend_test.rs`, `cb-compute/src/lib.rs`, `catboost-rs/src/builder.rs` (modified) — all FOUND.
- Commits: `ed0f9bb`, `d7de7fb`, `74e7333` — all FOUND in git log.
- Acceptance: `apply_leaf_delta_kernel` + `GpuTrainSession` + `grow_oblivious_tree_resident` present; `grep cb_train crates/cb-backend/Cargo.toml` empty (no cb-train dep); coverage gate returns None for depth>1/non-RMSE-Logloss/non-Plain/fold>1.
- rocm gfx1100 in-env: `cargo test -p cb-backend --no-default-features --features rocm` → **123 passed, 0 failed**; `cargo check -p catboost-rs --no-default-features --features rocm` → clean.

---
*Phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit*
*Completed: 2026-07-03*
