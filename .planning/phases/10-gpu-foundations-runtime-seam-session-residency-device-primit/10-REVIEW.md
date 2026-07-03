---
phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit
reviewed: 2026-07-03T03:03:50Z
depth: standard
files_reviewed: 28
files_reviewed_list:
  - crates/catboost-rs/src/builder.rs
  - crates/cb-backend/src/gpu_backend.rs
  - crates/cb-backend/src/gpu_backend_test.rs
  - crates/cb-backend/src/gpu_runtime/cindex.rs
  - crates/cb-backend/src/gpu_runtime/der_seams.rs
  - crates/cb-backend/src/gpu_runtime/mod.rs
  - crates/cb-backend/src/gpu_runtime/session.rs
  - crates/cb-backend/src/gpu_runtime/session_residency.rs
  - crates/cb-backend/src/kernels.rs
  - crates/cb-backend/src/kernels/apply_leaf_delta.rs
  - crates/cb-backend/src/kernels/cindex.rs
  - crates/cb-backend/src/kernels/compression.rs
  - crates/cb-backend/src/kernels/fill_transform.rs
  - crates/cb-backend/src/kernels/partitions.rs
  - crates/cb-backend/src/kernels/pointwise_hist.rs
  - crates/cb-backend/src/kernels/reduce.rs
  - crates/cb-backend/src/kernels/scan.rs
  - crates/cb-backend/src/kernels/segmented_scan.rs
  - crates/cb-backend/src/kernels/sort.rs
  - crates/cb-backend/src/kernels/update_part_props.rs
  - crates/cb-compute/src/lib.rs
  - crates/cb-compute/src/runtime.rs
  - crates/cb-train/src/boosting.rs
  - crates/cb-train/tests/device_seam_test.rs
  - bench/RESULTS.md
  - bench/cuda_oracle.ipynb
  - bench/fixtures/README.md
  - bench/generator.py
findings:
  critical: 1
  warning: 3
  info: 2
  total: 6
status: issues_found
---

# Phase 10: Code Review Report

**Reviewed:** 2026-07-03
**Depth:** standard
**Files Reviewed:** 28
**Status:** issues_found

## Summary

Phase 10 lands the GPU foundations: the `GpuBackend` runtime seam, the per-fit
device-resident `GpuTrainSession`, the bit-packed cindex (GPUT-15), the depth-1
device grow loop, and the Kaggle CUDA oracle harness. The der seams, the cindex
packer, the reduce/scan/partition kernels, the coverage-gate declines
(depth>1 / non-RMSE-Logloss / non-Plain / fold>1 / non-zero bias / Newton), and
the RAII teardown are careful and well-guarded: `checked_*` arithmetic, host-side
value-range validation, typed errors instead of panics, no `unwrap` in production,
and the HIP-safe finite `f32::MIN` sentinel instead of an `-inf` literal.

However, the device coverage gate has a hole that **turns a documented CPU
fallback into a hard training failure** for essentially every realistic dataset:
the gate never checks that the uniform `n_bins` is one the histogram fill can
actually dispatch. `begin` commits the whole fit to the device (all-or-nothing,
D-10-01) and the launch then errors at grow time for any `n_bins ∉
{2,16,32,64,128,256}` — the common case under the default 254-border
quantization. That is CR-01. Two benchmark-oracle findings (Cosine-vs-L2
mislabel and the weighted-der reference) weaken or could falsely fail the
committed depth-1 reference the Kaggle gate compares against.

## Critical Issues

### CR-01: Device coverage gate does not reject an un-dispatchable `n_bins`; commits the fit then hard-fails instead of falling back to CPU

**File:** `crates/cb-backend/src/gpu_runtime/session.rs:150-169` (`GpuTrainSession::begin`); interacts with `crates/cb-train/src/boosting.rs:2957-3012` and `crates/cb-backend/src/gpu_runtime/mod.rs:749-773` (`hist2_launch_resident` dispatch)

**Issue:**
`begin` only declines a degenerate `n_bins == 0` (session.rs:167); it accepts any
other `n_bins` and returns `Ok(Some(session))`, which the host reads as
`covered = true` and uses to **commit the entire fit to the device path** (the
D-10-01 all-or-nothing decision, `boosting.rs:2962-2985`). But the histogram fill
`hist2_launch_resident` (mod.rs:749-773) can only dispatch a fixed set of line
sizes — `BINARY_BINS (2)`, `HALF_BYTE_BINS (16)`, and the non-binary
`{32, 64, 128, 256}` — and returns
`CbError::Degenerate("... expects n_bins in {32,64,128,256} ...")` for anything
else.

`device_n_bins` is `max_f(feature_borders[f].len()) + 1`
(`boosting.rs:2113-2116`, `quantize_feature_major`). With the builder default
`border_count = 254` (`crates/cb-data/src/quantize.rs`), a feature with 254
borders yields `n_bins = 255`, which is not in the dispatchable set. Because
`begin` already committed the fit, `grow_tree_on_device` propagates the
`Degenerate` error up through `train_inner` (`boosting.rs:2996-3012`) and the fit
**fails hard** rather than declining to the byte-unchanged CPU grower — a direct
violation of the D-04 "decline → CPU fallback, never a hard failure" contract.

Reachability (GPU builds — `wgpu`/`cuda`/`rocm`): the covered gate is
`depth==1 && Plain && fold==1 && (RMSE|Logloss|CrossEntropy) && bias==0 &&
Gradient/Simple && supported score fn`. A user calling
`CatBoostBuilder::new().loss(Loss::Logloss).depth(1).fit(&pool)` hits it directly
(Logloss ⇒ `bias == 0`, since `starting_approx` only means-seeds RMSE —
`boosting.rs:1105-1107`), as does
`...loss(Rmse).boost_from_average(false).depth(1)`. In both cases any feature
whose max border count is not exactly `1/15/31/63/127/255` (the overwhelming
majority of real data at default settings) trips the failure.

This escaped the suite because every device-grow test pins `n_bins = 32`
(`gpu_backend_test.rs:170`, `session_residency.rs:209`), the one non-binary value
in the dispatch set.

**Fix:** Make the coverage gate decline (→ CPU fallback) for any `n_bins` the fill
cannot dispatch, so the fit is never committed to a device path that will error.
Add to `GpuTrainSession::begin`, alongside the existing degenerate check:

```rust
// session.rs, in begin(), after the `n == 0 || n_features == 0 || n_bins == 0` decline:
// The device histogram fill (hist2_launch_resident) only dispatches these line
// sizes. Any other n_bins would commit the fit then hard-fail at grow time —
// decline to the CPU path (D-04) instead of a hard failure.
if !matches!(n_bins, 2 | 16 | 32 | 64 | 128 | 256) {
    return Ok(None);
}
```

Prefer sourcing the set from the kernel constants (`BINARY_BINS`,
`HALF_BYTE_BINS`, and the `{32,64,128,256}` non-binary widths) so the gate and the
dispatch cannot drift. Optionally mirror the predicate in `device_host_eligible`
(`boosting.rs`) for symmetry, but the `begin` decline is the load-bearing fix
since it is the single all-or-nothing commit point.

## Warnings

### WR-01: Depth-1 benchmark oracle computes the L2 score but is labeled "Cosine"

**File:** `bench/generator.py:230-284` (`serial_depth1_tree`); mislabeled at `bench/generator.py:232` and `bench/fixtures/README.md:61`

**Issue:**
The docstring and `fixtures/README.md` state the committed depth-1 reference uses
the **Cosine** split score, but the code computes the **L2 / variance-reduction**
score:

```python
score = sl * sl / (wl + l2) + sr * sr / (wr + l2)   # this is L2, not Cosine
```

The device Cosine arm (`kernels.rs:3196-3203`) divides that same numerator by
`sqrt(1e-100 + Σ avg²·w)` — a per-candidate denominator. Since
`argmax(L2) ≠ argmax(Cosine)` in general (the denominator varies by candidate),
the `best_feature` / `best_bin` / `best_border` / `leaf_left` / `leaf_right`
recorded in `expected_depth1_tree.json` can select a **different split** than the
device Cosine path the Kaggle notebook validates against — either a false oracle
failure or, if they coincide on this seed, an oracle silently weaker than its
stated `≤1e-5` Cosine claim.

**Fix:** Either compute the true Cosine score in the reference (divide the folded
numerator by `sqrt(1e-100 + Σ (sum/(w+l2))² · w)` over the two leaves, matching
`score.rs` / `find_optimal_split_kernel`), or drive the device side of the depth-1
oracle with the L2 score and update the docstring + README to say "L2 score" so
reference and system-under-test use the same score function.

### WR-02: Depth-1 oracle reference uses weighted der in the numerator; the shipped path uses unweighted der

**File:** `bench/generator.py:256,270-272,279-280`

**Issue:**
`serial_depth1_tree` folds `der1 = der1 * weights` and then sums the **weighted**
der into the score numerator and leaf value (`sl = der1[left].sum()`,
`leaf = lr * sl / (wl + l2)`). The shipped path sums the **unweighted** der:
`reduce_leaf_stats` puts `sum_f64(deltas)` (raw der1) into `sum_weighted_delta`
(`crates/cb-compute/src/histogram.rs:82`), and the device histogram channel 0 is
`Σ der1` unweighted (`kernels.rs:712`, `bin_sums[cell].fetch_add(d)` with
`d = der1[obj]`). The two agree only because this fixture pins unit object weights
(`generator.py:313`); a future weighted fixture would make the committed reference
diverge from the device/CPU result it is meant to certify.

**Fix:** Drop the `der1 = der1 * weights` multiply (fold weight only into the
`wl`/`wr` counts) so the reference numerator matches the shipped unweighted-der
convention, keeping the oracle valid if a weighted fixture is added.

### WR-03: Device fit ignores object weights in the leaf/score numerator with no uniform-weight gate

**File:** `crates/cb-train/src/boosting.rs:2925-2950` (`device_host_eligible`)

**Issue:**
The device grow path sums `Σ der1` (unweighted) into histogram channel 0 and
estimates leaves via `calc_average(Σ der1, Σ weight, l2)` — weight enters only the
denominator, never the numerator. This mirrors the in-repo CPU path (WR-02), so
cpu-vs-gpu builds agree, but it is not how upstream CatBoost weights the leaf
numerator (`Σ w·der`, `TBucketStats::SumWeightedDelta`). The eligibility gate does
not require uniform weights, so a genuinely weighted pool takes the device path
and produces leaf values that cannot meet the `≤1e-5` upstream-parity bar the
milestone targets. Latent gap inherited rather than introduced here, but silently
reachable via the device seam.

**Fix:** Until weighted-der is wired into the histogram/leaf math, add
`&& weights.iter().all(|&w| w == 1.0)` (or an explicit no-non-unit-weight
predicate) to `device_host_eligible`, or fold weight into the der contribution
(`d * w`) in `pointwise_hist2_*` / `partition_update` and the CPU
`reduce_leaf_stats` together. Document the choice against the upstream
`SumWeightedDelta` convention.

## Info

### IN-01: `read_bin` computes `offset + obj` in `u32` before widening to `usize`

**File:** `crates/cb-backend/src/kernels.rs:2677-2680`

**Issue:** `let word = cindex[(offset + obj) as usize];` adds two `u32`s and then
casts. `device_arrays()` guarantees `offset` fits `u32`, but `offset + obj` is
evaluated in `u32` and would wrap before the `usize` cast if it exceeded
`u32::MAX`. Not reachable in practice (it needs a >4-billion-element `words`
buffer, i.e. >16 GB), so informational — but the ordering is a latent foot-gun if
larger buffers ever appear.

**Fix:** Widen before adding: index with `(offset as usize + obj as usize)`
(the `obj as u32` argument can stay), matching the `usize`-domain index arithmetic
used elsewhere in the kernels.

### IN-02: `bench/RESULTS.md` primitive-gate bars are inconsistent with the README/notebook "bit-exact" claim

**File:** `bench/RESULTS.md:51-56` vs `bench/fixtures/README.md:44-57`

**Issue:** The RESULTS template bars scan / reduce-by-key / segmented-reduce at
`≤1e-4`, while `fixtures/README.md` groups the integer/index primitives under
"bit-exact" and the deterministic reduce strategies are asserted byte-identical
in-tree (`kernels/reduce.rs`). A reviewer filling the Kaggle template from the
RESULTS bars could record a looser pass than the deterministic primitives actually
achieve.

**Fix:** Align the RESULTS bar labels with the README (mark the integer/index
primitives "bit-exact"; keep the float reduces at their stated tolerance) so the
sign-off log states one consistent bar per primitive.

---

_Reviewed: 2026-07-03_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
