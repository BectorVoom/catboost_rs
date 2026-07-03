---
phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit
reviewed: 2026-07-03T03:03:50Z
depth: standard
files_reviewed: 23
files_reviewed_list:
  - crates/cb-backend/src/kernels.rs
  - crates/cb-backend/src/kernels/scan.rs
  - crates/cb-backend/src/kernels/segmented_scan.rs
  - crates/cb-backend/src/kernels/reduce.rs
  - crates/cb-backend/src/kernels/sort.rs
  - crates/cb-backend/src/kernels/partitions.rs
  - crates/cb-backend/src/kernels/fill_transform.rs
  - crates/cb-backend/src/kernels/compression.rs
  - crates/cb-backend/src/kernels/update_part_props.rs
  - crates/cb-backend/src/kernels/cindex.rs
  - crates/cb-backend/src/kernels/apply_leaf_delta.rs
  - crates/cb-backend/src/kernels/pointwise_hist.rs
  - crates/cb-backend/src/gpu_runtime/mod.rs
  - crates/cb-backend/src/gpu_runtime/cindex.rs
  - crates/cb-backend/src/gpu_runtime/session.rs
  - crates/cb-backend/src/gpu_runtime/session_residency.rs
  - crates/cb-backend/src/gpu_runtime/der_seams.rs
  - crates/cb-backend/src/gpu_backend.rs
  - crates/cb-compute/src/runtime.rs
  - crates/cb-compute/src/lib.rs
  - crates/cb-train/src/boosting.rs
  - crates/catboost-rs/src/builder.rs
  - bench/generator.py
findings:
  critical: 2
  warning: 3
  info: 2
  total: 7
status: issues_found
---

# Phase 10: Code Review Report

**Reviewed:** 2026-07-03T03:03:50Z
**Depth:** standard
**Files Reviewed:** 23
**Status:** issues_found

## Summary

Phase 10 lands a large, well-documented CubeCL device-primitive library (scan,
segmented-scan, reduce/reduce-by-key, sort/reorder, TDataPartition update,
fill/gather/vector ops, bit-compression, bit-packed cindex + `read_bin`) plus the
`Runtime::begin_device_training`/`grow_tree_on_device`/`end_device_training` seam,
a per-fit `GpuTrainSession` residency wrapper, and the cb-train boosting-loop wiring
that lets a depth-1 RMSE/Logloss/CrossEntropy Plain fit run entirely on the device.
The primitive kernels themselves are careful and consistent (bounds guards, checked
host-side arithmetic, no `-inf` literals, no `unwrap`/`expect`/`panic`/indexing in
production code, thorough self-oracles). However, the boosting-loop integration
that decides *when* a fit is safe to hand to the device grower has two silent
correctness gaps that let a fully default-shaped user request produce a
**materially wrong trained model with no error, warning, or test coverage** —
these are the headline findings below. A handful of lower-severity
documentation/consistency issues round out the report.

## Critical Issues

### CR-01: Device grow path ignores `boost_from_average` — silently drops the model bias

**Status:** FIXED (2026-07-03, commit 8957ed5) — Option (a) conservative CPU fallback: `device_host_eligible` now requires `bias == 0.0`. Regression test `device_declines_nonzero_starting_bias_boost_from_average` added (commit 128002b).

**File:** `crates/cb-train/src/boosting.rs:2925-2938` (the `device_host_eligible` gate) and `crates/cb-backend/src/gpu_runtime/session.rs:213-248` (`GpuTrainSession::begin` — resident `approx_h` always initialized to all-zero)

**Issue:**

`train_inner` computes the starting approximant/bias *before* the device gate is
evaluated:

```rust
let bias = starting_approx(params, target);   // boosting.rs:2489
let mut approx = ... vec![bias; approx_dimension * n] ...   // boosting.rs:2510
```

`starting_approx` (boosting.rs:1105-1111) returns the **target mean** whenever
`params.boost_from_average && loss == Loss::Rmse` — i.e. exactly the
`CatBoostBuilder` **default** (`boost_from_average: true`, `builder.rs:107`) for
the RMSE loss, one of the three losses the device path supports.

The `device_host_eligible` predicate (boosting.rs:2925-2938) does **not** check
`params.boost_from_average` or the computed `bias`, and `GpuTrainSession::begin`
(session.rs) is never given the bias — it unconditionally seeds its own resident
`approx_h` to `vec![0.0; n]` (session.rs:222-224, comment: *"the RMSE-from-zero
MVP; boost_from_average is out of scope"*). `GpuTrainSession::grow_one` does not
even accept `approx` as a parameter (session.rs:270) — only `target` — so the
session's residual `der1 = target - approx_h` is computed against the WRONG
(zero) starting point whenever `bias != 0`.

Meanwhile, in `train_inner`'s device branch, the device's (wrongly-derived)
leaf deltas are added onto the *host* `approx` array, which DOES start at
`bias`:

```rust
for (i, &leaf) in device_leaf_of.iter().enumerate() {
    if let (Some(a), Some(&lv)) = (approx.get_mut(i), device_leaf_values.get(leaf)) {
        *a += lv;   // boosting.rs:3066-3072 — approx[i] = bias + lr*leaf_value
    }
}
```

The result is a self-inconsistent, silently wrong model: the device grows every
tree as if training against a zero-mean target while the stored model output is
`bias + Σ(device tree contributions)`. This reproduces neither the CPU
`boost_from_average=true` fit nor a legitimate `boost_from_average=false` fit.

**Reachability:** any user who takes the default `CatBoostBuilder` (which
already has `boost_from_average: true`, `boosting_type: Plain`,
`score_function: Cosine` — all device-eligible by default) and calls
`.depth(1)` on an RMSE regression, compiled against any GPU feature
(`wgpu`/`cuda`/`rocm`), triggers this bug with zero indication anything is
wrong. `crates/cb-train/tests/device_seam_test.rs::device_params()` explicitly
sets `boost_from_average: false` to sidestep this case (see the comment "so
the bias is 0, keeping the staged assertion a pure tree contribution") — i.e.
the non-zero-bias case was never exercised by any test in this phase.

**Fix:** either (a) reject the device path when `bias != 0.0` (extend
`device_host_eligible` with `bias == 0.0`, or equivalently
`!(params.boost_from_average && matches!(params.loss, Loss::Rmse))`), or (b)
thread the bias into `begin_device_training`/`GpuTrainSession::begin` so the
resident `approx_h` is seeded to `vec![bias; n]` instead of `vec![0.0; n]`.
Option (a) is the minimal, low-risk fix for this phase; add a regression test
with `boost_from_average: true` (the builder default) at `depth(1)` asserting
device- and CPU-grown models agree (or that the device path correctly declines).

```rust
// boosting.rs, device_host_eligible:
let device_host_eligible = group_spans.is_none()
    && ordered_learning_perm.is_none()
    && materialized_ctr_features.is_empty()
    && structure_fold_columns.iter().all(Vec::is_empty)
    && !penalties_active
    && params.monotone_constraints.is_empty()
    && params.grow_policy == EGrowPolicy::SymmetricTree
    && approx_dimension == 1
    && !is_multiclass
    && !is_multilabel
    && matches!(params.bootstrap_type, EBootstrapType::No)
    && params.random_strength == 0.0
    && eval_sets.is_empty()
    && matrix.n_features() > 0
    && bias == 0.0;   // <-- NEW: the device session always starts approx at 0
```

---

### CR-02: Device grow path always uses the Gradient leaf method — silently ignores `LeafMethod::Newton`

**Status:** FIXED (2026-07-03, commit 1d111ce) — Option (a) conservative CPU fallback: `device_host_eligible` now requires `matches!(params.leaf_method, LeafMethod::Gradient | LeafMethod::Simple)`. Regression test `device_declines_newton_leaf_method_on_covered_loss` added (commit 128002b).

**File:** `crates/cb-train/src/boosting.rs:2925-2938` (`device_host_eligible` — no `leaf_method` check), `crates/cb-backend/src/gpu_runtime/mod.rs:2082-2090` (`grow_oblivious_tree_resident`, always calls `cb_compute::calc_average`), `crates/cb-compute/src/runtime.rs:1012-1026` (`begin_device_training` signature carries no `leaf_method` parameter)

**Issue:**

The CPU oblivious-tree path dispatches the leaf-value formula on
`params.leaf_method` (`compute_leaf_deltas`, boosting.rs:1412-1459):
`LeafMethod::Gradient`/`Simple` use `calc_average(Σder1, Σweight, l2)`, while
`LeafMethod::Newton` uses `Σder1 / (-Σder2*weight + l2)` — a genuinely
different formula whenever `der2` is not a per-object constant. For `Loss::Rmse`
`der2 == -1.0` for every object so the two formulas happen to coincide bit-for-bit
(harmless), but for `Loss::Logloss`/`Loss::CrossEntropy` — two of the three
losses the device path supports — `der2[i] = -p(1-p)` varies per object, so
Newton and Gradient diverge.

The device path (`grow_oblivious_tree_resident`, mod.rs:2082-2090) unconditionally
computes leaf values via `cb_compute::calc_average(sum, cnt, scaled_l2)` — the
Gradient/Simple formula — with no way to request the Newton formula: neither
`Runtime::begin_device_training` nor `GpuTrainSession::begin`/`grow_one` accepts
a `leaf_method` parameter at all, and `device_host_eligible` in `train_inner`
never inspects `params.leaf_method` before committing the whole fit to the
device (boosting.rs:2925-2938).

**Reachability:** a user selecting `Loss::Logloss` (or `CrossEntropy`) with
`.leaf_method(LeafMethod::Newton)` — the leaf-estimation method upstream
CatBoost actually defaults to for Logloss — combined with `.depth(1)`,
default `Plain` boosting, and a default/`Cosine` score function on a
GPU-featured build, silently gets Gradient-method leaves instead of the
requested Newton leaves, with no error and no divergence signal (both paths
"succeed").

**Fix:** extend `device_host_eligible` to require
`matches!(params.leaf_method, LeafMethod::Gradient | LeafMethod::Simple)`
(Simple is byte-identical to Gradient, `leaf.rs:159-162`) until a Newton arm is
implemented on the device seam. Add a device_seam_test.rs case that asserts
`LeafMethod::Newton` + `Loss::Logloss` + otherwise-device-eligible params
falls back to the CPU path (or errors), rather than silently taking the
device branch.

```rust
&& matches!(params.leaf_method, cb_compute::LeafMethod::Gradient | cb_compute::LeafMethod::Simple)
```

## Warnings

### WR-01: Stale/misleading doc comment claims the grow loop never reads back `leaf_of`

**File:** `crates/cb-backend/src/gpu_runtime/mod.rs:1411-1414`
**Issue:** The doc comment on `read_u32_handle` states *"the grow loop itself
never reads the bulk routing back (D-05) — this is the test seam"*. That is no
longer accurate: `grow_oblivious_tree_resident` (the production per-tree path
`GpuTrainSession::grow_one` calls every boosting iteration) calls
`read_u32_handle(client, leaf_of_h)` at step (8) of every tree
(mod.rs:2104), i.e. an `n`-length device→host read-back *does* cross the seam
on every production tree today — contradicting the "no per-tree read-back"
invariant asserted throughout `session.rs`'s own module doc ("only... ONE
`leaf_of` read-back at the end of each tree — the SAME crossing class as the
part-stats"). The 10-07 SUMMARY documents this as a known, deferred
optimization ("production hot-path optimization... deferred until the CPU
fallback is unwired"), but the in-code comment on `read_u32_handle` itself is
now factually wrong and will mislead a future maintainer who reads only the
code.
**Fix:** Update the `read_u32_handle` doc comment to state that it is
currently used by both the cross-oracle test seam AND the production
`grow_oblivious_tree_resident` end-of-tree structure read (documenting it as
accepted, tracked debt) rather than claiming the grow loop "never" reads it back.

### WR-02: `bench/fixtures/README.md` and `generator.py` overstate what `cuda_oracle.ipynb` actually cross-validates

**File:** `bench/generator.py:339-355`, `bench/fixtures/README.md:1-54`
**Issue:** `write_fixtures()` emits, and the manifest sha256-pins,
`expected_inclusive_scan.npy`, `expected_exclusive_scan.npy`,
`expected_segmented_scan.npy`, `expected_sort_perm.npy`,
`expected_reduce_by_key_{keys,vals}.npy`, `expected_segmented_reduce.npy`,
`expected_cindex_packed_f0_bits8.npy`, and `cindex_small.npy`. The README
states: *"These files are the...expected values that `bench/cuda_oracle.ipynb`
loads on the Kaggle CUDA image to run the correctness gate (BENCH-01)."*
Inspecting the notebook shows this is not the case: the primitive/cindex
correctness gate (cell 5) runs exclusively via `cargo test --features cuda ...
scan segmented sort reorder reduce cindex update_part_props` — the in-tree
Rust self-oracle (device vs. an *inline Rust* serial reference) — and never
loads any of the `expected_*` `.npy` files. Only `X_small.npy`,
`y_small_reg.npy`, `y_small_bin.npy`, and `expected_depth1_tree.json` are
actually read by the notebook (cell 6). A repo-wide grep confirms no Rust
test, Python script, or notebook cell anywhere loads
`expected_inclusive_scan.npy` / `expected_sort_perm.npy` / etc. — they are
generated, sha256'd, and then never consumed.
**Fix:** either wire a Python-side comparison of these fixtures against the
Rust primitive test output (giving the claimed independent numpy-vs-Rust
cross-check), or correct the README/generator docstrings to state that the
primitive/cindex correctness gate is a Rust-internal self-oracle only, and
that these committed `.npy` files are presently unused reference data (or
drop them from the committed manifest to avoid churn/confusion).

### WR-03: Pre-existing one-hot pairwise-histogram double-counts weight into the same cell (carried-forward debt, in reviewed-file scope)

**File:** `crates/cb-backend/src/kernels.rs:1054-1066` (`pairwise_hist_nonbinary_kernel`, `one_hot` branch) and `crates/cb-backend/src/kernels.rs:1173-1184` (`pairwise_hist_8bit_atomics_kernel`, same pattern)
**Issue:** In the `one_hot` branch, the same cell is `fetch_add`ed twice with
the same weight:
```rust
bin_sums[cell1].fetch_add(w);
bin_sums[cell1].fetch_add(w);   // same cell, same value — doubles the weight
bin_sums[cell2].fetch_add(w);
bin_sums[cell2].fetch_add(w);
```
This predates Phase 10 (introduced in Phase 7.4, tracked in project memory as
"WR-01 pairwise one-hot fill double-writes (no one_hot oracle)") and the
`kernels.rs` file is squarely in this phase's review scope. The `one_hot`
overlay is "threaded now but exercised only by [a future] Plan" per the
kernel doc comment, so it is not yet reachable from any shipped one-hot
pairwise fixture, but as written it will silently double the accumulated
weight for every one-hot pairwise histogram cell the day it is wired up.
Flagging for visibility since Phase 10 builds substantial new infrastructure
(`read_bin`, resident sessions) directly adjacent to this code without
touching it.
**Fix:** when the one-hot overlay is exercised, either write `w` once to each
of two distinct channels (matching the non-one-hot "two channels" shape) or
document explicitly that one-hot pairwise intentionally uses 2w-per-cell
semantics with an oracle that asserts it. Track under the existing WR-01 debt
item; no action required this phase unless a later phase starts consuming the
`one_hot=true` arm.

## Info

### IN-01: `device_host_eligible` duplicates knowledge of the backend's own coverage gate without a single source of truth

**File:** `crates/cb-train/src/boosting.rs:2925-2938` vs `crates/cb-backend/src/gpu_runtime/session.rs:154-164`
**Issue:** Both `train_inner`'s host predicate and `GpuTrainSession::begin`
independently encode "depth==1, RMSE/Logloss/CrossEntropy, Plain, fold==1,
supported score fn" (the backend re-derives loss/score support via
`map_der_kernel`/`map_score_fn`; the host predicate does not check loss/score
at all and instead relies entirely on the backend call for that half). This
is deliberate per the 10-08 SUMMARY (two composed gates, host sees things the
backend cannot), but CR-01/CR-02 show the split has a gap: fields the CPU
*does* use for leaf estimation (`leaf_method`) and initial state
(`boost_from_average`/bias) are visible only to the host, not passed to the
backend, and not checked by the host gate either. A short "device
prerequisites" comment/struct co-located with `begin_device_training`'s
signature enumerating every `BoostParams` field the device grower is blind to
(and therefore must be host-gated) would make this class of gap easier to
catch in review.
**Fix:** non-blocking; consider a `debug_assert!`/doc block listing the
BoostParams fields NOT passed to `begin_device_training` as a checklist for
future device-coverage extensions.

### IN-02: `quantize_feature_major` computes a shared `n_bins` across all features from the max border count, wasting bits for narrower features

**File:** `crates/cb-train/src/boosting.rs:2106-2127`
**Issue:** `n_bins = max_f(feature_borders[f].len()) + 1` is used uniformly
for every feature's `pack_cindex` bucket count (`session.rs:201`,
`vec![n_bins; n_features]`), even though most features typically have fewer
borders. This is not a correctness bug (every feature's bin values are
`<= borders.len() < n_bins`, so `pack_cindex`'s per-feature bit-width sizing
via `feature_bits(n_bins)` never truncates), but it does size every feature's
packed field to the WIDEST feature's bit-width rather than its own, giving up
some of the bit-packing's memory-efficiency benefit (a CLAUDE.md first-class
constraint) for the depth-1 MVP's `n_features`-small case. Not urgent, but
worth tightening once `pack_cindex` grows a per-feature bucket-count entry
point from `quantize_feature_major`.
**Fix:** thread real per-feature border counts (`feature_borders[f].len() + 1`)
into `pack_cindex` instead of the uniform max, when convenient.

---

_Reviewed: 2026-07-03T03:03:50Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
