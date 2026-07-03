---
phase: 11-depth-1-partition-aware-histograms-reduction-determinism-new
reviewed: 2026-07-03T00:00:00Z
depth: standard
files_reviewed: 10
files_reviewed_list:
  - crates/cb-backend/src/kernels.rs
  - crates/cb-backend/src/gpu_runtime/mod.rs
  - crates/cb-backend/src/gpu_runtime/pairwise.rs
  - crates/cb-backend/src/kernels/grow_loop.rs
  - crates/cb-compute/tests/depth6_reference_test.rs
  - crates/cb-compute/Cargo.toml
  - bench/generator.py
  - bench/fixtures/manifest.json
  - bench/fixtures/expected_depth6_tree.json
  - bench/RESULTS.md
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 11: Code Review Report

**Reviewed:** 2026-07-03
**Depth:** standard
**Files Reviewed:** 10
**Status:** issues_found

## Summary

Reviewed the Phase-11 depth-6 partition-aware grow loop, the fixed-point `Atomic<u64>`
reduction-determinism path, the histogram subtraction trick, the Newton der2 leaf
estimation, the depth-6 CPU-oracle fixture generator/cross-check, and the Kaggle
sign-off harness. The correctness core is well-guarded: overflow-checked buffer sizing,
host-side value-range guards, typed errors instead of panics, `f32::MIN` sentinels in
`#[cube]` kernels (no `-inf` landmine), and self-oracles that assert structure EXACT and
leaf values ≤1e-4 on rocm/cuda. The channel-swap (channel-0 = weight in the partition
fill vs channel-0 = der1 in the depth-1 fill) is internally consistent between the
partition fill and the partition scorer.

No BLOCKER-class defects were proven in the scored/returned tree path — the depth-6
structure and leaf-value oracles would catch those in-env. However, several real
defects sit just outside what those oracles exercise: the subtraction trick is
dead/incorrect, the new `Atomic<u64>` fill silently widened the backend requirement of
pre-existing depth-1 tests that were never re-guarded, and the "reduction determinism"
guarantee is narrower than the phase framing claims (it covers the histogram fill but
not the leaf-stat reduce that produces leaf values).

No structural findings block was provided.

## Narrative Findings (AI reviewer)

## Warnings

### WR-01: Subtraction trick is computed-then-discarded, uses the wrong sibling pairing, and its self-oracle validates a partition convention production never produces

**File:** `crates/cb-backend/src/gpu_runtime/mod.rs:2366-2383` (and kernel `crates/cb-backend/src/kernels.rs:3748-3777`; test `crates/cb-backend/src/kernels/grow_loop.rs:1668-1723`)

**Issue:** In `grow_oblivious_tree_into` the subtraction trick is run per level but its
result is thrown away:

```rust
let _bigger = launch_subtract_histograms_into( … )?;   // result discarded
```

The comment concedes the direct `hist_h` fill (all `2^level` slots) is what actually
gets scored, so:

1. **The D-04 "memory-lean" claim is not realized.** The kernel fills every partition
   slot directly *and then also* runs the subtraction, doing strictly more work, not
   less. The optimization delivers zero benefit.
2. **The sibling pairing is wrong for the forward-bit routing.** The loop uses
   `smaller_child = pair * 2 + 1` and `parent slot = pair`, i.e. it assumes children are
   `2k` / `2k+1`. But `partition_split_kernel` uses forward-bit routing
   (`new_leaf |= 1 << level`), so the two children of parent `p` at level `L` are `p`
   and `p | (1 << (L-1))` — never `2p` / `2p+1` (except at level 1). If anyone later
   wires `_bigger` into the scorer, it will subtract the wrong parent/child slots and
   produce corrupt histograms.
3. **The self-oracle tests a different convention than production.** `partition_aware_hist_matches_cpu_scatter`
   fabricates `parent_of = child_of >> 1` (a `2k`/`2k+1` pairing) and asserts the
   subtraction against *that*. It therefore passes without ever exercising the
   forward-bit pairing the real grow loop produces — a green test that does not certify
   the production path.

Because the scored/returned tree uses the direct fill, current output is correct, so
this is not a BLOCKER — but it is dead code, a latent correctness bug, an unmet D-04
optimization claim, and a misleading self-oracle.

**Fix:** Either (a) delete the discarded per-level subtraction call from
`grow_oblivious_tree_into` and drop the "memory-lean" claim until it is actually wired
in, or (b) wire the derived `bigger` into the score step and correct the sibling
pairing to the forward-bit convention:

```rust
// children of parent p at level L: p (pass=0) and p | (1 << (level-1)) (pass=1)
let sibling_bit = 1usize << (level - 1);
for p in 0..n_parents {
    let child0 = p;                // directly filled
    let child1 = p | sibling_bit;  // derive via parent - child0 (pick smaller by size)
    …
}
```

and update `partition_aware_hist_matches_cpu_scatter` to route `leaf_of` through the
real forward-bit split sequence rather than a synthetic `child >> 1` parent.

### WR-02: Depth-1/depth-2 grow & boosting tests route through the `Atomic<u64>` fixed-point fill but were never re-guarded; production `grow_oblivious_tree` performs no u64-atomic capability gate

**File:** `crates/cb-backend/src/kernels/grow_loop.rs:569, 661, 750, 1170` (tests); `crates/cb-backend/src/gpu_runtime/mod.rs:2355, 1760-1861` (production)

**Issue:** Phase 11 replaced the depth-1 grow-loop histogram fill with
`launch_partition_hist2_into` at *every* level (including level 0), so
`partition_hist2_nonbinary_kernel` (which accumulates into `&Array<Atomic<u64>>`) is now
on the depth-1 path too. The new depth-6 / Newton / partition-hist tests correctly gate
on `cfg!(any(feature = "rocm", feature = "cuda"))` because "cpu/wgpu lack `Atomic<u64>`
add" (their own SKIP comments). But the pre-existing depth-1 grow/boosting tests were
not re-guarded:

- `single_tree::matches_cpu_greedy_search` (569)
- `single_tree::cosine_matches_cpu_cosine_greedy_search` (661)
- `single_tree::depth_gt_one_is_device_covered` (750)
- `multi_tree::matches_cpu_multi_tree_boosting` (1170)

All four call `grow_oblivious_tree` / `grow_boosting_pass` → `launch_partition_hist2_into`
→ the `Atomic<u64>` kernel, with no backend guard. By the code's own admission they will
**error (not skip)** on cpu/wgpu, which is inconsistent with the guarded depth-6 tests.

Worse, production `grow_oblivious_tree_into` calls `launch_partition_hist2_into`
unconditionally, while `launch_partition_hist2_resident_into` claims (mod.rs:1823-1824)
"the caller gates this path on the device's advertised capability." No caller performs a
`device_supports_..._u64_atomic` check — the tests gate at compile time via `cfg!`, and
production does not gate at all. On a wgpu build, `grow_oblivious_tree` would attempt to
launch a kernel wgpu cannot run.

**Fix:** Add the same `if !cfg!(any(feature = "rocm", feature = "cuda")) { return; }`
skip guard to the four unguarded tests (matching the depth-6 tests), and in production
add a real `device_supports_f64_atomic_add`-style `Atomic<u64>` capability check in
`grow_oblivious_tree_into` (or its launcher) that surfaces a typed `CbError` before
launch on a backend without u64 atomics. Correct the resident launcher comment to match
whichever gate actually exists.

### WR-03: Reduction determinism covers only the histogram fill — the leaf-stat reduce (`partition_update_kernel`) still uses non-deterministic float atomics, so leaf values / predictions are not bit-deterministic

**File:** `crates/cb-backend/src/kernels.rs:3546-3581` (`partition_update_kernel`)

**Issue:** The phase's headline determinism property (fixed-point `Atomic<u64>`,
order-independent integer add — GPUT-06) is applied to `partition_hist2_nonbinary_kernel`
only. The leaf statistics that feed the actual leaf values are produced by
`partition_update_kernel`, which still merges with a naked float atomic:

```rust
part_stats[part * 3usize].fetch_add(d);          // Atomic<F>::fetch_add — float, order-dependent
part_stats[part * 3usize + 1usize].fetch_add(w);
part_stats[part * 3usize + 2usize].fetch_add(h);
```

Consequently: tree STRUCTURE (splits + `leaf_of`) is deterministic (it derives from the
fixed-point histogram scoring), but LEAF VALUES and therefore model PREDICTIONS are not
bit-identical run to run, even on rocm/cuda. The determinism self-oracle
(`partition_hist_reduce_zero_spread`) only checks the *fill*; leaf-value spread is
untested. This is narrower than the "reduction determinism" phase title and the
RESULTS.md "Per-tree run-to-run spread … 0 spread" gate imply (that gate holds for
structure, not for leaf values / predictions).

Float ulp-level spread compounded over hundreds of trees stays well within the ε=1e-4
bar, so this is not a correctness BLOCKER — but a strict bit-reproducibility claim for
predictions is not met.

**Fix:** Either (a) route `partition_update_kernel` through the same fixed-point
`Atomic<u64>` accumulate path as the histogram fill (encode `d`/`w`/`h`, integer atomic
add, decode on read-back) to make leaf values bit-deterministic, or (b) explicitly scope
the determinism guarantee to tree structure in the phase docs / RESULTS.md and add a
run-to-run leaf-value spread REPORT so the residual float-order variance is characterized
rather than implied to be zero.

### WR-04: Fixed-point encode silently wraps for accumulated magnitudes above ~2^33, with no guard

**File:** `crates/cb-backend/src/kernels.rs:3605-3608` (`fixedpoint_encode`) and the `Atomic<u64>` accumulation in `partition_hist2_nonbinary_kernel:3718-3719`

**Issue:** `fixedpoint_encode` scales by `2^30` and stores the two's-complement `i64`
bits in the `Atomic<u64>`. The accumulated per-bin sum is exact only while
`|sum| · 2^30 < 2^63`, i.e. `|sum| < 2^33 ≈ 8.6e9`. Beyond that the wrapping `u64` add
silently overflows into a wrong (sign-flipped) value with no error and no saturation.
For the committed workloads (correctness 2000×10; speed ~1e6×50 with O(1)–O(10) targets)
this is safe, but a large-`n` or large-magnitude der1/weight fixture could cross the
threshold undetected. The decode doc mentions the `2^53` float-exactness bound but
nothing enforces the tighter `2^33` fixed-point-range bound.

**Fix:** Document the `|Σ| < 2^33` fixed-point range precondition next to
`REDUCE_FIXEDPOINT_SCALE_F64`, and either add a host-side pre-launch magnitude estimate
(e.g. `n * max|der1| * 2^30 < i64::MAX`) that surfaces a typed `CbError::OutOfRange`, or
lower `k` / widen the accumulator strategy for very large `n`.

## Info

### IN-01: Depth-1 serial reference skips degenerate splits while depth-6 and the device do not

**File:** `bench/generator.py:304-305` (`serial_depth1_tree`) vs `:346-357` (`_cosine_split_score` / `serial_depth6_tree`)

**Issue:** `serial_depth1_tree` skips any candidate with an empty side
(`if wl <= 0.0 or wr <= 0.0: continue`), whereas `_cosine_split_score` (depth-6) and the
device partition scorer deliberately permit empty sides (`avg = 0/(0+l2) = 0`). On the
2000×10 gaussian fixture an interior border producing an all-one-side split is unlikely,
so the depth-1 fixture is probably unaffected — but a device depth-1 split landing on a
degenerate side would not match `expected_depth1_tree.json`. The two references use
different candidate-admissibility rules for the same "Cosine best split" contract.

**Fix:** Make `serial_depth1_tree` consistent with the device (score degenerate sides
with the `l2`-seeded zero-average fold rather than `continue`), or document why depth-1
uses a stricter admissibility rule than depth-6.

### IN-02: Misleading capability-gating comment in the resident partition-fill launcher

**File:** `crates/cb-backend/src/gpu_runtime/mod.rs:1822-1824`

**Issue:** The comment states "wgpu lacks u64 atomics — the caller gates this path on the
device's advertised capability," but no caller (production `grow_oblivious_tree_into`,
the tests, or `grow_boosting_pass`) performs a device-capability query; gating is
compile-time `cfg!` in tests and absent in production (see WR-02). The comment overstates
the safety actually present.

**Fix:** Update the comment to reflect the real gate once WR-02 is addressed.

### IN-03: Per-object fixed-point rounding makes the device histogram a quantized approximation of the CPU `sum_f64` reference

**File:** `crates/cb-backend/src/kernels.rs:3718-3719` (encode-per-contribution) vs `crates/cb-compute` `reduce_leaf_stats`

**Issue:** Each per-object contribution is rounded to the `2^30` grid before the integer
atomic add, so the device bin sum is `Σ round(vᵢ·2^30)/2^30`, not the CPU
`sum_f64(vᵢ)`. Per-term error ≤ 2^-31; over `n` objects this is bounded by
`n·2^-31` (≈2.3e-7 for n=500, ≈4.7e-4 at n≈1e6). Comfortably inside the ε=1e-4 grow bar
for the tested sizes, but the margin shrinks with `n`, and the CUDA-oracle preds compound
this across ~200 trees. This is inherent to the fixed-point strategy and acknowledged by
the tolerance framing — noted so the shrinking headroom at large `n` is on record for the
Kaggle sign-off.

**Fix:** No change required for the committed fixtures. Track the effective error vs `n`
in the Kaggle per-tree diagnostic so the 1e-4 margin at ~1e6 rows is confirmed, not
assumed.

---

_Reviewed: 2026-07-03_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
