---
phase: 21-cpu-split-finding-histogram-rewrite
reviewed: 2026-07-05T12:01:38Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - crates/cb-compute/src/histogram.rs
  - crates/cb-compute/src/lib.rs
  - crates/cb-train/src/tree.rs
  - crates/cb-train/Cargo.toml
  - Cargo.toml
findings:
  critical: 1
  warning: 4
  info: 1
  total: 6
status: issues_found
---

# Phase 21: Code Review Report

**Reviewed:** 2026-07-05T12:01:38Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

This phase replaces the per-candidate full-dataset rescan (`assign_leaves` +
`reduce_leaf_stats`) with a per-`(leaf, feature, bin)` `BucketHistogram`, an
`O(n_bins)` prefix scan, and a level-transition subtraction trick, then
parallelizes the per-feature work with rayon. The stated parity bar is
**bit-exact (`==`) against the prior object-order score math**.

**Rayon determinism is handled correctly** — every parallel site is
order-preserving: `bins.par_chunks_mut(n_objects)` writes disjoint feature
columns, and `(0..n_features).into_par_iter().map(..).collect::<Vec<Vec<_>>>()`
is an `IndexedParallelIterator` whose `collect` preserves index order before
`flatten`, reproducing the sequential feature-asc × border-asc enumeration. No
cross-thread float reduction is introduced. This is a genuine strength and not a
finding.

**The central problem is that the histogram rewrite does not, and structurally
cannot, meet the bit-exact bar.** `cb_core::sum_f64` is a deliberately
non-compensated, order-sensitive sequential fold (its own module doc: "Any
reordering of additions perturbs the result … and breaks the ≤1e-5 oracle gate").
The new code changes the addition order in two independent ways (bin-grouped
summation, and the parent−sibling subtraction trick). The code's own comments
concede this ("bit-exact … *on a benign fixture*"; "adversarial ULP ties are
gated by the downstream oracle suite"). Because the split winner is chosen with a
strict `>` first-wins tie-break, a single-ULP score perturbation can flip which
candidate wins and diverge the tree structure. That is the BLOCKER below.

The remaining findings concern silent-wrong-result defensive fallbacks and
memory-efficiency regressions (a first-class project constraint), several of which
are acknowledged as deferred but are flagged here for the record.

## Critical Issues

### CR-01: Histogram bin-grouped summation + subtraction trick violate the bit-exact parity bar

**File:** `crates/cb-compute/src/histogram.rs:377-393` (build), `:426-473` (scan), `crates/cb-train/src/tree.rs:710-792` (`hist_add` / `advance` subtraction trick)
**Issue:**
The parity bar for this phase is bit-exact equality with the prior score math,
whose `LeafStats` came from `reduce_leaf_stats`: for each child leaf, **all**
member objects were gathered in ascending object order and reduced by a **single**
`sum_f64` (one strictly left-to-right `f64` fold).

The new path produces the same `LeafStats` by a different addition tree:

1. `build_bucket_histogram` sums each `(leaf, feature, bin)` cell over its members
   (object order), then
2. `scan_border_to_leaf_stats` re-sums those per-bin totals across bins
   (`w_false = sum_f64(bins 0..=border)`), i.e. `Σ_bins ( Σ_objects_in_bin )`.

`sum_f64` is explicitly non-associative and non-compensated
(`crates/cb-core/src/reduction.rs:25-38`), so `Σ_bins(Σ_obj) ≠ Σ_obj` on
general float inputs. The FALSE-child objects are interleaved across bins in the
old object-order fold but grouped by bin in the new one — the two results differ
by ULPs on non-benign data. `crates/cb-compute/src/histogram.rs:420-424` admits
this directly: "On a benign fixture (values exactly representable) this is
bit-exact … adversarial ULP ties are gated by the downstream oracle suite."

`GrowScratch::advance` compounds the divergence: from level 1 onward the scored
histogram is derived as `child_large = parent − sibling_small`
(`tree.rs:774,790`, via `BucketHistogram::remove`), and
`sum_all − sum_small ≠ sum_large` under `sum_f64` rounding. So every level after
the root scores candidates from a subtraction-derived histogram that is not
bit-equal to a fresh object-order reduction.

Because `select_best_candidate` (`tree.rs:311-322`) uses a strict `>` first-wins
tie-break, a sub-ULP score change can flip the chosen split, changing tree
structure and cascading beyond the ≤1e-5 oracle tolerance. This is the exact
"summation-order / tie-break divergence" the phase brief calls out as a real bug
against the bit-exact bar.

Note the nuance for the fixer: upstream CatBoost itself uses histograms + the
subtraction trick, so this path may be *more* faithful to real CatBoost than the
prior rescan — but it is **not** bit-exact to the prior Rust score math, which is
the stated bar. Resolution requires one of: (a) confirm the accepted bar is the
≤1e-5 oracle (not `==`) and correct the "bit-exact"/"byte-for-byte" claims in the
doc comments (`histogram.rs:186,415-417`; `tree.rs:44-46,291-292,714-724`) so the
contract is not overstated; or (b) treat it as a genuine parity regression and
restructure so the child reduction reproduces the object-order fold.

**Fix:**
```text
Before merge, verify the phase's actual gate:
- If the bar is the ≤1e-5 oracle: run the full oracle/e2e train→predict suite
  on a NON-benign fixture (real-valued gradients, not exactly-representable
  toy data) and record the observed max deviation in the phase VERIFICATION.
  Then downgrade the "bit-exact"/"byte-for-byte" wording in the doc comments to
  "≤1e-5 oracle-equivalent" so the code does not claim a guarantee it cannot keep.
- If the bar is truly `==`: the histogram+subtraction path cannot satisfy it for
  arbitrary float inputs; the summation order must match reduce_leaf_stats
  (single object-order fold per child), which is incompatible with the
  subtraction trick. That contradiction must be resolved at the plan level.
```

## Warnings

### WR-01: `BucketHistogram::remove` silently returns the receiver unchanged on shape mismatch

**File:** `crates/cb-compute/src/histogram.rs:268-290`
**Issue:**
On any shape mismatch `remove` returns `self.clone()` — i.e. `a - b` silently
yields `a`, dropping the `b` operand entirely rather than surfacing the error.
This is threaded through `GrowScratch::hist_add` (`tree.rs:710-714`), which is
built from three chained `remove` calls (`a + b = a − (0 − b)`). If any operand
shape ever drifts (e.g. a future refactor of `n_bins`/`approx_dim`), `hist_add`
would silently return `a`, producing a numerically wrong histogram with no panic
and no error — the worst failure mode under a parity bar. The current call sites
happen to keep shapes equal, so this is latent, but a silent wrong-value branch on
a parity-critical primitive should fail loudly (or be unreachable by construction)
rather than fabricate a plausible result.
**Fix:** Return `Option<BucketHistogram>`/`CbResult<_>` from `remove` (and
propagate through `hist_add`/`advance`), or `debug_assert!` the shape equality so a
mismatch is caught in tests instead of silently corrupting the histogram. Do not
return `self.clone()` as a "defensive" success.

### WR-02: `build_bucket_histogram` allocates one heap `Vec` per histogram cell-channel on every call

**File:** `crates/cb-compute/src/histogram.rs:360-393`
**Issue:**
The builder allocates `members: Vec<Vec<f64>>` of length
`n_leaves * n_features * n_bins * n_channels`, i.e. one growable `Vec<f64>` per
`(leaf, feature, bin, channel)`, purely to route each cell through `sum_f64` and
satisfy the D-08 "no raw float fold" ban. For a modest tree (n_features=50,
n_bins=128, depth-6 → n_leaves=64, dim=1 → n_channels=2) that is ~820k small heap
allocations *per level*, and the per-leaf grower (`best_split_for_leaf`,
`tree.rs:1114-1123`) plus the subtraction trick (`advance`, up to 3 builds per
transition) call it repeatedly. This directly regresses the first-class
memory-efficiency constraint the phase brief flags. The equivalent object-order
scatter-add into a single flat `Vec<f64>` accumulator (`data[cell] += v` in object
order) would be numerically identical to the gather-then-`sum_f64` (same
left-to-right order per cell) at a fraction of the allocations.
**Fix:** Accumulate into a flat `vec![0.0f64; total]` with an object-order
scatter-add helper that is itself the sanctioned sequential fold (extend
`cb_core`'s reduction module with a `scatter_add`-style primitive so the D-08 ban
is honored without the per-cell `Vec`). The comment at `histogram.rs:356-359`
already flags scratch reuse as deferred to 21-05 — this allocation shape should be
part of that work and is called out here so it is not lost.

### WR-03: `scan_border_to_leaf_stats` is not the advertised `O(n_bins)` prefix scan — it re-gathers and re-sums per border

**File:** `crates/cb-compute/src/histogram.rs:426-490`
**Issue:**
`scan_borders_to_leaf_stats` calls `scan_border_to_leaf_stats` once per border,
and each call, for every parent leaf and every dimension, allocates fresh
`bin_weight` / `bin_delta` `Vec`s of length `n_bins` (`:440-441,448-450`) and
`sum_f64`s the whole `0..=border` / `border+1..n_bins` prefix again from scratch.
For a feature with `B` borders that is `O(B · n_bins)` work and `O(B · P · (1+D))`
`Vec` allocations, not the `O(n_bins)` running prefix the docstrings and module
header repeatedly claim ("ONE O(n_bins) prefix scan per border",
`tree.rs:846-848,933`). This is both a memory-efficiency regression (in scope) and
a misleading contract. Note: it does not change correctness, and any true running
prefix must keep the ascending-bin `sum_f64` order to stay consistent with CR-01.
**Fix:** Compute the per-bin weight/delta rows once per (parent, feature), then
derive each border's FALSE/TRUE split from a single left-to-right running prefix
plus the leaf total (`true = total − false`, keeping the ascending-bin fold), or
at minimum correct the "O(n_bins)" wording to reflect the actual `O(B·n_bins)`
cost and per-border allocation.

### WR-04: `GrowScratch::advance` rebuilds the parent histogram it already holds, doing ~3 full passes per level transition

**File:** `crates/cb-train/src/tree.rs:760-792`
**Issue:**
Both subtraction-trick branches build three full `O(n_objects)` histograms per
transition — e.g. the `n_true <= n_false` branch builds `true_high` (the small
sibling, necessary), `true_low`, and `base_low` (`:770-772`). `base_low` is a
from-scratch rebuild of the *parent* histogram (all objects at `pget(o)` in the
low slots), which is exactly the data already resident in `self.hist` (identical
objects, bins, and object order). The whole point of the subtraction trick is to
build only the smaller sibling and derive the rest from the retained parent; as
written it pays 3 full binning passes plus multiple `n_objects`-length `leaf_of`
`Vec` allocations (`true_high`, `true_low`, `base_low`, `next_leaf_of`), so it is
slower and more allocation-heavy than a single fresh rebuild would be. This
regresses the first-class memory constraint and defeats the optimization's stated
purpose.
**Fix:** Derive the larger sibling from the retained `self.hist` (relocated into
the correct forward-bit slots) minus the freshly-built smaller sibling, eliminating
the `base_low`/`base_high` rebuild and the extra `true_low`/`false_high` passes —
one small-sibling pass per transition instead of three full passes.

## Info

### IN-01: `build_bucket_histogram` computes `cell_base` with unchecked multiplication, contradicting the "returned empty rather than panicking" guarantee

**File:** `crates/cb-compute/src/histogram.rs:349-354,376`
**Issue:**
`total` is computed with `checked_mul(..).unwrap_or(0)` and documented (`:349`,
`:328-329`) as returning an empty histogram on overflow "rather than panicking"
(T-21-02). But if `total` did overflow to `0`, the object loop still runs and
computes `cell_base = ((leaf * n_features + feature) * n_bins + bin) * n_channels`
(`:376`) with **unchecked** `usize` arithmetic, which panics on overflow in debug
builds — so the no-panic guarantee is not actually upheld in the overflow case it
claims to handle. Unreachable in practice under `MAX_DEPTH = 16` and realistic
feature counts, hence Info, but the stated invariant is not met.
**Fix:** Either derive `cell_base` with `checked_mul`/`checked_add` and `continue`
on `None`, or document that the overflow guarantee holds only because the depth cap
bounds `total` below `usize::MAX` (and drop the "returned empty rather than
panicking" wording for the arithmetic that is not actually checked).

---

_Reviewed: 2026-07-05T12:01:38Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
