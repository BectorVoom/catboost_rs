---
phase: 02-data-layer-pool-quantization-reduction
reviewed: 2026-06-13T00:00:00Z
depth: standard
files_reviewed: 16
files_reviewed_list:
  - crates/cb-core/src/error.rs
  - crates/cb-core/src/lib.rs
  - crates/cb-core/src/reduction.rs
  - crates/cb-data/src/borders.rs
  - crates/cb-data/src/cat_hash.rs
  - crates/cb-data/src/ingest/arrow.rs
  - crates/cb-data/src/ingest/mod.rs
  - crates/cb-data/src/ingest/owned.rs
  - crates/cb-data/src/ingest/polars.rs
  - crates/cb-data/src/lib.rs
  - crates/cb-data/src/nan_mode.rs
  - crates/cb-data/src/pool.rs
  - crates/cb-data/src/quantize.rs
  - crates/cb-data/src/quantized_pool.rs
  - crates/cb-data/src/weights.rs
  - scripts/check-no-raw-float-sum.sh
findings:
  critical: 2
  warning: 6
  info: 5
  total: 13
status: fixes_applied
fix:
  fixed_at: 2026-06-13T00:00:00Z
  fixed: [CR-01, CR-02, WR-01, WR-02, WR-03, WR-04, WR-05, WR-06]
  also_addressed: [IN-01]   # bin_of/nan_bin debug_assert guard, same as WR-06
  deferred: [IN-02, IN-03, IN-04, IN-05]   # Info-only; not in fix scope
  verification:
    cargo_test_workspace: pass
    check_no_raw_float_sum: pass
    clippy_cb_core_lib: pass
    clippy_cb_data_lib: pass
    phase2_oracles: pass   # borders x2, cat_hash x1, quantize x1, weights x2
  notes: >
    IN-02 (dead insert_sentinel/appends_max_sentinel) is resolved transitively:
    CR-01 now wires insert_sentinel into the quantize driver for NanMode::Max.
    WR-01 reproduces the libstdc++ std::priority_queue heap tie-break and is the
    one logic-sensitive change — flagged for human verification, though all
    borders/quantize oracles remain green which is strong evidence of parity.
status_note: >
    All 2 Critical and all 6 Warning findings fixed and committed atomically on
    branch gsd-reviewfix/02 (fast-forwarded into the working branch). New
    regression tests added for CR-01 (NaN top-bin under Max), CR-02 (null->NaN +
    categorical-null rejection on Arrow and Polars), WR-01 (permutation
    invariance + constant column), and WR-02 (pair id bounds).
---

# Phase 2: Code Review Report

**Reviewed:** 2026-06-13
**Depth:** standard
**Files Reviewed:** 16
**Status:** issues_found

## Summary

Reviewed the phase-02 data layer: the deterministic reduction primitive, the CityHash64
port + perfect-hash remap, GreedyLogSum border selection, NanMode/bin assignment, the
quantization driver, the immutable QuantizedPool, auto class weights, and the
owned/Arrow/Polars ingestion seam.

The summation discipline (D-07/D-08) is held well: every float fold in library code
routes through `cb_core::sum_f64`. The CityHash port reads as a faithful, panic-free
transcription with consistent `wrapping_*` arithmetic. Error handling is typed
(`thiserror`), and no `unwrap`/`expect`/`panic` appears on production paths.

However, two parity-critical correctness defects exist. First, the quantization driver
**never inserts the `NanMode::Max` sentinel**, so a public-API caller using `nan_mode = Max`
on a NaN-bearing column collides NaN with the largest real-value bin instead of giving NaN
a dedicated top bin — a divergence from upstream on a reachable path. Second, Arrow/Polars
ingestion reads the raw value buffer and **ignores the Arrow null bitmap**, so null entries
are silently materialized as undefined values (typically `0.0`) and the NaN-in-categorical
guard never sees them. Both undermine the ≤10⁻⁵ parity mandate.

Several parity/robustness warnings follow (tie-break order in the greedy split, unvalidated
ranking pairs, inconsistent error-variant usage).

## Critical Issues

### CR-01: `NanMode::Max` sentinel is never inserted by the quantize driver — NaN collides with the top real bin

**File:** `crates/cb-data/src/quantize.rs:90-108`
**Issue:**
`QuantizeParams.nan_mode` is public and accepts `NanMode::Max`. When a float column contains
a `NaN`, `feature_mode = params.nan_mode` (line 81-85). Border selection is then driven only
by `prepend_min_sentinel = feature_mode.prepends_min_sentinel()` (line 90), and
`select_borders_greedy_logsum` only ever prepends the *Min* sentinel — it has no `Max` path.
The `f32::MAX` border that `nan_mode::insert_sentinel`/`appends_max_sentinel` are supposed to
append is **never added** to `borders_f32`.

At bin-assignment time, `nan_bin(NanMode::Max, &borders_f32)` returns `borders_f32.len()`
(nan_mode.rs:141) — the bin *above* the highest real border. But because the `f32::MAX`
sentinel is absent, real values greater than the highest real border *also* land in
`borders_f32.len()` via `bin_of`. NaN and the largest real values are therefore packed into
the **same** bin.

Upstream appends `numeric_limits<float>::max()` precisely so NaN gets a dedicated top bin that
no finite value can reach (`quantization.cpp` NanMode insertion; documented in nan_mode.rs:14-19).
This is a parity-breaking correctness bug on a reachable public path (the default `Min` masks it,
which is why `appends_max_sentinel`/`insert_sentinel` are dead — see IN-02).

**Fix:** Wire the Max sentinel into the driver. After border selection, apply the appended
sentinel for `Max` features before packing/bin assignment, e.g.:
```rust
let borders_f32: Vec<f32> = if feature_mode.appends_max_sentinel() {
    crate::nan_mode::insert_sentinel(feature_mode, &borders_f32_real)
} else {
    borders_f32_real
};
```
and ensure `select_borders_greedy_logsum` reserves the budget (`reserved_border_budget`) for the
Max case symmetrically to Min. Add an oracle test for a NaN-bearing column under `nan_mode = Max`.

### CR-02: Arrow/Polars ingestion ignores the Arrow null bitmap — nulls become undefined values, NaN-in-categorical guard bypassed

**File:** `crates/cb-data/src/ingest/arrow.rs:64-77` (and `polars.rs:73-81`)
**Issue:**
`arrow_f64_column` reads `typed.values()` — the raw contiguous **value buffer** — and copies it
out with `values.to_vec()`. For a `Float64Array` carrying a validity (null) bitmap, the slots
corresponding to null entries hold **arbitrary/undefined** payload (commonly `0.0`), and
`values()` does not consult the null bitmap at all. Consequences:

1. A null feature value is silently materialized as `0.0` (or whatever the buffer holds) instead
   of the missing-value `NaN` the rest of the pipeline expects. Downstream quantization will bin
   it as a real `0.0` rather than routing it through NanMode handling — a silent correctness/parity
   divergence (a Pandas/Polars `null`/`NaN` is exactly how missing values arrive).
2. The categorical guard (lines 66-73) iterates `values` and checks `is_nan()`, but a null is **not**
   stored as NaN in the value buffer — so a smuggled missing categorical value passes the
   `NanInCategorical` check (threat T-02-14) undetected.

The doc comment claims a "zero-copy view of the contiguous backing buffer" but never reconciles
nulls.

**Fix:** Either reject any column with `null_count() > 0` as a typed error, or normalize nulls to
`NaN` (for numeric features) / reject them (for categorical) by consulting the validity bitmap:
```rust
if typed.null_count() > 0 {
    if categorical {
        return Err(CbError::NanInCategorical { column: column_index });
    }
    // numeric: map nulls -> NaN explicitly
    return Ok((0..typed.len())
        .map(|i| if typed.is_null(i) { f64::NAN } else { typed.value(i) })
        .collect());
}
```
The Polars path inherits this via the shared funnel; additionally confirm `cont_slice()` /
`f64()` behavior on a nullable column rather than assuming non-null.

## Warnings

### WR-01: Greedy-split tie-break uses first-occurrence, not the C++ `priority_queue` pop order

**File:** `crates/cb-data/src/borders.rs:267-307`
**Issue:** Upstream `GreedySplit` pops from `std::priority_queue<TBinType>` keyed on `Score()`.
On **tied** best-scores the heap's pop order is determined by heap structure, not insertion order.
The Rust `arg_max_score` (lines 297-307) deterministically returns the **first** bin with the max
score (`bin.best_score > best_score` keeps the earliest on ties). When two bins share the top score,
the two implementations may split different bins, producing different intermediate boundaries.
Because final borders are deduped into a set and sorted, this only diverges when the tie leads to a
different final border set — but the project's own oracle exercises duplicate columns where ties are
plausible. For a bit-exact parity target this is a real risk.
**Fix:** Reproduce the STL binary-heap tie ordering, or prove (via an oracle test over
duplicate/constant columns and small `max_borders`) that tie-break order cannot change the final
sorted border set. Document the proof next to `arg_max_score` if the latter.

### WR-02: Ranking `pairs` are never validated against `n_rows` in the ingestion seam

**File:** `crates/cb-data/src/ingest/owned.rs:129-176` (also `pool.rs:62`)
**Issue:** `into_pool` validates every feature/metadata column length but **skips `pairs`
entirely** — `Pair.winner_id` / `loser_id` are `u32` row indices that are never bounds-checked
against `n_rows`. The doc on `Pool` (pool.rs:33-34) claims a Pool from this path is "guaranteed
internally length-consistent," which is false for pairs. A pair referencing a non-existent row is a
latent out-of-bounds for any downstream code that indexes objects by pair id (threats T-02-04/05 that
this seam is meant to close).
**Fix:** In `into_pool`, validate `winner_id < n_rows && loser_id < n_rows` for every pair and return
`CbError::OutOfRange` (or a dedicated variant) otherwise. Update the `Pool` doc to scope its
consistency guarantee accurately.

### WR-03: `cont_slice()` / value-buffer read assumes a NaN-free, null-free Polars column

**File:** `crates/cb-data/src/ingest/polars.rs:73-81`
**Issue:** `column.rechunk()` then `f64()?.cont_slice()?` is documented as erroring on a "multi-chunk
/ nullable column," but `cont_slice` on many Polars versions returns the contiguous value buffer for a
single-chunk column **even when it has nulls** (nulls present as buffer garbage, validity tracked
separately), or silently materializes nulls. Combined with CR-02, a nullable Polars feature can reach
`arrow_f64_column` with its null information stripped. The "non-contiguous after rechunk" error message
also conflates two distinct failure modes (nullability vs. fragmentation).
**Fix:** Explicitly check `column.null_count() == 0` (or handle nulls) before `cont_slice`, and split
the error messages for the nullable vs. non-contiguous cases so the boundary contract is enforced, not
assumed.

### WR-04: `check_len` builds an `OutOfRange` string instead of the dedicated `LengthMismatch` variant

**File:** `crates/cb-data/src/ingest/owned.rs:119-127`
**Issue:** `CbError::LengthMismatch { column, expected, actual }` exists for exactly this case and is
used by the Arrow path (`arrow.rs:120-124`), but the owned path hand-formats the same information into
`CbError::OutOfRange(String)`. Callers that `match` on the error type to react to a shape mismatch
(PYAPI-05 maps typed errors to Python exceptions) cannot distinguish a length mismatch from any other
range violation here, and the two ingestion paths report the same logical error with different variants.
**Fix:** Return `CbError::LengthMismatch { column: name.into(), expected, actual }` from `check_len`,
matching the Arrow path.

### WR-05: `max_summary_weight_f32` applies a `.max(0.0)` floor absent from upstream

**File:** `crates/cb-data/src/weights.rs:82-88`
**Issue:** Upstream computes `maxSummaryClassWeight = *MaxElement(summaryClassWeights...)` with no zero
floor (`calc_class_weights.cpp:79`). The Rust adds `.max(0.0)`. For non-negative weight sums this is a
no-op, but it diverges from the oracle if any summary weight is negative (e.g., a future signed-weight
path) and is undocumented behavior that masks a genuine difference rather than asserting the
precondition. It also silently substitutes `0.0` for the `NEG_INFINITY` seed when the slice is empty —
but an empty slice means `class_count == 0`, already rejected upstream of this call, making the floor
dead defensiveness that hides intent.
**Fix:** Drop `.max(0.0)` to match upstream's plain max, and instead assert/guarantee non-empty
(`class_count > 0` is already enforced in `summary_class_weights`), keeping the max bit-faithful.

### WR-06: `bin_of` cast `index as u32` can silently truncate for a categorical-sized border set

**File:** `crates/cb-data/src/nan_mode.rs:72-85`
**Issue:** `bin_of` is `pub` and generic over any `borders: &[f32]`. The comment asserts border counts
are "bounded by the budget (<= 65535 for float)," but the function does not enforce that — a caller
passing `> u32::MAX` borders (or, more realistically, a future categorical reuse) would truncate
`index as u32`. The invariant is documented but unchecked at the boundary of a public function.
**Fix:** Either restrict the function's contract (make it `pub(crate)` and centralize the budget
assertion), or `debug_assert!(borders.len() <= u32::MAX as usize)` and document that float callers
guarantee `< 65536`.

## Info

### IN-01: `nan_bin` / `bin_of` return `... as u32` on `usize` lengths without an explicit cap note at the call boundary

**File:** `crates/cb-data/src/nan_mode.rs:141`
**Issue:** `borders.len() as u32` mirrors the same unchecked-cast pattern as WR-06 for the NaN top-bin.
Harmless under the documented float budget, but the cast is silent.
**Fix:** Same remediation as WR-06; a single `debug_assert` covering both casts suffices.

### IN-02: Dead public API — `insert_sentinel` and `appends_max_sentinel` have no production callers

**File:** `crates/cb-data/src/nan_mode.rs:56-60, 95-115`
**Issue:** Both are exported (`lib.rs:40`) and exercised only by `nan_mode_test.rs`; the quantize driver
never calls them (root cause of CR-01). Until the Max path is wired, this is dead production surface.
**Fix:** Wire them in via CR-01; once wired, this resolves itself. If Max is intentionally deferred,
gate the public export and add a `// TODO(phase):` note per the project's TODO convention.

### IN-03: `total_object_weight` allocates an all-ones `Vec<f64>` solely to sum to a known count

**File:** `crates/cb-data/src/borders.rs:225-226, 250-255`
**Issue:** `select_borders_greedy_logsum` builds `vec![1.0_f64; values.len()]` and folds it through
`sum_f64` only to assert it equals `values.len()` in a `debug_assert`. This is an allocation purely to
satisfy the routing contract in the unweighted path, against the crate's first-class memory-efficiency
constraint. The result is discarded (`let _ = total_weight`).
**Fix:** When weights are genuinely needed (weighted binarization, a later plan), pass the real weight
slice. For the unweighted path, drop the throwaway allocation and document that the count is exact;
the D-07 routing contract is already satisfied where actual sums occur.

### IN-04: `Bin::left_border` / lookups use `.unwrap_or(f32::NAN)` masking would-be logic errors

**File:** `crates/cb-data/src/borders.rs:121, 153-154, 170, 186`
**Issue:** `values.get(idx).copied().unwrap_or(f32::NAN)` is panic-free, but substituting `NaN` for an
out-of-range index would silently corrupt a border (NaN propagates) rather than surfacing the indexing
bug. Under the current call structure these indices are always in range, so the fallback is unreachable
— but it converts a contract violation into silent data corruption instead of a detectable failure.
**Fix:** Where the index is provably valid, prefer a `debug_assert!`-guarded access or document the
invariant; reserve `unwrap_or(NaN)` for genuinely optional reads.

### IN-05: `pop_at` uses `Vec::remove` (O(n) shift) instead of documenting why order is preserved

**File:** `crates/cb-data/src/borders.rs:309-317`
**Issue:** The comment explains `swap_remove` is avoided to keep tie-break order deterministic — good —
but this couples correctness (WR-01) to an O(n) removal. Not a perf finding (out of v1 scope); flagged
only because the determinism rationale is the same one WR-01 questions. If WR-01 is resolved by matching
the STL heap order, this `remove`-based ordering should be revisited together.
**Fix:** Resolve alongside WR-01; if first-occurrence ordering is proven equivalent, keep and reference
the proof here.

---

_Reviewed: 2026-06-13_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
