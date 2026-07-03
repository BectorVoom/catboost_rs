---
phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit
fixed_at: 2026-07-03T00:00:00Z
review_path: .planning/phases/10-gpu-foundations-runtime-seam-session-residency-device-primit/10-REVIEW.md
iteration: 1
findings_in_scope: 6
fixed: 6
skipped: 0
status: all_fixed
---

# Phase 10: Code Review Fix Report

**Fixed at:** 2026-07-03
**Source review:** .planning/phases/10-gpu-foundations-runtime-seam-session-residency-device-primit/10-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 6 (fix_scope = all → Critical + Warning + Info)
- Fixed: 6
- Skipped: 0

All fixes were applied inside an isolated git worktree, verified with a targeted
`cargo check` for each touched crate (cb-backend, cb-train) plus a regenerated,
byte-verified fixture set for the benchmark oracle, and committed atomically
per finding.

## Fixed Issues

### CR-01: Device coverage gate does not reject an un-dispatchable `n_bins`

**Files modified:** `crates/cb-backend/src/gpu_runtime/session.rs`
**Commit:** 42fcf75
**Applied fix:** Added a dispatchable-`n_bins` decline to `GpuTrainSession::begin`,
immediately after the existing degenerate (`n == 0 || n_features == 0 ||
n_bins == 0`) check:

```rust
if !matches!(n_bins, 2 | 16 | 32 | 64 | 128 | 256) {
    return Ok(None);
}
```

The histogram fill `hist2_launch_resident` can only dispatch the line sizes
`BINARY_BINS (2)`, `HALF_BYTE_BINS (16)`, and the non-binary `{32,64,128,256}`.
Any other `n_bins` (e.g. the default 254-border quantization → `n_bins = 255`)
previously committed the whole fit to the device (D-10-01 all-or-nothing) and
then hard-failed at grow time — violating the D-04 "decline → CPU fallback,
never a hard failure" contract. The gate now returns `Ok(None)` so those fits
fall back to the byte-unchanged CPU grower. A comment ties the constant set back
to the kernel dispatch so the two cannot silently drift.

**Verification note (human/logic):** This is a coverage-gate (decline) change.
It compiles (`cargo check -p cb-backend`) and the decline semantics are
straightforward, but the *set* `{2,16,32,64,128,256}` must remain in lock-step
with `hist2_launch_resident`'s dispatch. Recommend a follow-up test that fits with
`n_bins ∉` the dispatch set and asserts the CPU-fallback (rather than a hard
error), since the existing suite pins `n_bins = 32` only.

### WR-01: Depth-1 benchmark oracle computed L2 but was labeled "Cosine"

**Files modified:** `bench/generator.py`, `bench/fixtures/expected_depth1_tree.json`,
`bench/fixtures/manifest.json`
**Commit:** 6da44a2 (shared with WR-02)
**Applied fix:** Replaced the L2 / variance-reduction numerator
(`sl²/(wl+l2) + sr²/(wr+l2)`) in `serial_depth1_tree` with the true **Cosine**
score matching the device Cosine arm (`kernels.rs` `find_optimal_split_kernel`
and `score.rs`): the same folded numerator divided by
`sqrt(1e-100 + Σ (sum/(w+l2))²·w)` over the two leaves. The docstring and
`fixtures/README.md` already state "Cosine score", so the code now matches the
documented (and shipped-device) score function. Regenerated the committed
`expected_depth1_tree.json` + `manifest.json` and re-verified byte-for-byte with
`python3 bench/generator.py --check bench/fixtures` (17/17 reproduce).

On this seed the selected split (`best_feature`, `leaf_left`, `leaf_right`) is
unchanged for both `rmse` and `logloss`; only the recorded `score` value now
reflects Cosine, so the oracle's `≤1e-5` claim is now against the correct score.

### WR-02: Depth-1 oracle reference used weighted der; shipped path uses unweighted der

**Files modified:** `bench/generator.py`, `bench/fixtures/expected_depth1_tree.json`,
`bench/fixtures/manifest.json`
**Commit:** 6da44a2 (shared with WR-01)
**Applied fix:** Dropped the `der1 = der1 * weights` multiply in
`serial_depth1_tree`, folding object weight only into the `wl`/`wr` count
denominators. This matches the shipped unweighted-der convention
(`reduce_leaf_stats` → `sum_f64(deltas)`; device histogram channel 0 =
`Σ der1` unweighted). Numerically identical on this unit-weight fixture, but the
reference now stays valid if a weighted fixture is ever added.

WR-01 and WR-02 both rewrite the same `serial_depth1_tree` block and both feed the
single regenerated fixture, so they are committed together (they cannot be split
into two independent generator runs).

### WR-03: Device fit ignored object weights in the leaf/score numerator with no uniform-weight gate

**Files modified:** `crates/cb-train/src/boosting.rs`
**Commit:** 447dd83
**Applied fix:** Added `&& weights.iter().all(|&w| w == 1.0)` to
`device_host_eligible`. The device grow path sums unweighted `Σ der1` into
histogram channel 0 and estimates leaves via `calc_average(Σ der1, Σ weight, l2)`
(weight enters only the denominator), which does not match upstream CatBoost's
`Σ w·der` (`SumWeightedDelta`). Until weighted-der is wired through the
histogram/leaf math, a genuinely weighted pool now falls back to the CPU grower
(D-04) rather than silently producing leaves that cannot meet the `≤1e-5`
upstream-parity bar. The float-equality predicate is consistent with the existing
`bias == 0.0` / `random_strength == 0.0` clauses in the same gate.

**Verification note (human/logic):** Eligibility-gate (decline) change; compiles
(`cargo check -p cb-train`). Documented against the upstream `SumWeightedDelta`
convention in an inline comment.

### IN-01: `read_bin` computed `offset + obj` in `u32` before widening to `usize`

**Files modified:** `crates/cb-backend/src/kernels.rs`
**Commit:** 52eb7bc
**Applied fix:** Widened before adding — `cindex[offset as usize + obj as usize]`
instead of `cindex[(offset + obj) as usize]` — so the index is computed in the
`usize` domain (matching the `feature * n_bins + bin` cell arithmetic in the
histogram consumers) rather than in `u32` where the sum could wrap. Not reachable
today (needs a >4-billion-element cindex), but removes the latent foot-gun.

**Verification note (GPU):** This is a `#[cube]` device-kernel edit. It compiles
under the default `cpu` feature (`cargo check -p cb-backend`) and is a pure
index-cast reordering that mirrors existing `usize`-domain patterns in the same
file — no float types, no `-inf` literal introduced. Per the project's CubeCL
discipline, a rocm in-env run of the cindex/histogram device oracles is
recommended as the final confirmation (the orchestrator/verifier discharges GPU,
since GPU tests cannot run inside this fixer).

### IN-02: `bench/RESULTS.md` primitive-gate bars inconsistent with README/notebook

**Files modified:** `bench/RESULTS.md`
**Commit:** 0406783
**Applied fix:** Aligned the RESULTS sign-off template with the authoritative bars
in `bench/fixtures/README.md` and `bench/cuda_oracle.ipynb` (notebook lines
17–20). Verified against the ground truth: the notebook gate itself states the
float primitive reduces (scan, segmented scan, reduce-by-key vals,
segmented-reduce, `update_part_props`) are `≤1e-4`, while integer/INDEX outputs
(sort perm, reduce-by-key **keys**, cindex) are **bit-exact**. `kernels/reduce.rs`
"byte-identical" is run-to-run DETERMINISM, not CPU bit-parity, so the float bars
must stay at `≤1e-4` — tightening them (as the finding's premise implies) would
falsely contradict the real gate and could cause spurious oracle failures.

The concrete edit: split the single ambiguous `reduce-by-key` row into
`reduce-by-key (keys)` → bit-exact and `reduce-by-key (vals)` → `≤1e-4` (matching
the README's `expected_reduce_by_key_{keys,vals}` split), and added a header note
documenting the bar semantics so a reviewer filling the template records one
consistent, correct bar per primitive.

---

_Fixed: 2026-07-03_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
