# Phase 11: Depth>1 Partition-Aware Histograms + Reduction Determinism + Newton Der2 - Pattern Map

**Mapped:** 2026-07-03
**Files analyzed:** 10 (new/modified) — 8 extend existing files, 2 are harness edits
**Analogs found:** 10 / 10 (every new behavior has an in-repo analog; this phase is composition + one new kernel behavior, not new infrastructure)

> **Standing landmines (apply to EVERY `cb-backend` file below):**
> - **No `cb-train` dep in `cb-backend`** — transcribe the CPU-reference formula inline (`newton_leaf_delta`, `calc_average`, `scale_l2_reg` are one-liners; copy them, don't import).
> - **No `-inf` float literals inside `#[cube]`** — use `f32::MIN` sentinel (HIP/gfx1100 JIT rejects `double(-inf)`); host code may keep `f64::NEG_INFINITY`.
> - **Deterministic reduce mandatory** — the histogram/partition atomic merge must use the LOCKED fixed-point `Atomic<u64>` path (SPIKE-REDUCTION §5b), never f64 `Atomic::fetch_add` for the accumulator; gfx1100 has no f64 atomic-add.
> - **Never read a `Handle` through a client other than the one that allocated it** — one `&client` threads the whole tree.
> - Run `cargo test -p cb-backend --no-default-features --features rocm <kernel>` after ANY `#[cube]` change (cpu/wgpu false-pass the `-inf` landmine).

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/cb-backend/src/kernels.rs` (new partition-aware `pointwise_hist2` `#[cube]` + subtraction-trick `#[cube]` + `partition_update_kernel` +der2 channel) | kernel (`#[cube]`) | streaming / event-driven (per-object atomic scatter) | `pointwise_hist2_nonbinary_kernel` (kernels.rs:656-719); `partition_update_kernel` (kernels.rs:3531-3561) | exact (role+flow) |
| `crates/cb-backend/src/gpu_runtime/mod.rs` (`grow_oblivious_tree_into` — remove depth>1 reject, per-level partition-aware score; `launch_partition_update_into` +der2; `read_part_stats_f64` widen; new partition-hist launch fn) | service (device grow driver) | request-response (per-level O(1) BestSplit host↔device) | `grow_oblivious_tree_into` (mod.rs:1711-1840); `launch_partition_update_into` (mod.rs:1534-1592) | exact (self-extension) |
| `crates/cb-backend/src/gpu_runtime/der_seams.rs` (wire `LoglossHessian` der2 + `const_der` RMSE into Newton path) | service (der seam) | transform (approx→der2, no readback) | `DerUnaryKernel::LoglossHessian` / `launch_der_unary_handle` (der_seams.rs:287-367) | reuse (no new code) |
| `crates/cb-backend/src/gpu_runtime/mod.rs` (Newton leaf value + `apply_leaf_delta` refinement) | service | transform (device leaf update) | `apply_leaf_delta_kernel` (kernels.rs:587-598) / `launch_apply_leaf_delta_into` (mod.rs:1881) | exact |
| `crates/cb-backend/src/kernels/reduce.rs` (consume fixed-point accumulator as step 0) | utility (deterministic reduce) | batch reduce | `block_reduce_fixedpoint_kernel` / `run_fixedpoint_reduce` (reduce.rs:731) + `REDUCE_FIXEDPOINT_SCALE_F64` | reuse (LOCKED) |
| `crates/cb-backend/src/kernels/grow_loop.rs` (depth-6 RMSE+Logloss + Newton self-oracle) | test (self-oracle) | request-response | `matches_cpu_greedy_search` (grow_loop.rs:475), `cpu_boosting_pass` (grow_loop.rs:733), `depth_gt_one_is_tracked_forward_dependency` (grow_loop.rs:653) | exact |
| `crates/cb-compute/src/histogram.rs` (`reduce_leaf_der2` — the Newton oracle) | model (CPU oracle) | batch reduce | `reduce_leaf_der2` (histogram.rs:100-121) — **already exists, reuse as-is** | reuse (present) |
| `crates/cb-compute/src/leaf.rs` (`newton_leaf_delta` — leaf math oracle) | model (CPU oracle) | transform | `newton_leaf_delta` (leaf.rs:145-154) — **already exists, reuse as-is** | reuse (present) |
| `bench/generator.py` (extend to depth-6 RMSE+Logloss configs, D-03) | config / fixture generator | batch / file-I/O | `serial_depth1_tree` (generator.py:230) + `write_fixtures` (generator.py:313) | role-match (extend) |
| `bench/cuda_oracle.ipynb` + `bench/RESULTS.md` (final-ε gate + per-tree diagnostic + depth-6 speed cells) | test / config (Kaggle harness) | request-response / batch | existing notebook depth-1 oracle cells + `RESULTS.md` sign-off log | role-match (extend) |

## Pattern Assignments

### `crates/cb-backend/src/kernels.rs` — partition-aware `pointwise_hist2` (GPUT-05, kernel, streaming)

**Analog:** `pointwise_hist2_nonbinary_kernel` (kernels.rs:656-719). The new `fullPass=false` variant is this kernel keyed by `leaf_of[obj]` into `2^level` slots.

**Core atomic-scatter pattern to replicate** (kernels.rs:687-718) — grid-stride loop, per-feature `read_bin`, FROZEN interleaved cell index, in-kernel atomic merge:
```rust
let stride = CUBE_COUNT * (CUBE_DIM as usize);
let mut i = ABSOLUTE_POS;
while i < n {
    let obj_u = indices[i];
    let obj = obj_u as usize;
    let d = der1[obj];
    let w = weight[obj];
    let mut feature = 0usize;
    while feature < n_features_usize {
        let bin = read_bin(cindex, offsets[feature], obj_u, shifts[feature], masks[feature]) as usize;
        let cell = (feature * n_bins + bin) * 2usize;   // ← EXTEND: prepend leaf offset
        bin_sums[cell].fetch_add(d);
        bin_sums[cell + 1usize].fetch_add(w);
        feature += 1usize;
    }
    i += stride;
}
```

**What changes for depth>1 (D-04 / Pattern 1 / A2):**
- Add a `leaf_of: &Array<u32>` input; read `let part = leaf_of[obj] as usize;` (same routing as `partition_update_kernel`, kernels.rs:3546).
- Cell index gains the leaf/partition stride — upstream `leafId * binFeatureCount * statCount + statId * binFeatureCount + binFeatureIndex` (§6.4). Concretely prepend `part * (n_features * n_bins * HIST_CHANNELS)` to the FROZEN `(feature * n_bins + bin) * 2` index. **Pin the exact addressing in the depth-6 fixture (A2).**
- **Channel-0 semantics (A2, MEDIUM confidence):** for the Logloss-Newton path channel-0 may carry Σder2 (hessian) rather than Σweight. Cross-check device split scores vs the CPU reference for the first few trees, not just the final prediction.
- **Deterministic accumulate:** swap the naked `fetch_add` for the LOCKED fixed-point `Atomic<u64>` accumulator (see reduce.rs analog below) — the atomic-order comment at kernels.rs:621-636 explicitly flags this merge as the non-deterministic source that GPUT-06 must remove.

**Subtraction-trick kernel (D-04, Pattern 2) — new `#[cube]`, no direct analog but mirror the elementwise cell-loop shape.** Upstream `SubstractHistogramsImpl`: `hist[bigChild][cell] = hist[parent][cell] - hist[smallChild][cell]` per bin, **clamping `statId==0` (weight/hessian channel) to `max(., 0)`** (numerical-underflow guard — LANDMINE: omitting the clamp yields tiny negative weights that poison the score denominator). Keep only **parent-level** histograms resident; reuse the parent slot for the next level.

---

### `crates/cb-backend/src/gpu_runtime/mod.rs` — `grow_oblivious_tree_into` depth>1 (GPUT-05, service, request-response)

**Analog:** the function itself (mod.rs:1711-1840). The extension is surgical — the per-level loop already exists; only the score step and the reject change.

**Remove this reject** (mod.rs:1740-1748):
```rust
if depth > 1 {
    return Err(CbError::OutOfRange(format!(
        "grow_oblivious_tree supports depth <= 1 until the per-partition \
         (fullPass = false) histogram fill lands ..."
    )));
}
```

**The per-level loop to thread through** (mod.rs:1785-1822) — the score step at :1792 is the ONLY line that must change; `partition_split` (:1810), `partition_update`, `leaf_of`, read-back stay unchanged except the +der2 channel:
```rust
for level in 0..depth {
    // (1) SWAP THIS whole-dataset score for a partition-aware (fullPass = L>0) fill
    //     keyed by the resident leaf_of, then the subtraction trick, then per-active-leaf score:
    let (best, _scores) = launch_find_optimal_split_pointwise_into(
        client, der1, weight, cindex, indices, n_bins, n_features, scaled_l2, score_fn,
    )?;
    let split = best.ok_or_else(|| CbError::Degenerate(/* ... */))?;
    splits.push((split.feature_id, split.bin_id));
    // (3) UNCHANGED — forward-bit doc routing leaf_of |= (bit << level) (Pitfall 6):
    leaf_of_h = launch_partition_split_into(
        client, der1_h.clone(), cindex_h.clone(), indices_h.clone(), leaf_of_h,
        n, cindex_stride, split.feature_id, split.bin_id, level as u32,
    )?;
}
```

**D-05 boundary contract (verified):** only the O(1) `BestSplit` descriptor per level + the final `2^depth` part-stats (`read_part_stats_f64`, mod.rs:1839) cross host↔device. Never read the full histogram/partition buffer to host (the FORBIDDEN hybrid). All host-side length/overflow guards (`checked_shl` for `2^depth` at :1753, `checked_mul` for `n_features*n` at :1759) are already present — extend them for the +der2 channel and `2^level` slot sizing.

---

### `crates/cb-backend/src/gpu_runtime/mod.rs` — `launch_partition_update_into` +Σder2 channel (GPUT-07, service)

**Analog:** `launch_partition_update_into` (mod.rs:1534-1592) and `partition_update_kernel` (kernels.rs:3531-3561).

**Kernel body to extend** (kernels.rs:3544-3560) — add a 3rd channel `Σ(der2·weight)`; the `part * 2` stride becomes `part * 3`:
```rust
while i < n {
    let obj = indices[i] as usize;
    let part = leaf_of[obj] as usize;
    let d = der1[obj];
    let w = weight[obj];
    if part * 2usize + 1usize < part_stats.len() {   // ← widen guard to part*3 + 2
        part_stats[part * 2usize].fetch_add(d);
        part_stats[part * 2usize + 1usize].fetch_add(w);
        // ← ADD: part_stats[part*3 + 2].fetch_add(der2_weighted[obj])
    }
    i += stride;
}
```

**Launch-side change** (mod.rs:1547): `let part_stats_len = n_parts.checked_mul(2)` → `checked_mul(3)`; add a `der2` (or `weighted_der2`) input handle. **Weighting LANDMINE (A3 / Pitfall 2):** der1/der2 handles are UNWEIGHTED (Phase 7.2 contract); the weight is folded HERE the same way der1's channel is. `reduce_leaf_der2` (the oracle) consumes `der2·weight` — match that folding exactly, or the `−Σder2 + scaledL2` denominator mis-scales.

---

### Newton leaf value + refinement (GPUT-07, D-01) — `der_seams.rs` reuse + `apply_leaf_delta`

**der2 handle analog** (der_seams.rs:287-367): reuse `DerUnaryKernel::LoglossHessian` (der2 = −p(1−p), p = sigmoid(approx), target-independent) via `launch_der_unary_handle` (der_seams.rs:304). For RMSE der2 = const −1 via `const_der_handle` (no kernel). Both are no-readback `*_handle` seams — **do not read them to host** (Handle bound to its allocating client).

**Leaf-value formula to transcribe inline** (leaf.rs:145-154 — copy, do NOT add a cb-train dep):
```rust
pub fn newton_leaf_delta(sum_der: f64, sum_der2: f64, scaled_l2: f64) -> f64 {
    let denom = -sum_der2 + scaled_l2;
    if denom == 0.0 { 0.0 } else { sum_der / denom }   // 0/0 empty-leaf guard only
}
```
RMSE collapses: der2 = −1 ⇒ −Σder2 = Σweight ⇒ `newton_leaf_delta` reduces to `calc_average` (leaf.rs:83) — RMSE needs no new leaf math; **Logloss is the genuinely-new path.**

**Device refinement update analog** (kernels.rs:587-598) — `apply_leaf_delta_kernel` keeps the approx update device-resident (D-01, no per-iteration readback):
```rust
if ABSOLUTE_POS < approx.len() {
    let leaf = leaf_of[ABSOLUTE_POS] as usize;
    approx[ABSOLUTE_POS] += lr[0] * leaf_values[leaf];
}
```
**Pin `leaf_estimation_iterations = 1` in the fixture (D-02 / A1 / Pitfall 1):** the CPU oracle does a SINGLE closed-form Newton step — NO iterative walker, NO backtracking (`cb-train/src/boosting.rs`: "leaf_estimation_iterations = 1 for this slice; auto-forced"). A multi-step loop would diverge from the parity target; if iterations>1 is ever wanted, build the iterative walker in `cb-compute` FIRST.

---

### `crates/cb-backend/src/kernels/reduce.rs` — deterministic accumulator (GPUT-06, step 0, LOCKED)

**Analog / consume as-is:** `block_reduce_fixedpoint_kernel`, `run_fixedpoint_reduce` (reduce.rs:731-760), `REDUCE_FIXEDPOINT_SCALE_F64` (k=30), and the `AtomicFinalizePath` capability enum (reduce.rs:668-731). This is the LOCKED SPIKE-REDUCTION §5b winner — do not re-open. Fixed-point finalize decode pattern (reduce.rs:760): `let sum = (bits as i64) as f64 / REDUCE_FIXEDPOINT_SCALE_F64;`. Backends lacking `Atomic<u64>` add report the downgrade (reduce.rs:673-677 capability probe), never silently switch.

---

### `crates/cb-backend/src/kernels/grow_loop.rs` — depth-6 + Newton self-oracle (GPUT-05/07/14, test)

**Analog:** the existing depth-1 self-oracles. Replicate their structure at depth-6:
- `matches_cpu_greedy_search` (grow_loop.rs:475) — grows the device tree, asserts (B) per-object `leaf_of == cpu_leaf_index`, then (C) reports leaf-value divergence vs the CPU reference. Extend to depth-6 and swap `calc_average` for `newton_leaf_delta` on the Logloss arm.
- `cpu_boosting_pass` (grow_loop.rs:733) — the serial multi-tree reference (forward-bit `leaf_of`, per-leaf `calc_average` over ordered `sum_f64`, `approx += lr * leaf_values`). This is the pattern for the across-run ε=1e-4 check.
- `leaf_of_matches_cpu_leaf_index` (grow_loop.rs:288) — extend the split sequence to depth-6.
- **Replace** `depth_gt_one_is_tracked_forward_dependency` (grow_loop.rs:653) — this test asserts depth>1 surfaces the typed reject; it must be inverted/removed once the reject is gone.
- Divergence helper `max_divergence` (grow_loop.rs:60) — max abs/rel over equal-length buffers at the ε bar.

---

### `crates/cb-compute/src/{histogram.rs,leaf.rs}` — CPU oracles (already present, reuse)

Both oracle functions already exist and are byte-unchanged (D-04). Do NOT modify — the device path matches THEM:
- `reduce_leaf_der2(leaf_of, weighted_der2, n_leaves)` (histogram.rs:100-121) — `weighted_der2[i] = der2·weight`; Σ per leaf via `cb_core::sum_f64` in canonical object order. The oracle for the new device Σder2 partition channel.
- `reduce_leaf_stats` (histogram.rs:49-87) — Σder1 / Σweight per leaf (existing depth-1 oracle).
- `newton_leaf_delta` (leaf.rs:145) / `calc_average` (leaf.rs:83) / `scale_l2_reg` (leaf.rs:99) — the leaf math. Transcribe (don't import) into `cb-backend` if a device-side host compute is needed.

---

### `bench/generator.py` — depth-6 RMSE+Logloss fixture + speed workload (D-03, config)

**Analog:** `serial_depth1_tree` (generator.py:230 — Cosine-score best split + `calc_average` leaves, RMSE `der1 = y − approx`, Logloss `der1 = y − sigmoid`) and `write_fixtures` (generator.py:313, emits `expected_depth1_tree.json` + sha-manifested `.npy`). Extend to a `serial_depth6_tree` (recursive per-level partition, Newton leaf on the Logloss arm) producing BOTH the ≤1e-4 correctness fixture AND the large-n speed workload from one seed. **Pin `leaf_estimation_iterations=1` and the score function in the emitted config (A1/A2).** Real named datasets (Higgs/Epsilon) stay deferred to Phase 14 (BENCH-03) — synthetic only.

---

### `bench/cuda_oracle.ipynb` + `bench/RESULTS.md` — Kaggle final-ε + per-tree diagnostic + speed (D-05, BENCH-02)

**Analog:** the notebook's existing depth-1 oracle cells + the `RESULTS.md` dated sign-off log (correctness is a BLOCKING gate before any speed number; no fabricated numbers). Extend with: (a) final-prediction ε=1e-4 gate over the full depth-6 run (RMSE+Logloss); (b) per-tree split-agreement + run-to-run spread diagnostic (D-05 — pinpoints the first divergent tree, evidences SC-3 "no split flips compounding"); (c) depth-6 device-vs-host-CPU-vs-official-CatBoost-GPU speed cells (warm-run/JIT-excluded, train-only). Append a dated Phase-11 block to `RESULTS.md`. **Human-gated external run — the subagent cannot execute it; the orchestrator/human discharges the gate.**

## Shared Patterns

### Deterministic reduction (GPUT-06, LOCKED)
**Source:** `crates/cb-backend/src/kernels/reduce.rs` (`block_reduce_fixedpoint_kernel`, `REDUCE_FIXEDPOINT_SCALE_F64` k=30, `AtomicFinalizePath`).
**Apply to:** the partition-aware histogram accumulator AND the partition-update Σ reduces. Consume as step 0; never use f64 `Atomic::fetch_add` for the many-cubes-contend accumulator.

### Host-side pre-launch validation (V5 input validation)
**Source:** `launch_pointwise_hist2_into` (mod.rs:557-575) — `checked_mul` stride guard, `LengthMismatch`/`OutOfRange` typed errors BEFORE launch; `grow_oblivious_tree_into` (mod.rs:1753-1768) — `checked_shl` for `2^depth`.
**Apply to:** every new launch — validate `leaf_of[obj] < 2^level`, `cindex.len()` stride, `2^depth * 3` buffer sizing with `checked_*` → typed `CbError::OutOfRange`; `launch_unchecked` only after host validation.

### No-readback device seam (D-01 / D-05)
**Source:** der_seams.rs `*_handle` entries (der_seams.rs:50-159, 304-367) — Handle returned, never read to host; bound to its allocating client.
**Apply to:** the der2 handle in the Newton path and every resident histogram/partition buffer. One `&client` threads the whole tree.

### Self-oracle divergence report at the ε bar
**Source:** `max_divergence` + the `(A) STRUCTURE assert / (C) VALUE report` split in grow_loop.rs (:60, :475, :510-538).
**Apply to:** the depth-6/Newton self-oracles — hard-assert structure (`leaf_of == cpu_leaf_index`, split sequence), report value divergence within ε=1e-4.

## No Analog Found

None. Every new behavior extends a Phase 7/10-validated in-repo pattern. The only genuinely-new `#[cube]` kernel is the **subtraction-trick** (`parent − smaller`, weight-channel `max(0)` clamp) — it has no direct analog but is a trivial elementwise cell-loop; mirror the `apply_leaf_delta_kernel` position-guarded shape (kernels.rs:587-598) and transcribe upstream `SubstractHistogramsImpl` (§6.4). **MEDIUM-confidence items to pin in the depth-6 fixture, not from a codebase analog:** exact `2^level` slot addressing and channel-0 (Σweight vs Σder2) semantics (RESEARCH A2) — cross-check device split scores against the CPU reference for the first few trees.

## Metadata

**Analog search scope:** `crates/cb-backend/src/kernels.rs`, `crates/cb-backend/src/kernels/{grow_loop,reduce,pointwise_hist,partitions,apply_leaf_delta}.rs`, `crates/cb-backend/src/gpu_runtime/{mod,der_seams}.rs`, `crates/cb-compute/src/{histogram,leaf}.rs`, `crates/cb-train/src/boosting.rs`, `bench/{generator.py,RESULTS.md,cuda_oracle.ipynb,fixtures/}`
**Files scanned:** ~15 source files + bench harness
**Pattern extraction date:** 2026-07-03
