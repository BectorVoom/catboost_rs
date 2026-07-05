# Phase 15: Debt Discharge & CUDA Oracle Re-establishment - Pattern Map

**Mapped:** 2026-07-05
**Files analyzed:** 11 (4 production edits, 3 test files, 1 new Python harness + JSON, 3 in-place doc rewrites, 3 bookkeeping edits)
**Analogs found:** 11 / 11 (every file has an in-repo analog at exact HEAD line numbers)

> This is a debt-discharge phase: almost every new/modified file copies an EXISTING sibling in the same file or directory. There is very little green-field. The planner should treat each analog below as "copy this shape, change these lines."

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/cb-backend/src/gpu_runtime/ranking.rs` (RV-13-01 `descending_order_per_query`, RV-13-02 `query_softmax_ders_host`) | gpu-runtime host driver | transform (host→device→host der) | self: sibling fns in same file; `compute_group_max_host` for the weight-aware max | exact (edit-in-place) |
| `crates/cb-backend/src/kernels/query_helper.rs` (RV-13-03 `compute_group_means_host`) | kernel host driver | transform (group reduction) | `remove_group_means_host` (same file, already guards `n==0`) | exact |
| `crates/cb-backend/src/gpu_runtime/pairwise.rs` (RV-13-04 `select_best_split_over_scores`) | gpu-runtime host driver | transform (argmax over host f64 scores) | self: the existing exact-`==` tie-break at `:1882-1898` | exact (edit-in-place) |
| `crates/cb-backend/src/gpu_runtime/ranking_stoch_test.rs` (RV-13-01/02 oracles, +2 tests) | test (self-oracle) | request-response (direct kernel invocation + ε assert) | same file's `yetirank_der_matches_frozen_cpu` | exact |
| `crates/cb-backend/src/kernels/query_helper_test.rs` (RV-13-03 oracle, +1 test) | test (self-oracle) | request-response | same file's `zero_weight_query_mean_is_zero` (`:169`) | exact (extends) |
| `crates/cb-backend/src/kernels/cholesky_solve_test.rs` (RV-13-04 oracle, +1 test) | test (self-oracle) | request-response | same file header + `device_backend_active` pattern | exact |
| `bench/phase15_cuda_oracle/oracle.py` (new) | harness / integration runner | batch (Part A gate → Part B/C timing) | `bench/phase14_cuda_signoff/oracle.py` | exact (clone + delta) |
| `bench/phase15_cuda_oracle/result.json` (new, from run) | data artifact | file-I/O | `bench/phase14_cuda_signoff/bench03-result.json` | exact (schema) |
| `bench/BENCH-03-SIGNOFF.md`, `bench/RESULTS.md` | docs / evidence | file-I/O (in-place rewrite) | themselves (TBD tables at `RESULTS.md:56-66,71`) | exact |
| `.planning/phases/15-.../15-EVIDENCE.md` (new) | docs / evidence | file-I/O | no direct analog — structure per D-10 (one entry per RV-13-0x) | role-match |
| `.planning/REQUIREMENTS.md`, `MILESTONES.md`, `STATE.md` | bookkeeping | file-I/O (checkbox/status flip) | exact cells located (RESEARCH Bookkeeping Surface) | exact |

## Pattern Assignments

### `crates/cb-backend/src/gpu_runtime/ranking.rs` — RV-13-01 tie order (gpu-runtime, transform)

**Analog:** self (the function is already written; the fix is confirmatory-or-defensive per Pitfall 1).

**Current function** (`ranking.rs:738-769`) — the stable complemented-key radix that the tie oracle must exercise:
```rust
#[cfg(not(feature = "wgpu"))]
fn descending_order_per_query(perturbed: &[f64], q_offsets: &[u32]) -> CbResult<Vec<u32>> {
    let n = perturbed.len();
    if n == 0 { return Ok(Vec::new()); }
    let mut head = vec![0u32; n];
    for w in q_offsets.windows(2) {
        let b = *w.first().unwrap_or(&0) as usize;
        if let Some(slot) = head.get_mut(b) { *slot = 1; }
    }
    // Complement the radix key so ONE stable ASCENDING pass yields the DESCENDING order,
    // ties preserved in ORIGINAL index order (WR-01). Non-negative f64 bits are monotone.
    let ord: Vec<u64> = perturbed.iter().map(|&v| !v.to_bits()).collect();
    let lo: Vec<u32> = ord.iter().map(|&b| b as u32).collect();
    let hi: Vec<u32> = ord.iter().map(|&b| (b >> 32) as u32).collect();
    let idx0: Vec<u32> = (0..n as u32).collect();
    let (_sk, order_lo) = segmented_radix_sort(&head, &lo, &idx0)?;
    let hi_re: Vec<u32> = order_lo.iter()
        .map(|&i| hi.get(i as usize).copied().unwrap_or(0)).collect();
    let (_sk2, order) = segmented_radix_sort(&head, &hi_re, &order_lo)?;
    Ok(order)
}
```
**Fix guidance (Pitfall 1 / A1):** the oracle is written FIRST. If it passes as-is, the "fix" is a doc-anchored comment + the oracle (a valid HARD-03 discharge — record honestly in `15-EVIDENCE.md`). Only if the tie oracle reveals a flip does the planner add a tertiary index radix key (`idx0` already carried through both passes). Do NOT churn working code speculatively. `segmented_radix_sort` (reused from `exact_quantile.rs:155`) — do not hand-roll a second sort.

---

### `crates/cb-backend/src/gpu_runtime/ranking.rs` — RV-13-02 weight>0 max-seed (gpu-runtime, transform)

**Analog:** `compute_group_max_host` (`query_helper.rs:436-463`, the weight-BLIND max currently seeded); the CPU parity target is `cb-compute/src/ranking_der.rs:257-266`.

**The bug** (`ranking.rs:472-475`) — max seeded over ALL objects, not weight>0:
```rust
let weight_col = weight_column(weights, n)?;
// Uniform-weight covered regime: max over all objects == the CPU max-over-(weight>0) seed.
let group_max = compute_group_max_host(approx, q_offsets)?;   // <-- weight-BLIND (the fix target)
```

**CPU reference to mirror** (`crates/cb-compute/src/ranking_der.rs:257-266`, transcribe FROZEN, never a live dep):
```rust
let mut max_approx = f64::MIN;
for i in 0..group.size() {
    let w = weight_at(i);
    let a = approx_slice.get(i).copied().unwrap_or(0.0);
    if w > 0.0 {                    // <-- weight>0-ONLY seed
        if a > max_approx { max_approx = a; }
    }
}
```
**Fix (Open Q2 → prefer host-side, no `#[cube]` edit):** compute a per-query max over `w>0` objects host-side (a weight-aware loop replacing the `compute_group_max_host(approx,…)` call). Empty-weight-group falls back to `f64::MIN` but the downstream `sum_weighted_targets > 0` guard short-circuits before `exp` (mirror CPU exactly). Do NOT merely edit the comment (Pitfall 2).

---

### `crates/cb-backend/src/kernels/query_helper.rs` — RV-13-03 `n==0` guard (kernel host driver, transform)

**Analog:** `remove_group_means_host` (same file, `:467+`) already guards `n==0` per research — copy that guard placement; and the existing `n_groups==0` guard in `compute_group_means_host` itself.

**Current guard block** (`query_helper.rs:379-383`) — guards `n_groups==0` but NOT `n==0`:
```rust
let n = values.len();
let n_groups = q_offsets.len().saturating_sub(1);
if n_groups == 0 { return Ok(Vec::new()); }     // existing guard
// PROPOSED (insert here, BEFORE the `#[cfg(not(wgpu))] { … client.create(…) }` block so no
// zero-length device buffer is bound — project HIP residency lesson):
if n == 0 { return Ok(vec![0.0; n_groups]); }    // all-empty groups → zero means (CPU queryAvrg 0)
```
**Fix (Pitfall 3 / A3):** the empty-group mean is `0.0` PER group → `vec![0.0; n_groups]`, NOT `Vec::new()` (callers expect `n_groups` entries). Placed before `client.create` (`:417-418`). Consider the same guard on sibling `compute_group_max_host` (`:436-463`) only if the oracle reaches it — stay inside the named-site boundary.

---

### `crates/cb-backend/src/gpu_runtime/pairwise.rs` — RV-13-04 tie-break (gpu-runtime, argmax transform)

**Analog:** self — the existing exact-`==` argmax inside `select_best_split_over_scores` (`:1807`).

**Current argmax over host-resident f64 scores** (`pairwise.rs:1882-1898`):
```rust
let mut best_score = f64::NEG_INFINITY;
let mut best_c = u32::MAX;
for &cand in best_idxs.iter() {
    if (cand as usize) >= n_candidates { continue; }
    let score = match scores.get(cand as usize) { Some(&s) => s, None => continue };
    // CURRENT: exact f64 tie-break — rarely fires across device-Cholesky vs wgpu host-scorer.
    let take = score > best_score || (score == best_score && cand < best_c);
    if take { best_score = score; best_c = cand; }
}
```
**Backend split that produces divergent scores** (`pairwise.rs:1830-1861`): `#[cfg(feature="wgpu")]` scorer runs the reduce in f32 / uses `cb_compute::calculate_pairwise_score` (host scorer, `:1768`); `#[cfg(not(wgpu))]` runs f64 device Cholesky. The two accumulation orders differ ~1e-13 on near-equal borders.

**Fix (Pitfall 4 / A2, recommended default):** near-equal-tolerant, lowest-index deterministic tie-break — treat `|a-b| <= tol * max(|a|,|b|,1.0)` as tied, break by lowest candidate index (`select_best_candidate` first-wins parity contract). `tol` sized just above the observed device-vs-host delta (~1e-9 relative; the oracle settles it). Do NOT try to force the two solves bit-identical (larger scope).

---

### `crates/cb-backend/src/gpu_runtime/ranking_stoch_test.rs` — RV-13-01/02 oracles (test, self-oracle)

**Analog:** the same file's existing `yetirank_der_matches_frozen_cpu` (`:127-143`). Copy verbatim: `#![cfg(not(feature = "wgpu"))]` gate (`:29`), `const TOL: f64 = 1e-4` (`:37`), `device_backend_active()` (`:42-44`), frozen-literal CPU reference (`:56-78`), and the record-only skip.

**The exact skip / assert idiom to copy** (`ranking_stoch_test.rs:137-142`):
```rust
if device_backend_active() {
    assert_der_close(&der1, &frozen_der1(), "yetirank der1");
    assert_der_close(&der2, &frozen_der2(), "yetirank der2");
} else {
    println!("yetirank: cpu backend — numeric ε assert skipped (WR-01 record-only)");
}
```
**New tests:**
- RV-13-01 `tie_order_matches_cpu_stable_descending` — feed `descending_order_per_query` tied perturbed values (`exp(approx)`+f32-Gumbel ties); assert the returned order == the frozen CPU stable-descending order. FROZEN reference generated offline (module-doc pattern at `:7-14`).
- RV-13-02 `softmax_weight_max_seed` — call `query_softmax_ders_host` on a weighted query whose global-max doc has weight ≤ 0; assert der1/der2 ≤ 1e-4 vs frozen CPU der. Pitfall 2: the fixture MUST place `w≤0` on the max-approx doc (else tautological).

**Non-tautology rule (Pattern 1):** frozen literals generated ONCE offline from the INDEPENDENT `cb-train`/`cb-compute` path — NEVER a live `cb-train` dep (feature-unification landmine).

---

### `crates/cb-backend/src/kernels/query_helper_test.rs` — RV-13-03 oracle (test, self-oracle)

**Analog:** the same file's `zero_weight_query_mean_is_zero` (`:167-185`) — extend it or add a sibling.

**Existing test to mirror** (`query_helper_test.rs:169-185`):
```rust
#[test]
fn zero_weight_query_mean_is_zero() {
    let values = vec![1.0, 2.0, 3.0, 4.0];
    let weights = vec![0.0, 0.0, 1.0, 1.0];
    let q_offsets = vec![0u32, 2, 4];
    let ref_means = cpu_group_means(&values, &weights, &q_offsets);
    assert_eq!(ref_means[0], 0.0, "CPU zero-weight query mean must be 0");
    let dev = compute_group_means_host(&values, &weights, &q_offsets)
        .expect("device group-means must not error");
    assert!(dev.iter().all(|v| v.is_finite()), "means must be finite");
    let div = max_abs_divergence(&dev, &ref_means);
    if device_backend_active() {
        assert!(div <= TOL, "device zero-weight mean diverged: {div:e} > {TOL:e}");
    }
}
```
**New test** `empty_group_means_no_fault` — `q_offsets=[0,0]` (`n==0`, `n_groups==1`); assert `compute_group_means_host` returns `Ok(vec![0.0])` (RIGHT length + value, not just "no panic" — Pitfall 3) and launches no zero-length device buffer. This is ALSO validated on rocm in-env (D-03: RV-13-03's fault-guard is the one hazard smoke-tested in-env).

---

### `crates/cb-backend/src/kernels/cholesky_solve_test.rs` — RV-13-04 oracle (test, self-oracle)

**Analog:** this file's header + `device_backend_active` (`:42-44`), `const TOL: f64 = 1e-4` (`:33`), `#![cfg(not(feature = "wgpu"))]` (`:27`), and its reuse of `cb_compute::pairwise_cholesky_solve` (`:30`, the shared SPD primitive — already a dep, NOT `cb-train`).

**New test** `pairwise_near_equal_border_tiebreak` — construct pairwise inputs with two borders whose TRUE scores fall inside the tolerance band; assert the device-Cholesky path (`not(wgpu)`) and the frozen wgpu host-scorer path (`calculate_pairwise_score`) select the SAME `BestSplit`. Pitfall 4: the fixture must use near-equal (not well-separated) borders to exercise the flip. Alternatively add to `pairwise_deriv_test.rs`; either sibling matches the pattern.

---

### `bench/phase15_cuda_oracle/oracle.py` — single-session runner (harness, batch)

**Analog:** `bench/phase14_cuda_signoff/oracle.py` (clone + delta). Reuse verbatim:
- **`gen(n, nf, nbins)`** (`phase14 oracle.py:58-84`) — the frozen numpy repro of `bench_grow_speed_test.rs::gen()` (integer-binned 0..31 f32, seed via `i*2654435761 + f*40503 mod 2^64`). Do NOT hand-roll a new generator (Don't-Hand-Roll).
- **Repo staging into `/tmp`** (`:121-144`) + rust toolchain + `CARGO_TARGET_DIR=/tmp/target` (`:146-154`) — keeps Kaggle output tiny (~1.8MB `git archive`, not 2.9G crates/).
- **Part A per-family gate → blocking `sys.exit(2)`** (`:156-205`): the `FAMILIES` table (`:160-171`) drives `cargo test --release --no-default-features --features cuda -p cb-backend -- <filters>`; `corr_pass = all(exit==0 and ran_any_tests)` (`:195`) → `sys.exit(2)` BEFORE any timing (D-05). The DIV_RE/SUMMARY_RE parsing (`:173-192`) harvests ε divergences into JSON.
- **Part C CatBoost-GPU informational arm** (`:207-296`) — Region N/A (`:231-238`), border_count=32, quantization caveat, warm untimed fit (`:251-260`) → keep for BENCH-03's informational `catboost_gpu_s` column (D-08).

**Phase-15 delta (NEW, per D-04/D-07):**
- ADD the 4 RV-13 oracles to Part A's filter set (`ranking_stoch`, `query_helper`, `cholesky_solve` already covered by the ranking/pairwise families — the two NEW test names ride the SAME `cargo test` invocations).
- ADD a Part B (BENCH-02 timing) that runs the **depth-1** and **depth-6** device+CPU grow rows in the SAME kernel session (warm/JIT-excluded/queue-drained/median-of-N, 20 iters/20 feat/32 bins; depth-1 on large-n `SPEED_CONFIG` ~1e6×50 per Pitfall 5). This is the depth grow-speed arm the Phase-14 clone does NOT have.
- Emit ONE `result.json` + ONE verdict (D-04) — do NOT stitch multiple sessions.

**Anti-pattern (from `aggregate.py`):** do NOT extend `aggregate.py`'s two-file `load_rows` — supersede it. Keep only its `GE20X_GATE = 20.0` verdict shape (`aggregate.py:45`, `speedup >= 20.0` roll-up at `:163-171`) for the BENCH-03 recompute reading the single json.

---

### `bench/BENCH-03-SIGNOFF.md` + `bench/RESULTS.md` — in-place evidence rewrite (docs, file-I/O)

**Analog:** themselves. Fill the TBD tables with real single-session numbers (no fabrication, D-09):
- `RESULTS.md:56-66` — depth-1 correctness oracle table (primitive scan/radix/reduce, cindex GPUT-15, depth-1 RMSE/Logloss `<=1e-5`), `GATE:` at `:66`.
- `RESULTS.md:71` — depth-1 large-n speed row + crossover note `:72` (Pitfall 5 / D-10-09 escalation at `:19-36` — record the crossover, do NOT force a device≥CPU pass).
- `RESULTS.md:129-132` — depth-6 Gate A/B rows.
- `BENCH-03-SIGNOFF.md` — rewrite the "Standing debt" section to reflect discharge; Region stays `N/A`, `catboost_gpu_s` stays informational (D-08).

---

### `.planning/phases/15-.../15-EVIDENCE.md` — per-hazard evidence (docs, new)

**Analog:** none direct — structure per D-10: one entry per RV-13-0x with { what diverged, the oracle, the fix, the passing result }. Honest recording especially for RV-13-01 (A1: "verified stable + oracle added" is a valid discharge if the oracle passes unchanged).

---

## Shared Patterns

### Pattern A — Frozen-CPU-reference self-oracle (non-tautological)
**Source:** `crates/cb-backend/src/gpu_runtime/ranking_stoch_test.rs:29-44` and module doc `:7-27`.
**Apply to:** ALL FOUR RV-13 oracles (RV-13-01/02 in `ranking_stoch_test.rs`, RV-13-03 in `query_helper_test.rs`, RV-13-04 in `cholesky_solve_test.rs`).
```rust
#![cfg(not(feature = "wgpu"))]
const TOL: f64 = 1e-4;                                   // D-07 GPU bar
fn device_backend_active() -> bool {
    cfg!(any(feature = "rocm", feature = "cuda"))         // ε assert skips on cpu (WR-01)
}
```
CPU reference is FROZEN literals generated offline from the INDEPENDENT `cb-train`/`cb-compute` path — NEVER a live `cb-train` dep (feature-unification landmine, CLAUDE.md). Numeric assert hard-fires only on real device; record-only on `cpu` (anti-false-pass).

### Pattern B — Host-side residency guard before `client.create`
**Source:** `crates/cb-backend/src/kernels/query_helper.rs:381-383` (existing `n_groups==0` guard) + `remove_group_means_host` (`:467+`, existing `n==0` guard).
**Apply to:** RV-13-03 fix — never bind/read a zero-length device handle (project HIP residency lesson). Return the correctly-sized empty result BEFORE `selected_client()` / `client.create`.

### Pattern C — Correctness-blocks-speed single-session gate
**Source:** `bench/phase14_cuda_signoff/oracle.py:194-205`.
**Apply to:** `bench/phase15_cuda_oracle/oracle.py` Part A → `sys.exit(2)` before Part B timing. `corr_pass = all(f["exit"]==0 and f["ran_any_tests"] …)`; no speed number is emitted on a failed pre-gate (D-05).

### Pattern D — `thiserror` typed errors, no `unwrap()` in production
**Source:** `query_helper.rs:397-415` (`CbError::Degenerate` / `CbError::OutOfRange`).
**Apply to:** all four production fixes. `.unwrap()`/indexing lives ONLY in `*_test.rs` (CLAUDE.md source/test separation is MANDATORY).

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `.planning/phases/15-.../15-EVIDENCE.md` | docs | file-I/O | No prior per-hazard evidence artifact in-repo; structure is dictated by D-10, not an existing template. (`bench/BENCH-03-SIGNOFF.md` is the closest tone/format reference.) |

## Metadata

**Analog search scope:** `crates/cb-backend/src/gpu_runtime/`, `crates/cb-backend/src/kernels/`, `crates/cb-compute/src/`, `bench/`, `.planning/`
**Files scanned:** 9 read at exact HEAD line numbers (4 production, 3 test, 1 python, plus bench markdown grep)
**Line-number caveat (Pitfall 6):** CONTEXT.md line numbers are from archived commit `0f457d9`; all excerpts above are re-anchored to HEAD by FUNCTION NAME, verified this session.
**Pattern extraction date:** 2026-07-05
