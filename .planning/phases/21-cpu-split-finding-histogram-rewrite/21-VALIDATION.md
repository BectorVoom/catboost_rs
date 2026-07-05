---
phase: 21
slug: cpu-split-finding-histogram-rewrite
status: planned
nyquist_compliant: true
wave_0_complete: true
created: 2026-07-05
---

# Phase 21 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (+ `approx` for ≤1e-5 float asserts); source/test files strictly separated (`*_test.rs`) |
| **Config file** | none — workspace `cargo test` |
| **Quick run command** | `cargo test -p cb-compute` (histogram unit + score parity) |
| **Full suite command** | `cargo test -p cb-compute -p cb-train` (all CPU oracle fixtures across losses/policies/CTR) |
| **Perf check (PERF-01/03)** | `CB_PERF=1 cargo test --release -p cb-train --test perf_baseline_test -- --nocapture` (n_bins-flat + before/after speedup) |
| **Estimated runtime** | ~60–180s full suite; perf grid minutes (release) |

---

## Sampling Rate

- **After every task commit:** Run the quick command for the crate touched.
- **After every plan wave:** Run the full suite — every shipped ≤1e-5 CPU oracle fixture MUST stay green/bit-exact (PERF-02 gate).
- **Before `/gsd-verify-work`:** Full suite green + perf check shows n_bins-flat timing (PERF-01) and a recorded before/after speedup (PERF-03).
- **Max feedback latency:** ~180 seconds (full suite).

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 21-01-01 | 01 | 1 | PERF-01, PERF-02 | T-21-01/02/03 | `.get()` defensive bin/leaf access; sum_f64 order | unit | `cargo test -p cb-compute histogram` | ✅ (histogram_test.rs exists; new cases added) | ⬜ pending |
| 21-01-02 | 01 | 1 | PERF-02 | T-21-03 | ordered sum_f64; no circular cb-train dep | unit | `cargo test -p cb-compute` | ✅ | ⬜ pending |
| 21-02-01 | 02 | 2 | PERF-01, PERF-02 | T-21-05/06 | subtraction-trick order; scratch bounded | oracle | `cargo test -p cb-train --test loss_oracle_test --test overfit_oracle_test --test regularization_oracle_test` | ✅ | ⬜ pending |
| 21-02-02 | 02 | 2 | PERF-01, PERF-02 | T-21-04 | perturbed RNG draw order preserved | oracle + perf | `cargo test -p cb-train` ; `CB_PERF=1 cargo test --release -p cb-train --test perf_baseline_test -- --nocapture` | ✅ | ⬜ pending |
| 21-03-01 | 03 | 3 | PERF-02 | T-21-07/08/09 | per-leaf hist order; fresh-rebuild fallback | oracle | `cargo test -p cb-train --test non_symmetric_grower_oracle_test --test region_e2e_test` | ✅ | ⬜ pending |
| 21-03-02 | 03 | 3 | PERF-02 | T-21-07 | full-suite parity gate | oracle | `cargo test -p cb-train` ; `cargo test -p cb-compute` | ✅ | ⬜ pending |
| 21-04-01 | 04 | 4 | PERF-02 | T-21-10/11/12 | CTR `bin > border` off-by-one guard | oracle | `cargo test -p cb-train --test plain_ctr_oracle_test --test tensor_ctr_oracle_test --test ctr_split_scoring_test` | ✅ | ⬜ pending |
| 21-04-02 | 04 | 4 | PERF-02 | T-21-11 | full-suite parity gate | oracle | `cargo test -p cb-train` ; `cargo test -p cb-compute` | ✅ | ⬜ pending |
| 21-05-01 | 05 | 5 | PERF-03 | T-21-13/14 | feature-independent ordered collect | build/unit | `cargo test -p cb-train tree::` | ✅ | ⬜ pending |
| 21-05-02 | 05 | 5 | PERF-03, PERF-02 | T-21-13 | byte-identical model across runs | unit + oracle | `cargo test -p cb-train --test rayon_determinism_test` ; `cargo test -p cb-train` ; `cargo test -p cb-compute` | ❌ W0 (rayon_determinism_test.rs created in 21-05) | ⬜ pending |
| 21-05-03 | 05 | 5 | PERF-03 | T-21-SC | rayon verdict OK; commit Cargo.lock | bench (manual-recorded) | `CB_PERF=1 cargo test --release -p cb-train --test bench_grow_speed_test -- --nocapture` | ✅ | ⬜ pending |

*(Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky. Task IDs are `{plan}-{task}`.)*

---

## Wave 0 Requirements (covered by plans — no separate wave 0)

- [x] Histogram equivalence unit tests (per-(feature,bin) `TBucketStats` accumulator + subtraction trick + prefix-scan-to-`LeafStats`, bit-exact vs `reduce_leaf_stats` and vs `score_candidate` raw score) → **Plan 21-01 Tasks 1-2**, in `crates/cb-compute/src/histogram_test.rs` (self-contained reference, no cb-train dep — W2).
- [x] Bit-exact parity invariant isolated as a unit test (the PERF-02 invariant) → **Plan 21-01 Task 2**.
- [x] `rayon` added to `[workspace.dependencies]` + `crates/cb-train/Cargo.toml` (NOT cb-compute) + byte-identical determinism test → **Plan 21-05 Tasks 1-2** (`crates/cb-train/tests/rayon_determinism_test.rs`).

*Existing CPU oracle fixtures (Phases 3–6) cover the end-to-end ≤1e-5 parity gate — no new e2e infra needed; the rewrite must not perturb them.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Before/after end-to-end speedup magnitude | PERF-03 | Wall-clock benchmark, machine-dependent absolute numbers | Run the perf check pre- and post-rewrite; record per-tree ms + speedup factor + n_bins-flatness in the plan SUMMARY. |

*All correctness behaviors (PERF-01 histogram equivalence, PERF-02 bit-exact fixtures) have automated verification.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (equivalence + determinism tests + rayon dep planned in 21-01/21-05)
- [x] No watch-mode flags
- [x] Feedback latency < 180s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-07-05
