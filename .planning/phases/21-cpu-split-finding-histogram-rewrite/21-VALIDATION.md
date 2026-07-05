---
phase: 21
slug: cpu-split-finding-histogram-rewrite
status: draft
nyquist_compliant: false
wave_0_complete: false
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
| 21-01-01 | 01 | 1 | PERF-01 | — / — | N/A | unit | `cargo test -p cb-compute histogram` | ❌ W0 | ⬜ pending |
| 21-0X-XX | — | — | PERF-02 | — / — | N/A | oracle | `cargo test -p cb-train` (existing ≤1e-5 fixtures) | ✅ | ⬜ pending |

*(Planner fills the full map. Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky)*

---

## Wave 0 Requirements

- [ ] `crates/cb-compute/src/histogram_scatter_test.rs` (or analog) — unit tests for the new per-(feature,bin) `TBucketStats` accumulator + subtraction trick + prefix-scan-to-`LeafStats` equivalence.
- [ ] A parity harness assertion that the histogram-produced `LeafStats` equal the current `reduce_leaf_stats` output bit-exact on a fixed fixture (the PERF-02 invariant, isolated as a unit test).
- [ ] `rayon` added to the relevant crate `[dependencies]` (PERF-03 wave only).

*Existing CPU oracle fixtures (Phases 3–6) cover the end-to-end ≤1e-5 parity gate — no new e2e infra needed; the rewrite must not perturb them.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Before/after end-to-end speedup magnitude | PERF-03 | Wall-clock benchmark, machine-dependent absolute numbers | Run the perf check pre- and post-rewrite; record per-tree ms + speedup factor + n_bins-flatness in the plan SUMMARY. |

*All correctness behaviors (PERF-01 histogram equivalence, PERF-02 bit-exact fixtures) have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
