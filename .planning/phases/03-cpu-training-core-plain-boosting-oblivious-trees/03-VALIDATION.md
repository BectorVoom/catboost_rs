---
phase: 3
slug: cpu-training-core-plain-boosting-oblivious-trees
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-13
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution. Derived from
> `03-RESEARCH.md` § Validation Architecture. Per-slice oracle locks (≤1e-5 abs) are the
> sampling rate — each additive knob (D-10) gets its own oracle-locked slice.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` + `cb-oracle` per-stage comparator (`compare_stage`, ≤1e-5 abs) |
| **Config file** | none (cargo native); fixtures under `crates/cb-oracle/fixtures/` |
| **Quick run command** | `cargo test -p <crate-under-edit>` (e.g. `-p cb-train`, `-p cb-compute`, `-p cb-backend`) |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | ~30–90 seconds (CPU-only; no GPU tests this phase) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p <crate-under-edit>` (quick)
- **After every plan wave:** Run `cargo test --workspace` (full)
- **Before `/gsd-verify-work`:** Full workspace green AND every slice's `compare_stage` oracle passing at ≤1e-5
- **Max feedback latency:** ~90 seconds

---

## Per-Task Verification Map

| Task ID | Slice / Wave | Requirement | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|--------------|-------------|-----------------|-----------|-------------------|-------------|--------|
| boundary | CubeCL seam (W1) | TRAIN-01 | no panic on degenerate input; `CbError` not `unwrap` | unit | `cargo test -p cb-backend kernels::gradient` | ❌ W0 | ⬜ pending |
| loss | first slice (W1) | TRAIN-01 | — | unit | `cargo test -p cb-compute loss::` | ❌ W0 | ⬜ pending |
| tie-break | first slice (W1) | TRAIN-02 | deterministic (parity integrity) | unit | `cargo test -p cb-train tree::tie_break` | ❌ W0 | ⬜ pending |
| first-slice oracle | first slice (W1) | TRAIN-01/02/03 | ordered host reduction via `cb-core::sum_f64` | oracle | `cargo test -p cb-train slice_first_oracle` | ❌ W0 | ⬜ pending |
| leaf methods | leaf wave (W2) | TRAIN-03 | `CalcAverage` guards count>0 | oracle | `cargo test -p cb-train leaf_methods_oracle` | ❌ W0 | ⬜ pending |
| bootstrap | sampling wave (W3) | TRAIN-04 | exact `randSeed+blockIdx`/`Advance(10)` draw order | oracle | `cargo test -p cb-train bootstrap_oracle` | ❌ W0 | ⬜ pending |
| regularization | reg wave (W4) | TRAIN-05 | normal-draw order reproduced (Box-Muller) | oracle | `cargo test -p cb-train regularization_oracle` | ❌ W0 | ⬜ pending |
| overfit | overfit wave (W5) | TRAIN-06 | — | unit+oracle | `cargo test -p cb-train overfit::` | ❌ W0 | ⬜ pending |
| eval metrics | eval wave (W6) | TRAIN-07 | — | oracle | `cargo test -p cb-train eval_metrics_oracle` | ❌ W0 | ⬜ pending |
| auto-LR | auto-LR wave (W7) | TRAIN-08 | — | unit | `cargo test -p cb-train autolr::` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky. `❌ W0` = test asset must be created in Wave 0.*

---

## Wave 0 Requirements

- [ ] Extend `crates/cb-oracle/generator/gen_fixtures.py` to emit per-slice training oracles: `splits`/`leaf_values` from `model.json`, per-iteration `staged.npy`, and a **binclf_skeleton** (Logloss) scenario mirroring `regression_skeleton`. Pin `bootstrap_type=No`, `random_strength=0`, explicit `boost_from_average`, `leaf_estimation_iterations`, `score_function` (resolve Open Q1 — recommend L2).
- [ ] New oracle scenarios: `leaf_methods/{gradient,newton,exact,simple}`, `bootstrap/{poisson,bayesian,bernoulli,mvs,no}`, `regularization/{l2,random_strength,bagging_temp}`, `overfit/{wilcoxon,inctodec,iter,use_best_model}`, `eval_metrics`, `autolr`.
- [ ] `model.json` parser in `cb-oracle` (or `cb-train` tests) extracting `oblivious_trees[i].splits` (float_feature_index, border) and `leaf_values` into `Vec<f64>` for `compare_stage(Stage::Splits/LeafValues, …)`.
- [ ] `cb-backend` build spike test: minimal `#[cube]` kernel on `CpuRuntime` (Open Q2) — must compile under deny-lints.
- [ ] Framework install: add `cubecl = { version = "0.10.0", features = ["cpu"] }` + `bytemuck` to `[workspace.dependencies]` and `cb-backend`'s manifest.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CubeCL `CpuRuntime` actually executes kernels (vs. compiles) | TRAIN-01 | First-time seam stand-up; build spike may surface env-specific issues | Run `cargo test -p cb-backend kernels::gradient -- --nocapture` and confirm kernel output matches a host-computed reference |

*All other phase behaviors have automated oracle/unit verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (fixtures, model.json parser, cubecl install)
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
