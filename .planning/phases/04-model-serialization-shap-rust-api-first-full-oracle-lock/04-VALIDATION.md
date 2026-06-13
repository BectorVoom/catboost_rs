---
phase: 4
slug: model-serialization-shap-rust-api-first-full-oracle-lock
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-13
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (workspace) + `approx` 0.5 for float asserts + `cb-oracle::compare_stage` (≤1e-5 gate) |
| **Config file** | none (cargo built-in); CI clippy gate is `cargo clippy --lib` per Phase 1 |
| **Quick run command** | `cargo test -p cb-model` (or the touched crate) |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | full workspace pulls cubecl-cpu's heavy MLIR dep — watch disk headroom (STATE.md caution) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p <crate>` for the touched crate + `cargo clippy --lib` (restriction lints)
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd-verify-work`:** Full suite green + the five success criteria oracle-locked
- **Max feedback latency:** per-crate test run (seconds); workspace run gated by MLIR build

---

## Per-Task Verification Map

> Populated during planning/Wave 0. Each phase requirement maps to an oracle or unit test below.

| Req ID | Behavior | Test Type | Automated Command | File Exists |
|--------|----------|-----------|-------------------|-------------|
| MODEL-01 | save→load reproduces Model; load upstream `.cbm` apply ≤1e-5; upstream loads ours ≤1e-5 | integration (oracle) | `cargo test -p cb-model cbm` | ❌ W0 (upstream `.cbm` fixture + tests) |
| MODEL-02 | apply runs with no GPU toolchain; predictions ≤1e-5 | integration | `cargo test -p cb-model apply` | ❌ W0 |
| MODEL-03 | PredictionValuesChange + Interaction ≤1e-5 | integration (oracle) | `cargo test -p cb-model fstr` | ❌ W0 (importance fixtures) |
| MODEL-04 | per-object SHAP matrix ≤1e-5; `sum(shap)==prediction` | integration (oracle) | `cargo test -p cb-model shap` | ❌ W0 (SHAP `.npy` fixture) |
| MODEL-06 | JSON export round-trips via `cb-oracle::model_json`; matches upstream schema | integration | `cargo test -p cb-model json` | ⚠️ parser exists; needs `leaf_weights` extension + export tests |
| LOSS-01 | Logloss/CrossEntropy/Focal train ≤1e-5 (splits/leaf/staged) | integration (oracle) | `cargo test -p cb-train loss` | ❌ W0 (CrossEntropy + Focal fixtures) |
| LOSS-06 | RawFormulaVal/Probability/LogProbability/Class/Exponent ≤1e-5 | integration (oracle) | `cargo test -p cb-model predict` | ❌ W0 (per-type prediction fixtures) |
| RAPI-01 | `CatBoostBuilder...fit(&pool)->Model`, predict end-to-end | integration | `cargo test -p catboost-rs builder` | ❌ W0 |
| RAPI-02 | `CatBoostError` variants + `#[from] CbError`; Result equality | unit | `cargo test -p catboost-rs error` | ❌ W0 |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Leaf-weights capture in `cb-train::train` (structural prerequisite for SHAP/fstr) — **first task**
- [ ] `cb-oracle::model_json` extension: add `leaf_weights` per tree
- [ ] `crates/cb-model/src/cbm_test.rs` — MODEL-01 (round-trip + bidirectional interop)
- [ ] `crates/cb-model/src/apply_test.rs` — MODEL-02 (apply ≤1e-5)
- [ ] `crates/cb-model/src/shap_test.rs` — MODEL-04 (SHAP matrix + local accuracy)
- [ ] `crates/cb-model/src/fstr_test.rs` — MODEL-03 (PredictionValuesChange + Interaction)
- [ ] `crates/cb-model/src/json_test.rs` — MODEL-06
- [ ] `crates/cb-model/src/predict_test.rs` — LOSS-06
- [ ] `crates/catboost-rs/src/builder_test.rs`, `error_test.rs` — RAPI-01/02
- [ ] New committed fixtures (generated offline, D-13): upstream `.cbm` (1-dim binclf + regression); SHAP `.npy`; PredictionValuesChange/Interaction `.npy`; per-prediction-type `.npy`; CrossEntropy + Focal training fixtures
- [ ] `flatc` provisioning + committed generated FlatBuffers bindings (flatc not installed in CI)

*Existing infra reused: `cb-oracle::compare_stage` (≤1e-5), `.npy` fixture readers, `model_json` parser, generator scaffold (`gen_fixtures.py`).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Fixture generation (offline) | All oracle reqs | Python `catboost` not importable in this env; `flatc` not installed | Generate fixtures offline per D-13 and commit `.npy`/`.cbm`/`model.json` artifacts before automated oracle tests can pass |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency acceptable (per-crate seconds)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
