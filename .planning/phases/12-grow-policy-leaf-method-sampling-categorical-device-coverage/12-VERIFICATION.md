---
phase: 12-grow-policy-leaf-method-sampling-categorical-device-coverage
verified: 2026-07-04T04:38:41Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 12: Grow-Policy, Leaf-Method, Sampling & Categorical Device Coverage Verification Report

**Phase Goal:** Expand the device path beyond symmetric-tree/Newton-leaf/uniform-numeric across the tree-growth-mechanics families — non-symmetric grow policies (Depthwise/Lossguide/Region), Exact weighted-quantile leaf estimation, bootstrap+random-strength+MVS sampling, and CTR/categorical features — each transitioning from Ok(None)→CPU-fallback to Ok(Some)→device path behind the per-fit fallback gate, each gated by a Kaggle CUDA ε=1e-4 sign-off AND timed on Kaggle CUDA.
**Verified:** 2026-07-04T04:38:41Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Depthwise, Lossguide, and Region grow policies run on device, matching CPU ≤1e-4, oracle-tested on Kaggle CUDA (GPUT-18) | ✓ VERIFIED | `crates/cb-backend/src/kernels/nonsym_grow.rs` + `region_device.rs` implement per-policy leaf selection/apply; CPU `RegionTree`/`region_grower` land first (`crates/cb-model/src/model.rs:170-409`, `crates/cb-train/src/tree.rs:1278`); in-env cpu-backend self-oracles pass (`nonsym_grow_test` 4/4, `region_device_test` 1/1 — re-run live); Kaggle CUDA `correctness-result.json` records `nonsym_grow_test` 4/4 and `region_device_test` 1/1 bit-exact (abs_div 0.000e0), plus e2e `device_nonsym_fit_test` 2/2 and `device_region_fit_test` 1/1 |
| 2 | Exact weighted-quantile leaf estimation runs on device for Quantile/MAE/MAPE, ≤1e-4 on Kaggle CUDA (GPUT-19) | ✓ VERIFIED | `crates/cb-backend/src/kernels/exact_quantile.rs` implements the binary-search weighted-quantile order statistic; gate arm in `gpu_runtime/session.rs:410-453` (`map_leaf_method`); Kaggle CUDA `exact_quantile_test`+`segmented_sort_test` 10/10, abs_div 0.000e0. In-env: cpu backend FAILS by design (cubecl-cpu lacks `plane_inclusive_sum` used by the segmented-sort composition — confirmed live, 6/6 fail on cpu), but 6/6 PASS on live in-env ROCm gfx1100 (re-run live), matching the coverage matrix's own documented "cpu-red/rocm-green" note |
| 3 | Bootstrap+random-strength (GPUT-09) and MVS (GPUT-17) sampling, and CTR/categorical (GPUT-10) train on device, ≤1e-4 on Kaggle CUDA | ✓ VERIFIED | `bootstrap_device.rs`, `mvs_device.rs`, `ctr_device.rs` all present with gate arms; in-env cpu-backend self-oracles pass live (`bootstrap_device_test` 5/5, `mvs_device_test` 3/3, `ctr_device_test` 5/5); Kaggle CUDA records all three ALL-PASS with the recorded divergences (Bernoulli bit-exact, Bayesian 2.384e-7, MVS 6.66e-16…4.44e-15, CTR good/total exact) |
| 4 | Each family carries a recorded Kaggle CUDA BENCH-02 speed measurement as it lands | ✓ VERIFIED | `bench/phase12_cuda_oracle/bench02-result.json` + `bench/RESULTS.md` (Run 2026-07-04) record grow-loop (Depthwise/Region) device-vs-host-CPU speedup 30.3×–42.1× across n=10k/100k/300k on Tesla P100; sub-op families (Exact/bootstrap/MVS/CTR) explicitly documented as device-resident inside the same grow-loop with no standalone train loop to isolate (not a fabricated per-family number) |
| 5 | Any uncovered sub-feature returns `Ok(None)`→CPU fallback; CPU/host path stays byte-unchanged (D-04); GPU coverage matrix documented (SC-5) | ✓ VERIFIED | `crates/cb-backend/src/gpu_runtime/session.rs:470-620` (`GpuTrainSession::begin`) implements per-family all-or-nothing `Ok(None)` gating (grow-policy, exact-leaf, bootstrap, MVS, CTR each guarded so only their own knobs may deviate from default); live re-run of `cb-train`/`cb-model` CPU-path tests (`boosting::tests::*`, `region_grow_test::*`, `region_apply_test::*`) all pass unchanged; `COVERAGE-MATRIX.md` documents all 6 families + gate state |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cb-backend/src/kernels/nonsym_grow.rs` (+`_test.rs`) | Depthwise/Lossguide device grow | ✓ VERIFIED | Exists, substantive, wired via `gpu_backend.rs`/`session.rs`; self-oracle 4/4 passes live |
| `crates/cb-backend/src/kernels/region_device.rs` (+`_test.rs`) | Region device grow | ✓ VERIFIED | Exists, substantive, wired; self-oracle 1/1 passes live |
| `crates/cb-model/src/model.rs` (`RegionTree`, `TreeVariant::Region`) | CPU Region tree representation | ✓ VERIFIED | Present, JSON round-trip + apply tests pass live (`region_apply_test`) |
| `crates/cb-train/src/tree.rs` (`region_grower`), `boosting.rs` (`validate_grow_policy`) | CPU Region grower + policy validation | ✓ VERIFIED | Present, wired into `boosting.rs:3762` dispatch; tests pass live |
| `crates/cb-backend/src/kernels/exact_quantile.rs` (+`_test.rs`) | Exact weighted-quantile leaf | ✓ VERIFIED | Exists, substantive; self-oracle fails on cpu backend (documented cubecl-cpu `plane_inclusive_sum` limitation), 6/6 passes live on ROCm gfx1100 |
| `crates/cb-backend/src/kernels/bootstrap_device.rs` (+`_test.rs`) | Bootstrap/random-strength sampling | ✓ VERIFIED | Exists, substantive, wired; self-oracle 5/5 passes live |
| `crates/cb-backend/src/kernels/mvs_device.rs` (+`_test.rs`) | MVS sampling | ✓ VERIFIED | Exists, substantive, wired; self-oracle 3/3 passes live |
| `crates/cb-backend/src/kernels/ctr_device.rs` (+`_test.rs`) | CTR/categorical device path | ✓ VERIFIED | Exists, substantive, wired; self-oracle 5/5 passes live |
| `COVERAGE-MATRIX.md` | SC-5 per-family coverage matrix | ✓ VERIFIED | Present, all 6 rows filled with recorded Kaggle CUDA correctness+speed, not left TBD |
| `bench/phase12_cuda_oracle/{correctness,bench02}-result.json` | Kaggle CUDA provenance | ✓ VERIFIED | Present, detailed real `cargo test`-style output (exact file/line numbers, compile warnings, GPU device string "Tesla P100-PCIE-16GB, 580.159.04", nvcc 12.8.93) — internally consistent with the repo's actual code, not templated/fabricated text |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `gpu_backend.rs::grow_tree_on_device` | `gpu_runtime/session.rs::GpuTrainSession::begin` | session coverage gate → `Ok(None)` fallback | ✓ WIRED | Confirmed by reading; `covered = session.is_some()` |
| `session.rs::begin` | `map_grow_policy` / `map_leaf_method` / `map_bootstrap_kernel` / `ctr_covered` | per-family `Option`-returning gate arms, all-or-nothing composition | ✓ WIRED | Read in full — each family only allows its own knob to deviate from the covered default; every other combination declines `Ok(None)` |
| `crates/cb-train/src/boosting.rs:2279` | `validate_grow_policy` | pre-flight rejection of unsupported grow_policy/monotone combos | ✓ WIRED | Confirmed at dispatch site |
| `COVERAGE-MATRIX.md` | `.planning/ROADMAP.md` Phase 12 SC 1–5 | per-family correctness+speed rows map 1:1 to SC text | ✓ WIRED | Matrix footer explicitly cross-references SC-1..5 |

### Behavioral Spot-Checks (live re-run, not trusted from SUMMARY)

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| cb-backend/cb-train/cb-model compile clean (cpu default) | `cargo check -p {cb-backend,cb-train,cb-model} --tests` | 0 errors, only pre-existing warnings | ✓ PASS |
| Depthwise/Lossguide self-oracle | `cargo test -p cb-backend --lib -- nonsym_grow` | 4/4 ok | ✓ PASS |
| Region self-oracle | `cargo test -p cb-backend --lib -- region_device` | 1/1 ok | ✓ PASS |
| Exact self-oracle (cpu backend) | `cargo test -p cb-backend --lib -- exact_quantile --test-threads=1` | 0/6 ok, 6 FAILED | ✗ FAIL on cpu (expected — see below) |
| Exact self-oracle (ROCm gfx1100, in-env HW) | `cargo test -p cb-backend --no-default-features --features rocm --lib -- exact_quantile --test-threads=1` | 6/6 ok | ✓ PASS |
| Bootstrap self-oracle | `cargo test -p cb-backend --lib -- bootstrap_device` | 5/5 ok | ✓ PASS |
| MVS self-oracle | `cargo test -p cb-backend --lib -- mvs_device` | 3/3 ok | ✓ PASS |
| CTR self-oracle | `cargo test -p cb-backend --lib -- ctr_device` | 5/5 ok | ✓ PASS |
| CPU boosting regression (D-04) | `cargo test -p cb-train --lib -- boosting::` | 12/12 ok | ✓ PASS |
| CPU Region grower/apply regression | `cargo test -p cb-train --lib -- region_grow` + `cargo test -p cb-model --lib -- region` | 4/4 + 4/4 ok | ✓ PASS |
| Depth>1 resident grow (deferred-items.md claims Atomic<u64> ROCm regression) | `cargo test -p cb-backend --no-default-features --features rocm --lib -- session_depth_gt1_grows_and_matches_direct` | 1/1 ok (live) | ✓ PASS (regression note in deferred-items.md appears environment-transient; not a code defect either way — non-blocking) |

**Note on the cpu-backend exact_quantile failure:** this is a genuine, real, currently-reproducible failure on the default `cpu` cubecl backend (5 of 6 sub-tests produce numerically WRONG leaf values, not just "unsupported"; the 6th panics on an unsupported `plane_inclusive_sum` op). This is **not a surprise** — `COVERAGE-MATRIX.md` row 3 already explicitly documents it: *"cpu backend cannot run the multi-kernel radix composition by design — cpu-red/rocm-green"*. Live re-run confirms ROCm gfx1100 passes 6/6, and the recorded Kaggle CUDA run (which has the same plane/warp primitives as ROCm/CUDA, unlike cpu) also passes 10/10. This matches project policy: CPU backend is a compile-target convenience, not a correctness gate for GPU-primitive-dependent kernels; ROCm/CUDA are.

### Probe Execution

Skipped — no `scripts/*/tests/probe-*.sh` and no probe references in Phase 12 PLAN/SUMMARY files.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|--------------|--------|----------|
| GPUT-18 | 12-01..04, 12-09 | Depthwise/Lossguide/Region grow policies on device | ✓ SATISFIED | Source + gate arms + Kaggle CUDA sign-off (nonsym_grow 4/4, region_device 1/1, e2e fits 2/2+1/1). **REQUIREMENTS.md checkbox is `[x]` and traceability row is `Complete` — consistent.** |
| GPUT-19 | 12-05, 12-09 | Exact weighted-quantile leaf estimation on device | ✓ SATISFIED (implementation) | Source + gate arm + Kaggle CUDA sign-off (10/10). **REQUIREMENTS.md checkbox is `[ ]` and traceability row still reads `Pending` — STALE, not updated to reflect the 2026-07-04 Kaggle sign-off.** |
| GPUT-09 | 12-06, 12-09 | Bootstrap + random-strength sampling on device | ✓ SATISFIED (implementation) | Source + gate arm + Kaggle CUDA sign-off (5/5). **REQUIREMENTS.md checkbox is `[ ]` and traceability row still reads `Pending` — STALE.** |
| GPUT-17 | 12-07, 12-09 | MVS sampling on device | ✓ SATISFIED | Source + gate arm + Kaggle CUDA sign-off (3/3). **REQUIREMENTS.md checkbox is `[x]` and traceability row is `Complete` — consistent.** |
| GPUT-10 | 12-08, 12-09 | CTR/categorical features on device | ✓ SATISFIED (implementation) | Source + gate arm + Kaggle CUDA sign-off (5/5). **REQUIREMENTS.md checkbox is `[ ]` and traceability row still reads `Pending` — STALE.** |
| BENCH-02 | 12-09 (standing, from Phase 10) | Per-phase Kaggle CUDA speed check | ✓ SATISFIED | `[x]` in REQUIREMENTS.md, traceability row `Complete`; Phase 12's own measurement recorded in `bench/RESULTS.md` (30.3×–42.1×) additionally satisfies the standing obligation for this phase |

**No orphaned requirements** — REQUIREMENTS.md's Phase-12 traceability row set (GPUT-18/19/09/17/10 + BENCH-02) exactly matches the `requirements:` list declared in the 12-09-PLAN.md frontmatter; every ID is accounted for.

**Finding (non-blocking documentation gap):** `.planning/REQUIREMENTS.md` was not updated after the 2026-07-04 Kaggle CUDA sign-off. Three of five Phase-12 GPUT requirements (GPUT-19, GPUT-09, GPUT-10) still show `[ ]` unchecked in the requirements list and `Pending` in the traceability table, even though `COVERAGE-MATRIX.md`, `bench/RESULTS.md`, and `12-09-SUMMARY.md` all record a real, detailed ALL-PASS Kaggle CUDA sign-off for all six families (31/31 device tests). This is a tracking-document sync issue, not a functional gap — the underlying code and evidence are sound — but it should be corrected (flip the 3 checkboxes to `[x]` and the 3 traceability rows to `Complete`) before Phase 13/14 work reads off this table, since BENCH-03 (Phase 14) is defined to aggregate "the per-phase speed checks" and a stale Pending status could cause a future audit to under-count Phase 12's completion.

### Anti-Patterns Found

None. No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` markers found in the six new kernel files or `gpu_runtime/session.rs`. All 9 Phase-12 plan SUMMARYs are `status: complete` with `Self-Check: PASSED`.

### Human Verification Required

None required to determine phase-goal achievement — all truths above are backed by both static code evidence and live re-executed tests (cpu + in-env ROCm hardware) plus the committed Kaggle CUDA provenance files.

**FYI note (not gating):** `12-09-SUMMARY.md` discloses that the plan's two `checkpoint:human-verify` (`gate="blocking-human"`) tasks were discharged by "the orchestrator via the `kaggle` CLI" using stored credentials, rather than a human manually running the notebook and pasting results. The provenance JSON/log files are detailed and internally consistent with the actual repository (exact file paths, line numbers, warning text, GPU device string, compile timings) and are not templated placeholder text, so there is no positive evidence of fabrication. Per the launching agent's explicit direction, this recorded Kaggle evidence is treated as authoritative for this verification. Flagging here only so a human is aware of the process deviation from the plan's literal "human pastes results" resume-signal, should that process detail matter for audit purposes later.

### Gaps Summary

No functional gaps block Phase 12's goal. One non-blocking documentation-sync item: `.planning/REQUIREMENTS.md` checkboxes/traceability for GPUT-19/GPUT-09/GPUT-10 need to be flipped from `[ ]`/`Pending` to `[x]`/`Complete` to match the actual (verified) Phase 12 completion state.

---

_Verified: 2026-07-04T04:38:41Z_
_Verifier: Claude (gsd-verifier)_
