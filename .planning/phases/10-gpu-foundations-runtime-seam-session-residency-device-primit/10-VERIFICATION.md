---
phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit
verified: 2026-07-03T03:14:08Z
status: human_needed
score: 5/5 must-haves structurally verified in-env; 0 failed; 5 items await the authoritative human-gated Kaggle CUDA run
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "Run bench/cuda_oracle.ipynb on a Kaggle CUDA (NVIDIA) instance: build the --features cuda wheel, run nvidia-smi, then the BLOCKING correctness gate (primitive/segmented-scan/sort/reduce-by-key/segmented-reduce/update_part_props <=1e-4 or bit-exact; cindex bit-exact; depth-1 RMSE/Logloss <=1e-5 Cosine + first-order calc_average leaves), then the warm-run/JIT-excluded train-only large-n (~1e6x50) speed measurement (device vs host-CPU baseline, vs official CatBoost GPU where comparable)."
    expected: "All 9 correctness rows in bench/RESULTS.md's run-log table PASS (GATE: PASS); the large-n speed row is filled with device/host-CPU/CatBoost-GPU wall-clock numbers and the device>=CPU crossover (or its documented absence per D-10-09) is recorded."
    why_human: "This milestone's own validation authority (ROADMAP.md, PROJECT.md) designates the Kaggle CUDA notebook as the SOLE authoritative GPU oracle for both correctness (<=1e-5 depth-1 / <=1e-4 primitives) and wall-clock speed (BENCH-01/02). There is no NVIDIA hardware in this environment; ROCm in-env is explicitly documented as smoke-only, never a gate. bench/RESULTS.md currently records only TBD placeholders (correctly, per D-do-not-fabricate) — no run has happened yet."
  - test: "Independently confirm (spot-check on a real device, or accept as tracked debt) the WR-01 code-review finding: grow_oblivious_tree_resident (crates/cb-backend/src/gpu_runtime/mod.rs:2104, called every boosting iteration via GpuTrainSession::grow_one) performs an n-length device->host read-back of leaf_of on EVERY production tree, not only for an oracle/test path."
    expected: "Either (a) accept this as intentional, disclosed, non-blocking debt (a performance/residency-completeness gap, not a correctness gap — leaf_of does not affect the trained model's numeric output) tracked for removal once the CPU fallback path is eventually retired, or (b) file a follow-up plan to eliminate the per-tree leaf_of crossing before Phase 11 optimizes depth>1 residency further."
    why_human: "ROADMAP.md Success Criterion 4 states literally 'only the O(1) BestSplit descriptor + 2^depth partition statistics cross host<->device per level' — the current code additionally crosses an n-length leaf_of buffer once per tree. This is a verified, reproducible fact (grep+read confirmed in this verification), already self-disclosed in 10-REVIEW.md WR-01 and in the 10-07 SUMMARY's 'Next Phase Readiness' section as a deferred hot-path optimization. It does not change correctness (the CPU-vs-device tree structure still matches), but it is a literal deviation from the roadmap's SC4 wording that a human should consciously accept or schedule closure for."
gaps: []
---

# Phase 10: GPU Foundations — Runtime Seam, Session Residency, Device-Primitive Library, Compressed Index, Depth-1 + Kaggle CUDA Oracle & Speed Harness Verification Report

**Phase Goal:** The from-scratch CubeCL device-primitive substrate (no CUB) and the device-resident compressed index land; the device grow loop becomes reachable from `fit()`; training data stays device-resident; depth-1 oblivious trees grow on device with the Cosine GPU-default score; the foundational Kaggle CUDA harness measures BOTH correctness (≤1e-5) AND wall-clock speed from the start.
**Verified:** 2026-07-03T03:14:08Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

Merged from ROADMAP.md's 5 Success Criteria (the roadmap contract) and the 9 plans' `must_haves.truths`. All in-repo/in-env checks below were independently re-run in this verification session (not taken from SUMMARY.md claims): builds, `cargo test`, greps, and direct code reads.

| # | Truth (source) | Status | Evidence |
|---|---|---|---|
| 1 | SC1: from-scratch device-primitive library (scan/segmented-scan/reduce/segmented-reduce/reduce-by-key/sort/reorder/bit-compression/TDataPartition update/fill-transform) runs on device with a deterministic reduction, matches CPU ≤1e-4, oracle-tested on Kaggle CUDA | ⚠️ In-env verified / Kaggle open | All kernels present in `crates/cb-backend/src/kernels.rs` (3955 lines); each has a separate-file self-oracle (`kernels/scan.rs` 10 tests, `segmented_scan.rs` 4, `reduce.rs` 8 incl. 32-launch zero-spread determinism harness at reduce.rs:812, `sort.rs` 5, `partitions.rs` 3, `fill_transform.rs` 3, `compression.rs` 4, `update_part_props.rs` 4, `cindex.rs` 6) — test counts match the SUMMARY-claimed rocm pass counts. `cargo check -p cb-backend --no-default-features --features rocm` compiles clean (re-run this session). Kaggle CUDA `<=1e-4` sign-off is the authoritative bar (ROADMAP + PROJECT validation-authority note) and is explicitly TBD in `bench/RESULTS.md` — not yet run. |
| 2 | SC2: bit-packed device-resident cindex (`TCFeature` Offset/Shift/Mask/OneHot) matches CPU quantized layout ≤1e-4, oracle-tested on Kaggle CUDA | ⚠️ In-env verified / Kaggle open | `crates/cb-backend/src/gpu_runtime/cindex.rs` (`TCFeature`, `pack_cindex`, checked arithmetic) + `crates/cb-backend/src/kernels.rs` (`read_bin` accessor) + `crates/cb-backend/src/kernels/cindex.rs` (6 tests, bit-exact oracle) all FOUND. Grep-confirmed every consumer (`pointwise_hist2` family, `partition_split_kernel`, `host_reference_hist2`) routes through `read_bin` — no remaining plain `cindex[feature*n+obj]` extraction. Kaggle CUDA bit-exact sign-off TBD in `bench/RESULTS.md`. |
| 3 | SC3: depth-1 oblivious tree (RMSE/Logloss, Plain, fold_count=1) grows fully on device with Cosine score, matches CPU ≤1e-5, **oracle-tested on Kaggle CUDA (not merely ROCm)** | ⚠️ In-env structure-verified / Kaggle open | `GpuTrainSession` (`session.rs`), `grow_oblivious_tree_resident` (`gpu_runtime/mod.rs:1968`), `apply_leaf_delta_kernel`/`launch_apply_leaf_delta_into` (`apply_leaf_delta.rs`), Cosine wired as `SCORE_FN_COSINE` default all FOUND and wired. `session_residency.rs` (3 tests) exercises a 5-tree device pass matching CPU multi-tree Cosine boosting **structure** exactly (per 10-07 SUMMARY, itself explicit that leaf-value numeric divergence is "REPORTED (not signed-off)"). ROADMAP explicitly requires the Kaggle run as authoritative for the ≤1e-5 bar — TBD in `bench/RESULTS.md`. |
| 4 | SC4: `Runtime` grow-tree seam reachable from `cb_train::train`; quantized matrix uploads exactly once per fit; approx/der1 stay device-resident (no per-tree der1 read-back); only O(1) BestSplit + `2^depth` partition stats cross host↔device per level (D-05) | ✓ VERIFIED, with one disclosed caveat | `cargo test -p cb-train --test device_seam_test` → **6/6 passed** (re-run this session): bin_id→border join, all-or-nothing rejection, out-of-range bin_id typed error, CR-01 (`boost_from_average`/non-zero bias) decline, CR-02 (Newton leaf method) decline, `Ok(false)`→CPU-fallback all pass. `cargo test -p cb-train --lib` → 231/231 passed (D-04 CPU no-regression, re-run this session). Code-read confirms `der1_h` is never read back (only passed/chained as a `Handle`) — matches the plan's "no n-length der1 read-back" wording exactly. **Caveat (see human_verification item #2):** `grow_oblivious_tree_resident` DOES read back an n-length `leaf_of` buffer once per tree in the production path (confirmed by direct code read of `gpu_runtime/mod.rs:2104`, called from every `GpuTrainSession::grow_one`) — this is an additional crossing beyond the roadmap's literal "only O(1) + 2^depth" wording, already self-disclosed as WR-01 in `10-REVIEW.md` and as deferred debt in the 10-07 SUMMARY. Non-blocking: it does not change model correctness, only bus traffic. |
| 5 | SC5/BENCH-01/02: reproducible Kaggle CUDA harness builds `--features cuda` wheel, runs correctness (blocking gate) then warm-run/JIT-excluded train-only speed; one generator sources both fixtures (D-06); D-10-09 small-n infeasibility surfaced explicitly; standing per-phase speed-check discipline established | ✓ Structurally VERIFIED / Kaggle run open | `bench/generator.py` parses and is deterministic (`--check` regenerates 17 fixtures bit-for-bit, per SUMMARY, files present); `bench/cuda_oracle.ipynb` is valid notebook JSON (13 cells, re-verified this session); `bench/RESULTS.md` records the D-10-09 escalation prominently ("large n" match, re-grepped this session) and contains ONLY `TBD` placeholders for measured values (no fabricated numbers — confirmed by direct read); `SPIKE-REDUCTION.md` contains a firm `Recommendation` (re-grepped this session). This is exactly the "harness must be structurally complete and reproducible in-env, with authoritative numeric sign-off deferred to the human Kaggle run" bar set for this verification — met. |

**Score:** 5/5 truths structurally verified in-env (code present, wired, builds, self-oracled, tests green); 0 failed; all 5 carry an explicitly-designed-in human-gated Kaggle CUDA sign-off that has not yet been run (correctly recorded as TBD, not fabricated).

### Critical Code-Review Findings — Confirmed Fixed

Two Critical findings from `10-REVIEW.md` were independently re-verified in this session (not merely trusted from the review doc):

- **CR-01** (device path silently ignored `boost_from_average`, causing a wrong-but-silent model bias): `crates/cb-train/src/boosting.rs:2937-2939` now includes `&& bias == 0.0` in `device_host_eligible`, with an inline comment explaining the CR-01 rationale. Regression test `device_declines_nonzero_starting_bias_boost_from_average` (boosting.rs test file) — **PASSED** in this session's `cargo test -p cb-train --test device_seam_test` run.
- **CR-02** (device path silently ignored `LeafMethod::Newton`): `boosting.rs:2942-2946` now includes `&& matches!(params.leaf_method, LeafMethod::Gradient | LeafMethod::Simple)`. Regression test `device_declines_newton_leaf_method_on_covered_loss` — **PASSED** in this session's run.

Both fixes are present in the code (not just claimed in SUMMARY.md) and both regression tests pass.

### Required Artifacts

All 23 artifacts named across the 9 plans' `must_haves.artifacts` were checked for existence, substance (non-trivial line counts, real logic, no placeholder returns), and wiring. All FOUND, all substantive, all wired — none are stubs.

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `crates/cb-backend/src/kernels.rs` | all `#[cube]` kernel bodies (scan/reduce/sort/partitions/fill/compression/update_part_props/cindex/apply_leaf_delta) | ✓ VERIFIED | 3955 lines; contains `segmented_scan`, `reduce_by_key`, `reorder_one_bit`, `update_part_props`, `read_bin`, `TCFeature`-adjacent accessor — all grep-confirmed present |
| `crates/cb-backend/src/kernels/{scan,segmented_scan,reduce,sort,partitions,fill_transform,compression,update_part_props,cindex,apply_leaf_delta}.rs` | separate-file self-oracles | ✓ VERIFIED | all 10 files FOUND with test counts matching SUMMARY claims; proper `#[cfg(test)] mod X;` mounting pattern in kernels.rs (source/test separation honored, no embedded `mod tests` in production kernel bodies) |
| `crates/cb-compute/src/runtime.rs` | `DeviceGrownTree` + 3 default-impl seam methods | ✓ VERIFIED | 1079 lines; `DeviceGrownTree`, `begin_device_training`/`grow_tree_on_device`/`end_device_training` present with `Ok(false)`/`Ok(None)`/`Ok(())` defaults; `grep cubecl crates/cb-compute/Cargo.toml` shows only the pre-existing D-03 comment — no cubecl/cb-backend dep added |
| `crates/cb-backend/src/gpu_runtime/cindex.rs` | `TCFeature` + host bit-packer + bit-exact oracle | ✓ VERIFIED | 258 lines; `TCFeature`, `pack_cindex`, checked-arithmetic guards present |
| `crates/cb-backend/src/gpu_runtime/session.rs` | `GpuTrainSession` residency wrapper | ✓ VERIFIED | 349 lines; one `ComputeClient` + persistent handles, `begin()` coverage gate, `grow_one()`, Drop-free (handles freed by client drop) |
| `crates/cb-backend/src/gpu_runtime/session_residency.rs` | residency + coverage-gate test | ✓ VERIFIED | 341 lines, 3 tests |
| `crates/cb-backend/src/gpu_backend.rs` | `RefCell<Option<GpuTrainSession>>` + 3 seam impls | ✓ VERIFIED | `RefCell` field, `begin_device_training`/`grow_tree_on_device`/`end_device_training` overrides present, `gpu_backend_test.rs` 7 tests |
| `crates/cb-train/src/boosting.rs` | device branch + bin→border join + `Ok(None)` fallback | ✓ VERIFIED | `grow_tree_on_device` call present at boosting.rs:2998; `device_host_eligible` gate; `DeviceSessionGuard` RAII teardown declared before the iteration loop (structurally guarantees `end_device_training` fires on every `?`-early-return, per Rust's Drop-on-scope-exit guarantee) |
| `crates/cb-train/tests/device_seam_test.rs` | CPU-only structural fold test | ✓ VERIFIED | 6 tests, all pass |
| `bench/generator.py`, `bench/cuda_oracle.ipynb`, `bench/fixtures/README.md`, `bench/RESULTS.md` | Kaggle harness | ✓ VERIFIED | all present, parse/validate cleanly, no fabricated numbers |
| `.planning/.../SPIKE-REDUCTION.md` | reduction-determinism spike report | ✓ VERIFIED | 160 lines, `Recommendation` section present |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `kernels/scan.rs` | `kernels.rs` | `launch::<F, SelectedRuntime>` two-level scan | ✓ WIRED | `full_scan`/`full_scan_into` present and called by scan.rs oracle |
| `kernels/reduce.rs` | `kernels.rs` | `AtomicFinalizePath`/`ReduceFinalizeStrategy` variance harness | ✓ WIRED | 32-launch harness present, asserts reported strategy matches capability |
| `kernels.rs` (sort/partitions) | `kernels.rs` (scan) | `reorder_one_bit`/partition update consume `full_scan_into` | ✓ WIRED | grep-confirmed `full_scan_into` reused by radix sort + partition update |
| `kernels/update_part_props.rs` | `kernels.rs` | `sum_f64` / 10-03 deterministic reduce winner reused | ✓ WIRED | fixed-order f64 tree reduce reused verbatim per SUMMARY, grep-confirmed present |
| `kernels.rs` (`read_bin`) | `gpu_runtime/cindex.rs` (`TCFeature`) | every histogram/partition bin read routes through `read_bin` | ✓ WIRED | grep across `gpu_runtime/` + `kernels/` finds no remaining plain `cindex[feature*n+obj]` extraction in migrated kernels |
| `gpu_backend.rs` | `gpu_runtime/session.rs` | `begin` builds+stores session; `grow` borrows; `end` drops | ✓ WIRED | `RefCell<Option<GpuTrainSession>>` lifecycle confirmed by direct read |
| `gpu_runtime/mod.rs` | `gpu_runtime/der_seams.rs` | `der1_h = der(approx_h, target)` chained, no read-back | ✓ WIRED | confirmed: `launch_der_binary_resident` called every tree, no `read_*_handle` call on `der1_h` anywhere in the resident path |
| `cb-train/src/boosting.rs` | `cb-compute/src/runtime.rs` | `train_inner` calls the seam, maps `(feature,bin_id)`→border | ✓ WIRED | `feature_borders[feature][bin_id]` resolution with checked `.get` range guard confirmed present |
| `bench/cuda_oracle.ipynb` | `bench/generator.py` | notebook imports generator for both fixture and speed workload | ✓ WIRED | confirmed via notebook cell content (D-06 single-source) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| cb-train CPU no-regression (D-04) | `cargo test -p cb-train --lib` | 231 passed, 0 failed | ✓ PASS |
| Device-seam structural fold correctness | `cargo test -p cb-train --test device_seam_test` | 6 passed, 0 failed | ✓ PASS |
| cb-backend rocm feature compiles | `cargo check -p cb-backend --no-default-features --features rocm` | Finished, 0 errors | ✓ PASS |
| catboost-rs rocm feature compiles (no cb-train leak) | `cargo check -p catboost-rs --no-default-features --features rocm` | Finished, 0 errors | ✓ PASS |
| Kaggle notebook JSON validity | `python3 -c "json.load(...)"` | valid, 13 cells | ✓ PASS |
| Generator determinism (parse-level) | `python3 -c "ast.parse(...)"` | parses cleanly | ✓ PASS |
| GPU (rocm/cuda) runtime kernel execution | — | not run (no GPU access in this verifier's execution context) | ? SKIP — orchestrator-discharged per-plan rocm runs (17/17, 8/8, 5/5, 3/3, 3/3, 4/4, 4/4, 6/6, 44/44, 123/123 reported across the 9 SUMMARYs) accepted as evidence per task instructions; structural artifacts + test code corroborate the claimed counts |

### Probe Execution

No `scripts/*/tests/probe-*.sh` convention or PLAN/SUMMARY-declared probes found for this phase (`find scripts -path '*/tests/probe-*.sh'` → none; this project's GPU verification uses `cargo test --features {rocm,cuda}` directly, not a probe-script convention). Skipped — not applicable.

### Requirements Coverage

All 9 requirement IDs assigned to Phase 10 (`GPUT-01, GPUT-02, GPUT-03, GPUT-04, GPUT-08, GPUT-15, GPUT-16, BENCH-01, BENCH-02`) are declared across the 9 plans' frontmatter and marked `Complete` / `[x]` in both `REQUIREMENTS.md`'s traceability table and `ROADMAP.md`'s phase checklist. Cross-referenced 1:1, no orphans, no gaps:

| Requirement | Plan(s) | Description | Status | Evidence |
|---|---|---|---|---|
| GPUT-01 | 10-02, 10-08 | Runtime grow-tree seam + reachable from `cb_train::train` | ✓ SATISFIED | seam methods + boosting.rs device branch, both tested |
| GPUT-02 | 10-07 | `GpuTrainSession` upload-once residency | ✓ SATISFIED | `session.rs::begin` uploads all handles once |
| GPUT-03 | 10-07 | approx/der1 device-resident, no per-tree der1 read-back | ✓ SATISFIED (with WR-01 caveat on `leaf_of`, non-blocking) | der1_h never read back; `leaf_of` (a different buffer) crosses once/tree — see human_verification #2 |
| GPUT-04 | 10-07, 10-08 | Depth-1 tree grows fully on device, matches CPU ≤1e-5 | ✓ SATISFIED in-env / Kaggle sign-off open | structural match confirmed; numeric ≤1e-5 sign-off is Kaggle-authoritative per ROADMAP |
| GPUT-08 | 10-07 | Cosine score wired as device default | ✓ SATISFIED | `SCORE_FN_COSINE` wired, `map_score_fn` defaults align with CatBoost's Cosine default |
| GPUT-15 | 10-06 | Bit-packed cindex + `read_bin` | ✓ SATISFIED | `TCFeature`, `pack_cindex`, `read_bin`, full consumer migration confirmed |
| GPUT-16 | 10-01, 10-03, 10-04, 10-05 | From-scratch device-primitive library | ✓ SATISFIED | scan/segmented-scan/reduce/segmented-reduce/reduce-by-key/sort/reorder/partitions/fill-transform/compression/update_part_props all present + self-oracled |
| BENCH-01 | 10-09 | Reproducible Kaggle CUDA harness (correctness+speed) | ✓ SATISFIED (harness); Kaggle run open | notebook + generator + RESULTS.md all present and valid |
| BENCH-02 | 10-09 | Standing per-phase speed-check discipline established | ✓ SATISFIED | RESULTS.md run-log template + D-10-09 escalation established for reuse by Phases 11-13 |

No orphaned requirements: `grep -E "Phase 10" .planning/REQUIREMENTS.md` maps exactly these 9 IDs to Phase 10, and all 9 appear in at least one plan's `requirements:` frontmatter field.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| `crates/cb-backend/src/gpu_runtime/mod.rs` | 2104 (`grow_oblivious_tree_resident`) | n-length `leaf_of` device→host read-back on every production tree, contradicting the literal ROADMAP SC4 "only O(1) BestSplit + 2^depth partition stats cross per level" wording | ⚠️ Warning (disclosed, non-blocking) | No correctness impact (verified: does not change trained-model output); a bus-traffic / residency-completeness gap already self-flagged as WR-01 in `10-REVIEW.md` and as deferred debt in the 10-07 SUMMARY ("deferred until the CPU fallback is unwired"). Surfaced as human_verification item #2 for an explicit accept/schedule-closure decision. |
| `bench/generator.py` / `bench/fixtures/README.md` | — | (WR-02, pre-existing in the review, re-confirmed) committed `expected_*.npy` fixtures are sha256-pinned but never loaded by any Rust test, Python script, or the notebook — the primitive/cindex correctness gate runs as a Rust-internal self-oracle only | ℹ️ Info | Documentation/fixture-hygiene issue, not a functional gap; does not affect the phase goal. Already disclosed in `10-REVIEW.md`. |
| `crates/cb-backend/src/kernels.rs` | 1054-1066, 1173-1184 | (WR-03, pre-existing debt from Phase 7.4, in-scope-by-touch only) one-hot pairwise histogram double-`fetch_add`s the same cell | ℹ️ Info | Pre-existing, not introduced by Phase 10, not reachable from any shipped fixture yet (per the code review). No action required this phase. |

No `TBD`/`FIXME`/`XXX` unreferenced debt markers found in any of the 18 phase-touched production files (grepped in this session). No `TODO`/`HACK`/`PLACEHOLDER` markers found. No `unwrap()`/`expect()`/`panic!()` found in production kernel/session/boosting code (grepped in this session, workspace-lint-enforced). Source/test separation honored throughout — every `#[cfg(test)]` in production files is a `mod X;` declaration pointing to a separate file, never an embedded test body.

### Human Verification Required

1. **Kaggle CUDA correctness + speed sign-off (BENCH-01/02, GPUT-04, GPUT-15, GPUT-16 numeric bars)**
   **Test:** Run `bench/cuda_oracle.ipynb` on a Kaggle CUDA (NVIDIA) instance end-to-end.
   **Expected:** All 9 correctness rows PASS (`GATE: PASS`) in `bench/RESULTS.md`'s run-log; the large-n speed row is filled with device/host-CPU/CatBoost-GPU numbers.
   **Why human:** No NVIDIA hardware exists in this environment; this milestone's own PROJECT.md/ROADMAP.md designate the Kaggle CUDA run as the sole authoritative GPU oracle. ROCm in-env is documented as smoke-only. The harness itself is structurally complete and reproducible (verified this session) — only the external numeric run is pending, and no numbers have been fabricated.

2. **WR-01 `leaf_of` per-tree read-back — accept as tracked debt or schedule closure**
   **Test:** Review `crates/cb-backend/src/gpu_runtime/mod.rs:2104` (`grow_oblivious_tree_resident`, step 8) and confirm whether the n-length `leaf_of` read-back on every production tree is acceptable to carry forward, or should be closed before Phase 11 builds further on this residency architecture.
   **Expected:** An explicit accept (documented, e.g., via a VERIFICATION override or a tracked backlog item) or a follow-up plan.
   **Why human:** This is a literal deviation from ROADMAP.md's Success Criterion 4 wording ("only... O(1)... + 2^depth... cross... per level"), independently re-confirmed by code read in this session (not merely trusted from the review doc). It does not affect correctness (the depth-1 device tree structure still matches CPU) and is already self-disclosed by the team's own code review (WR-01, Warning severity) and SUMMARY docs as intentional, temporary debt — but a human should consciously ratify that judgment rather than have a verifier silently pass over it.

### Gaps Summary

No blocking gaps. Both Critical code-review findings (CR-01 `boost_from_average` bias, CR-02 Newton leaf method) were independently re-verified as fixed in the code with passing regression tests — not merely trusted from SUMMARY.md. All 9 requirement IDs are satisfied with wired, substantive, tested artifacts; all builds (`cb-compute`, `cb-train`, `cb-backend --features rocm`, `catboost-rs --features rocm`) succeed; the full re-run `cb-train` suite (231 lib tests) and the phase's own `device_seam_test` (6 tests) are green. The phase's numeric correctness/speed bars are, by explicit milestone-wide design (not a Phase-10-specific shortfall), gated behind a human-run Kaggle CUDA notebook that has not yet been executed — `bench/RESULTS.md` correctly shows `TBD`, not fabricated numbers, satisfying this verification's mandate to fail only on a missing/non-reproducible harness (which is not the case here). One disclosed, non-blocking discrepancy (the `leaf_of` per-tree read-back vs. the roadmap's literal "O(1)+2^depth only" wording) is surfaced for explicit human acknowledgment rather than silently absorbed into a `passed` verdict.

---

_Verified: 2026-07-03T03:14:08Z_
_Verifier: Claude (gsd-verifier)_
