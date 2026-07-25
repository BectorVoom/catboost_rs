## Pass 3 — final revision verification

This is PLAN-CHECK pass 3 (final allowed pass) for GINF-01. Both of pass 2's
remaining findings were re-checked directly against the diffed sections of
SPEC.md / PLAN.md and against on-disk `apply.rs` (re-read verbatim,
`crates/cb-model/src/apply.rs:386-416`), plus a full from-scratch re-run of the
checker workflow (CodeGraph re-verification of every load-bearing symbol,
internal-consistency scan of both artifacts, and an adversarial look for
anything the two edits might have newly broken).

1. **[MAJOR pass-2 finding — ragged-column `n_objects` derivation] — RESOLVED
   (verified, not paper-over).**
   `crates/cb-model/src/apply.rs:397-399` re-read verbatim this pass: `let
   n_float = feature_values.first().map_or(0, Vec::len); let n_cat =
   cat_columns.first().map_or(0, Vec::len); let n_objects =
   n_float.max(n_cat);` — confirms, byte-for-byte, the rule the revision now
   states: for the numeric-only path (`cat_columns = &[]`, `n_cat == 0`),
   `n_objects` is governed EXCLUSIVELY by `feature_values.first()`'s length,
   not a max-over-all-columns rule. `apply.rs:404-407` (re-read verbatim) also
   confirms the per-cell NaN-pad citation is exact:
   `feature_values.iter().map(|col| col.get(obj).copied().unwrap_or(f32::NAN))`.
   Both citations are used correctly and at the exact cited line numbers in:
   - SPEC.md `predict_raw_on_device` doc comment (§4, lines 322-330): states
     the rule, cites both line ranges, explicitly contrasts
     "first-column-governs" vs "max-over-columns", and explains the
     truncation/NaN-pad asymmetry.
   - SPEC.md §6 AT-S5c (line 492): requires a NON-FIRST-column-LONGER-than-first
     case specifically because it is "the only combination that distinguishes
     'first-column-governs `n_objects`' from 'max-over-columns'".
   - PLAN.md TASK-05 Files (lines 431-438) and CodeGraph/Read evidence (lines
     449-455): same rule, same citations, explicitly flagged as "added per
     PLAN-CHECK pass 2 MAJOR finding".
   - PLAN.md TASK-05 Red (`predict_on_device_matches_cpu_ragged_columns`,
     lines 463-477): now explicitly requires BOTH a shorter-later-column case
     (a) AND a longer-later-column case (b), "both required (not
     optional/either-or)", and requires the assertion to check BOTH the
     returned vector's LENGTH and its VALUES against `predict_raw`'s
     truncated-to-first-column-length output — closing the exact zip-silently-
     truncates gap a weaker assertion would have left open. This is a genuine,
     not cosmetic, fix: an implementation using `n_objects =
     feature_values.iter().map(Vec::len).max()` would produce a length-4 output
     on case (b) against `predict_raw`'s length-2 output, and the
     required length-and-value assertion would catch that mismatch.
   Verdict: fully resolved, both in specification text and in the Red test's
   actual discriminating power.

2. **[MINOR pass-2 finding — stale SPEC §2 non-goals framing] — RESOLVED
   (verified).**
   SPEC.md §2 non-goals (lines 115-122) now reads: "**Bit-for-bit parity
   across backends/compilers.** The kernel's accumulation ASSOCIATION is
   locked to match the CPU oracle exactly (ascending per-tree order seeded at
   `0.0`, `bias` added once at the end ... §9 R1, corrected after PLAN-CHECK
   pass 1). What is NOT claimed exact is bit-identical output across
   DIFFERENT backends/hardware/compilers (rounding-mode, FMA-fusion, and
   codegen differences ...); parity there is to the `SCORE_BOUND` report
   convention ..., not exact." This precisely matches pass 2's required
   revision: it now distinguishes "accumulation ASSOCIATION is locked" from
   "cross-backend bit-identity is not claimed," removing the superseded
   "thread-local order, not order-locked" phrasing. Cross-checked against §9
   R1 (line 546, re-read verbatim) and TASK-03's kernel pseudocode/notes
   (PLAN.md lines 273-325, re-read verbatim) — both are consistent with this
   reworded non-goal; no remaining internal contradiction.

**Net result of the pass-2 revisions:** both findings genuinely resolved, with
verifiable, non-superficial fixes (a length-and-value-checking discriminating
test for finding 1; a reworded, now-internally-consistent non-goal for finding
2).

### New issue surfaced on this pass's full re-run

One new MINOR documentation-completeness gap was found (see Issues below); no
new MAJOR/CRITICAL/BLOCKER issue was found. All CodeGraph-verifiable structural
claims in the plan were re-confirmed against current on-disk source in this
pass (see CodeGraph Evidence below) and none have drifted or broken.

---

## Plan Check Result

**Verdict:** PASS
**Goal:** GINF-01 — a first-slice GPU inference/apply evaluator for float-only, scalar, oblivious catboost-rs models, with a `cpu`-backend (CubeCL `CpuRuntime`) numeric oracle against the shipped CPU `predict_raw`, additive to the existing crate architecture (`cb-model` stays backend-free; `cb-backend` stays independent of `cb-model`).
**Plan:** `.planning/plans/gpu-inference-evaluator/SPEC.md` + `.planning/plans/gpu-inference-evaluator/PLAN.md` (pass 3, post pass-2-revision)

### Summary
- Both pass-2 findings (the MAJOR ragged-column `n_objects` gap and the MINOR stale non-goals framing) were independently re-derived from on-disk source (`apply.rs:386-416`, re-read verbatim this pass) and confirmed genuinely resolved, not merely reworded: the new Red test's length-and-value assertion on a non-first-longer-column case is the specific test shape that distinguishes a correct "first-column-governs" implementation from an incorrect "max-over-columns" one.
- All four pass-1 findings (verified resolved in pass 2) remain resolved; nothing in the pass-2 revision touched those areas.
- A full from-scratch re-run of every load-bearing CodeGraph claim in the plan (Cargo.toml dependency edges, `is_onnx_exportable`'s guard order, `ObliviousTree`'s field shape, `launch_block_reduce_f64`'s launch shape, `apply_leaf_delta_kernel`'s per-object gather shape, `PackedCindex::device_arrays`'s overflow-guard shape, `CatBoostError::Train(#[from] cb_core::CbError)`, and `to_pyerr`'s exhaustive match) reconfirmed every citation exactly as claimed, with no drift since pass 2.
- One new MINOR issue was found on this pass's full re-run: SPEC §10 ("Traceability and sources"), which is explicitly the plan's line-range-cited source list, does not include `apply.rs:386-407` (`predict_raw_cat`, now load-bearing to the plan's own corrected S5 contract) even though the citation is correctly present and used everywhere it is normatively needed (§4 doc comment, TASK-05 Files/evidence). This is a completeness/consistency nit in a summary section, not a defect in the plan's substantive requirements, evidence, or tests — it does not block implementation and does not indicate any unverified structural claim.
- No unmitigated functional impact on existing dependents was found: `apply.rs`, `reduction.rs`, `kernels.rs`, `gpu_runtime/mod.rs`, `error.rs`, and the facade's existing `predict`/`predict_raw` remain read-only/additive-only under this plan, confirmed again this pass.

### Specification Coverage
- [x] GINF-01-S1 (guard) → TASK-01: re-verified against `is_onnx_exportable` (`crates/cb-model/src/export/onnx.rs:99-115`, deterministic non-symmetric → region → CTR order confirmed verbatim this pass) and `Model` fields (re-confirmed `ObliviousTree{splits: Vec<ModelSplit>, leaf_values: Vec<f64>, leaf_weights: Vec<f64>}` at `model.rs:254-264`).
- [x] GINF-01-S2 (flattener) → TASK-02: CSR-invariant assertion (tying offsets to `oblivious_trees[t].leaf_values.len()`) and round-trip oracle vs `predict_raw` unchanged from pass 2, still sound.
- [x] GINF-01-S3 (`#[cube]` kernel) → TASK-03: accumulation order (seed `0.0`, accumulate ascending `t`, `bias` added once after the loop) re-verified bit-exact-equivalent to `predict_raw_one`'s `model.bias + sum_f64(&oblivious)` association (`apply.rs:354`, `reduction.rs:32-38`, both re-read verbatim this pass).
- [x] GINF-01-S4 (launch helper) → TASK-04: `launch_block_reduce_f64` precedent re-confirmed verbatim this pass (`crates/cb-backend/src/gpu_runtime/mod.rs:220-260`) — `client.create`/`client.empty`/`CubeCount::Static`/`CubeDim`/`kernel::launch`/`client.read_one`→`CbError::Degenerate`/`bytemuck::cast_slice`, matching every plan citation.
- [x] GINF-01-S5 (facade) → TASK-05: the parity + reject tests are covered; the `n_objects`-derivation contract is now explicitly specified (SPEC §4, PLAN TASK-05 Files/evidence) and the ragged-column Red test is required to exercise the ONE discriminating case (non-first column longer than first) with a length-and-value assertion — closing pass 2's gap.
- [x] GINF-01-S6 (bench) → TASK-06: unchanged, non-gating, sufficient.

### CodeGraph Evidence
- `predict_raw_cat` (`crates/cb-model/src/apply.rs:386-416`) — re-read verbatim this pass (both via `Read` and via `codegraph_explore`, byte-identical). Lines 397-399: `let n_float = feature_values.first().map_or(0, Vec::len); let n_cat = cat_columns.first().map_or(0, Vec::len); let n_objects = n_float.max(n_cat);`. Lines 404-407: the per-cell `col.get(obj).copied().unwrap_or(f32::NAN)` gather. Both exactly match the SPEC/PLAN citations used to resolve pass 2's Issue #1.
- `predict_raw_one` (`crates/cb-model/src/apply.rs:318-355`) — re-read verbatim. Line 354: `model.bias + sum_f64(&oblivious) + sum_f64(&non_symmetric) + sum_f64(&region)`, reducing (for a guard-accepted oblivious-only model) to `model.bias + sum_f64(&oblivious)`. Confirms the kernel's locked association (TASK-03 pseudocode, PLAN.md lines 273-325) is correct and that SPEC §2's reworded non-goal (lines 115-122) is now accurate.
- `sum_f64` (`crates/cb-core/src/reduction.rs:32-38`) — re-read verbatim. Plain left-to-right fold seeded at `0.0`; unchanged since pass 1/2, still the structural match the kernel mirrors.
- `CbError` (`crates/cb-core/src/error.rs:17-105`) — re-read verbatim. `Unsupported(String)` at lines 86-92, `OutOfRange(String)` at lines 26-28, confirmed exactly as cited for the guard-reject vs overflow taxonomy.
- `PackedCindex::device_arrays` (`crates/cb-backend/src/gpu_runtime/cindex.rs:88-109`) — re-read verbatim via CodeGraph. `u32::try_from(f.offset).map_err(|_| CbError::OutOfRange(...))` confirms the overflow-guard precedent the flattener's `CbError::OutOfRange` design mirrors.
- `is_onnx_exportable` (`crates/cb-model/src/export/onnx.rs:99-115`) — re-read verbatim via CodeGraph. Confirmed deterministic check order non-symmetric → region → CTR (`model.ctr_data.is_some() || has_ctr_split`) → `Ok(())`, matching TASK-01's guard order exactly.
- `apply_leaf_delta_kernel` (`crates/cb-backend/src/kernels.rs:588-598`) — re-read verbatim via CodeGraph. `if ABSOLUTE_POS < approx.len() { let leaf = leaf_of[ABSOLUTE_POS] as usize; approx[ABSOLUTE_POS] += lr[0]*leaf_values[leaf]; }` confirms the per-object gather / bounds-guard shape TASK-03's kernel mirrors.
- `launch_block_reduce_f64` (`crates/cb-backend/src/gpu_runtime/mod.rs:220-260`) — re-read verbatim via CodeGraph. Confirms `client.create`/`client.empty`/`num_cubes = n.div_ceil(CUBE_DIM)`/`CubeCount::Static`/`CubeDim`/`kernel::launch::<f64, SelectedRuntime>`/`client.read_one`→`CbError::Degenerate`/`bytemuck::cast_slice`, matching TASK-04's citations exactly.
- `CatBoostError` (`crates/catboost-rs/src/error.rs:32-115`) — re-read in full this pass. `Train(#[from] cb_core::CbError)` confirmed at line 37; the enum is not `#[non_exhaustive]` but downstream `match`es are documented as expected to stay robust to new variants. Still the single conversion point TASK-05 needs.
- `to_pyerr` (`crates/catboost-rs-py/src/errors.rs:117-169`) — re-read in full this pass. Still an exhaustive `match` on `FacadeError` with no wildcard arm; `FacadeError::Train(c) => CatBoostError::new_err(c.to_string())` is generic over any `CbError` payload (`.to_string()` only, no sub-variant match) — confirms adding `CbError::Unsupported` to the taxonomy forces no py-side change. The uncommitted `regressor.rs`/`lib.rs` edits visible in git status are unrelated (predict/partial-dependence surfaces) and do not touch this arm.
- `ObliviousTree` (`crates/cb-model/src/model.rs:254-264`) — re-read verbatim. `{splits: Vec<ModelSplit>, leaf_values: Vec<f64>, leaf_weights: Vec<f64>}` confirmed, matching SPEC §3.2's reuse citation.
- Cargo.toml dependency edges — re-read in full this pass (not just grepped): `crates/cb-model/Cargo.toml` — `cb-train` is a NORMAL dependency (`default-features = false`, backend-passthrough only), `cb-backend` is a `[dev-dependencies]`-only entry (also `default-features = false`); `crates/catboost-rs/Cargo.toml` — `cb-model`, `cb-backend`, `cb-train` are all NORMAL dependencies (`default-features = false`, feature-passthrough); `crates/cb-backend/Cargo.toml` — owns `cubecl`/`bytemuck`, depends only on `cb-compute`/`cb-core`/`rayon`, no `cb-model` edge. All exactly match the plan's crate-placement decision (SPEC §3.1) and confirm no cycle is introduced.
- Impact assessment: no functional impact on existing dependents — `apply.rs`, `reduction.rs`, `kernels.rs`, `gpu_runtime/mod.rs`, `error.rs`, `errors.rs`, and the facade's existing `predict`/`predict_raw`/`predict_with` remain untouched/read-only oracles under this plan. `predict_raw`'s 44 existing callers (`partial_dependence.rs`, `fstr.rs`, `catboost-rs/src/model.rs`, plus 22+ test files, confirmed via CodeGraph blast-radius query) are unaffected by an additive `predict_raw_on_device` method.

### Issues

#### [MINOR] SPEC §10 "Traceability and sources" omits the now-load-bearing `predict_raw_cat` `n_objects`-derivation citation
- **Plan location:** SPEC.md §10, "CPU oracle / semantics" bullet (lists `apply.rs:{1-6, 136-140, 208-215, 318-355, 370}` but not `386-407`).
- **Requirement:** Internal consistency/completeness of the specification's own designated "sources" section, given the plan elsewhere (§4, §6 AT-S5c, PLAN.md TASK-05) now treats `apply.rs:397-399`/`404-407` as load-bearing evidence for a MAJOR-severity requirement fixed in this very revision.
- **Evidence:** SPEC.md §10 (lines ~572-574) enumerates specific `apply.rs` line ranges as "the" traceability/sources list, but does not include `386-407` even though §4's `predict_raw_on_device` doc comment (lines 322-330) and PLAN.md TASK-05's CodeGraph/Read evidence (lines 449-455) both correctly cite and rely on those exact lines.
- **Failure scenario:** Low risk: a future reader auditing "what does this plan depend on" via §10 alone (without reading §4/TASK-05 in full) could miss that the `n_objects`-derivation rule is a hard dependency on `predict_raw_cat`, and might, for example, refactor `predict_raw_cat`'s object-count logic without realizing GINF-01's facade contract mirrors it bit-for-bit.
- **Impact:** Documentation/maintainability only; the plan's own normative sections (§4, TASK-05) already carry the correct citation and would guide an implementer correctly regardless of §10's omission.
- **Required revision:** Add `386-407 predict_raw_cat (n_objects derivation + NaN-pad gather)` to SPEC §10's "CPU oracle / semantics" bullet, alongside the existing `apply.rs` line-range list.

### Implementation Order Review
1. TASK-01 (`crates/cb-model/**`) ∥ TASK-03 (`crates/cb-backend/**`) — disjoint crates, no write conflict. Unchanged and still valid.
2. TASK-02 (same file `gpu_apply.rs`, depends on TASK-01) ∥ TASK-04 (same file `gpu_runtime/mod.rs`, depends on TASK-03) — unchanged, still valid.
3. TASK-05 (`catboost-rs`) — correctly gated on TASK-02 and TASK-04; the ragged-column test and `n_objects`-derivation clarification (pass-2 fix) live entirely within this task and do not change its position in the order.
4. TASK-06 — correctly gated on TASK-05.
5. No intermediate broken-build state is introduced by the pass-2 revisions; both edits are scoped to SPEC prose and TASK-05's own Files/Red/evidence sections, none of which affect task sequencing or file ownership.

### Potential Bugs
- Ragged-column `n_objects` mismatch (max-over-columns vs first-column-only) — CLOSED this pass: the required Red test now specifically targets the one input shape (non-first column longer than first) that distinguishes the two candidate implementations, with an explicit length-and-value assertion.
- Residual backend FMA/rounding-mode divergence on non-`cpu` backends remains an accepted, documented, adequately mitigated risk (§9 R2/R3) — unchanged from pass 1/2.
- No new latent bug was found in this pass's adversarial review of the revised sections (the doc-comment wording, the AT-S5c wording, and the TASK-05 Red test wording are all mutually consistent and none introduce a new gap).

### Required Plan Revisions
1. SPEC §10: add `apply.rs:386-407` (`predict_raw_cat` — `n_objects` derivation + NaN-pad gather) to the "CPU oracle / semantics" traceability bullet, so the plan's designated sources list is complete relative to its own §4/TASK-05 citations. (MINOR, non-blocking.)

### Unverified Items
- Whether CubeCL's compiled float comparison (`v > b`) reproduces strict IEEE-754 NaN-is-unordered semantics identically across `cpu`/wgpu/cuda/rocm backends remains unverified against the CubeCL manual or test suite (unchanged from pass 2); the `cpu`-backend ragged-column test, once implemented, empirically confirms this for the `cpu` backend only — which is the plan's stated numeric gate (SPEC §9 R2/R3). Not a blocker; the plan already scopes numeric sign-off to `cpu`/ROCm and treats wgpu/cuda as compile-only.
- The exact `kernels::<child>` test-mount idiom for TASK-03's oracle test remains an acknowledged scaffolding detail to confirm at edit time (PLAN §4 unresolved blocker #1) — unchanged from pass 1/2, not a correctness blocker.
