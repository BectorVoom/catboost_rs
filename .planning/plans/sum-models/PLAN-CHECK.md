## Plan Check Result

**Verdict:** ISSUES_FOUND
**Goal:** Implement CatBoost `sum_models` (weighted model merge), first slice = float-only oblivious scale==1 models, ≤1e-5 oracle vs `catboost.sum_models`; acceptance = SM-01..SM-07 + §6 scenarios.
**Plan:** `.planning/plans/sum-models/PLAN.md` (spec `.planning/plans/sum-models/SPEC.md`)

### Summary
- The plan is structurally sound, TDD-complete, and its CodeGraph claims are ALL independently confirmed: `sum_f64` location/signature, `Model`/`ObliviousTree` fields, the `bias + Σ leaf` apply (no scale multiply), facade `from_canonical`/`as_canonical`/`CatBoostError::Model #[from]`, `load_cbm(&Path)`, fixture precedents, and the additive-`ModelError::Merge` claim (no exhaustive match on `ModelError` exists anywhere in the workspace).
- The merge math is correct for the stated scope.
- One MAJOR consistency gap: SPEC §2 promises a *typed error* for "non-default scale" models, but the canonical `Model` has NO scale field (`load_cbm` discards the wire `Scale`), so this case is UNDETECTABLE at the `sum_models` boundary and has no task — SM-04's "never a silent wrong merge" cannot be honored for it. Plus two MINOR verification hardening items.

### Specification Coverage
- [x] SM-01 single-model weight scaling (leaf+bias): TASK-02 Red `sum_models_single_scales_leaves` + empty-weights default. Confirmed apply = `model.bias + sum_f64(oblivious)+...` (apply.rs:354) ⇒ scaling all leaves by w and bias by w yields `w·rawpredict`.
- [x] SM-02 N-model tree concatenation: TASK-03 Red `sum_models_concats_trees`, end-to-end `predict_raw` assertion with zero bias to isolate.
- [x] SM-03 weighted bias sum via `sum_f64`: TASK-04 Red `sum_models_sums_bias` (bit-exact). `cb_core::sum_f64(&[f64])->f64` verified at reduction.rs:32.
- [x] SM-04 compatibility validation → `ModelError::Merge`: TASK-01 (empty) + TASK-05 (6 cases). NOTE: does not (and cannot) cover the non-default-scale non-goal — see MAJOR issue.
- [x] SM-05 facade `Model::sum_models`: TASK-06. `as_canonical()` (pub) + `from_canonical()` (pub(crate), reachable from the mounted in-crate test) + `CatBoostError::Model(#[from] cb_model::ModelError)` (error.rs:45) verified.
- [~] SM-06 Python surface: TASK-09, OPTIONAL/deferrable per SPEC. `estimator.rs::from_model` (56), `load_model_path` (296), `to_pyerr` `FacadeError::Model(m)=>…to_string()` (errors.rs:121) verified — no `to_pyerr` change needed. Deferral authorized by SPEC.
- [x] SM-07 oracle parity ≤1e-5: TASK-07 (frozen fixtures) + TASK-08 (integration test). `load_cbm(&Path)` (cbm.rs:817), `predict_raw(&Model,&[Vec<f32>])` (apply.rs:370), `cb_oracle::load_f64_vec` (exported), `cb-oracle` dev-dep in cb-model (Cargo.toml:53) verified.
- §6 scenarios 1-5 all map to tasks (TASK-08 w_1_1 / w_03_07, TASK-02 w=2.0, TASK-05 ctr + border).

### CodeGraph Evidence
- `sum_f64` in `crates/cb-core/src/reduction.rs:32` — `pub fn sum_f64(values:&[f64])->f64`, sequential fold (D-08). 130 callers. Matches plan exactly.
- `Model` in `crates/cb-model/src/model.rs:272` — fields `oblivious_trees, non_symmetric_trees, region_trees, bias:f64, float_feature_borders:Vec<Vec<f64>>, ctr_data:Option<CtrData>, approx_dimension:usize, class_to_label:Vec<f64>`. No scale field. Matches plan.
- `ObliviousTree` in `crates/cb-model/src/model.rs:254` — `splits:Vec<ModelSplit>, leaf_values, leaf_weights`. Matches.
- `predict_raw_one` in `crates/cb-model/src/apply.rs:318` — returns `model.bias + sum_f64(&oblivious) + sum_f64(&non_symmetric) + sum_f64(&region)`; NO scale multiply. Confirms merge math.
- `ModelError` in `crates/cb-model/src/error.rs:18` — variants Deserialize/SchemaVersion/Serialize/Json/Core/Io; NOT `#[non_exhaustive]`. Workspace grep for any `ModelError::<variant> =>` arm ⇒ NONE. Impact: adding `Merge(String)` is fully additive, zero downstream match-arm breakage. Plan claim CONFIRMED independently.
- `CatBoostError::Model(#[from] cb_model::ModelError)` in `crates/catboost-rs/src/error.rs:45` — confirmed; `?` in the facade converts automatically. No new facade variant needed.
- `from_canonical` (pub(crate), model.rs:38) / `as_canonical` (pub, model.rs:45) — both real; `Model{inner:cb_model::Model}`.
- `load_cbm(path:&Path)->Result<Model,ModelError>` (cbm.rs:817). Fixtures: `inputs/numeric_tiny/{X,y}.npy`, `fstr_loss_change/gen_fixtures.py`, `model_serde/regression/*` all present. No pre-existing `sum_models` symbol.

### Issues

#### [MAJOR] SPEC over-promises a typed error for non-default-scale models that is undetectable and untasked
- **Plan location:** SPEC §2 (Non-goals: "models with a non-default scale … Each is a **typed error**, never a silent wrong merge (SM-04)") vs SM-04 §5 / TASK-05 (which enumerate 6 rejection cases, none for scale).
- **Requirement:** SM-04 invariant "never a silent wrong merge."
- **Evidence:** Canonical `Model` (model.rs:272) has NO scale field; `decode_cbm`/`reconstruct_model` read `Bias`/`MultiBias` (`read_bias`, cbm.rs:1132) but the wire `Scale` (present in TModelTrees, model_generated.rs:3183) is discarded. `sum_models(&[&Model], …)` operates on already-decoded `Model`s, so scale is unrecoverable at the merge boundary — it cannot emit a typed error for a non-unit-scale input.
- **Failure scenario:** A model whose upstream `Scale != 1` is fed to `sum_models`. No validation can fire (nothing to inspect); the merge proceeds as if scale==1 and silently produces wrong leaf contributions — exactly the outcome SPEC §2 says is forbidden. (Note: single-model `predict_raw` is already wrong for such a model, so this is an inherited limitation, not newly introduced.)
- **Impact:** SM-04 acceptance guarantee is unmet for one enumerated non-goal; a traceability defect (a stated typed-error case with no task and no test).
- **Required revision:** Reconcile the SPEC — remove non-default-scale from the "typed error" list in §2 and instead document it (as R1/A1 already begins to) as an UN-CHECKABLE assumption of the first slice: inputs are workspace-produced scale==1 models, the merge cannot detect a baked non-unit scale, and the oracle (TASK-08) is the only backstop. Ensure TASK-05/SM-04 explicitly state scale is NOT among the guarded cases.

#### [MINOR] Oracle test cannot isolate a merge defect from an apply defect
- **Plan location:** TASK-07 / TASK-08.
- **Evidence:** Only `expected_w_1_1.npy` / `expected_w_03_07.npy` (summed-model predictions) are frozen. TASK-08 asserts merged==expected but freezes no per-model expected.
- **Failure scenario:** If the oracle fails, it cannot tell whether the defect is in the merge arithmetic (TASK-02/03/04/05) or in the apply path on these specific new fixtures. TASK-08's claim "the defect is in the core merge" is unproven.
- **Required revision:** Also freeze `expected_m0.npy`/`expected_m1.npy` (each model's own `.predict(X)`) and add a per-model `predict_raw(m_i) ≈ expected_m_i` sanity assertion in TASK-08 so a merge defect is isolated from an apply defect.

#### [MINOR] Fixture prediction-type not pinned to RawFormulaVal
- **Plan location:** TASK-07 generation recipe.
- **Evidence:** Rust compares via `predict_raw` (RawFormulaVal). The generator computes `cb.sum_models([m0,m1],w).predict(X)`. For a regression loss the CatBoost default IS RawFormulaVal, so parity holds — but the recipe does not pin it.
- **Required revision:** In `gen_fixtures.py`, generate expecteds with an explicit `prediction_type="RawFormulaVal"` (and record it in `config.json`) so the fixture cannot silently drift to a transformed output that would break the ≤1e-5 gate against `predict_raw`.

### Implementation Order Review
1. Order is valid and acyclic: `TASK-01 → 02 → (03,04,05 sequential, shared `model_sum.rs`) → 06 → 08`; `TASK-07` fully parallel (Python + new fixtures dir, no Rust-file conflict); `TASK-09` optional after 06.
2. Write-conflict handling is correct: 03/04/05 all edit `model_sum.rs`+`model_sum_test.rs` and are sequenced, not parallelized, despite sharing only TASK-02 as prerequisite.
3. Intermediate states build: TASK-01 leaves a compiling placeholder (`models.first().ok_or_else(...)` clone) with the empty-reject; each later task is Green-additive. No invalid intermediate state.
4. Minor note: TASK-01 edits `crates/cb-model/src/lib.rs`, which is already uncommitted-modified on this branch (git status). The edit is an additive `mod`/`pub use` line — low conflict risk; SPEC §7's "no edit to in-flight files" wording is slightly imprecise (lib.rs is in-flight) but non-blocking.

### Potential Bugs
- Non-unit-scale silent mis-merge (see MAJOR) — mitigated only by input-scope restriction + oracle.
- `weights` non-empty with `weights.len() != models.len()`: guarded by TASK-05 BEFORE the merge; the pre-TASK-05 `iter().zip(weights)` truncates rather than panics, so no invalid intermediate. OK.
- `weights` empty ⇒ all-ones (Q1/A4): handled in TASK-02; `Some(&[])` passed through facade also resolves to all-ones. OK.
- Multi-dim leaf scaling: `leaf_values` is dimension-major flat; scaling every element by `w` is correct for any `approx_dimension` (fixtures are scalar dim==1 anyway). OK.
- No NaN/Inf weight guard — not required by SPEC; acceptable.

### Required Plan Revisions
1. Reconcile SPEC §2 with SM-04/R1: state that non-default-scale is an un-checkable first-slice assumption (no scale field to inspect), NOT a typed-error case; keep the oracle as the only backstop. (MAJOR)
2. Freeze per-model `expected_m0.npy`/`expected_m1.npy` and add per-model apply-sanity assertions in TASK-08 to isolate merge defects. (MINOR)
3. Pin `prediction_type="RawFormulaVal"` in `gen_fixtures.py` and record it in `config.json`. (MINOR)

### Unverified Items
- Upstream `catboost.sum_models` exact signature/defaults (`ctr_merge_policy` default) — deferred by the plan to TASK-07 generation time against installed `catboost==1.2.10` (R2/A2). Not resolvable in this session; float-only models make `ctr_merge_policy` irrelevant. Acceptable as a plan-time deferral, not a structural risk.

---

## Plan Check Result — PASS 2 (re-review of revised SPEC/PLAN)

**Verdict:** PASS
**Goal:** Implement CatBoost `sum_models` (weighted model merge), first slice = float-only oblivious scale==1 models, ≤1e-5 oracle vs `catboost.sum_models`; acceptance = SM-01..SM-07 + §6 scenarios.
**Plan:** `.planning/plans/sum-models/PLAN.md` (spec `.planning/plans/sum-models/SPEC.md`)

### Summary
- All three PASS-1 issues (1 MAJOR + 2 MINOR) are resolved by the revision, and no regression was introduced. Every structural claim was re-verified via CodeGraph MCP this pass.

### Resolution of prior issues

- **[MAJOR — RESOLVED] Non-default scale over-promise.** SPEC §2 no longer lists non-default scale as a typed `ModelError::Merge`. The typed-error list is now exactly {CTR/categorical, non-symmetric/Region, differing float borders/feature counts, differing approx_dimension, differing class_to_label, text/embedding} — all *checkable* fields. A dedicated §2 paragraph ("Non-default scale — an UNCHECKABLE assumption, NOT a typed error") documents that the canonical `Model` has no scale field and the case is undetectable at the merge boundary, with the SM-07 oracle as the only backstop. PLAN §5-A1 mirrors this and states explicitly that NO task asserts a scale typed-error; SM-04 §5 / TASK-05 enumerate six rejection cases with no scale arm; SPEC §4 `# Errors` doc lists the same six with no scale. Fully consistent.
  - CodeGraph re-verification: `Model` (crates/cb-model/src/model.rs:272) has no scale field; `predict_raw_one` (crates/cb-model/src/apply.rs:318-354) = `model.bias + sum_f64(&oblivious) + sum_f64(&non_symmetric) + sum_f64(&region)` — no scale multiply. The only `Scale()` in the graph is `TModelCtr::Scale` (ctr_data_generated.rs:1999), a per-CTR wire field, not a model-level scale. Confirms scale is genuinely unrecoverable post-decode.

- **[MINOR 1 — RESOLVED] Merge-vs-apply isolation.** TASK-07 now freezes `expected_m0.npy` and `expected_m1.npy` (each model's own `predict(X, RawFormulaVal)`). TASK-08 adds `sum_models_oracle_inputs_apply`, which runs FIRST and asserts `predict_raw(m0)≈expected_m0` and `predict_raw(m1)≈expected_m1` (≤1e-5) before any summed assertion — so a summed failure with green per-model sanity provably isolates the defect to `sum_models` arithmetic. Sound isolation.

- **[MINOR 2 — RESOLVED] Prediction-type pin.** TASK-07 adds a MANDATORY "Prediction-type pin" clause: every expected `.npy` (per-model AND summed) is generated with explicit `prediction_type="RawFormulaVal"` and recorded in `config.json`; SPEC SM-07 states the same. Matches the Rust `predict_raw` comparison path.

### Regression check (nothing new broke)
- **Additive `ModelError::Merge`:** re-confirmed. `ModelError` (error.rs:18) is not `#[non_exhaustive]`; grep for any `ModelError::<variant> =>` match arm returns NOTHING; `to_pyerr` (catboost-rs-py/src/errors.rs:113-135) matches `FacadeError` variants and handles `FacadeError::Model(m)` opaquely via `m.to_string()`; `CatBoostError::Model(#[from] cb_model::ModelError)` (catboost-rs/src/error.rs:45) converts via `?`. Adding `Merge(String)` requires zero downstream match-arm edits.
- **Merge math:** unchanged and correct — apply is `bias + Σ leaf` (no scale), so scaling every leaf by `w_i` and each bias by `w_i`, concatenating trees, and summing biases via `cb_core::sum_f64` (reduction.rs:32) yields `Σ_i w_i·rawpredict(m_i)`.
- **Task ordering:** unchanged and valid/acyclic — `01→02→(03,04,05 sequential on model_sum.rs)→06→08`; TASK-07 fully parallel; TASK-09 optional after 06. Revision only added fixture files + one sanity test; no new inter-task edge.
- The revision touched only SPEC prose (scale wording), fixture artifacts (`expected_m0/m1.npy`, prediction-type pin in `gen_fixtures.py`/`config.json`), and one added test — all additive; no new production-symbol claim to break.

### Unverified items (unchanged, non-blocking)
- Upstream `catboost.sum_models` exact signature/defaults (`ctr_merge_policy` default) — deferred to TASK-07 generation time against installed `catboost==1.2.10` (R2/A2). Float-only models make `ctr_merge_policy` irrelevant; not a structural risk.

**Final verdict: PASS.**
