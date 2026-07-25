---
title: "ORCH-03 — Plan Check: Training snapshot / resume (slice 1) — EXTRA VERIFICATION (post-cap, pass 5)"
status: reviewed
verdict: PASS
reviewed_artifacts:
  - .planning/plans/snapshot-resume/SPEC.md
  - .planning/plans/snapshot-resume/PLAN.md
pass: 5
prior_pass_report: "pass 4 verdict was ISSUES_FOUND (CRITICAL-6: TASK-04's Cargo dependency fix was specified but its mandatory non-test `cargo build -p cb-train` validation step was never actually added to TASK-04's Validation list); this pass verifies that fix and re-runs the full checker process from scratch a fifth time"
---

## Verification pass 5

This is a FIFTH, explicitly user-requested review pass, continuing beyond the
spec-tdd-planner-skill's normal 3-pass cap, specifically to verify pass 4's
CRITICAL-6 finding was genuinely fixed (not just described as fixed) and to
re-run the complete checker process from scratch once more, including a fifth
independent, from-scratch re-enumeration of every `Loss` variant against the
fingerprint's LOSS PAYLOAD match block.

**CRITICAL-6 verification result: GENUINELY FIXED.** Direct `Read` of
`PLAN.md` TASK-04 confirms its "Validation:" bullet list (lines 358-375) now
opens with:

```
- **Validation:**
  - `cargo build -p cb-train` **(MANDATORY, run FIRST — added at the post-cap
    extra verification pass, pass 4, closing CRITICAL-6: ... `cargo test -p
    cb-train` CANNOT detect `serde`/`serde_json` being left under
    `[dev-dependencies]` instead of `[dependencies]`, since dev-dependencies
    ARE visible when compiling a crate's own test targets ...)**
  - `cargo test -p cb-train --lib snapshot`
  - `cargo clippy -p cb-train --lib --no-deps` (no `unwrap`/`panic` on encode/decode)
  - `cargo build -p cb-model` (downstream sanity: `cb-model` depends on
    `cb-train` normally — confirms the newly-normal `serde`/`serde_json` deps
    don't break the downstream build either)
```

This is a concrete, executable Validation bullet (not prose), it is
correctly ordered (plain build BEFORE `cargo test`/clippy, exactly as
required — a green `cargo build -p cb-train` proves the crate's LIBRARY
target, which does NOT see `[dev-dependencies]`, compiles with the new
`serde`/`serde_json` entries wherever they were actually placed), and its
justification is technically accurate (dev-dependencies genuinely ARE linked
into a crate's own test-target compilation, which is exactly why `cargo test`
alone could not have caught a misplaced entry — independently confirmed by
reading both `crates/cb-train/Cargo.toml` and `crates/cb-model/Cargo.toml` in
full this pass, byte-for-byte matching every citation in TASK-04/SPEC §1/§3).
The added `cargo build -p cb-model` downstream-sanity bullet is a reasonable
belt-and-suspenders check (`cb-model` is confirmed to depend on `cb-train`
normally, `Cargo.toml:26`).

**Non-blocking residual observation (carried, not a defect):** the fix was
applied ONLY to TASK-04 (where the Cargo.toml edit actually happens); pass
4's Required Revision #1 additionally suggested repeating an equivalent
build-only check through TASK-05..08. That repetition was NOT added. On
independent analysis (see "Potential Bugs" below) this is NOT a live gap:
`crates/cb-train/Cargo.toml` is edited ONLY in TASK-04 in this plan — none of
TASK-05/06/07/08 touch the manifest again — so the one CRITICAL-6 defect
mechanism (a misplaced dependency-table entry escaping `cargo test`) is fully
retired at the single point where the risky edit occurs. Repeating the plain
build in later tasks would be extra defense-in-depth, not a fix for a
currently-open path to the same failure. Downgraded to a MINOR
recommendation, not a blocker.

**Result: full re-run of the checker process finds no new blocking issue.**
Every structural claim in the SPEC/PLAN — the full 28-variant `Loss`
re-enumeration (fifth independent pass), `loss_approx_dimension` /
`is_grouped_loss`'s exact match sets, `Loss::validate`'s `MultiQuantile`
empty-only rejection, `draws_active`'s RNG-inertness gate, both
`cb-train`/`cb-model` `Cargo.toml` files, `train_inner`'s signature / return
type / exactly-3 call sites, `CustomObjectiveHandle`/`CustomMetricHandle`'s
`Arc::ptr_eq`-only equality, `greedy_tensor_search_oblivious_perturbed`'s
signature (no `min_data_in_leaf`), `BoostParams.loss`/`.eval_metric` field
types, the `metrics.rs` test-mount idiom, and the CPU backend's inert
`begin_device_training` default — was independently re-derived from the live
source this pass (not accepted from any prior pass's prose) and matches
every citation byte-for-byte. **Verdict: PASS.**

---

## Full Plan Check Result (pass 5, re-run from scratch)

**Verdict:** PASS
**Goal:** ORCH-03 slice 1 — a narrow, provably-resumable training snapshot/resume
surface (float-only, `Plain`, non-ranking, no-CTR, `bootstrap_type=No`,
`random_strength=0`, single-dimension) for `catboost-rs`'s training loop, with a
GATING VERIFICATION SPIKE (TASK-01) preceding any implementation.
**Plan:** `.planning/plans/snapshot-resume/{SPEC,PLAN}.md`

### Summary

Independent re-derivation (direct `Read` + `codegraph_explore`, not trusting
the plan's own citations) confirms: (1) pass 4's CRITICAL-6 fix — a
MANDATORY, correctly-ordered, technically-justified `cargo build -p
cb-train` Validation bullet in TASK-04, plus a `cargo build -p cb-model`
downstream sanity check — is genuinely present as an executable bullet, not
prose; (2) the `crates/cb-train/Cargo.toml` / `crates/cb-model/Cargo.toml`
fix instructions remain byte-accurate against the live files; (3) the
`Loss::MultiQuantile{alpha,delta}` fingerprint-payload gap (CRITICAL-4) stays
correctly closed; (4) a fifth, from-scratch, complete 28-variant
re-enumeration of `Loss` (`crates/cb-compute/src/runtime.rs:58-495`, read
start to finish) finds NO new variant and NO other single-dimension,
non-grouped parametric-payload gap; (5) RNG-inertness, the
`cb-model`↔`cb-train` dependency direction, the S6 `refingerprinted_k_run`
mechanism, and TASK-08's peek-based `resume_from` design are all
independently re-confirmed sound and unchanged.

### Specification Coverage

- [x] AT-S1 (additive param, D-04 anchor): TASK-01 + TASK-03 — `train_inner`
  private, signature `crates/cb-train/src/boosting.rs:2259-2271` returns
  `CbResult<(Model, BakedCtrData)>` (re-read directly), exactly 3 call sites
  confirmed this pass at `:2062` (`train_with_eval_sets`), `:2102`
  (`train_ranking`), `:2155` (`train_cat`) — matching `codegraph_explore`'s
  independent blast-radius count ("3 callers in crates/cb-train/src/boosting.rs").
- [x] AT-S2 (`from_raw_state` bit-exact round-trip): TASK-02 — re-read
  `crates/cb-core/src/rng.rs:137-223` in full this pass: `raw_state()` at
  `:221` returns `[r1.x, r1.c, r2.x, r2.c]`; `call_count()` at `:204`;
  `from_seed` at `:171`; `new` at `:158`; no restore ctor exists yet
  (plan-only). Design coherent.
- [x] AT-S3 (`decode(encode(s))==s`, no serde on `Model`/`ObliviousTree`): the
  DTO design is buildable, AND (this pass) the underlying Cargo-fix build
  gate is now genuinely present and correctly ordered in TASK-04 — see
  CodeGraph Evidence and the pass-5 preface above. No open item.
- [x] AT-S4 (fingerprint: any hyperparameter change ⇒ different fingerprint;
  mismatch ⇒ `Err`, never a silently wrong model): all 6 CRITICAL-1 fields,
  the 6 CRITICAL-3 parametric-loss variants, and the CRITICAL-4
  `MultiQuantile` payload are correctly, completely handled per a fifth
  independent, from-scratch 28-variant re-enumeration (see CodeGraph
  Evidence).
- [x] AT-S5a/b/c (per-tree write + scope guard incl. `Custom`): TASK-06 —
  anchored to real gate expressions (`is_grouped_loss`
  `boosting.rs:989-1001`, `loss_approx_dimension` `:1074-1096`, both re-read
  verbatim this pass); the two `Custom` predicates are syntactically sound
  given `BoostParams.loss: Loss` (`:167`, re-read) and `.eval_metric:
  Option<EvalMetric>` (`:225`, re-read).
- [x] AT-S6 (self-consistency, byte-identical): the `refingerprinted_k_run`
  mechanism (TASK-07) re-confirmed valid; `auto_learning_rate=false` ⇒
  `params.iterations` provably inert on the scoped first-K-trees path (the
  mechanism does not touch `train_inner`'s call sites or return type).
- [x] AT-S7 (facade `train_with_snapshot`): TASK-08 — peek-based design
  explicitly locked in the plan text (`train_with_snapshot` decodes
  `completed_iters` itself, before delegating, and does NOT grow
  `train_inner`'s return type or touch its 3 call sites); re-confirmed this
  pass against the live `train_inner` signature/return type/call sites.

### CodeGraph Evidence

- `Loss` enum, **full 28-variant re-enumeration, independently re-derived a
  FIFTH time this pass** — `crates/cb-compute/src/runtime.rs:58-495` (read
  start-to-finish directly via `Read`, not from any prior list): `Rmse`,
  `Logloss`, `CrossEntropy`, `Focal{alpha,gamma}`, `Mae`,
  `Quantile{alpha,delta}`, `LogCosh`, `Lq{q}`, `Huber{delta}`,
  `Expectile{alpha}`, `Poisson`, `Tweedie{variance_power}`, `Mape`,
  `MultiClass`, `MultiClassOneVsAll`, `MultiLogloss`, `MultiCrossEntropy`,
  `RmseWithUncertainty`, `MultiQuantile{alpha: Vec<f64>, delta}`,
  `QueryRmse`, `QuerySoftMax{lambda,beta}`, `PairLogit`,
  `PairLogitPairwise`, `LambdaMart{metric,sigma,top,norm}`,
  `YetiRank{permutations,decay}`, `YetiRankPairwise{permutations,decay}`,
  `StochasticRank{metric,sigma,mu,num_estimations}`,
  `Custom(CustomObjectiveHandle)`. IDENTICAL to passes 2/3/4's lists — no
  variant added since pass 4, no additional gap found. 7 non-parametric
  (discriminant-only, correct) + 6 parametric single-dim (payload hashed) +
  1 variable-dim with a dedicated always-hashing arm (`MultiQuantile`) + 5
  always-multi-dim (excluded by `approx_dimension != 1`) + 8 grouped/ranking
  (excluded by `is_grouped_loss`) + 1 `Custom` (excluded by its own explicit
  predicate) = 28. ✓
- `loss_approx_dimension` — `crates/cb-train/src/boosting.rs:1074-1096` —
  re-read this pass: `MultiClass`/`MultiClassOneVsAll` ⇒
  `build_class_remap(target).len().max(2)`; `MultiQuantile{alpha,..}` ⇒
  `alpha.len()`; `RmseWithUncertainty` ⇒ `2`; default arm `_ => 1`.
  Byte-identical to every prior pass's citation.
- `is_grouped_loss` — `crates/cb-train/src/boosting.rs:989-1001` — re-read,
  exact 8-member match (`QueryRmse | QuerySoftMax{..} | PairLogit |
  PairLogitPairwise | LambdaMart{..} | YetiRank{..} | YetiRankPairwise{..} |
  StochasticRank{..}`); `MultiQuantile` absent, confirming it is not
  excluded by the grouped-loss guard.
- `Loss::validate`, `MultiQuantile` arm — `crates/cb-compute/src/runtime.rs:
  665-683` — re-read directly this pass: rejects only an EMPTY `alpha`; a
  single-element `alpha` is legal, confirming `MultiQuantile{alpha:[a],
  delta:d}` at length 1 is real, constructible, in-scope.
- `crates/cb-train/Cargo.toml` (full 51-line file re-read this pass) —
  `[dependencies]` lines 21-31 (`cb-core`/`cb-data`/`cb-compute`/
  `cb-backend`/`rayon` — still no `serde` entry); `[dev-dependencies]` lines
  33-51, `serde_json.workspace = true` still at line 51 (expected — plan-only
  artifact, no production code authored yet, SPEC §11). `crates/cb-model/
  Cargo.toml` (full file re-read) confirms `serde.workspace = true` /
  `serde_json.workspace = true` at lines 37-38 under `[dependencies]` — the
  exact precedent TASK-04 cites. **TASK-04's fix instruction remains
  byte-accurate and complete against the live files.**
- **`cargo build -p cb-train` occurrence audit in `PLAN.md`, re-run this
  pass** — TASK-04's "Validation:" bullet list (`PLAN.md:358-375`, re-read
  verbatim) now begins with `cargo build -p cb-train` marked MANDATORY /
  run FIRST, with the exact justification text quoted in the pass-5 preface
  above; followed by `cargo test -p cb-train --lib snapshot`, `cargo clippy
  -p cb-train --lib --no-deps`, and `cargo build -p cb-model`. This closes
  CRITICAL-6 as specified. TASK-05 (`:525-527`), TASK-06 (`:624-627`),
  TASK-07 (`:723-726`), TASK-08 (`:785-788`) Validation lists still do not
  repeat a plain build — see "Potential Bugs" for why this is non-blocking.
- `train_inner` — `crates/cb-train/src/boosting.rs:2259-2271` (re-read) —
  signature confirmed, no `snapshot` param yet (plan-only); return type
  `CbResult<(Model, BakedCtrData)>` at `:2271`. Exactly 3 internal callers,
  re-confirmed via direct `Read` at `:2062` (`train_with_eval_sets`), `:2102`
  (`train_ranking`), `:2155` (`train_cat`) — matching `codegraph_explore`'s
  independently-computed blast radius ("3 callers ... boosting.rs").
- `draws_active` — `crates/cb-train/src/boosting.rs:3018` (re-read) —
  `!matches!(params.bootstrap_type, EBootstrapType::No) || perturb_active`;
  the RNG-inertness claim for `bootstrap_type=No, random_strength=0` holds.
- Dependency direction — re-read BOTH manifests directly this pass:
  `crates/cb-train/Cargo.toml:45` lists `cb-model` under `[dev-dependencies]`
  ONLY; `crates/cb-model/Cargo.toml:26` lists `cb-train = { path =
  "../cb-train", ... }` under `[dependencies]` (normal). Confirms the
  `cb-model → cb-train` build edge; `cb-train`'s only reference to `cb-model`
  is dev-only.
- `greedy_tensor_search_oblivious_perturbed` — `crates/cb-train/src/tree.rs:
  574-584` (re-read directly) — signature confirmed to have no
  `min_data_in_leaf` parameter; the field-10 fingerprint citation correction
  remains accurate.
- `metrics.rs:713-715` (re-read) — `#[cfg(test)] #[path = "metrics_test.rs"]
  mod tests;` confirmed exact.
- `EvalMetric::Custom(CustomMetricHandle)` — `crates/cb-train/src/metrics.rs:
  167` (re-read) — confirmed exact. `BoostParams.loss: Loss` — `boosting.rs:
  167` (re-read) — confirmed. `BoostParams.eval_metric: Option<EvalMetric>` —
  `boosting.rs:225` (re-read) — confirmed exact.
- `CustomObjectiveHandle`/`CustomMetricHandle` `Arc::ptr_eq` equality —
  `crates/cb-compute/src/custom.rs:117-169` (re-read in full this pass) —
  confirmed: pointer-identity-only `PartialEq` for both handles (`:134-142`,
  `:165-169`), no `Serialize`/`Deserialize` derive anywhere in the file.
  `Loss::Custom(CustomObjectiveHandle)` at `runtime.rs:494` (re-read)
  confirmed exact.
- **`begin_device_training` override audit (new this pass)** —
  `grep -rn "begin_device_training" crates/cb-backend/src/` returns
  occurrences ONLY in `gpu_backend.rs`/`gpu_backend_test.rs`; `cpu_runtime.rs`
  has NO override, so the default `CpuBackend`/`CpuRuntime` inherits the
  trait's default `Ok(false)` body (`cb-compute/src/runtime.rs:1230-1262`,
  re-read). Independently confirms SPEC §9 Open Question 4 / TASK-01 item 3:
  `device_active` (`boosting.rs:3158`, re-read: `device_host_eligible &&
  device_n_bins > 0` gates a call into `begin_device_training`, which is
  `false` on the default backend) is provably `false` on the scoped CPU
  path, so the scope guard's device exclusion is defense-in-depth, not the
  primary gate.

### Issues

None outstanding at BLOCKER/CRITICAL/MAJOR severity. See "Potential Bugs"
for two carried-forward MINOR, non-blocking observations.

### Implementation Order Review

1. TASK-01 (gating spike, read-only) ∥ TASK-02 (`cb-core` rng ctor) —
   confirmed independent, no conflict.
2. TASK-03 (Cargo.toml scaffold does NOT yet need serde) → TASK-04 (Cargo
   dependency fix + the now-present `cargo build -p cb-train` MANDATORY
   validation step, run FIRST, before any Red/Green test for TASK-04 is
   trusted) → TASK-05 → TASK-06 → TASK-07 → TASK-08 correctly drawn as a
   single sequential chain (all touch `lib.rs`/`snapshot.rs`/`boosting.rs`).
3. TASK-01's GO/NO-GO gate remains a genuine blocking prerequisite.
4. TASK-05 (fingerprint) correctly lands both the CRITICAL-3 fix (six named
   parametric variants) and the CRITICAL-4 fix (`MultiQuantile` payload)
   before TASK-06's scope guard is finalized.
5. TASK-07 (resume, keystone oracle) correctly sequenced after TASK-06/02;
   its `refingerprinted_k_run` mechanism independently reconfirmed sound.
6. The CRITICAL-6 validation gate now correctly precedes TASK-04's own
   Green/Refactor trust boundary: a green `cargo test` at TASK-04 no longer
   gives false confidence, because the plain `cargo build -p cb-train` (which
   the crate's own dependency-table placement can fail) runs first and is
   marked MANDATORY.

### Potential Bugs

- **(Resolved this pass) Production build failure hidden behind a green test
  run, via a Cargo section-placement slip during TASK-04** — CLOSED. The
  MANDATORY, correctly-ordered `cargo build -p cb-train` Validation bullet in
  TASK-04 now catches this before Green is trusted; a downstream `cargo
  build -p cb-model` check adds a second independent proof point.
- **(Analyzed, non-blocking) TASK-05..08 do not repeat a plain, non-test
  `cargo build -p cb-train`** — trigger considered: could a later task
  (TASK-05 fingerprint, TASK-06 write hook + scope guard, TASK-07 resume,
  TASK-08 facade) reintroduce the SAME "cargo test passes, cargo build
  fails" class of defect? Analysis: NONE of TASK-05/06/07/08 touch
  `crates/cb-train/Cargo.toml` again per the plan text — they only add
  functions/structs to already-`[dependencies]`-fixed `snapshot.rs` /
  `boosting.rs`. The ONE point where the Cargo-manifest section-placement
  mistake could occur (TASK-04) is now gated by the plain build BEFORE any
  Green step is trusted; once that gate passes, the dependency is proven
  correctly placed for the remainder of the plan, since no further Cargo.toml
  edit is planned. Repeating the check downstream would be extra
  defense-in-depth but closes no currently-open path to the CRITICAL-6
  failure mode. Recommended as a MINOR enhancement, not required for a PASS.
- **(Carried, unresolved, MINOR) Imprecise "conflict" framing in TASK-04's
  Cargo fix instruction** — TASK-04 still reads "promoting `serde_json` out
  of `[dev-dependencies]` if the existing dev-only entry would otherwise
  conflict — check at edit time" (`PLAN.md:320-321`, re-read this pass,
  unchanged from pass 4). Cargo permits the same crate in both
  `[dependencies]` and `[dev-dependencies]` simultaneously (no TOML key
  conflict); the wording implies a conditional outcome where none exists.
  Low risk (clarity only, not a functional risk) — carried forward as a
  MINOR, non-blocking wording fix.
- **Content-hash fingerprint cannot distinguish genuinely-different
  `feature_values` that coincidentally quantize to identical
  `feature_borders`** — pre-existing, documented, accepted residual design
  trade-off (SPEC AT-S4 frames it as "high probability", not a guarantee);
  not blocking, carried forward unchanged since pass 2/3.

### Required Plan Revisions

1. **(Minor, optional, not blocking)** Reword TASK-04's "if the existing
   dev-only entry would otherwise conflict — check at edit time" hedge to
   state plainly that no TOML conflict is possible; a lingering
   `[dev-dependencies]` duplicate is legal and only a cleanliness concern.
2. **(Minor, optional, not blocking)** Consider repeating a plain `cargo
   build -p cb-train` (or an equivalent standing check, e.g. once as a
   repo-wide CI-equivalent step) through TASK-05..08's Validation lists as
   pure defense-in-depth, even though no currently-open path to CRITICAL-6's
   failure mode remains once TASK-04's gate is in place.
3. (Already satisfied, no action) CRITICAL-6's core requirement — a
   MANDATORY, correctly-ordered, technically-justified `cargo build -p
   cb-train` Validation bullet in TASK-04 — independently re-verified
   genuinely present this pass (not merely described/mentioned in prose).
4. (Already satisfied, no action) CRITICAL-4's `MultiQuantile` fingerprint
   payload fix; CRITICAL-5's Cargo.toml edit instruction; CRITICAL-3's six
   named parametric variants + `Loss::Custom`/`EvalMetric::Custom`
   scope-guard exclusion; the `min_data_in_leaf` citation correction;
   TASK-08's peek-based `resume_from` design; the `refingerprinted_k_run` S6
   mechanism; the `metrics.rs` test-mount citation — all independently
   re-confirmed intact and unchanged this pass, a fifth time.

### Unverified Items

None outstanding for the scope of this pass. Every claim re-verified a fifth
time — the complete 28-variant `Loss` enum re-enumeration, each variant's
dimension/grouped classification, the `MultiQuantile`/`Loss::validate`
bounds, both `cb-train`/`cb-model` `Cargo.toml` files in full, the
`min_data_in_leaf` and `metrics.rs` citations, the `draws_active` gate, the
dependency-direction claim, `train_inner`'s signature/return-type/call-sites,
the `CustomObjectiveHandle`/`CustomMetricHandle` `Arc::ptr_eq`-only equality,
the CPU backend's inert `begin_device_training` default (new this pass), and
a document-level re-audit of TASK-04's Validation-list ordering — was
independently re-derived from the live source via direct `Read`/`Bash`
inspection and `codegraph_explore`, not accepted from any prior pass's prose
or the plan's own citations.
