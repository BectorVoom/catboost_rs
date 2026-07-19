# Plan Check Report — EXPORT-01 ONNX Export

**Checker agent:** `plan-checker` (project-installed, `~/.claude/agents/plan-checker.md`)
**Passes run:** 3 (this planning process's cap — "no more than three checker passes")
**Final status:** **ISSUES_FOUND on pass 3, with the sole remaining issue
fixed directly (not re-verified by a 4th pass) per the cap rule.** T0–T5 are
fully checker-confirmed across all 3 passes; T6/T7 carry one
applied-but-not-independently-re-verified fix. This is **not** a clean
checker `PASS` — see `SPEC.md`'s header note and the Final Disposition below.

---

## Pass 1 — verdict: ISSUES_FOUND

Reviewed the original `research.md` + `SPEC.md` + `PLAN.md`.

Findings:
1. **[CRITICAL]** AT-01f-1b (facade CTR-rejection test) instructed loading
   an existing CTR fixture, but `cb_model::load_cbm`/`load_json` never
   reconstruct `ModelSplit::Ctr`/`ctr_data` from any file (CTR-model loading
   is separate, unmerged work on `feat/23-ctr-model-loading`) — no fixture
   could ever exercise the rejection path as specified.
2. **[MAJOR]** Multiclass classifier leaf-value layout was unaddressed:
   upstream is leaf-major, this port's `ObliviousTree.leaf_values` is
   dimension-major; the only multiclass test (`post_transform=="SOFTMAX"`)
   could not catch a leaf/class transposition bug.
3. **[MINOR]** SPEC §9 risk 7 / PLAN T6 hedged the `PyCbError`/`to_pyerr`
   match-arm requirement as conditional ("verify... if it does not...")
   when it is a certain, compiler-enforced requirement (`to_pyerr` is an
   exhaustive match, no wildcard).
4. **[MINOR]** Binary classifier's `probability_tensor` output shape
   (`secondDim=2` even for 1-dim models) was undocumented and untested.

Everything else (guard predicate/order, reversed-split-order algorithm,
`BRANCH_GT`/complete-binary-tree indexing, zero-bias `base_values` gating,
binary asymmetric-bias trick, `ir_version=3`/opset-2 pin, `classlabels`
default fallback, task dependency ordering, backend-feature-passthrough
non-conflict) was independently verified via CodeGraph + live upstream
source fetch and confirmed correct.

## Pass 2 — verdict: ISSUES_FOUND

All 4 pass-1 issues confirmed genuinely fixed (not just reworded) via
independent CodeGraph re-verification (dimension-major indexing formula
cross-checked against `apply.rs`'s own `predict_raw_multi_cat` usage;
`to_pyerr`'s exhaustive-match shape re-confirmed; upstream `secondDim` rule
re-fetched and confirmed).

New findings surfaced by the revision itself:
1. **[MAJOR]** AT-01f-1b's revised fix (hand-construct the CTR model, wrap
   via `Model::from_canonical`) was correct in principle, but the plan left
   the test's LOCATION ambiguous (`crates/catboost-rs/tests/` OR an internal
   `#[cfg(test)]` module) with a non-functional fallback ("add a
   `#[cfg(test)]`-only constructor" — which does not solve the external
   `tests/`-crate visibility problem, since `#[cfg(test)]` items are absent
   from the normal build that `tests/` binaries link against).
   `Model::from_canonical` is `pub(crate)`, reachable only from an internal
   module.
2. **[MINOR]** `CatBoostError::Export`'s target Python exception class was
   left unspecified (no mapping to `CatBoostValueError`/`PyIOError`/base
   `CatBoostError`, no test).
3. **[MINOR]** The `to_pyerr`-mandatory-step instruction was textually
   placed under "Red (PyO3)" instead of "Green (PyO3)" (readability only).

## Pass 3 — verdict: ISSUES_FOUND

All 3 pass-2 issues confirmed genuinely fixed via independent verification
(`Model::from_canonical`'s `pub(crate)` visibility re-confirmed; the
`mod error_test;` internal-mount precedent at `crates/catboost-rs/src/lib.rs:51`
confirmed real and distinct from `crates/catboost-rs/tests/`; the new
`to_pyerr` `Export`-arm code confirmed syntactically sound and type-correct
against `OnnxExportError`'s actual shape; the `to_pyerr` step confirmed
correctly repositioned under "Green").

One NEW issue surfaced by this pass's holistic re-read (a consequence of the
pass-2 fix, not present in that form before):

1. **[CRITICAL]** The `to_pyerr` `Export`-arm code (a **production**
   function in `crates/catboost-rs-py/src/errors.rs`) names
   `cb_model::OnnxExportError` directly, but `crates/catboost-rs-py/Cargo.toml`
   declares `cb-model` under `[dev-dependencies]` ONLY (explicitly commented
   as "used only by `errors_test.rs`"). Cargo's dev-dependency extern
   prelude is available to `cargo test` compilation but NOT to the real
   `cargo build -p catboost-rs-py` / `maturin develop`/`build` production
   build — so `cargo test -p catboost-rs-py` would PASS while the actual
   wheel build FAILS to compile, for the ENTIRE crate (not just
   `save_onnx`), since `to_pyerr` is the shared chokepoint every PyO3 method
   routes through. The plan's own validation-command list never ran
   `cargo build -p catboost-rs-py`, creating a real risk this genuine defect
   gets masked and/or misattributed to the known, unrelated python3.14-link
   environmental red suite.

Checker's own final disposition (verbatim summary): *"T0–T5 (the
`cb-model`-only slice) may proceed... T6/T7 (facade+Python surfacing) should
not be implemented from this PLAN.md as currently written until the
Cargo.toml/re-export gap is closed... This should go through one additional,
narrowly-scoped revision (a small, mechanical fix, not a full re-plan)
before implementation of T6 begins; it does not need to revisit T0–T5,
EXPORT-01a–e, or the three already-verified pass-2 fixes."*

## Resolution (applied without a 4th checker pass — cap reached)

Per the process's 3-pass cap ("repeat this revision loop for no more than
three checker passes... if material issues remain after three passes, stop,
preserve the latest artifacts and checker report, and return the unresolved
issues as blockers"), the pass-3 CRITICAL finding was fixed directly in
`SPEC.md`/`PLAN.md` rather than triggering a 4th automated pass:

- `PLAN.md` T6 now opens with an explicit **T6-0** prerequisite sub-task:
  promote `cb-model` from `[dev-dependencies]` to `[dependencies]` in
  `crates/catboost-rs-py/Cargo.toml` (preserving `default-features = false`
  and its WR-03 feature-unification rationale, re-verified for the regular-
  dependency case), and add `cargo build -p catboost-rs-py` as a REQUIRED
  validation step (both in T6's own validation and the plan-wide Validation
  commands section) so this class of dev-dependency-masking defect cannot
  hide behind `cargo test` again.
- `SPEC.md` §7 "Modified" now lists `crates/catboost-rs-py/Cargo.toml` and
  `crates/catboost-rs-py/src/errors.rs` explicitly.
- `SPEC.md`'s header carries an explicit, honest status note (see the file)
  stating T0–T5 are fully checker-verified and T6/T7 carry one
  applied-but-not-independently-re-verified fix.

**This fix has NOT been independently re-verified by a 4th plan-checker
pass.** It is evidence-backed (the checker's own pass-3 report cites the
exact `Cargo.toml` lines and explains the Cargo dev-dependency mechanics
precisely) and narrowly scoped, but per the skill's guardrails
("Do not claim the plan is ready unless the latest verdict is PASS... Never
weaken, omit, or reinterpret checker findings merely to obtain a pass"),
this report does **not** claim a clean `PASS`. It should be flagged for a
second look — either a manual review or a 4th checker pass in a fresh
session — before or during T6's implementation, specifically re-checking:
1. Does `cb-model` as a REGULAR (not dev) dependency of `catboost-rs-py`
   actually preserve the WR-03 feature-unification guarantee (no unintended
   `cb-backend` default-`cpu` re-enablement) once workspace-wide feature
   unification is considered, not just the single-crate view?
2. Does `cargo build -p catboost-rs-py` (as now mandated) actually pass
   once `cb-model` is promoted, with no other transitive fallout?

## Overall Verdict

**ISSUES_FOUND → 3 rounds of fixes applied (2 independently re-verified by
the checker, 1 applied post-cap without re-verification).** T0–T5 are ready
for implementation with full checker confidence. T6–T7 are ready for
implementation with the T6-0 fix applied but flagged for a follow-up check.
This is the accurate, non-inflated status per the skill's guardrails — it is
deliberately NOT reported as a checker `PASS`.
