---
slug: borders-vs-score-fn-builder
status: diagnosed
trigger: isolate borders vs score function in builder_oracle_test
created: 2026-06-19
updated: 2026-06-19
goal: find_root_cause_only
---

# Debug: isolate borders vs score-function in builder_oracle_test

## Symptoms (pre-filled from /gsd-explore session)

**Expected behavior:** `cargo test -p catboost-rs --test builder_oracle_test` passes —
both `builder_regression_full_cycle` and `builder_binclf_full_cycle` reproduce the
upstream catboost 1.2.10 `model_serde/{regression,binclf}/predictions.npy` to ≤1e-5
through the public facade (`CatBoostBuilder::fit` → save → load → predict).

**Actual behavior:** Both tests fail at the **Predictions stage**:
- regression: expected `1.3333638`, actual `1.2025143` (diff ~0.131)
- binclf: expected `0.1985496`, actual `0.2018587` (diff ~0.0033)

The Rust↔Rust round-trip legs (.cbm and .json save→load→predict reproduce baseline)
PASS — only the upstream ≤1e-5 oracle leg diverges. Per-crate library oracles
(cb-compute / cb-train / cb-model) all pass ≤1e-5.

**Error messages:** `compare_stage(Stage::Predictions, ...)` returns an error; the
test panics with "upstream <= 1e-5 oracle diverged".

**Timeline:** Pre-existing since Phase 4 (RAPI facade). Mislabeled "facade default
drift" across Phases 6.1–6.6. Not caused by any 6.x phase.

**Reproduction:** `cargo test -p catboost-rs --test builder_oracle_test`

## Prior analysis (confirmed by code in /gsd-explore)

Two differences exist between the FAILING facade path and the PASSING cb-train
oracle path. Need to isolate which one (or both) drives the divergence:

1. **score_function** — facade hardcodes **Cosine**
   (`crates/catboost-rs/src/builder.rs:286` → `score_function_default()` →
   `crates/cb-train/src/boosting.rs:545` = `EScoreFunction::Cosine`), and exposes
   **NO `.score_function()` setter**. The fixtures were generated with **L2**
   (`crates/cb-oracle/fixtures/model_serde/{regression,binclf}/config.json:26`
   `"score_function": "L2"`). Score function governs split selection → divergent
   tree structure. PRIME SUSPECT.

2. **borders** — facade computes borders from the pool via
   `select_borders_greedy_logsum(col, 254, false)` (`builder.rs:327`), while the
   passing oracle path loads pinned borders from the fixture `model.json`. The test
   docstring CLAIMS the computed borders reproduce upstream exactly for
   `numeric_tiny` — UNVERIFIED. Possible secondary contributor.

Full write-up: `.planning/notes/builder-oracle-score-function-root-cause.md`.

## Investigation goal (this session)

Empirically isolate the divergence into {score_function, borders}. Specifically:

- **Borders check (do first, cheapest):** compute
  `select_borders_greedy_logsum(col, 254, false)` for each `numeric_tiny` column and
  compare bit-for-bit against the pinned `float_feature_borders` in
  `crates/cb-oracle/fixtures/model_serde/regression/model.json`. Match ⇒ borders
  exonerated, docstring claim holds.
- **Score-function check:** drive `cb_train::train` directly (bypassing the facade's
  missing setter) with `EScoreFunction::L2` + pinned fixture borders + the same
  config the test uses, then run save→load→predict. Converges to ≤1e-5 ⇒ score
  function is the sole cause.

State, with evidence, which of {score_function, borders} drives the divergence and
by how much. This is `find_root_cause_only` (diagnose) — do NOT change production
code or fixtures; a throwaway test harness / example binary is fine.

## Environment constraints

- Root disk ~100% full; the cb-compute test profile may fail to link. Prefer a
  minimal per-crate harness or `--example`; avoid full-workspace builds. See memory
  `disk-pressure-and-full-suite-verification`.
- Relevant memory: `cb-train-uses-l2-but-catboost-defaults-cosine` (catboost CPU
  default IS Cosine; cb-train default was fixed to Cosine in 05-19) — implies the
  FIXTURE's L2 is the non-default outlier, not the facade.

## Current Focus

- status: ROOT CAUSE CONFIRMED.
- root_cause: score_function mismatch. The facade hardcodes EScoreFunction::Cosine
  (builder.rs:286, no setter); the upstream fixtures were trained with L2
  (config.json params.score_function="L2"). Score function governs split selection,
  producing a divergent tree structure → divergent predictions (regression 5.6e-1,
  binclf 7.3e-2 off). Driving cb_train::train with L2 (everything else identical)
  converges to ≤1e-5 against both fixtures.
- borders: EXONERATED. Computed borders differ from pinned (49 vs 2/2/0/3, NOT
  bit-exact — docstring claim is false), but the difference is benign: with score
  function fixed, computed vs pinned borders give identical predictions to f64
  precision. Borders contribute nothing to the divergence.
- next_action: report diagnosis (find_root_cause_only — no production fix here).
- cleanup: throwaway example crates/catboost-rs/examples/score_borders_isolate.rs
  was created, run, and DELETED (no production code/fixtures modified).

## Resolution

root_cause: builder_oracle_test Predictions divergence is driven SOLELY by a
  score_function mismatch. The facade (CatBoostBuilder) hardcodes
  EScoreFunction::Cosine via score_function_default() (builder.rs:286) and exposes
  no .score_function() setter, whereas the upstream model_serde/{regression,binclf}
  fixtures were trained with score_function="L2" (config.json params). Cosine vs L2
  selects different splits → different tree structure → different predictions.
  Empirically isolated via a 2x2 (score_function x borders-source) experiment:
  flipping Cosine→L2 (borders held fixed) moves regression error from 5.555e-1 to
  2.404e-8 and binclf from 7.261e-2 to 2.843e-9; swapping computed↔pinned borders
  (score held fixed) changes predictions by < f64 print precision. Borders are
  exonerated.
fix: NOT APPLIED (find_root_cause_only). Two candidate fix directions — see below.
verification: pending (diagnose-only mode).
files_changed: []

## Fix direction (recommendation)

PREFERRED: expose a `.score_function(EScoreFunction)` setter on CatBoostBuilder and
have the test set it to L2 (matching the fixtures). Rationale:
  - The fixtures encode a deliberate, valid CatBoost configuration (L2). The bug is
    a MISSING facade capability (no score-function setter), not a wrong fixture.
  - L2 is the score function the upstream artifacts were trained under and the one
    every passing per-crate cb-train/cb-compute oracle already locks. Regenerating
    fixtures to Cosine would discard that locked oracle coverage and force
    re-pinning model.json/predictions.npy across binclf+regression.
  - This is the smallest change that closes the facade↔fixture gap AND adds a real
    public-API capability (score_function is a first-class catboost param).
  Caveat: per memory `cb-train-uses-l2-but-catboost-defaults-cosine`, catboost's CPU
  DEFAULT is Cosine — so keep score_function_default()=Cosine as the builder default
  (correct parity), and let the test opt into L2 explicitly to match its L2 fixtures.

ALTERNATIVE (not recommended): regenerate the model_serde fixtures with
  score_function=Cosine so the facade's hardcoded default matches. Cheaper in code
  but loses the L2 oracle lock and is more fragile (re-pins multiple committed
  artifacts).

## Evidence

- timestamp: 2026-06-19 (static code/config audit)
  checked: fixture configs + facade params construction
  found: regression config.json params.score_function = "L2" (binclf same per
    prior analysis); facade boost_params() pins score_function = score_function_default()
    = EScoreFunction::Cosine (builder.rs:286); NO .score_function() setter on the
    builder. All other train params (iterations=5, depth=2, lr=0.1, l2=3.0,
    random_strength=0, boost_from_average, leaf=Gradient, bootstrap=No, seed=0)
    match between facade and fixture config.
  implication: score_function is a confirmed param mismatch on the facade path;
    must quantify its effect vs borders empirically.
- timestamp: 2026-06-19 (model.json border audit)
  checked: regression model.json float_feature_borders (pinned, upstream-trained)
  found: f0=[0.5756795406341553, 0.7963352203369141] (2), f1=[-0.08364950865507126,
    0.28499358892440796] (2), f2=[] (0), f3=[-0.1585310399532318, 0.3004736602306366,
    1.0769416093826294] (3). nan_value_treatment=AsIs on all.
  implication: pinned borders are the reference for experiment 1's bit-compare.

- timestamp: 2026-06-19 (EXPERIMENT 1 — borders bit-compare, example score_borders_isolate)
  checked: select_borders_greedy_logsum(col, 254, false) vs pinned model.json borders,
    per regression feature, bit-for-bit (to_bits()).
  found: NOT bit-exact, and not even close in COUNT. Facade computes the FULL distinct
    set (n=49 per feature — every distinct value becomes a border, because the 254
    budget >> the 50-row distinct count). Pinned upstream borders are reduced:
    f0=2, f1=2, f2=0, f3=3. The docstring claim "those borders reproduce upstream's
    border selection exactly for numeric_tiny" is FALSE — they are wildly different.
  implication: borders are NOT identical. The prior-analysis assumption that borders
    match for numeric_tiny is WRONG. BUT see EXPERIMENT 2 — the border difference
    turns out to be BENIGN for the final predictions.
- timestamp: 2026-06-19 (EXPERIMENT 2 — score-function isolation, example)
  checked: cb_train::train + cb_model::predict_raw vs upstream predictions.npy, varying
    {score_function} x {borders source}.
  found (regression, expected[0]=1.3333638):
    - L2     + PINNED borders   : max_abs_diff = 2.404e-8  (CONVERGES ≤1e-5)
    - L2     + COMPUTED borders : max_abs_diff = 2.404e-8  (CONVERGES ≤1e-5)
    - Cosine + PINNED borders   : max_abs_diff = 5.555e-1  pred[0]=1.2025143
    - Cosine + COMPUTED borders : max_abs_diff = 5.555e-1  pred[0]=1.2025143  (== facade)
  found (binclf, expected[0]=0.1985496):
    - L2     + COMPUTED borders : max_abs_diff = 2.843e-9  (CONVERGES ≤1e-5)
    - Cosine + COMPUTED borders : max_abs_diff = 7.261e-2  pred[0]=0.2018587  (== facade)
  implication: score_function is the SOLE driver of the divergence. Holding score
    function fixed, swapping computed<->pinned borders changes the max_abs_diff by
    < 1e-15 (identical to printed precision). Holding borders fixed, swapping
    Cosine<->L2 moves the regression error from 5.6e-1 to 2.4e-8 (7 orders of
    magnitude). The facade's Cosine+computed result reproduces the EXACT failing
    test values (regression 1.2025143; binclf 0.2018587), confirming the harness
    faithfully replicates the facade path.

## Eliminated

- hypothesis: borders (facade-computed vs pinned) contribute to the Predictions
  divergence.
  evidence: With score_function held fixed (L2 or Cosine), substituting the pinned
    upstream borders for the facade's computed borders changes the prediction
    max_abs_diff by less than printed f64 precision (2.404e-8 in both L2 cases;
    5.555e-1 in both Cosine cases). The borders differ in count (49 vs 2/2/0/3) but
    this difference is BENIGN: for numeric_tiny the boosting reaches the same leaf
    partition either way (the extra computed borders are never selected as splits, or
    select equivalent thresholds). Borders are exonerated as a contributor.
  timestamp: 2026-06-19
