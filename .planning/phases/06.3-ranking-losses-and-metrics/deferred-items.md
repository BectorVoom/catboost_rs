# Phase 06.3 — Deferred Items

## [06.3-03] PairLogit / PairLogitPairwise leaf-estimation parity gap (DEFERRED)

**Status:** der + Cholesky pairwise-leaf path + `is_pairwise_scoring` routing all LAND
and are unit-tested green; the **end-to-end per-stage oracle** for PairLogit and
PairLogitPairwise is DEFERRED pending a leaf-estimation der2 parity fix.

**Symptom:** PairLogit tree-0 leaf values diverge from catboost 1.2.10.
- depth-1, l2=0, lr=1.0 probe: catboost leaves = `[0.7778 (=7/9), -1.5556 (=-14/9)]`;
  the transcribed der (p=0.5 at exp-approx=1) + standard per-object Newton leaf gives
  `[1.0, -1.3333]`.
- **Root cause (isolated):** the per-leaf SECOND-derivative SUM differs. The total
  `Σ der2` over all objects from the transcribed `competitor.Weight·p·(p-1)` formula
  (`error_functions.h:861-862`) is `-3.5` (7 pairs × 2 × -0.25 at p=0.5), but the value
  catboost's leaves imply is `-3.857 (=-27/7)`. A `Σ der2 = -3.857` is impossible from
  `p(p-1)` with `p ∈ [0,1]` (the quadratic `p(1-p)=0.2755` has no real root) AND the
  pair weights were confirmed to be exactly `1.0` (2-col and 3-col-weight-1 pools give
  identical leaves) AND the leaf assignment matches (leaf_weights `[8,4]`).
- **Conclusion (refined via Gradient/Newton probes):** the gap is a **pair-weight
  normalization** catboost applies to explicit `Pool.pairs`, NOT a der-formula error.
  Probing the SAME depth-1/l2=0/lr=1 tree with both leaf methods isolates the leaf der
  sums directly:
  - `Gradient` leaf (`sumDer/sumWeight`): catboost = `[7/36, -14/36]`. The transcribed
    `sumDer` (= `[2.0, -2.0]`) MATCHES, but the implied per-leaf `sumWeight` is
    `[72/7, 36/7] ≈ [10.29, 5.14]`, NOT the raw pair-incidence counts `[8, 6]`.
  - `Newton` leaf (`sumDer/-sumDer2`): catboost = `[7/9, -14/9]`; implied `-sumDer2 =
    [18/7, 9/7]`, not the raw `p(1-p)` sums `[2.0, 1.5]`.
  Both the Gradient `sumWeight` AND the Newton `-sumDer2` are scaled by the SAME
  non-uniform per-pair weight (the `7` denominator == total pair count; group-relative
  factors). A `1/(group-pair-count)` scheme reproduces the leaf values to ~2 s.f. but not
  ≤1e-5 — the exact upstream normalization (likely `GeneratePairs` / `bt.PairwiseWeights`
  group-weight folding in `data_providers.cpp` / `approx_calcer.cpp:444`) must be
  transcribed into `build_query_info`'s pair→`Competitor.weight` mapping (Plan 01 seam).
  The der1/der2 FORMULAS and the Cholesky pairwise-leaf solve are independently
  unit-test-verified (the latter bit-exact vs `pairwise_leaves_calculation_ut.cpp`).

**What landed (green):**
- `Loss::PairLogit` / `Loss::PairLogitPairwise` der over Competitors (inline exp), unit-tested.
- `pairwise_leaves.rs` Cholesky leaf solve (2×2 + general), bit-exact vs upstream UT.
- `is_pairwise_scoring` / `is_plain_only` predicates + boosting leaf-path routing.
- Frozen catboost 1.2.10 `PairLogit` + `PairLogitPairwise` fixtures (committed for the
  follow-up oracle).

**Follow-up:** add a plan transcribing the pairwise leaf-der reduction
(`approx_calcer_querywise.cpp` `AddLeafDersForQueries` der2 path) so the per-leaf SumDer2
matches upstream, then gate the `pairlogit_oracle` + `pairlogit_pairwise_oracle`
per-stage tests ≤1e-5 against the already-frozen fixtures. The fixtures are committed;
only the leaf-der2 reduction + the two oracle tests remain.

**[06.3-09 UPDATE] PARTIAL CLOSURE + refined root cause.** The pairwise SPLIT-SCORING
weight was wired: `bt.PairwiseWeights` (per-object Σ competitor.weight, `CalcPairwiseWeights`
`approx_updater_helpers.h:74-89`) now feeds the histogram `sumWeight` (`scoring.cpp:275-279`)
+ the Newton L2 scaling (`CalcDeltaNewtonBody`, `sumAllWeights = Σ pairwise / n`) for the
`UsesPairsForCalculation` losses (`uses_pairwise_weights` / `calc_pairwise_weights` in
`boosting.rs`). This advanced the PairLogit oracle Splits match from index 4 → index 6.

The plan's diagnosis (normalize `Competitor.weight`) was REFUTED empirically: the frozen
fixture's explicit pairs carry weight `1.0` (pairs.npy is `(7,2)` winner/loser only, no
weight col) and upstream sets `Competitor.Weight = pair.Weight` verbatim
(`data_providers.cpp:327-329`) — `Competitor.weight` is correct as-is.

**The TRUE remaining blocker (still DEFERRED — needs the instrumented trainer):** the
PairLogit LEAF-der2 reduction. At iter-0 tree-0 the `TPairLogitError` der is
bit-verified-identical to upstream (der1 leaf sums match), yet the Newton leaf deltas
diverge in a way NO global L2 / pairwise scaling reconciles: e.g. the single-object leaf
holding object 10 (its only pair (10,11) split across leaves) needs an effective
denominator of ~23 (delta 0.0064) where `sumDer/(-sumDer2 + scaledL2)` with the identical
der (`sumDer=0.5`, `sumDer2=-0.25`) yields ~3.75 (delta 0.04) — a ~6× gap, while leaf 0
needs denom 7.47 vs 5.5. The per-leaf denominators are mutually inconsistent with any
`-sumDer2 + C`, so the per-leaf SumDer2 (or a per-leaf coupling the pointwise path omits)
genuinely differs from a plain per-object der2 sum. Pinning it ≤1e-5 requires the
instrumented catboost 1.2.10 trainer's per-leaf SumDer/SumDer2 log, INFEASIBLE this
session (toolchain/disk, `catboost-instrumented-trainer-build`). The two oracle tests
(`pairlogit_oracle_test.rs` / `pairlogit_pairwise_oracle_test.rs`) are committed,
`#[ignore]`'d with this deferral, and run the FULL ≤1e-5 gate the moment `#[ignore]` is
removed — no tolerance weakened.

**LambdaMart (Wave B) shipped fully** — per-stage oracle ≤1e-5 GREEN (the listwise
positive-hessian Newton-denominator fix in `leaf.rs::newton_leaf_delta` was the key).

---

## 06.3-04 Wave C — YetiRank/YetiRankPairwise/StochasticRank end-to-end trainer fixture (DEFERRED, path c)

**Decision: ESCALATE-DON'T-WEAKEN (D-6.3-03b).** The Task-1 feasibility probe found the
instrumented catboost 1.2.10 TRAINER build infeasible this session: the persisted `/tmp`
clang-18 + built `_catboost` toolchain is ABSENT; no `clang`/`clang-18` on the host (apt
candidate only); disk at 95-97% / ~8-12G free — the documented link-failure regime
(`disk-pressure-and-full-suite-verification`). NO oracle weakened, NO `#[ignore]`, NO
fabricated fixtures.

**What landed (green, validated):**
- `Loss::YetiRank/YetiRankPairwise{permutations,decay}` + `Loss::StochasticRank{metric,
  sigma,mu,num_estimations}` + `StochasticRankMetric` + validate + defaults.
- `yetirank.rs` sampler (2-level seed + Gumbel + Classic weights) + `derive_query_seeds`.
- StochasticRank DCG/NDCG Monte-Carlo der (std_normal + SFA, der2=0) in `ranking_der.rs`.
- Boosting per-iteration competitor re-sample; YetiRankPairwise → Cholesky leaf (Plain).
- TWO standalone OFFLINE instrumented generators (`yetirank_oracle.cpp` /
  `stochasticrank_oracle.cpp`, ZERO catboost includes) — compile clean, RUN, and
  SELF-ORACLE bit-for-bit vs `cb-core::TFastRng64`/`std_normal`. RNG-draw ground truth
  frozen under `ranking_corpus/{yetirank,yetirank_pairwise,stochasticrank}/`.
- 3 per-stage oracles gating the RNG draw log ≤1e-5 (sampled competitor weights
  `0.192/0.098250/0.083250` + 2-level query seed + Gaussian noise stream), all GREEN.

**Deferred (OFFLINE closure):** the end-to-end per-stage TRAINER fixtures (trained
`model.json` + staged/predictions `.npy`) for YetiRank/YetiRankPairwise/StochasticRank,
needed for `compare_stage(Splits|LeafValues|StagedApprox|Predictions)` ≤1e-5 across full
training. The three oracle tests (`*_end_to_end_per_stage`) are WIRED to run the full gate
the moment each `model.json` lands (they currently assert the deferred-fixture invariant —
the RNG ground truth is committed — so they never silently pass). Closure recipe (toolchain
+ disk permitting): `crates/cb-oracle/generator/instrument_ranking_rng_README.md` STATUS
section — follow the `instrument_live_trainer_README.md` sudo-free clang-18 recipe, train
the three corpus configs with `CB_INSTRUMENT_LOG`, freeze the per-stage `.npy` + RNG log.

**Also deferred (scoped out, not gap-blocking):** StochasticRank non-DCG metrics
(PFound/ERR/MRR/FilteredDCG — admitted by the enum), YetiRank multi-thread (blockCount>1)
block-seed partition (fixtures are single-thread).

## [06.3-06] Pre-existing `clippy::indexing_slicing` errors in `stochastic_rank_group_der` (OUT OF SCOPE)

**Status:** DEFERRED — pre-existing, NOT introduced by 06.3-06.

`cargo clippy -p cb-compute --lib` reports 42 `error: indexing may panic` diagnostics
inside the production `stochastic_rank_group_der` body (`ranking_der.rs` ~742-829:
`noise[d]`, `scores[d]`, `cum_sum[pos+1]`, `cum_sum_up[..]`, `cum_sum_low[..]`,
`order[pos]`, `der1[doc_id]`, etc.). Confirmed pre-existing: stashing the 06.3-06
changes and re-running clippy on `HEAD` (00b299a) reproduces the identical 42-error
count. The 06.3-06 edits (`calc_dcg_metric_diff` pos_weights read + `lambdamart_ideal_ndcg`
sum_f64) add NO new indexing-panic sites — both use bounds-checked
`.get(..).copied().unwrap_or(..)` / iterator forms. No git hook or pre-commit config
gates clippy, so commits are unaffected. Out of scope per the executor scope-boundary
rule (only auto-fix issues directly caused by the current task). Remediation: convert the
`stochastic_rank_group_der` direct-index accesses to `.get(..).copied().unwrap_or(0.0)` in a
dedicated hardening pass (touches the parity-critical DCG cumulative-stats loop, so it must
be oracle-revalidated, not blind-fixed here).
