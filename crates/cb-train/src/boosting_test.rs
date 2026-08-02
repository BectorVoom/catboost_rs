//! Unit tests for the plain boosting loop's leaf-delta computation
//! ([`crate::boosting::compute_leaf_deltas`]), focused on the RESEARCH Pattern 3
//! Exact-alpha threading (Plan 06.1-03 / D-6.1-05): the Exact leaf branch must
//! thread the ACTIVE loss's `(alpha, delta)` into `exact_leaf_delta`, NOT the
//! hardcoded `QUANTILE_ALPHA` / `QUANTILE_DELTA` median constants.
//!
//! These are falsifiable regression catches: a revert of the threading (back to
//! the unconditional hardcoded 0.5) flips `quantile_alpha07_threads_alpha`.
//!
//! Dedicated test file (CLAUDE.md source/test separation — no inline
//! `#[cfg(test)]` in production source). Mounted via `#[path]` from `boosting.rs`,
//! so it can reach the private `compute_leaf_deltas`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_compute::{
    exact_leaf_delta, GroupSpan, LeafMethod, Loss, RankingCompetitor, QUANTILE_ALPHA,
    QUANTILE_DELTA,
};

use super::{calc_pairwise_weights, compute_leaf_deltas, uses_pairwise_weights};
use super::oblivious_from_grown;
use crate::tree::{CtrSplitSpec, Split};
use super::validate_score_function;
use cb_compute::EScoreFunction;

// WR-07 / CR-01: the second-order (Newton) split-score functions have no faithful
// CPU training implementation (the CPU scoring path produces only the first-order
// weight-count reduction, so they would silently degrade to L2 / Cosine). The
// formula-identity self-oracles in `cb-compute` cannot catch that wiring gap; the
// real guard is `validate_score_function`, which must REJECT them at train time.
// These falsifiable regression catches flip if that gate is ever removed.
#[test]
fn validate_score_function_rejects_newton_l2() {
    assert!(
        validate_score_function(EScoreFunction::NewtonL2).is_err(),
        "NewtonL2 must be rejected on the CPU training path (no der2-fill; would \
         silently degrade to L2)"
    );
}

#[test]
fn validate_score_function_rejects_newton_cosine() {
    assert!(
        validate_score_function(EScoreFunction::NewtonCosine).is_err(),
        "NewtonCosine must be rejected on the CPU training path (no der2-fill; would \
         silently degrade to Cosine)"
    );
}

#[test]
fn validate_score_function_accepts_first_order_variants() {
    // Cosine/L2 (shipped) plus the first-order GPU-only calcers (Solar/LOO/Sat),
    // which compute correctly from the first-order stats, are accepted.
    for sf in [
        EScoreFunction::Cosine,
        EScoreFunction::L2,
        EScoreFunction::SolarL2,
        EScoreFunction::LOOL2,
        EScoreFunction::SatL2,
    ] {
        assert!(
            validate_score_function(sf).is_ok(),
            "first-order score function {sf:?} must be accepted on the CPU path"
        );
    }
}

/// Run the Exact-leaf branch of `compute_leaf_deltas` over a single leaf whose
/// per-member residuals are exactly `residuals` (we feed `approx = 0`, `target =
/// residuals`, so the internal `target - approx` recovers them), unit weights, and
/// return the single leaf delta. `der2`/`weighted_der1` are unused by the Exact
/// branch (it works off the residuals), so they are filled trivially.
fn exact_single_leaf(loss: Loss, residuals: &[f64]) -> f64 {
    exact_single_leaf_dim(loss, residuals, 0)
}

/// As [`exact_single_leaf`] but for a specific output dimension index `dim_index`
/// (the MultiQuantile per-dimension `alpha[dim_index]` selector).
fn exact_single_leaf_dim(loss: Loss, residuals: &[f64], dim_index: usize) -> f64 {
    let n = residuals.len();
    let leaf_of = vec![0_usize; n]; // every object in leaf 0.
    let weighted_der1 = vec![0.0_f64; n];
    let der2 = vec![0.0_f64; n];
    let weights = vec![1.0_f64; n];
    let approx = vec![0.0_f64; n];
    let target = residuals.to_vec();

    let deltas = compute_leaf_deltas(
        LeafMethod::Exact,
        &loss,
        &leaf_of,
        &weighted_der1,
        &der2,
        &weights,
        &approx,
        &target,
        /* scaled_l2 */ 0.0,
        /* n_leaves */ 1,
        dim_index,
    );
    assert_eq!(deltas.len(), 1);
    deltas[0]
}

#[test]
fn quantile_alpha07_threads_alpha_not_hardcoded_half() {
    // Residuals [1,2,3,4,5], unit weights: the weighted 0.5-quantile is 3, the
    // weighted 0.7-quantile is 4 (DISTINCT). If the Exact branch threaded the
    // active Quantile{0.7} alpha, the leaf delta is the 0.7-quantile; if it
    // regressed to the hardcoded 0.5, it would be the 0.5-quantile — so this is a
    // falsifiable threading catch.
    let residuals = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let alpha = 0.7;
    let delta = QUANTILE_DELTA;

    let delta_07 = exact_single_leaf(Loss::Quantile { alpha, delta }, &residuals);

    // Anchor: the alpha-general exact_leaf_delta at alpha=0.7 (leaf.rs UNCHANGED).
    let residuals_f32: Vec<f32> = residuals.iter().map(|&r| r as f32).collect();
    let weights = vec![1.0_f64; residuals.len()];
    let expected_07 = exact_leaf_delta(&residuals_f32, &weights, alpha, delta);
    assert!(
        (delta_07 - expected_07).abs() < 1e-12,
        "Exact branch must thread alpha=0.7: got {delta_07}, expected {expected_07}"
    );

    // Sanity: the 0.7-quantile differs from the 0.5-quantile here, so the test
    // genuinely distinguishes threaded-0.7 from hardcoded-0.5.
    let expected_05 = exact_leaf_delta(&residuals_f32, &weights, 0.5, delta);
    assert!(
        (expected_07 - expected_05).abs() > 0.5,
        "test corpus must separate the 0.7- and 0.5-quantiles (got 0.7={expected_07}, 0.5={expected_05})"
    );
}

#[test]
fn quantile_alpha05_equals_mae_exact_leaf() {
    // MAE == Quantile{alpha=0.5, delta=1e-6} at the Exact-leaf level: the threaded
    // Quantile{0.5} leaf delta must equal the Mae leaf delta (which threads the
    // hardcoded QUANTILE_ALPHA/QUANTILE_DELTA == 0.5/1e-6) bit-for-bit.
    let residuals = [-2.5_f64, 0.0, 1.0, 3.25, 7.0, -4.5];

    let mae_delta = exact_single_leaf(Loss::Mae, &residuals);
    let q05_delta = exact_single_leaf(
        Loss::Quantile {
            alpha: QUANTILE_ALPHA,
            delta: QUANTILE_DELTA,
        },
        &residuals,
    );
    assert_eq!(
        mae_delta, q05_delta,
        "MAE Exact leaf must equal Quantile{{0.5}} Exact leaf (byte-stable)"
    );
}

#[test]
fn multiquantile_threads_per_dimension_alpha() {
    // MultiQuantile (D-6.2-05): the Exact leaf for dimension `d` must thread
    // alpha[d] (each dimension is an independent quantile). With alpha=[0.3,0.7],
    // dimension 0 takes the weighted 0.3-quantile and dimension 1 the weighted
    // 0.7-quantile of the SAME residuals — DISTINCT values. A regression that used
    // a single fixed alpha (or alpha[0] for every dim) flips dimension 1.
    let residuals = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let delta = QUANTILE_DELTA;
    let alpha = vec![0.3_f64, 0.7];

    let d0 = exact_single_leaf_dim(
        Loss::MultiQuantile {
            alpha: alpha.clone(),
            delta,
        },
        &residuals,
        0,
    );
    let d1 = exact_single_leaf_dim(
        Loss::MultiQuantile {
            alpha: alpha.clone(),
            delta,
        },
        &residuals,
        1,
    );

    // Each dimension must equal the alpha-general exact_leaf_delta at its own alpha.
    let residuals_f32: Vec<f32> = residuals.iter().map(|&r| r as f32).collect();
    let weights = vec![1.0_f64; residuals.len()];
    let expected_d0 = exact_leaf_delta(&residuals_f32, &weights, 0.3, delta);
    let expected_d1 = exact_leaf_delta(&residuals_f32, &weights, 0.7, delta);
    assert!((d0 - expected_d0).abs() < 1e-12, "dim 0 must thread alpha[0]=0.3");
    assert!((d1 - expected_d1).abs() < 1e-12, "dim 1 must thread alpha[1]=0.7");
    assert!(
        (d0 - d1).abs() > 0.5,
        "the two quantile levels must produce distinct leaf deltas (got d0={d0}, d1={d1})"
    );
}

#[test]
fn multiquantile_alpha07_dimension_equals_scalar_quantile07_leaf() {
    // The degenerate-equivalence anchor at the leaf level (D-6.2-05): a
    // MultiQuantile dimension at alpha=0.7 must produce the SAME Exact leaf delta as
    // the scalar Quantile{0.7} path (leaf.rs reused verbatim per dimension).
    let residuals = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let delta = QUANTILE_DELTA;

    let mq = exact_single_leaf_dim(
        Loss::MultiQuantile {
            alpha: vec![0.7],
            delta,
        },
        &residuals,
        0,
    );
    let scalar = exact_single_leaf(Loss::Quantile { alpha: 0.7, delta }, &residuals);
    assert_eq!(
        mq, scalar,
        "MultiQuantile{{[0.7]}} dimension-0 leaf must equal scalar Quantile{{0.7}} leaf"
    );
}

// --- LOSS-04 06.3-09: pairwise split-scoring / leaf weight (`bt.PairwiseWeights`) ---

/// `uses_pairwise_weights` selects exactly the `UsesPairsForCalculation` losses
/// (`enum_helpers.cpp:502` = YetiRank* | PairLogit*) — these drive split-scoring
/// `sumWeight` + L2 scaling off the per-object PAIRWISE weights, not the per-object
/// sample weights. A regression that drops PairLogit/PairLogitPairwise (or adds a
/// pointwise loss) flips this.
#[test]
fn uses_pairwise_weights_selects_only_pair_losses() {
    assert!(uses_pairwise_weights(&Loss::PairLogit));
    assert!(uses_pairwise_weights(&Loss::PairLogitPairwise));
    assert!(uses_pairwise_weights(&Loss::YetiRank {
        permutations: 10,
        decay: 0.85
    }));
    assert!(uses_pairwise_weights(&Loss::YetiRankPairwise {
        permutations: 10,
        decay: 0.85
    }));
    // Pointwise / querywise / listwise losses do NOT use pairwise weights.
    assert!(!uses_pairwise_weights(&Loss::Logloss));
    assert!(!uses_pairwise_weights(&Loss::QueryRmse));
    assert!(!uses_pairwise_weights(&Loss::LambdaMart {
        metric: cb_compute::LambdaMartMetric::Ndcg,
        sigma: 1.0,
        top: -1,
        norm: true
    }));
}

/// `calc_pairwise_weights` mirrors upstream `CalcPairwiseWeights`
/// (`approx_updater_helpers.h:74-89`): for every winner→loser competitor edge it
/// adds `competitor.weight` to BOTH the winner's and the loser's per-object slot,
/// so `pw[obj] = Σ competitor.weight` over all pairs incident on `obj`. This is the
/// histogram / leaf `sumWeight` (`bt.PairwiseWeights`) the pairwise-loss split
/// scoring consumes — NOT the uniform per-object `1.0`.
#[test]
fn calc_pairwise_weights_sums_competitor_weights_over_both_endpoints() {
    // One group [0,3): winner 0 -> loser 1 (w 1.0); winner 0 -> loser 2 (w 1.0);
    // winner 1 -> loser 2 (w 1.0). Object 0 is a winner twice (pw 2), object 1 is
    // winner once + loser once (pw 2), object 2 is loser twice (pw 2).
    let group = GroupSpan {
        begin: 0,
        end: 3,
        weight: 1.0,
        competitors: vec![
            vec![
                RankingCompetitor { id: 1, weight: 1.0 },
                RankingCompetitor { id: 2, weight: 1.0 },
            ],
            vec![RankingCompetitor { id: 2, weight: 1.0 }],
            Vec::new(),
        ],
    };
    let pw = calc_pairwise_weights(&[group], 3);
    // NON-uniform vs the old hardcoded 1.0: each object touched by 2 pairs -> 2.0.
    assert_eq!(pw, vec![2.0, 2.0, 2.0]);
    // The total pairwise weight is 2 x (number of pairs) = 6 (each pair scores both
    // endpoints), the value `scale_l2_reg` divides by `n` for the L2 scaling.
    let total: f64 = pw.iter().sum();
    assert!((total - 6.0).abs() < 1e-12);
}

/// A weighted-pair group: `competitor.weight` (not a uniform 1.0) is what gets
/// summed, and a group with NO competitors leaves its objects at pairwise weight
/// `0.0` (upstream `bt.PairwiseWeights` is zero-initialized, `Fill(..., 0)`).
#[test]
fn calc_pairwise_weights_honors_weights_and_empty_groups() {
    // Group A [0,2): winner 0 -> loser 1, weight 2.5.
    let group_a = GroupSpan {
        begin: 0,
        end: 2,
        weight: 1.0,
        competitors: vec![vec![RankingCompetitor { id: 1, weight: 2.5 }], Vec::new()],
    };
    // Group B [2,4): NO pairs -> both objects stay at 0.0.
    let group_b = GroupSpan {
        begin: 2,
        end: 4,
        weight: 1.0,
        competitors: vec![Vec::new(), Vec::new()],
    };
    let pw = calc_pairwise_weights(&[group_a, group_b], 4);
    assert_eq!(pw, vec![2.5, 2.5, 0.0, 0.0]);
}

// ===========================================================================
// T03 / SPEC-OH-01 — ObliviousTree carries ordered mixed-kind levels
// ===========================================================================

/// The trainer must stop discarding `GrownTree.level_kinds` at the persist step.
///
/// Until now `oblivious_from_grown`'s predecessor built `ObliviousTree` from
/// `grown.splits` + `ctr_splits` only, throwing the per-level interleaving away.
/// Since `cb_model`'s apply path treats STORED split order as the leaf-index bit
/// order, a tree whose level 0 is a CTR split and level 1 a float split had its
/// leaf indices transposed once it crossed the trainer→model boundary. Carrying
/// `level_kinds` through is the prerequisite for fixing that (T04) and for storing
/// mixed float/one-hot levels at all (T19).
#[test]
fn oblivious_tree_records_level_kinds_in_level_order() {
    use crate::tree::LevelKind;

    let grown_level_kinds = vec![
        LevelKind::Ctr {
            ctr_idx: 0,
            border: 0.5,
        },
        LevelKind::Float(0),
    ];

    let tree = oblivious_from_grown(
        vec![Split {
            feature: 0,
            border: 1.5,
        }],
        vec![CtrSplitSpec {
            projection: crate::TProjection::single(0),
            ctr_type: 0,
            prior_num: 0.5,
            prior_denom: 1.0,
            target_border_idx: 0,
            border: 0.5,
            shift: 0.0,
            scale: 1.0,
        }],
        Vec::new(),
        grown_level_kinds.clone(),
        vec![10.0, 20.0, 30.0, 40.0],
        vec![1.0, 1.0, 1.0, 1.0],
    );

    assert_eq!(
        tree.level_kinds, grown_level_kinds,
        "level_kinds must survive the persist step in LEVEL order (CTR at level 0, \
         float at level 1) — dropping it is what transposes leaf indices downstream"
    );
    assert!(
        tree.one_hot_splits.is_empty(),
        "T03 introduces the one_hot_splits field but never populates it (that is T19)"
    );
    // The pre-existing kind-grouped vectors are untouched.
    assert_eq!(tree.splits.len(), 1);
    assert_eq!(tree.ctr_splits.len(), 1);
}

/// A float-only tree persists EMPTY `level_kinds`, so downstream consumers take the
/// unchanged legacy path and the float-only output stays byte-identical
/// (SPEC-OH-31). This is the structural half of the no-regression guarantee.
#[test]
fn float_only_tree_persists_empty_level_kinds() {
    let tree = oblivious_from_grown(
        vec![
            Split {
                feature: 0,
                border: 1.5,
            },
            Split {
                feature: 1,
                border: 2.5,
            },
        ],
        Vec::new(),
        Vec::new(),
        Vec::new(),
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 1.0, 1.0, 1.0],
    );

    assert!(
        tree.level_kinds.is_empty(),
        "a float-only tree must persist EMPTY level_kinds so consumers keep the \
         byte-identical legacy path"
    );
    assert!(tree.one_hot_splits.is_empty());
    assert_eq!(tree.splits.len(), 2);
}

// ── SPEC-OH-04 — one-hot vs CTR cat-column partition ────────────────────────

/// SPEC-OH-04 — the one-hot-routed and CTR-eligible column lists are derived
/// from ONE `route_categorical` match, so they are disjoint BY CONSTRUCTION
/// (two independent filters could drift into double-counting a column, which
/// would materialize the same feature on both paths).
#[test]
fn one_hot_routed_columns_are_partitioned_disjointly_from_ctr_eligible() {
    // cardinalities [2, 5, 3] at one_hot_max_size = 3:
    //   card 2 <= 3 -> OneHot; card 5 > 3 -> Ctr; card 3 <= 3 -> OneHot.
    let (one_hot, ctr) = super::partition_cat_columns(&[2, 5, 3], 3);
    assert_eq!(one_hot, vec![0, 2]);
    assert_eq!(ctr, vec![1]);

    // A constant column (cardinality <= 1) routes to NEITHER list
    // (`route_categorical` -> `Skip`).
    let (one_hot, ctr) = super::partition_cat_columns(&[1, 2], 2);
    assert_eq!(one_hot, vec![1]);
    assert!(ctr.is_empty());

    // Disjointness, asserted rather than assumed.
    let (one_hot, ctr) = super::partition_cat_columns(&[1, 2, 3, 7, 9], 3);
    assert!(
        one_hot.iter().all(|c| !ctr.contains(c)),
        "the two lists must never share a column"
    );
}

// ── SPEC-OH-05 — one-hot bin columns + the exact bin -> raw-hash inverse ────

/// SPEC-OH-05 — a one-hot-routed column contributes its FIRST-SEEN
/// `PerfectHash` bin column, and the fit-wide `bin -> raw hash` table is its
/// EXACT inverse.
///
/// The inverse must be built by zipping the raw column with the bins
/// `perfect_hash_bins` returned (first-seen order), NOT by sorting the distinct
/// hashes ascending: `PerfectHash::remap_bounded` assigns `bin = map.len()` on
/// first sight, so `hash_by_bin[1]` is the SECOND value ENCOUNTERED, which for
/// this column is `"a"` — while the ascending-hash reading would give a
/// different value. The `hash_by_bin.len() == cardinality` assertion is what
/// makes a partially-filled table impossible.
#[test]
fn one_hot_columns_build_bins_and_an_exact_bin_to_hash_inverse() {
    let cat_columns = vec![vec![
        "b".to_owned(),
        "a".to_owned(),
        "b".to_owned(),
        "c".to_owned(),
    ]];
    let (bins, hash_by_bin) = super::build_one_hot_columns(&cat_columns, &[0])
        .expect("building one-hot columns must succeed");

    assert_eq!(bins, vec![vec![0, 1, 0, 2]], "first-seen bin assignment");
    assert_eq!(
        hash_by_bin,
        vec![vec![
            cb_data::calc_cat_feature_hash("b"),
            cb_data::calc_cat_feature_hash("a"),
            cb_data::calc_cat_feature_hash("c"),
        ]],
        "the inverse follows FIRST-SEEN bin order, not ascending hash order"
    );
    assert_eq!(hash_by_bin[0].len(), 3, "one entry per distinct value");

    // The inverse is exact: re-deriving each object's hash through the table
    // reproduces hashing the raw value directly.
    for (obj, raw) in cat_columns[0].iter().enumerate() {
        let bin = bins[0][obj] as usize;
        assert_eq!(hash_by_bin[0][bin], cb_data::calc_cat_feature_hash(raw));
    }
}

/// SPEC-OH-05 — only the LISTED one-hot columns are materialized, in the listed
/// order, so a mixed pool's CTR-routed columns never leak into `cat_bins`.
#[test]
fn build_one_hot_columns_materializes_only_the_listed_columns() {
    let cat_columns = vec![
        vec!["x".to_owned(), "y".to_owned()],
        vec!["p".to_owned(), "q".to_owned()],
        vec!["m".to_owned(), "n".to_owned()],
    ];
    let (bins, hash_by_bin) =
        super::build_one_hot_columns(&cat_columns, &[2, 0]).expect("must succeed");
    assert_eq!(bins.len(), 2);
    assert_eq!(hash_by_bin.len(), 2);
    // Position 0 is ABSOLUTE column 2 ("m","n"); position 1 is column 0.
    assert_eq!(hash_by_bin[0][0], cb_data::calc_cat_feature_hash("m"));
    assert_eq!(hash_by_bin[1][0], cb_data::calc_cat_feature_hash("x"));

    // An out-of-range absolute index is a typed error, never a silent empty
    // column that would make every one-hot split fail.
    assert!(super::build_one_hot_columns(&cat_columns, &[9]).is_err());
}

// ---------------------------------------------------------------------------
// E03 — characterization tests for `ctr_splits_for_tree`.
//
// CodeGraph reports this function as having NO covering tests, and E10 is about
// to change it (SPEC-CTRT-09). These two tests pin TODAY'S behavior verbatim so
// that E10's change is a visible, reviewed diff rather than a silent one. They
// are CHARACTERIZATION tests: they assert what the code does now, not what it
// ought to do. E10 is expected to update them deliberately.
//
// Falsifiability is established by the mutation check recorded in E03's
// completion evidence (§3.1), not by an initial Red — a characterization test
// passes on first write by construction.
// ---------------------------------------------------------------------------

#[test]
fn ctr_splits_for_tree_emits_one_spec_per_candidate_with_the_head_prior() {
    let cands = vec![
        crate::candidates::CtrCandidate {
            projection: crate::TProjection::from_features(&[0]),
            is_simple: true,
        },
        crate::candidates::CtrCandidate {
            projection: crate::TProjection::from_features(&[0, 1]),
            is_simple: false,
        },
    ];

    let specs = super::ctr_splits_for_tree(&cands, &[0.25, 0.75]);

    assert_eq!(specs.len(), 2, "one spec per candidate, in candidate order");

    for (i, spec) in specs.iter().enumerate() {
        // ONLY the head prior is used today; the tail (0.75) is dropped. E15 is
        // the task that makes the full prior list expand into separate splits.
        assert_eq!(
            spec.prior_num, 0.25,
            "spec {i}: only priors.first() is read today"
        );
        assert_eq!(spec.prior_denom, 1.0, "spec {i}");
        // Every candidate is emitted as Borders today regardless of the
        // configured type — this is exactly the inertness SPEC-CTRT-09 fixes.
        assert_eq!(
            spec.ctr_type,
            crate::ctr::ECtrType::Borders.as_i8(),
            "spec {i}: hard-coded Borders head"
        );
        assert_eq!(spec.target_border_idx, 0, "spec {i}");
        assert_eq!(spec.border, 0.0, "spec {i}");
        assert_eq!(spec.shift, 0.0, "spec {i}");
        assert_eq!(spec.scale, 1.0, "spec {i}");
        assert_eq!(
            spec.projection, cands[i].projection,
            "spec {i}: projection carried through unchanged, in order"
        );
    }
}

#[test]
fn ctr_splits_for_tree_empty_priors_defaults_to_half() {
    let cands = vec![
        crate::candidates::CtrCandidate {
            projection: crate::TProjection::from_features(&[0]),
            is_simple: true,
        },
        crate::candidates::CtrCandidate {
            projection: crate::TProjection::from_features(&[0, 1]),
            is_simple: false,
        },
    ];

    // Pins the `priors.first().copied().unwrap_or(0.5)` fallback.
    let specs = super::ctr_splits_for_tree(&cands, &[]);

    assert_eq!(specs.len(), 2);
    for (i, spec) in specs.iter().enumerate() {
        assert_eq!(spec.prior_num, 0.5, "spec {i}: empty-prior fallback is 0.5");
    }
}

// ---------------------------------------------------------------------------
// E02 — SPEC-CTRT-03 / acceptance A10: CPU-illegal CTR types are rejected with a
// typed error BEFORE any accumulation or tree growth.
//
// Upstream: catboost_options.cpp:504-509
//   CB_ENSURE(IsSupportedCtrType(CPU, ctrType),
//             "Ctr type " << ctrType << " is not implemented on CPU yet")
//
// Zero behavior change today: no site anywhere in the workspace pins a
// non-default ECtrType, and both defaults are Borders, so this guard rejects
// nothing that currently exists.
// ---------------------------------------------------------------------------

/// A fully-explicit `BoostParams` (Pitfall-6 discipline: every field pinned, so a
/// changed default cannot silently alter what this test exercises).
fn e02_params(simple: crate::ctr::ECtrType, combos: crate::ctr::ECtrType) -> crate::BoostParams {
    crate::BoostParams {
        loss: cb_compute::Loss::Logloss,
        iterations: 1,
        depth: 1,
        learning_rate: 0.1,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        leaf_method: cb_compute::LeafMethod::Gradient,
        bootstrap_type: crate::bootstrap::EBootstrapType::No,
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: 0,
        od_type: crate::overfit::EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        auto_learning_rate: false,
        one_hot_max_size: 1,
        permutation_count: 1,
        fold_len_multiplier: 2.0,
        simple_ctr: simple,
        simple_ctr_priors: crate::simple_ctr_priors_default(),
        counter_calc_method: crate::counter_calc_method_default(),
        boosting_type: crate::boosting_type_default(),
        max_ctr_complexity: 1,
        combinations_ctr: combos,
        combinations_ctr_priors: crate::combinations_ctr_priors_default(),
        score_function: crate::score_function_default(),
        has_time: false,
        feature_weights: crate::feature_weights_default(),
        first_feature_use_penalties: crate::first_feature_use_penalties_default(),
        per_object_feature_penalties: crate::per_object_feature_penalties_default(),
        penalties_coefficient: crate::penalties_coefficient_default(),
        monotone_constraints: crate::monotone_constraints_default(),
        grow_policy: crate::grow_policy_default(),
        max_leaves: crate::max_leaves_default(),
        min_data_in_leaf: crate::min_data_in_leaf_default(),
    }
}

#[test]
fn cpu_illegal_ctr_types_are_typed_unsupported_before_training() {
    use crate::ctr::ECtrType;

    // A 4-row, 1-cat-column corpus. Small on purpose: the rejection must happen
    // before any accumulation, so the data never actually gets trained on.
    let cat_cols = vec![vec![
        "a".to_owned(),
        "b".to_owned(),
        "a".to_owned(),
        "b".to_owned(),
    ]];
    let y = vec![0.0_f64, 1.0, 0.0, 1.0];
    let w = vec![1.0_f64; 4];

    // Case 1: simple_ctr = FloatTargetMeanValue (GPU-only, restrictions.h:18-48).
    let params = e02_params(ECtrType::FloatTargetMeanValue, ECtrType::Borders);
    let result = crate::train_cat(
        &cb_backend::CpuBackend,
        &[],
        &[],
        &cat_cols,
        &y,
        &w,
        &params,
        None,
    );
    match result {
        Err(cb_core::CbError::Unsupported(msg)) => {
            assert!(
                msg.contains("FloatTargetMeanValue"),
                "must name the type: {msg}"
            );
            assert!(
                msg.contains("not implemented on CPU yet"),
                "must mirror upstream catboost_options.cpp:504-509: {msg}"
            );
        }
        other => panic!("expected CbError::Unsupported, got {other:?}"),
    }

    // Case 2: combinations_ctr = FeatureFreq — the OTHER field must be checked
    // too, so a guard that only reads simple_ctr fails here.
    let params = e02_params(ECtrType::Borders, ECtrType::FeatureFreq);
    let result = crate::train_cat(
        &cb_backend::CpuBackend,
        &[],
        &[],
        &cat_cols,
        &y,
        &w,
        &params,
        None,
    );
    match result {
        Err(cb_core::CbError::Unsupported(msg)) => {
            assert!(msg.contains("FeatureFreq"), "must name the type: {msg}");
            assert!(
                msg.contains("not implemented on CPU yet"),
                "must mirror upstream: {msg}"
            );
        }
        other => panic!("expected CbError::Unsupported, got {other:?}"),
    }

    // Case 3 (the anti-over-rejection guard): the legal default still trains.
    let params = e02_params(ECtrType::Borders, ECtrType::Borders);
    let result = crate::train_cat(
        &cb_backend::CpuBackend,
        &[],
        &[],
        &cat_cols,
        &y,
        &w,
        &params,
        None,
    );
    assert!(
        result.is_ok(),
        "a CPU-legal CTR type must be unaffected, got {result:?}"
    );
}
