//! `sampling_unit` / `sampling_frequency`: what this engine implements, and what it
//! refuses — asserted, so neither the acceptance nor the refusal is an accident.
//!
//! Both are string-valued CatBoost parameters and both have exactly ONE value this
//! engine implements:
//!
//! * `sampling_unit = Object` — what [`cb_train::bootstrap`] already does (a keep-mask
//!   or weight per OBJECT). `Group` would draw once per query group and keep or drop
//!   it whole, which needs group spans the sampler has no notion of.
//! * `sampling_frequency = PerTree` — this engine draws once per tree, before the
//!   level loop, which IS upstream's `PerTree`. `PerTreeLevel` redraws at every level.
//!
//! The interesting case is the THIRD one, and it is why `PerTreeLevel` is not refused
//! outright: under `bootstrap_type = No` there is no draw at all, so the two
//! frequencies are BIT-IDENTICAL and the value is accepted. That was measured against
//! catboost 1.2.10, not assumed — `max |diff| = 0` at depths 1, 2 and 4 — and refusing
//! an inert value would reject configurations this engine reproduces exactly.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_backend::CpuBackend;
use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_core::CbError;
use cb_train::{
    train, BoostParams, EBootstrapType, EBoostingType, EGrowPolicy, EOverfittingDetectorType,
    ESamplingFrequency, ESamplingUnit, ExtraBoostParams,
};

/// A deterministic float corpus with enough spread to give every feature borders.
fn corpus() -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = 240;
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut cols: Vec<Vec<f32>> = vec![Vec::with_capacity(n); 3];
    let mut target = Vec::with_capacity(n);
    for _ in 0..n {
        let (a, b, c) = (next(), next(), next());
        cols[0].push(a as f32);
        cols[1].push(b as f32);
        cols[2].push(c as f32);
        target.push(2.0 * a - b + 0.5 * c);
    }
    let borders: Vec<Vec<f64>> = cols
        .iter()
        .map(|c| cb_data::select_borders_greedy_logsum_f32(c, 32, false))
        .collect();
    let weights = vec![1.0_f64; n];
    (cols, borders, target, weights)
}

fn params(bootstrap_type: EBootstrapType, extra: ExtraBoostParams) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 5,
        depth: 3,
        learning_rate: 0.3,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type,
        subsample: if matches!(bootstrap_type, EBootstrapType::Bernoulli | EBootstrapType::Mvs) {
            0.7
        } else {
            1.0
        },
        bagging_temperature: if matches!(bootstrap_type, EBootstrapType::Bayesian) {
            1.0
        } else {
            0.0
        },
        random_seed: 0,
        od_type: EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        auto_learning_rate: false,
        one_hot_max_size: cb_train::one_hot_max_size_default(),
        permutation_count: cb_train::permutation_count_default(),
        fold_len_multiplier: cb_train::fold_len_multiplier_default(),
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: EBoostingType::Plain,
        max_ctr_complexity: cb_train::max_ctr_complexity_default(),
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: EScoreFunction::L2,
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
        grow_policy: EGrowPolicy::SymmetricTree,
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
        extra,
    }
}

fn leaves(p: &BoostParams) -> Vec<f64> {
    let (cols, borders, target, weights) = corpus();
    let model = train(&CpuBackend, &cols, &borders, &target, &weights, p, None)
        .expect("fit must succeed");
    model
        .oblivious_trees
        .iter()
        .flat_map(|t| t.leaf_values.iter().copied())
        .collect()
}

/// Both defaults must leave the boosting loop BYTE-IDENTICAL — the standing
/// inert-at-default discipline for every parameter this wave added.
#[test]
fn sampling_defaults_are_inert() {
    let base = leaves(&params(EBootstrapType::Bernoulli, ExtraBoostParams::default()));
    let explicit = leaves(&params(
        EBootstrapType::Bernoulli,
        ExtraBoostParams {
            sampling_unit: ESamplingUnit::Object,
            sampling_frequency: ESamplingFrequency::PerTree,
            ..Default::default()
        },
    ));
    assert_eq!(
        base, explicit,
        "setting sampling_unit/sampling_frequency to their defaults must not change the model"
    );
}

/// `PerTreeLevel` is ACCEPTED under `bootstrap_type = No`, and produces the same model
/// as `PerTree`.
///
/// This is the measured fact the whole guard turns on: with no draw there is no sample
/// to redraw, so the frequency cannot matter. catboost 1.2.10 agrees exactly —
/// `max |diff| = 0` between the two frequencies at depths 1, 2 and 4 under
/// `bootstrap_type=No`. Refusing an inert value would reject configurations this
/// engine reproduces bit-for-bit.
#[test]
fn sampling_frequency_is_inert_without_a_draw() {
    let per_tree = leaves(&params(
        EBootstrapType::No,
        ExtraBoostParams {
            sampling_frequency: ESamplingFrequency::PerTree,
            ..Default::default()
        },
    ));
    let per_level = leaves(&params(
        EBootstrapType::No,
        ExtraBoostParams {
            sampling_frequency: ESamplingFrequency::PerTreeLevel,
            ..Default::default()
        },
    ));
    assert_eq!(
        per_tree, per_level,
        "under bootstrap_type=No there is NO draw, so the two sampling frequencies must \
         produce identical models — catboost 1.2.10 gives max|diff| = 0 here too"
    );
    // Not vacuous: the fit must actually have produced leaves to compare.
    assert!(
        !per_tree.is_empty() && per_tree.iter().any(|v| *v != 0.0),
        "the inert comparison is vacuous unless the fit produced non-trivial leaves"
    );
}

/// `PerTreeLevel` is REFUSED once the sampler actually draws — for every drawing
/// bootstrap type, not just the one that happened to be tested.
#[test]
fn per_tree_level_is_refused_for_every_drawing_sampler() {
    for bt in [
        EBootstrapType::Bernoulli,
        EBootstrapType::Bayesian,
        EBootstrapType::Mvs,
    ] {
        let p = params(
            bt,
            ExtraBoostParams {
                sampling_frequency: ESamplingFrequency::PerTreeLevel,
                ..Default::default()
            },
        );
        let (cols, borders, target, weights) = corpus();
        let err = train(&CpuBackend, &cols, &borders, &target, &weights, &p, None)
            .expect_err("PerTreeLevel with a drawing sampler must be refused");
        assert!(
            matches!(err, CbError::Unsupported(_)),
            "{bt:?}: expected Unsupported, got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("sampling_frequency") && msg.contains("PerTree"),
            "{bt:?}: the refusal must name the parameter and the value to use instead, got: {msg}"
        );
    }
}

/// `sampling_unit = Group` is REFUSED, and the message points at the reason rather
/// than just the name.
#[test]
fn group_sampling_unit_is_refused() {
    let p = params(
        EBootstrapType::Bernoulli,
        ExtraBoostParams {
            sampling_unit: ESamplingUnit::Group,
            ..Default::default()
        },
    );
    let (cols, borders, target, weights) = corpus();
    let err = train(&CpuBackend, &cols, &borders, &target, &weights, &p, None)
        .expect_err("sampling_unit=Group must be refused");
    assert!(matches!(err, CbError::Unsupported(_)), "got {err:?}");
    assert!(
        err.to_string().contains("sampling_unit"),
        "the refusal must name the parameter, got: {err}"
    );
}

/// Round-trip every spelling. A parser that silently accepted an unknown value would
/// let a typo train a DIFFERENT configuration than the caller wrote.
#[test]
fn spellings_round_trip_and_reject_unknowns() {
    for u in ESamplingUnit::all() {
        assert_eq!(ESamplingUnit::parse(u.as_str()), Some(u));
    }
    for f in ESamplingFrequency::all() {
        assert_eq!(ESamplingFrequency::parse(f.as_str()), Some(f));
    }
    // Case-sensitive, exactly like upstream's enum parser.
    assert_eq!(ESamplingUnit::parse("object"), None);
    assert_eq!(ESamplingUnit::parse("Groups"), None);
    assert_eq!(ESamplingFrequency::parse("pertree"), None);
    assert_eq!(ESamplingFrequency::parse("PerLevel"), None);
}
