//! PARAM-03 — the class-weighting surface and `ignored_features` through the
//! PUBLISHED facade.
//!
//! # Why these needed no new engine code
//!
//! `cb_data::weights` already carried the upstream-faithful class-weight
//! computation (`balanced_class_weights` / `sqrt_balanced_class_weights` /
//! `resolve_object_weights`, a bit-faithful port of
//! `calc_class_weights.cpp`), and `cb-data`'s `weights_oracle_test` already
//! gates it against the frozen upstream `class_weights/` fixture. What did not
//! exist was any path that APPLIED the result during a fit: the resolved
//! per-object weights never reached `train`. PARAM-03 is that path.
//!
//! So these tests deliberately do NOT re-derive the weight VALUES (the oracle
//! test owns that). They assert the application: that the weights reach the
//! trainer and change the model, that the mutually-exclusive controls are
//! rejected together rather than silently resolved, and that a control paired
//! with a loss having no notion of "class" is refused.
//!
//! `ignored_features` IS new, and is implemented by emptying the feature's
//! border set — so the assertions below check both halves of that choice: the
//! feature stops being splittable, AND the model's feature indexing is unchanged
//! (predict still takes the full-width pool).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use catboost_rs::{
    AutoClassWeights, CatBoostBuilder, IngestSource, Loss, OwnedColumns, Pool,
};

const N: usize = 120;

/// An imbalanced binary pool: feature 0 predicts the label, feature 1 is noise,
/// and class 1 is a 1-in-4 minority (so reweighting it visibly moves the fit).
fn binary_pool() -> Pool {
    let f0: Vec<f64> = (0..N).map(|i| i as f64).collect();
    let f1: Vec<f64> = (0..N).map(|i| ((i * 7) % 13) as f64).collect();
    let y: Vec<f64> = (0..N).map(|i| f64::from(u8::from(i % 4 == 0))).collect();
    OwnedColumns::new(vec![f0, f1], y)
        .into_pool()
        .expect("binary pool must build")
}

/// A continuous-target pool (no per-object class exists).
fn regression_pool() -> Pool {
    let f0: Vec<f64> = (0..N).map(|i| i as f64).collect();
    let y: Vec<f64> = f0.iter().map(|v| v * 0.5 + 1.0).collect();
    OwnedColumns::new(vec![f0], y)
        .into_pool()
        .expect("regression pool must build")
}

fn clf(iterations: usize) -> CatBoostBuilder {
    CatBoostBuilder::new()
        .loss(Loss::Logloss)
        .iterations(iterations)
        .depth(3)
        .learning_rate(0.2)
}

/// Explicit `class_weights` reach the trainer and CHANGE the model. An accepted
/// but unapplied weight vector is the whole failure this parameter had before
/// PARAM-03 (the computation existed; nothing consumed it).
#[test]
fn explicit_class_weights_change_the_model() {
    let pool = binary_pool();
    let base = clf(20).fit(&pool).expect("base fit");
    let weighted = clf(20)
        .class_weights(vec![1.0, 5.0])
        .fit(&pool)
        .expect("weighted fit");

    assert_ne!(
        base.predict(&pool).expect("predict"),
        weighted.predict(&pool).expect("predict"),
        "class_weights must reach the trainer"
    );
}

/// `class_weights = [1.0, 1.0]` is the IDENTITY: it is "active" as a parameter
/// but multiplies every weight by one, so the model must be unchanged.
///
/// This separates "the parameter is applied" from "the parameter perturbs
/// something": an implementation that, say, replaced the pool weights with the
/// class weights instead of MULTIPLYING would pass the test above and fail here.
#[test]
fn unit_class_weights_are_the_identity() {
    let pool = binary_pool();
    let base = clf(20).fit(&pool).expect("base fit");
    let unit = clf(20)
        .class_weights(vec![1.0, 1.0])
        .fit(&pool)
        .expect("unit-weight fit");

    assert_eq!(
        base.predict(&pool).expect("predict"),
        unit.predict(&pool).expect("predict"),
        "all-ones class weights must not change the model"
    );
}

/// `scale_pos_weight = w` is EXACTLY `class_weights = [1.0, w]` — asserted as an
/// equality between the two spellings rather than as two independent behaviours.
#[test]
fn scale_pos_weight_equals_the_two_element_class_weight_vector() {
    let pool = binary_pool();
    let scaled = clf(20)
        .scale_pos_weight(4.0)
        .fit(&pool)
        .expect("scale_pos_weight fit");
    let explicit = clf(20)
        .class_weights(vec![1.0, 4.0])
        .fit(&pool)
        .expect("class_weights fit");

    assert_eq!(
        scaled.predict(&pool).expect("predict"),
        explicit.predict(&pool).expect("predict"),
        "scale_pos_weight=w must be exactly class_weights=[1, w]"
    );
}

/// Both auto schemes are applied and DIFFER from each other — Balanced uses
/// `max/w`, SqrtBalanced `sqrt(max/w)`, so a facade that wired one enum arm to
/// both computations would be caught here.
#[test]
fn the_two_auto_schemes_are_applied_and_differ() {
    let pool = binary_pool();
    let base = clf(20).fit(&pool).expect("base fit");
    let balanced = clf(20)
        .auto_class_weights(AutoClassWeights::Balanced)
        .fit(&pool)
        .expect("balanced fit");
    let sqrt = clf(20)
        .auto_class_weights(AutoClassWeights::SqrtBalanced)
        .fit(&pool)
        .expect("sqrt fit");

    let b = base.predict(&pool).expect("predict");
    let bal = balanced.predict(&pool).expect("predict");
    let sq = sqrt.predict(&pool).expect("predict");
    assert_ne!(b, bal, "Balanced must change the model");
    assert_ne!(b, sq, "SqrtBalanced must change the model");
    assert_ne!(bal, sq, "the two schemes must not resolve to the same weights");
}

/// `AutoClassWeights::None` is the default and must be a true no-op.
#[test]
fn auto_class_weights_none_is_the_default_no_op() {
    let pool = binary_pool();
    let base = clf(20).fit(&pool).expect("base fit");
    let explicit_none = clf(20)
        .auto_class_weights(AutoClassWeights::None)
        .fit(&pool)
        .expect("None fit");
    assert_eq!(
        base.predict(&pool).expect("predict"),
        explicit_none.predict(&pool).expect("predict"),
    );
}

/// The three controls all write the SAME per-object weight, so combining them is
/// rejected rather than resolved by precedence (which would silently discard a
/// value the caller set on purpose).
#[test]
fn the_three_class_weight_controls_are_mutually_exclusive() {
    let pool = binary_pool();
    let cases: Vec<CatBoostBuilder> = vec![
        clf(5)
            .class_weights(vec![1.0, 2.0])
            .auto_class_weights(AutoClassWeights::Balanced),
        clf(5).class_weights(vec![1.0, 2.0]).scale_pos_weight(3.0),
        clf(5)
            .auto_class_weights(AutoClassWeights::Balanced)
            .scale_pos_weight(3.0),
    ];
    for builder in cases {
        let err = builder
            .fit(&pool)
            .expect_err("combining the class-weight controls must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("at most one"),
            "the error must say only one may be set, got: {msg}"
        );
    }
}

/// A class-weight control on a REGRESSION loss is refused: there is no per-object
/// class to weight, so applying it would be meaningless and ignoring it would be
/// silent.
#[test]
fn a_class_weight_control_with_a_regression_loss_is_rejected() {
    let pool = regression_pool();
    let err = CatBoostBuilder::new()
        .loss(Loss::Rmse)
        .iterations(5)
        .class_weights(vec![1.0, 2.0])
        .fit(&pool)
        .expect_err("class weights need a classification loss");
    let msg = err.to_string();
    assert!(msg.contains("classification"), "got: {msg}");
}

/// A non-integer label is refused rather than truncated by `as usize`, which
/// would silently bucket 0.7 and 0.2 into the same class.
#[test]
fn a_non_integer_class_label_is_rejected() {
    let f0: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let y: Vec<f64> = vec![0.0, 0.7, 1.0, 0.2, 0.0, 1.0, 0.5, 1.0];
    let pool = OwnedColumns::new(vec![f0], y)
        .into_pool()
        .expect("pool must build");
    let err = CatBoostBuilder::new()
        .loss(Loss::CrossEntropy)
        .iterations(5)
        .class_weights(vec![1.0, 2.0])
        .fit(&pool)
        .expect_err("a probabilistic target has no class to weight");
    assert!(err.to_string().contains("integer class labels"), "got: {err}");
}

/// A label class with no matching weight is refused rather than read
/// out-of-range.
#[test]
fn too_few_class_weights_for_the_observed_classes_is_rejected() {
    let pool = binary_pool();
    let err = clf(5)
        .class_weights(vec![1.0])
        .fit(&pool)
        .expect_err("2 classes need 2 weights");
    assert!(err.to_string().contains("2 classes"), "got: {err}");
}

// ─── ignored_features ────────────────────────────────────────────────────────

/// An ignored feature stops being splittable — asserted through the MODEL, not
/// through the predictions: the model must contain no split on that feature.
#[test]
fn an_ignored_feature_is_never_split_on() {
    let pool = binary_pool();
    let model = clf(25)
        .ignored_features(vec![0])
        .fit(&pool)
        .expect("fit with an ignored feature");

    let splits_on_zero = model
        .as_canonical()
        .oblivious_trees
        .iter()
        .flat_map(|t| t.splits.iter())
        .filter(|s| matches!(s, cb_model::ModelSplit::Float(f) if f.feature == 0))
        .count();
    assert_eq!(
        splits_on_zero, 0,
        "feature 0 was ignored, so no tree may split on it"
    );

    // The control: WITHOUT the parameter, feature 0 is the predictive one and IS
    // chosen — so the assertion above is not vacuously true.
    let unrestricted = clf(25).fit(&pool).expect("control fit");
    let control_splits = unrestricted
        .as_canonical()
        .oblivious_trees
        .iter()
        .flat_map(|t| t.splits.iter())
        .filter(|s| matches!(s, cb_model::ModelSplit::Float(f) if f.feature == 0))
        .count();
    assert!(
        control_splits > 0,
        "the control must actually split on feature 0, else the test proves nothing"
    );
}

/// Ignoring a feature does NOT renumber the others: the model still expects the
/// full-width pool, and predict works unchanged. This is the property that
/// motivated emptying the borders instead of dropping the column.
#[test]
fn ignoring_a_feature_preserves_the_model_feature_width() {
    let pool = binary_pool();
    let model = clf(10)
        .ignored_features(vec![0])
        .fit(&pool)
        .expect("fit with an ignored feature");
    assert_eq!(
        model.n_float_features(),
        2,
        "the ignored feature keeps its index; the model stays full-width"
    );
    model
        .predict(&pool)
        .expect("predict must still accept the full-width pool");
}

/// An empty `ignored_features` is a true no-op.
#[test]
fn an_empty_ignored_features_list_changes_nothing() {
    let pool = binary_pool();
    let base = clf(15).fit(&pool).expect("base fit");
    let empty = clf(15)
        .ignored_features(Vec::new())
        .fit(&pool)
        .expect("empty-list fit");
    assert_eq!(
        base.predict(&pool).expect("predict"),
        empty.predict(&pool).expect("predict"),
    );
}

/// An out-of-range index is rejected, not silently ignored — a typo that ignores
/// nothing defeats the purpose of the parameter.
#[test]
fn an_out_of_range_ignored_feature_index_is_rejected() {
    let pool = binary_pool();
    let err = clf(5)
        .ignored_features(vec![7])
        .fit(&pool)
        .expect_err("index 7 does not exist in a 2-feature pool");
    let msg = err.to_string();
    assert!(
        msg.contains("out of range") && msg.contains('7'),
        "got: {msg}"
    );
}
