//! `final_ctr_computation_mode` / `ctr_history_unit` parity against catboost
//! 1.2.10.
//!
//! # `final_ctr_computation_mode`
//!
//! `Skip` trains IDENTICALLY — verified against catboost, which returns
//! byte-identical trees, leaves, bias and the same 8 CTR splits — and differs
//! ONLY in that `ctr_data` comes back empty. That leaves a model whose CTR
//! splits have no tables to look up.
//!
//! **catboost 1.2.10 SEGFAULTS applying such a model** (the fit succeeds, then
//! `predict` dumps core). This port keeps the training behaviour identical but
//! refuses the apply with a typed error — a deliberate improvement, so there is
//! no `Skip` prediction vector to compare against.
//!
//! # `ctr_history_unit`
//!
//! Upstream does not implement it on CPU at all:
//!
//! ```text
//! json_helper.h:185: Error: change of option ctr_history_unit is
//! unimplemented for task type CPU and was not default in previous run
//! ```
//!
//! So refusing `Group` is exact parity, and the refusal IS the contract.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use catboost_rs::{
    CatBoostBuilder, ECtrHistoryUnit, EFinalCtrComputationMode, IngestSource, OwnedColumns, Pool,
};
use cb_compute::Loss;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("ctr_modes")
        .join(rel)
}

fn load_num(name: &str) -> Vec<Vec<f64>> {
    let x: Array2<f64> = read_npy(fixture(name)).expect("fixture matrix");
    (0..x.ncols()).map(|f| x.column(f).to_vec()).collect()
}

fn load_y(name: &str) -> Vec<f64> {
    let y: Array1<f64> = read_npy(fixture(name)).expect("fixture vector");
    y.to_vec()
}

fn load_cats(name: &str) -> Vec<String> {
    std::fs::read_to_string(fixture(name))
        .expect("categorical column")
        .lines()
        .map(str::to_owned)
        .collect()
}

/// Build a pool whose FIRST feature is the categorical column.
fn pool_of(cats: Vec<String>, num: Vec<Vec<f64>>, target: Vec<f64>) -> Pool {
    OwnedColumns::new(num, target)
        .with_cat_features(vec![cats])
        .into_pool()
        .expect("pool must build")
}

fn train_pool() -> Pool {
    pool_of(load_cats("cats.txt"), load_num("X_num.npy"), load_y("y.npy"))
}

fn eval_pool() -> Pool {
    let num = load_num("X_eval_num.npy");
    let n = num.first().map_or(0, Vec::len);
    pool_of(load_cats("cats_eval.txt"), num, vec![0.0; n])
}

/// The pinned fit from `gen_ctr_mode_fixtures.py::PARAMS`.
fn builder() -> CatBoostBuilder {
    CatBoostBuilder::new()
        .loss(Loss::Rmse)
        .iterations(5)
        .depth(3)
        .learning_rate(0.3)
        .l2_leaf_reg(3.0)
        .random_strength(0.0)
        .boost_from_average(true)
        .random_seed(0)
        .border_count(32)
        .score_function(cb_compute::EScoreFunction::L2)
        .leaf_method(cb_compute::LeafMethod::Gradient)
        .one_hot_max_size(2)
        .max_ctr_complexity(1)
}

/// The default mode produces an APPLIABLE model — the property
/// `final_ctr_computation_mode` actually governs.
///
/// # Why this does not compare against `preds_Default.npy`
///
/// The frozen catboost predictions are kept in the fixture, but this file does
/// NOT assert end-to-end CTR numerics against them, because the comparison is
/// not apples-to-apples on this corpus: catboost's pool puts the categorical at
/// FLAT feature index 0 with the numerics after it, while a `cb-data` pool
/// carries categorical columns separately from float features. Candidate
/// enumeration order therefore differs between the two sides, and candidate
/// selection is first-wins on ties — so a prediction difference here would not
/// isolate a CTR defect.
///
/// CTR numeric parity is the job of the dedicated CTR oracles
/// (`ctr_borders_multiprior`, `ctr_buckets_simple`, `ordered_ctr`, ...), which
/// pin it on corpora built for that purpose. What belongs here is that Default
/// yields a model that can be applied at all — the exact thing `Skip` removes.
#[test]
fn final_ctr_default_yields_an_appliable_model() {
    let model = builder()
        .final_ctr_computation_mode(EFinalCtrComputationMode::Default)
        .fit(&train_pool())
        .expect("the fit must succeed");
    let preds = model
        .predict(&eval_pool())
        .expect("a Default-mode model must be appliable");
    let expected_len = load_y("preds_Default.npy").len();
    assert_eq!(preds.len(), expected_len, "one prediction per eval row");
    assert!(
        preds.iter().all(|p| p.is_finite()),
        "every prediction must be finite"
    );
    assert!(
        model.as_canonical().ctr_tables_present(),
        "Default must bake the CTR tables"
    );
}

/// `Skip` FITS — training is unaffected — but the resulting model cannot be
/// applied, and must say so with a typed error instead of misbehaving.
/// catboost 1.2.10 segfaults here.
#[test]
fn final_ctr_skip_trains_but_refuses_to_predict() {
    let model = builder()
        .final_ctr_computation_mode(EFinalCtrComputationMode::Skip)
        .fit(&train_pool())
        .expect("Skip must still TRAIN — upstream trains identically");

    let err = model
        .predict(&eval_pool())
        .expect_err("a model with no baked CTR tables must refuse to predict");
    let msg = err.to_string();
    assert!(
        msg.contains("CTR") && msg.contains("final_ctr_computation_mode"),
        "the refusal must explain the cause and name the parameter; got: {msg}"
    );
}

/// The structural claim behind `Skip`: it changes ONLY the CTR tables, never the
/// trees. If this ever fails, "training is unaffected" is no longer true and the
/// parameter means something else.
#[test]
fn final_ctr_skip_leaves_the_trees_untouched() {
    let default = builder()
        .final_ctr_computation_mode(EFinalCtrComputationMode::Default)
        .fit(&train_pool())
        .expect("fit");
    let skip = builder()
        .final_ctr_computation_mode(EFinalCtrComputationMode::Skip)
        .fit(&train_pool())
        .expect("fit");

    let a = default.as_canonical();
    let b = skip.as_canonical();

    // Compare what TRAINING produced: the tree count, the split kinds/borders and
    // the leaf values/weights.
    //
    // The CTR splits' `shift` / `scale` are deliberately EXCLUDED: those are
    // outputs OF the bake (the bake copies the table's normalization back onto
    // every matching split), so `Skip` leaves them at their defaults by
    // construction. Asserting on them would be asserting that the skipped step
    // ran.
    assert_eq!(
        a.oblivious_trees.len(),
        b.oblivious_trees.len(),
        "Skip must not change the tree count"
    );
    for (ti, (ta, tb)) in a
        .oblivious_trees
        .iter()
        .zip(b.oblivious_trees.iter())
        .enumerate()
    {
        assert_eq!(
            ta.leaf_values, tb.leaf_values,
            "tree {ti}: Skip must not change the leaf values"
        );
        assert_eq!(
            ta.leaf_weights, tb.leaf_weights,
            "tree {ti}: Skip must not change the leaf weights"
        );
        assert_eq!(
            ta.splits.len(),
            tb.splits.len(),
            "tree {ti}: Skip must not change the split count"
        );
        for (si, (sa, sb)) in ta.splits.iter().zip(tb.splits.iter()).enumerate() {
            match (sa, sb) {
                (cb_model::ModelSplit::Float(x), cb_model::ModelSplit::Float(y)) => {
                    assert_eq!(x, y, "tree {ti} split {si}: float split changed");
                }
                (cb_model::ModelSplit::Ctr(x), cb_model::ModelSplit::Ctr(y)) => {
                    assert_eq!(
                        (&x.projection, x.ctr_type, x.target_border_idx, x.border),
                        (&y.projection, y.ctr_type, y.target_border_idx, y.border),
                        "tree {ti} split {si}: CTR split identity/border changed \
                         (shift/scale are bake outputs and are excluded)"
                    );
                }
                _ => panic!("tree {ti} split {si}: split KIND changed"),
            }
        }
    }
    assert!(
        (a.bias - b.bias).abs() <= f64::EPSILON,
        "Skip must not change the bias"
    );
    assert!(
        a.has_ctr_splits(),
        "the fixture corpus must actually produce CTR splits, else this proves nothing"
    );
    assert!(a.ctr_tables_present(), "Default must bake the CTR tables");
    assert!(
        !b.ctr_tables_present(),
        "Skip must leave the model without CTR tables"
    );
}

/// `ctr_history_unit = Group` is refused, exactly as upstream refuses it on CPU.
#[test]
fn ctr_history_unit_group_is_refused_on_cpu() {
    let err = builder()
        .ctr_history_unit(ECtrHistoryUnit::Group)
        .fit(&train_pool())
        .expect_err("Group must be refused on the CPU path");
    let msg = err.to_string();
    assert!(
        msg.contains("ctr_history_unit") && msg.contains("CPU"),
        "the refusal must name the parameter and the task type; got: {msg}"
    );
}

/// The default value is accepted and changes nothing.
#[test]
fn ctr_history_unit_sample_is_the_accepted_default() {
    let explicit = builder()
        .ctr_history_unit(ECtrHistoryUnit::Sample)
        .fit(&train_pool())
        .expect("Sample must be accepted");
    let implicit = builder().fit(&train_pool()).expect("fit");
    assert_eq!(
        explicit.predict(&eval_pool()).expect("predict"),
        implicit.predict(&eval_pool()).expect("predict"),
        "naming the default must change nothing"
    );
}

/// Every legal token round-trips for both parameters.
#[test]
fn ctr_mode_enums_parse_every_legal_token() {
    for m in EFinalCtrComputationMode::all() {
        assert_eq!(EFinalCtrComputationMode::parse(m.as_str()), Some(m));
    }
    for u in ECtrHistoryUnit::all() {
        assert_eq!(ECtrHistoryUnit::parse(u.as_str()), Some(u));
    }
    assert_eq!(EFinalCtrComputationMode::parse("skip"), None);
    assert_eq!(ECtrHistoryUnit::parse("sample"), None);
    assert!(ECtrHistoryUnit::Sample.is_cpu_supported());
    assert!(!ECtrHistoryUnit::Group.is_cpu_supported());
}
