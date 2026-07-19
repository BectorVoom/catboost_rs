//! FSTR-02 facade integration: `catboost_rs::Model::feature_importance_with_data`
//! for `LossFunctionChange` maps the model's trained loss name to the matching
//! Min-optimized `EvalMetric` final-error closure, delegates to
//! `cb_model::loss_function_change`, and rejects out-of-scope losses with a typed
//! `CatBoostError::UnsupportedLoss` (never a silent Logloss fallback). Parity is
//! checked against the committed upstream `catboost==1.2.10` per-loss LFC fixtures
//! (`cb-oracle/fixtures/fstr_loss_change/`) through the PUBLISHED facade.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use catboost_rs::{CatBoostError, FeatureImportanceType, Model, OwnedColumns, Pool};
use cb_data::ingest::IngestSource;
use cb_oracle::load_f64_vec;
use ndarray::Array2;
use ndarray_npy::read_npy;

const TOL: f64 = 1e-5;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// A `Pool` built from a per-loss `{tag}_X.npy` (float columns) + `{tag}_y.npy`
/// (the raw regression labels LossFunctionChange requires).
fn regression_pool(tag: &str) -> Pool {
    let x: Array2<f64> = read_npy(fixture(&format!("fstr_loss_change/{tag}_X.npy")))
        .unwrap_or_else(|e| panic!("{tag}_X.npy loads: {e:?}"));
    let float_features: Vec<Vec<f64>> = (0..x.ncols())
        .map(|fi| x.column(fi).iter().copied().collect())
        .collect();
    let label = load_f64_vec(&fixture(&format!("fstr_loss_change/{tag}_y.npy")))
        .unwrap_or_else(|e| panic!("{tag}_y.npy loads: {e:?}"));
    OwnedColumns::new(float_features, label)
        .into_pool()
        .unwrap_or_else(|e| panic!("{tag} pool builds: {e:?}"))
}

fn regression_model(tag: &str) -> Model {
    Model::load_cbm(&fixture(&format!("fstr_loss_change/{tag}_model.cbm")))
        .unwrap_or_else(|e| panic!("{tag}_model.cbm loads: {e:?}"))
}

/// Reproduce `{tag}_loss_function_change.npy` through the facade with the given
/// `loss` name; assert per-feature parity <= 1e-5.
fn assert_facade_lfc(tag: &str, loss: &str) {
    let model = regression_model(tag);
    let pool = regression_pool(tag);
    let scores = model
        .feature_importance_with_data(FeatureImportanceType::LossFunctionChange, &pool, loss)
        .unwrap_or_else(|e| panic!("{tag} facade LFC ok: {e:?}"));
    let got: Vec<f64> = scores.iter().map(|&(_, _, s)| s).collect();
    let want = load_f64_vec(&fixture(&format!("fstr_loss_change/{tag}_loss_function_change.npy")))
        .unwrap_or_else(|e| panic!("{tag}_loss_function_change.npy loads: {e:?}"));
    assert_eq!(got.len(), want.len(), "{tag} facade LFC length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= TOL,
            "{tag} facade LFC[{i}] diverges: got {g}, want {w} (|d|={})",
            (g - w).abs()
        );
    }
}

/// FL-03 / FL-04a+b: the facade routes the FULL Min-optimized numeric loss set
/// `{RMSE, MAE, MAPE, Quantile}` to the matching `EvalMetric` closure and matches
/// upstream `get_feature_importance('LossFunctionChange')` within 1e-5. (The
/// `Logloss` binary path is covered by cb-model's `fstr_oracle_test`.) Each call
/// is also the flat-routing success probe — `LossFunctionChange` reaches the
/// generalized `loss_function_change`, not a rejected arm.
#[test]
fn loss_change_rmse_facade() {
    assert_facade_lfc("rmse", "RMSE");
    // Case-insensitive name handling.
    assert_facade_lfc("rmse", "rmse");
}

#[test]
fn loss_change_mae_mape_quantile_facade() {
    assert_facade_lfc("mae", "MAE");
    assert_facade_lfc("mape", "MAPE");
    // Quantile default alpha 0.5 (matches the fixture's trained alpha).
    assert_facade_lfc("quantile", "Quantile");
    assert_facade_lfc("quantile", "Quantile:alpha=0.5");
}

/// FL-03 (review [3]): a cosmetically whitespaced loss name (`" RMSE "`) is
/// accepted — the allow-list base is trimmed just like `cb_train::parse_metric`,
/// so it is NOT spuriously rejected. The result matches the untrimmed name.
#[test]
fn loss_change_accepts_whitespaced_name() {
    let model = regression_model("rmse");
    let pool = regression_pool("rmse");
    let tidy = model
        .feature_importance_with_data(FeatureImportanceType::LossFunctionChange, &pool, "RMSE")
        .expect("RMSE ok");
    let spaced = model
        .feature_importance_with_data(FeatureImportanceType::LossFunctionChange, &pool, " RMSE ")
        .expect("whitespaced RMSE must be accepted, not UnsupportedLoss");
    assert_eq!(tidy, spaced);
}

/// FL-03 (review [1]): a malformed *param* on a supported base name is rejected
/// as `UnsupportedLoss`, but the message carries the underlying parse reason so
/// the caller sees the loss IS supported and only the param is wrong.
#[test]
fn loss_change_bad_param_message_carries_parse_reason() {
    let model = regression_model("rmse");
    let pool = regression_pool("rmse");
    match model.feature_importance_with_data(
        FeatureImportanceType::LossFunctionChange,
        &pool,
        "Quantile:beta=1",
    ) {
        Err(CatBoostError::UnsupportedLoss(m)) => {
            assert!(m.contains("Quantile:beta=1"), "must echo the descriptor: {m}");
            // The enriched message appends the parse_metric reason in parens,
            // distinguishing a bad-param from a genuinely unknown loss.
            assert!(m.contains('('), "must append the parse reason: {m}");
        }
        other => panic!("expected UnsupportedLoss with reason, got {other:?}"),
    }
}

/// FL-03: a Max-optimized / unknown loss is rejected with a typed
/// `UnsupportedLoss` error — NOT a silent (wrong) Logloss number. Also rejects
/// an out-of-scope-but-parseable metric (`MSLE`) and a malformed param on a
/// supported name.
#[test]
fn loss_change_rejects_max_metric() {
    let model = regression_model("rmse");
    let pool = regression_pool("rmse");
    for bad in ["AUC", "Accuracy", "R2", "NDCG", "MSLE", "not_a_loss", "quantile:beta=1"] {
        match model.feature_importance_with_data(
            FeatureImportanceType::LossFunctionChange,
            &pool,
            bad,
        ) {
            Err(CatBoostError::UnsupportedLoss(m)) => assert!(m.contains(bad)),
            other => panic!("expected UnsupportedLoss for `{bad}`, got {other:?}"),
        }
    }
}
