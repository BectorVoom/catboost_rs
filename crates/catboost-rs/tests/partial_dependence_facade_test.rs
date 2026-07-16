//! FSTR-03b facade integration: `catboost_rs::Model::partial_dependence` projects
//! a `Pool` to float columns, delegates to `cb_model::partial_dependence`, and maps
//! errors — reproducing the committed upstream `catboost==1.2.10` fixtures
//! (`cb-oracle/fixtures/partial_dependence/`) through the PUBLISHED facade.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use catboost_rs::{CatBoostError, Model, OwnedColumns, PdpError, Pool};
use cb_data::ingest::IngestSource;
use cb_oracle::{assert_abs_close, load_f64_vec};
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

/// A `Pool` built from `numeric_tiny/X.npy` as SoA `f64` float columns (no label
/// needed for partial dependence).
fn numeric_tiny_pool() -> Pool {
    let x: Array2<f64> =
        read_npy(fixture("inputs/numeric_tiny/X.npy")).expect("numeric_tiny/X.npy loads");
    let float_features: Vec<Vec<f64>> = (0..x.ncols())
        .map(|fi| x.column(fi).iter().copied().collect())
        .collect();
    let label = vec![0.0_f64; x.nrows()];
    OwnedColumns::new(float_features, label)
        .into_pool()
        .expect("numeric_tiny pool builds")
}

fn pdp_model() -> Model {
    Model::load_cbm(&fixture("partial_dependence/model.cbm")).expect("model.cbm loads")
}

/// FAC-01: the facade single- and two-feature curves reproduce the committed
/// upstream fixtures within 1e-5, and echo the requested feature indices.
#[test]
fn facade_single_and_pair_match_fixture() {
    let model = pdp_model();
    let pool = numeric_tiny_pool();

    let single = model
        .partial_dependence(&pool, &[3])
        .expect("single-feature PD ok");
    assert_eq!(single.features, vec![3]);
    assert_eq!(single.values.len(), single.grids[0].len());
    let exp_single = load_f64_vec(&fixture("partial_dependence/pdp_single_values.npy"))
        .expect("pdp_single_values.npy loads");
    assert_abs_close(&exp_single, &single.values, TOL).expect("facade single within TOL");

    let pair = model
        .partial_dependence(&pool, &[0, 3])
        .expect("pair PD ok");
    assert_eq!(pair.features, vec![0, 3]);
    assert_eq!(pair.values.len(), pair.grids[0].len() * pair.grids[1].len());
    let exp_pair = load_f64_vec(&fixture("partial_dependence/pdp_pair_values.npy"))
        .expect("pdp_pair_values.npy loads");
    assert_abs_close(&exp_pair, &pair.values, TOL).expect("facade pair within TOL");
}

/// FAC-02: the facade maps a wrong-width pool to `FeatureMismatch` and an invalid
/// `features` request to `PartialDependence(PdpError::…)`.
#[test]
fn facade_maps_errors() {
    let model = pdp_model(); // 4 float features
    let pool = numeric_tiny_pool();

    // Wrong-width pool: 2 float columns, model expects 4 -> FeatureMismatch
    // (raised by feature_columns before the PD call).
    let narrow = OwnedColumns::new(vec![vec![0.0_f64; 3], vec![1.0_f64; 3]], vec![0.0; 3])
        .into_pool()
        .expect("narrow pool builds");
    match model.partial_dependence(&narrow, &[0]) {
        Err(CatBoostError::FeatureMismatch(_)) => {}
        other => panic!("expected FeatureMismatch, got {other:?}"),
    }

    // Duplicate feature -> PartialDependence(DuplicateFeature).
    match model.partial_dependence(&pool, &[1, 1]) {
        Err(CatBoostError::PartialDependence(PdpError::DuplicateFeature { index })) => {
            assert_eq!(index, 1);
        }
        other => panic!("expected PartialDependence(DuplicateFeature), got {other:?}"),
    }

    // Out-of-range feature -> PartialDependence(FeatureIndexOutOfRange).
    match model.partial_dependence(&pool, &[99]) {
        Err(CatBoostError::PartialDependence(PdpError::FeatureIndexOutOfRange {
            index,
            n_float,
        })) => {
            assert_eq!(index, 99);
            assert_eq!(n_float, 4);
        }
        other => panic!("expected PartialDependence(FeatureIndexOutOfRange), got {other:?}"),
    }
}
