//! FSTR-03 partial-dependence oracle (PDP-03 / PDP-04): the single- and
//! two-feature partial-dependence curves reproduce upstream `catboost==1.2.10`
//! `plot_partial_dependence(pool, features, plot=False)[0]`
//! (== `_calc_partial_dependence`, one averaged value per BIN) within `<= 1e-5`.
//!
//! Fixtures are generated OFFLINE by
//! `crates/cb-oracle/fixtures/partial_dependence/gen_fixtures.py` (numeric-only
//! `CatBoostRegressor`, pinned ISOLATING params, `thread_count=1`). single_feature
//! = 3 (3 borders -> 4 bins); pair = [0, 3] (C-order row-major, feature 0 outer,
//! feature 3 inner). Values are upstream's, never hand-computed.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`; the
//! top-line `#![allow(...)]` mirrors the other cb-model oracle tests.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_model::{load_cbm, partial_dependence, Model};
use cb_oracle::{assert_abs_close, load_f64_vec};
use ndarray::Array2;
use ndarray_npy::read_npy;

const TOL: f64 = 1e-5;

/// Resolve a path under `cb-oracle/fixtures/` from cb-model's manifest dir.
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// The pinned partial-dependence model, loaded from the committed upstream `.cbm`.
fn pdp_model() -> Model {
    load_cbm(&fixture("partial_dependence/model.cbm")).expect("partial_dependence/model.cbm loads")
}

/// `numeric_tiny` X as per-feature `f32` SoA columns — the exact layout
/// `partial_dependence` / `predict_raw` consume.
fn numeric_tiny_columns() -> Vec<Vec<f32>> {
    let x: Array2<f64> =
        read_npy(fixture("inputs/numeric_tiny/X.npy")).expect("numeric_tiny/X.npy loads");
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

/// PDP-03 (AT-03a/b): single-feature curve == upstream, and grid/meta shape.
#[test]
fn single_feature_pdp_matches_upstream() {
    let model = pdp_model();
    let cols = numeric_tiny_columns();

    let pd = partial_dependence(&model, &cols, &[3]).expect("single-feature PD ok");

    // AT-03b: feature echoed; one grid; values length == grid length (n_bins).
    assert_eq!(pd.features, vec![3]);
    assert_eq!(pd.grids.len(), 1);
    assert_eq!(pd.values.len(), pd.grids[0].len());

    // AT-03a: values match upstream per-bin averages within 1e-5.
    let expected = load_f64_vec(&fixture("partial_dependence/pdp_single_values.npy"))
        .expect("pdp_single_values.npy loads");
    assert_abs_close(&expected, &pd.values, TOL).expect("single-feature PD within TOL");
}

/// PDP-04 (AT-04a/b): two-feature surface == upstream (C-order row-major),
/// and the row-major length invariant.
#[test]
fn pair_feature_pdp_matches_upstream() {
    let model = pdp_model();
    let cols = numeric_tiny_columns();

    let pd = partial_dependence(&model, &cols, &[0, 3]).expect("pair PD ok");

    // AT-04b: features echoed; two grids; row-major length = |g0| * |g1|.
    assert_eq!(pd.features, vec![0, 3]);
    assert_eq!(pd.grids.len(), 2);
    assert_eq!(pd.values.len(), pd.grids[0].len() * pd.grids[1].len());

    // AT-04a: values match upstream 2-D surface, C-order (feature 0 outer,
    // feature 3 inner), within 1e-5.
    let expected = load_f64_vec(&fixture("partial_dependence/pdp_pair_values.npy"))
        .expect("pdp_pair_values.npy loads");
    assert_abs_close(&expected, &pd.values, TOL).expect("pair PD within TOL");
}
