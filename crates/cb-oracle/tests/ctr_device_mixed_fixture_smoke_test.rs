//! GDC-06/T13 smoke gate: the frozen mixed float+cat CTR fixture loads and is
//! device-shaped — 2 float columns with the FULL 15-border set (16 bins, matching
//! the 16 CTR buckets `ctr_covered` requires), 1 categorical column, 64 rows.
//! The permutation-divergence guard is asserted cb-train-side (T15), where the
//! real structure/averaging permutations are computable.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_oracle::load_f64_vec;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("ctr_device_mixed")
        .join(rel)
}

#[test]
fn ctr_device_mixed_fixture_is_device_shaped() {
    let x: Array2<f32> = read_npy(fixture("X.npy")).expect("X.npy loads as f32 [N,F]");
    assert_eq!(x.nrows(), 64);
    assert_eq!(x.ncols(), 2, "two float columns — the device n_features axis");

    let cat: Array1<i32> = read_npy(fixture("X_cat.npy")).expect("X_cat.npy loads as i32 [N]");
    assert_eq!(cat.len(), 64);
    let card = cat.iter().collect::<std::collections::BTreeSet<_>>().len();
    assert!(card > 2, "cat column must be CTR-routed (cardinality {card} > 2)");

    let borders: Array2<f64> = read_npy(fixture("borders.npy")).expect("borders.npy [2,15]");
    assert_eq!(borders.nrows(), 2);
    assert_eq!(
        borders.ncols(),
        15,
        "the FULL 15-border set — 16 float bins must equal the 16 CTR buckets"
    );
    for row in borders.rows() {
        assert!(
            row.windows(2).into_iter().all(|w| w[0] < w[1]),
            "borders must be strictly ascending"
        );
    }

    let y = load_f64_vec(&fixture("y.npy")).unwrap();
    assert_eq!(y.len(), 64);
    assert!(y.iter().all(|&v| v == 0.0 || v == 1.0), "binclf labels");

    let predictions = load_f64_vec(&fixture("predictions.npy")).unwrap();
    assert_eq!(predictions.len(), 64);
    let mean = predictions.iter().sum::<f64>() / 64.0;
    let var = predictions.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / 64.0;
    assert!(var.sqrt() > 1e-6, "degenerate constant predictions");
}
