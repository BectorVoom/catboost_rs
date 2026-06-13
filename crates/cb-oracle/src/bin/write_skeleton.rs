//! One-off in-crate fixture generator (NO Python/numpy — numpy is not installed
//! this phase; the Python oracle env is Plan 03's job). Writes the skeleton
//! `predictions.npy` via `ndarray-npy::write_npy`, which round-trips f64
//! bit-exactly, so the committed file reads back identically.
//!
//! Run ONCE and commit the result:
//!   cargo run -p cb-oracle --bin write_skeleton
//!
//! A one-off generator binary may unwrap/panic, so the restriction lints are
//! allowed at file scope here.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::Path;

use ndarray::Array1;
use ndarray_npy::write_npy;

/// The canonical skeleton prediction values. These MUST stay in sync with the
/// reference vector built in `tests/skeleton_oracle_test.rs`.
const SKELETON_VALUES: [f64; 5] = [0.0, 0.25, -1.5, 3.14159, 2.71828];

fn main() {
    let out_dir = Path::new("crates/cb-oracle/fixtures/skeleton");
    std::fs::create_dir_all(out_dir).expect("create skeleton fixture dir");

    let arr = Array1::from(SKELETON_VALUES.to_vec());
    let out_path = out_dir.join("predictions.npy");
    write_npy(&out_path, &arr).expect("write predictions.npy");

    println!("wrote {} ({} values)", out_path.display(), SKELETON_VALUES.len());
}
