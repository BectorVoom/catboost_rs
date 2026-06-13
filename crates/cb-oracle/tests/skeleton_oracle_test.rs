//! Walking-skeleton end-to-end proof (INFRA-03 read/compare side): read the ONE
//! committed `.npy` fixture via the public API and assert it matches the
//! reference vector at absolute error <= 1e-5.
//!
//! Integration tests are separate compilation units, so the restriction-lint
//! exemption is the non-cfg `#![allow(...)]` form (Pitfall 1).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::Path;

/// Reference values — identical to those written by `src/bin/write_skeleton.rs`.
const SKELETON_VALUES: [f64; 5] = [0.0, 0.25, -1.5, 3.14159, 2.71828];

#[test]
fn skeleton_fixture_matches_reference_at_1e_5() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/skeleton/predictions.npy");

    let actual = cb_oracle::load_f64_vec(&path).expect("read committed skeleton predictions.npy");
    let expected = SKELETON_VALUES.to_vec();

    cb_oracle::assert_abs_close(&expected, &actual, 1e-5)
        .expect("skeleton oracle comparison must pass at 1e-5");
}
