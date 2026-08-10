//! GDC-09 (T10): `DeviceCtrConfig` carries an OPTIONAL averaging-permutation
//! column set (`DeviceCtrAveraging`) as a plain host type — `None` by default
//! (byte-unchanged non-CTR / legacy shape), populated only by a covered
//! two-permutation CTR fit (GDC-11).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use crate::runtime::{DeviceCtrAveraging, DeviceCtrColumn, DeviceCtrConfig};

#[test]
fn device_ctr_config_carries_optional_averaging() {
    // Default: no averaging half (and the derive-Default shape is preserved).
    assert!(DeviceCtrConfig::default().averaging.is_none());

    // T17 / DCTR-15: `projection_members` stated explicitly rather than left to `Default`
    // (an EMPTY list). One member ⇒ SIMPLE, matching this column's single `member_bins`
    // entry, so the fixture cannot misrepresent its arity to a future consumer.
    let column = DeviceCtrColumn {
        member_bins: vec![vec![0, 1, 2, 0]],
        prior: 0.5,
        borders: vec![0.25, 0.5, 0.75],
        projection_members: vec![0],
        ..DeviceCtrColumn::default()
    };
    let config = DeviceCtrConfig {
        permutation: vec![3, 1, 0, 2],
        target_class: vec![0, 1, 1, 0],
        columns: vec![column.clone()],
        averaging: Some(DeviceCtrAveraging {
            permutation: vec![2, 0, 3, 1],
            target_class: vec![0, 1, 1, 0],
            columns: vec![column],
        }),
        ..DeviceCtrConfig::default()
    };

    // Plain-host round-trip: Clone + PartialEq hold, and the two permutations are
    // independently addressable (the structure one must never alias the averaging
    // one — GDC-09's defining invariant is that they can differ).
    let cloned = config.clone();
    assert_eq!(cloned, config);
    let averaging = cloned.averaging.expect("populated averaging half");
    assert_ne!(
        averaging.permutation, config.permutation,
        "test shape must exercise DIVERGENT structure/averaging permutations"
    );
    assert_eq!(averaging.columns.len(), config.columns.len());
    assert_eq!(averaging.target_class.len(), config.target_class.len());
}
