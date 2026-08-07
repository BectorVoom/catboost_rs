//! FPP-15 (T04): [`FamilyTreeArgs`] is a plain-host, borrow-only descriptor.
//!
//! The load-bearing property is structural, so the assertions are mostly the fact that
//! this file COMPILES: every variant must be constructible from ordinary host literals
//! (`&[u32]` / `&[f64]` / `usize`) with no `cubecl` and no `cb-train` type in sight
//! (T-10-04). The field round-trips below keep that from degrading into a no-op test.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use crate::runtime::{DeviceGrownTree, FamilyTreeArgs, Runtime};
use cb_core::CbResult;

#[test]
fn family_tree_args_variants_are_plain_host_types() {
    let group_offsets: &[u32] = &[0, 3, 7];
    let ranking = FamilyTreeArgs::Ranking { group_offsets };
    match ranking {
        FamilyTreeArgs::Ranking { group_offsets: got } => {
            assert_eq!(got, &[0, 3, 7], "Ranking must round-trip its CSR prefix");
        }
        other => panic!("expected Ranking, got {other:?}"),
    }

    let pair_begin: &[u32] = &[0, 1];
    let pair_end: &[u32] = &[2, 3];
    let pair_weight: &[f64] = &[1.0, 0.5];
    let pairwise = FamilyTreeArgs::Pairwise {
        group_offsets,
        pair_begin,
        pair_end,
        pair_weight,
    };
    match pairwise {
        FamilyTreeArgs::Pairwise {
            group_offsets: g,
            pair_begin: b,
            pair_end: e,
            pair_weight: w,
        } => {
            assert_eq!(g.len(), 3);
            assert_eq!(b, &[0, 1]);
            assert_eq!(e, &[2, 3]);
            assert_eq!(w, &[1.0, 0.5]);
            assert_eq!(b.len(), w.len(), "one weight per pair");
        }
        other => panic!("expected Pairwise, got {other:?}"),
    }

    // DIM-MAJOR: approx_k[d * n + i]. Two dims over three objects.
    let approx_k: &[f64] = &[0.0, 1.0, 2.0, 10.0, 11.0, 12.0];
    let multi = FamilyTreeArgs::MultiOutput {
        approx_k,
        approx_dim: 2,
    };
    match multi {
        FamilyTreeArgs::MultiOutput {
            approx_k: a,
            approx_dim,
        } => {
            assert_eq!(approx_dim, 2);
            let n = a.len() / approx_dim;
            assert_eq!(n, 3);
            assert_eq!(a[1 * n + 0], 10.0, "dim-major indexing is approx_k[d * n + i]");
        }
        other => panic!("expected MultiOutput, got {other:?}"),
    }
}

/// A `Runtime` that overrides nothing: the trait default must still return `Ok(None)`
/// for both `None` and `Some(family)`, so adding the parameter changed no behaviour
/// (D-04). A backend that ignores `family` must behave exactly as it did before.
struct DefaultRuntime;

impl Runtime for DefaultRuntime {
    fn compute_gradients(
        &self,
        _loss: &crate::runtime::Loss,
        _approx: &[f64],
        _target: &[f64],
        _approx_dimension: usize,
    ) -> CbResult<crate::runtime::Derivatives> {
        Ok(crate::runtime::Derivatives {
            der1: Vec::new(),
            der2: Vec::new(),
        })
    }
}

#[test]
fn default_grow_tree_on_device_ignores_family_and_still_declines() {
    let rt = DefaultRuntime;
    let approx = [0.0_f64; 4];
    let target = [1.0_f64; 4];
    let group_offsets: &[u32] = &[0, 4];
    let family = FamilyTreeArgs::Ranking { group_offsets };

    let without: Option<DeviceGrownTree> = rt
        .grow_tree_on_device(&approx, &target, &[], None)
        .expect("default impl never errors");
    let with: Option<DeviceGrownTree> = rt
        .grow_tree_on_device(&approx, &target, &[], Some(&family))
        .expect("default impl never errors");

    assert!(without.is_none(), "the trait default declines to the CPU grow loop");
    assert!(
        with.is_none(),
        "a family descriptor must not make an unimplementing backend fabricate a tree \
         (T-10-05: never a fabricated device result)"
    );
}
