//! T26 / SPEC-OH-23 — the split-application test has a one-hot (EQUALITY) arm.
//!
//! # Why this lives here and not in `crates/cb-backend/tests/`
//!
//! The fns drive `super::launch_partition_split_packed_into` (`pub(crate)`) and build
//! their input with `super::cindex::pack_cindex` (`pub(crate)` in a `pub(crate)` module).
//! Neither is reachable from an integration test, so this is a `gpu_runtime` descendant —
//! which also means the test does NOT need to hand-roll the packed words.
//!
//! # What changes
//!
//! `partition_split_kernel` routes an object LEFT/RIGHT by testing its packed bin. With
//! `one_hot == false` the test is `bin > border` (threshold); with `one_hot == true` it is
//! `bin == value` (equality), matching the CPU `FeatureMatrix::passes_one_hot`
//! (`IsTrueOneHotFeature`). One kernel, one comptime flag — the comptime resolves the
//! other arm away entirely.

use super::cindex::pack_cindex;
use super::{launch_partition_split_packed_into, read_u32_handle};
use crate::SelectedRuntime;

fn client() -> cubecl::client::ComputeClient<SelectedRuntime> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    <SelectedRuntime as cubecl::Runtime>::client(&device)
}

/// Pack ONE column of bins and route every object through the packed split launcher,
/// returning the resulting `new_leaf_of`.
fn route(bins: &[u32], n_buckets: usize, one_hot: bool, bin: u32, level_bit: u32) -> Vec<u32> {
    let n = bins.len();
    let packed = pack_cindex(bins, &[n_buckets], &[one_hot], n).expect("pack_cindex must succeed");
    let (offsets, shifts, masks, _flags) = packed.device_arrays().expect("device_arrays");
    let client = client();

    let der1_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; n]));
    let words_h = client.create(cubecl::bytes::Bytes::from_elems(packed.words.clone()));
    let indices_h = client.create(cubecl::bytes::Bytes::from_elems(
        (0..n as u32).collect::<Vec<u32>>(),
    ));
    let leaf_of_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; n]));

    let out = launch_partition_split_packed_into(
        &client,
        der1_h,
        words_h,
        indices_h,
        leaf_of_h,
        n,
        packed.words.len(),
        offsets.first().copied().unwrap_or(0),
        shifts.first().copied().unwrap_or(0),
        masks.first().copied().unwrap_or(0),
        bin,
        level_bit,
        one_hot,
    )
    .expect("the packed split launch must succeed");

    read_u32_handle(&client, out).expect("the routing read-back must succeed")
}

/// Fn 1 — the one-hot arm routes on EQUALITY, so only the objects whose bin EQUALS the
/// split value take the bit. Under the threshold test, bin 2 would also pass.
#[test]
fn one_hot_partition_split_routes_on_equality() {
    let bins = [0u32, 1, 2, 1, 0];
    let got = route(&bins, 3, /* one_hot = */ true, /* bin = */ 1, 0);
    assert_eq!(
        got,
        vec![0u32, 1, 0, 1, 0],
        "an equality split must set the level bit for bins EQUAL to the value; the \
         threshold test would also let bin 2 through, giving [0, 1, 1, 1, 0]"
    );
}

/// Fn 2 — the float (threshold) routing is numerically unchanged by the one-hot arm.
#[test]
fn float_partition_split_is_unchanged_after_the_one_hot_arm() {
    let bins = [0u32, 1, 2, 1, 0];
    let got = route(&bins, 3, /* one_hot = */ false, /* bin = */ 1, 0);
    assert_eq!(
        got,
        vec![0u32, 0, 1, 0, 0],
        "the threshold arm must still be `bin > border` exactly"
    );

    // A non-zero level bit must land in the right position on the threshold arm too.
    let got_bit2 = route(&bins, 3, false, 0, 2);
    assert_eq!(
        got_bit2,
        vec![0u32, 4, 4, 4, 0],
        "the level bit position is independent of the split kind"
    );
}
