//! Self-oracle for the ordered-boosting device trajectory driver (Phase 13 Plan 08, GPUT-13): the
//! device [`crate::gpu_runtime::ordered::accumulate_ordered_trajectory`] must reproduce the frozen
//! CPU per-permutation approx TRAJECTORY — the sum over boosting iterations of the anti-leakage
//! body/tail approximant ([`crate::gpu_runtime::ordered::ordered_approx_delta`], transcribed from
//! `cb_train::boosting::ordered_approx_delta_simple`, D-06) — within the ε=1e-4 GPU bar (D-07).
//!
//! Three properties are covered on a FROZEN pinned fixture (a single learning fold, Open Q2):
//!
//! 1. **Trajectory parity** — the device resident trajectory after N iterations equals the host sum
//!    of the frozen per-iteration ordered deltas ≤1e-4 (the device add runs on rocm/cuda; on the
//!    default `cpu` backend the "device" IS the host, so the numeric assert is record-only to avoid a
//!    CPU-vs-CPU false-pass, WR-01).
//! 2. **Anti-leakage body prefix** — the BODY rows keep delta exactly `0` in every iteration (the
//!    estimation prefix is never updated historically).
//! 3. **Residency (single final read-back)** — the driver holds the trajectory handle DEVICE-RESIDENT
//!    across all N iterations and reads it back exactly ONCE (the multi-iteration accumulation is
//!    correct only if the resident handle persists across iterations — the session-residency
//!    precedent; the driver performs one `read_one`, outside the loop).
//!
//! Plus the coverage gate: an ordered fit (`boosting_type_is_plain == false`) declines to `Ok(None)`
//! (the per-tree ordered permutation-descriptor grow seam is a forward dependency — the pairwise /
//! ranking / multiclass precedent), never fabricating a Plain grow.
//!
//! Source/test separation (CLAUDE.md / AGENTS.md): the driver + coverage gate are production code;
//! ALL `#[test]` + `.unwrap()`/indexing live here. cb-backend must NEVER gain a `cb-train` dep even
//! in the test — the ordered delta reference is the driver's own inline transcription (the frozen CPU
//! oracle, not a `cb-train` call). The whole file is gated off the (f64-less) wgpu backend.

#![cfg(not(feature = "wgpu"))]

use cb_compute::{gradient_leaf_delta, DeviceTrainConfig, EScoreFunction, Loss};

use crate::gpu_runtime::ordered::{accumulate_ordered_trajectory, ordered_approx_delta, OrderedTree};
use crate::gpu_runtime::GpuTrainSession;

/// The ε=1e-4 device-vs-CPU bar (D-07; the GPU bar, looser than the CPU ref's own ≤1e-5).
const TOL: f64 = 1e-4;

/// Whether the trajectory add actually runs on a real device backend (rocm/cuda). On the default
/// `cpu` backend the "device" IS the host, so a numeric assert would be a CPU-vs-CPU false-pass
/// (WR-01) — record-only there, hard-assert on rocm/cuda.
fn device_backend_active() -> bool {
    cfg!(any(feature = "rocm", feature = "cuda"))
}

/// Max abs divergence over equal-length buffers (`INFINITY` on a length mismatch).
fn max_abs_divergence(device: &[f64], reference: &[f64]) -> f64 {
    if device.len() != reference.len() {
        return f64::INFINITY;
    }
    device
        .iter()
        .zip(reference.iter())
        .map(|(&d, &r)| (d - r).abs())
        .fold(0.0_f64, f64::max)
}

/// The FROZEN fixture: a small single-fold ordered-boosting problem. `n = 8` objects, two leaves,
/// a pinned learn permutation (the ordered seed's frozen output — a fixed shuffle here), and three
/// boosting iterations with distinct per-iteration leaf assignments + derivatives. `body_finish = 3`
/// (the estimation prefix), `tail_finish = n` (the whole tail approximated historically).
struct Fixture {
    n: usize,
    body_finish: usize,
    tail_finish: usize,
    n_leaves: usize,
    scaled_l2: f64,
    permutation: Vec<i32>,
    /// Per-iteration `(leaf_of, der)` — three trees over the SAME permutation/weights (unit weight).
    trees: Vec<(Vec<u32>, Vec<f64>)>,
}

fn frozen_fixture() -> Fixture {
    let n = 8usize;
    // Pinned learn permutation (a fixed shuffle of 0..8 — the frozen ordered seed output).
    let permutation: Vec<i32> = vec![5, 1, 7, 0, 3, 6, 2, 4];
    // Three boosting iterations; leaf assignments split the objects two ways, derivatives spread.
    let trees = vec![
        (
            vec![0u32, 0, 1, 1, 0, 1, 0, 1],
            vec![0.9_f64, -0.4, 0.3, 0.7, -0.6, 0.2, 0.5, -0.1],
        ),
        (
            vec![1u32, 0, 0, 1, 1, 0, 1, 0],
            vec![-0.3_f64, 0.8, -0.2, 0.4, 0.1, -0.5, 0.6, -0.7],
        ),
        (
            vec![0u32, 1, 1, 0, 0, 1, 1, 0],
            vec![0.2_f64, 0.5, -0.8, 0.1, -0.3, 0.9, -0.4, 0.6],
        ),
    ];
    Fixture {
        n,
        body_finish: 3,
        tail_finish: n,
        n_leaves: 2,
        scaled_l2: 3.0,
        permutation,
        trees,
    }
}

/// Build the borrowed [`OrderedTree`] descriptors for the fixture (unit weights — empty slice).
fn ordered_trees<'a>(fx: &'a Fixture, empty_weight: &'a [f64]) -> Vec<OrderedTree<'a>> {
    fx.trees
        .iter()
        .map(|(leaf_of, der)| OrderedTree {
            leaf_of,
            der,
            weights: empty_weight,
            permutation: &fx.permutation,
            body_finish: fx.body_finish,
            tail_finish: fx.tail_finish,
            n_leaves: fx.n_leaves,
            scaled_l2: fx.scaled_l2,
        })
        .collect()
}

/// The host reference trajectory: sum, per object, the frozen per-iteration ordered deltas (each
/// computed by the SAME `ordered_approx_delta` the driver folds on device — the device path differs
/// only in that the FOLD runs through the resident `apply_leaf_delta` add).
fn cpu_reference_trajectory(fx: &Fixture, empty_weight: &[f64]) -> Vec<f64> {
    let mut trajectory = vec![0.0_f64; fx.n];
    for tree in ordered_trees(fx, empty_weight) {
        let delta = ordered_approx_delta(&tree).expect("ordered delta must not error");
        for i in 0..fx.n {
            if let (Some(slot), Some(&d)) = (trajectory.get_mut(i), delta.get(i)) {
                *slot += d;
            }
        }
    }
    trajectory
}

/// Test 1: the device resident trajectory after N iterations equals the host sum of the frozen
/// per-iteration ordered deltas within `TOL` (record-only on the cpu backend, hard-assert on
/// rocm/cuda).
#[test]
fn device_trajectory_matches_frozen_cpu_reference() {
    let fx = frozen_fixture();
    let empty_weight: Vec<f64> = Vec::new();

    let device = accumulate_ordered_trajectory(&ordered_trees(&fx, &empty_weight), fx.n)
        .expect("device ordered trajectory must not error");
    let reference = cpu_reference_trajectory(&fx, &empty_weight);

    assert_eq!(device.len(), fx.n, "device trajectory is length n");
    assert_eq!(reference.len(), fx.n, "reference trajectory is length n");
    for &v in &device {
        assert!(v.is_finite(), "device trajectory value must be finite, got {v}");
    }
    let divergence = max_abs_divergence(&device, &reference);
    println!(
        "[ordered:trajectory] device-vs-CPU max abs divergence = {divergence:e} \
         (device_backend_active = {})",
        device_backend_active()
    );
    if device_backend_active() {
        assert!(
            divergence <= TOL,
            "device ordered trajectory diverged from frozen CPU: {divergence:e} > {TOL:e}\n\
             device = {device:?}\nreference = {reference:?}"
        );
    }
}

/// Test 2: the BODY rows keep delta exactly `0` in every iteration (the anti-leakage estimation
/// prefix is never historically updated). The body objects are the first `body_finish` learn-order
/// positions of the permutation.
#[test]
fn body_rows_keep_delta_zero() {
    let fx = frozen_fixture();
    let empty_weight: Vec<f64> = Vec::new();
    let body_objects: Vec<usize> = fx
        .permutation
        .iter()
        .take(fx.body_finish)
        .map(|&p| p as usize)
        .collect();

    for tree in ordered_trees(&fx, &empty_weight) {
        let delta = ordered_approx_delta(&tree).expect("ordered delta must not error");
        for &obj in &body_objects {
            assert_eq!(
                delta.get(obj).copied(),
                Some(0.0),
                "body object {obj} must keep delta 0 (anti-leakage estimation prefix)"
            );
        }
        // A tail object that appears in this tree's tail must be able to take a NON-zero delta
        // (sanity: the approximant is actually doing work), unless its running leaf average is 0.
        let any_tail_nonzero = fx
            .permutation
            .iter()
            .skip(fx.body_finish)
            .any(|&p| delta.get(p as usize).copied().unwrap_or(0.0) != 0.0);
        assert!(any_tail_nonzero, "at least one tail row must take a non-zero historical delta");
    }
}

/// Test 3 (residency): the driver holds the trajectory handle DEVICE-RESIDENT across all N
/// iterations and reads it back exactly once — the multi-iteration accumulation is correct ONLY if
/// the resident handle persists across iterations (a handle reset each iteration would drop earlier
/// trees' contributions). Assert the N-iteration resident accumulation equals the sum of each tree's
/// SINGLE-iteration trajectory (the same total, proving cross-iteration persistence), matching the
/// host reference within `TOL` on a real device.
#[test]
fn resident_trajectory_persists_across_iterations() {
    let fx = frozen_fixture();
    let empty_weight: Vec<f64> = Vec::new();

    // The full N-iteration resident accumulation (one final read-back inside the driver).
    let device_full = accumulate_ordered_trajectory(&ordered_trees(&fx, &empty_weight), fx.n)
        .expect("device full trajectory must not error");

    // Independently sum each tree's own single-iteration device trajectory. If the driver reset the
    // resident handle per iteration (no persistence), `device_full` would equal only the LAST tree's
    // delta, not this sum — so equality proves the handle persisted across iterations.
    let mut summed_singletons = vec![0.0_f64; fx.n];
    for tree in ordered_trees(&fx, &empty_weight) {
        let single = accumulate_ordered_trajectory(std::slice::from_ref(&tree), fx.n)
            .expect("device single-tree trajectory must not error");
        for i in 0..fx.n {
            if let (Some(slot), Some(&v)) = (summed_singletons.get_mut(i), single.get(i)) {
                *slot += v;
            }
        }
    }

    let divergence = max_abs_divergence(&device_full, &summed_singletons);
    println!(
        "[ordered:residency] full-vs-summed-singletons max abs divergence = {divergence:e} \
         (device_backend_active = {})",
        device_backend_active()
    );
    // This identity holds on EVERY backend (the fold is exact addition) — a broken resident handle
    // would fail it regardless of device presence.
    assert!(
        divergence <= TOL,
        "resident trajectory did not persist across iterations: {divergence:e} > {TOL:e}\n\
         full = {device_full:?}\nsummed_singletons = {summed_singletons:?}"
    );
}

/// The coverage gate: `GpuTrainSession::begin` declines an ordered fit (`boosting_type_is_plain ==
/// false`) to `Ok(None)` (the per-tree ordered permutation-descriptor grow seam is a forward
/// dependency — the pairwise / ranking / multiclass precedent), for BOTH a covered ordered config and
/// a genuinely uncovered one. Never a fabricated Plain grow, never an error while classifying.
#[test]
fn begin_declines_ordered_to_cpu() {
    let n = 8usize;
    let n_features = 2usize;
    let n_bins = 32usize;
    let weight = vec![1.0_f64; n];
    // Minimal valid feature-major cindex (bins < n_bins).
    let mut cindex = vec![0u32; n_features * n];
    for f in 0..n_features {
        for obj in 0..n {
            if let Some(slot) = cindex.get_mut(f * n + obj) {
                *slot = (obj % n_bins) as u32;
            }
        }
    }
    let scaled_l2 = 3.0_f64;
    let lr = 0.3_f64;

    let open = |loss: &Loss, depth: usize, plain: bool, folds: usize, cfg: &DeviceTrainConfig| {
        GpuTrainSession::begin(
            loss,
            depth,
            plain,
            folds,
            EScoreFunction::Cosine,
            &cindex,
            &weight,
            n,
            n_features,
            n_bins,
            lr,
            scaled_l2,
            cfg,
        )
        .expect("begin must not error while classifying ordered coverage")
        .is_some()
    };
    let def = DeviceTrainConfig::default();

    // Covered ordered config (RMSE, SymmetricTree, depth≥1, single fold) but ORDERED (not Plain) →
    // declines to CPU pending the per-tree ordered permutation-descriptor grow seam.
    assert!(
        !open(&Loss::Rmse, 2, false, 1, &def),
        "covered ordered RMSE declines to CPU pending the ordered grow seam"
    );
    assert!(
        !open(&Loss::Logloss, 2, false, 1, &def),
        "covered ordered Logloss declines to CPU pending the ordered grow seam"
    );
    // Genuinely uncovered ordered configs → also Ok(None) (all-or-nothing per family).
    assert!(
        !open(&Loss::Rmse, 0, false, 1, &def),
        "depth-0 ordered must decline"
    );
    assert!(
        !open(&Loss::Rmse, 2, false, 2, &def),
        "fold_count>1 ordered must decline"
    );
    // The Plain path is byte-unchanged (a covered Plain RMSE fit opens a session as before).
    assert!(
        open(&Loss::Rmse, 2, true, 1, &def),
        "covered Plain RMSE still opens a device session (byte-unchanged, D-04)"
    );
}

/// Sanity: the ordered delta gradient average matches `cb_compute::gradient_leaf_delta` (the shared
/// `calc_average`) for a single tail row over a single-object leaf — a direct numeric anchor that the
/// transcription reads the running leaf average, not some other statistic.
#[test]
fn single_tail_row_equals_gradient_leaf_delta() {
    // One object in the tail, its own leaf, empty body. Its delta = its own der / (weight + l2).
    let leaf_of = vec![0u32, 1];
    let der = vec![0.6_f64, -0.2];
    let permutation = vec![0i32, 1];
    let empty_weight: Vec<f64> = Vec::new();
    let scaled_l2 = 2.0_f64;
    let tree = OrderedTree {
        leaf_of: &leaf_of,
        der: &der,
        weights: &empty_weight,
        permutation: &permutation,
        body_finish: 0,
        tail_finish: 2,
        n_leaves: 2,
        scaled_l2,
    };
    let delta = ordered_approx_delta(&tree).expect("ordered delta must not error");
    // Object 0 is the first tail row in its own leaf: der 0.6, weight 1 → 0.6 / (1 + 2).
    assert!(
        (delta.first().copied().unwrap_or(0.0) - gradient_leaf_delta(0.6, 1.0, scaled_l2)).abs()
            <= 1e-12,
        "single tail row delta must equal gradient_leaf_delta(der, weight, l2)"
    );
}
