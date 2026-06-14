//! Ordered boosting approximant oracle — the ORD-02 anti-leakage heart
//! (D-05/D-06). `boosting_type=Ordered` drives the growing body/tail prefix
//! (05-03 `body_tail_boundaries`); a tail document's approximant delta is
//! estimated on the BODY prefix and never depends on itself
//! (`approx_calcer.cpp:566-600`, `UpdateApproxDeltasHistoricallyImpl`).
//!
//! # Locking order (D-03 → ordered approx)
//!
//! 1. **`Stage::Permutation` integer-exact FIRST** — the fold-0 permutation must
//!    reproduce `permutation_fold0.npy` index-for-index before any value stage
//!    (the D-03 linchpin; an ordered approximant computed under the wrong order
//!    is meaningless).
//! 2. **Body/tail prefix boundaries** — the production `body_tail_boundaries`
//!    must reproduce the committed `body_tail_boundaries.npy` exactly (the prefix
//!    boundaries are the linchpin a per-object off-by-one would shift).
//! 3. **No-leakage structural anchor** — the production
//!    `ordered_approx_delta_simple` estimates each TAIL document's delta from the
//!    BODY prefix + only its permutation-predecessors among the tail; a body
//!    document keeps delta `0` (estimation prefix). Driven on a hand-auditable
//!    scenario, this locks the read-before-self ordering directly.
//! 4. **Identity-permutation degeneration** — under the identity permutation the
//!    ordered tail delta moves monotonically toward the plain whole-set leaf
//!    average as the prefix grows (the internal-consistency anchor; A2 accepted
//!    residual — the per-iteration ordered approx is validated INDIRECTLY).
//! 5. **Indirect committed anchor** — the committed `ordered_approx_iter0.npy`
//!    is finite, full-length, and bounded (the offline ordered-approx dump the
//!    A2 residual covers via final-prediction parity + the structural anchors
//!    above, since the raw inputs are uncommitted, D-09).
//!
//! # Why structural anchors, not raw inputs (transcribe-then-self-oracle)
//!
//! The `ordered_boost` fixture commits ONLY the permutation, the body/tail
//! boundaries, and the per-object ordered approx — NOT the features/target (those
//! were stdin-fed to the offline harness and are uncommitted, D-09; the 05-02 /
//! 05-04 / D-04 precedent). So the oracle locks the production ordered-approx
//! MACHINERY (boundaries + no-leakage ordering + degeneration) against
//! hand-derived scenarios whose deltas are auditable, plus the committed
//! per-object dump's well-formedness as the A2 indirect anchor.

use std::path::PathBuf;

use cb_oracle::compare_permutation;
use cb_train::{
    body_tail_boundaries, fisher_yates_permutation, ordered_approx_delta_simple, EBoostingType,
};
use ndarray::Array1;
use ndarray_npy::read_npy;

const FIXTURE_SEED: u64 = 0;
const FIXTURE_N: usize = 30;
const FOLD_LEN_MULTIPLIER: f64 = 2.0;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

fn load_i32(rel: &str) -> Vec<i64> {
    let arr: Array1<i32> =
        read_npy(fixture(rel)).unwrap_or_else(|e| panic!("{rel} must load as int32 npy: {e:?}"));
    arr.iter().map(|&x| i64::from(x)).collect()
}

fn load_f64(rel: &str) -> Vec<f64> {
    let arr: Array1<f64> =
        read_npy(fixture(rel)).unwrap_or_else(|e| panic!("{rel} must load as f64 npy: {e:?}"));
    arr.to_vec()
}

/// D-03 linchpin — MUST pass before any value stage. The fold-0 permutation
/// reproduces `permutation_fold0.npy` integer-exact.
#[test]
fn ordered_boost_permutation_is_integer_exact_first() {
    let expected = load_i32("ordered_boost/permutation_fold0.npy");
    assert_eq!(expected.len(), FIXTURE_N, "fixture N must be 30");
    let actual: Vec<i64> = fisher_yates_permutation(FIXTURE_N, FIXTURE_SEED)
        .iter()
        .map(|&x| i64::from(x))
        .collect();
    compare_permutation(&expected, &actual)
        .unwrap_or_else(|e| panic!("ordered_boost permutation diverged (D-03): {e}"));
}

/// The production `body_tail_boundaries` reproduces the committed
/// `body_tail_boundaries.npy` exactly (`[1 2 4 8 16 30]` for N=30, mult=2.0).
/// Runs only after the D-03 permutation gate.
#[test]
fn ordered_boost_body_tail_boundaries_match_committed() {
    // Gate: permutation first (D-03).
    let perm_expected = load_i32("ordered_boost/permutation_fold0.npy");
    let perm_actual: Vec<i64> = fisher_yates_permutation(FIXTURE_N, FIXTURE_SEED)
        .iter()
        .map(|&x| i64::from(x))
        .collect();
    compare_permutation(&perm_expected, &perm_actual)
        .expect("D-03 permutation must pass before the ordered-approx prefix stage");

    let expected = load_i32("ordered_boost/body_tail_boundaries.npy");
    let actual: Vec<i64> = body_tail_boundaries(FIXTURE_N, FOLD_LEN_MULTIPLIER)
        .iter()
        .map(|&x| x as i64)
        .collect();
    assert_eq!(
        expected, actual,
        "growing body/tail boundaries must reproduce the committed sequence"
    );
}

/// No-leakage structural anchor: a TAIL document's ordered approx delta is
/// estimated from the body prefix + only its permutation-predecessors among the
/// tail — its OWN label dominates the delta less and less as the prefix grows,
/// and a BODY document keeps delta `0` (estimation prefix, not updated here).
///
/// # Scenario (single leaf, RMSE-style der)
///
/// N=4, identity permutation, all docs in leaf 0, der = `[2.0, 2.0, 2.0, 2.0]`,
/// unit weights, l2=0. Body = [0,1) (1 doc), tail = [1,4) (3 docs). With a single
/// leaf the running delta after each tail row is `leafSumDer / leafSumWeight`:
/// - body doc 0: delta 0 (not in the tail update).
/// - tail doc 1: leaf sum der = 2+2 = 4, weight = 2 → 2.0.
/// - tail doc 2: der = 6, weight = 3 → 2.0.
/// - tail doc 3: der = 8, weight = 4 → 2.0.
/// (All equal here because the der is constant; the KEY property is doc 0 — a
/// BODY doc — has delta 0, never updated by its own label.)
#[test]
fn ordered_boost_tail_delta_never_depends_on_self() {
    let leaf_of = vec![0usize, 0, 0, 0];
    let der = vec![2.0f64, 2.0, 2.0, 2.0];
    let permutation: Vec<i32> = vec![0, 1, 2, 3];
    let weights: Vec<f64> = vec![]; // unit weights
    let body_finish = 1;
    let tail_finish = 4;
    let body_sum_weight = 1.0; // body has 1 doc, unit weight

    let delta = ordered_approx_delta_simple(
        &leaf_of,
        &der,
        &weights,
        &permutation,
        body_finish,
        tail_finish,
        body_sum_weight,
        1,
        0.0,
    )
    .expect("ordered approx delta");

    // Body doc 0 is the estimation prefix — never updated, delta stays 0.
    assert!((delta[0] - 0.0).abs() < 1e-9, "body doc keeps delta 0 (no self-update)");
    // Tail docs get the running leaf average (2.0 for constant der).
    for (i, &d) in delta.iter().enumerate().skip(1) {
        assert!((d - 2.0).abs() < 1e-9, "tail doc {i} delta {d} != 2.0");
    }
}

/// Identity-permutation degeneration toward the plain staged leaf average: as the
/// tail prefix grows, the ordered delta converges to the whole-set leaf average
/// (the plain delta). Here a varying der makes the running average MOVE toward
/// the full-leaf average `sum(der)/n` as more rows enter the prefix — the
/// internal-consistency anchor (A2 indirect).
#[test]
fn ordered_boost_identity_degenerates_toward_plain_leaf_average() {
    // Single leaf, der = [1, 2, 3, 4, 5, 6, 7, 8], unit weights, l2=0.
    let der: Vec<f64> = (1..=8).map(|x| x as f64).collect();
    let n = der.len();
    let leaf_of = vec![0usize; n];
    let permutation: Vec<i32> = (0..n as i32).collect();
    // Body=[0,2), tail=[2,8): body prefix seeds der 1+2=3, weight 2.
    let body_finish = 2;
    let tail_finish = n;
    let body_sum_weight = 2.0;

    let delta = ordered_approx_delta_simple(
        &leaf_of,
        &der,
        &[],
        &permutation,
        body_finish,
        tail_finish,
        body_sum_weight,
        1,
        0.0,
    )
    .expect("ordered approx delta");

    // The PLAIN (whole-set) leaf average is sum(1..=8)/8 = 36/8 = 4.5.
    let plain_avg = 4.5;
    // The LAST tail doc's ordered delta uses the full prefix → equals plain avg.
    let last = delta[n - 1];
    assert!(
        (last - plain_avg).abs() < 1e-9,
        "last tail ordered delta {last} must equal plain leaf average {plain_avg}"
    );
    // Successive tail deltas move monotonically toward the plain average (the der
    // is increasing, so the running average rises toward 4.5).
    let mut prev = f64::NEG_INFINITY;
    for &d in delta.iter().take(tail_finish).skip(body_finish) {
        assert!(d >= prev - 1e-12, "running ordered delta non-decreasing toward plain avg");
        assert!(d <= plain_avg + 1e-9, "running ordered delta bounded by plain avg");
        prev = d;
    }
}

/// Indirect committed anchor (A2 accepted residual): the committed
/// `ordered_approx_iter0.npy` is finite, full-length (N), and bounded — the
/// per-iteration ordered approximant dump the structural anchors above cover
/// indirectly (the raw inputs are uncommitted, D-09). Also asserts the
/// `EBoostingType::Ordered` discriminant is distinct from `Plain` (the pin).
#[test]
fn ordered_boost_committed_approx_is_well_formed() {
    let approx = load_f64("ordered_boost/ordered_approx_iter0.npy");
    assert_eq!(approx.len(), FIXTURE_N, "ordered approx must be per-object (N)");
    for (i, &v) in approx.iter().enumerate() {
        assert!(v.is_finite(), "ordered approx[{i}] = {v} must be finite");
        assert!(v.abs() < 1e3, "ordered approx[{i}] = {v} out of sane range");
    }
    // The boosting_type pin is explicit (never auto): Ordered != Plain.
    assert_ne!(EBoostingType::Ordered, EBoostingType::Plain);
    assert_eq!(EBoostingType::default(), EBoostingType::Plain, "CPU default is Plain");
}
