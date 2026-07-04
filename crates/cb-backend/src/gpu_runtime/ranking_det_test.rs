//! Self-oracle for the deterministic ranking der driver (Phase 13 Plan 04, GPUT-22): the device
//! QueryRMSE / QuerySoftMax der ([`crate::gpu_runtime::ranking`]) must reproduce the FROZEN CPU
//! `cb_compute::calc_ders_for_queries` der (the LOSS-04 grouped der seam, itself oracle-tested to
//! ≤1e-5) over a 3-query uneven fixture within ε=1e-4 (the D-07 GPU bar). QueryCrossEntropy is
//! asserted INDEPENDENTLY DEFERRED (Open Q3): its coverage flag is `false` (→ the session gate maps
//! it to `Ok(None)` WITHOUT disabling QueryRMSE / QuerySoftMax), and its bounded per-query shift
//! search is exercised as a self-consistent root-find (`Σ w·sigmoid(approx + shift) ≈ Σ w·target`).
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the device der driver lives in the
//! production `crate::gpu_runtime::ranking` module; ALL assertions / `.unwrap()` / indexing live
//! here. The reference is the INDEPENDENT `cb_compute::calc_ders_for_queries` (a different, already
//! oracle-tested implementation), so this is NON-tautological.
//!
//! Runs over [`crate::SelectedRuntime`]. The serial f64 ranking der kernels execute on the cpu
//! cubecl backend (the `query_helper` group-reduction precedent), so length / finiteness /
//! gated-off logic hard-assert on EVERY backend; the numeric ε=1e-4 vs-CPU assertion SKIPS off
//! rocm/cuda (record-only) so a default `cpu` run does not masquerade as a GPU validation (WR-01
//! anti-false-pass — on `cpu` the "device" IS the host, a CPU-vs-CPU comparison).

#![cfg(not(feature = "wgpu"))]

use cb_compute::{calc_ders_for_queries, Derivatives, GroupSpan, Loss};

use crate::gpu_runtime::ranking::{
    query_cross_entropy_shifts_host, query_rmse_ders_host, query_softmax_ders_host,
    ranking_objective_covered, RankingObjective,
};

/// The ε=1e-4 device-vs-CPU bar (D-07; the GPU bar, looser than the CPU ref's own ≤1e-5).
const TOL: f64 = 1e-4;

/// Whether the numeric ε assertion runs on a REAL device (rocm/cuda) — else record-only (WR-01):
/// on the `cpu` backend the "device" IS the host, so a numeric assert would be a CPU-vs-CPU
/// false-pass rather than a GPU validation.
fn device_backend_active() -> bool {
    cfg!(any(feature = "rocm", feature = "cuda"))
}

/// A 3-query UNEVEN fixture (query sizes 3 / 2 / 4, `n = 9`) with varied approx / relevance targets.
/// Uniform object weights (the covered device ranking regime) → `weights` passed empty (both the CPU
/// reference and the device driver expand it to a `1.0` column).
fn uneven_fixture() -> (Vec<f64>, Vec<f64>, Vec<u32>) {
    let approx = vec![0.5, -0.2, 1.3, 0.1, 0.4, -0.7, 0.9, 0.2, -0.3];
    let target = vec![1.0, 0.0, 2.0, 0.0, 1.0, 3.0, 1.0, 0.0, 2.0];
    let q_offsets = vec![0u32, 3, 5, 9];
    (approx, target, q_offsets)
}

/// The per-group [`GroupSpan`] view the CPU `calc_ders_for_queries` slices on. QueryRMSE /
/// QuerySoftMax are querywise (not pairwise), so `competitors` is empty and the per-GROUP `weight`
/// is unused (their arms read the flat per-OBJECT weights).
fn group_spans(q_offsets: &[u32]) -> Vec<GroupSpan> {
    q_offsets
        .windows(2)
        .map(|w| GroupSpan {
            begin: w[0] as usize,
            end: w[1] as usize,
            weight: 1.0,
            competitors: Vec::new(),
        })
        .collect()
}

/// Flatten the CPU per-group [`Derivatives`] into flat per-object `(der1, der2)` in object order (the
/// order the device driver returns).
fn flatten(ders: &[Derivatives]) -> (Vec<f64>, Vec<f64>) {
    let mut der1 = Vec::new();
    let mut der2 = Vec::new();
    for d in ders {
        der1.extend_from_slice(&d.der1);
        der2.extend_from_slice(&d.der2);
    }
    (der1, der2)
}

fn max_abs_divergence(device: &[f64], reference: &[f64]) -> f64 {
    device
        .iter()
        .zip(reference.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Test 1: device QueryRMSE der == CPU `calc_ders_for_queries` within ε=1e-4.
#[test]
fn query_rmse_der_matches_cpu_within_epsilon() {
    let (approx, target, q_offsets) = uneven_fixture();
    let groups = group_spans(&q_offsets);
    let cpu = calc_ders_for_queries(&Loss::QueryRmse, &approx, &target, &[], &groups, 0)
        .expect("CPU QueryRMSE der must succeed on the in-range fixture");
    let (cpu_der1, cpu_der2) = flatten(&cpu);

    let (dev_der1, dev_der2) = query_rmse_ders_host(&approx, &target, &[], &q_offsets)
        .expect("device QueryRMSE der must succeed on the in-range fixture");

    assert_eq!(dev_der1.len(), cpu_der1.len(), "der1 length must match CPU");
    assert_eq!(dev_der2.len(), cpu_der2.len(), "der2 length must match CPU");
    assert!(
        dev_der1.iter().all(|v| v.is_finite()) && dev_der2.iter().all(|v| v.is_finite()),
        "device QueryRMSE der must be finite"
    );

    let d1 = max_abs_divergence(&dev_der1, &cpu_der1);
    let d2 = max_abs_divergence(&dev_der2, &cpu_der2);
    println!(
        "[ranking QueryRMSE] der1 max_div={d1:e} der2 max_div={d2:e} (device_backend_active={})",
        device_backend_active()
    );
    if device_backend_active() {
        assert!(d1 <= TOL, "device QueryRMSE der1 diverged from CPU: {d1:e} > {TOL:e}");
        assert!(d2 <= TOL, "device QueryRMSE der2 diverged from CPU: {d2:e} > {TOL:e}");
    }
}

/// Test 2: device QuerySoftMax der == CPU `calc_ders_for_queries` within ε=1e-4.
#[test]
fn query_softmax_der_matches_cpu_within_epsilon() {
    let (approx, target, q_offsets) = uneven_fixture();
    let groups = group_spans(&q_offsets);
    // The catboost QuerySoftMax defaults: beta = 1.0, lambda = 0.01.
    let beta = 1.0_f64;
    let lambda = 0.01_f64;
    let cpu = calc_ders_for_queries(
        &Loss::QuerySoftMax { lambda, beta },
        &approx,
        &target,
        &[],
        &groups,
        0,
    )
    .expect("CPU QuerySoftMax der must succeed on the in-range fixture");
    let (cpu_der1, cpu_der2) = flatten(&cpu);

    let (dev_der1, dev_der2) =
        query_softmax_ders_host(&approx, &target, &[], &q_offsets, beta, lambda)
            .expect("device QuerySoftMax der must succeed on the in-range fixture");

    assert_eq!(dev_der1.len(), cpu_der1.len(), "der1 length must match CPU");
    assert_eq!(dev_der2.len(), cpu_der2.len(), "der2 length must match CPU");
    assert!(
        dev_der1.iter().all(|v| v.is_finite()) && dev_der2.iter().all(|v| v.is_finite()),
        "device QuerySoftMax der must be finite"
    );

    let d1 = max_abs_divergence(&dev_der1, &cpu_der1);
    let d2 = max_abs_divergence(&dev_der2, &cpu_der2);
    println!(
        "[ranking QuerySoftMax] der1 max_div={d1:e} der2 max_div={d2:e} (device_backend_active={})",
        device_backend_active()
    );
    if device_backend_active() {
        assert!(d1 <= TOL, "device QuerySoftMax der1 diverged from CPU: {d1:e} > {TOL:e}");
        assert!(d2 <= TOL, "device QuerySoftMax der2 diverged from CPU: {d2:e} > {TOL:e}");
    }
}

/// Test 3: QueryCrossEntropy is INDEPENDENTLY gated off (Open Q3) — its coverage flag is `false`
/// (→ `Ok(None)` at the session level) WITHOUT disabling QueryRMSE / QuerySoftMax — and its bounded
/// per-query shift search is a self-consistent root-find.
#[test]
fn query_cross_entropy_gated_off_and_shift_search_self_consistent() {
    // (a) Independent gate: QueryCrossEntropy is NOT covered; QueryRMSE / QuerySoftMax ARE. So the
    // session ranking gate maps QueryCrossEntropy to Ok(None) without touching the covered arms.
    assert!(
        !ranking_objective_covered(RankingObjective::QueryCrossEntropy),
        "QueryCrossEntropy must be INDEPENDENTLY gated off (Open Q3) — coverage flag must be false"
    );
    assert!(
        ranking_objective_covered(RankingObjective::QueryRmse),
        "QueryRMSE must stay covered even though QueryCrossEntropy is deferred"
    );
    assert!(
        ranking_objective_covered(RankingObjective::QuerySoftMax { beta: 1.0, lambda: 0.01 }),
        "QuerySoftMax must stay covered even though QueryCrossEntropy is deferred"
    );

    // (b) Bounded shift-search self-consistency: with fractional (probability-like) targets a
    // feasible per-query shift exists solving F(b) = Σ w·sigmoid(approx + b) = Σ w·target. The
    // returned shift must satisfy that equation (a genuine root-find, NOT a der-parity claim — the
    // full der oracle is deferred with the arm).
    let approx = vec![0.5, -0.2, 1.3, 0.1, 0.4, -0.7, 0.9, 0.2, -0.3];
    let target = vec![0.5, 0.2, 0.8, 0.3, 0.6, 0.1, 0.9, 0.4, 0.7];
    let q_offsets = vec![0u32, 3, 5, 9];

    let shifts = query_cross_entropy_shifts_host(&approx, &target, &[], &q_offsets)
        .expect("QueryCrossEntropy shift search must succeed on the in-range fixture");
    assert_eq!(shifts.len(), q_offsets.len() - 1, "one shift per query");
    assert!(shifts.iter().all(|s| s.is_finite()), "every per-query shift must be finite");

    let mut worst = 0.0_f64;
    for (g, w) in q_offsets.windows(2).enumerate() {
        let (begin, end) = (w[0] as usize, w[1] as usize);
        let shift = shifts.get(g).copied().unwrap_or(0.0);
        let mut f = 0.0_f64;
        let mut t = 0.0_f64;
        for i in begin..end {
            let a = approx.get(i).copied().unwrap_or(0.0);
            f += 1.0 / (1.0 + (-(a + shift)).exp());
            t += target.get(i).copied().unwrap_or(0.0);
        }
        worst = worst.max((f - t).abs());
    }
    println!(
        "[ranking QueryCrossEntropy] shift self-consistency worst |F-T|={worst:e} \
         (device_backend_active={})",
        device_backend_active()
    );
    if device_backend_active() {
        assert!(
            worst <= 1e-3,
            "QueryCrossEntropy bounded shift search did not converge: worst |F-T|={worst:e} > 1e-3"
        );
    }
}
