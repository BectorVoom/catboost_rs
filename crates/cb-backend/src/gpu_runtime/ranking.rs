//! GPUT-22 (Phase 13 Plan 04, Wave 4): the deterministic **query/listwise objective device
//! driver** — QueryRMSE, QuerySoftMax, and QueryCrossEntropy — transcribing the CPU
//! `cb_compute::ranking_der::calc_ders_for_queries` der math onto the device path over the
//! Plan-03 shared query-grouping infrastructure ([`crate::kernels::query_helper`]). This is the
//! amortization payoff of that grouping substrate: the group means / group max / per-query bias
//! removal the ranking der needs are ALL computed by the Plan-03 kernels; this driver adds only the
//! per-objective pointwise der arithmetic (RMSE), the per-query softmax der (QuerySoftMax), and the
//! bounded per-query shift search (QueryCrossEntropy).
//!
//! # What lives here (production, NOT `#[cfg(test)]`)
//!
//! - [`query_rmse_ders_host`] — QueryRMSE: `RemoveGroupMeans` on the resident residual (the
//!   `queryAvrg` mean-removal from the Plan-03 group-means kernel) then the pointwise
//!   `der1 = (residual − queryAvrg)·w`, `der2 = −w` in [`ranking_rmse_der_kernel`].
//! - [`query_softmax_ders_host`] — QuerySoftMax: the per-query softmax der using the Plan-03
//!   `ComputeGroupMax` shift + the weighted exp-share `p`, in [`query_softmax_der_kernel`].
//! - [`query_cross_entropy_shifts_host`] — QueryCrossEntropy's per-query bisection + Newton shift
//!   search ([`query_cross_entropy_shift_kernel`]), a BOUNDED root-find (Open Q3). This arm is
//!   INDEPENDENTLY gated: it is deferred (not in the covered ranking set, [`ranking_objective_covered`])
//!   because it has no `cb_compute::ranking_der` CPU der oracle / `Loss` variant yet, so it can be
//!   shipped structurally without over-running QueryRMSE / QuerySoftMax.
//!
//! # Parity discipline — the covered regime is uniform-weight
//!
//! The covered device ranking regime is UNIFORM object weights (`weights` empty → `1.0` column, the
//! same expansion [`crate::kernels::query_helper::compute_group_means_host`] performs). Under
//! uniform weights the Plan-03 `ComputeGroupMax` (max over ALL objects) equals the CPU's
//! max-over-`weight>0` seed, and the group means fixed-point reduction matches
//! `group_reduce_weighted` / `cb_core::sum_f64` within ε=1e-4 (proven by the Plan-03 self-oracle).
//! Each per-query serial der loop walks `[begin, end)` in ascending doc order — the SAME order the
//! CPU `calc_ders_for_queries` per-group loop uses — so the serial single-thread accumulation is
//! deterministic and order-matches the CPU reference by construction (no atomic needed).
//!
//! # f64/u64-typed seam (WR-01)
//!
//! The der accumulates in f64 and the group infra uses u64 fixed-point / u64 RNG; WGSL has neither
//! f64 nor u64, so a genuine `wgpu` backend surfaces a typed [`CbError::OutOfRange`] rather than an
//! opaque JIT crash (the [`crate::kernels::query_helper`] / [`crate::kernels::mvs_device`]
//! precedent). No `-inf` literal in any `#[cube]` body (the shift search uses a FINITE bracket). No
//! `unwrap`/`expect`/`panic`/indexing in production (workspace lints + D-13).

use cubecl::prelude::*;
use cubecl::server::Handle;

use cb_core::{CbError, CbResult};

use crate::kernels::query_helper::{
    compute_group_ids_host, compute_group_max_host, compute_group_means_host,
    remove_group_means_host,
};
use crate::SelectedRuntime;

/// The deterministic query/listwise objective this driver computes the der for. QueryRMSE and
/// QuerySoftMax are the COVERED deterministic arms (a `cb_compute::ranking_der` CPU der oracle
/// exists for each — [`ranking_objective_covered`] returns `true`). QueryCrossEntropy is the
/// INDEPENDENTLY-DEFERRED arm (Open Q3): its bounded per-query shift search is landed structurally
/// but it has no CPU der oracle / `Loss` variant yet, so it is NOT covered (the session gate returns
/// `Ok(None)` for it without disabling QueryRMSE / QuerySoftMax).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum RankingObjective {
    /// QueryRMSE — per-query mean-removed residual (`error_functions.h:879-933`).
    QueryRmse,
    /// QuerySoftMax — per-query weighted softmax der (`error_functions.cpp:540-576`).
    QuerySoftMax {
        /// The inverse-temperature `Beta` (`QuerySoftMax:beta`, default `1.0`).
        beta: f64,
        /// The L2 regularization `LambdaReg` (`QuerySoftMax:lambda`, default `0.01`).
        lambda: f64,
    },
    /// QueryCrossEntropy — per-query bisection/Newton shift search (Open Q3, INDEPENDENTLY gated
    /// off: no CPU der oracle yet, so it is deferred rather than shipped as unverified der parity).
    QueryCrossEntropy,
}

/// Whether `objective` is in the COVERED deterministic ranking set (has a `cb_compute::ranking_der`
/// CPU der oracle this driver is self-oracled against). QueryRMSE / QuerySoftMax → `true`;
/// QueryCrossEntropy → `false` (Open Q3, independently deferred behind its own `Ok(None)` gate so
/// QueryRMSE / QuerySoftMax ship even though QueryCrossEntropy's der oracle is not landed).
#[must_use]
pub(crate) fn ranking_objective_covered(objective: RankingObjective) -> bool {
    match objective {
        RankingObjective::QueryRmse | RankingObjective::QuerySoftMax { .. } => true,
        RankingObjective::QueryCrossEntropy => false,
    }
}

/// QueryCrossEntropy shift-search BISECTION iteration budget (Open Q3, T-13-07 DoS mitigation): a
/// FIXED count (mirrors the `mvs_device` `MVS_BISECTION_ITERS` monotone-bracket precedent) so the
/// per-query loop can never run unbounded. Upstream `query_cross_entropy.cu` runs 8 bisection
/// halvings; a `2·BRACKET / 2^8` bracket is < 0.24 before Newton refines.
const QCE_SHIFT_BISECTION_ITERS: u32 = 8;

/// QueryCrossEntropy shift-search NEWTON refinement budget (Open Q3, T-13-07): a FIXED count of
/// Newton steps after the bisection bracket (upstream runs 5). Bounded — never unbounded.
const QCE_SHIFT_NEWTON_ITERS: u32 = 5;

/// The FINITE shift-search bracket `[−BRACKET, +BRACKET]` (Pattern D — NO `-inf` literal in a
/// `#[cube]` body). A logit shift of ±30 saturates every sigmoid, bracketing any feasible root.
const QCE_SHIFT_BRACKET: f64 = 30.0;

// ===========================================================================
// #[cube] ranking der kernels
// ===========================================================================

/// QueryRMSE pointwise der over the per-query MEAN-REMOVED residual `centered[d] = (target[d] −
/// approx[d]) − queryAvrg[qid]` (the `RemoveGroupMeans` output from the Plan-03 grouping infra) and
/// the per-object `weights[d]`. Doc-parallel (`ABSOLUTE_POS`-indexed):
/// `der1[d] = centered[d]·w`, `der2[d] = −1·w` (`error_functions.h:901-907`
/// `TQueryRmseError::CalcDersForQueries` — the querywise der folds the weight IN). Order-independent
/// (each doc writes its own slot).
#[cube(launch)]
fn ranking_rmse_der_kernel(
    centered: &Array<f64>,
    weights: &Array<f64>,
    der1: &mut Array<f64>,
    der2: &mut Array<f64>,
) {
    let d = ABSOLUTE_POS;
    if d < centered.len() {
        let w = weights[d];
        der1[d] = centered[d] * w;
        der2[d] = -1.0f64 * w;
    }
}

/// QuerySoftMax per-query der (`error_functions.cpp:540-576` `TQuerySoftMaxError`). Serial (unit 0)
/// per query, consuming the Plan-03 `ComputeGroupMax` shift `group_max[g]` (the max-shift before
/// `exp`, applied per query). `params = [beta, lambda]`. For each query `g` over `[begin, end)`:
/// - `sumWTargets = Σ_{w>0, t>0} w·t`; if `≤ 0` every der in the query is `0` (no divide).
/// - else `sumExpApprox = Σ exp(Beta·(approx − maxApprox))·w`, `p = expApprox·w / sumExpApprox`,
///   `der2 = Beta·sumWTargets·(Beta·p·(p−1) − LambdaReg)`, `der1 = Beta·(−sumWTargets·p + w·t)` for
///   `w>0`; `der1 = der2 = 0` for `w≤0`.
/// The serial `[begin, end)` walk matches the CPU per-group ascending order exactly (deterministic).
#[cube(launch)]
fn query_softmax_der_kernel(
    approx: &Array<f64>,
    target: &Array<f64>,
    weights: &Array<f64>,
    q_offsets: &Array<u32>,
    group_max: &Array<f64>,
    params: &Array<f64>,
    der1: &mut Array<f64>,
    der2: &mut Array<f64>,
) {
    if ABSOLUTE_POS == 0 {
        let beta = params[0];
        let lambda = params[1];
        let n_groups = q_offsets.len() - 1;
        let mut g = 0usize;
        while g < n_groups {
            let begin = q_offsets[g];
            let end = q_offsets[g + 1];
            let max_a = group_max[g];

            // sumWTargets = Σ_{w>0, t>0} w·t (max-shift is applied below; this term is shift-free).
            let mut sum_wt = 0.0f64;
            let mut d = begin;
            while d < end {
                let w = weights[d as usize];
                let t = target[d as usize];
                if w > 0.0f64 {
                    if t > 0.0f64 {
                        sum_wt += w * t;
                    }
                }
                d += 1u32;
            }

            if sum_wt > 0.0f64 {
                // sumExpApprox = Σ exp(Beta·(approx − maxApprox))·w.
                let mut sum_exp = 0.0f64;
                let mut de = begin;
                while de < end {
                    let w = weights[de as usize];
                    let a = approx[de as usize];
                    sum_exp += f64::exp(beta * (a - max_a)) * w;
                    de += 1u32;
                }
                let mut dd = begin;
                while dd < end {
                    let w = weights[dd as usize];
                    let mut e1 = 0.0f64;
                    let mut e2 = 0.0f64;
                    if w > 0.0f64 {
                        if sum_exp > 0.0f64 {
                            let a = approx[dd as usize];
                            let t = target[dd as usize];
                            let p = (f64::exp(beta * (a - max_a)) * w) / sum_exp;
                            e2 = beta * sum_wt * (beta * p * (p - 1.0f64) - lambda);
                            e1 = beta * (-sum_wt * p + w * t);
                        }
                    }
                    der1[dd as usize] = e1;
                    der2[dd as usize] = e2;
                    dd += 1u32;
                }
            } else {
                // sumWTargets ≤ 0 → every der in the query is 0 (error_functions.cpp:571-576).
                let mut dz = begin;
                while dz < end {
                    der1[dz as usize] = 0.0f64;
                    der2[dz as usize] = 0.0f64;
                    dz += 1u32;
                }
            }
            g += 1usize;
        }
    }
}

/// QueryCrossEntropy per-query BOUNDED shift search (Open Q3; T-13-07 DoS mitigation). Serial (unit
/// 0) per query. Finds the per-query logit shift `b` solving the monotone equation
/// `F(b) = Σ w·sigmoid(approx + b) = Σ w·target` by a FIXED-count bisection over the FINITE bracket
/// `[−BRACKET, +BRACKET]` (no `-inf` literal, Pattern D) followed by a FIXED-count Newton refine
/// (`F'(b) = Σ w·p·(1−p)`). Both loop counts are compile-time constants — the search can never run
/// unbounded (the DoS threat). `shift_out[g]` is the per-query shift.
///
/// NOTE: this arm is INDEPENDENTLY DEFERRED ([`ranking_objective_covered`] returns `false` for
/// QueryCrossEntropy) — the shift search is a genuine, self-consistent root-find (verifiable
/// `F(shift) ≈ Σ w·t`), but the full QueryCrossEntropy der has no `cb_compute::ranking_der` CPU
/// oracle / `Loss` variant yet, so the session gate declines it rather than shipping unverified der
/// parity. Landing the bounded search here keeps QueryCrossEntropy ready without over-running the
/// covered QueryRMSE / QuerySoftMax arms.
#[cube(launch)]
fn query_cross_entropy_shift_kernel(
    approx: &Array<f64>,
    target: &Array<f64>,
    weights: &Array<f64>,
    q_offsets: &Array<u32>,
    params: &Array<f64>,
    shift_out: &mut Array<f64>,
) {
    if ABSOLUTE_POS == 0 {
        // The FINITE bracket is a RUNTIME scalar (params[0]) — NOT the comptime const directly:
        // seeding `lo`/`hi` from a comptime f64 makes them comptime-typed, which then clashes when
        // reassigned from the runtime `mid` (a `NativeExpand<f64>` type mismatch). The host passes
        // `QCE_SHIFT_BRACKET` in as runtime data (the `mvs_device` runtime-bracket precedent).
        let bracket = params[0];
        let n_groups = q_offsets.len() - 1;
        let mut g = 0usize;
        while g < n_groups {
            let begin = q_offsets[g];
            let end = q_offsets[g + 1];

            // Target mass T = Σ w·target (the value F(b) must reach).
            let mut t_mass = 0.0f64;
            let mut d = begin;
            while d < end {
                t_mass += weights[d as usize] * target[d as usize];
                d += 1u32;
            }

            // Bisection over the FINITE bracket; F(b) = Σ w·sigmoid(approx + b) is monotone in b.
            let mut lo = 0.0f64 - bracket;
            let mut hi = bracket;
            let mut it = 0u32;
            while it < QCE_SHIFT_BISECTION_ITERS {
                let mid = 0.5f64 * (lo + hi);
                let mut f = 0.0f64;
                let mut k = begin;
                while k < end {
                    let z = approx[k as usize] + mid;
                    let p = 1.0f64 / (1.0f64 + f64::exp(0.0f64 - z));
                    f += weights[k as usize] * p;
                    k += 1u32;
                }
                if f < t_mass {
                    lo = mid;
                } else {
                    hi = mid;
                }
                it += 1u32;
            }
            let mut b = 0.5f64 * (lo + hi);

            // Newton refine (bounded): b ← b − (F(b) − T) / F'(b).
            let mut nit = 0u32;
            while nit < QCE_SHIFT_NEWTON_ITERS {
                let mut f = 0.0f64;
                let mut fp = 0.0f64;
                let mut k = begin;
                while k < end {
                    let z = approx[k as usize] + b;
                    let p = 1.0f64 / (1.0f64 + f64::exp(0.0f64 - z));
                    let w = weights[k as usize];
                    f += w * p;
                    fp += w * p * (1.0f64 - p);
                    k += 1u32;
                }
                if fp > 0.0f64 {
                    let num = f - t_mass;
                    let step = num / fp;
                    b = b - step;
                }
                nit += 1u32;
            }
            shift_out[g] = b;
            g += 1usize;
        }
    }
}

// ===========================================================================
// Host launch wrappers
// ===========================================================================

/// Reject the (impossible) wgpu f64/u64 path with a typed error (WR-01), mirroring
/// [`crate::kernels::query_helper`]. Kept in one place so every entry point agrees.
#[cfg(feature = "wgpu")]
fn wgpu_reject() -> CbError {
    CbError::OutOfRange(
        "device ranking der requires f64 + u64 device channels; the wgpu backend has neither \
         (WR-01). Use the rocm/cuda/cpu backend for the query der."
            .to_owned(),
    )
}

/// The selected-runtime client (one per call, mirroring [`crate::kernels::query_helper`]).
#[cfg(not(feature = "wgpu"))]
fn selected_client() -> cubecl::client::ComputeClient<SelectedRuntime> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    <SelectedRuntime as cubecl::Runtime>::client(&device)
}

/// Expand the caller's `weights` to a uniform `1.0` column when empty (the covered regime), else
/// validate the length. Mirrors [`crate::kernels::query_helper::compute_group_means_host`] so the
/// der driver and the group reductions agree on the weight column.
fn weight_column(weights: &[f64], n: usize) -> CbResult<Vec<f64>> {
    if weights.is_empty() {
        return Ok(vec![1.0; n]);
    }
    if weights.len() != n {
        return Err(CbError::Degenerate(format!(
            "ranking der: weights len {} != n {n}",
            weights.len()
        )));
    }
    Ok(weights.to_vec())
}

/// Validate the flat ranking der inputs (shared by every objective). `approx` / `target` are length
/// `n`; `q_offsets` has `n_groups + 1` entries covering `[0, n)`.
fn validate_ranking_inputs(approx: &[f64], target: &[f64], q_offsets: &[u32]) -> CbResult<()> {
    if approx.len() != target.len() {
        return Err(CbError::Degenerate(format!(
            "ranking der: approx len {} != target len {}",
            approx.len(),
            target.len()
        )));
    }
    if q_offsets.is_empty() {
        return Err(CbError::Degenerate(
            "ranking der: q_offsets must have n_groups + 1 entries (got 0)".to_owned(),
        ));
    }
    Ok(())
}

/// QueryRMSE device der: the per-query mean-removed residual then the pointwise der. Composes the
/// Plan-03 grouping infra — `ComputeGroupMeans` (the `queryAvrg` numerator/denominator fixed-point
/// reduction), `ComputeGroupIds`, `RemoveGroupMeans` (`residual − queryAvrg`) — with
/// [`ranking_rmse_der_kernel`]. Returns `(der1, der2)` (each length `n`), matching
/// `cb_compute::calc_ders_for_queries` for `Loss::QueryRmse` within ε=1e-4.
pub(crate) fn query_rmse_ders_host(
    approx: &[f64],
    target: &[f64],
    weights: &[f64],
    q_offsets: &[u32],
) -> CbResult<(Vec<f64>, Vec<f64>)> {
    validate_ranking_inputs(approx, target, q_offsets)?;
    let n = approx.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    #[cfg(feature = "wgpu")]
    {
        let _ = (weights, q_offsets);
        return Err(wgpu_reject());
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let weight_col = weight_column(weights, n)?;
        // residual[i] = target[i] - approx[i] (input prep; the group reduction is on-device).
        let residuals: Vec<f64> = approx
            .iter()
            .zip(target.iter())
            .map(|(&a, &t)| t - a)
            .collect();
        // queryAvrg per group = Σ(residual·w)/Σw (Plan-03 fixed-point group-means reduction).
        let query_avrg = compute_group_means_host(&residuals, &weight_col, q_offsets)?;
        // qids[d] = the doc's query id (Plan-03 ComputeGroupIds).
        let qids = compute_group_ids_host(q_offsets, n)?;
        // centered[d] = residual[d] - queryAvrg[qids[d]] (Plan-03 RemoveGroupMeans).
        let centered = remove_group_means_host(&residuals, &qids, &query_avrg)?;

        // Pointwise der: der1 = centered·w, der2 = -w.
        let client = selected_client();
        let centered_h = client.create(cubecl::bytes::Bytes::from_elems(centered));
        let w_h = client.create(cubecl::bytes::Bytes::from_elems(weight_col));
        let der1 = client.empty(n * std::mem::size_of::<f64>());
        let der2 = client.empty(n * std::mem::size_of::<f64>());
        let block = 64u32;
        let cubes = n.div_ceil(block as usize).max(1) as u32;
        ranking_rmse_der_kernel::launch::<SelectedRuntime>(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim { x: block, y: 1, z: 1 },
            unsafe { ArrayArg::from_raw_parts(centered_h, n) },
            unsafe { ArrayArg::from_raw_parts(w_h, n) },
            unsafe { ArrayArg::from_raw_parts(der1.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(der2.clone(), n) },
        );
        let der1_v = read_f64(&client, der1, "query_rmse_der1")?;
        let der2_v = read_f64(&client, der2, "query_rmse_der2")?;
        Ok((der1_v, der2_v))
    }
}

/// QuerySoftMax device der: the per-query weighted softmax der using the Plan-03 `ComputeGroupMax`
/// shift. Returns `(der1, der2)` (each length `n`), matching `cb_compute::calc_ders_for_queries` for
/// `Loss::QuerySoftMax { lambda, beta }` within ε=1e-4 (covered uniform-weight regime).
pub(crate) fn query_softmax_ders_host(
    approx: &[f64],
    target: &[f64],
    weights: &[f64],
    q_offsets: &[u32],
    beta: f64,
    lambda: f64,
) -> CbResult<(Vec<f64>, Vec<f64>)> {
    validate_ranking_inputs(approx, target, q_offsets)?;
    let n = approx.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    #[cfg(feature = "wgpu")]
    {
        let _ = (weights, q_offsets, beta, lambda);
        return Err(wgpu_reject());
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let weight_col = weight_column(weights, n)?;
        // Per-query max approx (Plan-03 ComputeGroupMax) — the max-shift before exp. Uniform-weight
        // covered regime: max over all objects == the CPU max-over-(weight>0) seed.
        let group_max = compute_group_max_host(approx, q_offsets)?;
        let n_groups = q_offsets.len().saturating_sub(1);

        let client = selected_client();
        let approx_h = client.create(cubecl::bytes::Bytes::from_elems(approx.to_vec()));
        let target_h = client.create(cubecl::bytes::Bytes::from_elems(target.to_vec()));
        let w_h = client.create(cubecl::bytes::Bytes::from_elems(weight_col));
        let off_h = client.create(cubecl::bytes::Bytes::from_elems(q_offsets.to_vec()));
        let max_h = client.create(cubecl::bytes::Bytes::from_elems(group_max));
        let params_h = client.create(cubecl::bytes::Bytes::from_elems(vec![beta, lambda]));
        let der1 = client.empty(n * std::mem::size_of::<f64>());
        let der2 = client.empty(n * std::mem::size_of::<f64>());
        // Serial single-thread launch (unit 0 loops the queries).
        query_softmax_der_kernel::launch::<SelectedRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 1, y: 1, z: 1 },
            unsafe { ArrayArg::from_raw_parts(approx_h, n) },
            unsafe { ArrayArg::from_raw_parts(target_h, n) },
            unsafe { ArrayArg::from_raw_parts(w_h, n) },
            unsafe { ArrayArg::from_raw_parts(off_h, q_offsets.len()) },
            unsafe { ArrayArg::from_raw_parts(max_h, n_groups) },
            unsafe { ArrayArg::from_raw_parts(params_h, 2) },
            unsafe { ArrayArg::from_raw_parts(der1.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(der2.clone(), n) },
        );
        let der1_v = read_f64(&client, der1, "query_softmax_der1")?;
        let der2_v = read_f64(&client, der2, "query_softmax_der2")?;
        Ok((der1_v, der2_v))
    }
}

/// QueryCrossEntropy per-query BOUNDED shift search (Open Q3; INDEPENDENTLY deferred). Returns the
/// per-query logit shifts (length `n_groups`) solving `Σ w·sigmoid(approx + shift) = Σ w·target` via
/// the fixed-count bisection + Newton [`query_cross_entropy_shift_kernel`]. This is a genuine,
/// self-consistent root-find (the returned shift satisfies the equation) — NOT a der claim; the full
/// QueryCrossEntropy der stays gated off until its CPU oracle lands.
pub(crate) fn query_cross_entropy_shifts_host(
    approx: &[f64],
    target: &[f64],
    weights: &[f64],
    q_offsets: &[u32],
) -> CbResult<Vec<f64>> {
    validate_ranking_inputs(approx, target, q_offsets)?;
    let n = approx.len();
    let n_groups = q_offsets.len().saturating_sub(1);
    if n == 0 || n_groups == 0 {
        return Ok(Vec::new());
    }
    #[cfg(feature = "wgpu")]
    {
        let _ = (weights, q_offsets);
        return Err(wgpu_reject());
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let weight_col = weight_column(weights, n)?;
        let client = selected_client();
        let approx_h = client.create(cubecl::bytes::Bytes::from_elems(approx.to_vec()));
        let target_h = client.create(cubecl::bytes::Bytes::from_elems(target.to_vec()));
        let w_h = client.create(cubecl::bytes::Bytes::from_elems(weight_col));
        let off_h = client.create(cubecl::bytes::Bytes::from_elems(q_offsets.to_vec()));
        // The FINITE shift bracket passed as runtime data (see the kernel comment).
        let params_h = client.create(cubecl::bytes::Bytes::from_elems(vec![QCE_SHIFT_BRACKET]));
        let out = client.empty(n_groups * std::mem::size_of::<f64>());
        query_cross_entropy_shift_kernel::launch::<SelectedRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 1, y: 1, z: 1 },
            unsafe { ArrayArg::from_raw_parts(approx_h, n) },
            unsafe { ArrayArg::from_raw_parts(target_h, n) },
            unsafe { ArrayArg::from_raw_parts(w_h, n) },
            unsafe { ArrayArg::from_raw_parts(off_h, q_offsets.len()) },
            unsafe { ArrayArg::from_raw_parts(params_h, 1) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), n_groups) },
        );
        read_f64(&client, out, "query_cross_entropy_shift")
    }
}

/// Read a resident `f64` handle back to host, mapping a failure to [`CbError::Degenerate`] (WR-05),
/// never a silent zero buffer.
#[cfg(not(feature = "wgpu"))]
fn read_f64(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    handle: Handle,
    who: &str,
) -> CbResult<Vec<f64>> {
    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("ranking {who} f64 read-back failed: {e:?}")))?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}
