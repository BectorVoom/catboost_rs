//! GPUT-02/03: the per-fit device-resident training session — the residency wrapper that
//! owns ONE [`cubecl::client::ComputeClient`] plus the persistent device handles for the
//! whole fit (the quantized matrix in BOTH the packed-cindex `words` layout the histogram
//! reads AND the plain feature-major layout the partition split reads, the per-feature
//! TCFeature table, the weights, the object-visiting indices, the target, and the running
//! `approx`/`der1`), uploaded ONCE at [`GpuTrainSession::begin`] and cloned into the
//! resident grow geometry ([`super::grow_oblivious_tree_resident`]) per tree — NO per-tree
//! re-upload.
//!
//! # Residency invariants (T-10-18 / Pitfall 3)
//!
//! Every handle is allocated by, and read back ONLY through, `self.client` — a CubeCL
//! Handle is bound to its originating client, so the session threads that ONE client
//! through every launch and read-back. `end` (Drop) frees the client + handles
//! deterministically.
//!
//! # No-read-back contract (must-have truth 3 / D-05)
//!
//! The running `approx` is updated ON DEVICE (`apply_leaf_delta`) and the residual `der1`
//! is recomputed device-side from it (the resident der seam); NO n-length der1/approx
//! read-back crosses per tree — only the O(1) BestSplit per level, the `2^depth` part-stats,
//! and ONE `leaf_of` read-back at the end of each tree (the structure oracle seam, the SAME
//! crossing class as the part-stats — the 10-02 `DeviceGrownTree` contract).
//!
//! # Coverage gate (D-10-01/02)
//!
//! `begin` runs the classification where the session is created: it returns `None`
//! (→ CPU fallback, D-04) unless `depth >= 1` AND the loss is RMSE/Logloss/CrossEntropy AND
//! Plain boosting AND `fold_count == 1` AND the score function is one of the supported five.
//! (Phase 12 Plan 01 / GPUT-18: depth>1 is device-covered via the Phase-11 partition-aware
//! substrate — the former depth-1-only restriction is lifted.)
//! Cosine is the depth-1 device default (GPUT-08), honored from the passed `EScoreFunction`.
//!
//! # Landmine (project memory)
//!
//! cb-backend must NEVER gain a `cb-train` dependency (the feature-unification landmine);
//! this session reaches only `cb_compute` (a normal dep) for the host types
//! (`Loss`/`EScoreFunction`/`DeviceGrownTree`) and transcribes any CPU reference inline.

use cubecl::server::Handle;

use cb_compute::{
    DeviceBootstrapType, DeviceCtrConfig, DeviceGrownTree, DeviceGrowPolicy, DeviceTrainConfig,
    EScoreFunction, Loss,
};
use cb_core::{CbError, CbResult, TFastRng64};

use crate::gpu_runtime::cindex::pack_cindex;
use crate::gpu_runtime::{
    grow_oblivious_tree_resident, launch_der_binary_resident, upload_channel_floats,
    DerBinaryKernel, PAIRWISE_BUCKET_WEIGHT_PRIOR_REG_DEFAULT,
};
use crate::kernels::bootstrap_device::{
    fold_weights_resident, launch_bootstrap_weights_resident, DeviceBootstrapKind,
};
use crate::kernels::mvs_device::launch_mvs_weights_resident;
use crate::kernels::ctr_device::{
    binarize_ctr_column_resident, combine_projection_bins, launch_ordered_ctr_resident,
};
use crate::kernels::exact_quantile::device_exact_leaf_delta;
use crate::kernels::nonsym_grow::{grow_nonsym_tree, NonsymPolicy};
use crate::kernels::region_device::grow_region_tree;
use crate::kernels::{SCORE_FN_COSINE, SCORE_FN_L2, SCORE_FN_LOO_L2, SCORE_FN_SAT_L2, SCORE_FN_SOLAR_L2};
use crate::SelectedRuntime;

/// Map a host [`DeviceGrowPolicy`] to the device non-symmetric grow strategy, or `None`
/// for the oblivious / symmetric path (which stays on the resident grow loop) and for the
/// not-yet-covered Region policy (Plan 04). Follows the `map_score_fn` / `map_der_kernel`
/// `Option`-returning template (Pattern A — the family-gated coverage flip). Phase 12 Plan 03
/// flips Depthwise / Lossguide `Ok(None)` → device; Region stays `None` until Plan 04.
fn map_grow_policy(grow_policy: DeviceGrowPolicy) -> Option<NonsymPolicy> {
    match grow_policy {
        DeviceGrowPolicy::Depthwise => Some(NonsymPolicy::Depthwise),
        DeviceGrowPolicy::Lossguide => Some(NonsymPolicy::Lossguide),
        // SymmetricTree → the resident oblivious path (not a non-sym policy); Region → Plan 04.
        DeviceGrowPolicy::SymmetricTree | DeviceGrowPolicy::Region => None,
    }
}

/// The bootstrap gate arm outcome (Phase 12 Plan 06, GPUT-09; Plan 07 GPUT-17) — the `map_score_fn`
/// / `map_grow_policy` `Option`-returning template, widened because `No` is a covered non-draw (NOT
/// a decline) and MVS is its OWN device path (derivative reduction, distinct from the RNG-only draw):
/// - `NoDraw` — `bootstrap_type == No`: the byte-unchanged covered default (no device draw).
/// - `Device(kind)` — Bernoulli/Bayesian/Poisson: draw the device-resident sample per tree.
/// - `Mvs` — Minimal-Variance Sampling (Plan 07): the device per-block threshold + reweight over
///   the resident derivatives (covered when the caller supplies `mvs_lambda`; else declines).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BootstrapArm {
    NoDraw,
    Device(DeviceBootstrapKind),
    Mvs,
}

/// Map a host [`DeviceBootstrapType`] to the device bootstrap arm (Pattern A). Bernoulli/Bayesian/
/// Poisson flip to the device RNG draw (Plan 06); MVS routes to its device reduction (Plan 07); `No`
/// is the covered non-draw default.
fn map_bootstrap_kernel(bootstrap_type: DeviceBootstrapType) -> BootstrapArm {
    match bootstrap_type {
        DeviceBootstrapType::No => BootstrapArm::NoDraw,
        DeviceBootstrapType::Bernoulli => BootstrapArm::Device(DeviceBootstrapKind::Bernoulli),
        DeviceBootstrapType::Bayesian => BootstrapArm::Device(DeviceBootstrapKind::Bayesian),
        DeviceBootstrapType::Poisson => BootstrapArm::Device(DeviceBootstrapKind::Poisson),
        DeviceBootstrapType::Mvs => BootstrapArm::Mvs,
    }
}

/// `mean(|der|)^2` — the MVS iter-0 `GetLambda(...)` (`mvs.cpp:37-79`): the squared mean gradient
/// magnitude via the ordered [`cb_core::sum_f64`] (D-05). Transcribed inline (NEVER a `cb-train`
/// dep, Pattern B). Used when the caller does not pin `config.mvs_lambda`.
fn mvs_lambda_from_der(derivatives: &[f64]) -> f64 {
    if derivatives.is_empty() {
        return 0.0;
    }
    let mags: Vec<f64> = derivatives.iter().map(|&d| (d * d).sqrt()).collect();
    let mean = cb_core::sum_f64(&mags) / derivatives.len() as f64;
    mean * mean
}

/// Whether a single-permutation CTR config is DEVICE-COVERED this wave (Phase 12 Plan 08, GPUT-10,
/// Pattern A). Covered when: a CTR config is present; its permutation + target-class span all `n`
/// objects (the single-permutation regime, Open Q3); every CTR column has at least one member and
/// binarizes to EXACTLY `n_bins` buckets (`borders.len() + 1 == n_bins`) so the extra CTR cindex
/// columns join the UNIFORM-`n_bins` resident histogram cleanly; and the f64 CTR seam exists on
/// this backend (NOT wgpu, WR-02). A multi-fold / multi-permutation CTR is NOT covered — it is
/// declined by the `fold_count != 1` gate upstream (Open Q3, deferred behind `Ok(None)`). Every
/// OTHER family flag must still be the covered default (no bootstrap / MVS / exact / leaf-cap) —
/// D-10-01 all-or-nothing PER family; the caller ANDs those in.
fn ctr_covered(config: &DeviceTrainConfig, n: usize, n_bins: usize) -> bool {
    if cfg!(feature = "wgpu") {
        return false;
    }
    let Some(ctr) = config.ctr.as_ref() else {
        return false;
    };
    ctr.permutation.len() == n
        && ctr.target_class.len() == n
        && !ctr.columns.is_empty()
        && ctr
            .columns
            .iter()
            .all(|col| !col.member_bins.is_empty() && col.borders.len() + 1 == n_bins)
}

/// Compute the ADDITIONAL binarized-CTR cindex columns for a covered CTR config ON device (Phase 12
/// Plan 08, GPUT-10): for each CTR column, fold its member categories into the (combined) bin column
/// (A5), accumulate the ordered read-before-increment target statistic resident across the
/// permutation ([`launch_ordered_ctr_resident`]), binarize the CTR VALUES into cindex bins on device
/// ([`binarize_ctr_column_resident`]), and read back ONLY the final integer bin column (the CTR
/// VALUES never touch the host — the same host-pack-once discipline every plain feature follows,
/// A2). Returns one `Vec<u32>` (object order) per CTR column, each with bins in `0..n_bins`.
///
/// # Errors
/// [`CbError`] propagated from the device CTR launch / read-back; the caller only invokes this once
/// [`ctr_covered`] has confirmed the shape invariants, so the length guards here are defensive.
fn build_ctr_cindex_columns(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    ctr: &DeviceCtrConfig,
    n: usize,
) -> CbResult<Vec<Vec<u32>>> {
    let mut columns = Vec::with_capacity(ctr.columns.len());
    for col in &ctr.columns {
        // Fold the member categories into one (combined) bin column + its distinct-bucket count.
        let (bins, buckets) = if col.member_bins.len() == 1 {
            let single = col.member_bins.first().cloned().unwrap_or_default();
            let buckets = single.iter().copied().max().map_or(0, |m| m as usize + 1);
            (single, buckets)
        } else {
            combine_projection_bins(&col.member_bins, n)?
        };
        let res = launch_ordered_ctr_resident(
            client,
            &ctr.permutation,
            &bins,
            &ctr.target_class,
            col.prior,
            buckets,
            n,
        )?;
        let bin_h = binarize_ctr_column_resident(client, &res.value, &col.borders, n)?;
        let bytes = client
            .read_one(bin_h)
            .map_err(|e| CbError::Degenerate(format!("CTR cindex column read-back failed: {e:?}")))?;
        columns.push(bytemuck::cast_slice::<u8, u32>(&bytes).to_vec());
    }
    Ok(columns)
}

/// The per-fit device bootstrap state (Phase 12 Plan 06, GPUT-09). `Some` iff the fit committed to
/// a covered oblivious grow with a non-`No` `bootstrap_type`. Holds the CONTINUOUS training stream
/// on the validated [`cb_core::TFastRng64`] (seeded from `config.rng_seed`); [`GpuTrainSession::grow_one`]
/// snapshots its O(1) base state per tree, draws the device-resident sample, folds it into the
/// resident weight, then advances the stream by the draws that tree consumed — so the keep-mask /
/// weights stay device-resident (no per-tree host round-trip, D-08) while the host manages only the
/// O(1) stream position on the sanctioned RNG (never a `cb_train` dep).
struct BootstrapState {
    kind: DeviceBootstrapKind,
    /// The persistent continuous training stream (seeded once from `config.rng_seed`).
    rng: TFastRng64,
    /// Bernoulli/Poisson subsample rate (`config.sample_rate`).
    sample_rate: f64,
    /// Bayesian bagging temperature — the catboost default `1.0` (config carries no field yet;
    /// the covered device Bayesian regime uses the default, MVP-scoped).
    bagging_temperature: f64,
}

/// The per-fit device MVS state (Phase 12 Plan 07, GPUT-17). `Some` iff the fit committed to a
/// covered oblivious grow with `bootstrap_type == Mvs` (and the caller pinned `config.mvs_lambda`).
/// Like [`BootstrapState`] it holds the CONTINUOUS training stream on the validated
/// [`cb_core::TFastRng64`] (seeded from `config.rng_seed`); [`GpuTrainSession::grow_one`] takes the
/// ONE main-stream `GenRand()` as the per-tree `rand_seed`, draws the device-resident MVS sample
/// (per-block threshold + reweight over the RESIDENT derivatives), folds it into the resident
/// weight, then advances the stream by the draws that tree consumed (D-08 — no per-tree host
/// round-trip of the mask). The λ is `config.mvs_lambda` (the caller's `GetLambda(...)`), or the
/// iter-0 `mean(|der|)^2` derived from the resident derivatives when unpinned (MVP scope).
struct MvsState {
    /// The persistent continuous training stream (seeded once from `config.rng_seed`).
    rng: TFastRng64,
    /// The MVS subsample rate (`config.sample_rate`, f32-rounded downstream).
    sample_rate: f64,
    /// The caller-pinned `GetLambda(...)` (`config.mvs_lambda`); `None` ⇒ iter-0 `mean(|der|)^2`.
    lambda: Option<f64>,
    /// The covered-loss der kind, driving the host der re-derivation for the unpinned-λ path.
    der_kernel: DerBinaryKernel,
}

/// The per-fit non-symmetric device-grow state (Phase 12 Plan 03). Held when the fit commits
/// to a Depthwise / Lossguide device grow. Unlike the resident oblivious session this path is
/// host-DRIVEN (the per-node device split scorer reads a host doc-subset per candidate), so it
/// keeps HOST copies of the quantized bins + weights and re-derives the per-object der1 from
/// the caller's `approx` each tree via [`host_der1`] (the covered regime is unit weights,
/// bias 0, so `der1 = target - approx` for RMSE / `target - sigmoid(approx)` for Logloss
/// matches the CPU `compute_gradients` used by the reference model, bit-for-bit).
struct NonsymState {
    policy: NonsymPolicy,
    /// Feature-major quantized bins (`bins[f * n + i]`).
    bins: Vec<u32>,
    /// Per-object weight (the covered regime is all-`1.0`).
    weight: Vec<f64>,
    max_depth: usize,
    /// Leaf cap (Lossguide); `usize::MAX` == unbounded (Depthwise / no cap).
    max_leaves: usize,
    min_data_in_leaf: usize,
    /// The covered-loss der kind, driving the host der1 re-derivation.
    der_kernel: DerBinaryKernel,
}

/// The per-fit device Region-grow state (Phase 12 Plan 04, GPUT-18, D-03a). `Some` iff the
/// fit committed to a `grow_policy=Region` device grow. Like [`NonsymState`] the Region path
/// is host-DRIVEN (the per-frontier device split scorer reads a host doc-subset per level),
/// so it keeps HOST copies of the quantized bins + weights and re-derives the per-object der1
/// from the caller's `approx` each tree via [`host_der1`] (the covered regime is unit weights /
/// bias 0, so `der1 = target - approx` for RMSE / `target - sigmoid(approx)` for Logloss
/// matches the CPU `compute_gradients` bit-for-bit). Region has `MaxLeaves = max_depth + 1`;
/// there is no leaf cap (unlike Lossguide), so no `max_leaves` field.
struct RegionState {
    /// Feature-major quantized bins (`bins[f * n + i]`).
    bins: Vec<u32>,
    /// Per-object weight (the covered regime is all-`1.0`).
    weight: Vec<f64>,
    /// Region depth (`MaxLeaves = max_depth + 1`).
    max_depth: usize,
    min_data_in_leaf: usize,
    /// The covered-loss der kind, driving the host der1 re-derivation.
    der_kernel: DerBinaryKernel,
}

/// The per-fit device Exact-leaf state (Phase 12 Plan 05, GPUT-19). `Some` iff the fit
/// committed to a SymmetricTree oblivious grow with `exact_leaf` set for a covered
/// quantile-family loss. [`GpuTrainSession::grow_one`] then REPLACES the resident grow's
/// Newton/`calc_average` leaf values with the device weighted-quantile order statistic
/// ([`device_exact_leaf_delta`]) per leaf, from the host residuals `target - approx`.
///
/// The tree STRUCTURE is grown by the resident RMSE-residual-der path (`der_kernel =
/// RmseGradient`, the MVP structural der); only the leaf VALUES become the Exact order
/// statistic. Upstream quantile-der split parity + the full-tree Kaggle oracle are the
/// Plan-09 sign-off; here the leaf-VALUE numerics are locked ≤1e-4 by the
/// `kernels::exact_quantile` self-oracle (D-09).
struct ExactLeafState {
    /// The quantile level α (loss param for [`Loss::Quantile`], else config default 0.5).
    alpha: f64,
    /// The quantile deadzone δ (loss param for [`Loss::Quantile`], else config default 1e-6).
    delta: f64,
    /// MAPE selects the `weightsWithTargets[i] = weight_i/max(1,|target_i|)` divisor (A4).
    mape: bool,
    /// Host copy of the per-object weight (the covered regime is all-`1.0`); folded into the
    /// per-leaf weighted quantile (× the MAPE divisor when `mape`).
    weight: Vec<f64>,
}

/// The per-fit device PAIRWISE state (Phase 13 Plan 01, GPUT-11). `Some` iff the fit committed to
/// a covered pairwise-scoring device path (a `*Pairwise` loss whose leaf VALUES solve the
/// `(leaf_count-1)×(leaf_count-1)` SPD pairwise system, `is_pairwise_scoring`). The device pairwise
/// histogram (reusing the Phase-7.4 4-channel `pairwise_hist` kernels) + the per-leaf
/// `MakePairwiseDerivatives`/`MakePointwiseDerivatives` matrix assembly
/// ([`crate::gpu_runtime::launch_pairwise_assemble_system_into`]) run device-side; the batched
/// Cholesky SOLVE lands in Plan 02 (GPUT-21). The per-tree pair/group descriptor seam (the
/// `Runtime::grow_tree_on_device` seam carries only `approx`/`target` today) is ALSO Plan 02's
/// wiring, so a covered pairwise fit currently declines to the byte-unchanged CPU grower (D-04
/// no-regression) rather than fabricating a pointwise grow on a pairwise fit. This state carries
/// the coverage DECISION + the regularization priors the Plan-02 grow consumes.
#[allow(dead_code)]
struct PairwiseState {
    /// The L2 diagonal reg (`l2_leaf_reg` / `L2Reg`); the per-leaf system diagonal prior.
    l2_diag_reg: f64,
    /// The pairwise bucket-weight prior reg (`bayesian_matrix_reg` / `PairwiseNonDiagReg`,
    /// default [`PAIRWISE_BUCKET_WEIGHT_PRIOR_REG_DEFAULT`]).
    pairwise_bucket_weight_prior_reg: f64,
}

/// Phase 13 Plan 01 (GPUT-11): the pairwise coverage gate, mirroring the `map_leaf_method` /
/// `map_bootstrap_kernel` `Option`-returning family-gated template (Pattern A). Returns
/// `Some(PairwiseState)` iff the fit is a covered pairwise-scoring config — a `*Pairwise` loss
/// (`is_pairwise_scoring`), depth ≥ 1, Plain boosting, single fold, `SymmetricTree`, a Phase-7.4
/// pairwise-fill-supported `n_bins` (`1 << bits`, bits in {5,6,7} == {32,64,128}), and every OTHER
/// family flag still the covered default (D-10-01 all-or-nothing PER family: no
/// bootstrap/MVS/exact-leaf/CTR/leaf-cap). Returns `None` for any uncovered config → the
/// byte-unchanged CPU grower (never a fabricated device result). `l2_diag_reg` is the caller's raw
/// L2 leaf reg feeding the per-leaf system diagonal.
fn map_pairwise_coverage(
    loss: &Loss,
    config: &DeviceTrainConfig,
    depth: usize,
    boosting_type_is_plain: bool,
    fold_count: usize,
    n_bins: usize,
    l2_diag_reg: f64,
) -> Option<PairwiseState> {
    if !cb_compute::is_pairwise_scoring(loss) {
        return None;
    }
    if depth == 0 || !boosting_type_is_plain || fold_count != 1 {
        return None;
    }
    if config.grow_policy != DeviceGrowPolicy::SymmetricTree {
        return None;
    }
    // The device pairwise histogram fill is the Phase-7.4 one-byte non-binary family (bits in
    // {5,6,7}); keep this set in sync with `launch_pairwise_split_score_into`'s `n_bins` match.
    if !matches!(n_bins, 32 | 64 | 128) {
        return None;
    }
    // All-or-nothing per family: no other non-default family flag may be set.
    let family_default = config.bootstrap_type == DeviceBootstrapType::No
        && config.mvs_lambda.is_none()
        && !config.exact_leaf
        && config.ctr.is_none()
        && config.max_leaves.is_none();
    if !family_default {
        return None;
    }
    Some(PairwiseState {
        l2_diag_reg,
        pairwise_bucket_weight_prior_reg: PAIRWISE_BUCKET_WEIGHT_PRIOR_REG_DEFAULT,
    })
}

/// Whether `loss` is a DETERMINISTIC query/listwise ranking loss with a device der driver landed in
/// Phase 13 Plan 04 (GPUT-22): QueryRMSE / QuerySoftMax. These funnel through the shared Plan-03
/// query-grouping infra + the [`crate::gpu_runtime::ranking`] der driver, self-oracled against
/// `cb_compute::calc_ders_for_queries` at ε=1e-4. (QueryCrossEntropy is INDEPENDENTLY deferred —
/// Open Q3 — and has no `Loss` variant yet, so it is not reachable here.)
fn is_deterministic_ranking_loss(loss: &Loss) -> bool {
    matches!(loss, Loss::QueryRmse | Loss::QuerySoftMax { .. })
}

/// Whether `loss` is a STOCHASTIC query/listwise ranking loss with a device der driver landed in
/// Phase 13 Plan 05 (GPUT-22, D-08): YetiRank (pointwise leaf) / YetiRankPairwise (the PFound-F
/// pairwise-leaf arm). Both ride the SAME in-query bootstrap-sampled competitor der
/// ([`crate::gpu_runtime::ranking::yetirank_ders_host`] / `pfound_f_ders_host`), self-oracled against
/// the FROZEN pinned-seed `yetirank_sample_pairs` + `calc_ders_for_queries` reference at ε=1e-4.
fn is_stochastic_ranking_loss(loss: &Loss) -> bool {
    matches!(loss, Loss::YetiRank { .. } | Loss::YetiRankPairwise { .. })
}

/// Whether `loss` is ANY device-covered query/listwise ranking loss (deterministic OR stochastic) —
/// the gate that routes the fit through [`map_ranking_coverage`] (Phase 13 Plans 04 + 05, GPUT-22).
fn is_ranking_loss(loss: &Loss) -> bool {
    is_deterministic_ranking_loss(loss) || is_stochastic_ranking_loss(loss)
}

/// The per-fit device RANKING state (Phase 13 Plan 04, GPUT-22). `Some` iff the fit committed to a
/// covered deterministic query objective — the [`crate::gpu_runtime::ranking::RankingObjective`] the
/// device der driver computes over the Plan-03 grouping infra. Like [`PairwiseState`] this carries
/// the coverage DECISION: the der driver + self-oracle land THIS plan; the per-tree query-descriptor
/// grow seam (the `Runtime::grow_tree_on_device` seam carries only `approx`/`target` today) is a
/// forward dependency, so a covered ranking fit currently declines to the byte-unchanged CPU grower
/// (D-04 no-regression) rather than fabricating a pointwise grow on a ranking fit.
#[allow(dead_code)]
struct RankingState {
    /// The covered query objective the device der driver computes (QueryRMSE / QuerySoftMax). The
    /// independently-deferred QueryCrossEntropy arm (Open Q3) is never stored here — it maps to
    /// `Ok(None)` without disabling the covered arms.
    objective: crate::gpu_runtime::ranking::RankingObjective,
}

/// Phase 13 Plan 04 (GPUT-22): the ranking coverage gate, mirroring the `map_pairwise_coverage` /
/// `map_leaf_method` `Option`-returning family-gated template (Pattern A). Returns
/// `Some(RankingState)` iff the fit is a COVERED deterministic query config — a QueryRMSE /
/// QuerySoftMax loss ([`is_deterministic_ranking_loss`]), depth ≥ 1, Plain boosting, single fold,
/// SymmetricTree, and every OTHER family flag still the covered default (D-10-01 all-or-nothing PER
/// family: no bootstrap / MVS / exact-leaf / CTR / leaf-cap). The QueryCrossEntropy arm is
/// INDEPENDENTLY gated off ([`crate::gpu_runtime::ranking::ranking_objective_covered`] returns
/// `false`), so this returns `None` for it without disabling QueryRMSE / QuerySoftMax. Returns
/// `None` for any uncovered config → the byte-unchanged CPU grower (never a fabricated device
/// result).
fn map_ranking_coverage(
    loss: &Loss,
    config: &DeviceTrainConfig,
    depth: usize,
    boosting_type_is_plain: bool,
    fold_count: usize,
) -> Option<RankingState> {
    let objective = match *loss {
        Loss::QueryRmse => crate::gpu_runtime::ranking::RankingObjective::QueryRmse,
        Loss::QuerySoftMax { lambda, beta } => {
            crate::gpu_runtime::ranking::RankingObjective::QuerySoftMax { beta, lambda }
        }
        // Phase 13 Plan 05 (D-08): the stochastic pair. YetiRank → pointwise-leaf arm;
        // YetiRankPairwise → the PFound-F pairwise-leaf arm. Both share the sampled-pair device der.
        Loss::YetiRank { permutations, decay } => {
            crate::gpu_runtime::ranking::RankingObjective::YetiRank { permutations, decay }
        }
        Loss::YetiRankPairwise { permutations, decay } => {
            crate::gpu_runtime::ranking::RankingObjective::PFoundF { permutations, decay }
        }
        // Not a ranking loss with a landed device der driver.
        _ => return None,
    };
    // Independent per-objective gate: QueryCrossEntropy (Open Q3) is deferred even though its
    // bounded shift search is landed — decline it here without disabling QueryRMSE / QuerySoftMax.
    if !crate::gpu_runtime::ranking::ranking_objective_covered(objective) {
        return None;
    }
    if depth == 0 || !boosting_type_is_plain || fold_count != 1 {
        return None;
    }
    if config.grow_policy != DeviceGrowPolicy::SymmetricTree {
        return None;
    }
    // All-or-nothing per family: no other non-default family flag may be set.
    let family_default = config.bootstrap_type == DeviceBootstrapType::No
        && config.mvs_lambda.is_none()
        && !config.exact_leaf
        && config.ctr.is_none()
        && config.max_leaves.is_none();
    if !family_default {
        return None;
    }
    Some(RankingState { objective })
}

/// The per-fit device MULTI-OUTPUT state (Phase 13 Plan 07, GPUT-12). `Some` iff the fit committed
/// to a covered multi-output loss — the [`crate::gpu_runtime::multiclass::MulticlassObjective`] the
/// device block driver solves (COUPLED softmax vs DIAGONAL separable, RESEARCH Pitfall 3). Like
/// [`PairwiseState`] / [`RankingState`] this carries the coverage DECISION: the block-leaf driver +
/// self-oracle land THIS plan (`grow_multiclass_block` over the Plan-06 K-dim Newton block solve),
/// but the per-tree SHARED multi-dim grow seam (the `Runtime::grow_tree_on_device` seam carries only
/// scalar `approx`/`target` today, not the `K`-dimensional approx / block leaf) is a forward
/// dependency, so a covered multi-output fit currently declines to the byte-unchanged CPU grower
/// (D-04 no-regression) rather than fabricating a SCALAR pointwise grow (`approx_dim == 1`) on a
/// multi-output fit — the pairwise / ranking gate precedent.
#[allow(dead_code)]
struct MulticlassState {
    /// The covered multi-output objective the block driver solves (coupled softmax / diagonal
    /// separable). MultiQuantile (a different exact-quantile leaf estimator, Plan 09) is NOT stored
    /// here — it maps to `Ok(None)` without disabling the covered arms.
    objective: crate::gpu_runtime::multiclass::MulticlassObjective,
}

/// Phase 13 Plan 07 (GPUT-12): the multi-output coverage gate, mirroring the `map_pairwise_coverage`
/// / `map_ranking_coverage` `Option`-returning family-gated template (Pattern A). Returns
/// `Some(MulticlassState)` iff the fit is a COVERED multi-output config — a MultiClass /
/// MultiClassOneVsAll / MultiLogloss / MultiCrossEntropy / RMSEWithUncertainty loss
/// ([`crate::gpu_runtime::multiclass::map_multiclass_objective`]), depth ≥ 1, Plain boosting, single
/// fold, SymmetricTree, and every OTHER family flag still the covered default (D-10-01 all-or-nothing
/// PER family: no bootstrap / MVS / exact-leaf / CTR / leaf-cap). MultiQuantile declines (its exact
/// leaf is Plan 09). Returns `None` for any uncovered config → the byte-unchanged CPU grower.
fn map_multiclass_coverage(
    loss: &Loss,
    config: &DeviceTrainConfig,
    depth: usize,
    boosting_type_is_plain: bool,
    fold_count: usize,
) -> Option<MulticlassState> {
    let objective = crate::gpu_runtime::multiclass::map_multiclass_objective(loss)?;
    if depth == 0 || !boosting_type_is_plain || fold_count != 1 {
        return None;
    }
    if config.grow_policy != DeviceGrowPolicy::SymmetricTree {
        return None;
    }
    // All-or-nothing per family: no other non-default family flag may be set.
    let family_default = config.bootstrap_type == DeviceBootstrapType::No
        && config.mvs_lambda.is_none()
        && !config.exact_leaf
        && config.ctr.is_none()
        && config.max_leaves.is_none();
    if !family_default {
        return None;
    }
    Some(MulticlassState { objective })
}

/// The per-fit device ORDERED-boosting state (Phase 13 Plan 08, GPUT-13). `Some` iff the fit
/// committed to a covered ordered-boosting config — the per-permutation historical approx trajectory
/// the device driver ([`crate::gpu_runtime::ordered`]) keeps RESIDENT across iterations, reproducing
/// the frozen CPU `ordered_approx_delta_simple` body/tail approximant at ε=1e-4 (body rows keep delta
/// 0). Like [`RankingState`] / [`MulticlassState`] this carries the coverage DECISION: the ordered
/// trajectory driver + self-oracle land THIS plan; the per-tree ordered permutation-descriptor grow
/// seam (the `Runtime::grow_tree_on_device` seam carries only scalar `approx`/`target` today, not the
/// learn-order permutation + body/tail boundary an ordered grow needs) is a forward dependency, so a
/// covered ordered fit currently declines to the byte-unchanged CPU grower (D-04 no-regression)
/// rather than fabricating a Plain (leakage-prone) pointwise grow on an ordered fit.
#[allow(dead_code)]
struct OrderedState {
    /// The covered ordered der/leaf method — the simple Gradient/RMSE approximant
    /// (`ordered_approx_delta_simple`, `gradient_leaf_delta` = `calc_average`). Newton / other
    /// approximants are NOT stored here — they map to `Ok(None)` without disabling the covered arm.
    der_kernel: DerBinaryKernel,
}

/// Phase 13 Plan 08 (GPUT-13): the ordered-boosting coverage gate, mirroring the
/// `map_ranking_coverage` / `map_multiclass_coverage` `Option`-returning family-gated template
/// (Pattern A). Returns `Some(OrderedState)` iff the fit is a COVERED ordered config — a covered
/// simple-approximant loss ([`map_der_kernel`]: RMSE / Logloss / CrossEntropy der), depth ≥ 1, single
/// fold, SymmetricTree, and every OTHER family flag still the covered default (D-10-01 all-or-nothing
/// PER family: no bootstrap / MVS / exact-leaf / CTR / leaf-cap). Ordered boosting itself is signalled
/// by the caller's `!boosting_type_is_plain` at the call site (there is no `permutation_count` device
/// knob — the frozen fixture pins it, Open Q2 single learning-fold). Returns `None` for any uncovered
/// ordered config → the byte-unchanged CPU grower (never a fabricated device result).
fn map_ordered_coverage(
    loss: &Loss,
    config: &DeviceTrainConfig,
    depth: usize,
    fold_count: usize,
) -> Option<OrderedState> {
    let der_kernel = map_der_kernel(loss)?;
    if depth == 0 || fold_count != 1 {
        return None;
    }
    if config.grow_policy != DeviceGrowPolicy::SymmetricTree {
        return None;
    }
    // All-or-nothing per family: no other non-default family flag may be set.
    let family_default = config.bootstrap_type == DeviceBootstrapType::No
        && config.mvs_lambda.is_none()
        && !config.exact_leaf
        && config.ctr.is_none()
        && config.max_leaves.is_none();
    if !family_default {
        return None;
    }
    Some(OrderedState { der_kernel })
}

/// The per-fit device LANGEVIN / SGLB state (Phase 13 Plan 09, GPUT-20). `Some` iff the fit
/// committed to a covered Langevin config — the noise the device driver ([`crate::kernels::langevin`])
/// adds to the RESIDENT reduced derivatives per tree (`coefficient · std_normal(seed_i)`), reproducing
/// the frozen pinned-seed CPU `coefficient · std_normal` sequence at ε=1e-4. Like [`RankingState`] /
/// [`OrderedState`] this carries the coverage DECISION: the AddLangevinNoise kernel + self-oracle land
/// THIS plan; there is no device Langevin CONFIG knob yet (the noise coefficient rides a forward seam
/// alongside the per-tree grow descriptor), so a covered Langevin fit currently declines to the
/// byte-unchanged CPU grower (D-04 no-regression) rather than fabricating an un-noised device grow.
/// A `*Pairwise` + Langevin config is NOT covered ([`crate::kernels::langevin::langevin_covered_loss`]
/// is false for `is_pairwise_scoring` — upstream `pairwise_oracle.h` `CB_ENSURE`, A4), so it falls
/// back to CPU (`Ok(None)`), reinforcing the pairwise arm's own decline.
#[allow(dead_code)]
struct LangevinState {
    /// The covered pointwise der kind, driving the host der re-derivation the Langevin noise layers
    /// onto (RMSE / Logloss / CrossEntropy — the same covered family as the sampling families).
    der_kernel: DerBinaryKernel,
}

/// Phase 13 Plan 09 (GPUT-20): the Langevin/SGLB coverage gate, mirroring the `map_ordered_coverage`
/// / `map_ranking_coverage` `Option`-returning family-gated template (Pattern A). Returns
/// `Some(LangevinState)` iff the fit is a COVERED Langevin config — a covered pointwise der loss
/// ([`crate::kernels::langevin::langevin_covered_loss`]: RMSE / Logloss / CrossEntropy, and EXPLICITLY
/// NOT a pairwise-scoring loss, A4), depth ≥ 1, Plain boosting, single fold, SymmetricTree, and every
/// OTHER family flag still the covered default (D-10-01 all-or-nothing PER family: no bootstrap / MVS /
/// exact-leaf / CTR / leaf-cap). A `*Pairwise` + Langevin config → `None` (Langevin not supported on
/// the pairwise oracle) → the byte-unchanged CPU grower. Returns `None` for any uncovered config.
fn map_langevin_coverage(
    loss: &Loss,
    config: &DeviceTrainConfig,
    depth: usize,
    boosting_type_is_plain: bool,
    fold_count: usize,
) -> Option<LangevinState> {
    if !crate::kernels::langevin::langevin_covered_loss(loss) {
        return None;
    }
    let der_kernel = map_der_kernel(loss)?;
    if depth == 0 || !boosting_type_is_plain || fold_count != 1 {
        return None;
    }
    if config.grow_policy != DeviceGrowPolicy::SymmetricTree {
        return None;
    }
    // All-or-nothing per family: no other non-default family flag may be set.
    let family_default = config.bootstrap_type == DeviceBootstrapType::No
        && config.mvs_lambda.is_none()
        && !config.exact_leaf
        && config.ctr.is_none()
        && config.max_leaves.is_none();
    if !family_default {
        return None;
    }
    Some(LangevinState { der_kernel })
}

/// Re-derive the UN-weighted per-object der1 on the host from the caller's `approx` + `target`
/// for the covered losses, transcribing `cb_compute`'s documented formulas (RMSE:
/// `der1 = target - approx`; Logloss / CrossEntropy: `der1 = target - sigmoid(approx)`). The
/// covered regime is unit weights, so this UN-weighted der1 feeds the histogram numerator and
/// `calc_average` denominator exactly as the CPU reference does.
fn host_der1(der_kernel: DerBinaryKernel, approx: &[f64], target: &[f64]) -> Vec<f64> {
    approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| match der_kernel {
            DerBinaryKernel::RmseGradient => t - a,
            DerBinaryKernel::LoglossGradient => {
                let p = 1.0_f64 / (1.0_f64 + (-a).exp());
                t - p
            }
        })
        .collect()
}

/// The per-fit device-resident training session (GPUT-02): one client + the persistent
/// device handles + the frozen per-fit configuration. Constructed by [`Self::begin`] only
/// when the coverage gate passes; dropped (client + handles freed) at end of fit.
pub struct GpuTrainSession {
    /// The ONE client that allocated every handle below — all read-backs go through it.
    client: cubecl::client::ComputeClient<SelectedRuntime>,

    // --- Persistent device handles, uploaded ONCE at `begin` (cloned per tree) ---
    /// The plain feature-major quantized bins (`cindex[feature * n + obj]`), for the
    /// partition split's degenerate-TCFeature read.
    plain_cindex_h: Handle,
    /// The bit-packed grouped cindex `words`, for the histogram's `read_bin` accessor.
    cindex_words_h: Handle,
    /// Per-feature TCFeature `Offset` array (packed-layout addressing).
    offsets_h: Handle,
    /// Per-feature TCFeature `Shift` array.
    shifts_h: Handle,
    /// Per-feature TCFeature `Mask` array.
    masks_h: Handle,
    /// The object-visiting order (identity `0..n` for the whole-dataset root).
    indices_h: Handle,
    /// The per-object weight (channel float type), folded downstream by the histogram.
    weight_h: Handle,
    /// The regression/classification target (channel float type), for the der recompute.
    /// Set on the FIRST `grow_one` (the seam supplies `target` per tree, not at `begin`).
    target_h: Option<Handle>,
    /// The running per-object approx (channel float type), resident + updated ON DEVICE.
    approx_h: Handle,
    /// The running residual gradient feeding the next tree (resident). Set on first grow.
    der1_h: Option<Handle>,

    // --- Frozen per-fit configuration ---
    num_words: usize,
    n: usize,
    n_bins: usize,
    /// The (possibly PADDED) partition-histogram line width the resident fill dispatches
    /// ({32,64,128,256}; == `n_bins` when it is already a dispatched width). Only the
    /// oblivious resident grow reads it; the nonsym/Region/CTR paths keep the ACTUAL
    /// `n_bins`.
    n_bins_line: usize,
    n_features: usize,
    cindex_stride: usize,
    depth: usize,
    scaled_l2: f64,
    score_fn: u32,
    learning_rate: f64,
    der_kernel: DerBinaryKernel,
    /// The frozen per-fit device config (Phase 12 Plan 01, Open Q2). Stored so the later
    /// Phase-12 waves (grow policy, sampling, exact leaf, CTR) read their knobs from ONE
    /// place without re-threading the `begin` argument list. In Plan 01 only the covered
    /// (default) regime reaches here, so no field is consumed yet.
    #[allow(dead_code)]
    config: DeviceTrainConfig,
    /// The per-fit non-symmetric grow state (Phase 12 Plan 03). `Some` iff the fit committed
    /// to a Depthwise / Lossguide device grow — [`Self::grow_one`] then dispatches to the
    /// host-driven [`grow_nonsym_tree`] instead of the resident oblivious loop. `None` for the
    /// oblivious / symmetric path (byte-unchanged, D-04).
    nonsym: Option<NonsymState>,
    /// The per-fit Exact-leaf state (Phase 12 Plan 05, GPUT-19). `Some` iff the fit committed
    /// to a SymmetricTree oblivious grow with device Exact leaves (covered quantile-family
    /// loss + `exact_leaf`). `None` for the Newton/default path (byte-unchanged, D-04).
    exact_leaf: Option<ExactLeafState>,
    /// The per-fit bootstrap state (Phase 12 Plan 06, GPUT-09). `Some` iff the fit committed to a
    /// covered oblivious grow with a non-`No` `bootstrap_type` — `grow_one` then draws the
    /// device-resident sample per tree and folds it into the resident weight. `None` for the
    /// no-subsampling default (byte-unchanged, D-04).
    bootstrap: Option<BootstrapState>,
    /// The per-fit MVS state (Phase 12 Plan 07, GPUT-17). `Some` iff the fit committed to a covered
    /// oblivious grow with `bootstrap_type == Mvs` — `grow_one` then draws the device-resident MVS
    /// sample (per-block threshold + reweight over the resident derivatives) and folds it into the
    /// resident weight. Mutually exclusive with [`Self::bootstrap`] (distinct `bootstrap_type`).
    mvs: Option<MvsState>,
    /// The per-fit Region-grow state (Phase 12 Plan 04, GPUT-18). `Some` iff the fit committed to
    /// a `grow_policy=Region` device grow — [`Self::grow_one`] then dispatches to the host-driven
    /// [`grow_region_tree`] (a walk-until-diverge PATH, `MaxLeaves = max_depth + 1`) instead of
    /// the resident oblivious loop / non-sym node graph. `None` for every other grow policy.
    region: Option<RegionState>,
    /// The per-fit pairwise state (Phase 13 Plan 01, GPUT-11). `Some` iff the fit committed to a
    /// covered pairwise-scoring device path; `None` for every pointwise path (byte-unchanged,
    /// D-04). Carries the coverage DECISION + regularization priors the Plan-02 grow wiring
    /// consumes (the per-tree pair/group descriptor seam lands in Plan 02, so no covered-pairwise
    /// session is constructed yet — the field is the landed structural seam, like `config`).
    #[allow(dead_code)]
    pairwise: Option<PairwiseState>,
    /// The per-fit ranking state (Phase 13 Plan 04, GPUT-22). `Some` iff the fit committed to a
    /// covered deterministic query objective (QueryRMSE / QuerySoftMax); `None` for every pointwise
    /// / pairwise path (byte-unchanged, D-04). Carries the coverage DECISION the ranking der driver
    /// consumes (the per-tree query-descriptor grow seam is a forward dependency, so no covered-
    /// ranking session is constructed yet — the field is the landed structural seam, like `pairwise`
    /// / `config`). QueryCrossEntropy is INDEPENDENTLY deferred (Open Q3) and never stored here.
    #[allow(dead_code)]
    ranking: Option<RankingState>,
    /// The per-fit multi-output state (Phase 13 Plan 07, GPUT-12). `Some` iff the fit committed to a
    /// covered multi-output loss (MultiClass / OneVsAll / MultiLogloss / MultiCrossEntropy /
    /// RMSEWithUncertainty); `None` for every scalar / pairwise / ranking path (byte-unchanged,
    /// D-04). Carries the coverage DECISION the block-leaf driver consumes (the per-tree shared
    /// multi-dim grow seam is a forward dependency, so no covered-multi-output session is constructed
    /// yet — the field is the landed structural seam, like `pairwise` / `ranking` / `config`).
    #[allow(dead_code)]
    multiclass: Option<MulticlassState>,
    /// The per-fit ordered-boosting state (Phase 13 Plan 08, GPUT-13). `Some` iff the fit committed
    /// to a covered ordered-boosting config (a covered simple-approximant loss over the historical
    /// per-permutation trajectory); `None` for every Plain / pairwise / ranking / multi-output path
    /// (byte-unchanged, D-04). Carries the coverage DECISION the ordered trajectory driver consumes
    /// (the per-tree ordered permutation-descriptor grow seam is a forward dependency, so no
    /// covered-ordered session is constructed yet — the field is the landed structural seam, like
    /// `pairwise` / `ranking` / `multiclass` / `config`).
    #[allow(dead_code)]
    ordered: Option<OrderedState>,
    /// The per-fit Langevin/SGLB state (Phase 13 Plan 09, GPUT-20). `Some` iff the fit committed to a
    /// covered Langevin config (a covered pointwise der loss with the seeded-Gaussian noise layered
    /// on the resident der); `None` for every non-Langevin path AND for every `*Pairwise` + Langevin
    /// config (Langevin unsupported on the pairwise oracle, A4 → CPU fallback). Carries the coverage
    /// DECISION the AddLangevinNoise driver consumes (the noise coefficient / per-tree grow seam is a
    /// forward dependency, so no covered-Langevin session is constructed yet — the field is the landed
    /// structural seam, like `pairwise` / `ranking` / `multiclass` / `ordered` / `config`).
    #[allow(dead_code)]
    langevin: Option<LangevinState>,
}

/// Map a host [`EScoreFunction`] to the device score-calcer selector, or `None` if the score
/// function has no device arm (→ the session declines, CPU fallback). Cosine is the catboost
/// CPU default and the depth-1 device default (GPUT-08).
fn map_score_fn(score_function: EScoreFunction) -> Option<u32> {
    match score_function {
        EScoreFunction::Cosine => Some(SCORE_FN_COSINE),
        EScoreFunction::L2 => Some(SCORE_FN_L2),
        EScoreFunction::SolarL2 => Some(SCORE_FN_SOLAR_L2),
        EScoreFunction::LOOL2 => Some(SCORE_FN_LOO_L2),
        EScoreFunction::SatL2 => Some(SCORE_FN_SAT_L2),
        // The Newton (der2-weighted) score functions are GPU-only upstream and have no
        // depth-1 MVP device arm here — decline to the CPU path (D-04), never a wrong score.
        EScoreFunction::NewtonL2 | EScoreFunction::NewtonCosine => None,
    }
}

/// Round a quantized bin count UP to the next partition-histogram line width the resident
/// fill dispatches (`1 << bits`, bits 5..=8 → {32,64,128,256}), or `None` for a >8-bit
/// quantization (n_bins > 256 → CPU fallback, D-04). The padding cells (`n_bins..width`)
/// stay zero in every histogram and their phantom borders are excluded from the split
/// argmin, so padding is score-invariant (see `GpuTrainSession::begin`).
fn pad_hist_line_bins(n_bins: usize) -> Option<usize> {
    match n_bins {
        1..=32 => Some(32),
        33..=64 => Some(64),
        65..=128 => Some(128),
        129..=256 => Some(256),
        _ => None,
    }
}

/// Map a host [`Loss`] to the resident der-seam kernel, or `None` if the loss has no device
/// der arm in the covered set (RMSE / Logloss / CrossEntropy). CrossEntropy shares Logloss's
/// sigmoid-gradient kernel exactly (D-09).
fn map_der_kernel(loss: &Loss) -> Option<DerBinaryKernel> {
    match *loss {
        Loss::Rmse => Some(DerBinaryKernel::RmseGradient),
        Loss::Logloss | Loss::CrossEntropy => Some(DerBinaryKernel::LoglossGradient),
        _ => None,
    }
}

/// The device leaf-estimation method (Phase 12 Plan 05, GPUT-19, D-09). `Newton` is the
/// default closed-form (`calc_average` / der2) path, byte-unchanged. `Exact` is the device
/// weighted-quantile ORDER STATISTIC for the Quantile/MAE/MAPE family — DISTINCT from Newton
/// (`g/(h+ε)`), reproducing `cb-compute/src/leaf.rs::exact_leaf_delta` on device (Pitfall 6).
/// `mape` selects the `weightsWithTargets[i] = weight_i/max(1,|target_i|)` divisor (A4).
#[derive(Debug, Clone, Copy, PartialEq)]
enum DeviceLeafMethod {
    Newton,
    Exact { alpha: f64, delta: f64, mape: bool },
}

/// The exact-leaf gate arm (`map_leaf_method`) — mirrors the `map_score_fn` / `map_der_kernel`
/// / `map_grow_policy` `Option`-returning family-gated template (Pattern A). Returns:
/// - `Some(Newton)` when exact-leaf is NOT requested → the default path, unchanged (D-04).
/// - `Some(Exact{..})` when `config.exact_leaf` is set AND the loss is in the covered
///   quantile family (MAE / Quantile / MAPE, A4) → route the leaf VALUES to the device Exact
///   order statistic.
/// - `None` when exact-leaf is requested for a loss OUTSIDE the quantile family → decline to
///   the CPU path (the prohibition: uncovered stays `Ok(None)`, never a wrong device result).
///
/// α/δ come from the loss's own params for [`Loss::Quantile`], else the config's
/// `quantile_alpha`/`quantile_delta` (whose defaults `0.5`/`1e-6` are the MAE / MAPE median).
fn map_leaf_method(config: &DeviceTrainConfig, loss: &Loss) -> Option<DeviceLeafMethod> {
    if !config.exact_leaf {
        return Some(DeviceLeafMethod::Newton);
    }
    match *loss {
        Loss::Mae => Some(DeviceLeafMethod::Exact {
            alpha: config.quantile_alpha,
            delta: config.quantile_delta,
            mape: false,
        }),
        Loss::Quantile { alpha, delta } => {
            Some(DeviceLeafMethod::Exact { alpha, delta, mape: false })
        }
        Loss::Mape => Some(DeviceLeafMethod::Exact {
            alpha: config.quantile_alpha,
            delta: config.quantile_delta,
            mape: true,
        }),
        // Exact leaf requested for a non-quantile loss → decline (never a wrong device leaf).
        _ => None,
    }
}

impl GpuTrainSession {
    /// Open a device-resident training session, running the coverage gate (D-10-02). Returns
    /// `Ok(None)` when the config is NOT covered (depth==0, non-RMSE/Logloss, non-Plain,
    /// fold_count>1, or an unsupported score function) so the caller falls back to the CPU
    /// path (D-04). When covered, it validates the quantized bins host-side, packs the
    /// cindex (10-06), and uploads ALL resident handles ONCE onto one client, initialising
    /// the running approx to all-zero (the RMSE-from-zero MVP; the target/der1 are set on the
    /// first `grow_one`, since the seam supplies `target` per tree).
    ///
    /// `bins_feature_major` is the quantized bin matrix (`cindex[feature * n + obj]`, length
    /// `n_features * n`); `weight` the per-object weight (length `n`); `scaled_l2` the
    /// per-tree `scale_l2_reg` output. Object indices default to the identity `0..n` (the
    /// whole-dataset root order). No `unwrap`/`expect`/`panic`/indexing (workspace lints +
    /// D-13); never reads a 0-len handle.
    #[allow(clippy::too_many_arguments)]
    pub fn begin(
        loss: &Loss,
        depth: usize,
        boosting_type_is_plain: bool,
        fold_count: usize,
        score_function: EScoreFunction,
        bins_feature_major: &[u32],
        weight: &[f64],
        n: usize,
        n_features: usize,
        n_bins: usize,
        learning_rate: f64,
        scaled_l2: f64,
        config: &DeviceTrainConfig,
    ) -> CbResult<Option<Self>> {
        // --- Coverage gate (D-10-02): classification lives where the session is created. ---
        // The device path grows a depth>=1 Plain oblivious tree over a single fold with an
        // RMSE/Logloss loss and a supported score function; ANYTHING else declines to CPU.
        //
        // Phase 12 Plan 01 (GPUT-18, A3 gap): the former `depth != 1` force-decline is GONE —
        // depth>1 is DEVICE-COVERED via the already-shipped Phase-11 partition-aware substrate
        // (`grow_oblivious_tree_resident` loops `0..depth`, keying each level's fill on the
        // resident `leaf_of` + the per-active-leaf score/argmin). Only the
        // Plain/fold_count==1/covered-loss/score/n_bins guards remain; every still-uncovered
        // config returns `Ok(None)` → the byte-unchanged CPU grower (D-10-01 all-or-nothing).
        // Phase 13 Plan 08 (GPUT-13): the ordered-boosting coverage gate. Ordered boosting
        // (`EBoostingType::Ordered`, `!boosting_type_is_plain`) keeps a per-permutation HISTORICAL
        // approx trajectory (the anti-leakage body/tail approximant — a TAIL row is estimated from
        // the BODY prefix plus only the tail rows that precede it in the learn permutation; body rows
        // keep delta 0) that the device `crate::gpu_runtime::ordered` driver reproduces
        // device-resident across iterations, self-oracled against the frozen CPU
        // `ordered_approx_delta_simple` at ε=1e-4. `map_ordered_coverage` returns `Some(OrderedState)`
        // for a covered ordered config, `None` otherwise. The ordered trajectory driver + self-oracle
        // land THIS plan; the per-tree ordered permutation-descriptor grow seam (the
        // `Runtime::grow_tree_on_device` seam carries only scalar `approx`/`target` today, not the
        // learn-order permutation + body/tail boundary an ordered grow needs) is a forward
        // dependency, so BOTH the covered and uncovered ordered branches decline to the
        // byte-unchanged CPU grower (D-04 no-regression) — NEVER a fabricated Plain (leakage-prone)
        // pointwise grow on an ordered fit (the pairwise / ranking / multiclass gate precedent).
        if !boosting_type_is_plain {
            let _ordered = map_ordered_coverage(loss, config, depth, fold_count);
            return Ok(None);
        }
        if depth == 0 || fold_count != 1 {
            return Ok(None);
        }
        // Phase 13 Plan 01 (GPUT-11): the pairwise coverage gate. A `*Pairwise` loss
        // (`is_pairwise_scoring`) solves its leaf VALUES via the SPD pairwise system, NOT the
        // pointwise Gradient/Newton estimator — a fundamentally different grow. `map_pairwise_coverage`
        // returns `Some(PairwiseState)` for a covered pairwise config (proven by the
        // `pairwise_deriv` self-oracle), `None` otherwise. The device pairwise histogram
        // (Phase-7.4 reuse) + the per-leaf `MakePairwise/PointwiseDerivatives` matrix ASSEMBLY
        // ([`crate::gpu_runtime::launch_pairwise_assemble_system_into`]) land THIS plan; the batched
        // Cholesky SOLVE + the per-tree pair/group descriptor seam (the `Runtime::grow_tree_on_device`
        // seam carries only `approx`/`target` today) land in Plan 02 (GPUT-21). Until Plan 02 wires
        // the grow, BOTH the covered and uncovered pairwise branches decline to the byte-unchanged
        // CPU grower (D-04 no-regression) — NEVER a fabricated pointwise grow on a pairwise fit.
        if cb_compute::is_pairwise_scoring(loss) {
            match map_pairwise_coverage(
                loss,
                config,
                depth,
                boosting_type_is_plain,
                fold_count,
                n_bins,
                scaled_l2,
            ) {
                // Covered pairwise config: device histogram reuse + per-leaf system assembly are
                // landed + self-oracled this plan; the per-tree grow seam is Plan 02. Decline to CPU.
                Some(_pairwise) => return Ok(None),
                // Uncovered pairwise config → CPU grower (D-10-01 all-or-nothing per family).
                None => return Ok(None),
            }
        }
        // Phase 13 Plan 04 (GPUT-22): the ranking coverage gate. A deterministic query/listwise loss
        // (QueryRMSE / QuerySoftMax) computes its der over the shared Plan-03 query-grouping infra +
        // the `crate::gpu_runtime::ranking` der driver, self-oracled against
        // `cb_compute::calc_ders_for_queries` at ε=1e-4. `map_ranking_coverage` returns
        // `Some(RankingState)` for a covered config, `None` otherwise (QueryCrossEntropy is
        // INDEPENDENTLY deferred — Open Q3 — without disabling the covered arms). The der driver +
        // self-oracle land THIS plan; the per-tree query-descriptor grow seam (the
        // `Runtime::grow_tree_on_device` seam carries only `approx`/`target` today) is a forward
        // dependency, so BOTH the covered and uncovered ranking branches decline to the
        // byte-unchanged CPU grower (D-04 no-regression) — NEVER a fabricated pointwise grow on a
        // ranking fit (the pairwise-gate precedent).
        // Phase 13 Plan 05 (GPUT-22, D-08) EXTENDS this branch to the STOCHASTIC pair (YetiRank /
        // PFound-F). NOTE: `YetiRankPairwise` (the PFound-F leaf path) is ALSO `is_pairwise_scoring`,
        // so it is intercepted by the pairwise branch ABOVE (which likewise declines to CPU, the grow
        // seam being a forward dependency) — its device der driver + self-oracle
        // (`ranking::pfound_f_ders_host`) still land this plan and are exercised directly by the
        // `ranking_stoch_test` self-oracle. YetiRank (pointwise leaf, NOT pairwise-scoring) reaches
        // here and records its ranking coverage decision via `map_ranking_coverage`.
        if is_ranking_loss(loss) {
            match map_ranking_coverage(loss, config, depth, boosting_type_is_plain, fold_count) {
                // Covered ranking config: der driver + self-oracle landed; grow seam is a forward
                // dependency. Decline to CPU (never a fabricated pointwise grow on a ranking fit).
                Some(_ranking) => return Ok(None),
                // Uncovered ranking config → CPU grower (D-10-01 all-or-nothing per family).
                None => return Ok(None),
            }
        }
        // Phase 13 Plan 07 (GPUT-12): the multi-output coverage gate. A multi-output loss (MultiClass
        // softmax / MultiClassOneVsAll / MultiLogloss / MultiCrossEntropy / RMSEWithUncertainty)
        // grows ONE shared tree whose leaf VALUES are a `leaf_count × K` block solved by the Plan-06
        // K-dim Newton block solve (COUPLED softmax vs DIAGONAL separable) — a fundamentally
        // different leaf estimator than the scalar Gradient/Newton path. `map_multiclass_coverage`
        // returns `Some(MulticlassState)` for a covered multi-output config (proven by the
        // `multiclass` block-driver self-oracle), `None` otherwise (MultiQuantile's exact-quantile
        // leaf is Plan 09). The block-leaf driver (`grow_multiclass_block`) + self-oracle land THIS
        // plan; the per-tree SHARED multi-dim grow seam (the `Runtime::grow_tree_on_device` seam
        // carries only scalar `approx`/`target` today, not the `K`-dimensional approx / block leaf)
        // is a forward dependency, so BOTH the covered and uncovered multi-output branches decline to
        // the byte-unchanged CPU grower (D-04 no-regression) — NEVER a fabricated SCALAR pointwise
        // grow (`approx_dim == 1`) on a multi-output fit (the pairwise / ranking gate precedent).
        if crate::gpu_runtime::multiclass::map_multiclass_objective(loss).is_some() {
            match map_multiclass_coverage(loss, config, depth, boosting_type_is_plain, fold_count) {
                // Covered multi-output config: block driver + self-oracle landed; grow seam is a
                // forward dependency. Decline to CPU (never a fabricated scalar grow on a
                // multi-output fit).
                Some(_multiclass) => return Ok(None),
                // Uncovered multi-output config → CPU grower (D-10-01 all-or-nothing per family).
                None => return Ok(None),
            }
        }
        // Phase 13 Plan 09 (GPUT-20): the Langevin/SGLB gate. `map_langevin_coverage` records whether
        // this is a covered Langevin config — a covered pointwise der loss with the seeded-Gaussian
        // noise layered on the resident der (`crate::kernels::langevin::langevin_covered_loss` is false
        // for `is_pairwise_scoring`, so a `*Pairwise` + Langevin config is NOT covered — Langevin is
        // unsupported on the pairwise oracle, A4 — and that config is already declined by the pairwise
        // arm above). There is no device Langevin CONFIG knob yet (the noise coefficient + per-tree
        // grow descriptor ride a forward seam), so — like the pairwise / ranking / multiclass / ordered
        // structural seams — no covered-Langevin session is constructed here; the decision is recorded
        // and a covered pointwise fit proceeds on the byte-unchanged path (the AddLangevinNoise kernel +
        // self-oracle land THIS plan, exercised directly by the langevin self-oracle).
        let _langevin = map_langevin_coverage(loss, config, depth, boosting_type_is_plain, fold_count);
        // Phase 12 Plan 03 (GPUT-18): the grow-policy arm. `map_grow_policy` returns the device
        // non-symmetric strategy for Depthwise / Lossguide (flipped ON this plan), or `None` for
        // the oblivious / symmetric path (which is the byte-unchanged Plan-01 resident loop) and
        // for the not-yet-covered Region policy (Plan 04, declines below).
        let nonsym_policy = map_grow_policy(config.grow_policy);
        // Phase 12 Plan 04 (GPUT-18, D-03a): the Region arm. Region is neither the oblivious
        // resident path nor a non-symmetric node graph — it grows a walk-until-diverge PATH
        // (`MaxLeaves = depth + 1`) via the host-driven device scorer. It is intercepted as a
        // THIRD branch below (`map_grow_policy` returns `None` for it, so `region_active` guards
        // the `None` arm from falling through to the SymmetricTree covered-regime gate).
        let region_active = config.grow_policy == DeviceGrowPolicy::Region;

        // Phase 12 Plan 05 (GPUT-19): the exact-leaf gate arm. `map_leaf_method` returns the
        // device leaf method — `Newton` (default, unchanged) or the `Exact` weighted-quantile
        // order statistic when `exact_leaf` is set for a covered quantile-family loss (MAE /
        // Quantile / MAPE, A4). An exact-leaf request for a NON-quantile loss declines to CPU
        // (prohibition: uncovered stays `Ok(None)`, never a wrong device leaf).
        let leaf_method = match map_leaf_method(config, loss) {
            Some(m) => m,
            None => return Ok(None),
        };

        // Phase 12 Plan 06 (GPUT-09): the bootstrap arm. `map_bootstrap_kernel` returns the device
        // draw family for Bernoulli/Bayesian/Poisson (flipped ON this plan), the covered non-draw
        // default for `No`, or a decline for MVS (Plan 07). Only reachable on the oblivious path
        // (the nonsym `family_default` below still requires `bootstrap_type == No`).
        let bootstrap_arm = map_bootstrap_kernel(config.bootstrap_type);

        match nonsym_policy {
            None if region_active => {
                // Phase 12 Plan 04 (GPUT-18): the Region family gate. Region owns only its own
                // knob (`grow_policy=Region`, depth → `MaxLeaves = depth + 1`); every OTHER family
                // flag must still be the covered default (D-10-01 all-or-nothing PER family: no
                // subsampling / MVS / exact leaf / CTR / leaf cap). Otherwise decline to CPU.
                let family_default = config.bootstrap_type == DeviceBootstrapType::No
                    && config.mvs_lambda.is_none()
                    && !config.exact_leaf
                    && config.ctr.is_none()
                    && config.max_leaves.is_none();
                if !family_default {
                    return Ok(None);
                }
            }
            None => {
                // Region is intercepted above (`region_active`). SymmetricTree
                // rides the Plan-01 covered-regime gate, EXTENDED (Plan 05) to also admit the
                // exact-leaf quantile-family regime, and (Plan 06) the bootstrap regime (D-10-01
                // all-or-nothing: any OTHER non-default family flag still routes to the CPU grower).
                if config.grow_policy != DeviceGrowPolicy::SymmetricTree {
                    return Ok(None);
                }
                // MVS with NO caller-pinned λ is not covered (the iter-0 `mean(|der|)^2` auto-λ
                // needs the resident der, MVP-deferred) — decline explicitly so it falls to CPU.
                if bootstrap_arm == BootstrapArm::Mvs && config.mvs_lambda.is_none() {
                    return Ok(None);
                }
                // The exact-leaf regime is covered when ONLY `exact_leaf` (+ its α/δ) is
                // non-default (every other family flag still the covered default).
                let exact_covered = matches!(leaf_method, DeviceLeafMethod::Exact { .. })
                    && config.bootstrap_type == DeviceBootstrapType::No
                    && config.mvs_lambda.is_none()
                    && config.ctr.is_none()
                    && config.max_leaves.is_none();
                // The bootstrap regime is covered when ONLY `bootstrap_type` (+ its rate/seed) is
                // non-default (Newton leaf, symmetric, single fold, no MVS/exact/CTR/leaf-cap).
                let bootstrap_covered = matches!(bootstrap_arm, BootstrapArm::Device(_))
                    && !config.exact_leaf
                    && config.mvs_lambda.is_none()
                    && config.ctr.is_none()
                    && config.max_leaves.is_none();
                // Phase 12 Plan 07 (GPUT-17): the MVS regime is covered when ONLY `bootstrap_type ==
                // Mvs` (+ its rate/seed/λ) is non-default — the caller pins `mvs_lambda` (checked
                // above), and every OTHER family flag is the covered default (Newton leaf, symmetric,
                // single fold, no other-bootstrap/exact/CTR/leaf-cap). D-10-01 all-or-nothing PER
                // family.
                let mvs_covered = bootstrap_arm == BootstrapArm::Mvs
                    && config.mvs_lambda.is_some()
                    && !config.exact_leaf
                    && config.ctr.is_none()
                    && config.max_leaves.is_none();
                // Phase 12 Plan 08 (GPUT-10): the CTR regime is covered when ONLY the CTR config
                // (single-permutation, n_bins-binarized columns) is non-default — every other
                // family flag still the covered default (Newton leaf, symmetric, single fold, no
                // bootstrap/MVS/exact/leaf-cap). D-10-01 all-or-nothing PER family.
                let ctr_is_covered = ctr_covered(config, n, n_bins)
                    && matches!(bootstrap_arm, BootstrapArm::NoDraw)
                    && !config.exact_leaf
                    && config.mvs_lambda.is_none()
                    && config.max_leaves.is_none();
                if !config.is_covered_regime()
                    && !exact_covered
                    && !bootstrap_covered
                    && !ctr_is_covered
                    && !mvs_covered
                {
                    return Ok(None);
                }
            }
            Some(_) => {
                // Depthwise / Lossguide: every OTHER family flag must still be the default
                // covered regime (no subsampling / MVS / exact leaf / CTR). Only `grow_policy`
                // and `max_leaves` (the Lossguide cap) may be non-default — those are THIS
                // family's own knobs (D-10-01 all-or-nothing PER family).
                let family_default = config.bootstrap_type == DeviceBootstrapType::No
                    && config.mvs_lambda.is_none()
                    && !config.exact_leaf
                    && config.ctr.is_none();
                if !family_default {
                    return Ok(None);
                }
            }
        }
        // For the exact-leaf oblivious grow the STRUCTURE der is the RMSE residual der (the MVP
        // structural der — the split histogram is driven by `target - approx`; the leaf VALUES
        // are OVERWRITTEN by the device Exact order statistic in `grow_one`). Upstream
        // quantile-der split parity + the full-tree Kaggle oracle are the Plan-09 sign-off; the
        // leaf-VALUE numerics are locked ≤1e-4 by the `kernels::exact_quantile` self-oracle.
        // Otherwise (Newton path) the covered der seam (RMSE / Logloss); a non-covered loss
        // declines. Exact is only reachable on the oblivious path (nonsym declined exact above).
        let der_kernel = match leaf_method {
            DeviceLeafMethod::Exact { .. } => DerBinaryKernel::RmseGradient,
            DeviceLeafMethod::Newton => match map_der_kernel(loss) {
                Some(k) => k,
                None => return Ok(None),
            },
        };
        let score_fn = match map_score_fn(score_function) {
            Some(s) => s,
            None => return Ok(None),
        };

        // Degenerate dimensions decline (an empty problem has nothing to grow on device).
        if n == 0 || n_features == 0 || n_bins == 0 {
            return Ok(None);
        }

        // The resident oblivious partition-histogram fill dispatches only the non-binary
        // {32,64,128,256} line widths (`1 << bits`, bits 5..=8). Any OTHER `n_bins` — e.g. the
        // real-world `border_count=32` → 33 bins, or the CatBoost default 254 borders → 255
        // bins — is handled by PADDING the histogram LINE width up to the next dispatched
        // family width (`n_bins_line`): the padding cells stay zero (they contribute nothing
        // to any left/right sum) and their phantom borders are excluded from the split argmin
        // (`n_bins_used` in `find_optimal_split_partition_kernel`), so the scored candidate
        // set — and therefore the chosen splits — stay bit-identical to the unpadded CPU
        // enumeration. Before this padding, every non-family width silently declined to the
        // CPU grower here, which made the device path unreachable from any real `.fit()`
        // (the SILENT-CPU-FALLBACK root cause of the 2026-07 GPU-speed investigation).
        // Only `n_bins > 256` (a >8-bit quantization) still declines (D-04).
        // The non-symmetric (Plan 03) / Region (Plan 04) grows score via the whole-subset
        // `pointwise_hist2` path per node, which keeps its own dispatch — skip for them.
        let n_bins_line = if nonsym_policy.is_none() && !region_active {
            match pad_hist_line_bins(n_bins) {
                Some(w) => w,
                None => return Ok(None),
            }
        } else {
            n_bins
        };

        // --- Host-side validation (V5): the resident grow skips per-tree guards, so validate
        // the value ranges ONCE here (T-10-18 residency + the histogram value-range contract).
        let cindex_stride = n_features.checked_mul(n).ok_or_else(|| {
            CbError::OutOfRange(format!(
                "n_features ({n_features}) * n ({n}) overflows usize (cindex stride)"
            ))
        })?;
        if bins_feature_major.len() != cindex_stride {
            return Err(CbError::LengthMismatch {
                column: "bins_feature_major".to_owned(),
                expected: cindex_stride,
                actual: bins_feature_major.len(),
            });
        }
        if weight.len() != n {
            return Err(CbError::LengthMismatch {
                column: "weight".to_owned(),
                expected: n,
                actual: weight.len(),
            });
        }
        // Every quantized bin must fit the dispatched line size (`n_bins`); a value >= n_bins
        // would write bin_sums out of bounds in the non-binary fill (which does not mask).
        if let Some(&bad) = bins_feature_major.iter().find(|&&b| (b as usize) >= n_bins) {
            return Err(CbError::OutOfRange(format!(
                "cindex bin value {bad} >= n_bins ({n_bins}); would write bin_sums out of bounds"
            )));
        }

        // --- One client owns every handle for the whole fit (Pitfall 3). ---
        let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
        let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

        // Phase 12 Plan 08 (GPUT-10): the CTR arm. When a single-permutation CTR config is covered
        // (`ctr_covered` confirmed the shape invariants at the gate), accumulate the ordered
        // read-before-increment target statistic for each CTR column ON device (resident across the
        // permutation, D-06), binarize the CTR VALUES into ADDITIONAL cindex columns on device, and
        // APPEND them to the feature-major bin matrix (each binarized to `n_bins` buckets, so the
        // uniform-`n_bins` resident histogram consumes them like any plain feature). The CTR VALUES
        // never touch the host — only the final integer bin columns are host-packed (the A2 cindex
        // discipline). The full-tree grow numerics over the augmented cindex are the Kaggle CUDA
        // sign-off (Plan 09); GPUT-10 stays Pending until then.
        let ctr_is_covered = nonsym_policy.is_none()
            && !region_active
            && ctr_covered(config, n, n_bins)
            && bootstrap_arm == BootstrapArm::NoDraw
            && !config.exact_leaf
            && config.mvs_lambda.is_none()
            && config.max_leaves.is_none();
        let (eff_bins, eff_n_features): (Vec<u32>, usize) = if ctr_is_covered {
            if let Some(ctr) = config.ctr.as_ref() {
                let ctr_columns = build_ctr_cindex_columns(&client, ctr, n)?;
                let mut augmented = bins_feature_major.to_vec();
                for col in &ctr_columns {
                    augmented.extend_from_slice(col);
                }
                (augmented, n_features + ctr_columns.len())
            } else {
                (bins_feature_major.to_vec(), n_features)
            }
        } else {
            (bins_feature_major.to_vec(), n_features)
        };
        let eff_cindex_stride = eff_n_features.checked_mul(n).ok_or_else(|| {
            CbError::OutOfRange(format!(
                "eff_n_features ({eff_n_features}) * n ({n}) overflows usize (cindex stride)"
            ))
        })?;

        // --- Pack the cindex (10-06) + build the device arrays ONCE (incl. CTR columns). ---
        // Pack with the (possibly padded) line width `n_bins_line`: the mask covers every
        // real bin value (`< n_bins <= n_bins_line`) losslessly, and the resident fill /
        // scorer address cells by the SAME padded width.
        let n_buckets_per_feature = vec![n_bins_line; eff_n_features];
        let packed = pack_cindex(&eff_bins, &n_buckets_per_feature, n)?;
        let (offsets_v, shifts_v, masks_v) = packed.device_arrays()?;
        let num_words = packed.words.len();

        // Identity object-visiting order (whole-dataset root): indices[i] = i.
        let indices: Vec<u32> = (0..n as u32).collect();

        // Upload ALL resident handles ONCE. der1/weight/approx are channel-typed (f32 on
        // wgpu, f64 elsewhere) via the shared helper; cindex/indices/TCFeature are u32.
        let plain_cindex_h = client.create(cubecl::bytes::Bytes::from_elems(eff_bins.clone()));
        let cindex_words_h = client.create(cubecl::bytes::Bytes::from_elems(packed.words.clone()));
        let offsets_h = client.create(cubecl::bytes::Bytes::from_elems(offsets_v));
        let shifts_h = client.create(cubecl::bytes::Bytes::from_elems(shifts_v));
        let masks_h = client.create(cubecl::bytes::Bytes::from_elems(masks_v));
        let indices_h = client.create(cubecl::bytes::Bytes::from_elems(indices));
        let weight_h = upload_channel_floats(&client, weight);
        // The running approx starts all-zero (the RMSE-from-zero MVP; boost_from_average is
        // out of scope, the cross-oracle uses the SAME zero start).
        let approx_h = upload_channel_floats(&client, &vec![0.0_f64; n]);

        // Phase 12 Plan 03: capture the non-symmetric grow state (host-driven; keeps host
        // copies of the bins + weights and re-derives der1 from the caller's approx per tree).
        let nonsym = nonsym_policy.map(|policy| NonsymState {
            policy,
            bins: bins_feature_major.to_vec(),
            weight: weight.to_vec(),
            max_depth: depth,
            max_leaves: config.max_leaves.unwrap_or(usize::MAX),
            min_data_in_leaf: config.min_data_in_leaf,
            der_kernel,
        });

        // Phase 12 Plan 04: capture the Region-grow state (host-driven; keeps host copies of the
        // bins + weights and re-derives der1 from the caller's approx per tree, exactly like the
        // non-sym path). `grow_one` dispatches to `grow_region_tree` (walk-until-diverge PATH).
        let region = region_active.then(|| RegionState {
            bins: bins_feature_major.to_vec(),
            weight: weight.to_vec(),
            max_depth: depth,
            min_data_in_leaf: config.min_data_in_leaf,
            der_kernel,
        });

        // Phase 12 Plan 05: capture the Exact-leaf state (oblivious path only; nonsym declined
        // exact above). `grow_one` overwrites the Newton leaf values with the device weighted
        // quantile per leaf, from the host residuals + this weight copy.
        let exact_leaf = match leaf_method {
            DeviceLeafMethod::Exact { alpha, delta, mape } if nonsym_policy.is_none() => {
                Some(ExactLeafState { alpha, delta, mape, weight: weight.to_vec() })
            }
            _ => None,
        };

        // Phase 12 Plan 06: capture the bootstrap state (oblivious path only; nonsym requires
        // `bootstrap_type == No`). Seed the CONTINUOUS training stream ONCE from `config.rng_seed`;
        // `grow_one` snapshots its O(1) base per tree and draws the device-resident sample.
        let bootstrap = match bootstrap_arm {
            BootstrapArm::Device(kind) if nonsym_policy.is_none() => Some(BootstrapState {
                kind,
                rng: TFastRng64::from_seed(config.rng_seed),
                sample_rate: f64::from(config.sample_rate),
                bagging_temperature: 1.0,
            }),
            _ => None,
        };

        // Phase 12 Plan 07 (GPUT-17): capture the MVS state (oblivious path only; the coverage gate
        // already required the caller-pinned `mvs_lambda`). Seeds the CONTINUOUS training stream ONCE
        // from `config.rng_seed`; `grow_one` takes the per-tree `rand_seed` and draws the resident
        // MVS sample over the resident derivatives. Mutually exclusive with `bootstrap`.
        let mvs = match bootstrap_arm {
            BootstrapArm::Mvs if nonsym_policy.is_none() => Some(MvsState {
                rng: TFastRng64::from_seed(config.rng_seed),
                sample_rate: f64::from(config.sample_rate),
                lambda: config.mvs_lambda,
                der_kernel,
            }),
            _ => None,
        };

        Ok(Some(Self {
            client,
            plain_cindex_h,
            cindex_words_h,
            offsets_h,
            shifts_h,
            masks_h,
            indices_h,
            weight_h,
            target_h: None,
            approx_h,
            der1_h: None,
            num_words,
            n,
            n_bins,
            n_bins_line,
            n_features: eff_n_features,
            cindex_stride: eff_cindex_stride,
            depth,
            scaled_l2,
            score_fn,
            learning_rate,
            der_kernel,
            config: config.clone(),
            nonsym,
            exact_leaf,
            bootstrap,
            mvs,
            region,
            // Pointwise path: no pairwise state (byte-unchanged, D-04). A covered pairwise fit is
            // gated + declined ABOVE (the pairwise arm returns Ok(None) pending the Plan-02
            // per-tree pair/group seam), so this construction is only ever reached pointwise.
            pairwise: None,
            // Pointwise path: no ranking state (byte-unchanged, D-04). A covered ranking fit is
            // gated + declined ABOVE (the ranking arm returns Ok(None) pending the per-tree
            // query-descriptor grow seam), so this construction is only ever reached pointwise.
            ranking: None,
            // Scalar path: no multi-output state (byte-unchanged, D-04). A covered multi-output fit
            // is gated + declined ABOVE (the multiclass arm returns Ok(None) pending the per-tree
            // shared multi-dim grow seam), so this construction is only ever reached scalar.
            multiclass: None,
            // Plain path: no ordered state (byte-unchanged, D-04). A covered ordered fit is gated +
            // declined ABOVE (the ordered arm returns Ok(None) pending the per-tree ordered
            // permutation-descriptor grow seam), and this construction is only ever reached on the
            // Plain (`boosting_type_is_plain`) path, so it is never ordered.
            ordered: None,
            // Pointwise path: no Langevin state (byte-unchanged, D-04). A covered Langevin fit records
            // its coverage decision via `map_langevin_coverage` but declines to CPU (the noise
            // coefficient + per-tree grow seam are a forward dependency), and a `*Pairwise` + Langevin
            // config is intercepted / declined above (A4), so this construction is only ever reached
            // on a non-Langevin pointwise fit.
            langevin: None,
        }))
    }

    /// The object count this session was opened for (the seam validates the passed
    /// `approx`/`target` against it).
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// The EFFECTIVE resident feature count — the plain feature count plus any binarized CTR
    /// columns appended during `begin` (Phase 12 Plan 08, GPUT-10). Equals the input `n_features`
    /// unless a covered CTR config augmented the cindex. The resident histogram loop iterates this
    /// many features.
    #[must_use]
    pub fn n_features_effective(&self) -> usize {
        self.n_features
    }

    /// Grow ONE tree over the resident state, advancing the device-resident boosting: it
    /// recomputes the residual `der1` from the resident approx device-side (no read-back),
    /// grows a depth-1 oblivious tree over the resident matrix (uploaded once at `begin`),
    /// updates the resident approx ON DEVICE via `apply_leaf_delta`, and chains `der1` for
    /// the next tree — returning the host-typed [`DeviceGrownTree`] (`leaf_values` UNSCALED,
    /// the 10-02 contract; `leaf_of` populated length `n` for the structure oracle).
    ///
    /// `target` is uploaded ONCE on the first call (the seam supplies it per tree; it is
    /// fixed for the whole fit) and reused thereafter; a `target` whose length disagrees with
    /// a prior call surfaces a typed [`CbError`]. The resident approx is authoritative for the
    /// device pass (in the covered Plain/fold=1/from-zero regime it tracks the caller's approx
    /// exactly). No `unwrap`/`expect`/`panic`/indexing (workspace lints + D-13).
    pub fn grow_one(&mut self, approx: &[f64], target: &[f64]) -> CbResult<DeviceGrownTree> {
        if target.len() != self.n {
            return Err(CbError::LengthMismatch {
                column: "target".to_owned(),
                expected: self.n,
                actual: target.len(),
            });
        }

        // Phase 12 Plan 04 (GPUT-18, D-03a): the REGION arm. The fit committed to a
        // `grow_policy=Region` device grow — re-derive der1 on the host from the caller's
        // `approx` + target (the covered regime is unit weights / bias 0, so this matches the CPU
        // `compute_gradients` bit-for-bit) and grow the Region PATH via the host-driven device
        // scorer. Emits a `region_path` + `depth+1` leaf values; the boosting Region fold arm
        // materializes the `RegionTree`. Checked BEFORE the non-sym arm (they are mutually
        // exclusive — `region` and `nonsym` are never both `Some`).
        if let Some(rg) = self.region.as_ref() {
            if approx.len() != self.n {
                return Err(CbError::LengthMismatch {
                    column: "approx".to_owned(),
                    expected: self.n,
                    actual: approx.len(),
                });
            }
            let der1 = host_der1(rg.der_kernel, approx, target);
            return grow_region_tree(
                &der1,
                &rg.weight,
                &rg.bins,
                self.n,
                self.n_bins,
                self.n_features,
                rg.max_depth,
                rg.min_data_in_leaf,
                self.scaled_l2,
                self.score_fn,
            );
        }

        // Phase 12 Plan 03 (GPUT-18): the NON-SYMMETRIC arm. The fit committed to a Depthwise /
        // Lossguide device grow — re-derive der1 on the host from the caller's `approx` + target
        // (the covered regime is unit weights / bias 0, so this matches the CPU
        // `compute_gradients` bit-for-bit) and grow the non-symmetric node graph via the
        // host-driven device scorer. Emits `step_nodes` / `node_id_to_leaf_id` + per-node splits
        // + `calc_average` leaf values; the boosting fold arm materializes the `NonSymmetricTree`.
        if let Some(ns) = self.nonsym.as_ref() {
            if approx.len() != self.n {
                return Err(CbError::LengthMismatch {
                    column: "approx".to_owned(),
                    expected: self.n,
                    actual: approx.len(),
                });
            }
            let der1 = host_der1(ns.der_kernel, approx, target);
            return grow_nonsym_tree(
                ns.policy,
                &der1,
                &ns.weight,
                &ns.bins,
                self.n,
                self.n_bins,
                self.n_features,
                ns.max_depth,
                ns.max_leaves,
                ns.min_data_in_leaf,
                self.scaled_l2,
                self.score_fn,
            );
        }

        // Phase 12 Plan 05 (GPUT-19): the EXACT-LEAF oblivious arm re-syncs the resident approx
        // from the caller's host `approx` each tree. The leaf-value override at the end advances
        // the TRUE approx via the device Exact order statistic (NOT the Newton leaves the
        // resident grow applies on-device), so the resident approx must track the caller —
        // otherwise the next tree's der1 + split structure would drift from the true,
        // exact-leaf-advanced approx. The extra per-tree upload is confined to the exact path.
        if self.exact_leaf.is_some() {
            if approx.len() != self.n {
                return Err(CbError::LengthMismatch {
                    column: "approx".to_owned(),
                    expected: self.n,
                    actual: approx.len(),
                });
            }
            let target_h = match &self.target_h {
                Some(h) => h.clone(),
                None => {
                    let h = upload_channel_floats(&self.client, target);
                    self.target_h = Some(h.clone());
                    h
                }
            };
            self.approx_h = upload_channel_floats(&self.client, approx);
            self.der1_h = Some(launch_der_binary_resident(
                &self.client,
                self.approx_h.clone(),
                target_h,
                self.der_kernel,
                self.n,
            )?);
        }

        // Upload the target ONCE (first call), then initialise the resident der1 from the
        // resident (zero-start) approx device-side — NO read-back. (The exact-leaf arm above
        // already set both handles from the fresh approx, so this init is skipped for it.)
        if self.target_h.is_none() {
            let target_h = upload_channel_floats(&self.client, target);
            let der1_h = launch_der_binary_resident(
                &self.client,
                self.approx_h.clone(),
                target_h.clone(),
                self.der_kernel,
                self.n,
            )?;
            self.target_h = Some(target_h);
            self.der1_h = Some(der1_h);
        }

        // Clone-out the resident der1/target for this grow (share, not copy).
        let der1_h = match &self.der1_h {
            Some(h) => h.clone(),
            None => {
                return Err(CbError::Degenerate(
                    "grow_one: resident der1 handle missing after init".to_owned(),
                ))
            }
        };
        let target_h = match &self.target_h {
            Some(h) => h.clone(),
            None => {
                return Err(CbError::Degenerate(
                    "grow_one: resident target handle missing after init".to_owned(),
                ))
            }
        };

        // Phase 12 Plan 06 (GPUT-09): the BOOTSTRAP arm. When a covered non-`No` `bootstrap_type`
        // is active, draw the device-resident per-object sample from the CONTINUOUS stream's O(1)
        // base state, fold it into a per-tree weight (`tree_weight = weight * sample`, on device),
        // and advance the stream by the draws this tree consumed. The base `weight_h` is untouched
        // (reused next tree); the default (no-bootstrap) path passes it through byte-unchanged (D-04).
        // The RNG/client borrows are scoped so the O(1) stream advance (mutable `self.bootstrap`)
        // completes before the launch reads `self.client`.
        let mut bootstrap_params: Option<(DeviceBootstrapKind, [u64; 4], u64, f64, f64)> = None;
        if let Some(bs) = self.bootstrap.as_mut() {
            let base = bs.rng.raw_state();
            // Bayesian consumes ONE main-stream draw for `rand_seed`; the per-block streams branch
            // off it. Bernoulli/Poisson draw sequentially from the base (advanced by `n` below).
            let rand_seed = match bs.kind {
                DeviceBootstrapKind::Bayesian => bs.rng.gen_rand(),
                DeviceBootstrapKind::Bernoulli | DeviceBootstrapKind::Poisson => 0,
            };
            bootstrap_params =
                Some((bs.kind, base, rand_seed, bs.sample_rate, bs.bagging_temperature));
            // Advance the continuous stream to the next tree's phase (Bernoulli/Poisson consume one
            // `gen_rand` per object; Bayesian already advanced by the single `rand_seed` draw).
            match bs.kind {
                DeviceBootstrapKind::Bernoulli | DeviceBootstrapKind::Poisson => {
                    // IN-02: Bernoulli consumes exactly one `gen_rand` per object, so the
                    // `n`-draw advance is draw-faithful. Poisson (Knuth) consumes a VARIABLE
                    // number of draws per object, so this advance is a deterministic-but-
                    // arbitrary phase, NOT aligned to the draws actually consumed. That is
                    // fine under the current scope (Poisson is validated for determinism
                    // only — same seed ⇒ same weights — and has no CPU oracle). If a Poisson
                    // parity oracle is ever added, make the kernel emit its consumed-draw
                    // count (or advance the stream on-device) so this matches consumption.
                    bs.rng.advance(self.n as u64)
                }
                DeviceBootstrapKind::Bayesian => {}
            }
        }

        // Phase 12 Plan 07 (GPUT-17): the MVS arm (mutually exclusive with the bootstrap arm above —
        // a distinct `bootstrap_type`). Take the ONE main-stream `GenRand()` as the per-tree
        // `rand_seed` (the per-block streams branch off it); the MVS kernel reduces over the resident
        // derivatives (`der1_h`). λ is the caller-pinned `config.mvs_lambda` (the coverage gate
        // required it), or the iter-0 `mean(|der|)^2` derived from the caller approx as a defensive
        // fallback. With `sample_rate >= 1.0` upstream short-circuits to all-`1.0` with ZERO draws —
        // pass the weight through unchanged and consume nothing.
        let mut mvs_params: Option<(u64, f64, f64)> = None;
        if let Some(mvs) = self.mvs.as_mut() {
            if mvs.sample_rate < 1.0 {
                let lambda = match mvs.lambda {
                    Some(l) => l,
                    None => {
                        if approx.len() != self.n {
                            return Err(CbError::LengthMismatch {
                                column: "approx".to_owned(),
                                expected: self.n,
                                actual: approx.len(),
                            });
                        }
                        mvs_lambda_from_der(&host_der1(mvs.der_kernel, approx, target))
                    }
                };
                // rand_seed = the ONE main-stream draw; then two `performRandomChoice=false`
                // compensation draws (`bootstrap()` Mvs branch) advance the stream to the next tree.
                let rand_seed = mvs.rng.gen_rand();
                mvs.rng.gen_rand();
                mvs.rng.gen_rand();
                mvs_params = Some((rand_seed, mvs.sample_rate, lambda));
            }
        }

        let sample_h = if let Some((kind, base, rand_seed, rate, temp)) = bootstrap_params {
            Some(launch_bootstrap_weights_resident(
                &self.client,
                kind,
                base,
                rand_seed,
                rate,
                temp,
                self.n,
            )?)
        } else if let Some((rand_seed, rate, lambda)) = mvs_params {
            Some(launch_mvs_weights_resident(
                &self.client,
                &der1_h,
                rand_seed,
                rate,
                lambda,
                self.n,
            )?)
        } else {
            None
        };
        let tree_weight_h = match sample_h {
            Some(s) => Some(fold_weights_resident(
                &self.client,
                &self.weight_h,
                &s,
                self.n,
            )?),
            None => None,
        };
        let weight_ref = tree_weight_h.as_ref().unwrap_or(&self.weight_h);

        // Grow one tree over the resident handles; take ownership of the resident approx
        // (updated in place on device) and swap it back afterwards.
        let approx_h = self.approx_h.clone();
        let (tree, approx_next, der1_next) = grow_oblivious_tree_resident(
            &self.client,
            approx_h,
            &der1_h,
            weight_ref,
            &self.plain_cindex_h,
            &self.cindex_words_h,
            &self.offsets_h,
            &self.shifts_h,
            &self.masks_h,
            &self.indices_h,
            &target_h,
            self.num_words,
            self.n,
            self.n_bins_line,
            self.n_bins,
            self.n_features,
            self.cindex_stride,
            self.depth,
            self.scaled_l2,
            self.score_fn,
            self.learning_rate,
            self.der_kernel,
        )?;

        // Advance the resident state for the next tree.
        // IN-03: in the EXACT-leaf arm these two handles are overwritten from the
        // freshly re-uploaded caller approx at the top of the NEXT call (der1/approx
        // re-sync), so this Newton-updated writeback is discarded there — correct
        // (the re-sync guarantees it), just redundant for the exact path. Kept
        // unconditional so the non-exact (Newton-resident) arm keeps its carried
        // state; the exact arm's extra der launch is harmless.
        self.approx_h = approx_next;
        self.der1_h = Some(der1_next);

        // Phase 12 Plan 05 (GPUT-19): OVERRIDE the Newton/calc_average leaf values with the
        // device Exact weighted-quantile order statistic per leaf (from the host residuals
        // `target - approx`), when the exact-leaf arm is active. The tree STRUCTURE + `leaf_of`
        // stay from the resident grow; only the leaf VALUES become the Exact order statistic
        // (≤1e-4 vs `exact_leaf_delta`, D-09). UNSCALED (learning_rate applied downstream — the
        // DeviceGrownTree contract), exactly like `exact_leaf_delta`.
        let leaf_values = match self.exact_leaf.as_ref() {
            Some(ex) => {
                compute_exact_leaf_values(&tree.leaf_of, tree.leaf_values.len(), approx, target, ex)?
            }
            None => tree.leaf_values,
        };

        Ok(DeviceGrownTree {
            splits: tree.splits,
            leaf_values,
            // Scalar oblivious emission: one value per leaf ⇒ approx_dim == 1 (the
            // block collapses to the flat scalar vector, byte-unchanged, D-04).
            approx_dim: 1,
            leaf_of: tree.leaf_of,
            // Oblivious / symmetric emission: the non-symmetric node-graph carrier stays EMPTY
            // (byte-unchanged, D-04). Only the Plan-03 non-sym device grow fills these.
            step_nodes: Vec::new(),
            node_id_to_leaf_id: Vec::new(),
            // Oblivious emission carries NO region path (Plan 04 Region grow fills it).
            region_path: Vec::new(),
        })
    }
}

/// Compute the per-leaf device Exact weighted-quantile leaf values (Phase 12 Plan 05,
/// GPUT-19). Groups the leaf members by `leaf_of[i]`, forms each member's residual
/// `target_i - approx_i` (widened through `f32`, matching upstream's `TVector<float>
/// leafSamples`) and weight (`weightsWithTargets[i] = weight_i/max(1,|target_i|)` for MAPE, A4;
/// else the object weight), then calls [`device_exact_leaf_delta`] per leaf. Returns the
/// UNSCALED per-leaf delta (length `n_leaves`); an empty leaf is `0.0` (the CPU-ref guard). No
/// `unwrap`/`expect`/`panic`/indexing (D-13).
fn compute_exact_leaf_values(
    leaf_of: &[u32],
    n_leaves: usize,
    approx: &[f64],
    target: &[f64],
    ex: &ExactLeafState,
) -> CbResult<Vec<f64>> {
    let n = leaf_of.len();
    let mut per_res: Vec<Vec<f32>> = vec![Vec::new(); n_leaves];
    let mut per_w: Vec<Vec<f64>> = vec![Vec::new(); n_leaves];
    for i in 0..n {
        let leaf = leaf_of.get(i).copied().unwrap_or(0) as usize;
        let a = approx.get(i).copied().unwrap_or(0.0);
        let t = target.get(i).copied().unwrap_or(0.0);
        let r = (t - a) as f32;
        let w = if ex.mape {
            // IN-01: upstream folds the object weight into the MAPE weight
            // (`weightsWithTargets[i] = weight_i / max(1, |target_i|)`). Multiply
            // in `ex.weight` so the weighted case stays correct if the exact path
            // is ever wired with non-unit weights (unit-weight regime unchanged).
            ex.weight.get(i).copied().unwrap_or(1.0) / f64::max(1.0, t.abs())
        } else {
            ex.weight.get(i).copied().unwrap_or(1.0)
        };
        if let (Some(pr), Some(pw)) = (per_res.get_mut(leaf), per_w.get_mut(leaf)) {
            pr.push(r);
            pw.push(w);
        }
    }
    let mut values = vec![0.0_f64; n_leaves];
    for l in 0..n_leaves {
        let res = per_res.get(l).map_or(&[][..], |v| v.as_slice());
        let wts = per_w.get(l).map_or(&[][..], |v| v.as_slice());
        if let Some(slot) = values.get_mut(l) {
            *slot = device_exact_leaf_delta(res, wts, ex.alpha, ex.delta)?;
        }
    }
    Ok(values)
}
