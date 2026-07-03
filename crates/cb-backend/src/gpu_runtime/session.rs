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
    DeviceBootstrapType, DeviceGrownTree, DeviceGrowPolicy, DeviceTrainConfig, EScoreFunction, Loss,
};
use cb_core::{CbError, CbResult};

use crate::gpu_runtime::cindex::pack_cindex;
use crate::gpu_runtime::{
    grow_oblivious_tree_resident, launch_der_binary_resident, upload_channel_floats,
    DerBinaryKernel,
};
use crate::kernels::nonsym_grow::{grow_nonsym_tree, NonsymPolicy};
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
        if depth == 0 || !boosting_type_is_plain || fold_count != 1 {
            return Ok(None);
        }
        // Phase 12 Plan 03 (GPUT-18): the grow-policy arm. `map_grow_policy` returns the device
        // non-symmetric strategy for Depthwise / Lossguide (flipped ON this plan), or `None` for
        // the oblivious / symmetric path (which is the byte-unchanged Plan-01 resident loop) and
        // for the not-yet-covered Region policy (Plan 04, declines below).
        let nonsym_policy = map_grow_policy(config.grow_policy);
        match nonsym_policy {
            None => {
                // Region is a distinct policy with no device kernel yet (Plan 04) — decline it
                // explicitly (never let it fall through to the oblivious path). SymmetricTree
                // rides the Plan-01 covered-regime gate unchanged (D-10-01 all-or-nothing: any
                // OTHER non-default family flag still routes to the CPU grower).
                if config.grow_policy != DeviceGrowPolicy::SymmetricTree {
                    return Ok(None);
                }
                if !config.is_covered_regime() {
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
        let der_kernel = match map_der_kernel(loss) {
            Some(k) => k,
            None => return Ok(None),
        };
        let score_fn = match map_score_fn(score_function) {
            Some(s) => s,
            None => return Ok(None),
        };

        // Degenerate dimensions decline (an empty problem has nothing to grow on device).
        if n == 0 || n_features == 0 || n_bins == 0 {
            return Ok(None);
        }

        // The device histogram fill (`hist2_launch_resident`) only dispatches these line
        // sizes: BINARY_BINS (2), HALF_BYTE_BINS (16), and the non-binary {32,64,128,256}
        // widths. Any other `n_bins` (e.g. the default 254-border quantization → n_bins=255)
        // would commit the whole fit to the device (D-10-01 all-or-nothing) and then hard-fail
        // at grow time. Decline to the byte-unchanged CPU grower (D-04) instead of a hard
        // failure. Keep this set in sync with `hist2_launch_resident`'s dispatch.
        // This dispatch restriction applies ONLY to the resident oblivious partition-histogram
        // fill. The non-symmetric grow (Plan 03) scores via the whole-subset `pointwise_hist2`
        // path per node, which has no such line-size restriction — so skip the check for it.
        if nonsym_policy.is_none() && !matches!(n_bins, 2 | 16 | 32 | 64 | 128 | 256) {
            return Ok(None);
        }

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

        // --- Pack the cindex (10-06) + build the device arrays ONCE. ---
        let n_buckets_per_feature = vec![n_bins; n_features];
        let packed = pack_cindex(bins_feature_major, &n_buckets_per_feature, n)?;
        let (offsets_v, shifts_v, masks_v) = packed.device_arrays()?;
        let num_words = packed.words.len();

        // --- One client owns every handle for the whole fit (Pitfall 3). ---
        let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
        let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

        // Identity object-visiting order (whole-dataset root): indices[i] = i.
        let indices: Vec<u32> = (0..n as u32).collect();

        // Upload ALL resident handles ONCE. der1/weight/approx are channel-typed (f32 on
        // wgpu, f64 elsewhere) via the shared helper; cindex/indices/TCFeature are u32.
        let plain_cindex_h = client.create(cubecl::bytes::Bytes::from_elems(bins_feature_major.to_vec()));
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
            n_features,
            cindex_stride,
            depth,
            scaled_l2,
            score_fn,
            learning_rate,
            der_kernel,
            config: config.clone(),
            nonsym,
        }))
    }

    /// The object count this session was opened for (the seam validates the passed
    /// `approx`/`target` against it).
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
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

        // Upload the target ONCE (first call), then initialise the resident der1 from the
        // resident (zero-start) approx device-side — NO read-back.
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

        // Grow one tree over the resident handles; take ownership of the resident approx
        // (updated in place on device) and swap it back afterwards.
        let approx_h = self.approx_h.clone();
        let (tree, approx_next, der1_next) = grow_oblivious_tree_resident(
            &self.client,
            approx_h,
            &der1_h,
            &self.weight_h,
            &self.plain_cindex_h,
            &self.cindex_words_h,
            &self.offsets_h,
            &self.shifts_h,
            &self.masks_h,
            &self.indices_h,
            &target_h,
            self.num_words,
            self.n,
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
        self.approx_h = approx_next;
        self.der1_h = Some(der1_next);

        Ok(DeviceGrownTree {
            splits: tree.splits,
            leaf_values: tree.leaf_values,
            leaf_of: tree.leaf_of,
            // Oblivious / symmetric emission: the non-symmetric node-graph carrier stays EMPTY
            // (byte-unchanged, D-04). Only the Plan-03 non-sym device grow fills these.
            step_nodes: Vec::new(),
            node_id_to_leaf_id: Vec::new(),
        })
    }
}
