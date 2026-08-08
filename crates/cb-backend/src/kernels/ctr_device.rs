//! GPUT-10 (Phase 12 Plan 08, W6): device ordered / one-hot / tensor CTR accumulation, the
//! highest-uncertainty categorical family. Ordered target-statistic CTRs accumulate ON device,
//! resident across the learn permutation (D-06 — no per-fit host round-trip of the CTR values),
//! read the prefix statistic BEFORE incrementing (read-before-increment, the no-leakage
//! invariant), and are binarized into ADDITIONAL cindex columns the histogram loop already
//! reads (the CTR→cindex JOIN below).
//!
//! # What lives here (production, NOT `#[cfg(test)]`)
//!
//! A serial `#[cube]` ordered-prefix kernel that transcribes CatBoost's
//! `online_ctr.cpp:300-307` `CalcQuantizedCtrs` simple-binclf path (mirrored inline — the kernel
//! body cannot reach `cb_train`, and cb-backend must NEVER gain a `cb-train` dep, the
//! feature-unification landmine, Pattern B). The prefix is INHERENTLY SEQUENTIAL (each document
//! reads its bucket's running `(N0, N1)` before adding its own label), so — like the bootstrap
//! draw ([`crate::kernels::bootstrap_device`]) — it runs on unit 0 as a serial device scan. It
//! stays device-resident (the per-bucket count scratch + the per-object output live on the
//! client for the whole fit), and it needs NO `Atomic<u64>` (the ordered binclf prefix is EXACT
//! INTEGER counting, not a float reduction — Pattern C's deterministic reduce is only required
//! for the FLOAT CTR sums, which this binclf ordered-TS path does not take).
//!
//! - **Ordered target statistic (Borders)** — per object `good = N[1]`, `total = N[0] + N[1]`
//!   read BEFORE `++N[targetClass]`; `value = (good + prior) / (total + 1)`
//!   ([`crate::kernels::ctr_device`] mirrors `calc_ctr.rs::calc_ctr_online`). The FIRST document
//!   in a bucket reads the PRIOR alone (`good = total = 0`) — it never sees its own label.
//! - **One-hot** — the SAME prefix kernel with the bucket key = the raw (small-cardinality)
//!   category bin instead of the perfect-hash TS bin. No separate kernel (A5 — the device CTR
//!   math is shared); only the bucket source differs, and both cross the seam as plain `bins`.
//! - **Tensor / feature-combination** — a host projection pre-step ([`combine_projection_bins`])
//!   folds each object's member category hashes into one combined key (`TProjection::combined_hash`
//!   / `fold_cat_hash` / `calc_hash`, transcribed inline) and remaps to dense first-seen bins;
//!   the SAME ordered-prefix kernel then runs on the combined bins (A5).
//!
//! # CTR → cindex binarization JOIN
//!
//! [`binarize_ctr_kernel`] binarizes the accumulated device CTR values into bin indices on
//! device (`bin = #{borders < value}`, the upstream `> bin` threshold convention every cindex
//! consumer already uses), producing ADDITIONAL cindex columns the histogram loop reads with no
//! host round-trip of the CTR VALUES. The border tables are the CPU ≤1e-5 quantization reference
//! (uploaded once per fit; quantization stays host — the A2 cindex discipline extended to CTR).
//!
//! # f64-typed seam (WR-02, shared with the bootstrap seam)
//!
//! The CTR value `(good + prior) / (total + 1)` and its borders are f64; WGSL has neither f64
//! nor u64, so a genuine `wgpu` backend surfaces a typed [`CbError::OutOfRange`] rather than an
//! opaque JIT crash (the rocm/cuda/cpu path is unaffected — the cpu backend runs the serial
//! scan self-oracle in-env). No `-inf` literal in any `#[cube]` body (Pattern D). No
//! `unwrap`/`expect`/`panic`/indexing in production (workspace lints + D-13); no `cb-train` dep.

use std::collections::HashMap;

use cubecl::prelude::*;
use cubecl::server::Handle;

use cb_core::{CbError, CbResult};

use crate::SelectedRuntime;

/// `MAGIC_MULT` (`projection.cpp` `TProjection::CalcHash`, `0x4906ba494954cb65`) — transcribed
/// inline for the host tensor-combination projection (Pattern B; no `cb-train` dep).
const MAGIC_MULT: u64 = 0x4906_ba49_4954_cb65;

// ===========================================================================
// Host tensor / feature-combination projection (A5) — plain host, no device.
// ===========================================================================

/// `TProjection::CalcHash` (`calc_hash`, transcribed): `MAGIC_MULT * (a + MAGIC_MULT * b)`, wrapping.
#[must_use]
fn calc_hash(a: u64, b: u64) -> u64 {
    MAGIC_MULT.wrapping_mul(a.wrapping_add(MAGIC_MULT.wrapping_mul(b)))
}

/// `fold_cat_hash` (`projection.rs`, transcribed): fold one member's category hash into the
/// running combined key with C++'s `(ui64)(int)hash` sign-extension.
#[must_use]
fn fold_cat_hash(running: u64, cat_hash: u32) -> u64 {
    let extended = i64::from(cat_hash as i32) as u64;
    calc_hash(running, extended)
}

/// Combine several categorical member bin columns (each `member_bins[m][obj]` an already-hashed
/// category code, feature-combination member order) into ONE combined-projection bin column plus
/// the distinct-bucket count (A5, `TProjection::combined_hash` + first-seen remap). Per object the
/// member codes are folded via [`fold_cat_hash`] into a combined key, then keys are remapped to
/// dense first-seen bins (the insertion-order perfect-hash remap the online accumulation keys on).
/// The combined bins feed the SAME [`ordered_ctr_prefix_kernel`] as a plain single feature (A5).
///
/// # Errors
/// [`CbError::LengthMismatch`] if any member column length disagrees with `n`.
pub(crate) fn combine_projection_bins(
    member_bins: &[Vec<u32>],
    n: usize,
) -> CbResult<(Vec<u32>, usize)> {
    for (m, col) in member_bins.iter().enumerate() {
        if col.len() != n {
            return Err(CbError::LengthMismatch {
                column: format!("ctr projection member {m}"),
                expected: n,
                actual: col.len(),
            });
        }
    }
    let mut remap: HashMap<u64, u32> = HashMap::new();
    let mut combined: Vec<u32> = Vec::with_capacity(n);
    for obj in 0..n {
        let mut key: u64 = 0;
        for col in member_bins {
            if let Some(&code) = col.get(obj) {
                key = fold_cat_hash(key, code);
            }
        }
        let next = remap.len() as u32;
        let bin = *remap.entry(key).or_insert(next);
        combined.push(bin);
    }
    let bucket_count = remap.len();
    Ok((combined, bucket_count))
}

// ===========================================================================
// #[cube] serial ordered-prefix CTR kernel (read-before-increment, resident scratch)
// ===========================================================================

/// Ordered target-statistic CTR over the learn permutation, read-before-increment (the no-leakage
/// invariant). Serial single-thread (unit 0) — the prefix is inherently sequential. `perm[p]` is
/// the object index at learn-order position `p`; `bins[doc]` its categorical bucket; `class[doc]`
/// its binclf class in `{0, 1}`; `prior` the additive CTR prior numerator (length-1). `counts` is
/// the resident per-bucket `[N0, N1]` scratch (length `2 * bucket_count`, PRE-ZEROED by the host).
///
/// Per position: read the bucket's `(N0, N1)` BEFORE incrementing → `total = N0 + N1`, `good` per
/// `mode` (below), `value[doc] = (good + prior) / (total + 1)`, then `++counts[2*bucket+class]`.
/// Outputs are OBJECT order (indexed by `doc`): `good`/`total` are exact integer counts (u32),
/// `value` is f64. No float reduction (integer counting is exact — Pattern C not needed here); no
/// `-inf` (Pattern D); every index derives from a bounds-validated host bucket count.
///
/// # `mode = [is_buckets, target_border_idx]` — the numerator selector (DCTR-06)
///
/// A length-2 `u32` runtime array, both elements HOST-VALIDATED by
/// [`launch_ordered_ctr_resident`]. Only the NUMERATOR varies with the CTR type; the denominator
/// is the bucket total in every mode. The generic rule is `online_class_prefix`
/// (`cb-train/src/ctr/online.rs:552-569`), itself the single-border collapse of upstream
/// `UpdateGoodCount` (`online_ctr.cpp:115-121`,
/// `if (ctrType == Buckets) *goodCount = curCount; else *goodCount -= curCount;` applied
/// cumulatively from `goodCount = Total`):
///
/// ```text
/// Buckets    -> N[b]
/// everything -> Total - Σ_{c ≤ b} N[c]
/// ```
///
/// **This kernel implements the explicit 2-class collapse**, valid because the simple ordered CTR
/// is binclf: `SIMPLE_CLASSES_COUNT == 2` (`cb-train/src/ctr/online.rs:52`), so the prefix is
/// exactly `[N0, N1]` and `Total = N0 + N1`:
///
/// | mode | `good` |
/// |---|---|
/// | `Buckets @ b = 0` | `N0` |
/// | `Buckets @ b = 1` | `N1` |
/// | `Borders @ b = 0` | `Total - N0 == N1` (the historical hard-coded value) |
/// | `Borders @ b = 1` | `0` — UNREACHABLE, pinned (see the body) |
///
/// A multiclass ordered CTR would need the generic loop instead; it is out of this seam's scope
/// (the host guard rejects `target_border_idx > 1`).
#[cube(launch)]
fn ordered_ctr_prefix_kernel(
    perm: &Array<u32>,
    bins: &Array<u32>,
    class: &Array<u32>,
    prior: &Array<f64>,
    mode: &Array<u32>,
    counts: &mut Array<u32>,
    good: &mut Array<u32>,
    total: &mut Array<u32>,
    value: &mut Array<f64>,
) {
    if ABSOLUTE_POS == 0 {
        let pr = prior[0];
        let is_buckets = mode[0];
        let b = mode[1];
        let n = perm.len();
        let mut p = 0usize;
        while p < n {
            let doc = perm[p] as usize;
            let bucket = bins[doc] as usize;
            let base = 2usize * bucket;
            // READ the prefix counts BEFORE incrementing (online_ctr.cpp:303-304).
            let n0 = counts[base];
            let n1 = counts[base + 1usize];
            let t = n0 + n1;
            // DCTR-06 numerator selection — the SIMPLE_CLASSES_COUNT == 2 collapse of
            // `online_class_prefix` (see this function's doc comment). `if`/`else` STATEMENTS
            // with a defaulted `let mut` (CubeCL conditionals manual: never an `if` EXPRESSION).
            // The scan is serial on unit 0, so the per-iteration branch costs nothing and
            // divergence is not a concern.
            let mut g = 0u32;
            if is_buckets == 1u32 {
                if b == 0u32 {
                    g = n0;
                } else {
                    g = n1;
                }
            } else if b == 0u32 {
                g = t - n0;
            }
            // The remaining case is Borders@1, which is UNREACHABLE
            // (`target_border_count(Borders, 2) == 1`, `ctr_helper.h:35-42`, and the host guard
            // never admits it). It is PINNED to the arithmetic value `Total - (N0 + N1) == 0`
            // rather than left undefined — carried by the `let mut g = 0u32` default above, so no
            // arm re-assigns a value that is never read.
            good[doc] = g;
            total[doc] = t;
            value[doc] = (f64::cast_from(g) + pr) / (f64::cast_from(t) + 1.0);
            // INCREMENT after read (learn set): ++N[class[doc]]. class is 0/1.
            let slot = base + (class[doc] as usize);
            counts[slot] = counts[slot] + 1u32;
            p += 1usize;
        }
    }
}

/// Binarize accumulated CTR VALUES into cindex bin indices on device (the CTR→cindex JOIN):
/// `bin[i] = #{ borders[j] : value[i] > borders[j] }` — the upstream `> bin` threshold convention
/// every cindex consumer already uses (so the emitted column drops straight into the histogram
/// loop). Elementwise, bounds-guarded (the host launches enough cubes to cover `out_bins`).
/// Generic over `F: Float` (AGENTS.md generics-float); no `-inf` literal (Pattern D).
#[cube(launch)]
fn binarize_ctr_kernel<F: Float>(values: &Array<F>, borders: &Array<F>, out_bins: &mut Array<u32>) {
    if ABSOLUTE_POS < out_bins.len() {
        let v = values[ABSOLUTE_POS];
        let k = borders.len();
        let mut bin = 0u32;
        let mut j = 0usize;
        while j < k {
            if v > borders[j] {
                bin += 1u32;
            }
            j += 1usize;
        }
        out_bins[ABSOLUTE_POS] = bin;
    }
}

// ===========================================================================
// Host launch wrappers (device-resident Handle + readback oracle wrapper)
// ===========================================================================

/// Reject the (impossible) wgpu f64/u64 CTR path with a typed error (WR-02), mirroring the
/// bootstrap seam. Kept in one place so every entry point agrees.
#[cfg(feature = "wgpu")]
fn wgpu_reject() -> CbError {
    CbError::OutOfRange(
        "device CTR requires an f64 device channel for the ordered target statistic; the wgpu \
         backend has none (WR-02). Use the rocm/cuda/cpu backend for the CTR accumulation."
            .to_owned(),
    )
}

/// The resident device CTR outputs for one categorical feature/projection: the three OBJECT-order
/// buffers held on the client WITHOUT read-back (D-06 residency). `value` feeds the binarize JOIN
/// into extra cindex columns; `good`/`total` are the integer prefix counts (kept for the oracle /
/// downstream diagnostics).
pub(crate) struct ResidentCtr {
    /// Per-object good count `N[1]` read before the label (u32, object order).
    #[allow(dead_code)] // consumed by the #[cfg(test)] ctr_device_test self-oracle (source/test separation)
    pub good: Handle,
    /// Per-object total count `N[0] + N[1]` read before the label (u32, object order).
    #[allow(dead_code)] // consumed by the #[cfg(test)] ctr_device_test self-oracle (source/test separation)
    pub total: Handle,
    /// Per-object online CTR value `(good + prior) / (total + 1)` (f64, object order).
    pub value: Handle,
}

/// `cb_train::ctr::ECtrType::Borders` — the i8 discriminant, transcribed BY VALUE (Pattern B /
/// C-3: `cb-backend` must never gain a `cb-train` dep). The full upstream list is
/// `0 Borders, 1 Buckets, 2 BinarizedTargetMeanValue, 3 FloatTargetMeanValue, 4 Counter,
/// 5 FeatureFreq` (`cb-train/src/ctr/mod.rs:96-108`; upstream `restrictions.h:20-32` limits the
/// CPU task type to `{Borders, Buckets, BinarizedTargetMeanValue, Counter}`).
pub(crate) const CTR_TYPE_BORDERS: i8 = 0;
/// `cb_train::ctr::ECtrType::Buckets` — see [`CTR_TYPE_BORDERS`] for the transcription rationale.
pub(crate) const CTR_TYPE_BUCKETS: i8 = 1;

/// The largest `target_border_idx` reachable at binclf: `SIMPLE_CLASSES_COUNT == 2`
/// (`cb-train/src/ctr/online.rs:52`) ⇒ the per-bucket class prefix is `[N0, N1]`, so a selector
/// above `1` names a class that does not exist.
const MAX_TARGET_BORDER_IDX: u32 = 1;

/// Accumulate the ordered CTR for one feature/projection ON device, resident across the
/// permutation (D-06), returning the resident output handles WITHOUT reading them back. `bins` is
/// the per-object bucket (object order); `perm` the learn permutation; `class` the binclf class;
/// `bucket_count` the distinct-bucket count (`max(bins) + 1`, host-validated). `client` owns the
/// scratch + outputs for the whole fit (residency, Pitfall 3). Empty `n` short-circuits.
///
/// # The `(ctr_type, target_border_idx)` numerator selector (DCTR-06)
///
/// The denominator (`total`, the bucket total) is the same for every CTR type; only the NUMERATOR
/// differs. `online_class_prefix` (`cb-train/src/ctr/online.rs:552-569`, the collapse of upstream
/// `UpdateGoodCount`, `online_ctr.cpp:115-121`) selects it as
/// `Buckets → N[b]`, everything else → `Total - Σ_{c ≤ b} N[c]`. Both arguments are host-validated
/// here and crossed into the kernel as the 2-element `mode` array `[is_buckets, target_border_idx]`
/// — see [`ordered_ctr_prefix_kernel`] for the 2-class collapse it implements.
/// `(CTR_TYPE_BORDERS, 0)` is the historical behaviour (`good = N1`) and is byte-unchanged.
///
/// # Errors
/// [`CbError::OutOfRange`] on the wgpu f64 path (WR-02); [`CbError::LengthMismatch`] if
/// `bins`/`class` disagree with `perm`; [`CbError::OutOfRange`] if `ctr_type` is not a class-prefix
/// type this kernel implements, or `target_border_idx` exceeds the binclf class count.
#[cfg_attr(feature = "wgpu", allow(unused_variables))]
pub(crate) fn launch_ordered_ctr_resident(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    perm: &[u32],
    bins: &[u32],
    class: &[u32],
    prior: f64,
    bucket_count: usize,
    n: usize,
    ctr_type: i8,
    target_border_idx: u32,
) -> CbResult<ResidentCtr> {
    // DCTR-06: validate the numerator selector host-side, mirroring the bin/class guards below.
    // `Counter` / `BinarizedTargetMeanValue` are NOT class-prefix types (their numerator is not
    // derivable from one bucket's class counts) and have their own device paths; admitting them
    // here would silently return the Borders numerator — a WRONG answer, not a worse one.
    if ctr_type != CTR_TYPE_BORDERS && ctr_type != CTR_TYPE_BUCKETS {
        return Err(CbError::OutOfRange(format!(
            "ctr_type {ctr_type} is not a class-prefix CTR type; the ordered prefix kernel \
             implements only Borders ({CTR_TYPE_BORDERS}) and Buckets ({CTR_TYPE_BUCKETS})"
        )));
    }
    if target_border_idx > MAX_TARGET_BORDER_IDX {
        return Err(CbError::OutOfRange(format!(
            "ctr target_border_idx {target_border_idx} > {MAX_TARGET_BORDER_IDX} (binclf has \
             SIMPLE_CLASSES_COUNT == 2 classes)"
        )));
    }
    if bins.len() != n || class.len() != n || perm.len() != n {
        return Err(CbError::LengthMismatch {
            column: "ctr ordered inputs".to_owned(),
            expected: n,
            actual: bins.len().min(class.len()).min(perm.len()),
        });
    }
    // WR-02: guard bin/class value ranges host-side before dispatch. The kernel indexes
    // `counts[2*bucket + class]` (length `2 * bucket_count`), so a bin >= bucket_count or a
    // class not in {0,1} is an out-of-bounds device access (UB). Mirror the histogram seam's
    // host-side bin guard rather than trusting callers to keep bins dense.
    if let Some(&bad) = bins.iter().find(|&&b| (b as usize) >= bucket_count) {
        return Err(CbError::OutOfRange(format!(
            "ctr bin value {bad} >= bucket_count ({bucket_count})"
        )));
    }
    if let Some(&bad) = class.iter().find(|&&c| c > 1) {
        return Err(CbError::OutOfRange(format!(
            "ctr class {bad} not in {{0,1}}"
        )));
    }
    if n == 0 {
        return Ok(ResidentCtr {
            good: client.empty(0),
            total: client.empty(0),
            value: client.empty(0),
        });
    }

    #[cfg(feature = "wgpu")]
    {
        return Err(wgpu_reject());
    }

    #[cfg(not(feature = "wgpu"))]
    {
        // Pre-zeroed per-bucket [N0, N1] scratch (2 * bucket_count u32); at least length 2 so an
        // all-zero-bin degenerate feature still has a valid bucket-0 slot.
        let scratch_len = bucket_count.max(1) * 2;
        let counts_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; scratch_len]));
        let perm_h = client.create(cubecl::bytes::Bytes::from_elems(perm.to_vec()));
        let bins_h = client.create(cubecl::bytes::Bytes::from_elems(bins.to_vec()));
        let class_h = client.create(cubecl::bytes::Bytes::from_elems(class.to_vec()));
        let prior_h = client.create(cubecl::bytes::Bytes::from_elems(vec![prior]));
        // DCTR-06 `mode = [is_buckets, target_border_idx]` (both host-validated above).
        let is_buckets = u32::from(ctr_type == CTR_TYPE_BUCKETS);
        let mode_h = client.create(cubecl::bytes::Bytes::from_elems(vec![
            is_buckets,
            target_border_idx,
        ]));
        let good_h = client.empty(n * std::mem::size_of::<u32>());
        let total_h = client.empty(n * std::mem::size_of::<u32>());
        let value_h = client.empty(n * std::mem::size_of::<f64>());

        // Serial single-thread launch (unit 0 loops the permutation); one cube, one unit.
        let count = CubeCount::Static(1, 1, 1);
        let dim = CubeDim { x: 1, y: 1, z: 1 };
        ordered_ctr_prefix_kernel::launch::<SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(perm_h, n) },
            unsafe { ArrayArg::from_raw_parts(bins_h, n) },
            unsafe { ArrayArg::from_raw_parts(class_h, n) },
            unsafe { ArrayArg::from_raw_parts(prior_h, 1) },
            unsafe { ArrayArg::from_raw_parts(mode_h, 2) },
            unsafe { ArrayArg::from_raw_parts(counts_h, scratch_len) },
            unsafe { ArrayArg::from_raw_parts(good_h.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(total_h.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(value_h.clone(), n) },
        );
        Ok(ResidentCtr {
            good: good_h,
            total: total_h,
            value: value_h,
        })
    }
}

/// Binarize resident CTR VALUES into an ADDITIONAL cindex bin column ON device (the CTR→cindex
/// JOIN), returning the resident bin handle WITHOUT read-back. `value_h` is the resident f64 CTR
/// value buffer ([`launch_ordered_ctr_resident`]); `borders` the per-CTR-column border table
/// (uploaded once per fit). The emitted `u32` bins use the `> bin` threshold convention the
/// histogram loop already reads. Empty short-circuits.
///
/// # Errors
/// [`CbError::OutOfRange`] on the wgpu f64 path (WR-02).
#[cfg(not(feature = "wgpu"))]
pub(crate) fn binarize_ctr_column_resident(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    value_h: &Handle,
    borders: &[f64],
    n: usize,
) -> CbResult<Handle> {
    if n == 0 {
        return Ok(client.empty(0));
    }
    let out = client.empty(n * std::mem::size_of::<u32>());
    let borders_h = client.create(cubecl::bytes::Bytes::from_elems(borders.to_vec()));
    let num_cubes = n.div_ceil(32).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim { x: 32, y: 1, z: 1 };
    binarize_ctr_kernel::launch::<f64, SelectedRuntime>(
        client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(value_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(borders_h, borders.len()) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), n) },
    );
    Ok(out)
}

/// The wgpu stub of [`binarize_ctr_column_resident`] — the CTR seam is f64 and wgpu has none, so
/// this path is never reached (the accumulation already rejected wgpu), but the symbol must exist
/// for the session's `cfg`-independent call site.
#[cfg(feature = "wgpu")]
pub(crate) fn binarize_ctr_column_resident(
    _client: &cubecl::client::ComputeClient<SelectedRuntime>,
    _value_h: &Handle,
    _borders: &[f64],
    _n: usize,
) -> CbResult<Handle> {
    Err(wgpu_reject())
}

/// Host-readback wrapper over the device ordered CTR (the self-oracle seam): accumulate the
/// resident CTR, then read the three buffers back to host `Vec`s (OBJECT order). This is NOT the
/// residency path (that keeps the handles on-device); it is the device-vs-CPU oracle exerciser. A
/// read-back failure surfaces [`CbError::Degenerate`] (WR-05), never a silent zero buffer.
///
/// Returns `(good, total, value)` in object order; `good`/`total` are widened to `i64` to match
/// the CPU reference's integer prefix schema.
///
/// Fixed at the historical `(Borders, target_border_idx = 0)` numerator; DCTR-06's
/// [`compute_ordered_ctr_host_mode`] is the seam that exercises the other reachable modes.
#[allow(dead_code)] // consumed by the #[cfg(test)] ctr_device_test self-oracle (source/test separation)
pub(crate) fn compute_ordered_ctr_host(
    perm: &[u32],
    bins: &[u32],
    class: &[u32],
    prior: f64,
    bucket_count: usize,
) -> CbResult<(Vec<i64>, Vec<i64>, Vec<f64>)> {
    compute_ordered_ctr_host_mode(
        perm,
        bins,
        class,
        prior,
        bucket_count,
        CTR_TYPE_BORDERS,
        0,
    )
}

/// DCTR-06: [`compute_ordered_ctr_host`] with the `(ctr_type, target_border_idx)` numerator
/// selector exposed, so the self-oracle can drive every mode reachable at binclf —
/// `(Borders, 0)`, `(Buckets, 0)`, `(Buckets, 1)` — against `online_class_prefix`.
///
/// # Errors
/// As [`launch_ordered_ctr_resident`] (including the host-validated selector range).
#[allow(dead_code)] // consumed by the #[cfg(test)] ctr_device_test self-oracle (source/test separation)
pub(crate) fn compute_ordered_ctr_host_mode(
    perm: &[u32],
    bins: &[u32],
    class: &[u32],
    prior: f64,
    bucket_count: usize,
    ctr_type: i8,
    target_border_idx: u32,
) -> CbResult<(Vec<i64>, Vec<i64>, Vec<f64>)> {
    let n = perm.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    let res = launch_ordered_ctr_resident(
        &client,
        perm,
        bins,
        class,
        prior,
        bucket_count,
        n,
        ctr_type,
        target_border_idx,
    )?;
    let good_b = client
        .read_one(res.good)
        .map_err(|e| CbError::Degenerate(format!("CubeCL CTR good read-back failed: {e:?}")))?;
    let total_b = client
        .read_one(res.total)
        .map_err(|e| CbError::Degenerate(format!("CubeCL CTR total read-back failed: {e:?}")))?;
    let value_b = client
        .read_one(res.value)
        .map_err(|e| CbError::Degenerate(format!("CubeCL CTR value read-back failed: {e:?}")))?;
    let good = bytemuck::cast_slice::<u8, u32>(&good_b)
        .iter()
        .map(|&g| i64::from(g))
        .collect();
    let total = bytemuck::cast_slice::<u8, u32>(&total_b)
        .iter()
        .map(|&t| i64::from(t))
        .collect();
    let value = bytemuck::cast_slice::<u8, f64>(&value_b).to_vec();
    Ok((good, total, value))
}

/// Host-readback wrapper over the device CTR→cindex binarize JOIN (the self-oracle seam):
/// accumulate the resident CTR, binarize its values into an extra cindex column on device, then
/// read that bin column back. Returns the per-object bin indices (`> bin` convention). A read-back
/// failure surfaces [`CbError::Degenerate`] (WR-05).
#[cfg(not(feature = "wgpu"))]
#[allow(dead_code)] // consumed by the #[cfg(test)] ctr_device_test self-oracle (source/test separation)
pub(crate) fn binarize_ctr_column_host(
    perm: &[u32],
    bins: &[u32],
    class: &[u32],
    prior: f64,
    bucket_count: usize,
    borders: &[f64],
) -> CbResult<Vec<u32>> {
    let n = perm.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    let res = launch_ordered_ctr_resident(
        &client,
        perm,
        bins,
        class,
        prior,
        bucket_count,
        n,
        CTR_TYPE_BORDERS,
        0,
    )?;
    let bins_h = binarize_ctr_column_resident(&client, &res.value, borders, n)?;
    let bytes = client
        .read_one(bins_h)
        .map_err(|e| CbError::Degenerate(format!("CubeCL CTR cindex read-back failed: {e:?}")))?;
    Ok(bytemuck::cast_slice::<u8, u32>(&bytes).to_vec())
}
