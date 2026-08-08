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

/// Device **Counter** CTR statistic (DCTR-09) — the whole-set per-bucket tally with a CONSTANT
/// denominator. This is a **sibling** of [`ordered_ctr_prefix_kernel`], deliberately NOT a mode on
/// it: the prefix kernel's loop is permutation-driven and read-before-increment, whereas Counter
/// has no permutation and reads the FINAL tally, so branching the prefix loop would carry a dead
/// mode through the hot path for no shared arithmetic.
///
/// # Upstream
/// `CalcOnlineCTRCounter` (`online_ctr.cpp:503-568`) with `CountOnlineCTRTotal` +
/// `counterCTRDenominator` (`online_ctr.cpp:713-729`); mirrored on the CPU by
/// `cb_train::ctr::online::online_counter_column` (`cb-train/src/ctr/online.rs:493-521`)
/// composed with `calc_ctr_online` at `ctr_feature.rs:296-309`:
///
/// ```text
/// totals[b]   = #{obj : bins[obj] == b}     // WHOLE learn set, no permutation
/// denominator = max_b totals[b]             // CONSTANT across objects
/// value[obj]  = (totals[bins[obj]] + prior) / (denominator + 1)
/// ```
///
/// `total[obj] = denominator` for every object is deliberate — it mirrors
/// `denoms = vec![denominator; n]` (`ctr_feature.rs:304`), so the emitted `(good, total, value)`
/// triple has the same shape as the ordered path's and quantizes through the SAME f64 border table
/// (`quantize_in_f32 == false`, C-7). Counter is **permutation independent**
/// (`IsPermutationDependentCtrType(Counter) == false`, `ctr_type.cpp:43-56`), which this kernel
/// gets structurally: it has no `perm` parameter.
///
/// # `counter_calc_method = Full` is NOT implemented and is unreachable on device
/// Upstream widens the tally AND the max denominator over the learn set plus every eval set
/// (`CountOnlineCTRTotal`, `online_ctr.cpp:713-729`; the CPU mirror is `online_counter_column`'s
/// `extra_bins`). This seam carries **no eval bins at all**, so the widening cannot be expressed
/// here, and the fit declines to the CPU whenever an eval set is present — T13 pins that boundary
/// with a negative test. Do not add an `extra_bins` argument without that test moving first.
///
/// # Shape
/// Serial single-thread (unit 0), three passes: tally, max, map. Counter columns are built ONCE
/// per fit inside `begin` (never per tree), so this is not a hot path and the serial form buys
/// D-06 residency with no atomics (`SPEC.md` §9). `counts` is the per-bucket tally scratch
/// (length `bucket_count`), **PRE-ZEROED by the host** — a reused non-zeroed buffer silently
/// doubles the tally. Every index derives from a host-validated bound
/// ([`launch_counter_ctr_resident`] range-checks `bins` against `bucket_count`). Generic over
/// `F: Float` for the value channel (AGENTS.md generics-float); the tally is `Array<u32>` because
/// it is an EXACT integer count. `while` with an explicit counter, `if` STATEMENTS only, no
/// `-inf` literal (Pattern D).
#[cube(launch)]
fn counter_ctr_kernel<F: Float>(
    bins: &Array<u32>,
    prior: &Array<F>,
    counts: &mut Array<u32>,
    good: &mut Array<u32>,
    total: &mut Array<u32>,
    value: &mut Array<F>,
) {
    if ABSOLUTE_POS == 0 {
        let pr = prior[0];
        let n = bins.len();
        let k = counts.len();
        // PASS 1 — tally the WHOLE set. No read-before-increment here: unlike the ordered
        // prefix, every object sees the FINAL tally (upstream counts first, then maps).
        let mut i = 0usize;
        while i < n {
            let b = bins[i] as usize;
            counts[b] = counts[b] + 1u32;
            i += 1usize;
        }
        // PASS 2 — the CONSTANT denominator: `max_b totals[b]` (`counterCTRDenominator`).
        let mut m = 0u32;
        let mut j = 0usize;
        while j < k {
            let c = counts[j];
            if c > m {
                m = c;
            }
            j += 1usize;
        }
        // PASS 3 — map each object onto its bucket's total and the shared denominator.
        let mut d = 0usize;
        while d < n {
            let b = bins[d] as usize;
            let t = counts[b];
            good[d] = t;
            total[d] = m;
            // `calc_ctr_online`'s denominator is a HARD `+1` (`calc_ctr.rs:77-80`), so it is
            // written as the exact integer one: `F::new(1.0)` would take an `f32` literal and
            // trip `float_literal_f32_fallback` on the generic parameter.
            value[d] = (F::cast_from(t) + pr) / (F::cast_from(m) + F::from_int(1));
            d += 1usize;
        }
    }
}

/// Device **BinarizedTargetMeanValue** (BTMV) CTR prefix (DCTR-12) — the read-before-increment
/// `(Sum, Count)` bucket history. A **sibling** of [`ordered_ctr_prefix_kernel`], deliberately not a
/// mode on it: BTMV's per-bucket state is a running FLOAT sum of binarized targets, not a class
/// count vector, so there is no shared accumulator to branch over (only the loop shape coincides).
///
/// # Upstream
/// `CalcOnlineCTRMean` (`online_ctr.cpp:437-501`) over `TCtrMeanHistory`
/// (`online_ctr.h:369-401`), with the added value `targetClass / targetBorderCount` where
/// `targetBorderCount = targetClassesCount - 1` (`online_ctr.cpp:762`). The CPU mirror is
/// `cb_train::ctr::online::online_mean_prefix` (`cb-train/src/ctr/online.rs:298-356`) composed
/// with `calc_ctr_online` (`calc_ctr.rs:77-80`) at `ctr_feature.rs:284-294`:
///
/// ```text
/// read (s, c) = hist[bucket]                       // BEFORE folding this document in
/// value[doc]  = (f64(s) + prior) / (c + 1)         // the ONLY widening point
/// hist[bucket].sum   = s + class[doc] / divisor    // an f32 add (TCtrMeanHistory::Add)
/// hist[bucket].count = c + 1
/// ```
///
/// # The `Array<f32>` accumulator is a PARITY CONTRACT, not an oversight
/// AGENTS.md mandates generics-float for new kernels; `sums` / `out_sum` are the **documented
/// exception** (`PLAN.md` §2.4): a buffer whose WIDTH is a parity contract stays concrete.
/// `TCtrMeanHistory::Sum` is a `float` upstream (`online_ctr.h:373`), the CPU side pins that with
/// `cb_train::ctr::online_test::btmv_sum_is_accumulated_in_f32_not_f64` (`online.rs:294`), and
/// widening the device history to `f64` "for precision" would silently break parity. Everything
/// else here is generic over `F: Float` — `prior` and `value` are the f64 CTR channel (WR-02).
///
/// # Why the width is only testable off the production path
/// At binclf `SIMPLE_CLASSES_COUNT == 2` (`cb-train/src/ctr/online.rs:52`) ⇒ `divisor == 1` ⇒ the
/// added values are exactly `{0.0, 1.0}`, every partial sum is an integer `<= n`, and f32 is exact
/// on integers to 2^24. **An f32 and an f64 accumulator are bit-identical for every reachable
/// binclf input** (`PLAN.md` §6 C-2), so no production-regime value comparison can discriminate the
/// width. `divisor` is therefore a RUNTIME scalar: the self-oracle drives a synthetic
/// `classes = 4` ⇒ `divisor = 3`, whose addends `{0, 1/3, 2/3, 1}` are inexact in f32 and do
/// separate the two widths (`btmv_f32_accumulation_width_is_load_bearing`). Production launches
/// `divisor = 1` and nothing else.
///
/// # Shape
/// Serial single-thread (unit 0) — the prefix is inherently sequential, exactly like
/// [`ordered_ctr_prefix_kernel`]. `sums`/`cnts` are the resident per-bucket history (length
/// `bucket_count`, **PRE-ZEROED by the host**; a reused non-zeroed buffer silently continues a
/// previous fit's history). Every index derives from a host-validated bound
/// ([`launch_btmv_ctr_resident`] range-checks `perm`, `bins` and `class`). `while` with an explicit
/// counter, no iterator adapters, no `if` EXPRESSION, no `-inf` literal (Pattern D).
#[cube(launch)]
fn btmv_ctr_prefix_kernel<F: Float>(
    perm: &Array<u32>,
    bins: &Array<u32>,
    class: &Array<u32>,
    prior: &Array<F>,
    divisor: &Array<f32>,
    sums: &mut Array<f32>,
    cnts: &mut Array<u32>,
    out_sum: &mut Array<f32>,
    out_cnt: &mut Array<u32>,
    value: &mut Array<F>,
) {
    if ABSOLUTE_POS == 0 {
        let pr = prior[0];
        let dv = divisor[0];
        let n = perm.len();
        let mut p = 0usize;
        while p < n {
            let doc = perm[p] as usize;
            let bucket = bins[doc] as usize;
            // READ the (Sum, Count) history BEFORE folding this document's own target in
            // (`online_ctr.cpp:168-185`, the no-leakage invariant).
            let s = sums[bucket];
            let c = cnts[bucket];
            out_sum[doc] = s;
            out_cnt[doc] = c;
            // The ONLY widening point: `calc_ctr_online(f64::from(s), c, prior)`. The hard `+1`
            // denominator is an integer, so `F::from_int(1)` — `F::new(1.0)` takes an f32 literal
            // and trips `float_literal_f32_fallback` under a generic `F`.
            value[doc] = (F::cast_from(s) + pr) / (F::cast_from(c) + F::from_int(1));
            // `TCtrMeanHistory::Add(targetClass / targetBorderCount)` — a single running **f32**
            // add. Keeping this in f32 is the parity contract documented above; do not widen it.
            sums[bucket] = s + (f32::cast_from(class[doc]) / dv);
            cnts[bucket] = c + 1u32;
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

/// The resident device **BinarizedTargetMeanValue** outputs for one categorical feature/projection
/// (DCTR-12): the three OBJECT-order buffers held on the client WITHOUT read-back (D-06 residency).
/// This is BTMV's counterpart to [`ResidentCtr`] — the numerator channel is a running FLOAT `sum`,
/// not an integer `good` count, which is exactly why it is a separate type.
///
/// `sum` is **f32-wide by parity contract** (`TCtrMeanHistory::Sum` is a `float` upstream,
/// `online_ctr.h:373`; the CPU pin is `cb_train::ctr::online_test::btmv_sum_is_accumulated_in_f32_not_f64`)
/// while `value` is the f64 CTR channel — the widening happens only at the value computation. See
/// [`btmv_ctr_prefix_kernel`].
pub(crate) struct ResidentCtrMean {
    /// Per-object `TCtrMeanHistory::Sum` read before the document's own target (**f32**, object
    /// order). The width is load-bearing — see the struct docs.
    #[allow(dead_code)] // consumed by the #[cfg(test)] ctr_device_test self-oracle (source/test separation)
    pub sum: Handle,
    /// Per-object `TCtrMeanHistory::Count` read before the document's own target (u32, object order).
    #[allow(dead_code)] // consumed by the #[cfg(test)] ctr_device_test self-oracle (source/test separation)
    pub count: Handle,
    /// Per-object online CTR value `(f64(sum) + prior) / (count + 1)` (f64, object order).
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
/// `cb_train::ctr::ECtrType::Counter` — see [`CTR_TYPE_BORDERS`] for the transcription rationale.
/// Unlike Borders/Buckets this is NOT a class-prefix type: it routes to
/// [`launch_counter_ctr_resident`], never to [`launch_ordered_ctr_resident`] (whose host guard
/// rejects it, DCTR-06 / T08).
pub(crate) const CTR_TYPE_COUNTER: i8 = 4;
/// `cb_train::ctr::ECtrType::BinarizedTargetMeanValue` — see [`CTR_TYPE_BORDERS`] for the
/// transcription rationale.
///
/// BTMV IS a permutation-dependent read-before-increment online prefix (unlike
/// [`CTR_TYPE_COUNTER`]), but its numerator is a running FLOAT `TCtrMeanHistory::Sum`
/// (`online_ctr.h:373`) rather than a class count, so it cannot be derived from one bucket's
/// `[N0, N1]`. It routes to [`launch_btmv_ctr_resident`] — which returns a
/// [`ResidentCtrMean`], not a [`ResidentCtr`] — and never to [`launch_ordered_ctr_resident`],
/// whose host guard rejects this discriminant (DCTR-06 / T08).
pub(crate) const CTR_TYPE_BTMV: i8 = 2;

/// The BTMV `targetBorderCount` divisor on the simple (binclf) CTR path:
/// `targetClassesCount - 1` with `SIMPLE_CLASSES_COUNT == 2`
/// (`cb-train/src/ctr/online.rs:52`; `online_ctr.cpp:762`). **Every production launch uses
/// this value and nothing else** — the runtime `divisor` parameter of
/// [`launch_btmv_ctr_resident`] exists so DCTR-12's f32-width detector can drive a synthetic
/// multiclass regime, which is the only regime in which the accumulator width is numerically
/// observable (PLAN §6 C-2).
pub(crate) const BTMV_DIVISOR_BINCLF: u32 = 1;

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

/// Accumulate the **Counter** CTR for one feature/projection ON device, resident (D-06), returning
/// the resident output handles WITHOUT reading them back (DCTR-09). `bins` is the per-object bucket
/// (object order); `bucket_count` the distinct-bucket count (`max(bins) + 1`, host-validated);
/// `prior` the additive CTR prior numerator.
///
/// # No permutation, no class, no target border — by construction
/// Counter is **permutation independent** (`IsPermutationDependentCtrType(Counter) == false`,
/// `ctr_type.cpp:43-56`) and is not a class-prefix statistic at all: its numerator is the whole-set
/// bucket total and its denominator the MAX bucket total, neither derivable from one bucket's class
/// counts. That is why this is a separate entry point rather than a widening of
/// [`launch_ordered_ctr_resident`], whose host guard **rejects** `ctr_type == Counter` precisely so
/// a Counter column can never silently receive the Borders numerator (T08).
///
/// The emitted `(good, total, value)` triple has the ordered path's shape — `total` is the
/// CONSTANT denominator repeated per object, mirroring `denoms = vec![denominator; n]`
/// (`ctr_feature.rs:304`) — so [`binarize_ctr_column_resident`] and the existing per-column border
/// table apply unchanged (C-7).
///
/// # Errors
/// [`CbError::OutOfRange`] on the wgpu f64 path (WR-02) or if any bin is `>= bucket_count`;
/// [`CbError::LengthMismatch`] if `bins` disagrees with `n`.
#[cfg_attr(feature = "wgpu", allow(unused_variables))]
pub(crate) fn launch_counter_ctr_resident(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    bins: &[u32],
    prior: f64,
    bucket_count: usize,
    n: usize,
) -> CbResult<ResidentCtr> {
    if bins.len() != n {
        return Err(CbError::LengthMismatch {
            column: "ctr counter bins".to_owned(),
            expected: n,
            actual: bins.len(),
        });
    }
    // WR-02: guard the bin range host-side before dispatch. The kernel indexes `counts[bucket]`
    // (length `bucket_count`), so a bin >= bucket_count is an out-of-bounds device access (UB).
    // Same guard as `launch_ordered_ctr_resident`, minus the class check (Counter reads no class).
    if let Some(&bad) = bins.iter().find(|&&b| (b as usize) >= bucket_count) {
        return Err(CbError::OutOfRange(format!(
            "ctr bin value {bad} >= bucket_count ({bucket_count})"
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
        // PRE-ZEROED per-bucket tally scratch (one u32 per bucket); at least length 1 so an
        // all-zero-bin degenerate feature still has a valid bucket-0 slot. A reused non-zeroed
        // buffer would silently double the tally, so this allocation is never shared.
        let scratch_len = bucket_count.max(1);
        let counts_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; scratch_len]));
        let bins_h = client.create(cubecl::bytes::Bytes::from_elems(bins.to_vec()));
        let prior_h = client.create(cubecl::bytes::Bytes::from_elems(vec![prior]));
        let good_h = client.empty(n * std::mem::size_of::<u32>());
        let total_h = client.empty(n * std::mem::size_of::<u32>());
        let value_h = client.empty(n * std::mem::size_of::<f64>());

        // Serial single-thread launch (unit 0 runs the three passes); one cube, one unit.
        let count = CubeCount::Static(1, 1, 1);
        let dim = CubeDim { x: 1, y: 1, z: 1 };
        // The kernel is generic over `F: Float`; the launch pins `f64` because the CTR seam is
        // f64 end to end (WR-02) and `binarize_ctr_column_resident` binarizes at f64.
        counter_ctr_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(bins_h, n) },
            unsafe { ArrayArg::from_raw_parts(prior_h, 1) },
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

/// Accumulate the **BinarizedTargetMeanValue** CTR for one feature/projection ON device, resident
/// across the permutation (D-06), returning the resident output handles WITHOUT reading them back
/// (DCTR-12). `bins` is the per-object bucket (object order); `perm` the learn permutation;
/// `class` the per-object target class; `divisor` the upstream `targetBorderCount`
/// (`targetClassesCount - 1`, `online_ctr.cpp:762` — **`1` in every production launch**, since the
/// simple CTR path is binclf); `bucket_count` the distinct-bucket count (`max(bins) + 1`).
///
/// # Not routed through [`launch_ordered_ctr_resident`] — by construction
/// BTMV is not a class-prefix statistic: its numerator is a running float `Sum` of binarized
/// targets, not a class count, so it cannot be derived from one bucket's `[N0, N1]`. The ordered
/// launcher's host guard **rejects** `ctr_type == BinarizedTargetMeanValue` precisely so a BTMV
/// column can never silently receive the Borders numerator (T08), and that guard stays. This
/// entry point takes no `ctr_type` and no `target_border_idx` at all.
///
/// The emitted `value` channel is f64 and feeds [`binarize_ctr_column_resident`] and the existing
/// per-column border table UNCHANGED (C-7: BTMV's CPU f32 quantizer was measured bit-identical to
/// the f64 border table for every prior in `[0, 1]` — research spike Q2, 4,504,501 pairs/prior).
/// No per-type border handling exists anywhere.
///
/// # Errors
/// [`CbError::OutOfRange`] on the wgpu f64 path (WR-02), if `divisor == 0` (a division by zero on
/// device), if any `perm` entry is `>= n`, if any bin is `>= bucket_count`, or if any target class
/// exceeds `divisor` (`targetClass <= targetClassesCount - 1`);
/// [`CbError::LengthMismatch`] if `bins`/`class` disagree with `perm`.
#[cfg_attr(feature = "wgpu", allow(unused_variables))]
pub(crate) fn launch_btmv_ctr_resident(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    perm: &[u32],
    bins: &[u32],
    class: &[u32],
    prior: f64,
    divisor: u32,
    bucket_count: usize,
    n: usize,
) -> CbResult<ResidentCtrMean> {
    // `divisor = targetClassesCount - 1` floored at 1 by the CPU (`online.rs:321`); a zero would
    // be a device division by zero, so it is refused rather than silently repaired.
    if divisor == 0 {
        return Err(CbError::OutOfRange(
            "ctr BTMV divisor (targetBorderCount = targetClassesCount - 1) must be >= 1".to_owned(),
        ));
    }
    if bins.len() != n || class.len() != n || perm.len() != n {
        return Err(CbError::LengthMismatch {
            column: "ctr btmv inputs".to_owned(),
            expected: n,
            actual: bins.len().min(class.len()).min(perm.len()),
        });
    }
    // WR-02: guard every value the kernel turns into an index, host-side, before dispatch. The
    // kernel indexes `bins[perm[p]]`, `class[perm[p]]` and `sums[bucket]`/`cnts[bucket]`
    // (length `bucket_count`), so an out-of-range permutation entry or bin is an out-of-bounds
    // device access (UB).
    if let Some(&bad) = perm.iter().find(|&&d| (d as usize) >= n) {
        return Err(CbError::OutOfRange(format!(
            "ctr btmv permutation index {bad} >= n ({n})"
        )));
    }
    if let Some(&bad) = bins.iter().find(|&&b| (b as usize) >= bucket_count) {
        return Err(CbError::OutOfRange(format!(
            "ctr bin value {bad} >= bucket_count ({bucket_count})"
        )));
    }
    // `targetClass` ranges over `0 ..= targetClassesCount - 1 == divisor`, so the added value
    // `class / divisor` lies in `[0, 1]`. A larger class is a caller bug, not a value to clamp.
    if let Some(&bad) = class.iter().find(|&&c| c > divisor) {
        return Err(CbError::OutOfRange(format!(
            "ctr btmv target class {bad} > targetBorderCount ({divisor})"
        )));
    }
    if n == 0 {
        return Ok(ResidentCtrMean {
            sum: client.empty(0),
            count: client.empty(0),
            value: client.empty(0),
        });
    }

    #[cfg(feature = "wgpu")]
    {
        return Err(wgpu_reject());
    }

    #[cfg(not(feature = "wgpu"))]
    {
        // PRE-ZEROED per-bucket (Sum, Count) history; at least length 1 so an all-zero-bin
        // degenerate feature still has a valid bucket-0 slot. Freshly allocated per call and never
        // shared — a reused non-zeroed buffer would silently continue a previous fit's history.
        let scratch_len = bucket_count.max(1);
        let sums_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0f32; scratch_len]));
        let cnts_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; scratch_len]));
        let perm_h = client.create(cubecl::bytes::Bytes::from_elems(perm.to_vec()));
        let bins_h = client.create(cubecl::bytes::Bytes::from_elems(bins.to_vec()));
        let class_h = client.create(cubecl::bytes::Bytes::from_elems(class.to_vec()));
        let prior_h = client.create(cubecl::bytes::Bytes::from_elems(vec![prior]));
        // The divisor crosses the seam as f32 because the addend `class / divisor` is computed in
        // the f32 accumulator's own width (`class as f32 / divisor` on the CPU, `online.rs:321`
        // and `:351`). It is an exact small integer, so the cast is lossless.
        let divisor_h = client.create(cubecl::bytes::Bytes::from_elems(vec![divisor as f32]));
        // The per-document sum output is f32-WIDE BY PARITY CONTRACT (see `ResidentCtrMean`).
        let out_sum_h = client.empty(n * std::mem::size_of::<f32>());
        let out_cnt_h = client.empty(n * std::mem::size_of::<u32>());
        let value_h = client.empty(n * std::mem::size_of::<f64>());

        // Serial single-thread launch (unit 0 loops the permutation); one cube, one unit.
        let count = CubeCount::Static(1, 1, 1);
        let dim = CubeDim { x: 1, y: 1, z: 1 };
        // The kernel is generic over `F: Float` for the prior/value channel; the launch pins `f64`
        // because the CTR seam is f64 end to end (WR-02) and `binarize_ctr_column_resident`
        // binarizes at f64. The `sums`/`out_sum` accumulator stays `f32` inside the kernel.
        btmv_ctr_prefix_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(perm_h, n) },
            unsafe { ArrayArg::from_raw_parts(bins_h, n) },
            unsafe { ArrayArg::from_raw_parts(class_h, n) },
            unsafe { ArrayArg::from_raw_parts(prior_h, 1) },
            unsafe { ArrayArg::from_raw_parts(divisor_h, 1) },
            unsafe { ArrayArg::from_raw_parts(sums_h, scratch_len) },
            unsafe { ArrayArg::from_raw_parts(cnts_h, scratch_len) },
            unsafe { ArrayArg::from_raw_parts(out_sum_h.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(out_cnt_h.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(value_h.clone(), n) },
        );
        Ok(ResidentCtrMean {
            sum: out_sum_h,
            count: out_cnt_h,
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

/// DCTR-09 host-readback wrapper over the device **Counter** CTR (the self-oracle seam):
/// accumulate the resident Counter column, then read the three buffers back to host `Vec`s (OBJECT
/// order). This is NOT the residency path (that keeps the handles on-device); it is the
/// device-vs-CPU oracle exerciser. A read-back failure surfaces [`CbError::Degenerate`] (WR-05),
/// never a silent zero buffer.
///
/// Returns `(good, total, value)`; `good[obj]` is the whole-set tally of `obj`'s bucket and
/// `total[obj]` the CONSTANT max-bucket denominator, both widened to `i64` to match the CPU
/// reference's integer schema.
///
/// # Errors
/// As [`launch_counter_ctr_resident`], plus [`CbError::Degenerate`] on a read-back failure.
#[allow(dead_code)] // consumed by the #[cfg(test)] ctr_device_test self-oracle (source/test separation)
pub(crate) fn compute_counter_ctr_host(
    bins: &[u32],
    prior: f64,
    bucket_count: usize,
) -> CbResult<(Vec<i64>, Vec<i64>, Vec<f64>)> {
    let n = bins.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    let res = launch_counter_ctr_resident(&client, bins, prior, bucket_count, n)?;
    let good_b = client.read_one(res.good).map_err(|e| {
        CbError::Degenerate(format!("CubeCL Counter CTR good read-back failed: {e:?}"))
    })?;
    let total_b = client.read_one(res.total).map_err(|e| {
        CbError::Degenerate(format!("CubeCL Counter CTR total read-back failed: {e:?}"))
    })?;
    let value_b = client.read_one(res.value).map_err(|e| {
        CbError::Degenerate(format!("CubeCL Counter CTR value read-back failed: {e:?}"))
    })?;
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

/// DCTR-09 host-readback wrapper over the device Counter CTR→cindex binarize JOIN (the self-oracle
/// seam): accumulate the Counter column on device, binarize its values against the per-column
/// border table on device, then read that bin column back. Returns the per-object bin indices
/// (`> bin` convention). A read-back failure surfaces [`CbError::Degenerate`] (WR-05).
///
/// # Errors
/// As [`launch_counter_ctr_resident`], plus [`CbError::Degenerate`] on a read-back failure.
#[cfg(not(feature = "wgpu"))]
#[allow(dead_code)] // consumed by the #[cfg(test)] ctr_device_test self-oracle (source/test separation)
pub(crate) fn binarize_counter_column_host(
    bins: &[u32],
    prior: f64,
    bucket_count: usize,
    borders: &[f64],
) -> CbResult<Vec<u32>> {
    let n = bins.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    let res = launch_counter_ctr_resident(&client, bins, prior, bucket_count, n)?;
    let bins_h = binarize_ctr_column_resident(&client, &res.value, borders, n)?;
    let bytes = client.read_one(bins_h).map_err(|e| {
        CbError::Degenerate(format!("CubeCL Counter CTR cindex read-back failed: {e:?}"))
    })?;
    Ok(bytemuck::cast_slice::<u8, u32>(&bytes).to_vec())
}

/// DCTR-12 host-readback wrapper over the device **BTMV** CTR (the self-oracle seam): accumulate
/// the resident `(Sum, Count)` prefix, then read the three buffers back to host `Vec`s (OBJECT
/// order). This is NOT the residency path (that keeps the handles on-device); it is the
/// device-vs-CPU oracle exerciser. A read-back failure surfaces [`CbError::Degenerate`] (WR-05),
/// never a silent zero buffer.
///
/// Returns `(sum, count, value)`. **`sum` is `Vec<f32>`, not `Vec<f64>`** — widening it here would
/// hide exactly the accumulator width DCTR-12 exists to pin (`ResidentCtrMean`); `count` is widened
/// to `i64` to match the CPU reference's `TCtrMeanHistory::Count` schema.
///
/// # Errors
/// As [`launch_btmv_ctr_resident`], plus [`CbError::Degenerate`] on a read-back failure.
#[allow(dead_code)] // consumed by the #[cfg(test)] ctr_device_test self-oracle (source/test separation)
pub(crate) fn compute_btmv_ctr_host(
    perm: &[u32],
    bins: &[u32],
    class: &[u32],
    prior: f64,
    divisor: u32,
    bucket_count: usize,
) -> CbResult<(Vec<f32>, Vec<i64>, Vec<f64>)> {
    let n = perm.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    let res = launch_btmv_ctr_resident(
        &client,
        perm,
        bins,
        class,
        prior,
        divisor,
        bucket_count,
        n,
    )?;
    let sum_b = client
        .read_one(res.sum)
        .map_err(|e| CbError::Degenerate(format!("CubeCL BTMV CTR sum read-back failed: {e:?}")))?;
    let count_b = client.read_one(res.count).map_err(|e| {
        CbError::Degenerate(format!("CubeCL BTMV CTR count read-back failed: {e:?}"))
    })?;
    let value_b = client.read_one(res.value).map_err(|e| {
        CbError::Degenerate(format!("CubeCL BTMV CTR value read-back failed: {e:?}"))
    })?;
    let sum = bytemuck::cast_slice::<u8, f32>(&sum_b).to_vec();
    let count = bytemuck::cast_slice::<u8, u32>(&count_b)
        .iter()
        .map(|&c| i64::from(c))
        .collect();
    let value = bytemuck::cast_slice::<u8, f64>(&value_b).to_vec();
    Ok((sum, count, value))
}

/// DCTR-12 (C-2.1) read-back of the resident BTMV `sum` buffer as RAW BYTES, so the output-width
/// pin can assert the on-device element width without the test constructing a `ComputeClient`
/// (source/test separation). Deliberately does NOT interpret the bytes.
///
/// **Scope**: this observes the per-document OUTPUT buffer, not the per-bucket accumulator — an
/// `Array<f64>` bucket history feeding an `f32` output would produce the same byte length. The
/// numeric width proof is `btmv_f32_accumulation_width_is_load_bearing`.
///
/// # Errors
/// As [`launch_btmv_ctr_resident`], plus [`CbError::Degenerate`] on a read-back failure.
#[allow(dead_code)] // consumed by the #[cfg(test)] ctr_device_test self-oracle (source/test separation)
pub(crate) fn read_btmv_sum_bytes(
    perm: &[u32],
    bins: &[u32],
    class: &[u32],
    prior: f64,
    divisor: u32,
    bucket_count: usize,
) -> CbResult<Vec<u8>> {
    let n = perm.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    let res = launch_btmv_ctr_resident(
        &client,
        perm,
        bins,
        class,
        prior,
        divisor,
        bucket_count,
        n,
    )?;
    let sum_b = client
        .read_one(res.sum)
        .map_err(|e| CbError::Degenerate(format!("CubeCL BTMV CTR sum read-back failed: {e:?}")))?;
    Ok(sum_b.to_vec())
}

/// DCTR-12 host-readback wrapper over the device BTMV CTR→cindex binarize JOIN (the self-oracle
/// seam): accumulate the BTMV column on device, binarize its values against the per-column border
/// table on device, then read that bin column back. Returns the per-object bin indices
/// (`> bin` convention). A read-back failure surfaces [`CbError::Degenerate`] (WR-05).
///
/// # Errors
/// As [`launch_btmv_ctr_resident`], plus [`CbError::Degenerate`] on a read-back failure.
#[cfg(not(feature = "wgpu"))]
#[allow(dead_code)] // consumed by the #[cfg(test)] ctr_device_test self-oracle (source/test separation)
pub(crate) fn binarize_btmv_column_host(
    perm: &[u32],
    bins: &[u32],
    class: &[u32],
    prior: f64,
    divisor: u32,
    bucket_count: usize,
    borders: &[f64],
) -> CbResult<Vec<u32>> {
    let n = perm.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    let res = launch_btmv_ctr_resident(
        &client,
        perm,
        bins,
        class,
        prior,
        divisor,
        bucket_count,
        n,
    )?;
    let bins_h = binarize_ctr_column_resident(&client, &res.value, borders, n)?;
    let bytes = client.read_one(bins_h).map_err(|e| {
        CbError::Degenerate(format!("CubeCL BTMV CTR cindex read-back failed: {e:?}"))
    })?;
    Ok(bytemuck::cast_slice::<u8, u32>(&bytes).to_vec())
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
