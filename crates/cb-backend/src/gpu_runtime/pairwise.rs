//! IN-03: Phase 7.4/7.5 device-resident PAIRWISE seam (4-channel histogram fill,
//! scan/update, make-derivatives, split score, and the pairwise grow-loop driver),
//! mechanically relocated out of the oversized `gpu_runtime.rs` with ZERO logic changes.
//! All public items are re-exported from `gpu_runtime` (`pub use pairwise::*`) so every
//! existing `crate::gpu_runtime::X` path still resolves. `use super::*` brings in the
//! shared parent items this seam consumes (`CUBE_DIM`, `BestSplit`, `GrownTree`,
//! `grow_oblivious_tree`, the partition / scan-update / read-back helpers, the
//! cubecl/cb-core imports).
#![allow(unused_imports)]
use super::*;

// ===========================================================================
// Phase 7.4 — the device-resident 4-channel WEIGHT-ONLY pairwise histogram FILL seam
// (GPU-01 histogram slice). The general one-byte non-binary
// `ComputePairwiseHistogramOneByte{5,6,7}Bits` analog: pair_i/pair_j (= `uint2* pairs`)
// + per-pair weight + cindex in -> 4-channel `binSums` device handle out, NO host
// round-trip (D-7.4-03). DISTINCT from the 7.2/7.3 der1/der2/pointwise seam: NO der1
// input, and the histogram is 4-channel weight-only (histId in {0,1,2,3}), NEVER the
// 7.3 2-channel der1/weight layout. The FROZEN 4-channel layout + the
// pair-handles-in -> binSums-handle-out seam this defines are reused UNCHANGED by Plans
// B/C/D/E and the 7.5 pairwise score/split consumer.
// ===========================================================================

/// The number of channels in a pairwise-histogram cell — the `* 4` in the FROZEN
/// pairwise `binSums` index (`histId in {0,1,2,3}`). Naming the `4` removes the magic
/// literal from the layout arithmetic (it is the channel count, NOT a stride/warp
/// literal — D-09). **This is 4, NEVER 2** (the 7.3 pointwise channel count) — Pitfall 2.
pub(crate) const PAIR_HIST_CHANNELS: usize = 4;

/// Compute the FROZEN pairwise `binSums` buffer length for the single-tree fill:
/// `histLineSize = PAIR_HIST_CHANNELS * totalBinFeatures` collapses, for `part = fold =
/// 0` and a single feature group with `FirstFoldIndex = 0`, to
/// `n_features * n_bins * PAIR_HIST_CHANNELS` floats. See [`launch_pairwise_hist_handle`]
/// for the full index formula.
#[inline]
fn pair_hist_binsums_len(n_bins: usize, n_features: usize) -> usize {
    n_features * n_bins * PAIR_HIST_CHANNELS
}

/// Overflow-checked companion to [`pair_hist_binsums_len`]. Returns `None` if
/// `n_features * n_bins * PAIR_HIST_CHANNELS` overflows `usize`, so the host seam can
/// reject a degenerate dimension with a typed range error instead of wrapping silently.
#[inline]
fn pair_hist_binsums_len_checked(n_bins: usize, n_features: usize) -> Option<usize> {
    n_features
        .checked_mul(n_bins)
        .and_then(|v| v.checked_mul(PAIR_HIST_CHANNELS))
}

/// Fill the device-resident 4-channel WEIGHT-ONLY pairwise histogram (5/6/7-bit
/// non-binary) on the compile-time [`SelectedRuntime`] and return `binSums` as a DEVICE
/// BUFFER HANDLE — WITHOUT reading it back (SC-3 / D-7.4-03 / Pitfall 2/5). This is the
/// load-bearing hand-off seam the 7.5 pairwise score/split path plugs into: the returned
/// histogram handle stays on-device, consumed with no host round-trip.
///
/// # FROZEN pairwise `binSums` device-handle layout (D-7.4-03 / Pitfall 2)
///
/// The histogram is a flat `[partCount * histLineSize]`-floats buffer with the 4
/// weight-only channels (`histId in {0,1,2,3}`) interleaved per (feature, bin):
///
/// ```text
/// histLineSize = PAIR_HIST_CHANNELS * totalBinFeatures      (totalBinFeatures = n_features * n_bins)
/// index(feature, bin, histId) = (feature * n_bins + bin) * PAIR_HIST_CHANNELS + histId
/// ```
///
/// mirroring upstream `pairwise_hist_one_byte_5bit.cuh:255-256` (`4 * (maxFoldCount * f
/// + fold) + histId`) + `split_properties_helpers.cuh`'s `Compare` predicate. This
/// phase delivers the SINGLE-TREE fill: `partCount = foldCount = 1`, one feature group
/// with `FirstFoldIndex = 0`, so the index collapses to the kernel's write index and
/// the buffer length is [`pair_hist_binsums_len`]. The `fullPass = false` multi-part
/// offset (`ShiftPartAndBinSumsPtr`'s else-branch) and the `BuildBinaryFeatureHistograms`
/// transform are 7.5 FORWARD DEPENDENCIES (RESEARCH Open Q3), NOT filled here —
/// documented, not silently cut. This raw 4-channel fill is what 7.5 reduces. This
/// layout is FROZEN across Plans B/C/D/E and the 7.5 seam. **Anti-pattern: any `* 2`
/// here silently breaks the 7.5 seam (Pitfall 2) — the buffer is ALWAYS `* 4`.**
///
/// # Inputs (D-7.4-03)
///
/// `pair_i`/`pair_j` (two parallel `u32` arrays = upstream `uint2* pairs`, length
/// `n_pairs`, OBJECT ids), `pair_weight` (the ONLY per-pair value — NO der1, length
/// `n_pairs`); `cindex` (length `n_features * n_objects`, feature-major:
/// `cindex[feature * n_objects + obj]` is object `obj`'s quantized bin for `feature` —
/// stride over OBJECTS, NOT pairs, Pitfall 3); `n_objects` the object count; `n_bins`
/// is `1 << bits`; `n_features` the feature-group width; `bits` in {5,6,7}; `one_hot`
/// threaded for Plan E.
///
/// # Atomic merge (D-03 / Pitfall 1)
///
/// The cross-thread merge into `binSums` is ALWAYS the in-kernel `Atomic<F>::fetch_add`.
/// The only capability adaptation is the channel float type: f64 on rocm/cuda/cpu, f32
/// on wgpu (WGSL has no f64 atomics — RESEARCH A1).
///
/// Empty input (`n_pairs == 0` or `n_features == 0` or `n_bins == 0`) short-circuits to
/// a zero-length handle with NO launch and NO read-back (Pitfall 5). Mismatched
/// pair_i/pair_j/pair_weight/cindex lengths surface [`CbError::LengthMismatch`] BEFORE
/// launch (T-07.4-01); out-of-range pair ids / bins surface [`CbError::OutOfRange`]
/// (T-07.4-02). No `unwrap`/`expect`/`panic`/indexing in this production helper.
#[allow(clippy::too_many_arguments)]
pub fn launch_pairwise_hist_handle(
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
    bits: u32,
    one_hot: bool,
) -> CbResult<Handle> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_pairwise_hist_into(
        &client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features, bits, one_hot,
    )
}

/// The ONE pairwise-histogram launch geometry (IN-02 — one place, not duplicated per
/// public entry point). Transfers `pair_i`/`pair_j`/`pair_weight`/`cindex` onto
/// `client`, zero-initialises the 4-channel `binSums` buffer, launches the non-binary
/// pairwise fill kernel, and returns the `binSums` Handle WITHOUT reading it back. The
/// caller owns the `client` lifecycle so a read-back (the self-oracle wrapper) uses the
/// SAME client that allocated the handle — a CubeCL Handle is bound to its originating
/// client.
#[allow(clippy::too_many_arguments)]
fn launch_pairwise_hist_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
    bits: u32,
    one_hot: bool,
) -> CbResult<Handle> {
    let n_pairs = pair_i.len();

    // Shape guards (T-07.4-01): the kernel reads pair_i[p]/pair_j[p]/pair_weight[p] for
    // each pair, and cindex[feature * n_objects + obj] for each feature. A mismatch would
    // read out of bounds on the device — surface a typed error BEFORE launch (no panic).
    if pair_j.len() != n_pairs {
        return Err(CbError::LengthMismatch {
            column: "pair_j".to_owned(),
            expected: n_pairs,
            actual: pair_j.len(),
        });
    }
    if pair_weight.len() != n_pairs {
        return Err(CbError::LengthMismatch {
            column: "pair_weight".to_owned(),
            expected: n_pairs,
            actual: pair_weight.len(),
        });
    }

    // Overflow guards FIRST — before any unchecked product is formed. The cindex stride
    // `n_features * n_objects` and the binSums length `n_features * n_bins * 4` are
    // products of unbounded caller-supplied dimensions. Reject a degenerate dimension
    // with a typed range error BEFORE the product is ever computed unchecked, then REUSE
    // the checked product for the length guard below (never re-multiplying).
    let cindex_stride = n_features.checked_mul(n_objects).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_objects ({n_objects}) overflows usize (cindex stride)"
        ))
    })?;
    if pair_hist_binsums_len_checked(n_bins, n_features).is_none() {
        return Err(CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) * {PAIR_HIST_CHANNELS} overflows usize (binSums length)"
        )));
    }

    // Now the cindex length guard uses the already-checked product, never re-multiplying.
    if cindex.len() != cindex_stride {
        return Err(CbError::LengthMismatch {
            column: "cindex".to_owned(),
            expected: cindex_stride,
            actual: cindex.len(),
        });
    }

    // Value-range guards (T-07.4-02): the length guards above bound only buffer
    // *positions*; the *values* inside pair_i/pair_j (object ids) and cindex (bins) drive
    // unchecked device array indices. Validate them HOST-SIDE so a malformed object id or
    // bin surfaces a typed `CbError::OutOfRange` rather than an out-of-bounds device
    // read/store (UB). BOTH pair endpoints index cindex (Pitfall 3).
    if let Some(&bad) = pair_i.iter().find(|&&ix| (ix as usize) >= n_objects) {
        return Err(CbError::OutOfRange(format!(
            "pair_i value {bad} >= n_objects ({n_objects}); object id would read cindex out of bounds"
        )));
    }
    if let Some(&bad) = pair_j.iter().find(|&&ix| (ix as usize) >= n_objects) {
        return Err(CbError::OutOfRange(format!(
            "pair_j value {bad} >= n_objects ({n_objects}); object id would read cindex out of bounds"
        )));
    }
    // Every cindex bin must fit the dispatched line size (`n_bins`); a value >= n_bins
    // would write `bin_sums` out of bounds (the non-binary kernel does not mask).
    if let Some(&bad) = cindex.iter().find(|&&b| (b as usize) >= n_bins) {
        return Err(CbError::OutOfRange(format!(
            "cindex bin value {bad} >= n_bins ({n_bins}); would write bin_sums out of bounds"
        )));
    }

    // Empty fill: hand back a zero-length handle (no launch, no read-back — Pitfall 5).
    // 7.5 still receives a valid (empty) histogram handle.
    if n_pairs == 0 || n_features == 0 || n_bins == 0 {
        return Ok(client.empty(0));
    }

    // Validate the bit-width belongs to the one-byte non-binary family (5/6/7-bit). The
    // border count `n_bins` MUST equal `1 << bits`; anything else is not this family.
    if bits < 5 || bits > 7 || n_bins != (1usize << bits) {
        return Err(CbError::Degenerate(format!(
            "pairwise_hist non-binary fill expects bits in {{5,6,7}} with n_bins == 1 << bits, \
             got bits={bits} n_bins={n_bins}"
        )));
    }

    // Launch geometry: enough cubes to cover `n_pairs` (the grid-stride loop handles any
    // surplus via the total-thread-count stride).
    let num_cubes = n_pairs.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    let bin_sums_len = pair_hist_binsums_len(n_bins, n_features);
    let cindex_handle = client.create(cubecl::bytes::Bytes::from_elems(cindex.to_vec()));
    let pair_i_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_i.to_vec()));
    let pair_j_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_j.to_vec()));

    // Channel float-type dispatch (RESEARCH A1 / Pitfall 1): the in-kernel atomic merge
    // needs `Atomic<F>::fetch_add` device-side; HIP (rocm) and CUDA support / emulate the
    // f64 atomic add, so the channel is f64 there (D-03), while wgpu's WGSL has NO f64
    // atomics, so the wgpu arm uses an f32 channel (read back and UPCAST to f64). The
    // buffer length (the FROZEN 4-channel layout) is channel-type independent. NO
    // read-back here (SC-3 / Pitfall 5). `from_raw_parts` consumes each input handle; the
    // output is cloned so the original stays returnable on-device.
    #[cfg(feature = "wgpu")]
    {
        let weight_f32: Vec<f32> = pair_weight.iter().map(|&v| v as f32).collect();
        let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(weight_f32));
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; bin_sums_len]));
        pairwise_hist_nonbinary_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(pair_i_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(pair_j_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(weight_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
            n_features as u32,
            n_objects as u32,
            bits,
            one_hot,
        );
        Ok(h)
    }

    #[cfg(not(feature = "wgpu"))]
    {
        let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_weight.to_vec()));
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; bin_sums_len]));
        pairwise_hist_nonbinary_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(pair_i_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(pair_j_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(weight_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
            n_features as u32,
            n_objects as u32,
            bits,
            one_hot,
        );
        Ok(h)
    }
}

/// Read a pairwise `binSums` device handle back to a host `Vec<f64>`, transparently
/// UPCASTING the f32 channel on the wgpu arm (RESEARCH A1) and reading the f64 channel
/// directly elsewhere. Centralizes the channel-type read so both the readback wrapper
/// and the `kernels::pairwise_hist` oracle observe the SAME f64 layout regardless of
/// backend. A read-back failure surfaces [`CbError::Degenerate`].
fn read_pair_binsums_f64(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    handle: Handle,
) -> CbResult<Vec<f64>> {
    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("CubeCL device read-back failed: {e:?}")))?;
    #[cfg(feature = "wgpu")]
    {
        Ok(bytemuck::cast_slice::<u8, f32>(&bytes)
            .iter()
            .map(|&v| f64::from(v))
            .collect())
    }
    #[cfg(not(feature = "wgpu"))]
    {
        Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
    }
}

/// Host-readback wrapper over the pairwise-histogram fill: launch the non-binary fill
/// device-resident, then read the 4-channel `binSums` handle back to a host `Vec<f64>`.
/// This is the seam the all-backend self-oracle exercises (it compares the device
/// histogram to the ordered host reference); it is NOT the histogram hand-off path (that
/// is [`launch_pairwise_hist_handle`], which never reads back).
///
/// The launch geometry lives in ONE place ([`launch_pairwise_hist_into`]); this wrapper
/// constructs the client ONCE and uses that SAME client for both the launch and the
/// read-back, so the handle is read by the client that allocated it. A device read-back
/// failure surfaces as [`CbError::Degenerate`], never a silent all-zero buffer
/// masquerading as a valid histogram. Empty input returns an empty `Vec` (no launch).
#[allow(clippy::too_many_arguments)]
pub fn launch_pairwise_hist(
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
    bits: u32,
    one_hot: bool,
) -> CbResult<Vec<f64>> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

    if pair_i.is_empty() || n_features == 0 || n_bins == 0 {
        return Ok(Vec::new());
    }

    let handle = launch_pairwise_hist_into(
        &client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features, bits, one_hot,
    )?;

    read_pair_binsums_f64(&client, handle)
}

/// Launch the 8-bit-atomics pairwise histogram fill device-resident and return the
/// 4-channel `binSums` Handle WITHOUT reading it back (the SC-3 / D-7.4-03 hand-off path).
/// This is the SEPARATE launch arm for the structurally DISTINCT 8-bit-atomics family
/// (D-7.4-02 — upstream `pairwise_hist_one_byte_8bit_atomics.cuh`): the 256-bin x
/// 4-channel line exceeds the per-block shared-memory budget, so the kernel accumulates
/// via TRUE GLOBAL atomics. It reuses the Plan A FROZEN 4-channel layout
/// ([`pair_hist_binsums_len`], index `(feature * n_bins + bin) * 4 + histId`), the guard
/// block, the backend-dispatched channel, the empty short-circuit, and the no-readback
/// seam UNCHANGED — the only differences are `n_bins = 256` (validated below) and the
/// dispatched kernel symbol.
///
/// `n_bins` MUST be `256` (the 8-bit line size); anything else is rejected with
/// [`CbError::Degenerate`]. Inputs, guards, and the empty/typed-error semantics match
/// [`launch_pairwise_hist_handle`] (see its docs). No `unwrap`/`expect`/`panic`/indexing
/// in this production helper.
#[allow(clippy::too_many_arguments)]
pub fn launch_pairwise_hist_8bit_handle(
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
    one_hot: bool,
) -> CbResult<Handle> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_pairwise_hist_8bit_into(
        &client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features, one_hot,
    )
}

/// The ONE 8-bit-atomics pairwise launch geometry (IN-02 — one place). Clones
/// [`launch_pairwise_hist_into`] with `n_bins` pinned to `256` and dispatches
/// [`pairwise_hist_8bit_atomics_kernel`] (direct global atomics, no shared-mem
/// pre-reduce). Reuses the Plan A guard block, the 4-channel layout, the
/// backend-dispatched channel, and the no-readback seam unchanged. The caller owns the
/// `client` lifecycle so a read-back uses the SAME client that allocated the handle.
#[allow(clippy::too_many_arguments)]
fn launch_pairwise_hist_8bit_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
    one_hot: bool,
) -> CbResult<Handle> {
    let n_pairs = pair_i.len();

    // Shape guards (T-07.4-07): identical to the non-binary arm — a mismatch would read
    // out of bounds on the device. Surface a typed error BEFORE launch (no panic).
    if pair_j.len() != n_pairs {
        return Err(CbError::LengthMismatch {
            column: "pair_j".to_owned(),
            expected: n_pairs,
            actual: pair_j.len(),
        });
    }
    if pair_weight.len() != n_pairs {
        return Err(CbError::LengthMismatch {
            column: "pair_weight".to_owned(),
            expected: n_pairs,
            actual: pair_weight.len(),
        });
    }

    // Overflow guards FIRST (the 256-bin line makes `n_features * 256 * 4` overflow
    // checking load-bearing — T-07.4-07). Reuse the checked cindex stride for the length
    // guard below, never re-multiplying.
    let cindex_stride = n_features.checked_mul(n_objects).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_objects ({n_objects}) overflows usize (cindex stride)"
        ))
    })?;
    if pair_hist_binsums_len_checked(n_bins, n_features).is_none() {
        return Err(CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) * {PAIR_HIST_CHANNELS} overflows usize (binSums length)"
        )));
    }

    if cindex.len() != cindex_stride {
        return Err(CbError::LengthMismatch {
            column: "cindex".to_owned(),
            expected: cindex_stride,
            actual: cindex.len(),
        });
    }

    // Value-range guards (T-07.4-07): BOTH pair endpoints index cindex (Pitfall 3); every
    // cindex bin must fit the 256-bin line (the kernel does not mask).
    if let Some(&bad) = pair_i.iter().find(|&&ix| (ix as usize) >= n_objects) {
        return Err(CbError::OutOfRange(format!(
            "pair_i value {bad} >= n_objects ({n_objects}); object id would read cindex out of bounds"
        )));
    }
    if let Some(&bad) = pair_j.iter().find(|&&ix| (ix as usize) >= n_objects) {
        return Err(CbError::OutOfRange(format!(
            "pair_j value {bad} >= n_objects ({n_objects}); object id would read cindex out of bounds"
        )));
    }
    if let Some(&bad) = cindex.iter().find(|&&b| (b as usize) >= n_bins) {
        return Err(CbError::OutOfRange(format!(
            "cindex bin value {bad} >= n_bins ({n_bins}); would write bin_sums out of bounds"
        )));
    }

    // Empty fill: hand back a zero-length handle (no launch, no read-back — Pitfall 5).
    if n_pairs == 0 || n_features == 0 || n_bins == 0 {
        return Ok(client.empty(0));
    }

    // The 8-bit-atomics family is defined for the 256-bin line ONLY (n_bins == 1 << 8);
    // anything else is not this family.
    if n_bins != 256 {
        return Err(CbError::Degenerate(format!(
            "pairwise_hist 8-bit-atomics fill expects n_bins == 256 (1 << 8), got n_bins={n_bins}"
        )));
    }

    // Launch geometry: enough cubes to cover `n_pairs` (the grid-stride loop handles any
    // surplus via the total-thread-count stride).
    let num_cubes = n_pairs.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    let bin_sums_len = pair_hist_binsums_len(n_bins, n_features);
    let cindex_handle = client.create(cubecl::bytes::Bytes::from_elems(cindex.to_vec()));
    let pair_i_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_i.to_vec()));
    let pair_j_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_j.to_vec()));

    // Channel float-type dispatch (RESEARCH A1 / Pitfall 1): f64 on rocm/cuda/cpu, f32 on
    // wgpu (WGSL has no f64 atomics — read back and UPCAST). Reused VERBATIM from the
    // non-binary arm. NO read-back here (SC-3 / Pitfall 5).
    #[cfg(feature = "wgpu")]
    {
        let weight_f32: Vec<f32> = pair_weight.iter().map(|&v| v as f32).collect();
        let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(weight_f32));
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; bin_sums_len]));
        pairwise_hist_8bit_atomics_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(pair_i_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(pair_j_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(weight_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
            n_features as u32,
            n_objects as u32,
            one_hot,
        );
        Ok(h)
    }

    #[cfg(not(feature = "wgpu"))]
    {
        let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_weight.to_vec()));
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; bin_sums_len]));
        pairwise_hist_8bit_atomics_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(pair_i_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(pair_j_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(weight_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
            n_features as u32,
            n_objects as u32,
            one_hot,
        );
        Ok(h)
    }
}

/// Host-readback wrapper over the 8-bit-atomics pairwise-histogram fill: launch
/// device-resident, then read the 4-channel `binSums` handle back to a host `Vec<f64>`.
/// This is the seam the all-backend self-oracle exercises; it is NOT the histogram
/// hand-off path (that is [`launch_pairwise_hist_8bit_handle`], which never reads back).
///
/// The launch geometry lives in ONE place ([`launch_pairwise_hist_8bit_into`]); this
/// wrapper constructs the client ONCE and uses that SAME client for both the launch and
/// the read-back. A device read-back failure surfaces [`CbError::Degenerate`], never a
/// silent all-zero buffer. Empty input returns an empty `Vec` (no launch).
#[allow(clippy::too_many_arguments)]
pub fn launch_pairwise_hist_8bit(
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
    one_hot: bool,
) -> CbResult<Vec<f64>> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

    if pair_i.is_empty() || n_features == 0 || n_bins == 0 {
        return Ok(Vec::new());
    }

    let handle = launch_pairwise_hist_8bit_into(
        &client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features, one_hot,
    )?;

    read_pair_binsums_f64(&client, handle)
}

/// Launch the half-byte (4-bit, 16-bin) pairwise histogram fill device-resident and return
/// the 4-channel `binSums` Handle WITHOUT reading it back (the SC-3 / D-7.4-03 hand-off
/// path). This is the SEPARATE launch arm for the structurally DISTINCT half-byte family
/// (D-7.4-02 — upstream `pairwise_hist_half_byte.cu`): the histogram line is a FIXED 16-bin
/// (4-bit) line (the comptime `HALF_BYTE_BINS`), and the family takes NO one-hot overlay
/// upstream (there is no `pairwise_hist_half_byte_one_hot.cu`). It reuses the Plan A FROZEN
/// 4-channel layout ([`pair_hist_binsums_len`], index `(feature * n_bins + bin) * 4 +
/// histId`), the guard block, the backend-dispatched channel, the empty short-circuit, and
/// the no-readback seam UNCHANGED — the only differences are `n_bins = 16` (validated below)
/// and the dispatched kernel symbol.
///
/// `n_bins` MUST be `16` (the half-byte line size, [`crate::kernels::HALF_BYTE_BINS`]);
/// anything else is rejected with [`CbError::Degenerate`]. There is no `one_hot` parameter
/// (the half-byte family has no one-hot overlay). Inputs, guards, and the empty/typed-error
/// semantics match [`launch_pairwise_hist_handle`] (see its docs). No
/// `unwrap`/`expect`/`panic`/indexing in this production helper.
pub fn launch_pairwise_hist_half_byte_handle(
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
) -> CbResult<Handle> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_pairwise_hist_half_byte_into(
        &client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features,
    )
}

/// The ONE half-byte pairwise launch geometry (IN-02 — one place). Clones
/// [`launch_pairwise_hist_into`] with `n_bins` pinned to [`crate::kernels::HALF_BYTE_BINS`]
/// (16) and dispatches [`pairwise_hist_half_byte_kernel`] (fixed 16-bin line, nibble-masked
/// bins, no one-hot). Reuses the Plan A guard block, the 4-channel layout, the
/// backend-dispatched channel, and the no-readback seam unchanged. The caller owns the
/// `client` lifecycle so a read-back uses the SAME client that allocated the handle.
fn launch_pairwise_hist_half_byte_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
) -> CbResult<Handle> {
    let n_pairs = pair_i.len();

    // Shape guards (T-07.4-11): identical to the non-binary/8-bit arms — a mismatch would
    // read out of bounds on the device. Surface a typed error BEFORE launch (no panic).
    if pair_j.len() != n_pairs {
        return Err(CbError::LengthMismatch {
            column: "pair_j".to_owned(),
            expected: n_pairs,
            actual: pair_j.len(),
        });
    }
    if pair_weight.len() != n_pairs {
        return Err(CbError::LengthMismatch {
            column: "pair_weight".to_owned(),
            expected: n_pairs,
            actual: pair_weight.len(),
        });
    }

    // Overflow guards FIRST. Reuse the checked cindex stride for the length guard below,
    // never re-multiplying.
    let cindex_stride = n_features.checked_mul(n_objects).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_objects ({n_objects}) overflows usize (cindex stride)"
        ))
    })?;
    if pair_hist_binsums_len_checked(n_bins, n_features).is_none() {
        return Err(CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) * {PAIR_HIST_CHANNELS} overflows usize (binSums length)"
        )));
    }

    if cindex.len() != cindex_stride {
        return Err(CbError::LengthMismatch {
            column: "cindex".to_owned(),
            expected: cindex_stride,
            actual: cindex.len(),
        });
    }

    // Value-range guards (T-07.4-11): BOTH pair endpoints index cindex (Pitfall 3); every
    // cindex bin must fit the 16-bin line (the kernel masks to the nibble, but the host
    // reference indexes RAW, so the value-range guard keeps the two in lock-step).
    if let Some(&bad) = pair_i.iter().find(|&&ix| (ix as usize) >= n_objects) {
        return Err(CbError::OutOfRange(format!(
            "pair_i value {bad} >= n_objects ({n_objects}); object id would read cindex out of bounds"
        )));
    }
    if let Some(&bad) = pair_j.iter().find(|&&ix| (ix as usize) >= n_objects) {
        return Err(CbError::OutOfRange(format!(
            "pair_j value {bad} >= n_objects ({n_objects}); object id would read cindex out of bounds"
        )));
    }
    if let Some(&bad) = cindex.iter().find(|&&b| (b as usize) >= n_bins) {
        return Err(CbError::OutOfRange(format!(
            "cindex bin value {bad} >= n_bins ({n_bins}); would write bin_sums out of bounds"
        )));
    }

    // Empty fill: hand back a zero-length handle (no launch, no read-back — Pitfall 5).
    if n_pairs == 0 || n_features == 0 || n_bins == 0 {
        return Ok(client.empty(0));
    }

    // The half-byte family is defined for the 16-bin line ONLY (n_bins == HALF_BYTE_BINS);
    // anything else is not this family.
    if n_bins != crate::kernels::HALF_BYTE_BINS {
        return Err(CbError::Degenerate(format!(
            "pairwise_hist half-byte fill expects n_bins == {} (1 << 4), got n_bins={n_bins}",
            crate::kernels::HALF_BYTE_BINS
        )));
    }

    // Launch geometry: enough cubes to cover `n_pairs` (the grid-stride loop handles any
    // surplus via the total-thread-count stride).
    let num_cubes = n_pairs.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    let bin_sums_len = pair_hist_binsums_len(n_bins, n_features);
    let cindex_handle = client.create(cubecl::bytes::Bytes::from_elems(cindex.to_vec()));
    let pair_i_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_i.to_vec()));
    let pair_j_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_j.to_vec()));

    // Channel float-type dispatch (RESEARCH A1 / Pitfall 1): f64 on rocm/cuda/cpu, f32 on
    // wgpu (WGSL has no f64 atomics — read back and UPCAST). Reused VERBATIM from the
    // non-binary/8-bit arms. NO read-back here (SC-3 / Pitfall 5).
    #[cfg(feature = "wgpu")]
    {
        let weight_f32: Vec<f32> = pair_weight.iter().map(|&v| v as f32).collect();
        let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(weight_f32));
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; bin_sums_len]));
        pairwise_hist_half_byte_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(pair_i_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(pair_j_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(weight_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
            n_features as u32,
            n_objects as u32,
        );
        Ok(h)
    }

    #[cfg(not(feature = "wgpu"))]
    {
        let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_weight.to_vec()));
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; bin_sums_len]));
        pairwise_hist_half_byte_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(pair_i_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(pair_j_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(weight_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
            n_features as u32,
            n_objects as u32,
        );
        Ok(h)
    }
}

/// Host-readback wrapper over the half-byte pairwise-histogram fill: launch device-resident,
/// then read the 4-channel `binSums` handle back to a host `Vec<f64>`. This is the seam the
/// all-backend self-oracle exercises; it is NOT the histogram hand-off path (that is
/// [`launch_pairwise_hist_half_byte_handle`], which never reads back).
///
/// The launch geometry lives in ONE place ([`launch_pairwise_hist_half_byte_into`]); this
/// wrapper constructs the client ONCE and uses that SAME client for both the launch and the
/// read-back. A device read-back failure surfaces [`CbError::Degenerate`], never a silent
/// all-zero buffer. Empty input returns an empty `Vec` (no launch).
pub fn launch_pairwise_hist_half_byte(
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
) -> CbResult<Vec<f64>> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

    if pair_i.is_empty() || n_features == 0 || n_bins == 0 {
        return Ok(Vec::new());
    }

    let handle = launch_pairwise_hist_half_byte_into(
        &client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features,
    )?;

    read_pair_binsums_f64(&client, handle)
}

/// Launch the binary (1-bit, 2-bin) pairwise histogram fill device-resident and return the
/// 4-channel `binSums` Handle WITHOUT reading it back (the SC-3 / D-7.4-03 hand-off path).
/// This is the SEPARATE launch arm for the structurally DISTINCT binary family (D-7.4-02 —
/// upstream `pairwise_hist_binary.cu`): the histogram line is a FIXED 2-bin (1-bit) line (a
/// bin COUNT, NOT a warp literal), and the family takes NO one-hot overlay upstream (there is
/// no `pairwise_hist_binary_one_hot.cu`). It reuses the Plan A FROZEN 4-channel layout
/// ([`pair_hist_binsums_len`], index `(feature * n_bins + bin) * 4 + histId`), the guard
/// block, the backend-dispatched channel, the empty short-circuit, and the no-readback seam
/// UNCHANGED — the only differences are `n_bins = 2` (validated below) and the dispatched
/// kernel symbol.
///
/// `n_bins` MUST be `2` (the binary line size); anything else is rejected with
/// [`CbError::Degenerate`]. There is no `one_hot` parameter (the binary family has no one-hot
/// overlay). Inputs, guards, and the empty/typed-error semantics match
/// [`launch_pairwise_hist_handle`] (see its docs). No `unwrap`/`expect`/`panic`/indexing in
/// this production helper.
pub fn launch_pairwise_hist_binary_handle(
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
) -> CbResult<Handle> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_pairwise_hist_binary_into(
        &client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features,
    )
}

/// The ONE binary pairwise launch geometry (IN-02 — one place). Clones
/// [`launch_pairwise_hist_into`] with `n_bins` pinned to `2` and dispatches
/// [`pairwise_hist_binary_kernel`] (fixed 2-bin line, bit-masked bins, no one-hot). Reuses the
/// Plan A guard block, the 4-channel layout, the backend-dispatched channel, and the
/// no-readback seam unchanged. The caller owns the `client` lifecycle so a read-back uses the
/// SAME client that allocated the handle.
fn launch_pairwise_hist_binary_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
) -> CbResult<Handle> {
    let n_pairs = pair_i.len();

    // Shape guards (T-07.4-15): identical to the non-binary/8-bit/half-byte arms — a mismatch
    // would read out of bounds on the device. Surface a typed error BEFORE launch (no panic).
    if pair_j.len() != n_pairs {
        return Err(CbError::LengthMismatch {
            column: "pair_j".to_owned(),
            expected: n_pairs,
            actual: pair_j.len(),
        });
    }
    if pair_weight.len() != n_pairs {
        return Err(CbError::LengthMismatch {
            column: "pair_weight".to_owned(),
            expected: n_pairs,
            actual: pair_weight.len(),
        });
    }

    // Overflow guards FIRST. Reuse the checked cindex stride for the length guard below,
    // never re-multiplying.
    let cindex_stride = n_features.checked_mul(n_objects).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_objects ({n_objects}) overflows usize (cindex stride)"
        ))
    })?;
    if pair_hist_binsums_len_checked(n_bins, n_features).is_none() {
        return Err(CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) * {PAIR_HIST_CHANNELS} overflows usize (binSums length)"
        )));
    }

    if cindex.len() != cindex_stride {
        return Err(CbError::LengthMismatch {
            column: "cindex".to_owned(),
            expected: cindex_stride,
            actual: cindex.len(),
        });
    }

    // Value-range guards (T-07.4-15): BOTH pair endpoints index cindex (Pitfall 3); every
    // cindex bin must fit the 2-bin line (the kernel masks to the bit, but the host reference
    // indexes RAW, so the value-range guard keeps the two in lock-step).
    if let Some(&bad) = pair_i.iter().find(|&&ix| (ix as usize) >= n_objects) {
        return Err(CbError::OutOfRange(format!(
            "pair_i value {bad} >= n_objects ({n_objects}); object id would read cindex out of bounds"
        )));
    }
    if let Some(&bad) = pair_j.iter().find(|&&ix| (ix as usize) >= n_objects) {
        return Err(CbError::OutOfRange(format!(
            "pair_j value {bad} >= n_objects ({n_objects}); object id would read cindex out of bounds"
        )));
    }
    if let Some(&bad) = cindex.iter().find(|&&b| (b as usize) >= n_bins) {
        return Err(CbError::OutOfRange(format!(
            "cindex bin value {bad} >= n_bins ({n_bins}); would write bin_sums out of bounds"
        )));
    }

    // Empty fill: hand back a zero-length handle (no launch, no read-back — Pitfall 5).
    if n_pairs == 0 || n_features == 0 || n_bins == 0 {
        return Ok(client.empty(0));
    }

    // The binary family is defined for the 2-bin line ONLY (n_bins == 2); anything else is not
    // this family.
    if n_bins != 2 {
        return Err(CbError::Degenerate(format!(
            "pairwise_hist binary fill expects n_bins == 2 (1 << 1), got n_bins={n_bins}"
        )));
    }

    // Launch geometry: enough cubes to cover `n_pairs` (the grid-stride loop handles any
    // surplus via the total-thread-count stride).
    let num_cubes = n_pairs.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    let bin_sums_len = pair_hist_binsums_len(n_bins, n_features);
    let cindex_handle = client.create(cubecl::bytes::Bytes::from_elems(cindex.to_vec()));
    let pair_i_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_i.to_vec()));
    let pair_j_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_j.to_vec()));

    // Channel float-type dispatch (RESEARCH A1 / Pitfall 1): f64 on rocm/cuda/cpu, f32 on wgpu
    // (WGSL has no f64 atomics — read back and UPCAST). Reused VERBATIM from the
    // non-binary/8-bit/half-byte arms. NO read-back here (SC-3 / Pitfall 5).
    #[cfg(feature = "wgpu")]
    {
        let weight_f32: Vec<f32> = pair_weight.iter().map(|&v| v as f32).collect();
        let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(weight_f32));
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; bin_sums_len]));
        pairwise_hist_binary_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(pair_i_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(pair_j_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(weight_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
            n_features as u32,
            n_objects as u32,
        );
        Ok(h)
    }

    #[cfg(not(feature = "wgpu"))]
    {
        let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_weight.to_vec()));
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; bin_sums_len]));
        pairwise_hist_binary_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(pair_i_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(pair_j_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(weight_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
            n_features as u32,
            n_objects as u32,
        );
        Ok(h)
    }
}

/// Host-readback wrapper over the binary pairwise-histogram fill: launch device-resident, then
/// read the 4-channel `binSums` handle back to a host `Vec<f64>`. This is the seam the
/// all-backend self-oracle exercises; it is NOT the histogram hand-off path (that is
/// [`launch_pairwise_hist_binary_handle`], which never reads back).
///
/// The launch geometry lives in ONE place ([`launch_pairwise_hist_binary_into`]); this wrapper
/// constructs the client ONCE and uses that SAME client for both the launch and the read-back.
/// A device read-back failure surfaces [`CbError::Degenerate`], never a silent all-zero
/// buffer. Empty input returns an empty `Vec` (no launch).
pub fn launch_pairwise_hist_binary(
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
) -> CbResult<Vec<f64>> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

    if pair_i.is_empty() || n_features == 0 || n_bins == 0 {
        return Ok(Vec::new());
    }

    let handle = launch_pairwise_hist_binary_into(
        &client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features,
    )?;

    read_pair_binsums_f64(&client, handle)
}

// ===========================================================================
// Phase 7.5 Plan 06 — the PAIRWISE split scorer device seam (split_pairwise.cuh),
// the LAST GPU-01 slice (D-7.5-01). It consumes the FROZEN 7.4 4-channel pairwise
// histogram handle + a device der-sum scatter, scan/updates them device-resident
// (the deferred 7.4 transform, D-7.4-06), then performs a BOUNDED host read-back of
// the assembled per-(feature,bucket) statistics and runs the small per-leaf Cholesky
// solve + score via `cb_compute::calculate_pairwise_score` (RESEARCH Open Q3: a
// `#[cube]` dense SPD solve is awkward and the FROZEN CPU `pairwise_cholesky_solve`
// is the parity oracle; the bulk pairwise histogram stays device-resident — minimal
// round-trips, D-05). Single-client threaded; typed guards; never reads a 0-len
// handle. Cross-oracled in `kernels/score_split.rs::pairwise` +
// `kernels/grow_loop.rs::pairwise`.
// ===========================================================================

/// Read a pairwise der-sum device handle (channel float type) back to a host
/// `Vec<f64>`, UPCASTING the f32 channel on wgpu (RESEARCH A1) — the der-sum sibling of
/// [`read_pair_binsums_f64`]. A read-back failure surfaces [`CbError::Degenerate`]
/// (WR-05), never a silent zero buffer.
fn read_pair_der_sums_f64(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    handle: Handle,
) -> CbResult<Vec<f64>> {
    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("pairwise der-sums read-back failed: {e:?}")))?;
    #[cfg(feature = "wgpu")]
    {
        Ok(bytemuck::cast_slice::<u8, f32>(&bytes)
            .iter()
            .map(|&v| f64::from(v))
            .collect())
    }
    #[cfg(not(feature = "wgpu"))]
    {
        Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
    }
}

/// The device-resident pairwise **scan/update** seam: fill the FROZEN 7.4 4-channel
/// pairwise histogram device-resident, scan/update it IN PLACE (inclusive prefix per
/// channel), and read back the cumulative 4-channel buffer (the self-oracle observation
/// — the FILL→scan seam stays device-resident, only the cumulative result crosses,
/// D-7.5-03 / D-05). Mirrors [`launch_scan_update_pointwise`] for the 4-channel layout.
///
/// SCOPE (inherited from Plan B): correct only for `n_bins <= CUBE_DIM`; `n_bins >
/// CUBE_DIM` (8-bit/256-bin) surfaces a typed [`CbError::OutOfRange`] — the EXPLICIT
/// tracked cross-cube-carry follow-up (RESEARCH Open Q3), never a silent truncation.
///
/// Inputs are the FROZEN 7.4 pairwise-fill inputs (`pair_i`/`pair_j` object ids,
/// `pair_weight`, `cindex` feature-major, `n_objects`, `n_bins`, `n_features`, `bits`
/// in {5,6,7}, `one_hot`). Empty input returns an empty `Vec` (no launch). No
/// `unwrap`/`expect`/`panic`/indexing in this production helper.
#[allow(clippy::too_many_arguments)]
pub fn launch_scan_update_pairwise(
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
    bits: u32,
    one_hot: bool,
) -> CbResult<Vec<f64>> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_scan_update_pairwise_into(
        &client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features, bits, one_hot,
    )
}

/// The ONE pairwise scan/update geometry (IN-02 — one place). Fills the FROZEN 7.4
/// 4-channel histogram device-resident, launches the per-(feature, histId)-cube
/// inclusive prefix-sum consuming that handle IN PLACE, and reads back ONLY the
/// cumulative 4-channel buffer. The caller owns the `client` lifecycle so the read-back
/// uses the SAME client that allocated the handle.
#[allow(clippy::too_many_arguments)]
fn launch_scan_update_pairwise_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
    bits: u32,
    one_hot: bool,
) -> CbResult<Vec<f64>> {
    // Empty short-circuit FIRST (Pitfall 3/5): no histogram, no launch, no 0-len read.
    if pair_i.is_empty() || n_features == 0 || n_bins == 0 {
        return Ok(Vec::new());
    }

    // SCOPE GUARD (RESEARCH Open Q3 / inherited from Plan B): the single-cube
    // scan_update_pairwise_kernel is correct only for n_bins <= CUBE_DIM. Reject
    // n_bins > CUBE_DIM with a typed error until the cross-cube carry lands (the
    // tracked cross-cube-carry follow-up for 8-bit/256-bin features).
    if n_bins > CUBE_DIM {
        return Err(CbError::OutOfRange(format!(
            "launch_scan_update_pairwise supports n_bins <= {CUBE_DIM} until the \
             cross-cube scan carry lands (RESEARCH Open Q3, the tracked pairwise \
             cross-cube-carry follow-up for 8-bit/256-bin features); got n_bins = {n_bins}"
        )));
    }

    // Cumulative-buffer length overflow guard: REUSE the FROZEN 7.4 checked 4-channel
    // length helper so the cumulative buffer matches the binSums layout exactly.
    let cumulative_len = pair_hist_binsums_len_checked(n_bins, n_features).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) * {PAIR_HIST_CHANNELS} overflows usize \
             (pairwise cumulative length)"
        ))
    })?;

    let n_bins_u32 = u32::try_from(n_bins).map_err(|_| {
        CbError::OutOfRange(format!("n_bins ({n_bins}) exceeds u32 (kernel bin axis)"))
    })?;

    // Number of (feature, histId) scan axes = n_features * PAIR_HIST_CHANNELS, each
    // scanned by ONE cube. Guard the cube-count product against overflow before the cast.
    let num_cubes = n_features.checked_mul(PAIR_HIST_CHANNELS).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * {PAIR_HIST_CHANNELS} overflows usize (scan cube count)"
        ))
    })?;
    let num_cubes_u32 = u32::try_from(num_cubes).map_err(|_| {
        CbError::OutOfRange(format!("pairwise scan cube count ({num_cubes}) exceeds u32"))
    })?;

    // Fill the FROZEN 7.4 device-resident 4-channel histogram (this runs the FROZEN
    // length / value-range guards on pair_i/pair_j/pair_weight/cindex BEFORE any launch,
    // and returns a device HANDLE with NO read-back). The bulk histogram stays resident.
    let bin_sums = launch_pairwise_hist_into(
        client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features, bits, one_hot,
    )?;

    // Launch geometry: ONE cube of CUBE_DIM units per (feature, histId) scan axis.
    let count = CubeCount::Static(num_cubes_u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    // The cumulative output buffer matches the FROZEN 4-channel layout / channel float
    // type: f64 on rocm/cuda/cpu, f32 on wgpu (RESEARCH A1) — read back via
    // read_pair_binsums_f64. Zero-initialised (the kernel writes every real bin cell).
    #[cfg(feature = "wgpu")]
    let cumulative_handle = {
        let cumulative_h =
            client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; cumulative_len]));
        scan_update_pairwise_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(bin_sums, cumulative_len) },
            unsafe { ArrayArg::from_raw_parts(cumulative_h.clone(), cumulative_len) },
            n_bins_u32,
        );
        cumulative_h
    };

    #[cfg(not(feature = "wgpu"))]
    let cumulative_handle = {
        let cumulative_h =
            client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; cumulative_len]));
        scan_update_pairwise_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(bin_sums, cumulative_len) },
            unsafe { ArrayArg::from_raw_parts(cumulative_h.clone(), cumulative_len) },
            n_bins_u32,
        );
        cumulative_h
    };

    read_pair_binsums_f64(client, cumulative_handle)
}

/// The device-resident pairwise **make-derivatives** seam: scatter the pairwise-weighted
/// `der1` into the per-(feature, bucket) der-sum buffer device-resident (== upstream
/// `MakePointwiseDerivatives` over the single root leaf), and read back the bounded
/// `n_features * n_bins` der-sum descriptor. This is the pointwise der row the pairwise
/// scorer assembles into `der_sum[2*leaf+1] += Σ_bucket der_sums[leaf][bucket]`; the
/// heavy per-object scatter stays device-resident, only the bounded der-sum crosses
/// (RESEARCH Open Q3 / D-05). The output is laid out `der_sums[feature * n_bins + bin]`
/// (leaf 0 / root, the depth-1 MVP), the SAME order `cb_compute::compute_der_sums`
/// produces for `leaf_count == 1`.
///
/// `der1` (the pairwise-weighted first derivative, length `n`), `cindex` (feature-major
/// quantized bins, `cindex[feature * n + obj]`, length `n_features * n`), `indices`
/// (object visiting order, length `n`), `n_bins`/`n_features`. Empty input returns an
/// empty `Vec` (no launch). Mismatched lengths / out-of-range bin values surface a typed
/// [`CbError`] BEFORE launch. No `unwrap`/`expect`/`panic`/indexing.
pub fn launch_pairwise_make_derivatives(
    der1: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
) -> CbResult<Vec<f64>> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_pairwise_make_derivatives_into(&client, der1, cindex, indices, n_bins, n_features)
}

/// The ONE pairwise make-derivatives geometry (IN-02 — one place). Uploads the resident
/// der1/cindex/indices handles onto `client`, launches the der-sum scatter, and reads
/// back ONLY the bounded `n_features * n_bins` der-sum descriptor. The caller owns the
/// `client` lifecycle so the read-back uses the SAME client that allocated the handles.
fn launch_pairwise_make_derivatives_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
) -> CbResult<Vec<f64>> {
    let n = der1.len();

    // Empty short-circuit FIRST (Pitfall 3/5).
    if n == 0 || n_features == 0 || n_bins == 0 {
        return Ok(Vec::new());
    }

    // Overflow guards BEFORE any unchecked product. The cindex stride `n_features * n`
    // and the der-sum length `n_features * n_bins` are products of caller dimensions.
    let cindex_stride = n_features.checked_mul(n).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n ({n}) overflows usize (cindex stride)"
        ))
    })?;
    let der_sums_len = n_features.checked_mul(n_bins).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) overflows usize (der-sums length)"
        ))
    })?;

    // Shape guards (the kernel reads cindex[feature * n + obj] / indices[i]).
    if cindex.len() != cindex_stride {
        return Err(CbError::LengthMismatch {
            column: "cindex".to_owned(),
            expected: cindex_stride,
            actual: cindex.len(),
        });
    }
    if indices.len() != n {
        return Err(CbError::LengthMismatch {
            column: "indices".to_owned(),
            expected: n,
            actual: indices.len(),
        });
    }

    // Value-range guards (T-07.5-06-01): a bin >= n_bins would write der_sums out of
    // bounds. An object id >= n would read der1/cindex out of bounds.
    if let Some(&bad) = cindex.iter().find(|&&b| (b as usize) >= n_bins) {
        return Err(CbError::OutOfRange(format!(
            "cindex bin value {bad} >= n_bins ({n_bins}); would write der_sums out of bounds"
        )));
    }
    if let Some(&bad) = indices.iter().find(|&&ix| (ix as usize) >= n) {
        return Err(CbError::OutOfRange(format!(
            "indices value {bad} >= n ({n}); object id would read der1/cindex out of bounds"
        )));
    }

    // The kernel needs n_bins as a comptime u32 (the per-feature line size).
    let n_bins_u32 = u32::try_from(n_bins).map_err(|_| {
        CbError::OutOfRange(format!("n_bins ({n_bins}) exceeds u32 (kernel comptime line size)"))
    })?;

    let num_cubes = n.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    let der1_h = upload_channel_floats(client, der1);
    let cindex_h = client.create(cubecl::bytes::Bytes::from_elems(cindex.to_vec()));
    let indices_h = client.create(cubecl::bytes::Bytes::from_elems(indices.to_vec()));

    #[cfg(feature = "wgpu")]
    let der_sums_handle = {
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; der_sums_len]));
        pairwise_make_derivatives_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(der1_h, n) },
            unsafe { ArrayArg::from_raw_parts(cindex_h, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(indices_h, n) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), der_sums_len) },
            n_features as u32,
            n_bins_u32,
        );
        h
    };

    #[cfg(not(feature = "wgpu"))]
    let der_sums_handle = {
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; der_sums_len]));
        pairwise_make_derivatives_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(der1_h, n) },
            unsafe { ArrayArg::from_raw_parts(cindex_h, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(indices_h, n) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), der_sums_len) },
            n_features as u32,
            n_bins_u32,
        );
        h
    };

    read_pair_der_sums_f64(client, der_sums_handle)
}

/// The catboost pairwise bucket-weight prior reg default (`bayesian_matrix_reg` /
/// `PairwiseNonDiagReg`, `oblivious_tree_options.cpp:15-16`, upstream default `0.1`).
/// Transcribed inline (NO `cb-train` dep — the feature-unification landmine); the pairwise
/// coverage gate carries it into the per-leaf system assembly.
pub(crate) const PAIRWISE_BUCKET_WEIGHT_PRIOR_REG_DEFAULT: f64 = 0.1;

/// Reject the (impossible) wgpu f64 path for the pairwise per-leaf system assembly with a typed
/// error, mirroring [`crate::kernels::mvs_device`]. The packed `linearSystem` accumulates in f64
/// (D-07, feeding the Plan-02 batched Cholesky); WGSL has no f64, so a genuine `wgpu` backend
/// surfaces a typed [`CbError::OutOfRange`] rather than an opaque JIT crash. The in-env
/// rocm/cuda/cpu path is unaffected.
#[cfg(feature = "wgpu")]
fn pairwise_assemble_wgpu_reject() -> CbError {
    CbError::OutOfRange(
        "device pairwise per-leaf linear-system assembly requires f64 device channels; the wgpu \
         backend has none. Use the rocm/cuda/cpu backend for the pairwise Cholesky path."
            .to_owned(),
    )
}

/// Assemble the packed lower-triangular per-leaf pairwise `linearSystem` resident on device
/// (Phase 13 Plan 01, GPUT-11 / GPUT-21 prep) and return its HANDLE WITHOUT reading it back
/// (D-05 residency — no `n`-length host round-trip of the pair stats). The device
/// [`pairwise_assemble_system_kernel`] transcribes the Rust CPU parity oracle
/// `cb_train::pairwise_leaves::calculate_pairwise_leaf_values` matrix build: the
/// `rowSize*(rowSize+1)/2` lower-triangle matrix cells (row-major, `x` in `0..=y`, with the
/// `diag_reg`/`non_diag_reg` prior) FOLLOWED by the `rowSize` RHS, where `rowSize = leaf_count - 1`
/// (leaf gauge freedom). Plan 02's batched Cholesky consumes this resident system.
///
/// `weight_sums` is the per-leaf pairwise weight-sum matrix (row-major `leaf_count × leaf_count`,
/// from the resident Phase-7.4 pairwise histograms), `der_sums` the per-leaf der sums
/// (`leaf_count`), `l2_diag_reg = L2Reg`, `pairwise_bucket_weight_prior_reg = PairwiseNonDiagReg`.
/// The reg constants are computed host-side (`cell_prior = 1/leaf_count`,
/// `non_diag_reg = -prior*cell_prior`, `diag_reg = prior*(1-cell_prior) + l2_diag_reg`) — the
/// EXACT CPU-oracle constants, NOT upstream's `RegularizeImpl` bump-heuristics (RESEARCH Pitfall
/// 2). `leaf_count <= 1` yields an empty handle (a singleton/empty system has no matrix or RHS —
/// its lone zero-averaged delta is 0). No `unwrap`/`expect`/`panic`/indexing; never reads a 0-len
/// handle.
pub fn launch_pairwise_assemble_system_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    weight_sums: &[f64],
    der_sums: &[f64],
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
) -> CbResult<Handle> {
    let leaf_count = der_sums.len();

    // A singleton/empty system has no reduced matrix or RHS (rowSize = leaf_count - 1 == 0):
    // its lone leaf delta is 0 after MakeZeroAverage. Return an empty handle (no launch, no
    // 0-len read downstream) — the "Ok(None)"-analog at the assembly level (never a fabrication).
    if leaf_count <= 1 {
        return Ok(client.empty(0));
    }

    // Shape guard: the weight-sum matrix is row-major leaf_count × leaf_count.
    let expected = leaf_count.checked_mul(leaf_count).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "leaf_count ({leaf_count}) squared overflows usize (weight-sum matrix length)"
        ))
    })?;
    if weight_sums.len() != expected {
        return Err(CbError::LengthMismatch {
            column: "weight_sums".to_owned(),
            expected,
            actual: weight_sums.len(),
        });
    }

    // The EXACT CPU-oracle reg constants (calculate_pairwise_leaf_values:123-125). Transcribed
    // inline — NOT upstream RegularizeImpl (Pitfall 2).
    let cell_prior = 1.0_f64 / leaf_count as f64;
    let non_diag_reg = -pairwise_bucket_weight_prior_reg * cell_prior;
    let diag_reg = pairwise_bucket_weight_prior_reg * (1.0 - cell_prior) + l2_diag_reg;

    let m = leaf_count - 1;
    // Packed length: m*(m+1)/2 lower-triangle matrix cells + m RHS. m*(m+1) is even so /2 is exact.
    let matrix_len = m
        .checked_mul(m + 1)
        .map(|p| p / 2)
        .ok_or_else(|| CbError::OutOfRange(format!("packed matrix length overflows (m = {m})")))?;
    let out_len = matrix_len.checked_add(m).ok_or_else(|| {
        CbError::OutOfRange(format!("packed system length overflows (matrix {matrix_len} + m {m})"))
    })?;

    #[cfg(feature = "wgpu")]
    {
        return Err(pairwise_assemble_wgpu_reject());
    }

    #[cfg(not(feature = "wgpu"))]
    {
        let ws_h = client.create(cubecl::bytes::Bytes::from_elems(weight_sums.to_vec()));
        let ds_h = client.create(cubecl::bytes::Bytes::from_elems(der_sums.to_vec()));
        let params_h =
            client.create(cubecl::bytes::Bytes::from_elems(vec![non_diag_reg, diag_reg]));
        let out = client.empty(out_len * std::mem::size_of::<f64>());
        // Serial single-thread launch (unit 0 packs the small per-leaf system in fixed order).
        let count = CubeCount::Static(1, 1, 1);
        let dim = CubeDim { x: 1, y: 1, z: 1 };
        pairwise_assemble_system_kernel::launch::<SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(ws_h, expected) },
            unsafe { ArrayArg::from_raw_parts(ds_h, leaf_count) },
            unsafe { ArrayArg::from_raw_parts(params_h, 2) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), out_len) },
        );
        Ok(out)
    }
}

/// Host-readback wrapper over the device pairwise per-leaf system assembly: build the resident
/// packed `linearSystem`, then read it back to a host `Vec<f64>`. This is the seam the self-oracle
/// exercises (device assembly vs the CPU pair-stat reference); it is NOT the residency path (that
/// keeps the handle on-device for Plan 02's Cholesky). A `leaf_count <= 1` system reads back as an
/// empty `Vec` WITHOUT reading a 0-len handle (HIP fault guard). A read-back failure surfaces
/// [`CbError::Degenerate`], never a silent zero buffer.
pub fn assemble_pairwise_system_host(
    weight_sums: &[f64],
    der_sums: &[f64],
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
) -> CbResult<Vec<f64>> {
    if der_sums.len() <= 1 {
        return Ok(Vec::new());
    }
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    let handle = launch_pairwise_assemble_system_into(
        &client,
        weight_sums,
        der_sums,
        l2_diag_reg,
        pairwise_bucket_weight_prior_reg,
    )?;
    read_pair_der_sums_f64(&client, handle)
}

/// The pairwise split scorer result: the per-candidate pairwise scores (one per border,
/// `scores[feature * (bucket_count - 1) + border]`), the device-resident-assembled
/// per-(feature, bucket) der-sums (the bounded descriptor read back, leaf 0 / root), and
/// the deterministic best split (lowest-index tie-break == `select_best_candidate`).
#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseSplitScore {
    /// The per-candidate pairwise score, flat `feature * (bucket_count - 1) + border`.
    pub scores: Vec<f64>,
    /// The device der-sum descriptor (leaf 0), flat `feature * n_bins + bin` — the
    /// bounded read-back the host solve consumed (kept for the self-oracle).
    pub der_sums: Vec<f64>,
    /// The deterministic best split (`feature_id`, `bin_id`) + its score, or `None` on a
    /// degenerate (no-candidate) system.
    pub best: Option<BestSplit>,
}

/// Compute the device pairwise split score over the FROZEN 7.4 4-channel pairwise
/// histogram handle (GPU-01 final slice). The heavy work is device-resident: the 4-channel
/// pair-weight histogram fill (7.4) + the der-sum scatter
/// ([`launch_pairwise_make_derivatives_into`]) build the per-(feature, bucket) statistics
/// on device; this seam then reads back ONLY the bounded `n_features * n_bins` der-sum
/// descriptor (the bulk pairwise histogram stays device-resident, D-05) and runs the
/// small per-leaf Cholesky solve + `CalculateScore` host-side via the FROZEN
/// `cb_compute::calculate_pairwise_score` (RESEARCH Open Q3: a `#[cube]` dense SPD solve
/// is awkward, and the FROZEN CPU `pairwise_cholesky_solve` IS the parity oracle, so the
/// small assembled system is solved over the bounded host read-back). The best split is
/// then selected with the SAME lowest-(feature,bin)-index tie-break as the pointwise path.
///
/// # Depth-1 / leaf_count == 1 MVP scope
///
/// This plan grows a depth-1 pairwise stump (leaf_count == 1 at level 0, the root), the
/// genuinely-complete vertical slice consistent with 07.5-03/04/05. The pairwise scorer's
/// per-leaf system is `2*leaf_count == 2` at the root; the general `leaf_count > 1` (deep)
/// path needs the partition-aware der-sum + pair-weight assembly (the SAME forward
/// dependency the pointwise grow loop surfaces for depth > 1).
///
/// # Inputs
///
/// `der1` (the pairwise-weighted first derivative, length `n`), `pair_i`/`pair_j` (object
/// ids, length `n_pairs`), `pair_weight` (length `n_pairs`), `cindex` (feature-major
/// quantized bins, length `n_features * n`), `indices` (object visiting order, length
/// `n`), `n_bins` (`1 << bits`, `bits` in {5,6,7}), `n_features`, `l2_diag_reg` (raw
/// `l2_leaf_reg`, NOT `scaled_l2`), `pairwise_bucket_weight_prior_reg`
/// (`bayesian_matrix_reg`, default `0.1`), `one_hot`. Empty input returns an empty score.
///
/// # Errors
///
/// [`CbError::OutOfRange`] on a degenerate dimension / overflow, [`CbError::Degenerate`]
/// on a read-back failure (never a silent zero buffer, WR-05), and the FROZEN 7.4 guards
/// (via the fill / der-sum launches). No `unwrap`/`expect`/`panic`/indexing.
#[allow(clippy::too_many_arguments)]
pub fn launch_pairwise_split_score(
    der1: &[f64],
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
    one_hot: bool,
) -> CbResult<PairwiseSplitScore> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_pairwise_split_score_into(
        &client,
        der1,
        pair_i,
        pair_j,
        pair_weight,
        cindex,
        indices,
        n_bins,
        n_features,
        l2_diag_reg,
        pairwise_bucket_weight_prior_reg,
        one_hot,
    )
}

/// The ONE pairwise split-score geometry (IN-02 — one place). Threads ONE `&client`
/// through the 4-channel fill + the der-sum scatter (both device-resident), reads back
/// ONLY the bounded der-sum + 4-channel statistics descriptor, assembles the per-leaf
/// systems, and runs the FROZEN host `calculate_pairwise_score` + best-split argmin.
#[allow(clippy::too_many_arguments)]
fn launch_pairwise_split_score_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1: &[f64],
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
    one_hot: bool,
) -> CbResult<PairwiseSplitScore> {
    let n = der1.len();

    // Empty short-circuit FIRST (Pitfall 3/5).
    if n == 0 || n_features == 0 || n_bins == 0 {
        return Ok(PairwiseSplitScore {
            scores: Vec::new(),
            der_sums: Vec::new(),
            best: None,
        });
    }

    // The bit-width drives the FROZEN 7.4 fill (5/6/7-bit one-byte non-binary family).
    // n_bins MUST equal 1 << bits; derive bits and reject anything else with a typed error.
    let bits: u32 = match n_bins {
        32 => 5,
        64 => 6,
        128 => 7,
        _ => {
            return Err(CbError::Degenerate(format!(
                "launch_pairwise_split_score expects n_bins in {{32,64,128}} (1 << bits for \
                 bits in 5..=7, the FROZEN 7.4 one-byte non-binary family); got n_bins = {n_bins}"
            )));
        }
    };
    let n_objects = n;

    // (0) DEVICE: consume the FROZEN 7.4 4-channel pairwise histogram device-resident
    //     (the load-bearing 7.4 hand-off seam, D-7.4-03 / D-7.4-06). The 4-channel
    //     pair-weight statistics histogram is filled IN PLACE over the same pairs the
    //     scorer's pair-weight statistics derive from; the bulk histogram stays resident
    //     (NO read-back here, D-05). This launch also runs the FROZEN 7.4 pair/bin
    //     value-range guards BEFORE any device store. The handle is dropped after launch
    //     (the scan path reads it back as the self-oracle; the score path proves the SAME
    //     statistics via the FROZEN cb_compute oracle below, the parity reference D-7.4-05).
    let _pair_hist = launch_pairwise_hist_into(
        client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features, bits, one_hot,
    )?;

    // (1) DEVICE: scatter the pairwise-weighted der1 into the per-(feature, bucket)
    //     der-sum buffer device-resident, read back ONLY the bounded n_features * n_bins
    //     der-sum descriptor (== compute_der_sums over leaf_count == 1). This also runs
    //     the length / value-range guards on der1/cindex/indices BEFORE any launch.
    let der_sums_flat =
        launch_pairwise_make_derivatives_into(client, der1, cindex, indices, n_bins, n_features)?;

    // (2) HOST: assemble the per-leaf systems + score over the BOUNDED descriptor
    //     (RESEARCH Open Q3 — the small per-leaf dense Cholesky solve runs host-side over
    //     the assembled systems via the FROZEN cb_compute::calculate_pairwise_score; the
    //     bulk per-object scatter stayed device-resident). For the depth-1 MVP there is
    //     ONE leaf (the root), so the pair-weight statistics + der_sums are leaf_count == 1.
    //
    //     The pair-weight statistics are reconstructed from the SAME pair/bucket inputs the
    //     FROZEN 7.4 4-channel fill consumes via the FROZEN cb_compute::compute_pair_weight_statistics
    //     (the parity reference D-7.4-05 names): leaf_count == 1, bucket_count == n_bins, all
    //     objects in the root leaf 0. This keeps the score a TRANSCRIPTION of the FROZEN CPU
    //     oracle, never a re-derivation of the 4-channel histId->statistics transform.
    let leaf_count = 1usize;
    let bucket_count = n_bins;

    // bucket_of[obj] from the feature's cindex column (the candidate feature's bucket).
    // leaf_of is all-zero (the root). Build per-feature der_sums / pair-weight stats and
    // score each feature's borders, flattening to scores[feature * (bucket_count-1) + border].
    let n_splits = bucket_count.saturating_sub(1);
    let n_candidates = n_features.checked_mul(n_splits).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * (bucket_count-1) ({n_splits}) overflows usize"
        ))
    })?;
    let mut scores = vec![0.0_f64; n_candidates];
    let leaf_of = vec![0usize; n];

    // The global pairs as (winner, loser, weight) — the FROZEN compute_pair_weight_statistics
    // input shape (the SAME pairs the 7.4 fill consumes).
    let pairs: Vec<(usize, usize, f64)> = pair_i
        .iter()
        .zip(pair_j.iter())
        .zip(pair_weight.iter())
        .map(|((&i, &j), &w)| (i as usize, j as usize, w))
        .collect();

    for feature in 0..n_features {
        // bucket_of[obj] = the candidate feature's quantized bin (feature-major cindex).
        let mut bucket_of = vec![0usize; n];
        for obj in 0..n {
            let bin = cindex.get(feature * n + obj).copied().unwrap_or(0) as usize;
            if let Some(slot) = bucket_of.get_mut(obj) {
                *slot = bin;
            }
        }

        // der_sums[leaf=0][bucket] = the device-scattered der-sum row for this feature
        // (the bounded device descriptor, leaf 0). Shaped [leaf_count][bucket_count].
        let mut feat_der_sums = vec![vec![0.0_f64; bucket_count]; leaf_count];
        for bucket in 0..bucket_count {
            let v = der_sums_flat.get(feature * n_bins + bucket).copied().unwrap_or(0.0);
            if let Some(cell) = feat_der_sums.get_mut(0).and_then(|row| row.get_mut(bucket)) {
                *cell = v;
            }
        }

        // pair_weight_statistics[leaf][leaf][bucket] via the FROZEN cb_compute oracle over
        // the SAME pairs / bucket assignment (leaf_count == 1, the root).
        let pair_weight_statistics = cb_compute::compute_pair_weight_statistics(
            &pairs,
            leaf_count,
            bucket_count,
            &leaf_of,
            &bucket_of,
        )?;

        // The per-feature, per-border pairwise scores via the FROZEN host scorer (the small
        // per-leaf Cholesky solve runs here over the bounded assembled system).
        let feat_scores = cb_compute::calculate_pairwise_score(
            &feat_der_sums,
            &pair_weight_statistics,
            bucket_count,
            l2_diag_reg,
            pairwise_bucket_weight_prior_reg,
        )?;

        for border in 0..n_splits {
            let s = feat_scores.get(border).copied().unwrap_or(f64::NEG_INFINITY);
            if let Some(slot) = scores.get_mut(feature * n_splits + border) {
                *slot = s;
            }
        }
    }

    // (3) DEVICE: the deterministic best-split argmax (== select_best_candidate strict
    //     first-wins / lowest-index tie-break) over the host-solved scores, threaded
    //     through the device select_best_split_kernel so the selection is device-resident
    //     (only the O(1) winner crosses back). Empty candidate set -> no best.
    let best = if n_candidates == 0 {
        None
    } else {
        select_best_split_over_scores(client, &scores, n_splits)?
    };

    Ok(PairwiseSplitScore {
        scores,
        der_sums: der_sums_flat,
        best,
    })
}

/// Run the device [`select_best_split_kernel`] over a host-solved per-candidate score
/// vector and finish the O(blocks) across-block argmax host-side with the SAME
/// lowest-index tie-break (== `select_best_candidate`). Returns the winning
/// [`BestSplit`] (`feature_id = candidate / n_splits`, `bin_id = candidate % n_splits`),
/// or `None` if no candidate beat the sentinel. A read-back failure surfaces
/// [`CbError::Degenerate`] (WR-05), never a silent zero buffer.
fn select_best_split_over_scores(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    scores: &[f64],
    n_splits: usize,
) -> CbResult<Option<BestSplit>> {
    let n_candidates = scores.len();
    if n_candidates == 0 {
        return Ok(None);
    }
    let n_candidates_u32 = u32::try_from(n_candidates).map_err(|_| {
        CbError::OutOfRange(format!("n_candidates ({n_candidates}) exceeds u32 (argmin axis)"))
    })?;

    // A SINGLE cube of CUBE_DIM units strides over all candidates and block-reduces to one
    // winner (the FROZEN pointwise argmin geometry).
    let num_cubes = 1usize;
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    #[cfg(feature = "wgpu")]
    let (best_gain_handle, best_idx_handle) = {
        let scores_f32: Vec<f32> = scores.iter().map(|&v| v as f32).collect();
        let scores_h = client.create(cubecl::bytes::Bytes::from_elems(scores_f32));
        let best_gain_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; num_cubes]));
        let best_idx_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; num_cubes]));
        select_best_split_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(scores_h, n_candidates) },
            unsafe { ArrayArg::from_raw_parts(best_gain_h.clone(), num_cubes) },
            unsafe { ArrayArg::from_raw_parts(best_idx_h.clone(), num_cubes) },
            n_candidates_u32,
        );
        (best_gain_h, best_idx_h)
    };

    #[cfg(not(feature = "wgpu"))]
    let (best_gain_handle, best_idx_handle) = {
        let scores_h = client.create(cubecl::bytes::Bytes::from_elems(scores.to_vec()));
        let best_gain_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; num_cubes]));
        let best_idx_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; num_cubes]));
        select_best_split_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(scores_h, n_candidates) },
            unsafe { ArrayArg::from_raw_parts(best_gain_h.clone(), num_cubes) },
            unsafe { ArrayArg::from_raw_parts(best_idx_h.clone(), num_cubes) },
            n_candidates_u32,
        );
        (best_gain_h, best_idx_h)
    };

    let best_gains = read_scores_f64(client, best_gain_handle)?;
    let best_idx_bytes = client
        .read_one(best_idx_handle)
        .map_err(|e| CbError::Degenerate(format!("pairwise best-idx read-back failed: {e:?}")))?;
    let best_idxs: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&best_idx_bytes).to_vec();

    // Finish the across-block argmax. The device reduce (run in f32 on wgpu,
    // gpu_runtime.rs:4101) only nominates each block's winning candidate INDEX; the
    // actual comparison and tie-break MUST use the exact host-resident f64 `scores`
    // (WR-03 / IN-01), never the f32-collapsed device `gain`. Two host-distinct f64
    // scores can collapse to one f32 tie on-device and then resolve by index, picking a
    // different split than the CPU oracle's strict-`>` over f64. Re-resolving over f64
    // `scores` here removes that near-tie flip risk; `best_gains` is intentionally not
    // read for the comparison (it stays the device round-trip's structural artifact).
    // Highest score wins; on an EXACT f64 tie the LOWER candidate index wins (strict
    // first-wins parity == select_best_candidate).
    let _ = &best_gains; // device-reduced gains: structural artifact only (see WR-03).
    let mut best_score = f64::NEG_INFINITY;
    let mut best_c = u32::MAX;
    for &cand in best_idxs.iter() {
        if (cand as usize) >= n_candidates {
            continue;
        }
        // Exact host f64 score for this nominated candidate.
        let score = match scores.get(cand as usize) {
            Some(&s) => s,
            None => continue,
        };
        let take = score > best_score || (score == best_score && cand < best_c);
        if take {
            best_score = score;
            best_c = cand;
        }
    }

    if (best_c as usize) < n_candidates && n_splits > 0 {
        let feature = (best_c as usize) / n_splits;
        let bin = (best_c as usize) % n_splits;
        Ok(Some(BestSplit {
            feature_id: feature as u32,
            bin_id: bin as u32,
            score: best_score as f32,
            gain: best_score as f32,
        }))
    } else {
        Ok(None)
    }
}

/// Grow ONE oblivious tree device-resident over the compile-time [`SelectedRuntime`]
/// using the PAIRWISE split scorer path (GPU-01 ranking slice). The pairwise sibling of
/// [`grow_oblivious_tree`]: per level it chains the FROZEN 7.4 4-channel pairwise fill +
/// the device der-sum scatter -> [`launch_pairwise_split_score`] (the per-leaf
/// linear-system build + the host Cholesky solve, RESEARCH Open Q3) -> ONE O(1)
/// [`BestSplit`] read-back -> the host integer split decision -> the loss-agnostic Plan-C
/// [`launch_partition_split_into`] (forward-bit doc-routing) -> [`launch_partition_update_into`]
/// (per-partition Σ der1 / Σ weight reduce), over persistent device handles threaded
/// through ONE `ComputeClient`. At the leaves it reads back ONLY the `2^depth` part-stats
/// and computes leaf values via the FROZEN `cb_compute::calc_average`. The bulk pairwise
/// histogram / partition / doc-routing NEVER crosses to host per level (D-05).
///
/// # MVP scope (depth == 1) — the partition-aware forward dependency
///
/// The MVP grows a depth-1 pairwise stump (leaf_count == 1 at level 0, the root) — the
/// genuinely-complete vertical slice consistent with 07.5-03/04/05. The pairwise scorer's
/// per-leaf system is `2*leaf_count == 2` at the root. A `depth > 1` tree scores each
/// level over the CURRENT 2^level partitions (the partition-aware der-sum + pair-weight
/// assembly), the SAME forward dependency the pointwise [`grow_oblivious_tree`] surfaces;
/// `depth > 1` returns a typed [`CbError::OutOfRange`] until it lands (documented, NOT
/// silently cut). PairLogit/ranking, foldCount == 1, Plain.
///
/// # Inputs
///
/// `der1` (the pairwise-weighted first derivative, length `n`), `weight` (channel 1,
/// length `n`), `pair_i`/`pair_j`/`pair_weight` (the global competitor pairs), `cindex`
/// (feature-major quantized bins, length `n_features * n`), `indices` (object visiting
/// order, length `n`), `n_bins` (`1 << bits`, bits in {5,6,7}), `n_features`, `depth`,
/// `l2_diag_reg` (raw `l2_leaf_reg`), `pairwise_bucket_weight_prior_reg`
/// (`bayesian_matrix_reg`, default `0.1`), `scaled_l2` (the per-tree leaf-value scaling),
/// `one_hot`.
///
/// # Errors
///
/// [`CbError::OutOfRange`] if `depth > 1` or a dimension overflows; [`CbError::Degenerate`]
/// if a level finds no candidate split or a read-back fails (never a silent zero buffer);
/// the FROZEN 7.4 guards (via the fill / score / partition launches). No
/// `unwrap`/`expect`/`panic`/indexing.
#[allow(clippy::too_many_arguments)]
pub fn grow_oblivious_tree_pairwise(
    der1: &[f64],
    weight: &[f64],
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
    depth: usize,
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
    scaled_l2: f64,
    one_hot: bool,
) -> CbResult<GrownTree> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    grow_oblivious_tree_pairwise_into(
        &client,
        der1,
        weight,
        pair_i,
        pair_j,
        pair_weight,
        cindex,
        indices,
        n_bins,
        n_features,
        depth,
        l2_diag_reg,
        pairwise_bucket_weight_prior_reg,
        scaled_l2,
        one_hot,
    )
}

/// The ONE pairwise grow-loop geometry (IN-02 — one place). Uploads the resident
/// der1/weight/cindex/indices/leaf_of handles ONCE onto `client`, runs the per-depth
/// pairwise launch chain over those persistent handles, and reads back ONLY the O(1)
/// BestSplit per level + the final `2^depth` part-stats. The caller owns the `client`
/// lifecycle so every read-back uses the SAME client that allocated the handles.
#[allow(clippy::too_many_arguments)]
fn grow_oblivious_tree_pairwise_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1: &[f64],
    weight: &[f64],
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
    depth: usize,
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
    scaled_l2: f64,
    one_hot: bool,
) -> CbResult<GrownTree> {
    let n = der1.len();

    // Empty short-circuit (Pitfall 3/5).
    if n == 0 || n_features == 0 || n_bins == 0 || depth == 0 {
        return Ok(GrownTree {
            splits: Vec::new(),
            leaf_of: vec![0u32; n],
            leaf_values: Vec::new(),
            part_stats: Vec::new(),
        });
    }

    // MVP scope guard: the depth-1 pairwise stump scores over the single root leaf
    // (leaf_count == 1). A depth>1 level scores over 2^level partitions (the
    // partition-aware der-sum + pair-weight assembly) — the EXPLICIT tracked forward
    // dependency (the SAME class as the pointwise grow loop's depth>1 guard). Reject with
    // a typed error rather than fabricating a wrong-structure deep tree.
    if depth > 1 {
        return Err(CbError::OutOfRange(format!(
            "grow_oblivious_tree_pairwise supports depth <= 1 until the partition-aware \
             (per-leaf, leaf_count > 1) pairwise der-sum + pair-weight assembly lands (the \
             depth-1 stump scores the single root leaf; a depth>1 level scores over 2^level \
             partitions — the EXPLICIT tracked forward dependency); got depth = {depth}"
        )));
    }

    // 2^depth leaf count, overflow-checked.
    let n_leaves = 1usize
        .checked_shl(depth as u32)
        .ok_or_else(|| CbError::OutOfRange(format!("2^depth overflows usize (depth = {depth})")))?;

    let cindex_stride = n_features.checked_mul(n).ok_or_else(|| {
        CbError::OutOfRange(format!("n_features ({n_features}) * n ({n}) overflows usize"))
    })?;
    if cindex.len() != cindex_stride {
        return Err(CbError::LengthMismatch {
            column: "cindex".to_owned(),
            expected: cindex_stride,
            actual: cindex.len(),
        });
    }

    // Resident device handles uploaded ONCE (the D-05 persistent-buffer contract).
    let der1_h = upload_channel_floats(client, der1);
    let weight_h = upload_channel_floats(client, weight);
    let cindex_h = client.create(cubecl::bytes::Bytes::from_elems(cindex.to_vec()));
    let indices_h = client.create(cubecl::bytes::Bytes::from_elems(indices.to_vec()));
    let mut leaf_of_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; n]));

    let mut splits: Vec<(u32, u32)> = Vec::with_capacity(depth);

    for level in 0..depth {
        // (1) Device pairwise fill + der-sum scatter + score + deterministic argmax over
        //     the CURRENT (root) partition. The bulk 4-channel histogram + der-sum scatter
        //     stay device-resident IN the score launch; only the O(1) BestSplit crosses back.
        let scored = launch_pairwise_split_score_into(
            client,
            der1,
            pair_i,
            pair_j,
            pair_weight,
            cindex,
            indices,
            n_bins,
            n_features,
            l2_diag_reg,
            pairwise_bucket_weight_prior_reg,
            one_hot,
        )?;

        // (2) The O(1) host integer split decision. No candidate -> a degenerate dataset.
        let split = scored.best.ok_or_else(|| {
            CbError::Degenerate(format!(
                "grow_oblivious_tree_pairwise level {level}: no candidate split (degenerate)"
            ))
        })?;
        splits.push((split.feature_id, split.bin_id));

        // (3) Device partition-split (forward-bit doc-routing, level -> bit level == the CPU
        //     `leaf_index` convention) — IN-PLACE on device, NO read-back here (D-05).
        leaf_of_h = launch_partition_split_into(
            client,
            der1_h.clone(),
            cindex_h.clone(),
            indices_h.clone(),
            leaf_of_h,
            n,
            cindex_stride,
            split.feature_id,
            split.bin_id,
            level as u32,
        )?;
    }

    // (4) Device partition-update: the per-partition Σ der1 / Σ weight / Σ(der2·weight) reduce
    //     over the final 2^depth partitions, device-resident. Pairwise leaf estimation uses
    //     the `calc_average` (channels 0/1) arm — der2 = const -1 fills the (unused) hessian
    //     channel so the 3-channel launch is well-formed (GPUT-07 stride-3 part-stats).
    let der2_rmse_h = upload_channel_floats(client, &vec![-1.0_f64; n]);
    let part_stats_h = launch_partition_update_into(
        client,
        der1_h.clone(),
        weight_h.clone(),
        der2_rmse_h,
        indices_h.clone(),
        leaf_of_h.clone(),
        n,
        n_leaves,
    )?;

    // (5) ONE read-back of the 2^depth part-stats (the ONLY bulk-data crossing besides the
    //     O(1) per-level BestSplit — D-05). A read-back failure surfaces CbError::Degenerate.
    let part_stats = read_part_stats_f64(client, part_stats_h)?;

    // (6) Host leaf values via the FROZEN cb_compute::calc_average formula (count>0 guard).
    //     part_stats is stride-3 [Σder1, Σweight, Σ(der2·weight)]; pairwise reads channels 0/1.
    let mut leaf_values = vec![0.0_f64; n_leaves];
    for leaf in 0..n_leaves {
        let sum = part_stats.get(leaf * 3).copied().unwrap_or(0.0);
        let cnt = part_stats.get(leaf * 3 + 1).copied().unwrap_or(0.0);
        if let Some(slot) = leaf_values.get_mut(leaf) {
            *slot = cb_compute::calc_average(sum, cnt, scaled_l2);
        }
    }

    // (7) The per-object leaf assignment (the SC-3 structure observation): the grow loop
    //     never reads the bulk routing back (D-05); this single read-back at the END is
    //     the oracle seam (the same crossing class as the final part-stats).
    let leaf_of = read_u32_handle(client, leaf_of_h)?;

    Ok(GrownTree {
        splits,
        leaf_of,
        leaf_values,
        part_stats,
    })
}
