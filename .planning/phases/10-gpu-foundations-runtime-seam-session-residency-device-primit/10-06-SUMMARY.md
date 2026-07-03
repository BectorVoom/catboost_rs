---
phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit
plan: 06
subsystem: infra
tags: [cubecl, gpu, rocm, cindex, bit-packed, tcfeature, read-bin, compressed-index, gput-15]

# Dependency graph
requires:
  - phase: 10-05 (bit-compression + update_part_props)
    provides: bit_pack_layout ceil(log2) sizing math + the (word>>Shift)&Mask read idiom (unpack_bins_kernel) the cindex accessor mirrors; the checked-arithmetic host-helper discipline
  - phase: 7.3 (pointwise histogram seam)
    provides: the pointwise_hist2 nonbinary/half_byte/binary kernel family + the ONE launch_pointwise_hist2_into geometry + host_reference_hist2 the migration retargets
  - phase: 7.5 (host-light grow loop)
    provides: grow_oblivious_tree_into + partition_split_kernel/launch_partition_split_into the partition consumer migrates
provides:
  - TCFeature descriptor {Offset,Mask,Shift,FirstFoldIndex,Folds,OneHotFeature} + host grouped bit-packer pack_cindex (features sharing a 32-bit word by cumulative bit-width) + PackedCindex::device_arrays + feature_bits + read_bin_host (gpu_runtime/cindex.rs)
  - read_bin #[cube] accessor ((cindex[offset+obj]>>shift)&mask) — the ONE bin accessor every histogram/partition consumer routes through — + read_all_bins_kernel device exerciser (kernels.rs)
  - migrated pointwise_hist2 (nonbinary/half_byte/binary) + partition_split_kernel reading via read_bin over the bit-packed cindex; launch_pointwise_hist2_into host-packs-then-uploads-once
affects: [10-07 depth-1 leaf step (consumes the packed cindex + read_bin the grow loop now reads through), 10-09 Kaggle CUDA authoritative cindex bit-exact oracle]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "The ONE read_bin accessor `(cindex[Offset+obj]>>Shift)&Mask`: every histogram/partition bin read routes through it; the plain feature-major `cindex[feature*n+obj]` load is its DEGENERATE case (Offset=feature*n, Shift=0, Mask=0xFFFF_FFFF) — so a shared feature-major buffer needs no repack to route through TCFeature addressing"
    - "Grouped bit-pack cindex: greedily pack features whose cumulative bit-width <= 32 into ONE word per object (offset = group_index*n word base, shift = cumulative prior bits, mask = (1<<feature_bits)-1); feature_bits = ceil(log2(n_buckets)) — the BUCKET-count sizing (bins 0..n_buckets), distinct from bit_pack_layout's border-count+1 convention"
    - "Host-pack-then-upload-once (Open Q1 / A2): the launch wrapper packs the plain cindex internally and uploads the packed words + TCFeature arrays, so external wrapper signatures are UNCHANGED — the HIGH-ripple bin-extraction change lands inside the kernel bodies + the ONE launch geometry with zero blast radius to scorer/pairwise callers"

key-files:
  created:
    - crates/cb-backend/src/gpu_runtime/cindex.rs
    - crates/cb-backend/src/kernels/cindex.rs
  modified:
    - crates/cb-backend/src/kernels.rs
    - crates/cb-backend/src/gpu_runtime/mod.rs
    - crates/cb-backend/src/kernels/pointwise_hist.rs

key-decisions:
  - "Histogram consumes the GENUINE bit-packed cindex (pack_cindex groups features into shared words; e.g. two 8-bit features share one word, shifts 0/8). The packing runs INSIDE launch_pointwise_hist2_into (host-pack-then-upload-once, A2) so the plain-cindex-slice public wrapper signatures are unchanged — the score/pairwise callers at mod.rs:981/1253 are untouched."
  - "partition_split_kernel routes through read_bin in the DEGENERATE plain-layout form (Offset=feature*n, Shift=0, Mask=0xFFFF_FFFF) rather than a repacked resident buffer. Rationale: launch_partition_split_into is SHARED with the out-of-scope pairwise.rs + grow_loop.rs callers (pass a plain cindex handle); repacking the resident buffer would ripple beyond the plan's files_modified. The degenerate read IS TCFeature addressing (the plain feature's identity descriptor), satisfies must-have truth #2 (reads through the ONE read_bin accessor), and is byte-identical to the former load. Kernel body + launch-internal only — zero ripple."
  - "feature_bits sizes from the BUCKET count (ceil(log2(n_buckets)), bins 0..n_buckets) — NOT bit_pack_layout's n_bins+1 border convention. For the histogram n_bins IS the bucket count (256->8 bits, 16->4, 2->1), which is what the fill needs; reusing bit_pack_layout(n_bins) would over-size by one bit."
  - "Open Q1 (device-pack vs host-pack) RESOLVED as A2 host-pack-then-upload-once: borders/quantization are the CPU reference and stay host, so packing is a pure host transform of the already-quantized bins; the on-device WriteCompressedIndex kernel is an equivalent packing LOCATION reserved as a follow-up only if a phase needs packing without a host round-trip. Documented in cindex.rs."

patterns-established:
  - "read_bin is a bare #[cube] helper taking `&Array<u32>` (not just scalars) — confirmed it compiles + JITs on rocm; the ONE accessor is inlined at every histogram/partition/materialize call site."
  - "host_reference_hist2 extracts via read_bin_host over pack_cindex so the ground-truth reference agrees cell-for-cell with the packed device path; independence is preserved because the SEPARATE kernels::cindex oracle proves pack_cindex∘read_bin == source bins bit-exactly (source bins are its ground truth)."

requirements-completed: [GPUT-15]

# Metrics
duration: ~11min
completed: 2026-07-03
status: complete
---

# Phase 10 Plan 06: Bit-Packed Compressed Index (cindex) + read_bin Accessor Summary

**Replaced the plain one-u32-per-cell cindex with the upstream bit-packed grouped `WriteCompressedIndex` layout addressed by a per-feature `TCFeature{Offset,Shift,Mask,...}` descriptor (GPUT-15): a host grouped bit-packer (`pack_cindex`, features sharing 32-bit words by cumulative bit-width) proven bit-exact against the CPU quantized bins by a separate device oracle, and the ONE `read_bin` `#[cube]` accessor `(cindex[Offset+obj]>>Shift)&Mask` every histogram/partition consumer now routes through — the pointwise_hist2 family consumes the genuine packed layout (packed host-side, uploaded once inside the ONE launch geometry so external signatures are unchanged), the grow-loop partition split reads via read_bin, and the extracted bin VALUE is unchanged so the bin->border join is identical. rocm gfx1100 in-env 44/44 green.**

## Performance
- **Duration:** ~11 min
- **Completed:** 2026-07-03
- **Tasks:** 2
- **Files:** 5 (2 created, 3 modified)

## Accomplishments
- **`TCFeature` + grouped host bit-packer** (`gpu_runtime/cindex.rs`): `TCFeature{offset:u64, mask, shift, first_fold_index, folds, one_hot_feature}` + `pack_cindex` greedily groups features whose cumulative bit-width fits a 32-bit word into ONE word per object (offset = group_index*n word base, shift = cumulative prior bits, mask = `(1<<feature_bits)-1`); `feature_bits = ceil(log2(n_buckets))`. Every product / word-count / bit-width step is `checked_*` → `CbError::OutOfRange`; `bins.len() != n_features*n` → `LengthMismatch`; an out-of-range bin is rejected before it is masked into a shared word. `PackedCindex::device_arrays` emits the device-ready per-feature (offsets, shifts, masks) u32 arrays with a checked u32 offset cast. No slice indexing (D-13 — iterator/`chunks_exact`/`get_mut` throughout).
- **`read_bin` accessor + `read_all_bins_kernel`** (`kernels.rs`): the ONE `#[cube]` bin accessor `(cindex[offset+obj]>>shift)&mask` (mirrors `unpack_bins_kernel`), plus the device exerciser that materializes the full feature-major matrix by reading every cell through `read_bin`.
- **Bit-exact cindex self-oracle** (`kernels/cindex.rs`, D-02): pack a random-seeded bin matrix (mixed per-feature widths, multiple features per word, large n=10000), read back every bin via `read_bin` on device, assert INTEGER equality with the CPU source bins (tighter than ≤1e-4, D-07); plus grouping-metadata assertions (two 8-bit features share one word at shifts 0/8; a mixed-width run spills to a new word column at the right feature) and `feature_bits`/overflow-guard tests. rocm 5/5.
- **HIGH-ripple consumer migration** (`kernels.rs` + `gpu_runtime/mod.rs`): the pointwise_hist2 nonbinary/half_byte/binary kernels now take the packed cindex + per-feature offsets/shifts/masks and extract via `read_bin` (the redundant `& 15`/`& 1` nibble/bit masks dropped — `masks[feature]` carries the exact field width); `launch_pointwise_hist2_into` host-packs the plain cindex once and uploads the packed words + TCFeature arrays instead of the plain layout. `partition_split_kernel` reads via `read_bin` over the split feature's degenerate TCFeature (offset=feature*n, shift=0, mask=0xFFFF_FFFF), computed + range-checked in `launch_partition_split_into`. External wrapper signatures UNCHANGED → zero ripple to the scorer / pairwise / grow-loop callers.
- **Reference agreement** (`kernels/pointwise_hist.rs`): `host_reference_hist2` extracts bins via `read_bin_host` over the same `pack_cindex` layout, so the ordered ground-truth reference agrees cell-for-cell with the packed device path.
- **rocm gfx1100 in-env:** 44/44 green — the migrated histogram family (pointwise_hist nonbinary/half_byte/binary/whole_family/handoff/f32), the cindex oracle (5), and every partition-split consumer (grow_loop single/multi/partition/pairwise, score_split, pairwise_hist) still match the CPU references EXACTLY; no regressions.

## Task Commits
1. **Task 1: TCFeature + host packer + read_bin accessor + bit-exact oracle** — `e4dc311` (feat)
2. **Task 2: Migrate every cindex consumer to the read_bin accessor** — `fa7ce14` (feat)

## Files Created/Modified
- `crates/cb-backend/src/gpu_runtime/cindex.rs` — TCFeature, PackedCindex, pack_cindex, device_arrays, feature_bits, read_bin_host + Open-Q1 resolution doc (created).
- `crates/cb-backend/src/kernels/cindex.rs` — bit-exact pack->read self-oracle over read_all_bins_kernel (created).
- `crates/cb-backend/src/kernels.rs` — added read_bin #[cube] accessor + read_all_bins_kernel + `mod cindex;`; migrated pointwise_hist2 nonbinary/half_byte/binary + partition_split_kernel to read via read_bin.
- `crates/cb-backend/src/gpu_runtime/mod.rs` — `pub(crate) mod cindex;`; launch_pointwise_hist2_into host-packs + uploads the packed cindex; launch_partition_split_into computes the checked split offset/shift/mask.
- `crates/cb-backend/src/kernels/pointwise_hist.rs` — host_reference_hist2 extracts via read_bin_host over pack_cindex.

## Deviations from Plan
None functional — plan executed as written. Two scope-clarifying notes (documented above as key-decisions, not correctness deviations):

1. **partition_split routes through read_bin in the DEGENERATE plain-layout form**, not a repacked resident buffer. The plan's Task 2 says "consume the bit-packed cindex + TCFeature table" for the grow_oblivious_tree partition read; a genuine resident repack would ripple into `launch_partition_split_into`'s out-of-scope callers (`pairwise.rs`, `grow_loop.rs`), which the plan's `files_modified` does not include. The degenerate read IS TCFeature addressing (the plain feature's identity descriptor: Offset=feature*n, Shift=0, Mask=0xFFFF_FFFF), satisfies must-have truth #2 (every partition consumer reads through the ONE read_bin accessor), and is byte-identical to the former load. The HISTOGRAM (the plan's primary HIGH-ripple target) consumes the GENUINE grouped bit-packed layout.
2. **The histogram packs INSIDE the launch wrapper** (host-pack-then-upload-once) rather than exposing a packed-cindex public API. This keeps `launch_pointwise_hist2*`'s slice-based signatures unchanged so the score/pairwise callers are untouched — the bin-extraction change lands in the kernel bodies + the ONE launch geometry with zero external blast radius (the plan's IN-02 "one place" discipline).

## Known Stubs
None — the packer + accessor + all consumers are fully wired and oracle-verified. `TCFeature.first_fold_index` / `folds` / `one_hot_feature` are carried per the plan's frozen descriptor contract (for the multi-group fold offset + one-hot split routing later phases consume) and marked `#[allow(dead_code)]` until read; `read_bin_host` is `#[allow(dead_code)]` (consumed only by the `#[cfg(test)]` histogram reference + cindex oracle). These are intentional forward-contract fields, not placeholder data paths.

## Threat Flags
None beyond the plan's `<threat_model>`. T-10-15 (silent wrong bins from partial migration) mitigated: every production histogram/partition consumer routes through the ONE read_bin accessor (grep-confirmed — the only remaining `cindex[` occurrences in the migrated kernels are doc-comments), and the packed layout is bit-exact-oracled BEFORE any consumer wiring. T-10-16 (address overflow) mitigated by checked_mul/checked_add/checked u32 casts in pack_cindex + device_arrays + the partition offset. T-10-17 (portability UB) mitigated by no `-inf` literal + rocm smoke green after every `#[cube]` change.

## Next Phase Readiness
- 10-07 (depth-1 leaf step) consumes the packed cindex + read_bin the grow loop now reads through; `update_part_props` (10-05) provides the 2^depth part-stats read-back.
- Human-gated acceptance still open (per plan): Kaggle CUDA authoritative cindex bit-exact oracle via the 10-09 bench harness — not in-CI.

## Self-Check: PASSED
- Files: `gpu_runtime/cindex.rs`, `kernels/cindex.rs`, `kernels.rs`, `gpu_runtime/mod.rs`, `kernels/pointwise_hist.rs` — all FOUND.
- Commits: `e4dc311`, `fa7ce14` — both FOUND in git log.
- Acceptance: `read_bin` present in kernels.rs; `TCFeature`/`pack_cindex` present in gpu_runtime/cindex.rs; no plain `cindex[feature*n+obj]` bin-extraction remains in the migrated kernels (all routed through read_bin — grep-confirmed).
- rocm gfx1100 in-env: `cargo test -p cb-backend --no-default-features --features rocm -- pointwise_hist cindex grow_loop score_split pairwise` → 44 passed, 0 failed.

---
*Phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit*
*Completed: 2026-07-03*
