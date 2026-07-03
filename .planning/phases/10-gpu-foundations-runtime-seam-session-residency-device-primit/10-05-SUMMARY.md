---
phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit
plan: 05
subsystem: infra
tags: [cubecl, gpu, rocm, bit-compression, pack-unpack, update-part-props, deterministic-reduction, gput-16]

# Dependency graph
requires:
  - phase: 10-03 (reduce family)
    provides: deterministic reduce winner (fixed-order f64 tree reduce, BLOCK_REDUCE_SHMEM, CUBE_DIM_X stride), segmented_reduce_kernel structural template, cb_core::sum_f64 ordered baseline
  - phase: 10-04 (sort / partitions)
    provides: TDataPartition {Offset,Size} contiguous partition boundary arrays (part_offsets) that update_part_props consumes; the u32-key #[cube] + phantom-float precedent (update_partition_sizes_kernel)
provides:
  - pack_bins_kernel / unpack_bins_kernel (bit-compression pack/unpack, u32 keys, comptime bits_per_key/keys_per_word/mask; one-thread-per-word race-free pack; bit-exact round-trip)
  - bit_pack_layout host helper (bits_per_key=ceil(log2(n_bins+1)), keys_per_word=32/bits_per_key, mask, num_words; checked_add/checked_shl/checked_mul -> CbError::OutOfRange)
  - update_part_props_kernel (per-partition Σstat over part_offsets, f64 accumulation, fixed-order tree reduce — deterministic; writes 2^depth partition totals)
affects: [10-06 cindex (consumes bit_pack_layout + read_bin math), 10-07 depth-1 leaf step (consumes update_part_props part-stat read-back)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "One-thread-per-output-word bit-pack (each thread loops over keys_per_word fields into a PRIVATE accumulator) => no cross-lane |= race => deterministic, order-independent"
    - "Host-side bit_pack_layout computes comptime pack geometry with fully checked arithmetic; the ONLY validated host->device numeric surface for the pack (T-10-12)"
    - "update_part_props = segmented reduce specialized to the 2^depth partition boundary array, reusing the 10-03 fixed-order f64 tree-reduce winner verbatim (no new finalize)"

key-files:
  created:
    - crates/cb-backend/src/kernels/compression.rs
    - crates/cb-backend/src/kernels/update_part_props.rs
  modified:
    - crates/cb-backend/src/kernels.rs

key-decisions:
  - "Bit-pack packs a SINGLE ≤8-bit bin column into keys_per_word fields per 32-bit word (the compression primitive the cindex consumes); 'multiple features sharing a word' is demonstrated by keys_per_word>1 — each slot is a distinct field extracted by its own Shift. The grouped multi-FEATURE cindex layout (TCFeature Offset/Shift) is 10-06's job; this plan ships the packing math it builds on."
  - "pack is one-thread-per-WORD (loops the keys_per_word fields) rather than one-thread-per-KEY with a shared |= : the per-word ownership makes the OR race-free and deterministic without atomics (upstream compression.cu BLOCK_SIZE-strided packing)."
  - "update_part_props reuses the 10-03 fixed-order f64 tree reduce (the shipped deterministic winner) rather than the fixed-point u64 atomic path: one cube per partition has no cross-cube contention, so the tree reduce is already deterministic with zero run-to-run spread."

patterns-established:
  - "Integer-primitive #[cube] carries an <F: Float> phantom (`let _ = F::new(0.0)`) purely for launch-signature uniformity (mirrors update_partition_sizes_kernel); the body is pure u32."

requirements-completed: [GPUT-16]

# Metrics
duration: ~30min
completed: 2026-07-03
status: complete
---

# Phase 10 Plan 05: Bit-Compression (pack/unpack) + Per-Partition Stat Aggregation (update_part_props) Summary

**The final two GPUT-16 primitives: a from-scratch CubeCL bit-compression pack/unpack (one-thread-per-word race-free packing of an ≤8-bit bin column into shared 32-bit words, comptime `bits_per_key`/`keys_per_word`/`mask` from a checked `bit_pack_layout` host helper, bit-exact round-trip) that feeds the 10-06 bit-packed cindex, and `update_part_props` (per-partition Σstat over the 10-04 partition boundary array, reusing the 10-03 fixed-order f64 tree-reduce winner) that feeds the 10-07 depth-1 leaf step — both self-oracled against inline serial references and green on rocm gfx1100 in-env (8/8).**

## Performance
- **Duration:** ~30 min
- **Completed:** 2026-07-03
- **Tasks:** 2
- **Files:** 3 (1 modified, 2 created)

## Accomplishments
- **Bit-compression pack/unpack** (`pack_bins_kernel` / `unpack_bins_kernel`): pack ORs `(bins[key] & Mask) << (slot * bitsPerKey)` with ONE thread owning each output word (loops its `keys_per_word` fields into a private accumulator — no cross-lane `|=` race, deterministic); unpack extracts key `i`'s field via `(words[i/keysPerWord] >> ((i%keysPerWord)*bitsPerKey)) & Mask`. `bits_per_key`/`keys_per_word`/`mask` are `#[comptime]` (JIT-resolved bit geometry).
- **`bit_pack_layout` host helper**: `bits_per_key = ceil(log2(n_bins+1))` (clamped `1..=32`), `keys_per_word = 32/bits_per_key`, `mask = (1<<bits_per_key)-1`, `num_words = ceil(n/keys_per_word)`. Every overflow-capable step (`n_bins+1`, the mask shift, the word-count reconstruction) is `checked_add`/`checked_shl`/`checked_mul` → `CbError::OutOfRange` (T-10-12 — the only validated host→device numeric surface for the pack).
- **`update_part_props_kernel`**: one cube per partition over `part_offsets` (`num_parts+1` contiguous boundary array from 10-04), each lane widens `stat[i]` to **f64** and folds a `CUBE_DIM_X`-strided slice, then the 10-03 fixed-order shared-mem tree reduce combines lanes → the `2^depth` partition totals (`Σder1`/`Σweight`) the leaf step reads back. Empty partition (`start==end`) → `0`.
- **Bit-exact compression self-oracle** (`kernels/compression.rs`, D-02): inline serial pack/unpack reference; behaviour example (four 2-bit keys in one word → `39`), random ui8 columns across 8 widths × 4 seeds (`n=1000`), large-n full-byte width (`n=10000`, `keys_per_word=4`), and `bit_pack_layout` geometry + overflow-guard tests (`n_bins=u32::MAX` rejected). Integer equality — tighter than ≤1e-4 (D-07).
- **Ordered serial update_part_props self-oracle** (`kernels/update_part_props.rs`, D-02): derives object→partition from `part_offsets`, asserts every partition id in range (the `pointwise_hist` range discipline generalized), folds each partition via `cb_core::sum_f64` in ascending object order; behaviour example (`[3,12]`), empty-partition case (`[10,0,50]`), f32/f64 varied sizes (several `>CUBE_DIM` so the grid-stride intra-partition fold runs).
- **rocm gfx1100 in-env:** 8/8 green (4 compression + 4 update_part_props); no regressions.

## Task Commits
1. **Task 1: Bit-compression pack/unpack (bit-exact)** — `aaed896` (feat)
2. **Task 2: Per-partition stat aggregation (update_part_props)** — `20b1d43` (feat)

_Kernel + oracle committed together per task (the self-oracle cannot compile without the kernel it exercises — GPU-kernel TDD constraint; tdd_mode inactive for this phase, per 10-01/10-03 precedent). To keep each commit independently compilable, the Task-2 kernel body + its `mod` decl were added to `kernels.rs` only in the Task-2 commit (the Task-1 commit contains ONLY the compression additions)._

## Files Created/Modified
- `crates/cb-backend/src/kernels.rs` — added `pack_bins_kernel`, `unpack_bins_kernel`, `BitPackLayout` + `bit_pack_layout`, `update_part_props_kernel`; added `mod compression;` and `mod update_part_props;`.
- `crates/cb-backend/src/kernels/compression.rs` — bit-exact pack∘unpack self-oracle + `bit_pack_layout` guard tests (created).
- `crates/cb-backend/src/kernels/update_part_props.rs` — ordered serial per-partition Σstat self-oracle (created).

## Deviations from Plan
None — plan executed as written. Two scope-clarifying notes (not functional deviations):

1. **Pack is one-thread-per-WORD, not one-thread-per-KEY with a shared `|=`.** The plan action text reads "each thread ORs `(bins[i] & Mask) << Shift` into the shared word at its Offset". A literal cross-thread `|=` into a shared word is a data race (non-deterministic — violates the phase's deterministic-reduction mandate). The upstream `compression.cu` packs BLOCK_SIZE-strided with each word owned by one thread; that is what shipped — race-free and deterministic, same packed bytes. Documented in the kernel doc-comment.
2. **`bit_pack_layout` / `BitPackLayout` carry `#[allow(dead_code)]`.** They are consumed by the 10-05 self-oracle (test) now and by the 10-06 bit-packed cindex builder (production) next; the allow keeps the default (cpu, non-test) build warning-free until 10-06 wires the production consumer.

## Known Stubs
None — both primitives are fully wired and oracle-verified; no placeholder/mock data paths.

## Threat Flags
None beyond the plan's `<threat_model>`. T-10-12 (pack Offset/word-count tampering) mitigated by `checked_add`/`checked_shl`/`checked_mul` in `bit_pack_layout` + a `n_bins=u32::MAX` rejection test; T-10-13 (nondeterministic stat) mitigated by the 10-03 fixed-order f64 tree reduce (no float atomics); T-10-14 (portability UB) mitigated by no `-inf` literal + rocm smoke green after every `#[cube]` change.

## Next Phase Readiness
- The primitive library is COMPLETE (GPUT-16): scan/segmented-scan, reduce/segmented-reduce/reduce-by-key, sort/reorder, partitions, fill/gather/vector, bit-compression, and update_part_props all ship with green self-oracles.
- 10-06 (cindex) consumes `bit_pack_layout` + the `(word >> Shift) & Mask` read math directly (wire `bit_pack_layout` into the `TCFeature` builder and flip its `#[allow(dead_code)]`).
- 10-07 (depth-1 leaf step) consumes `update_part_props_kernel` for the `2^depth` `Σder1`/`Σweight` partition read-back.
- Human-gated acceptance still open (per plan): Kaggle CUDA authoritative pack (bit-exact) + update_part_props (≤1e-4) via the 10-09 bench harness — not in-CI.

## Self-Check: PASSED
- Files: `crates/cb-backend/src/kernels.rs`, `crates/cb-backend/src/kernels/compression.rs`, `crates/cb-backend/src/kernels/update_part_props.rs` — all FOUND.
- Commits: `aaed896`, `20b1d43` — all FOUND.
- Acceptance: `update_part_props` present in kernels.rs; `pack_bins_kernel`/`unpack_bins_kernel`/`bit_pack_layout` present.
- rocm gfx1100 in-env: `cargo test -p cb-backend --no-default-features --features rocm -- compression update_part_props` → 8 passed, 0 failed.

---
*Phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit*
*Completed: 2026-07-03*
