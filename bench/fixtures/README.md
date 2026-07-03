# `bench/fixtures/` — committed CUDA-oracle fixtures

These files are the **repo-committed, seeded inputs + CPU-path expected values** that
`bench/cuda_oracle.ipynb` loads on the Kaggle CUDA image to run the correctness gate
(BENCH-01). They are produced by a single seeded generator (`bench/generator.py`,
D-06) so the depth-1 `<=1e-5` correctness fixture and the large-n speed workload can
never drift apart.

## Commit discipline

- **Only the small-n correctness fixtures are committed.** The large-n speed workload
  (`SPEED_CONFIG`, ~1e6x50) is **regenerated on the fly** from its seed in the notebook
  — never committed (it would be ~200 MB and adds nothing a seed cannot reproduce).
- **`manifest.json` is the contract.** It records the configs, seeds, and the `sha256`
  of every committed fixture. Regenerate + verify with:

  ```bash
  python3 bench/generator.py --write bench/fixtures   # (re)emit
  python3 bench/generator.py --check bench/fixtures   # verify byte-for-byte reproduce
  ```

  `--check` regenerates into a temp dir and diffs the shas against `manifest.json`.
  A mismatch means the generator changed: regenerate, re-review the diff, and commit
  the new fixtures **and** manifest together.
- **Determinism guarantee.** `generator.py` uses the legacy `numpy.random.RandomState`
  (Mersenne-Twister), whose stream is stable across numpy versions — so these bytes
  reproduce on the Kaggle image regardless of its numpy version. Do **not** switch to
  `numpy.random.default_rng`; that would break cross-version reproducibility.

## Fixture format

All arrays are `numpy.save` `.npy` (self-describing dtype + shape). Depth-1 tree
references are JSON.

### Correctness inputs

| File | Shape / dtype | Meaning |
|------|---------------|---------|
| `X_small.npy` | `(2000, 10)` f32 | depth-1 design matrix (`CORRECTNESS_CONFIG`) |
| `y_small_reg.npy` | `(2000,)` f32 | continuous regression target (RMSE oracle) |
| `y_small_bin.npy` | `(2000,)` f32 | `{0,1}` target (Logloss oracle) |
| `cindex_small.npy` | `(2000, 10)` i32 | quantized bins (`bin = #borders strictly exceeded`) |

### Standalone primitive references (bit-exact / `<=1e-4`)

| Input | Expected | Primitive |
|-------|----------|-----------|
| `prim_values.npy` (256 f64) | `expected_inclusive_scan.npy` | inclusive scan |
| `prim_values.npy` | `expected_exclusive_scan.npy` | exclusive scan |
| `prim_values.npy`, `prim_seg_heads.npy` | `expected_segmented_scan.npy` | segmented scan (resets at heads) |
| `prim_keys.npy` (reversed) | `expected_sort_perm.npy` | stable radix sort / reorder (bit-exact perm) |
| `prim_keys.npy`, `prim_values.npy` | `expected_reduce_by_key_{keys,vals}.npy` | reduce-by-key |
| `prim_values.npy`, `prim_offsets.npy` | `expected_segmented_reduce.npy` | segmented reduce |
| `cindex_small.npy[:, :1]` | `expected_cindex_packed_f0_bits8.npy` | bit-packed cindex (8-bit fields, GPUT-15) |

Float reduces accumulate in **float64** (mirroring the device f64 re-accumulation —
see `SPIKE-REDUCTION.md`); integer/index primitives are **bit-exact**.

### Depth-1 tree reference — `expected_depth1_tree.json`

First-order **`calc_average`** oblivious depth-1 stump (Cosine score, L2 = 3.0,
learning-rate 0.1), for both `rmse` and `logloss`. Each entry records
`best_feature`, `best_bin`, `best_border`, `score`, `leaf_left`, `leaf_right`.

> **Logloss is pinned to FIRST-ORDER `calc_average` leaves, not Newton der2.** The
> device depth-1 path computes first-order leaves; Newton (der2) is Phase 11 / GPUT-07
> (RESEARCH line 318, CONTEXT scope anchor). Getting this leaf method right is the
> single most likely reason a naive Logloss oracle would miss `<=1e-5`.

> **Bordering caveat.** The numpy reference uses uniform-quantile borders. This is a
> *self-consistent* reference for the standalone primitive/cindex fixtures — **not** a
> claim of bit-parity with CatBoost's GreedyLogSum. The **authoritative** depth-1 oracle
> in the notebook is **device-vs-Rust-CPU** (both use the Rust bordering); this numpy
> reference is a sanity anchor.
