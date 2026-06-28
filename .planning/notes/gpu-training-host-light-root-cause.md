---
title: GPU training is "host-light" — 20× slowdown is the MVP boundary, not a defect
date: 2026-06-28
context: /gsd-explore session investigating "GPU kernel training >20× slower than official"
---

# GPU training root cause: the tree-growth inner loop runs on the host CPU

The reported >20× slowdown vs official CatBoost GPU is **architectural, by design of the
current MVP** — not a bug. The GPU backend computes derivatives on-device and nothing
else; the entire greedy tree search runs serially on the host CPU while the GPU sits idle.
A device-resident grow loop was built in Phase 7.5 but **was never wired into the training
pipeline.**

## What runs where, per boosting iteration (verified by code read)

| Step | Where | Evidence |
|------|-------|----------|
| der1 / der2 | **DEVICE** ✅ | `crates/cb-backend/src/gpu_backend.rs:67-146` (the only `Runtime` trait method implemented) |
| read der1/der2 → host | sync stall | `crates/cb-train/src/boosting.rs:2996` `ders.der1.clone()` |
| histogram build | **HOST** ❌ | `crates/cb-train/src/tree.rs:609-635` nested `for feature → for border` |
| split scoring | **HOST** ❌ | `tree.rs:439-471` `reduce_leaf_stats()` CPU fold |
| BestSplit | **HOST** ❌ | `tree.rs:291-302` `select_best_candidate()` |
| partition / leaf assign | **HOST** ❌ | `tree.rs:384-395` `assign_leaves()` |
| leaf values | **HOST** ❌ | `tree.rs:1390-1447` |

So ~95% of training (the whole inner loop) is CPU. Choosing a GPU backend can be **slower
than pure CPU**, because it adds a per-tree device→host read-back without removing any host
work.

## The integration gap (the "design error")

- The boosting loop `cb_train::train::<R: Runtime>` is generic over `Runtime`, but the
  `Runtime` trait only exposes `compute_gradients()`. There is **no trait seam** for
  "grow a tree on device," so `fit()` always falls through to the CPU
  `greedy_tensor_search_*` growers (`boosting.rs:3203-3345`).
- Entry wiring: `crates/catboost-rs/src/builder.rs:333-371` selects `GpuBackend` for
  `wgpu`/`cuda`/`rocm` but that backend only provides gradients.
- **A device-resident grow loop already exists** — Phase 7.5's `grow_boosting_pass()` at
  `crates/cb-backend/src/gpu_runtime/mod.rs:1890-2043`. It does per-level histogram +
  score + split on-device and reads back only `2^depth` leaf-stats per level. **It is only
  ever called from tests** (`kernels/grow_loop.rs`), never from `cb-train`. The fast path
  was built and left unwired.

## MVP limits of the existing device grower (Phase 7.5)

`grow_boosting_pass` is depth-1 only; no Newton der2; skips CTR / pairwise / ordered-boosting
/ multiclass paths. Wiring it as-is would only accelerate the simplest case.

## Verdict

Bottleneck = **(a) compute on host instead of device** (primary), with a minor
**(b) per-tree read-back sync**. There is **no** tiny-kernel-launch problem because no
training kernels launch at all. The fix is not a tuning pass — it is moving the inner loop
onto the device and exposing a `Runtime` grow-tree seam.

## Constraints for any fix

- In-env GPU is **AMD gfx1100 / ROCm only** (no CUDA here). CubeCL kernels are portable
  (cuda/rocm/wgpu from one source) → develop + validate correctness ≤1e-5 in-env on ROCm.
- Official CatBoost GPU is **CUDA-only**; head-to-head **speed** benchmark must run on
  NVIDIA hardware → user will run it on a **Kaggle CUDA notebook**.
- Landmine (memory `phase75-grow-loop-outcome`): never add a `cb-train` dependency to
  `cb-backend` — feature unification breaks the rocm runtime. Transcribe CPU refs inline.
