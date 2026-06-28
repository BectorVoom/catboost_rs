---
phase: 08-python-bindings-dual-api-packaging
plan: 08
subsystem: gpu-backend-genericization
tags: [gpu, generics, cubecl, runtime, backend-selection, rocm, wgpu, cuda, maturin, wheels]
gap_closure: true
requires:
  - "cb-backend SelectedRuntime alias (cpu/wgpu/cuda/rocm) — lib.rs:45-65"
  - "Phase-7.2 der seam (launch_der_binary/launch_der_unary/launch_der_param/const_der_handle)"
  - "cb_train::train<R: Runtime> (already generic)"
  - "08-07 maturin/abi3 packaging + pyproject-rocm.toml"
provides:
  - "GpuBackend: a single zero-sized cb_compute::Runtime impl generic over SelectedRuntime (one path for wgpu/cuda/rocm)"
  - "feature-gated facade backend selection (CpuBackend for cpu, GpuBackend for wgpu/cuda/rocm)"
  - "catboost-rs-py wgpu/cuda feature passthrough"
  - "in-env rocm wheel (catboost_rs_rocm-0.1.0-cp312-abi3-manylinux_2_39_x86_64.whl) + bit-exact GPU parity"
affects:
  - "crates/cb-backend (new gpu_backend module)"
  - "crates/catboost-rs builder fit() path"
  - "crates/catboost-rs-py feature surface"
tech-stack:
  added: []
  patterns:
    - "compile-time backend selection by Cargo feature (no runtime match)"
    - "GPU der computed via the reused Phase-7.2 der seam (no new #[cube] kernels)"
    - "typed-CbError rejection for losses without a GPU der kernel (no silent fallback)"
key-files:
  created:
    - crates/cb-backend/src/gpu_backend.rs
    - crates/cb-backend/src/gpu_backend_test.rs
  modified:
    - crates/cb-backend/src/lib.rs
    - crates/catboost-rs/src/builder.rs
    - crates/catboost-rs-py/Cargo.toml
    - crates/catboost-rs-py/pyproject-rocm.toml
decisions:
  - "GpuBackend self-oracles against the cb-compute::loss host baseline (CpuBackend is cpu-gated and cannot co-compile under rocm), within the <=1e-4 Phase-7 tolerance"
  - "GPU MVP loss set = exactly the Phase-7.2 der-kernel-backed losses: Rmse, Logloss/CrossEntropy, Mae, Quantile, Focal; all others typed-rejected"
  - "rocm wheel requires ROCM_PATH + LD_PRELOAD of system libhiprtc/libamdhip64 at runtime (wheel-env, not a code defect)"
metrics:
  duration: "~7 min"
  completed: 2026-06-23
  tasks: 2
  files-created: 2
  files-modified: 4
---

# Phase 8 Plan 08: Generic GPU Backend + ROCm Wheel Summary

Made the facade train path generic over the CubeCL backend by introducing a single zero-sized `GpuBackend` that implements `cb_compute::Runtime` over `SelectedRuntime` via the existing Phase-7.2 der seam, feature-gated the facade's backend selection (CpuBackend vs GpuBackend), added wgpu/cuda passthrough to the Python crate, then built and bit-exactly validated the rocm wheel in-env on gfx1100 — discharging the rocm-wheel deliverable 08-07 deferred.

## What Shipped

- **`GpuBackend` (one impl, all GPU backends).** A zero-sized `pub struct GpuBackend` in `crates/cb-backend/src/gpu_backend.rs` implementing the 2-method `cb_compute::Runtime` trait. `compute_gradients` mirrors `CpuBackend`'s shape validation (zero-dim reject, non-divisible-length reject, per-object `target.len()==n` check, empty short-circuit) and per-dimension outer launch loop, but routes the per-loss math to the Phase-7.2 der seam over `SelectedRuntime`. No concrete runtime is named — wgpu/cuda/rocm all flow through the one impl. `compute_gradients_grouped` is inherited (the host-side default), not overridden.
- **GPU der loss support (MVP):** `Rmse` (binary grad + const `-1.0` der2), `Logloss`/`CrossEntropy` (binary grad + unary hessian), `Mae` and `Quantile` (param grad + const `0.0` der2), `Focal` (param grad + param hessian). Every other loss returns a typed `CbError::OutOfRange` naming it as not-yet-supported on the GPU backend (parity gap, not a bug) — no silent fallback, no panic. **No new `#[cube]` kernels authored.**
- **Feature-gated backend selection** in `crates/catboost-rs/src/builder.rs`: `#[cfg(feature="cpu")] CpuBackend` / `#[cfg(any(wgpu,cuda,rocm))] GpuBackend`, bound to a single `backend` local fed to the already-generic `cb_train::train`. No cpu-only symbol is referenced under a non-cpu build.
- **wgpu/cuda passthrough** added to `crates/catboost-rs-py/Cargo.toml` alongside cpu/rocm.
- **ROCm wheel built + validated in-env** (gfx1100 / ROCm 7.1.1): `catboost_rs_rocm-0.1.0-cp312-abi3-manylinux_2_39_x86_64.whl`.

## Verification

- **Compile gates (the core deliverable — generics):** `cargo check -p catboost-rs --no-default-features --features {cpu,wgpu,rocm,cuda}` ALL succeed. cuda passes the compile-gate too (cudarc loads the toolkit at runtime, not build time), so there is no toolkit-deferred gap — no failure named `CpuBackend` or any of our symbols under any backend.
- **cpu path byte-unchanged:** `cargo test -p catboost-rs-py --features cpu` green (29 passed).
- **One impl, no duplication:** a single `impl Runtime for GpuBackend` serves all three GPU backends (grep-verified).
- **cb-backend has NO cb-train dependency** (Cargo.toml verified) — the feature-unification landmine is preserved; GpuBackend lives in cb-backend beside the kernels.
- **In-env GPU unit tests (rocm, gfx1100):** the 5 new `gpu_backend_test` tests pass on real GPU — RMSE and Logloss der1/der2 match the `cb-compute::loss` host baseline within tolerance (observed bit-exact), unsupported-loss + zero-dim + empty cases return the typed errors. Full `cb-backend` rocm suite: 80 passed / 0 failed (was 75; +5 new), no regression.
- **In-env GPU train/predict parity (the rocm wheel oracle):** the cpu wheel and the rocm wheel, trained on identical data+params:
  - RMSE regression (20 iters, depth 3): predictions **bit-exact** (max_abs_diff = 0.000e+00).
  - Logloss classification (15 iters, depth 3): `predict_proba` **bit-exact** (max_abs_diff = 0.000e+00).
  - Both are 0.0, far inside the <=1e-4 Phase-7 GPU tolerance (D-04).

## ROCm Wheel — DISCHARGED

08-07's deferred "rocm wheel builds in-env" deliverable is **DISCHARGED**. Artifact: `target/wheels/catboost_rs_rocm-0.1.0-cp312-abi3-manylinux_2_39_x86_64.whl` (distribution `catboost-rs-rocm`, module `catboost_rs`, the D-08 two-distribution layout).

Build command (in-env, gfx1100 / ROCm 7.1.1; `cp pyproject-rocm.toml pyproject.toml` first, restore after):

```
cd crates/catboost-rs-py
maturin build --no-default-features --features rocm --release
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] maturin wheel-repair needs `patchelf`**
- **Found during:** Task 2 (first rocm `maturin build`).
- **Issue:** The compile succeeded but maturin's shared-lib bundling step failed: `Failed to execute 'patchelf'`.
- **Fix:** `pip install patchelf` into the build venv (a standard, well-known binary — not a package-legitimacy risk, so no human-verify checkpoint). Rebuild succeeded.
- **Files modified:** none (env install); documented in pyproject-rocm.toml.

**2. [Rule 3 - Blocking] rocm-wheel GPU `fit()` segfault in the bundled HIP JIT**
- **Found during:** Task 2 (GPU fit smoke under the installed rocm wheel).
- **Issue:** `fit()` segfaulted at 1 iter / depth 1 inside `cubecl_hip::compute::context::HipContext::compile_kernel` → `__strlen_evex` (a null JIT-log strlen). The wheel-bundled, patchelf-renamed `libhiprtc` loses the path resolution to its companion comgr/LLVM bitcode, so kernel JIT fails. The SAME kernels JIT cleanly under `cargo test --features rocm` — it is a wheel-runtime env issue, not a code defect.
- **Fix:** run the wheel against the SYSTEM ROCm libs so the JIT finds its data files: `ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm LD_PRELOAD=/opt/rocm/lib/libhiprtc.so.7:/opt/rocm/lib/libamdhip64.so.7 python ...`. With this, GPU fit/predict succeed and are bit-exact vs cpu. Documented the requirement in pyproject-rocm.toml (a forward packaging task is to bundle the comgr bitcode tree so the LD_PRELOAD is unnecessary).
- **Files modified:** crates/catboost-rs-py/pyproject-rocm.toml (runtime-env note).

### Plan-text adjustment (documented, not a code deviation)

- The plan's Task-1 test asks GpuBackend to "match CpuBackend". `CpuBackend` is `#[cfg(feature="cpu")]`-gated and cannot co-compile in a `--features rocm` test binary. The test instead self-oracles GpuBackend against the `cb-compute::loss` host baseline functions (`rmse_der1/der2`, `logloss_der1/der2`) — the SAME baseline the existing Phase-7.2 der-seam self-oracle uses — which is equivalent (CpuBackend launches kernels derived from those same baselines) and is the only buildable form under a non-cpu feature.

## Human-Authorization Gates (NOT actioned — recorded only)

Per the plan, no publish and no push/PR were performed. Recorded for the human:
- Publish (rocm distribution): `maturin upload target/wheels/catboost_rs_rocm-0.1.0-*.whl` (or `twine upload`) — NOT run.
- Push/PR: not run. Commits are local on `main`.

## Commits

- `3bef3da` feat(08-08): generic GpuBackend Runtime + feature-gated backend selection
- `6065215` feat(08-08): build + validate rocm wheel in-env (discharge 08-07 deferral)

## Known Stubs

None. The unsupported-loss arms are intentional typed-error parity gaps (documented in code + this summary), not stubs feeding empty data to a UI.

## Forward Dependencies (out of scope, documented)

- GPU der kernels for the remaining losses (multiclass, multilabel, ranking, smooth/positive-domain, RMSEWithUncertainty, custom) — these grow the 7.2 seam in later GPU phases; GpuBackend rejects them today.
- The device-resident grow loop (Phase-7 `grow_oblivious_tree`/`grow_boosting_pass`, depth>1 partition-aware histograms, Newton, GPU multiclass der) — 08-08 is GPU-DERIVATIVES through the proven host training loop, as Phase-8 CONTEXT scopes GPU kernel work out.
- Bundling the ROCm comgr bitcode tree into the rocm wheel so no `LD_PRELOAD`/`ROCM_PATH` is needed at runtime.

## Self-Check: PASSED

- FOUND: crates/cb-backend/src/gpu_backend.rs
- FOUND: crates/cb-backend/src/gpu_backend_test.rs
- FOUND: target/wheels/catboost_rs_rocm-0.1.0-cp312-abi3-manylinux_2_39_x86_64.whl
- FOUND: commit 3bef3da
- FOUND: commit 6065215
- FOUND: single `impl Runtime for GpuBackend`
- OK: cb-backend has no cb-train dependency
