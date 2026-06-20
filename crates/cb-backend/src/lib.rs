#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]
//! `cb-backend` — the sole feature-gated runtime-alias owner (D-02). As of
//! Phase 3 (D-01) the `cpu` arm stands up the real CubeCL `CpuRuntime` and owns
//! the `#[cube]` kernels (D-03 keeps `cubecl` out of `cb-compute`). As of Phase
//! 7.1 (D-7.1-01) the `wgpu`/`cuda`/`rocm` arms resolve to the real CubeCL GPU
//! runtimes (`WgpuRuntime`/`CudaRuntime`/`HipRuntime`) and the `#[cube]` kernels
//! + the `gpu_runtime` launch helpers compile under every backend. Selection is
//! purely compile-time — there is no runtime `match` over backends.

/// First-class `#[cube]` kernels (D-01/D-03). Elementwise loss kernels do only
/// order-independent per-element work; the Phase-7.1 `block_reduce_kernel`
/// (D-7.1-04..09) reduces on-device but emits one partial per cube, leaving the
/// parity-critical FINAL fold to the host via `cb-core::sum_f64` (D-02/D-05/D-06).
/// The production bodies live in `kernels.rs`; the spike/oracle tests live in the
/// dedicated child modules (`kernels::gradient`, `kernels::scatter`,
/// `kernels::reduce`). Mounted under ALL backends (D-7.1-01): the GPU primitives
/// must build/run on wgpu/cuda/rocm, not only cpu.
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub mod kernels;

// `gpu_runtime` (the Phase-7.1 generic launch helpers over `SelectedRuntime`,
// D-7.1-04) is mounted in Task 2 alongside its file + the reduce kernel/oracle,
// with the same all-backend gate predicate as `kernels` above.

/// The CubeCL `CpuRuntime` as `cb-compute`'s abstract `Runtime` (D-01/D-03):
/// launches the elementwise `#[cube]` kernels and returns UN-reduced per-object
/// buffers for the host to fold (D-02). The GPU runtimes implement the same
/// trait additively in Phase 7.
#[cfg(feature = "cpu")]
pub mod cpu_runtime;

#[cfg(feature = "cpu")]
pub use cpu_runtime::CpuBackend;

#[cfg(all(test, feature = "cpu"))]
mod cpu_runtime_test;

/// Compile-time-selected runtime alias. One `cfg` arm per backend feature. Under
/// `cpu` this is CubeCL's `CpuRuntime` (D-01); the GPU arms replace `()` in
/// Phase 7. Selection is purely compile-time (D-02) — no runtime `match`.
#[cfg(feature = "cpu")]
pub type SelectedRuntime = cubecl::cpu::CpuRuntime;

/// `wgpu` backend (GPU-04, D-7.1-01): CubeCL's `WgpuRuntime`. Builds and runs on
/// dev machines with no ROCm/CUDA toolchain. The mutual-exclusion `not(...)` chain
/// is preserved verbatim — only the RHS changed from `()`.
#[cfg(all(feature = "wgpu", not(feature = "cpu")))]
pub type SelectedRuntime = cubecl::wgpu::WgpuRuntime;

/// CUDA backend (GPU-05, D-7.1-01): CubeCL's `CudaRuntime`. Compile-gated only
/// (no NVIDIA hardware in-env, D-07). The mutual-exclusion `not(...)` chain is
/// preserved verbatim — only the RHS changed from `()`.
#[cfg(all(feature = "cuda", not(feature = "cpu"), not(feature = "wgpu")))]
pub type SelectedRuntime = cubecl::cuda::CudaRuntime;

/// ROCm backend (GPU-01/GPU-02, D-7.1-01): CubeCL's `HipRuntime` (the cubecl
/// facade names the AMD/ROCm runtime `hip`). The in-env gfx1100 (wave32) oracle
/// path. The mutual-exclusion `not(...)` chain is preserved verbatim — only the
/// RHS changed from `()`.
#[cfg(all(feature = "rocm", not(feature = "cpu"), not(feature = "wgpu"), not(feature = "cuda")))]
pub type SelectedRuntime = cubecl::hip::HipRuntime;
