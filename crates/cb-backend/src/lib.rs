#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]
//! `cb-backend` — the sole feature-gated runtime-alias owner (D-02). As of
//! Phase 3 (D-01) the `cpu` arm stands up the real CubeCL `CpuRuntime` and owns
//! the `#[cube]` kernels (D-03 keeps `cubecl` out of `cb-compute`); the
//! `wgpu`/`cuda`/`rocm` arms remain inert placeholders until Phase 7. Selection
//! is purely compile-time — there is no runtime `match` over backends.

/// First-class `#[cube]` kernels (D-01/D-03). Kernels do only order-independent
/// elementwise work; every parity-critical reduction is finalized host-side via
/// `cb-core::sum_f64` (D-02/D-05/D-06). The production body lives in `kernels.rs`;
/// its spike tests live in the dedicated `kernels_test.rs` file, declared from
/// `kernels.rs` as the child module `gradient` (mounted at `kernels::gradient`
/// so `cargo test kernels::gradient` selects them).
#[cfg(feature = "cpu")]
pub mod kernels;

/// Compile-time-selected runtime alias. One `cfg` arm per backend feature. Under
/// `cpu` this is CubeCL's `CpuRuntime` (D-01); the GPU arms replace `()` in
/// Phase 7. Selection is purely compile-time (D-02) — no runtime `match`.
#[cfg(feature = "cpu")]
pub type SelectedRuntime = cubecl::cpu::CpuRuntime;

/// `wgpu` backend placeholder (inert in Phase 1).
#[cfg(all(feature = "wgpu", not(feature = "cpu")))]
pub type SelectedRuntime = ();

/// CUDA backend placeholder (inert in Phase 1).
#[cfg(all(feature = "cuda", not(feature = "cpu"), not(feature = "wgpu")))]
pub type SelectedRuntime = ();

/// ROCm backend placeholder (inert in Phase 1).
#[cfg(all(feature = "rocm", not(feature = "cpu"), not(feature = "wgpu"), not(feature = "cuda")))]
pub type SelectedRuntime = ();
