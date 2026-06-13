#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]
//! `cb-backend` — the sole feature-gated runtime-alias owner (D-02). The concrete
//! CubeCL runtime is wired in Phase 7 (D-05); in Phase 1 every backend arm is an
//! inert placeholder selected at compile time with zero runtime dispatch.

/// Compile-time-selected runtime alias. One `cfg` arm per backend feature; the
/// real runtime type replaces `()` in Phase 7. Selection is purely compile-time
/// (D-02) — there is no runtime `match` over backends.
#[cfg(feature = "cpu")]
pub type SelectedRuntime = ();

/// `wgpu` backend placeholder (inert in Phase 1).
#[cfg(all(feature = "wgpu", not(feature = "cpu")))]
pub type SelectedRuntime = ();

/// CUDA backend placeholder (inert in Phase 1).
#[cfg(all(feature = "cuda", not(feature = "cpu"), not(feature = "wgpu")))]
pub type SelectedRuntime = ();

/// ROCm backend placeholder (inert in Phase 1).
#[cfg(all(feature = "rocm", not(feature = "cpu"), not(feature = "wgpu"), not(feature = "cuda")))]
pub type SelectedRuntime = ();
