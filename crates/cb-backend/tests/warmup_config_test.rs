//! SPD-03: `ensure_compile_cache` must actually flip CubeCL's disk compilation
//! cache on (it is `None` by default — `CompilationConfig::default()`), else every
//! process pays full driver JIT and the warm-up module silently loses its second
//! mitigation. This is process-global state, so the assertion lives alone in its
//! own integration-test binary (one process, no sibling test can latch the config
//! first).

#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
#[test]
fn ensure_compile_cache_flips_the_disk_cache_on() {
    cb_backend::gpu_runtime::warmup::ensure_compile_cache();
    let cfg = <cubecl::config::CubeClRuntimeConfig as cubecl::config::RuntimeConfig>::get();
    assert!(
        cfg.compilation.cache.is_some(),
        "ensure_compile_cache ran first in this process, but the compilation cache \
         is still None — the config set was silently swallowed"
    );
}
