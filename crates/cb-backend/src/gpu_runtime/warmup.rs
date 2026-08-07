//! Fit-time device warm-up + compilation-cache bootstrap (SPD-03).
//!
//! A cold GPU fit pays the FULL JIT compilation of every kernel it launches on
//! the calling thread, inline with training: the 2026-08-08 P100 diagnostic shows
//! a ~1334 ms FIRST tree against ~16 ms steady-state, and a 1222 ms `begin-raw`
//! that is mostly first-launch compilation. Official CatBoost ships precompiled
//! device code, so its first fit pays none of this. Two mitigations live here:
//!
//! 1. [`ensure_compile_cache`] turns on CubeCL's disk compilation cache (off by
//!    default — `CompilationConfig::default().cache == None`), so every process
//!    after the first on a machine deserializes PTX/bytecode instead of invoking
//!    the driver compiler.
//! 2. [`spawn_fit_warmup`] runs a MINIATURE device fit (n=2048, nf=2) on a
//!    background thread at `fit()` entry, so kernel compilation overlaps the
//!    ~1.5 s of host-side pool ingestion + fit-prep instead of serializing after
//!    them. Every launch-relevant `#[comptime]` key — `bits`/line width (from
//!    `n_bins`), the score function, the loss's der kernel — is shape-independent,
//!    so the tiny fit compiles the exact variants the real fit will request.
//!    The real n_bins is unknown until borders are built (border subsampling can
//!    shrink it), so the warm-up sweeps the dispatched line-width classes
//!    starting from the `border_count` hint.
//!
//! The warm-up thread is marked via a thread-local so (a) the hist-fill probe
//! never latches an arm from meaningless tiny-shape timings and (b) no
//! `CB_GPU_PROF` line is ever emitted from a warm-up launch (the bench's device
//! probe counts those lines as device-activation evidence).
//!
//! Everything here is best-effort: a warm-up failure only means the real fit
//! compiles on demand, exactly as before this module existed.

use cb_compute::{DeviceTrainConfig, EScoreFunction, Loss};
use cb_core::CbResult;

use super::session::GpuTrainSession;

thread_local! {
    /// Set on the warm-up thread only (see module doc).
    static WARMUP_THREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the CURRENT thread is a warm-up thread (gates probe latching and every
/// `CB_GPU_PROF` print — see [`crate::gpu_runtime::gpu_prof_enabled`]).
pub(crate) fn warmup_thread_active() -> bool {
    WARMUP_THREAD.with(std::cell::Cell::get)
}

/// Enable CubeCL's disk compilation cache once per process, BEFORE the first
/// client is created. Best-effort: if another component already configured
/// CubeCL (config latched or a user `cubecl.toml` was set programmatically),
/// the `set` panics internally and is swallowed — never overriding a user
/// choice, never failing the fit. The default `CacheConfig::Target` resolves to
/// the workspace `target/` in a dev checkout and to the per-user cache
/// directory for an installed wheel.
pub fn ensure_compile_cache() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        use cubecl::config::{cache::CacheConfig, CubeClRuntimeConfig, RuntimeConfig};
        let mut cfg = CubeClRuntimeConfig::default();
        cfg.compilation.cache = Some(CacheConfig::Target);
        // Silence the default panic hook for the expected already-configured
        // panic (the hook swap is process-global but lasts only this call, once
        // per process, at fit entry).
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            CubeClRuntimeConfig::set(cfg);
        }));
        std::panic::set_hook(prev_hook);
        drop(result);
    });
}

/// Spawn the background warm-up for an upcoming device fit (see module doc).
/// `border_count_hint` is the requested per-feature border count (the REAL
/// post-subsample bin count may be smaller — the sweep covers the other line
/// widths after the hinted one). Detached: the fit never joins or waits on it;
/// the CubeCL server serializes concurrent compile requests internally, so the
/// races are benign (whoever arrives first compiles, the other hits the cache).
pub fn spawn_fit_warmup(
    loss: Loss,
    depth: usize,
    border_count_hint: usize,
    score_function: EScoreFunction,
) {
    ensure_compile_cache();
    // Once-per-key guard: within a process each kernel variant compiles at most
    // once (CubeCL's in-memory cache), so re-warming the same
    // loss/depth/width/score key on every fit — oracle suites and bench grids
    // call `fit` hundreds of times — would only burn threads and device time.
    static WARMED: std::sync::Mutex<Option<std::collections::HashSet<String>>> =
        std::sync::Mutex::new(None);
    let key = format!("{loss:?}|{depth}|{border_count_hint}|{score_function:?}");
    if let Ok(mut guard) = WARMED.lock() {
        if !guard.get_or_insert_with(Default::default).insert(key) {
            return;
        }
    }
    // The dispatched histogram line widths (`pad_hist_line_bins` families). Warm
    // the hinted class first — it is the likeliest real width — then the rest,
    // smallest first (cheapest compiles early).
    let hint = border_count_hint.saturating_add(1).next_power_of_two().clamp(32, 256);
    let mut classes: Vec<usize> = vec![hint];
    for w in [64usize, 32, 128, 256] {
        if !classes.contains(&w) {
            classes.push(w);
        }
    }
    let spawned = std::thread::Builder::new()
        .name("cb-gpu-warmup".to_owned())
        .spawn(move || {
            WARMUP_THREAD.with(|c| c.set(true));
            for n_bins in classes {
                // Best-effort per class; an uncovered config simply declines.
                let _ = warm_one_class(&loss, depth, n_bins, score_function);
            }
        });
    // A failed spawn only means no warm-up; the fit compiles on demand.
    drop(spawned);
}

/// Run one miniature device fit at the given histogram line width: open a raw
/// session (compiles quantize+pack, const-fill, upload plumbing) and grow two
/// trees (compiles der, both hist-fill arms via the probe's compile-only warm-up
/// branch, scan/score, split, stats read, leaf apply). Errors are the caller's
/// signal to move on — never surfaced to the fit.
fn warm_one_class(
    loss: &Loss,
    depth: usize,
    n_bins: usize,
    score_function: EScoreFunction,
) -> CbResult<()> {
    let n = 2048_usize;
    let nf = 2_usize;
    let depth = depth.clamp(1, 8);
    let n_borders = n_bins.saturating_sub(1).max(1);
    // f32 midpoints — round-trip f64→f32→f64 exactly by construction, so the raw
    // channel's border gate admits them (the same property real borders have).
    let borders: Vec<f64> = (0..n_borders).map(|k| f64::from(k as f32 + 0.5)).collect();
    let feature_borders: Vec<Vec<f64>> = (0..nf).map(|_| borders.clone()).collect();
    // Spread values across the full bin range so every level has a split to find.
    let columns: Vec<Vec<f32>> = (0..nf)
        .map(|f| (0..n).map(|i| ((i.wrapping_mul(f + 1)) % n_bins) as f32).collect())
        .collect();
    let weight = vec![1.0_f64; n];
    let target: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let config = DeviceTrainConfig::default();
    let session = GpuTrainSession::begin_raw(
        loss,
        depth,
        /* boosting_type_is_plain = */ true,
        /* fold_count = */ 1,
        score_function,
        &columns,
        &feature_borders,
        &weight,
        n,
        nf,
        n_bins,
        /* learning_rate = */ 0.05,
        /* scaled_l2 = */ 3.0,
        &config,
    )?;
    if let Some(mut session) = session {
        let approx = vec![0.0_f64; n];
        for _ in 0..2 {
            let _ = session.grow_one(&approx, &target, &[])?;
        }
    }
    Ok(())
}
