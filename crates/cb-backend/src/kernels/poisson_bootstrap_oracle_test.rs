//! Upstream oracle for the device Poisson bootstrap kernel.
//!
//! `bootstrap_type=Poisson` is the ONE bootstrap arm with no CPU sampler anywhere: upstream
//! CatBoost's own validator rejects it on the CPU task type ("poisson bootstrap is not
//! supported on CPU", `bootstrap_options.cpp:29`), so the CUDA kernel IS the specification.
//! The reference here is therefore NOT a frozen CatBoost-Python run (impossible — the
//! per-object GPU bootstrap weights are not observable through any public API) but
//! `cb-oracle/generator/poisson_bootstrap_oracle.cpp`: a verbatim HOST transcription of
//! upstream `PoissonBootstrapImpl` + `random_gen.cuh`, compiled by g++ and frozen into
//! `cb-oracle/fixtures/bootstrap_poisson/`.
//!
//! That makes this test NON-tautological in the way that matters: the fixture is produced by
//! a separate program, in a different language, compiled by a different compiler, running on
//! the CPU — while the value under test is produced by a `#[cube]` kernel JIT-compiled to
//! GPU ISA. Agreement is evidence that the transcription is faithful, not that two copies of
//! the same code agree with themselves.
//!
//! Three geometries are gated because the object → seed map depends on the launch shape
//! (`numBlocks = min(ceil(seeds/256), ceil(n/256))`): `one_pass` (stride > n), `grid_wrap`
//! (stride < n, so each thread draws for several objects in sequence), and `wide` (the
//! production 79-block shape). Each is drawn TWICE over the same seed buffer, so the
//! cross-tree seed carry-over is gated too.
//!
//! Source/test separation is mandatory (CLAUDE.md): the kernel lives in the production
//! `kernels::bootstrap_device` module and every assertion / `unwrap` / index lives here.
//!
//! The assertions SKIP off rocm/cuda: the draw is u64/f64 and the cpu/wgpu backends cannot
//! execute it, and a silently "passing" cpu run would be a false pass (WR-01 discipline).

use std::path::PathBuf;

use crate::kernels::bootstrap_device::{
    create_poisson_seeds, draw_poisson_weights_host, poisson_alpha, poisson_grid,
    POISSON_BLOCK_SIZE, POISSON_SEEDS_SIZE,
};

/// Whether the device draw actually runs on this backend (u64/f64 kernel → rocm/cuda only).
fn device_backend_active() -> bool {
    cfg!(any(feature = "rocm", feature = "cuda"))
}

fn fixture(scenario: &str, file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("bootstrap_poisson")
        .join(scenario)
        .join(file)
}

/// The frozen scenario metadata (`config.json`) the generator wrote.
struct Scenario {
    n: usize,
    seeds_size: usize,
    subsample: f64,
    seed0: u64,
    rounds: usize,
    stride: usize,
    expected: Vec<f64>,
}

fn load(scenario: &str) -> Scenario {
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture(scenario, "config.json")).unwrap())
            .unwrap();
    let get_u = |k: &str| cfg[k].as_u64().unwrap() as usize;
    let weights: ndarray::Array1<f64> =
        ndarray_npy::read_npy(fixture(scenario, "weights.npy")).unwrap();
    Scenario {
        n: get_u("n"),
        seeds_size: get_u("seeds_size"),
        subsample: cfg["subsample"].as_f64().unwrap(),
        seed0: cfg["seed0"].as_u64().unwrap(),
        rounds: get_u("rounds"),
        stride: get_u("stride"),
        expected: weights.to_vec(),
    }
}

/// SplitMix64 over `seed0 + GOLDEN * (i + 1)` — the seed derivation the oracle `.cpp` pins,
/// transcribed here so both sides start from a byte-identical seed buffer. (Production uses
/// `create_poisson_seeds`, which draws from the validated `TFastRng64`; seed PROVENANCE is
/// not what this oracle gates — the object → weight map given a buffer is.)
fn splitmix64_seeds(seed0: u64, count: usize) -> Vec<u64> {
    (0..count)
        .map(|i| {
            let mut z = seed0.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        })
        .collect()
}

fn check_scenario(name: &str) {
    if !device_backend_active() {
        eprintln!("[poisson oracle {name}] skipped — needs rocm/cuda (u64/f64 device draw)");
        return;
    }
    let sc = load(name);
    // The launch geometry must be the one the oracle used, or objects draw from other seeds.
    let (_blocks, stride) = poisson_grid(sc.seeds_size, sc.n);
    assert_eq!(stride, sc.stride, "[{name}] launch stride diverged from the oracle geometry");

    let seeds = splitmix64_seeds(sc.seed0, sc.seeds_size);
    let got = draw_poisson_weights_host(&seeds, sc.subsample, sc.n, sc.rounds).unwrap();

    assert_eq!(got.len(), sc.expected.len(), "[{name}] weight count");
    // BIT-FOR-BIT: these are exact small integers on both sides; a tolerance here would hide
    // a real divergence in the draw sequence.
    let mismatches: Vec<usize> = (0..got.len()).filter(|&i| got[i] != sc.expected[i]).collect();
    assert!(
        mismatches.is_empty(),
        "[{name}] {} of {} device Poisson weights differ from upstream; first at index {} \
         (device {} vs upstream {})",
        mismatches.len(),
        got.len(),
        mismatches[0],
        got[mismatches[0]],
        sc.expected[mismatches[0]],
    );

    // Anti-false-pass: an all-zero (or constant) buffer would compare equal to nothing useful
    // if the fixture were also degenerate, so assert the reference itself carries real signal.
    let round0 = &sc.expected[..sc.n];
    let mean = round0.iter().sum::<f64>() / sc.n as f64;
    let lambda = f64::from(poisson_alpha(sc.subsample));
    assert!(
        (mean - lambda).abs() < 0.15,
        "[{name}] fixture mean {mean} is not a Poisson({lambda}) sample"
    );
    assert!(
        round0.contains(&0.0) && round0.iter().any(|&w| w >= 2.0),
        "[{name}] fixture is degenerate (no zeros or no multi-counts)"
    );
    // The second round must differ from the first: the seed buffer advances in place, so a
    // kernel that failed to write `seeds[t] = s` back would replay round 0 exactly.
    let round1 = &sc.expected[sc.n..];
    assert_ne!(round0, round1, "[{name}] fixture rounds are identical — seed carry-over untested");
    println!(
        "[poisson oracle {name}] n={} seeds={} lambda={:.6} stride={} rounds={} \
         mean={:.4} — bit-for-bit vs upstream",
        sc.n, sc.seeds_size, lambda, stride, sc.rounds, mean
    );
}

#[test]
fn poisson_one_pass_matches_upstream_bit_for_bit() {
    check_scenario("one_pass");
}

#[test]
fn poisson_grid_wrap_matches_upstream_bit_for_bit() {
    check_scenario("grid_wrap");
}

#[test]
fn poisson_wide_matches_upstream_bit_for_bit() {
    check_scenario("wide");
}

/// `GetPoissonLambda()` (`bootstrap_options.h:31-34`) — including the `subsample >= 1` case,
/// which upstream really does answer with `-1`. Pure host arithmetic; no GPU needed.
#[test]
fn poisson_lambda_matches_upstream_formula() {
    for &(subsample, expected) in &[
        (0.66_f64, 1.078_810_1_f32),
        (0.8_f64, 1.609_438_f32),
        // -log(1 - 0.5) is exactly ln 2.
        (0.5_f64, std::f32::consts::LN_2),
    ] {
        let got = poisson_alpha(subsample);
        assert!(
            (got - expected).abs() < 1e-6,
            "lambda({subsample}) = {got}, expected {expected}"
        );
        // e^-lambda == 1 - subsample: the identity that makes Poisson keep, on average, a
        // `subsample` fraction of the objects. This is WHY upstream picked this lambda.
        assert!(((-f64::from(got)).exp() - (1.0 - subsample)).abs() < 1e-6);
    }
    assert_eq!(poisson_alpha(1.0), -1.0, "upstream returns -1 for subsample >= 1");
    assert_eq!(poisson_alpha(1.5), -1.0);
}

/// The launch geometry mirrors `PoissonBootstrap` (`bootstrap.cu:66-70`) exactly, including
/// the `min` against the seed-buffer size that caps parallelism at 65536 threads.
#[test]
fn poisson_grid_matches_upstream_launch_geometry() {
    assert_eq!(poisson_grid(POISSON_SEEDS_SIZE, 1000), (4, 1024));
    assert_eq!(poisson_grid(1024, 4096), (4, 1024));
    assert_eq!(poisson_grid(POISSON_SEEDS_SIZE, 20000), (79, 20224));
    // Above the seed buffer the block count saturates at 256 — objects then wrap.
    assert_eq!(
        poisson_grid(POISSON_SEEDS_SIZE, 1_000_000),
        (256, POISSON_SEEDS_SIZE)
    );
    assert_eq!(POISSON_SEEDS_SIZE % POISSON_BLOCK_SIZE, 0);
}

/// A degenerate configuration must fail loudly rather than train on all-zero weights.
#[test]
fn poisson_rejects_subsample_at_or_above_one() {
    if !device_backend_active() {
        eprintln!("[poisson oracle] reject test skipped — needs rocm/cuda");
        return;
    }
    let seeds = splitmix64_seeds(1, POISSON_BLOCK_SIZE);
    let err = draw_poisson_weights_host(&seeds, 1.0, 64, 1).unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("subsample"), "unexpected error: {msg}");
}

/// Determinism: the same seed buffer must yield the same weights across independent client
/// sessions (WR01-S13's ≤1e-7 run-to-run budget, met exactly here).
#[test]
fn poisson_draw_is_deterministic() {
    if !device_backend_active() {
        eprintln!("[poisson oracle] determinism test skipped — needs rocm/cuda");
        return;
    }
    let seeds = splitmix64_seeds(9_999, POISSON_SEEDS_SIZE);
    let a = draw_poisson_weights_host(&seeds, 0.8, 5000, 1).unwrap();
    let b = draw_poisson_weights_host(&seeds, 0.8, 5000, 1).unwrap();
    assert_eq!(a, b, "device Poisson draw is not deterministic for a pinned seed buffer");
}

/// `create_poisson_seeds` must produce a full, non-degenerate buffer from the fit seed — a
/// zero-filled or short buffer would silently collapse the sample.
#[test]
fn poisson_seed_buffer_is_well_formed() {
    if !device_backend_active() {
        eprintln!("[poisson oracle] seed-buffer test skipped — needs rocm/cuda");
        return;
    }
    let device = <crate::SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <crate::SelectedRuntime as cubecl::Runtime>::client(&device);
    let handle = create_poisson_seeds(&client, 42, POISSON_SEEDS_SIZE);
    let bytes = client.read_one(handle).unwrap();
    let seeds: &[u64] = bytemuck::cast_slice(&bytes);
    assert_eq!(seeds.len(), POISSON_SEEDS_SIZE);
    assert!(seeds.iter().all(|&s| s != 0), "a zero seed word would freeze that thread's stream");
    let unique: std::collections::HashSet<u64> = seeds.iter().copied().collect();
    assert_eq!(unique.len(), seeds.len(), "seed buffer must not repeat words");
}
