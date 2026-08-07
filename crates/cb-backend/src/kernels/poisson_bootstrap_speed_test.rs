//! Kernel-level speed evidence for the device Poisson bootstrap draw.
//!
//! The other bootstrap arms MUST run serially on one thread: Bernoulli and Bayesian
//! mirror upstream's CPU samplers, whose randomness is a single sequential `TFastRng64`
//! stream, and reproducing it bit-for-bit forbids any reordering. That is a correctness
//! constraint, not an oversight.
//!
//! Poisson has no such constraint, because its upstream definition is itself a CUDA
//! kernel: `PoissonBootstrapImpl` runs `numBlocks * 256` threads, each owning one seed
//! word and walking objects with a grid stride. Transcribing upstream faithfully
//! therefore makes the draw parallel — up to 65536 threads — and being faithful and
//! being fast are the same act here rather than a trade-off.
//!
//! This mount measures that: the Poisson draw against the SERIAL Bernoulli draw at the
//! same object count. Bernoulli is the honest baseline because it is the shape the
//! Poisson kernel itself had before this work (a single-thread `while` loop over `n`),
//! so the ratio is a like-for-like reading of what the parallel transcription bought.
//!
//! Sizing matters for this to measure anything. At 300k objects BOTH draws finish in
//! tens of milliseconds and the reading is dominated by launch + read-back overhead
//! (~14 ms of fixed cost), which compresses the ratio to a meaningless ~2.6x. The
//! object count below is therefore large enough that the serial loop — which is O(n) on
//! ONE thread — dominates its own measurement, and each draw is timed as the best of
//! three runs so a scheduling hiccup cannot decide the result.
//!
//! The assertion is still deliberately loose — this is a regression guard, not a
//! certified speedup — but the gap it guards is an order of magnitude, so a loose bar
//! still catches a collapse back to serial.
//!
//! rocm/cuda only; cpu/wgpu print a SKIP line rather than passing silently.
//!
//! # Why this test is `#[ignore]`d (measured 2026-08-07, gfx1151)
//!
//! It needs an UNCONTENDED device, and a package-wide `cargo test -p cb-backend --lib`
//! does not provide one: ~260 tests share the GPU, and the default harness runs them on
//! as many threads as there are cores. The kernel is fine — the measurement is not:
//!
//! | condition | parallel | serial | ratio |
//! |---|---|---|---|
//! | alone (×3) | 32.7 / 33.0 / 33.2 ms | ~350 ms | **10.5–10.7×** — passes, 2× the bar |
//! | inside the full `--lib` suite (×3) | 218 / 186 / 171 ms | 723 / 703 / 627 ms | 3.3–3.8× — fails |
//!
//! Both arms slow down under load, but not equally: the parallel (GPU-bound) arm degrades
//! ~6× while the serial arm degrades only ~2×. The ratio is a quotient of two quantities
//! with different contention sensitivity, so it collapses while the code under test is
//! untouched. Every failure observed was of this kind — the baseline commit fails it 3/3
//! too.
//!
//! Three fixes were considered; only isolation survives scrutiny:
//!
//!  * **A fixed wall-clock budget** (e.g. "under 60 ms") does NOT work. It is *more*
//!    contention-sensitive, not less: the parallel arm measures 171–293 ms under load, so
//!    any budget near its uncontended 33 ms fails for exactly the same reason.
//!  * **A process-wide GPU mutex** would work but only if EVERY GPU-touching test in the
//!    crate took it — ~260 sites, for one test's benefit.
//!  * **Retrying for a quiet window** is already effectively in place and already
//!    insufficient: [`best_of_three`] takes the minimum of three runs per arm, and the
//!    contention is sustained across the whole suite rather than a transient hiccup, so
//!    all three samples are equally contended.
//!
//! So the test is `#[ignore]`d and run in its own process, where the measurement means
//! what it claims. `run_device_tests.sh` does this; by hand it is:
//!
//! ```text
//! cargo test -p cb-backend --no-default-features --features rocm \
//!     --lib kernels::poisson_bootstrap_speed_test -- --ignored --nocapture
//! ```
//!
//! `#[ignore]` here means "needs an isolated device", NOT "known-broken" — the assertion
//! below is live and unweakened, and `MIN_SPEEDUP` was deliberately NOT lowered, since a
//! bar tuned to contended numbers would measure the scheduler instead of the kernel.

use std::time::Instant;

use crate::kernels::bootstrap_device::{
    draw_bootstrap_weights_host, draw_poisson_weights_host, DeviceBootstrapKind,
    POISSON_SEEDS_SIZE,
};

fn device_backend_active() -> bool {
    cfg!(any(feature = "rocm", feature = "cuda"))
}

/// The serial single-thread draw is an order of magnitude slower at this size; require
/// only 5x so timing noise cannot fail the build, while a regression to serial still would.
const MIN_SPEEDUP: f64 = 5.0;

/// Object count large enough that the one-thread loop dominates its own measurement
/// rather than the fixed launch + read-back cost (see the module docs).
const N: usize = 2_000_000;

/// Best-of-N timing: the minimum is the least noisy estimator of a fixed workload.
fn best_of_three(mut f: impl FnMut() -> f64) -> f64 {
    (0..3).map(|_| f()).fold(f64::MAX, f64::min)
}

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

#[test]
#[ignore = "perf: needs an uncontended device — run in its own process (see the module \
            docs, or run_device_tests.sh)"]
fn poisson_parallel_draw_outpaces_the_serial_stream_draw() {
    if !device_backend_active() {
        eprintln!("[poisson speed] skipped — needs rocm/cuda");
        return;
    }
    let seeds = splitmix64_seeds(20_260_731, POISSON_SEEDS_SIZE);
    let base = cb_core::TFastRng64::from_seed(42).raw_state();

    // Warm both paths first: the first launch pays kernel JIT compilation, which would
    // otherwise dominate whichever draw happened to run first.
    let warm_p = draw_poisson_weights_host(&seeds, 0.8, 4096, 1).unwrap();
    let warm_b =
        draw_bootstrap_weights_host(DeviceBootstrapKind::Bernoulli, base, 0, 0.8, 0.0, 4096)
            .unwrap();
    assert_eq!(warm_p.len(), 4096);
    assert_eq!(warm_b.len(), 4096);

    let mut poisson = Vec::new();
    let poisson_s = best_of_three(|| {
        let t = Instant::now();
        poisson = draw_poisson_weights_host(&seeds, 0.8, N, 1).unwrap();
        t.elapsed().as_secs_f64()
    });
    let mut bernoulli = Vec::new();
    let serial_s = best_of_three(|| {
        let t = Instant::now();
        bernoulli =
            draw_bootstrap_weights_host(DeviceBootstrapKind::Bernoulli, base, 0, 0.8, 0.0, N)
                .unwrap();
        t.elapsed().as_secs_f64()
    });

    // Both draws must have actually produced n weights — a short buffer would make the
    // timing meaningless.
    assert_eq!(poisson.len(), N);
    assert_eq!(bernoulli.len(), N);
    // And the Poisson draw must carry real signal, not an all-zero fast path.
    let mean = poisson.iter().sum::<f64>() / N as f64;
    assert!(
        mean > 1.0 && mean < 2.5,
        "Poisson draw mean {mean} is not a lambda=1.609 sample; the timing is measuring \
         a degenerate kernel"
    );

    let speedup = serial_s / poisson_s;
    println!(
        "[poisson speed] n={N}: parallel Poisson {:.1}ms vs serial stream draw {:.1}ms \
         -> {speedup:.1}x",
        poisson_s * 1e3,
        serial_s * 1e3,
    );
    assert!(
        speedup >= MIN_SPEEDUP,
        "the Poisson draw is only {speedup:.1}x the serial single-thread draw (< {MIN_SPEEDUP}x); \
         the grid-stride transcription of upstream's PoissonBootstrapImpl has regressed to a \
         serial loop"
    );
}
