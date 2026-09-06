//! Self-oracle for [`crate::launch_geometry`] (source/test separation is mandatory —
//! CLAUDE.md / AGENTS.md: the production body carries no `#[cfg(test)]` block).
//!
//! Two properties are under test, and they pull in opposite directions, which is the
//! point:
//!
//! 1. **Coverage is absolute.** The elementwise kernels this geometry serves are
//!    one-shot and bounds-guarded (`if ABSOLUTE_POS < n { out[ABSOLUTE_POS] = .. }`),
//!    with no grid-stride loop to pick up a shortfall. A geometry that spans fewer
//!    than `n` units silently leaves the tail of the output buffer at whatever
//!    `client.empty()` returned — a wrong answer with no error. Every other property
//!    here is subordinate to this one.
//! 2. **The width must actually shrink on the CPU runtime.** That is the entire
//!    optimization; a test suite that only checked coverage would pass unchanged
//!    against the hard-coded 32-wide geometry this replaced.

use cubecl::Runtime;
use cubecl::prelude::CubeCount;

use crate::launch_geometry::{
    LaneCost, barrier_cube_dim, der_line_size, has_planes, launch_1d, SCALAR_LINE,
};

/// Total units the grid spans — the span `ABSOLUTE_POS` takes over the whole launch.
fn total_units(count: &CubeCount, dim: cubecl::prelude::CubeDim) -> usize {
    let cubes = match count {
        CubeCount::Static(x, y, z) => (*x as usize) * (*y as usize) * (*z as usize),
        _ => panic!("launch_1d must return a Static cube count"),
    };
    cubes * dim.num_elems() as usize
}

fn client() -> cubecl::client::ComputeClient<crate::SelectedRuntime> {
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    <crate::SelectedRuntime as Runtime>::client(&device)
}

/// THE correctness property: the grid must span at least `n` units for every `n`,
/// including the awkward ones (primes, one-past a cube boundary, one-below).
///
/// A shortfall here is not a slowdown, it is silent data corruption — the kernels have
/// no grid-stride loop to recover the tail.
#[test]
fn grid_always_covers_every_element() {
    let client = client();
    for &n in &[
        0usize, 1, 2, 3, 7, 8, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 999, 1000,
        1024, 4095, 4096, 4097, 65_537, 1_000_003,
    ] {
        let (count, dim) = launch_1d(&client, n, LaneCost::compute(1));
        assert!(
            total_units(&count, dim) >= n,
            "geometry for n={n} spans {} units, short of {n} — the tail of the output \
             buffer would never be written",
            total_units(&count, dim)
        );
        assert!(dim.num_elems() >= 1, "n={n} produced an empty cube dim");
    }
}

/// A degenerate empty input must still be a launchable grid, not `Static(0, 0, 0)`.
///
/// This preserves the `.max(1)` in the geometry this replaced. The kernels' bounds
/// guard makes the launch a no-op; what matters is that an empty column does not turn
/// into a backend-specific zero-extent launch failure (the HIP backend is unforgiving
/// about zero-length work).
#[test]
fn empty_input_still_yields_a_launchable_grid() {
    let client = client();
    let (count, dim) = launch_1d(&client, 0, LaneCost::compute(1));
    assert!(
        total_units(&count, dim) >= 1,
        "n=0 must still produce a launchable (bounds-guarded, no-op) grid"
    );
    match count {
        CubeCount::Static(x, y, z) => {
            assert!(x >= 1 && y >= 1 && z >= 1, "n=0 produced a zero-extent cube count");
        }
        _ => panic!("expected a Static cube count"),
    }
}

/// The optimization itself, on the CPU runtime: a launch far below the per-unit work
/// threshold must collapse to ONE unit.
///
/// On `cubecl-cpu` the cube width is the number of OS-thread tasks dispatched per
/// launch (`runner.rs::execute_data` sends one task, one `MlirData` clone and one stop
/// message per unit, then blocks until all report back). The width this replaced was a
/// fixed 32, so a 100-element gradient paid 32 thread wake-ups for 100 subtractions.
///
/// This assertion is what fails if the hard-coded 32-wide geometry ever comes back.
#[cfg(feature = "cpu")]
#[test]
fn cpu_small_launch_collapses_to_a_single_unit() {
    let client = client();
    assert!(
        !has_planes(&client),
        "the cpu feature must select a runtime with no hardware planes"
    );
    for &n in &[1usize, 10, 100, 1000] {
        let (_, dim) = launch_1d(&client, n, LaneCost::compute(1));
        assert_eq!(
            dim.num_elems(),
            1,
            "n={n} is far below one unit's worth of work ({} scalar ops), so it must \
             dispatch a single task, not {} of them",
            32 * 1024,
            dim.num_elems()
        );
    }
}

/// The CPU width scales up with real work, but never past the reported core count and
/// never past the number of elements there are to write.
#[cfg(feature = "cpu")]
#[test]
fn cpu_width_scales_with_work_and_stays_bounded() {
    let client = client();
    let cores = client.properties().hardware.num_cpu_cores.unwrap_or(1).max(1);

    // Well past the threshold: the width should have grown beyond a single unit
    // (otherwise the helper is just a constant-1 function and the scaling is dead).
    let (_, big) = launch_1d(&client, 4_000_000, LaneCost::compute(1));
    assert!(
        big.num_elems() > 1,
        "4M elements must engage more than one unit on a {cores}-core host"
    );

    // ... but never past the cores, and never past the element count.
    for &n in &[1usize, 100, 100_000, 4_000_000, 100_000_000] {
        let (_, dim) = launch_1d(&client, n, LaneCost::compute(1));
        assert!(
            dim.num_elems() <= cores,
            "n={n}: width {} exceeds the {cores} reported cores — this oversubscribes \
             the worker pool, which cubecl-cpu grows to match the cube width",
            dim.num_elems()
        );
        assert!(
            (dim.num_elems() as usize) <= n.max(1),
            "n={n}: {} units for {n} elements leaves units with no work",
            dim.num_elems()
        );
    }
}

/// `work_per_lane` is what lets a cheap-per-lane kernel and an expensive-per-lane
/// kernel over the same `n` get different widths. If it were ignored, the parameter
/// would be decoration.
#[cfg(feature = "cpu")]
#[test]
fn cpu_width_responds_to_per_lane_work() {
    let client = client();
    let n = 10_000usize;
    let (_, cheap) = launch_1d(&client, n, LaneCost::compute(1));
    let (_, expensive) = launch_1d(&client, n, LaneCost::compute(1024));
    assert!(
        expensive.num_elems() > cheap.num_elems(),
        "the same {n} lanes at 1024x the per-lane work must earn more units \
         (cheap={}, expensive={})",
        cheap.num_elems(),
        expensive.num_elems()
    );
}

/// The UPPER bound the `ops`-only model was missing: a streaming (bandwidth-bound)
/// lane must stop earning units well before the core count, while a compute-bound lane
/// of the SAME per-lane op count keeps scaling to it.
///
/// This is the assertion that fails if the memory-saturation ceiling is ever dropped
/// and `launch_1d` goes back to reading "more total work" as "more units" for a kernel
/// that is already at the memory ceiling. On the 8-core M1 that mistake measured 2.7x
/// on `gradient_kernel` at n = 1M (the table at `STREAMING_LANE`), and it is invisible
/// to every other test here: the output stays bit-identical, it is only slower.
#[cfg(feature = "cpu")]
#[test]
fn cpu_streaming_width_stops_at_memory_saturation() {
    let client = client();
    let cores = client.properties().hardware.num_cpu_cores.unwrap_or(1).max(1) as usize;
    // A host that reports fewer than two cores cannot distinguish the two ceilings —
    // `cores / 2` and `cores` coincide at 1 — so there is nothing to assert.
    if cores < 2 {
        return;
    }

    // `n` far past the point where the op-count model wants every core, so the width is
    // decided by the ceiling and by nothing else.
    let n = 100_000_000usize;
    let (_, streaming) = launch_1d(&client, n, LaneCost::streaming(4));
    let (_, compute) = launch_1d(&client, n, LaneCost::compute(4));

    assert_eq!(
        compute.num_elems() as usize,
        cores.min(crate::launch_geometry::CPU_CUBE_DIM_MAX as usize),
        "a compute-bound lane must still scale to the full core count"
    );
    assert!(
        (streaming.num_elems() as usize) <= (cores / 2).max(1),
        "a streaming lane must be capped at the memory-saturation width ({} units),          not at the {cores} cores it was given — it got {}",
        (cores / 2).max(1),
        streaming.num_elems()
    );
    assert!(
        streaming.num_elems() >= 1,
        "the ceiling must never collapse the width to zero"
    );
}

/// The ceiling is an upper bound, not a floor: it must never PROMOTE a small launch
/// that the op-count model would keep at one unit.
#[cfg(feature = "cpu")]
#[test]
fn cpu_streaming_ceiling_never_widens_a_small_launch() {
    let client = client();
    for &n in &[1usize, 10, 100, 1000] {
        let (_, dim) = launch_1d(&client, n, LaneCost::streaming(4));
        assert_eq!(
            dim.num_elems(),
            1,
            "n={n} is far below one unit's worth of work, so the streaming ceiling must              leave it at a single task, not widen it to {}",
            dim.num_elems()
        );
    }
}

/// On a real GPU the width must be a whole number of planes, so no SIMD lane in a
/// wavefront sits idle, and must respect the device's units-per-cube limit.
#[cfg(any(feature = "cuda", feature = "rocm"))]
#[test]
fn gpu_width_is_plane_aligned_and_within_device_limits() {
    let client = client();
    assert!(
        has_planes(&client),
        "a cuda/rocm build must select a runtime reporting hardware planes"
    );
    let hardware = client.properties().hardware.clone();
    let plane = hardware.plane_size_max as usize;

    for &n in &[1usize, 33, 1000, 65_537, 1_000_003] {
        let (_, dim) = launch_1d(&client, n, LaneCost::compute(1));
        let units = dim.num_elems() as usize;
        assert!(
            units % plane == 0,
            "n={n}: width {units} is not a whole number of {plane}-wide planes — the \
             remainder lanes idle"
        );
        assert!(
            units <= hardware.max_units_per_cube as usize,
            "n={n}: width {units} exceeds the device's {} units-per-cube limit",
            hardware.max_units_per_cube
        );
    }
}

/// THE invariant that licenses this whole optimization: for an order-independent
/// elementwise kernel, the output buffer is **bit-identical** under any geometry that
/// covers `[0, n)`.
///
/// Geometry is only safe to change where the schedule cannot reach the result. That is
/// true here and false three modules over — the shared-memory block-reduce folds floats
/// in-cube, so its cube width picks the summation order and moves the answer at the ULP
/// level, and the Poisson draw's grid is pinned bit-for-bit to upstream. This test pins
/// the distinction rather than leaving it to a comment: it launches the real
/// `gradient_kernel` over the same input under a deliberately pathological 1-wide cube,
/// the old hard-coded 32-wide cube, and a 256-wide cube, and requires every output bit
/// to agree.
///
/// If a future edit routes a reducing or atomic-accumulating kernel through
/// [`launch_1d`], this test will NOT catch it — but the property it documents is the
/// one to check before doing so.
#[test]
fn elementwise_output_is_bit_identical_across_geometries() {
    use cubecl::prelude::{ArrayArg, CubeDim};

    let client = client();
    let n = 10_000usize;
    let approx: Vec<f64> = (0..n).map(|i| (i as f64) * 1e-3 - 5.0).collect();
    let target: Vec<f64> = (0..n).map(|i| (i as f64) * 7e-4 + 0.25).collect();

    let run = |width: u32| -> Vec<f64> {
        let a = client.create(cubecl::bytes::Bytes::from_elems(approx.clone()));
        let t = client.create(cubecl::bytes::Bytes::from_elems(target.clone()));
        let out = client.empty(n * std::mem::size_of::<f64>());
        let dim = CubeDim { x: width, y: 1, z: 1 };
        let cubes = n.div_ceil(width as usize).max(1) as u32;
        crate::kernels::gradient_kernel::launch::<f64, crate::SelectedRuntime>(
            &client,
            cubecl::prelude::CubeCount::Static(cubes, 1, 1),
            dim,
            // Width 1: this test allocates at exactly `n` (no padding), so only the scalar
            // line is admissible — see `SCALAR_LINE`.
            SCALAR_LINE,
            unsafe { ArrayArg::from_raw_parts(a, n) },
            unsafe { ArrayArg::from_raw_parts(t, n) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), n) },
        );
        let bytes = client.read_one(out).unwrap();
        bytemuck::cast_slice::<u8, f64>(&bytes).to_vec()
    };

    let narrow = run(1);
    assert_eq!(narrow.len(), n, "the 1-wide launch must still write all {n} elements");

    for &width in &[32u32, 256] {
        let other = run(width);
        // Compare BITS, not values: `==` would let a NaN or a -0.0/+0.0 difference pass.
        let mismatches = narrow
            .iter()
            .zip(other.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            mismatches, 0,
            "gradient_kernel output differs in {mismatches} of {n} elements between a \
             1-wide and a {width}-wide cube — this kernel is NOT order-independent and \
             must not use the adaptive geometry"
        );
    }

    // And the values are actually the gradient, not an all-zero buffer that would make
    // the bit-identity above vacuously true.
    assert!(
        narrow.iter().any(|v| *v != 0.0),
        "the output is entirely zero — the comparison above proved nothing"
    );
    for i in [0usize, 1, n / 2, n - 1] {
        assert_eq!(
            narrow[i].to_bits(),
            (target[i] - approx[i]).to_bits(),
            "element {i} is not the RMSE gradient `target - approx`"
        );
    }
}

/// A vectorized launch must reach the SAME width decision as the scalar launch it
/// replaces: `n / width` lanes at `width` times the per-lane cost is the same total
/// work, and the total is what the lower bound is calibrated on. If `per_vector` did
/// not scale the ops, an 8-wide launch would look 8x cheaper and under-parallelize.
#[cfg(feature = "cpu")]
#[test]
fn per_vector_keeps_the_width_decision_of_the_scalar_launch() {
    let client = client();
    for &n in &[8usize, 4096, 100_000, 1_000_000] {
        for width in [1usize, 2, 4, 8] {
            let lanes = n.div_ceil(width);
            let (_, scalar) = launch_1d(&client, n, LaneCost::compute(16));
            let (_, vector) = launch_1d(&client, lanes, LaneCost::compute(16).per_vector(width));
            assert_eq!(
                vector.num_elems(),
                scalar.num_elems(),
                "n={n} width={width}: the vector launch chose a different width"
            );
        }
    }
}

/// A vector streaming lane saturates memory on far fewer units than a scalar one (see
/// `STREAMING_LANE`, "Vector lanes"): its ceiling is `cores / 8`, never above the scalar
/// `cores / 2`, and never below one unit. The compute tier is untouched by the width.
#[cfg(feature = "cpu")]
#[test]
fn per_vector_narrows_the_streaming_ceiling_and_leaves_compute_alone() {
    let client = client();
    let cores = client.properties().hardware.num_cpu_cores.unwrap_or(1).max(1) as usize;
    if cores < 2 {
        return;
    }
    let n = 100_000_000usize;
    let (_, streaming) = launch_1d(&client, n / 8, LaneCost::streaming(4).per_vector(8));
    let (_, scalar_streaming) = launch_1d(&client, n, LaneCost::streaming(4));
    let (_, compute) = launch_1d(&client, n / 8, LaneCost::compute(16).per_vector(8));
    assert_eq!(
        streaming.num_elems() as usize,
        (cores / 8).max(1),
        "an 8-wide streaming lane must stop at cores / 8, got {}",
        streaming.num_elems()
    );
    assert!(streaming.num_elems() <= scalar_streaming.num_elems());
    assert_eq!(
        compute.num_elems() as usize,
        cores.min(crate::launch_geometry::CPU_CUBE_DIM_MAX as usize),
        "a vector compute-bound lane must still scale to the full core count"
    );
}

/// The device width is a power of two >= 1 everywhere, and the padded buffers the CPU
/// launcher builds must be a whole number of vectors; on the CPU runtime it is the
/// widest io-optimized width for f64 (> 1, the whole point), on a GPU it is the scalar
/// line until the device seams learn to pad.
#[test]
fn der_line_size_is_a_power_of_two_the_launcher_can_pad_to() {
    let client = client();
    let line = der_line_size(&client, std::mem::size_of::<f64>());
    assert!(line >= SCALAR_LINE);
    assert!(line.is_power_of_two(), "line {line} is not a power of two");
    if has_planes(&client) {
        assert_eq!(line, SCALAR_LINE, "device seams do not pad, so they must get width 1");
    } else {
        assert!(line > 1, "the CPU runtime advertises SIMD widths; got {line}");
    }
    for n in [1usize, 7, 8, 9, 1001] {
        let n_pad = n.div_ceil(line) * line;
        assert_eq!(n_pad % line, 0);
        assert!(n_pad >= n && n_pad - n < line);
    }
}

/// The barrier width never exceeds what was requested, is always a power of two (the
/// tree reductions halve `CUBE_DIM_X`), and on the CPU runtime never exceeds the core
/// count — one spinning unit per core is the whole fix for the ~1 s/cube spin-barrier
/// collapse measured at 32 units on 8 cores.
#[test]
fn barrier_cube_dim_is_a_power_of_two_capped_by_request_and_cores() {
    let client = client();
    let cores = client.properties().hardware.num_cpu_cores.unwrap_or(1).max(1) as usize;
    for requested in [1usize, 2, 8, 32, 256] {
        let width = barrier_cube_dim(&client, requested);
        assert!(width >= 1);
        assert!(width.is_power_of_two(), "requested={requested}: width {width}");
        assert!(width <= requested, "requested={requested}: width {width}");
        if has_planes(&client) {
            assert_eq!(width, requested, "a GPU launch keeps the width it asked for");
        } else {
            assert!(width <= cores, "requested={requested}: {width} units on {cores} cores");
            assert!(
                width * 2 > requested.min(cores),
                "requested={requested}: {width} is not the largest admissible power of two"
            );
        }
    }
    assert_eq!(barrier_cube_dim(&client, 0), 1, "a zero request still launches one unit");
}

/// The block-reduce family's memoized width is exactly `barrier_cube_dim` of its
/// requested 32, and the single-cube width covers its items without exceeding the
/// capacity the callers validate against.
#[test]
fn gpu_runtime_cube_dims_follow_the_barrier_rule() {
    let client = client();
    let width = crate::gpu_runtime::cube_dim();
    assert_eq!(width, barrier_cube_dim(&client, 32));
    for items in [0usize, 1, 2, 5, 8, 9, 17, 32] {
        let single = crate::gpu_runtime::single_cube_dim(items);
        assert!(single.is_power_of_two());
        assert!(single >= items, "items={items}: {single} units cannot give each its own");
        assert!(single >= width, "items={items}: never narrower than the batch width");
        assert!(single <= 32, "items={items}: never wider than the validated capacity");
    }
}
