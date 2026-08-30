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

use crate::launch_geometry::{has_planes, launch_1d};

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
        let (count, dim) = launch_1d(&client, n, 1);
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
    let (count, dim) = launch_1d(&client, 0, 1);
    assert_eq!(
        total_units(&count, dim) >= 1,
        true,
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
        let (_, dim) = launch_1d(&client, n, 1);
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
    let (_, big) = launch_1d(&client, 4_000_000, 1);
    assert!(
        big.num_elems() > 1,
        "4M elements must engage more than one unit on a {cores}-core host"
    );

    // ... but never past the cores, and never past the element count.
    for &n in &[1usize, 100, 100_000, 4_000_000, 100_000_000] {
        let (_, dim) = launch_1d(&client, n, 1);
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
    let (_, cheap) = launch_1d(&client, n, 1);
    let (_, expensive) = launch_1d(&client, n, 1024);
    assert!(
        expensive.num_elems() > cheap.num_elems(),
        "the same {n} lanes at 1024x the per-lane work must earn more units \
         (cheap={}, expensive={})",
        cheap.num_elems(),
        expensive.num_elems()
    );
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
        let (_, dim) = launch_1d(&client, n, 1);
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
