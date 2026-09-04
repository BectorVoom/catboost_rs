//! Hardware-adaptive launch geometry for the elementwise `#[cube]` kernels.
//!
//! # Why this module exists
//!
//! Every launch helper in this crate used to hard-code its geometry as a literal
//! `CUBE_DIM = 32` cube width plus a hand-rolled `n.div_ceil(32)` cube count. On a
//! GPU that is a reasonable (if conservative) default — 32 is one wave32 wavefront on
//! the in-env gfx1151, so no SIMD lane idles. On the **CubeCL CPU runtime it is a
//! pathology**, and the CPU runtime is this crate's DEFAULT backend
//! (`cb-backend/Cargo.toml`: `default = ["cpu"]`), i.e. what a caller gets unless they
//! compile with `--features cuda|rocm|wgpu`.
//!
//! The cost model is not a guess; it is `cubecl-cpu-0.10.0`'s
//! `compute/runner.rs::execute_data`:
//!
//! * It dispatches **one OS-thread task per unit in the CUBE** — the loop is
//!   `for unit_pos_x in 0..cube_dim.x { for y { for z { worker.send_task(..) } } }` —
//!   and each iteration additionally clones the `MlirEngine` and `MlirData` and sends
//!   a second stop-message, then the host blocks on an `mpsc` receive until every unit
//!   reports back. Host cost per launch is therefore **O(cube_dim)**, and it is paid
//!   whether `n` is 8 or 8 million.
//! * `cube_count` is NOT dispatched. It is written into the kernel's builtins
//!   (`mlir_data.builtin.set_cube_count`) and the compiled MLIR loops over it inside
//!   each unit task, with `CubePos` as a block argument
//!   (`compiler/visitor/mod.rs:261`). So the cube count is a *serial in-kernel loop*,
//!   not parallelism.
//! * `if cube_dim_size > self.workers.len() { self.workers.extend(..) }` — a cube
//!   wider than `std::thread::available_parallelism()` **permanently grows the worker
//!   pool** past the machine's hardware parallelism.
//!
//! Put together, a fixed 32-wide cube on the 16-core dev box dispatched 32 tasks (and
//! spawned 16 surplus worker threads, 2x oversubscribed) to compute a gradient over
//! `n` elements, no matter how small `n` was. At roughly a microsecond per task
//! wake-up, a 1000-element `der1 = target - approx` spent far longer in thread-pool
//! synchronization than in the 1000 subtractions.
//!
//! [`launch_1d`] replaces that with a width chosen from the runtime's own reported
//! hardware, keeping the plane-aligned behaviour on GPU and scaling the CPU width with
//! the actual amount of work.
//!
//! # Why this is not a parity risk
//!
//! Geometry changes what runs where, so it is only safe where the result does not
//! depend on the schedule. This helper is for kernels that are **order-independent**:
//! one bounds-guarded write per lane, `out[ABSOLUTE_POS] = f(in[ABSOLUTE_POS])`, no
//! reduction and no atomic. Every elementwise loss kernel in `kernels.rs` is of that
//! shape (D-02 leaves every parity-critical SUM to the host via `cb_core::sum_f64`),
//! so the output buffer is bit-identical for any geometry that covers `[0, n)`.
//!
//! It is deliberately NOT applied to three families:
//!
//! * **The Poisson bootstrap draw.** Its geometry is pinned bit-for-bit to upstream
//!   `bootstrap.cu:66-70` and asserted by
//!   `poisson_grid_matches_upstream_launch_geometry`. The geometry IS the oracle.
//! * **The shared-memory block-reduce family** (`gpu_runtime::CUBE_DIM`, coupled by
//!   `const _: () = assert!(..)` to `kernels::BLOCK_REDUCE_SHMEM`). Those kernels fold
//!   floats in-cube, so the cube width selects the summation order and hence the
//!   rounding; widening it would move results at the ULP level.
//! * **The partition/pointwise histogram family** (`HIST_CUBE_DIM`,
//!   `PART_UPDATE_CUBE_DIM`, both 256). Those widths are already the product of
//!   measured tuning against official CatBoost GPU, and their rationale is documented
//!   at the constants.

// The production caller is `cpu_runtime.rs`, which is `#[cfg(feature = "cpu")]`, so
// under a `--no-default-features --features rocm|cuda|wgpu` build nothing outside the
// self-oracle calls into here. The module is still mounted under every backend (rather
// than gated to `cpu`) on purpose: `launch_geometry_test` exercises the GPU branch —
// plane alignment, the device units-per-cube limit, and the bit-identity of an
// elementwise kernel across cube widths — against the REAL device, which is the only
// place that branch can be checked. Gating the module to `cpu` would delete those
// tests from the one build that can run them.
#![allow(dead_code)]

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::{CubeCount, CubeDim};

/// Scalar element-operations one CPU unit should be worth before a second unit is
/// allocated.
///
/// A `cubecl-cpu` unit is an OS-thread task whose wake-up costs on the order of a
/// microsecond; at ~3 GHz that microsecond buys a few thousand cycles, i.e. tens of
/// thousands of scalar ops. Below this threshold a second unit is a net loss, because
/// the dispatch and the `mpsc` round-trip outweigh the arithmetic it takes away from
/// the first unit.
const WORK_PER_CPU_UNIT: usize = 32 * 1024;

/// Ceiling on the CPU cube width, independent of what the runtime reports.
///
/// `num_cpu_cores` comes from the platform and can be anomalous (container CPU
/// quotas, hosts reporting hundreds of SMT siblings). Since the width is also the
/// worker-pool size the runtime will grow to, an unbounded value would spawn threads
/// without limit; 64 is well past the point where dispatch overhead dominates for the
/// elementwise kernels this helper serves.
const CPU_CUBE_DIM_MAX: u32 = 64;

/// Whether the selected runtime executes on hardware SIMD planes (GPU warps /
/// wavefronts) rather than OS worker threads.
///
/// `plane_size_max == 1` is the CPU runtime's signature: it has no plane concept, its
/// `sync_cube` is a software barrier rather than a hardware one, and its "shared
/// memory" is ordinary heap with no bandwidth advantage over the CPU cache it already
/// lives in. Callers use this to skip shared-memory staging and barrier-based
/// algorithms that only pay off on a real GPU.
pub(crate) fn has_planes<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client.properties().hardware.plane_size_max > 1
}

/// Launch geometry for a 1-D, order-independent elementwise kernel over `lanes`
/// elements, where each lane performs roughly `work_per_lane` scalar operations.
///
/// Returns a `(CubeCount, CubeDim)` whose total unit span always covers `[0, lanes)`,
/// so a kernel of the standard `if ABSOLUTE_POS < n { out[ABSOLUTE_POS] = .. }` shape
/// writes every element exactly once — the same contract the previous hard-coded
/// `(n.div_ceil(32), 32)` provided.
///
/// * **GPU** (`plane_size_max > 1`): delegates to `CubeDim::new`, which builds the
///   cube from the device's own plane size and unit-per-cube limit, so the width is a
///   whole number of wavefronts on any device instead of an assumption about one.
/// * **CPU** (`plane_size_max == 1`): scales the width with the total work
///   (`lanes * work_per_lane`) in [`WORK_PER_CPU_UNIT`] steps, capped at the reported
///   core count, at `lanes` (never more units than there is work), and at
///   [`CPU_CUBE_DIM_MAX`]. Small launches collapse to a single unit — one task, no
///   pool growth, no clone storm — while large ones still fill the machine.
///
/// `lanes == 0` yields a single-unit, single-cube launch rather than
/// `CubeCount::Static(0, 0, 0)`. The bounds guard inside the kernel makes it a no-op,
/// and this preserves the `.max(1)` behaviour of the geometry it replaces: the HIP
/// backend is unforgiving about zero-extent work, and a degenerate empty column must
/// not turn into a backend-specific launch failure.
pub(crate) fn launch_1d<R: Runtime>(
    client: &ComputeClient<R>,
    lanes: usize,
    work_per_lane: usize,
) -> (CubeCount, CubeDim) {
    // Read the device properties ONCE and reuse them for both the plane check and the
    // CPU core count below, rather than paying a second `client.properties()` call.
    let hardware = &client.properties().hardware;
    let on_gpu = hardware.plane_size_max > 1;

    let cube_dim = if on_gpu {
        CubeDim::new(client, lanes.max(1))
    } else {
        let cores = hardware.num_cpu_cores.unwrap_or(1).max(1) as usize;
        let total = lanes.saturating_mul(work_per_lane.max(1));
        // `cores.min(lanes.max(1))` is the upper clamp: never more units than cores,
        // and never more units than elements to write.
        let units = (total / WORK_PER_CPU_UNIT).clamp(1, cores.min(lanes.max(1)));
        CubeDim::new_1d((units as u32).min(CPU_CUBE_DIM_MAX))
    };

    // `lanes.max(1)` keeps the `lanes == 0` case a launchable 1-cube grid (the HIP
    // backend is unforgiving about zero-extent work) rather than
    // `calculate_cube_count_elemwise`'s own `CubeCount::Static(0, 0, 0)` short-circuit
    // for a truly empty span.
    //
    // Routing through cubecl-core's own helper — instead of packing the whole cube
    // count into the x dimension by hand — spreads a large cube count across x/y/z so
    // it respects the device's per-dimension `hardware.max_cube_count` (e.g. WebGPU's
    // ~65535-per-dimension cap). This is safe for every kernel this helper serves
    // because `ABSOLUTE_POS` is the fully linearized unit index across the WHOLE grid
    // ("the position of the working unit in the whole cube kernel, without regards to
    // cubes and axis" — cubecl-core's own doc), so an elementwise
    // `out[ABSOLUTE_POS] = ..` kernel is indifferent to how the cube count is factored
    // across dimensions.
    let cube_count = cubecl::calculate_cube_count_elemwise(client, lanes.max(1), cube_dim);

    (cube_count, cube_dim)
}
