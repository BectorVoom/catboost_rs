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

//! # Measured and rejected
//!
//! Two changes that the cost model above suggests, and that measurement on the 8-core
//! Apple M1 said not to make. They are recorded so the next pass does not spend its
//! budget re-deriving them.
//!
//! * **`launch_unchecked`.** CubeCL's default `launch` compiles in `ExecutionMode::
//!   Checked`, which rewrites every array read and write into a bounds-tested one
//!   (`cubecl-core`'s `post_processing/checked_io.rs`), and the elementwise kernels here
//!   already carry their own `if ABSOLUTE_POS < approx.len()` guard, so the check is
//!   pure duplication and `launch_unchecked` would be sound. It buys nothing: measured
//!   over `gradient_kernel` and `logloss_gradient_kernel` at n = 10k / 100k / 1M, the
//!   unchecked launch ranged from 1.35x faster to 0.86x slower — noise in both
//!   directions. These lanes make three array accesses against an operand already in a
//!   register; the bounds test is a compare-and-select that the launch cost dwarfs. It
//!   is worth revisiting only for a kernel with a deep inner loop of gathers (a
//!   histogram fill), which is where the technique was originally measured to pay.
//! * **`create_from_slice` instead of `create(Bytes::from_elems(v.to_vec()))`.** The
//!   `to_vec()` looks like a wasted 8 MB copy at n = 1M. Measured, the copy costs
//!   0.26 ms and the whole `create` 0.84 ms, while `create_from_slice` of the same
//!   bytes costs 1.16-1.38 ms — slower, not faster. The upload is worth attacking by
//!   doing FEWER of them (`cpu_runtime::DerInputs`), not by changing how one is spelled.

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
///
/// This is the LOWER bound on the unit count — when to add the *second* unit. It says
/// nothing about when to stop adding them; that is [`LaneCost::ceiling`].
const WORK_PER_CPU_UNIT: usize = 32 * 1024;

/// What one lane of a kernel costs, and which physical resource that cost is drawn
/// from. Both halves are needed to pick a CPU width, and they answer different
/// questions:
///
/// * `ops` sets the LOWER bound — how much `n` must grow before a second OS thread
///   earns back its ~3 us dispatch (see [`WORK_PER_CPU_UNIT`]).
/// * `bandwidth_bound` sets the UPPER bound — whether more threads can still convert
///   into throughput at all, or whether the kernel already saturates the memory
///   system, past which extra units only add dispatch cost and contention.
///
/// Modelling only `ops` (as this module first did) gets the second question exactly
/// backwards: it reads "more total work" as "more units", when for a streaming kernel
/// more work is precisely the regime where extra units stop paying. Measured cost of
/// that mistake: 2.7x on the cheap tier at n = 1M (see [`STREAMING_LANE`]).
#[derive(Clone, Copy)]
pub(crate) struct LaneCost {
    /// Approximate scalar operations one lane performs.
    ops: usize,
    /// `true` when the lane's runtime is dominated by the DRAM traffic it moves
    /// rather than by its arithmetic.
    bandwidth_bound: bool,
    /// Objects one lane processes as a single `Vector<F, N>` (1 for a scalar lane).
    width: usize,
}

impl LaneCost {
    /// A lane whose cost is ARITHMETIC — it carries a transcendental or otherwise does
    /// enough work per byte that additional cores still convert into throughput.
    pub(crate) const fn compute(ops: usize) -> Self {
        Self {
            ops,
            bandwidth_bound: false,
            width: 1,
        }
    }

    /// A lane whose cost is MEMORY TRAFFIC — a handful of arithmetic operations over
    /// operands it must stream from DRAM.
    pub(crate) const fn streaming(ops: usize) -> Self {
        Self {
            ops,
            bandwidth_bound: true,
            width: 1,
        }
    }

    /// The cost of one lane that processes `width` objects as one `Vector<F, N>`.
    ///
    /// A vectorized lane does `width` objects' worth of arithmetic, so the op count
    /// scales with it while the `bandwidth_bound` classification does not: the lane
    /// moves the same bytes per object, only in wider loads. Feeding this to
    /// [`launch_1d`] together with `lanes = n / width` keeps the launch's TOTAL work —
    /// the quantity the lower bound is calibrated on — identical to the scalar launch,
    /// so the width decision does not shift just because the kernel became wider.
    pub(crate) const fn per_vector(self, width: usize) -> Self {
        let width = if width == 0 { 1 } else { width };
        Self {
            ops: self.ops.saturating_mul(width),
            bandwidth_bound: self.bandwidth_bound,
            width,
        }
    }

    /// The most units this lane cost can still turn into throughput on a `cores`-core
    /// host.
    ///
    /// A compute-bound lane scales with the cores it is given, so its ceiling is the
    /// core count. A bandwidth-bound lane does not: once enough threads are streaming
    /// to saturate the memory system, the remaining cores add no bandwidth, only
    /// dispatch cost (~3 us per unit, measured) and contention for the same cache
    /// lines and prefetchers. That is rule R7 of the CPU-kernel design manual — "do
    /// not parallelize what is already at bandwidth" — as an upper bound rather than
    /// as advice.
    ///
    /// `cores / 2` is a MEASURED, deliberately conservative saturation point, not a
    /// derivation: no runtime property exposes memory bandwidth, so the width at which
    /// a stream saturates cannot be computed from `client.properties()`. It is
    /// expressed as a fraction of the core count rather than an absolute so that it
    /// degrades sensibly on hosts other than the one it was calibrated on. See
    /// [`STREAMING_LANE`] for the measurements it comes from.
    fn ceiling(self, cores: usize) -> usize {
        if !self.bandwidth_bound {
            cores
        } else if self.width > 1 {
            // A vector lane streams the same bytes per object in a fraction of the
            // instructions, so ONE unit already pulls what several scalar units did and
            // the memory system saturates at a correspondingly narrower launch. See
            // [`STREAMING_LANE`], "Vector lanes": measured optimum of 1 unit at every
            // size on the 8-core calibration host.
            (cores / 8).max(1)
        } else {
            (cores / 2).max(1)
        }
    }
}

/// Ceiling on the CPU cube width, independent of what the runtime reports.
///
/// `num_cpu_cores` comes from the platform and can be anomalous (container CPU
/// quotas, hosts reporting hundreds of SMT siblings). Since the width is also the
/// worker-pool size the runtime will grow to, an unbounded value would spawn threads
/// without limit; 64 is well past the point where dispatch overhead dominates for the
/// elementwise kernels this helper serves.
pub(crate) const CPU_CUBE_DIM_MAX: u32 = 64;

/// Whether the selected runtime executes on hardware SIMD planes (GPU warps /
/// wavefronts) rather than OS worker threads.
///
/// `plane_size_max == 1` is the CPU runtime's signature: it has no plane concept
/// (`PLANE_DIM == 1`, so `PLANE_POS == UNIT_POS`), its `sync_cube` is a software
/// barrier rather than a hardware one, and its "shared memory" is ordinary heap with
/// no bandwidth advantage over the CPU cache it already lives in.
///
/// It is used to pick the GEOMETRY branch in [`launch_1d`] and to assert in
/// `launch_geometry_test` that each backend feature selects the runtime kind it
/// claims. It is deliberately NOT a switch for running different ALGORITHMS per
/// backend: the `#[cube]` kernels are one source serving both, kept correct on the
/// sequential-cube CPU runtime by the trailing-`sync_cube()` rule documented under
/// SHARED-MEMORY CUBE INDEPENDENCE at the top of `kernels.rs`, not by branching
/// shared-memory staging away on CPU.
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
    lane: LaneCost,
) -> (CubeCount, CubeDim) {
    // Read the device properties ONCE and reuse them for both the plane check and the
    // CPU core count below, rather than paying a second `client.properties()` call.
    let hardware = &client.properties().hardware;
    let on_gpu = hardware.plane_size_max > 1;

    let cube_dim = if on_gpu {
        CubeDim::new(client, lanes.max(1))
    } else {
        let cores = hardware.num_cpu_cores.unwrap_or(1).max(1) as usize;
        let total = lanes.saturating_mul(lane.ops.max(1));
        // Three upper clamps, each for a different reason: never more units than the
        // lane cost can still convert into throughput ([`LaneCost::ceiling`] — the
        // core count for a compute-bound lane, the memory-saturation width for a
        // streaming one), and never more units than there are elements to write.
        let ceiling = lane.ceiling(cores).min(lanes.max(1));
        let units = (total / WORK_PER_CPU_UNIT).clamp(1, ceiling);
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

/// The unit count for a launch of a `sync_cube` (shared-memory / barrier) kernel whose
/// algorithm was written for `requested` units per cube.
///
/// On a GPU this IS `requested`: the width is a hardware barrier over resident SIMD
/// lanes. On the CPU runtime every unit is an OS thread and `sync_cube` is a pure spin
/// barrier (`cubecl-cpu` `compute_task.rs`: `spin_loop`, no yield). More spinning
/// threads than cores is not merely slow, it is catastrophic: each barrier has to wait
/// for the scheduler to rotate every descheduled spinner back in. Measured on the 8-core
/// Apple M1, `full_scan` at 32 units per cube costs **~1 second per cube** (945-1157 ms
/// over 4 / 8 / 16 cubes, results correct), so a 100 000-element scan is a 50-minute
/// launch. That is the "hang" every large-`n` shared-memory test on this backend showed.
///
/// So the CPU width is the largest power of two that is `<= requested` and `<= cores`:
/// one spinner per core at most, and a power of two because the tree reductions stride
/// by `CUBE_DIM_X / 2` (`block_reduce_kernel`) and the Hillis-Steele scans double a
/// stride up to `CUBE_DIM_X`. Every barrier kernel in this crate derives its strides
/// and slot counts from the `CUBE_DIM_X` builtin — never from a literal 32 (D-09) —
/// and sizes its `SharedMemory` from the comptime `requested` (e.g.
/// [`crate::kernels::BLOCK_REDUCE_SHMEM`]), which stays an upper bound on the width,
/// so a narrower launch is the same algorithm over fewer slots. Hosts that compute
/// `num_cubes` MUST use the width this returns, not the constant they requested.
pub(crate) fn barrier_cube_dim<R: Runtime>(client: &ComputeClient<R>, requested: usize) -> usize {
    let requested = requested.max(1);
    if has_planes(client) {
        return requested;
    }
    let cores = client
        .properties()
        .hardware
        .num_cpu_cores
        .unwrap_or(1)
        .max(1) as usize;
    let cap = requested.min(cores);
    // Largest power of two <= cap (cap >= 1, so this is >= 1).
    1usize << (usize::BITS - 1 - cap.leading_zeros())
}

/// The vector width (`N` of `Array<Vector<F, N>>`) for a launch whose buffers are
/// allocated at EXACTLY the element count — no padding. Width 1 is the only width
/// every element count is a multiple of, so it is the only one such a launch may use.
///
/// The resident GPU der seams (`gpu_runtime::der_seams`) hold `approx`/`target`/`der`
/// handles sized to `n` for the whole training session and never pad them, so they
/// launch the vectorized kernels at this width. A `Vector<F, 1>` lowers to the scalar
/// type on every backend, so this is the pre-vectorization codegen, not a slow path.
/// Widening the device seams needs the padding contract [`der_line_size`] documents
/// and a device to measure it on; it is deliberately not done blind.
pub(crate) const SCALAR_LINE: usize = 1;

/// The vector width (`N` of `Array<Vector<F, N>>`) for the elementwise derivative
/// kernels on `client`, for a launcher that PADS its buffers to a multiple of it.
///
/// On the CPU runtime this is the widest of the device's `io_optimized_vector_sizes`
/// for the element — 8 for `f64` (a 512-bit load width), which the JIT lowers to
/// native SIMD loads and arithmetic. That is the whole win of the vectorized kernels
/// on this backend; see VECTORIZED ELEMENTWISE DER KERNELS in `kernels.rs` for the
/// measurements. It is queried, not hard-coded, so a host with a different load width
/// gets its own optimum.
///
/// On a GPU runtime it is [`SCALAR_LINE`]: the device launch sites do not pad, and the
/// device benefit of wider lines has not been measured (no device on the host this
/// was written on). Callers must size every buffer they pass to a multiple of the
/// returned width — the kernels index whole vectors, so a trailing partial vector is
/// silently skipped, not bounds-checked.
pub(crate) fn der_line_size<R: Runtime>(client: &ComputeClient<R>, elem_bytes: usize) -> usize {
    if has_planes(client) {
        return SCALAR_LINE;
    }
    client
        .io_optimized_vector_sizes(elem_bytes)
        .next()
        .unwrap_or(SCALAR_LINE)
        .max(SCALAR_LINE)
}

// ===========================================================================
// Per-lane work classifications for `launch_1d`.
//
// These live HERE, next to the cost model they feed, rather than in `cpu_runtime`.
// They are consumed by BOTH `cpu_runtime` (which is `#[cfg(feature = "cpu")]`) and
// `gpu_runtime::der_seams` (which is not), so a home inside the cpu-gated module made
// every `--no-default-features --features wgpu|cuda|rocm` build fail to compile with
// `unresolved import crate::cpu_runtime`. `launch_geometry` is mounted under every
// backend, so one definition now serves both callers — preserving the single source of
// truth the coupling was introduced for, without the cfg breakage.
// ===========================================================================

/// The cost of one lane of a TRANSCENDENTAL elementwise loss kernel: an f64
/// `exp`/`tanh`/`ln`/`powf` plus a divide (Logloss, Focal, LogCosh, Lq, Poisson,
/// Tweedie). Compute-bound — the arithmetic per lane is large enough that additional
/// cores still convert into throughput, so its unit count is capped only by the core
/// count.
///
/// The `16` is CALIBRATED, not counted. The instruction count per lane ranges from one
/// subtraction to an f64 `exp` plus a divide, so no single honest static number exists;
/// what the constant really has to do is put the unit count in the right place across
/// the range of `n` a fit actually sees. Measured on the 16-core dev box,
/// `compute_gradients(Logloss)`, best-of-3 in ms/call:
///
/// | n         |  ops=4 | ops=16 |  ops=64 |
/// |-----------|--------|--------|---------|
/// | 10 000    |  0.250 |  0.434 |   2.638 |
/// | 50 000    |  2.850 |  1.449 |   2.448 |
/// | 100 000   |  6.610 |  3.238 |   4.668 |
/// | 300 000   | 15.698 | 11.323 |  14.160 |
/// | 1 000 000 | 46.305 | 36.204 |  45.276 |
///
/// `4` was the first estimate and it is the wrong answer: it under-parallelizes the
/// mid-range, reaching only 12 of 16 units at n=100k, which measured SLOWER than the
/// hard-coded 32-wide geometry it replaced. `64` overshoots the other way and splits
/// launches too small to pay for the split. `16` is the only one of the three that
/// beats the old geometry at every size tested.
///
/// This replaces the former `const CUBE_DIM: usize = 32`, which fixed the geometry
/// regardless of both the hardware and `n`. See [`crate::launch_geometry`] for the
/// cost model that motivates the change.
pub(crate) const TRANSCENDENTAL_LANE: LaneCost = LaneCost::compute(16);

/// The cost of one lane of a CHEAP elementwise kernel: pure arithmetic and/or a
/// branch, no transcendental call. `target - approx` (RMSE), a sign/compare/divide
/// (MAPE), or a compare-and-select (Quantile/MAE, Huber, Expectile).
///
/// Two facts about such a lane, and each drives one half of [`LaneCost`]:
///
/// **`ops = 4` — the lower bound.** A handful of scalar operations, not the ~20-30
/// cycles an f64 `exp` plus a divide costs. This is an ANALYTICAL estimate from the
/// kernels' own instruction counts (`kernels.rs`: `gradient_kernel` /
/// `mape_gradient_kernel` / `quantile_gradient_kernel` / `huber_*_kernel` /
/// `expectile_*_kernel`), not a re-run of the [`TRANSCENDENTAL_LANE`] benchmark on this
/// workload — that table was measured against Logloss, whose true per-lane cost is
/// close to its 16. Using that SAME 16 for a ~1-op kernel would overstate its cost 16x
/// and parallelize it earlier than the real workload justifies. `4` is a deliberately
/// conservative floor above the true op count (not `1`), so a mis-estimate errs toward
/// under- rather than over-parallelizing.
///
/// **`streaming` — the upper bound.** The lane reads two f64 and writes one: 24 bytes
/// of DRAM traffic against ~4 flops, an arithmetic intensity of ~0.17 ops/byte. It is
/// memory-bound by construction, so it saturates the memory system at a width far
/// below the core count and gets worse — not merely flat — beyond it. Measured on the
/// 8-core Apple M1 (4 performance + 4 efficiency cores), `gradient_kernel` (RMSE der1,
/// f64), best-of-9 with the repetitions INTERLEAVED across widths (the schedule
/// discipline of `bench/perf_param_cpu/FINDINGS.md`), ms per launch + read-back:
///
/// | n         | w=1   | w=2   | w=3   | w=4   | w=6   | w=8   |
/// |-----------|-------|-------|-------|-------|-------|-------|
/// | 10 000    | 0.031 | 0.022 | 0.026 | 0.039 | 0.058 | 0.058 |
/// | 100 000   | 0.132 | 0.078 | 0.067 | 0.061 | 0.078 | 0.093 |
/// | 300 000   | 0.367 | 0.216 | 0.200 | 0.166 | 0.206 | 0.248 |
/// | 1 000 000 | 1.113 | 0.659 | 0.507 | 0.554 | 1.022 | 1.357 |
/// | 3 000 000 | 3.313 | 2.070 | 2.697 | 2.210 | 3.779 | 4.534 |
///
/// The optimum sits at 2-4 units at EVERY size and degrades sharply past it: at n=1M
/// the 8-wide launch the uncapped model chose is **2.7x** the 3-wide optimum, and at
/// n=3M it is 2.2x. Capping at `cores / 2` (= 4 here) lands within ~10% of the best
/// width at every size measured — see [`LaneCost::ceiling`].
///
/// That this is bandwidth saturation and not merely the M1's slow efficiency cores is
/// what the transcendental tier shows: the same 8-wide launch that costs the streaming
/// kernel 2.7x costs `logloss_gradient_kernel` nothing measurable at n=1M (1.12-1.44 ms
/// at w=8 against 1.20-1.31 ms at w=4). Straggling E-cores would penalize both kernels;
/// only the one at the memory ceiling is penalized.
///
/// **Vector lanes saturate at ONE unit.** The table above is for SCALAR lanes. Once the
/// same kernels ran over `Vector<f64, 8>` (VECTORIZED ELEMENTWISE DER KERNELS in
/// `kernels.rs`), the sweep was repeated on the same host — same interleaved
/// discipline, median of 11, `client.profile` kernel time in ms, every output
/// bit-identical to the scalar launch:
///
/// | kernel (vec 8) | n    | w=1   | w=2   | w=3   | w=4   | w=8   |
/// |----------------|------|-------|-------|-------|-------|-------|
/// | rmse           | 100k | 0.047 | 0.055 | 0.056 | 0.065 | 0.079 |
/// | rmse           | 1M   | 0.442 | 0.714 | 0.630 | 0.601 | 0.714 |
/// | rmse           | 3M   | 1.370 | 2.265 | 2.038 | 2.059 | 2.729 |
/// | huber          | 1M   | 0.545 | 0.750 | 0.660 | 0.608 | 0.774 |
/// | quantile       | 1M   | 0.606 | 0.752 | 0.659 | 0.625 | 0.781 |
/// | quantile       | 3M   | 1.654 | 2.499 | 2.162 | 2.085 | 3.420 |
///
/// One unit is the optimum at every size (300k is a tie within noise), and the scalar
/// tier's `cores / 2` width costs 1.35-1.5x at n >= 1M. A single vector unit moves
/// 8 MB x 3 in 0.44 ms — ~55 GB/s, the practical DRAM ceiling of this host — so there
/// is nothing left for a second unit to add. [`LaneCost::ceiling`] therefore caps a
/// vector streaming lane at `cores / 8` (= 1 here), keeping the fraction form so a
/// wider host with less bandwidth per core still gets a second unit. The transcendental
/// tier is again unaffected: `logloss` at vec 8 still improves monotonically to w=8
/// (1M: 6.97 / 3.67 / 2.55 / 1.95 / 1.80 ms), so its ceiling stays the core count.
pub(crate) const STREAMING_LANE: LaneCost = LaneCost::streaming(4);
