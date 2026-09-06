# CPU-runtime kernel design pass — measured results

Machine: **8-core Apple M1 (4 performance + 4 efficiency cores)**, `cubecl` 0.10.0,
`--features cpu` (the crate default). Workload: `CpuBackend::compute_gradients`, the
only path that runs `#[cube]` kernels on the CPU backend.

Method follows the schedule discipline of `FINDINGS.md`: warm the JIT first (the first
launch of each `(kernel, comptime args, cube_dim)` includes a 25–60 ms MLIR compile),
then **median of 15**, and for the width sweep the repetitions are **interleaved across
widths** rather than run as a block per width.

The two changes below are geometry and data-transfer only. No kernel arithmetic
changed, so every output is bit-identical; `cpu_runtime_test`'s per-loss host-reference
parity tests and its `dim1_is_byte_identical_to_scalar_path` /
`multiquantile_dim1_equals_scalar_quantile` anchors all pass unchanged.

---

## 1. The launch width had no upper bound (manual R7)

`launch_1d` chose `units = lanes * work_per_lane / 32K`, clamped to the core count. That
models when to add the *second* unit and nothing about when to stop. For a kernel that
is memory-bound — 24 bytes of DRAM traffic against ~4 flops per lane — "more total
work" is precisely the regime where extra threads stop paying, and on this host they do
worse than nothing.

`gradient_kernel` (RMSE der1, f64), ms per launch + read-back, interleaved best-of-9:

| n | w=1 | w=2 | w=3 | w=4 | w=6 | w=8 |
|---|---|---|---|---|---|---|
| 10 000 | 0.031 | **0.022** | 0.026 | 0.039 | 0.058 | 0.058 |
| 100 000 | 0.132 | 0.078 | 0.067 | **0.061** | 0.078 | 0.093 |
| 300 000 | 0.367 | 0.216 | 0.200 | **0.166** | 0.206 | 0.248 |
| 1 000 000 | 1.113 | 0.659 | **0.507** | 0.554 | 1.022 | 1.357 |
| 3 000 000 | 3.313 | **2.070** | 2.697 | 2.210 | 3.779 | 4.534 |

The old model picked **w=8** at every `n` ≥ 100k. At n=1M that is **2.7×** the optimum,
at n=3M **2.2×**.

**It is bandwidth saturation, not the M1's slow efficiency cores.** The same 8-wide
launch that costs the streaming kernel 2.7× costs `logloss_gradient_kernel` — same
loads, same stores, plus an f64 `exp` — nothing measurable at n=1M (1.12–1.44 ms at w=8
against 1.20–1.31 ms at w=4). Straggling E-cores would penalize both; only the kernel at
the memory ceiling is penalized.

**Fix.** `launch_1d` now takes a `LaneCost { ops, bandwidth_bound }` instead of a bare
op count. `ops` keeps setting the lower bound; `bandwidth_bound` sets the upper one —
`cores` for a compute-bound lane, `cores / 2` (a measured, conservative saturation
point, expressed as a fraction so it degrades sensibly off this host) for a streaming
one. The two tier constants became `TRANSCENDENTAL_LANE` and `STREAMING_LANE`.

The transcendental tier is deliberately **unchanged**: its `ops = 16` was calibrated on
the 16-core box in `FINDINGS.md` and capping it here would be an unmeasured regression
risk there.

## 2. Every two-derivative loss uploaded its inputs twice (manual R6)

Each derivative had a self-contained launcher that re-created the client, re-uploaded
its inputs, launched, and read back. So `Loss::LogCosh` uploaded `approx` and `target`
**four** times for two kernels, and the separable multi-dimension loop and MultiQuantile
re-uploaded the shared `target` once **per dimension**.

On this runtime an upload costs most of a kernel launch. At n = 1M (f64, 8 MB/vector):

| piece | ms |
|---|---|
| `client.create(Bytes::from_elems(v.to_vec()))` | 0.84 |
| `client.empty(8 MB)` | 0.00 |
| `read_one(8 MB)` + cast to `Vec` | 0.34 |
| the `logloss_gradient_kernel` launch itself | ~1.2 |

**Fix.** `cpu_runtime::DerInputs` uploads `(approx, target)` once and every launch
clones the handles into itself. `with_approx` swaps in one new slice while keeping the
shared `target`, which is what the separable and MultiQuantile loops now use.

## Combined result

`CpuBackend::compute_gradients`, median-of-15, three runs each, ms per call:

| | baseline | + R7 cap | + R6 shared uploads | total |
|---|---|---|---|---|
| **n = 100 000** | | | | |
| Rmse | 0.592–0.596 | 0.495–0.525 | 0.466–0.476 | **1.26×** |
| Logloss | 0.940–0.983 | 0.933–1.139 | 0.773–0.788 | **1.23×** |
| LogCosh | 1.312–1.317 | 1.284–1.305 | 0.836–0.847 | **1.56×** |
| Huber | 1.063–1.132 | 1.041–1.062 | 0.631–0.667 | **1.68×** |
| **n = 1 000 000** | | | | |
| Rmse | 4.656–4.883 | 3.996–4.136 | 4.171–4.232 | **1.13×** |
| Logloss | 7.298–7.758 | 7.042–7.443 | 6.483–6.675 | **1.14×** |
| LogCosh | 8.663–10.064 | 8.641–9.016 | 7.144–7.239 | **1.28×** |
| Huber | 8.296–10.391 | 7.173–7.260 | 5.124–5.242 | **1.73×** |

Logloss and LogCosh are transcendental-tier, so their R7 column is unchanged by design
and the movement there is run-to-run noise; their gain is entirely R6.

## Measured and rejected

Recorded so the next pass does not re-derive them. Both are in the `launch_geometry`
module doc as well.

* **`launch_unchecked`.** CubeCL's default `launch` compiles in `ExecutionMode::Checked`,
  which rewrites every array access into a bounds-tested one, and these kernels already
  carry their own `if ABSOLUTE_POS < approx.len()` guard — so the check is duplication
  and removing it would be sound. Measured over `gradient_kernel` and
  `logloss_gradient_kernel` at n = 10k/100k/1M, it ranged from 1.35× faster to 0.86×
  slower: noise in both directions. These lanes make three array accesses against an
  operand already in a register. Worth revisiting only for a kernel with a deep inner
  loop of gathers.
* **`create_from_slice` instead of `create(Bytes::from_elems(v.to_vec()))`.** The
  `to_vec()` looks like a wasted 8 MB copy. It costs 0.26 ms and the whole `create`
  0.84 ms, while `create_from_slice` of the same bytes costs 1.16–1.38 ms — slower. The
  upload is worth attacking by doing fewer of them, not by respelling one.

## Two facts about this host worth knowing

Neither is a defect introduced here; both cost time to rediscover.

* **`cubecl-cpu` 0.10 implements no atomics.** A kernel accumulating into `Atomic<f64>`
  panics the device worker at MLIR-lowering time (`compiler/visitor/elem.rs:38`,
  `not yet implemented: atomic<f64>`) — on a worker thread, while the host blocks, so
  the visible symptom is a hang or a silently all-zero output buffer rather than an
  error. `gpu_runtime::device_supports_channel_atomic_add` gates every such fill before
  launch and is load-bearing on this backend, not defensive.
* **20 of the `cb-backend` lib tests cannot pass under `--features cpu`** for that
  reason (`kernels::pairwise_hist`, `kernels::score_split`): they `.unwrap()` the
  `CbError::Unsupported` the gate returns. They are rocm/cuda tests
  (`run_device_tests.sh`); `kernels::pointwise_hist` skips instead, via
  `channel_atomics_available()`. Verified identical on the pre-change tree.
* **Run the suite with `--test-threads=1`.** `sync_cube` on `cubecl-cpu` is a
  process-global spin barrier, so parallel test threads each launching shared-memory
  kernels thrash on it — a run wedged for ~50 minutes in `reorder_one_bit_scatter_kernel`
  before being killed.

---

# Pass 2 — profile-driven: vectorized kernels and a one-copy upload

Same machine (8-core Apple M1), same `cubecl` 0.10.0, same workload
(`CpuBackend::compute_gradients`). Pass 1 changed geometry and transfer *count*; this
pass changed the kernels' element type and the transfer *mechanism*. Every output is
still bit-identical: the `vec` sweep below asserts byte equality of every configuration
against the scalar single-unit launch, and `cpu_runtime_test` gained a padded-tail sweep
(`n` = 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 100, 1001) over six kernel families.

Reproduce (the harness lives in the crate now):

```
cargo run -p cb-backend --release --example der_kernel_profile -- e2e 15
cargo run -p cb-backend --release --example der_kernel_profile -- vec 11
```

## 3. Where the time actually went (sampling profile)

`sample` (macOS) over a loop of `compute_gradients` for RMSE / Logloss / Huber at
n = 1M, 4300 samples on the main thread. Heaviest leaf frames process-wide:

| leaf frame                          | samples | what it is                                   |
|-------------------------------------|--------:|----------------------------------------------|
| `semaphore_wait_trap`               |  34 120 | worker threads idle / host waiting on kernel |
| `exp` (libsystem_m)                 |   3 347 | the Logloss kernels' transcendental          |
| `_platform_memmove`                 |   2 283 | **host-side copies**                         |
| JIT'd kernel code (anonymous)       |  ~4 800 | the kernels themselves                       |

The memmove was not the kernels: 1 300 of the main thread's 4 300 samples were inside
`DerInputs::new`, and every one of them was a copy of the same 8 MB:

1. our `approx.to_vec()` (517 samples),
2. `Bytes::from_elems` → `try_enforce_runtime_align` → `alloc_with_data` — the
   runtime re-allocating the `Vec` to its own alignment (361),
3. `ComputeClient::do_create` → `Bytes::from_bytes_vec(data.to_vec())` — the client
   copying the `Bytes` **again** before enqueueing (416),
4. and, on the queue thread, `copy_from_slice` into a freshly `alloc_zeroed` pool
   page (paid as page faults during the copy).

Four copies to upload one vector. Measured in isolation, two 8 MB vectors:

| upload path                                              | ms (median of 21) |
|----------------------------------------------------------|------------------:|
| `client.create(Bytes::from_elems(v.to_vec()))` + sync    | 2.216             |
| `client.empty` + `client.get_resource` + one `memcpy`    | 0.760             |

**Fix.** On the CPU runtime the storage resource *is* host memory (`BytesResource`),
and `client.get_resource(handle)` hands it out after draining the stream.
`cpu_runtime::upload_padded` allocates with `client.empty`, maps the resource, and
copies the caller's slice straight into the pool — one copy, no intermediate `Vec`,
no alignment re-copy, no queue task. Pass 1's `create_from_slice` note still stands:
respelling the `create` call could not have helped, because the copies were inside it.

## 4. The kernels were scalar (manual: Dynamic Vectorization)

The elementwise der kernels were `Array<F>`: one load / one op / one store per object,
and on a memory-bound lane the per-object loop and bounds test dominate. All 19 now
take `Array<Vector<F, N>>` with the width `N` chosen per device at launch
(`launch_geometry::der_line_size`: the widest `io_optimized_vector_sizes` for `f64`,
8 on this runtime). Branches became `select_many` lane selects — both arms computed,
the scalar arm's value selected — which is what keeps the results bit-identical rather
than merely close.

Buffers are padded to a multiple of the width (`DerInputs`), because a kernel that
indexes whole vectors cannot bounds-check a trailing partial one: the CPU launcher pads
and truncates, so an arbitrary `n` (1 000 003 is in the e2e table) stays on the vector
path. The resident GPU der seams keep their exact-`n` handles and launch at width 1,
which lowers to the codegen they had.

Kernel-only time (`client.profile`), median of 11, interleaved, ms:

| kernel   | n    | scalar w=1 | vec8 w=1 | vec8 best-w | gain (w=1) |
|----------|------|-----------:|---------:|------------:|-----------:|
| rmse     | 100k |      0.127 |    0.047 | 0.047 (1)   | 2.7×       |
| rmse     | 1M   |      1.154 |    0.442 | 0.442 (1)   | 2.6×       |
| rmse     | 3M   |      3.350 |    1.370 | 1.370 (1)   | 2.4×       |
| huber    | 1M   |      1.533 |    0.545 | 0.545 (1)   | 2.8×       |
| quantile | 1M   |      1.736 |    0.606 | 0.606 (1)   | 2.9×       |
| logloss  | 1M   |      7.979 |    6.966 | 1.795 (8)   | 1.15×      |
| logloss  | 3M   |     24.081 |   21.013 | 5.774 (4)   | 1.15×      |

The transcendental tier gains only the wider loads and stores: the backend scalarizes
`exp` on a vector, so its cost is unchanged and it still needs the whole machine.

## 5. A vector streaming lane saturates memory on one unit (manual R7, revisited)

Pass 1 capped streaming lanes at `cores / 2` from a *scalar* sweep. Repeating that sweep
with the vector kernels moved the optimum to a single unit at every size:

| kernel (vec8) | n    | w=1   | w=2   | w=3   | w=4   | w=8   |
|---------------|------|------:|------:|------:|------:|------:|
| rmse          | 1M   | 0.442 | 0.714 | 0.630 | 0.601 | 0.714 |
| rmse          | 3M   | 1.370 | 2.265 | 2.038 | 2.059 | 2.729 |
| quantile      | 3M   | 1.654 | 2.499 | 2.162 | 2.085 | 3.420 |

One unit moves 3 × 8 MB in 0.44 ms, ~55 GB/s — this host's practical DRAM ceiling —
so a second unit only adds contention. `LaneCost` now carries the lane width, and the
streaming ceiling for a vector lane is `cores / 8` (= 1 here), kept as a fraction so a
host with less bandwidth per core still gets a second unit. The scalar ceiling and the
transcendental tier are untouched (logloss vec8 improves monotonically to w=8:
6.97 / 3.67 / 2.55 / 1.95 / 1.80 ms at 1M).

## Combined result

`CpuBackend::compute_gradients`, median of 15 (min in parentheses), ms per call.
"pass 1" is the tree this pass started from.

| | pass 1 | + one-copy upload + vec8 kernels + w=1 streaming | speedup |
|---|---|---|---|
| **n = 100 000** | | | |
| Rmse | 0.447 (0.388) | 0.210 (0.144) | **2.1×** |
| Logloss | 0.880 (0.795) | 0.639 (0.542) | **1.4×** |
| LogCosh | 0.899 (0.868) | 0.608 (0.535) | **1.5×** |
| Huber | 0.653 (0.547) | 0.311 (0.247) | **2.1×** |
| Quantile | 0.505 (0.456) | 0.215 (0.162) | **2.3×** |
| **n = 1 000 000** | | | |
| Rmse | 4.160 (3.786) | 2.708 (2.241) | **1.5×** |
| Logloss | 7.369 (7.250) | 5.703 (5.264) | **1.3×** |
| LogCosh | 7.105 (6.925) | 5.432 (5.068) | **1.3×** |
| Huber | 5.281 (4.605) | 3.420 (2.963) | **1.5×** |
| Quantile | 3.914 (3.631) | 2.539 (2.184) | **1.5×** |
| **n = 1 000 003** (not a vector multiple) | | | |
| Rmse | 4.304 (3.800) | 2.678 (2.060) | **1.6×** |
| Logloss | 7.371 (7.109) | 5.686 (5.236) | **1.3×** |
| LogCosh | 7.049 (6.851) | 5.309 (4.993) | **1.3×** |
| Huber | 4.999 (4.550) | 3.429 (2.900) | **1.5×** |
| Quantile | 4.028 (3.900) | 2.592 (2.158) | **1.6×** |

The "after" column is the middle of three consecutive runs; the runs agree to within 3%.
What is left at n = 1M for RMSE (2.7 ms): ~0.8 ms for the two uploads (one 8 MB memcpy
each, page faults included), ~0.45 ms kernel, ~0.25 ms read-back copy into the returned
`Vec`, ~0.13 ms for the constant `der2` fill, and the rest is the runtime's own
scheduling. The transcendental losses are now dominated by `exp` itself (≈ 1.8 ms per
kernel at w=8) — the next lever there is a vectorized `exp`, which the backend does not
provide.

## Measured and rejected, this pass

* **`Bytes::try_into_vec` for the read-back.** The read side was already zero-copy
  (`read_one` returns a view onto pool memory, 0.04 ms); the remaining 0.2 ms is the
  `to_vec` into the `Vec<f64>` the trait returns. `try_into_vec` cannot take that
  allocation over — the pool's allocation alignment is not `f64`'s — so it fails on
  every call (0 of 15) and the copy stays. Attacking it means changing the trait's
  return type, which is not a kernel change.
* **Wider units for vectorized streaming lanes.** See §5: every width above 1 was
  slower at n ≥ 1M and no better at 100k.

## 6. The shared-memory kernels were not hanging — they were 5000× oversubscribed

Every large-`n` test of a `SharedMemory` kernel on this backend (`scan`, `reduce`,
`sort`, `segmented_sort`, `exact_quantile`, the `ranking_stoch` sort path — 17 tests)
"hung": a 45 s watchdog killed each one, and one full-suite run sat at 741 % CPU for
2.5 hours in `full_scan_inclusive_f32_large_n`. Sampling the process put every thread
in `cubecl_cpu::compute_task::sync_cube`, but there was no barrier mismatch — the
scan simply had not finished:

| `full_scan`, 32 units/cube | cubes | ms      | ms / cube |
|----------------------------|------:|--------:|----------:|
| n = 128                    |     4 |  3 780  |     945   |
| n = 256                    |     8 |  9 261  |   1 158   |
| n = 512                    |    16 | 17 521  |   1 095   |

`sync_cube` on `cubecl-cpu` 0.10 is a pure spin barrier (`spin_loop`, never yields).
Every unit is an OS thread, so 32 units on 8 cores means each of the ~12 barriers a
scan cube takes has to wait for the scheduler to rotate the descheduled spinners back
in — tens of milliseconds per barrier, a second per cube, 50 minutes for the
100 000-element test. Small tests passed only because a single cube fits inside the
watchdog.

**Fix.** `launch_geometry::barrier_cube_dim(client, requested)` returns `requested` on
a GPU and, on the CPU runtime, the largest power of two `<= min(requested, cores)` (a
power of two because the tree reductions stride by `CUBE_DIM_X / 2`). The block-reduce
family's `gpu_runtime::CUBE_DIM = 32` stays the requested width and the `SharedMemory`
capacity; every launch site now sizes its cube, cube count and grid stride from the
memoized `gpu_runtime::cube_dim()`, and the three single-cube one-unit-per-item scans
(`launch_block_scan_f64`, `scan_update_pointwise`, `scan_update_pairwise`) use
`single_cube_dim(items)`, which covers their items without exceeding the validated
capacity. The kernels themselves were already width-agnostic (strides from `CUBE_DIM_X`,
never a literal 32 — D-09), so no kernel changed. Same scan, 8 units per cube:

| `full_scan`, 8 units/cube | cubes | ms    | ms / cube |
|---------------------------|------:|------:|----------:|
| n = 128                   |     4 |  0.62 |   0.155   |
| n = 512                   |    16 |  0.92 |   0.058   |
| n = 4 096                 |   128 |  5.14 |   0.040   |

Results are identical. The 17 tests now take well under a second each.

The 20 tests that FAILED rather than hung were the documented rocm/cuda-only atomics
set (`pairwise_hist`, `score_split`, `grow_loop::{pairwise,partition}`): they
`unwrap`ped the `CbError::Unsupported` the capability gate returns on this backend.
They now skip with the same `channel_atomics_available()` guard `pointwise_hist`
already used, so a missing runtime feature is reported as a skip, not as a numerical
regression.
