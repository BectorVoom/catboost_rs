# Spike Conventions

Patterns established across spike sessions. New spikes follow these unless the
question requires otherwise.

## Stack

- **Rust benchmarks/experiments**: an env-gated `#[test]` in the relevant crate's
  `tests/` dir (e.g. `crates/cb-train/tests/perf_baseline_test.rs`), INERT by
  default (`if std::env::var("CB_PERF").is_err() { eprintln!("SKIP …"); return; }`)
  so `cargo test` stays green. Print machine-greppable single-line records
  (`RSBENCH key=val …`). Always run perf tests with `--release`.
- **Oracle / head-to-head against upstream**: a Python script in the spike dir
  using the `.venv` (`catboost==1.2.10`), mirroring the Rust generator and params
  exactly, printing a matching greppable format (`CBBENCH key=val …`).

## Structure

- One dir per spike: `.planning/spikes/NNN-name/` with `README.md` (frontmatter +
  Results + Investigation Trail), companion scripts, and raw evidence (`*.txt`).
- Perf spikes report an **iters-normalized** metric (`per_tree_ms`) so a tiny
  iteration count stays representative and wall-clock bounded.

## Patterns

- **Isolate the layer under test.** To measure the host boosting loop without
  backend noise, use a device-declining `Runtime` (only impl `compute_gradients`;
  the `Ok(None)`/`Ok(false)` defaults force the CPU fallback path).
- **Separate algorithmic from constant-factor** cost by sweeping one axis at a
  time (n_rows, n_features, n_bins, depth) and reading the *slope*, not just the
  absolute gap. Linear-in-n_bins vs flat-in-n_bins is the histogram fingerprint.
- Match upstream generator with a portable hash (splitmix64) so Rust and Python
  train on the same-shaped data without exchanging files.

### Parallel-scaling spikes (005, 006)

- **Sweep threads with LOCAL pools, not the global one.** Build a
  `rayon::ThreadPoolBuilder::new().num_threads(p).build()` per thread count and run the
  work inside `pool.install(|| ...)`; report `speedup = t@1 / t@p`. Lets one process
  measure the whole 1/2/4/8/16-thread curve. Always warm up once inside the pool before
  timing (allocator / first-touch).
- **Separate the Amdahl ceiling from parallel inefficiency.** Microbench the serial phase
  vs the parallel phase in isolation to get `serial_fraction`, then compare the
  Amdahl-predicted ceiling `1/(f+(1-f)/p)` to the measured end-to-end speedup. A measured
  speedup *below* the ceiling means the parallel phase itself is weak (too-few tasks,
  in-closure allocation), not just Amdahl.
- **Prove parity by byte-identity, not tolerance.** For a restructuring that should NOT
  change numerics (e.g. feature-outer vs object-outer accumulation), assert
  `f64::to_bits()` equality cell-by-cell — a stronger, cheaper guard than the `<= 1e-5`
  oracle, and it certifies "no re-baseline needed".
- **Reuse production kernels in the prototype.** Benchmark strategies by composing the
  real `build_bucket_histogram` / `scan_and_score_borders` (feature-major bins let you
  slice one feature's column as `&bins[f*n..(f+1)*n]` and build a 1-feature histogram),
  so the spike's parity/scaling result transfers directly to the integration.
