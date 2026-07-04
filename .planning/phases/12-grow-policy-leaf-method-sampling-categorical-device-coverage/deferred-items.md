# Phase 12 — Deferred Items (out-of-scope discoveries)

Logged per the executor SCOPE BOUNDARY rule: pre-existing issues in files NOT
touched by the current plan. Do NOT fix inside the discovering plan.

## From Plan 12-01

- **Pre-existing clippy `indexing_slicing` denials in `crates/cb-backend/src/cpu_runtime.rs`**
  - `cpu_runtime.rs:696` — `obj_approx[d] = approx.get(d * n + i)...` (indexing may panic)
  - `cpu_runtime.rs:1025` — `let approx_d = &approx[d * n..d * n + n];` (slicing may panic)
  - Surfaced by `cargo clippy -p cb-backend` (NOT by `cargo build` / `cargo test`).
  - File is unrelated to Plan 12-01 (the plan touched `session.rs`, `gpu_backend.rs`,
    `session_residency.rs`, `session_depth_gt1_test.rs`, `gpu_runtime/mod.rs`,
    `runtime.rs`, `lib.rs`). Pre-existing on the Plan-12 base commit.
  - Suggested fix (future hardening plan): replace the raw index/slice with
    `.get(..)` / `get(d*n..d*n+n)` guarded reads returning a typed `CbError` on
    out-of-range, consistent with the workspace `indexing_slicing = "deny"` lint.

## Plan 12-02 (discovered during verification)

- Pre-existing `cargo clippy` debt in `cb-backend` (`indexing_slicing`, `slicing may panic`) and `cb-oracle` (`indexing may panic`) blocks a full-graph `cargo clippy -p cb-model`. Unrelated to the Region path work (those crates were not modified). `cargo build` / `cargo test -p cb-model` / `cargo test -p cb-train` all pass. Out of scope for 12-02 (SCOPE BOUNDARY — pre-existing, unrelated files).

## Plan 12-06 (discovered during execution)

- **In-env ROCm runtime currently does NOT advertise `Atomic<u64>` add for the resident partition histogram.** Every depth>=1 resident-grow test fails FAST (~0.12s) with `Unsupported("partition-aware histogram fill requires Atomic<u64> add ...")` at `gpu_runtime/mod.rs:1826` (`device_supports_u64_atomic_add(client)` returns `false`). This affects the PRE-EXISTING Plan-01/05/11 grow oracles identically (`session_depth_gt1_grows_and_matches_direct`, `session_exact_leaf_grows_finite_quantile_leaves`, `session_residency_matches_cpu_multi_tree_boosting`) — NOT introduced by Plan 06 (which touched only `bootstrap_device.rs`, `session.rs`, `cb-core/rng.rs`). The Plan-06 device bootstrap kernels use plain u64 **arithmetic** (not u64 atomics) and run bit-for-bit correctly on gfx1100 (7/7 bootstrap self-oracle + gate tests green). Memory note `phase10-03-reduce-determinism-spike` records gfx1100 DID advertise `Atomic<u64>` add previously, so this is an ENVIRONMENT/driver capability-state regression, not a code regression.
  - **Impact:** the whole resident-grow oracle suite is red on rocm in-env until the capability is restored (driver/runtime reload). The Plan-06 e2e wiring test `session_bootstrap_grows_finite_tree` gracefully SKIPS on this `Unsupported` capability error (WR-01 capability-skip pattern) rather than adding a new hard failure.
  - **Out of scope for 12-06** (SCOPE BOUNDARY — pre-existing, environment-wide, unrelated to the bootstrap draw). Suggested next step: diagnose the ROCm/HIP `atomic_type_usage(Atomic<u64>)` advertisement on gfx1100 (driver/ROCm version) before the Phase-12 verifier re-runs the grow oracles.
