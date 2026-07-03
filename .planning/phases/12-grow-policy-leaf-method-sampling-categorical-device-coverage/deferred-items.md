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
