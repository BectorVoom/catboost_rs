---
phase: 1
slug: workspace-lint-discipline-oracle-harness
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-13
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (`cargo test`) + `approx` (`abs_diff_eq!`) for float assertions |
| **Config file** | none yet — Wave 0 creates root `Cargo.toml` workspace + `rust-toolchain.toml` |
| **Quick run command** | `cargo test -p cb-core` (PRNG vectors) / `cargo test -p cb-oracle` (comparator) |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | ~30 seconds (skeleton; grows with later phases) |

---

## Sampling Rate

- **After every task commit:** Run `cargo build -p <crate>` + `cargo test -p <crate>` for the touched crate.
- **After every plan wave:** Run `cargo clippy --workspace -- -D warnings && cargo test --workspace && scripts/check-no-anyhow.sh`.
- **Before `/gsd-verify-work`:** Full suite green + GitHub Actions CPU lane green.
- **Max feedback latency:** ~30 seconds.

---

## Per-Task Verification Map

> Authoritative per-task rows are owned by the PLAN.md files; this table maps each phase requirement to its automated verification. Planner refines into task-level rows.

| Requirement | Wave | Behavior | Test Type | Automated Command | File Exists | Status |
|-------------|------|----------|-----------|-------------------|-------------|--------|
| INFRA-01 | 0 | Workspace builds with all crates + features stubbed | build/smoke | `cargo build --workspace` + `cargo build -p cb-backend --features wgpu,cuda,rocm` | ❌ W0 | ⬜ pending |
| INFRA-02 | 0 | Restriction lints deny; `anyhow` absent from core libs | lint/script | `cargo clippy --workspace -- -D warnings` + `scripts/check-no-anyhow.sh` | ❌ W0 | ⬜ pending |
| INFRA-03 | 0 | One committed `.npy` fixture read + oracle assertion ≤1e-5 | unit/integration | `cargo test -p cb-oracle` (compare) | ❌ W0 | ⬜ pending |
| INFRA-04 | 0 | Per-stage comparator (borders/splits/leaf/approx) callable | unit | `cargo test -p cb-oracle` | ❌ W0 | ⬜ pending |
| INFRA-05 | 0 | `TFastRng64` reproduces vendored test vectors exactly | unit | `cargo test -p cb-core rng` | ❌ W0 | ⬜ pending |
| INFRA-06 | 0 | No inline `#[cfg(test)]` in production modules | convention/script | CI grep gate for `#[cfg(test)]` in non-`*_test.rs` `src/*.rs` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `Cargo.toml` (workspace root) — `[workspace]`, `[workspace.lints]`, members
- [ ] `rust-toolchain.toml` — pin stable
- [ ] `crates/cb-core/src/rng.rs` + `rng_test.rs` — covers INFRA-05
- [ ] `crates/cb-core/src/error.rs` — `thiserror` base (INFRA-02)
- [ ] `crates/cb-oracle/src/{fixture,compare}.rs` + `*_test.rs` — covers INFRA-03/04
- [ ] `crates/cb-oracle/generator/{requirements.txt,gen_inputs.py,gen_fixtures.py}` — build-time oracle generator (catboost==1.2.10 + numpy; never in CI)
- [ ] `crates/cb-oracle/fixtures/...` — at least one committed input dataset + scenario for the walking skeleton
- [ ] `.github/workflows/ci.yml` — CPU lane only (build + clippy + anyhow grep + oracle tests)
- [ ] `scripts/check-no-anyhow.sh` (+ `#[cfg(test)]` grep for INFRA-06)
- [ ] Stub crates `cb-data`/`cb-compute`/`cb-backend`/`cb-train`/`cb-model`/`catboost-rs` — `lib.rs` + `Cargo.toml` (INFRA-01)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GitHub Actions CPU lane is green | INFRA-01/02/03 | Requires push to GitHub; cannot run inside the local sandbox | After push, confirm the `ci.yml` workflow run passes on the PR/commit |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
