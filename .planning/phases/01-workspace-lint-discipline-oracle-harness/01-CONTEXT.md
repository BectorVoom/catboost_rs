# Phase 1: Workspace, Lint Discipline & Oracle Harness - Context

**Gathered:** 2026-06-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Foundational, greenfield project scaffolding plus parity-testing infrastructure — so that every subsequent algorithm is "born oracle-gated and lint-clean." No CatBoost algorithm ships in this phase; the deliverables are the modular Cargo workspace, the lint/anyhow enforcement gates, the oracle harness (with per-stage comparison support and a frozen input corpus), and the exact `TFastRng64` PRNG port (INFRA-01 … INFRA-06).

Decisions made here are load-bearing for all 7 later phases (crate boundaries, the generic `R: Runtime` seam, the oracle fixture format, and the parity-gate mechanics).

</domain>

<decisions>
## Implementation Decisions

### Crate Decomposition (INFRA-01)
- **D-01:** Day-one workspace is fine-grained — all responsibility crates stubbed now so boundaries are fixed early and never refactored. Full set: `cb-core`, `cb-data`, `cb-compute`, `cb-backend`, `cb-train`, `cb-model`, `cb-oracle`, `catboost-rs` (facade).
- **D-02:** **One** backend crate, not four. `cb-backend` is the sole owner of all CubeCL / runtime-specific code, feature-gated `cpu`/`wgpu`/`cuda`/`rocm`, resolving to a single `cfg`-gated runtime type alias (zero runtime dispatch). There is NO per-backend crate. (User explicitly corrected the initial 4-crate proposal.)
- **D-03:** `cb-compute` stays **pure generic** — it owns the generic `R: Runtime` / `F: Float` algorithm boundary and must NOT depend on CubeCL. All CubeCL coupling lives in `cb-backend`. This keeps the Phase 3 generic seam clean.
- **D-04:** Naming convention: internal crates use the `cb-` prefix; the public/published facade crate (the one Rust users add to `Cargo.toml` and that the Phase 8 PyO3 bindings wrap) is named **`catboost-rs`**.
- **D-05:** Crate responsibilities map to phases: `cb-core` (errors/shared types, P1), `cb-data` (Pool/quantization, P2), `cb-compute` (generic boundary + kernels, P3), `cb-backend` (CubeCL runtime alias, P7), `cb-train` (boosting loop, P3), `cb-model` (serialize/SHAP, P4), `cb-oracle` (harness + fixtures, P1), `catboost-rs` (Builder facade, P4).

### Oracle Extraction Method (INFRA-03, INFRA-04)
- **D-06:** **Python-first hybrid.** The primary oracle generator is the pinned pip `catboost` package, used for everything it exposes: borders (`get_borders()`/quantization), tree splits + leaf values (`save_model(format='json')`), per-iteration approximants (`staged_predict`), SHAP / feature importance, and final predictions.
- **D-07:** **Oracle version pinned to `catboost==1.2.10`** — matches the vendored `catboost-master/` source (`catboost-master/catboost/python-package/catboost/version.py` = `1.2.10`) we are porting from. This pin is mandatory for algorithmic parity; do not bump it independently of the vendored source.
- **D-08 [informational]:** (Phase 5 deferral; no Phase 1 work) C++ instrumentation of `catboost-master/` is **deferred per-phase**, used ONLY where the Python API cannot reach the needed internal value (primarily Phase 5 ordered-CTR per-object target statistics, which is already flagged NEEDS DEEPER RESEARCH). Phase 1–4 require no C++ build.

### Fixture Storage & Scope (INFRA-03, INFRA-04)
- **D-09:** Fixture format is a **hybrid: `config.json` (metadata — seed, version, params) + `.npy` binary arrays** (borders, leaf values, approximants, predictions). Rationale: `.npy` gives exact, compact f64 round-trip; config metadata stays human-readable. Rust reads `.npy` via the `ndarray-npy` crate; Python writes via NumPy.
- **D-10:** Fixtures live under `cb-oracle/fixtures/`. Layout: `cb-oracle/fixtures/inputs/<dataset>/` for frozen input datasets; `cb-oracle/fixtures/<scenario>/` for per-phase expected-output fixtures that reference an input dataset.
- **D-11:** **Frozen shared input corpus.** Phase 1 generates & commits a small canonical set of synthetic INPUT datasets once (pinned seed; varied shapes — e.g. tiny numeric, small categorical, grouped/ranking). Every later phase draws inputs from this corpus and only adds its own expected-OUTPUT fixtures. This isolates algorithm drift from input drift across all 8 phases.
- **D-12:** Determinism (locked from ROADMAP success criteria, restated for downstream): all oracle generation uses a pinned seed and `thread_count=1`; comparison tolerance is absolute error ≤ 1e-5; fixtures are committed frozen so the generator does not run in CI.

### Lint & CI Enforcement (INFRA-02, INFRA-06)
- **D-13:** Deny-lints enforced via a single centralized **`[workspace.lints]`** table at the workspace root (`clippy::unwrap_used`, `clippy::expect_used`, `clippy::panic`, `clippy::indexing_slicing` = `"deny"`); each library crate opts in with `[lints] workspace = true`. DRY, native, no per-file attribute drift.
- **D-14:** `anyhow` ban is **structural**: core library crates (`cb-core`, `cb-data`, `cb-compute`, `cb-backend`, `cb-train`, `cb-model`) do NOT list `anyhow` in `[dependencies]` — making it impossible to use in non-test code. `anyhow` is permitted only as a dev-dependency (tests) and at app/binding edges. A CI grep (`scripts/check-no-anyhow.sh`) is the belt-and-suspenders backstop.
- **D-15:** `thiserror` is the library error strategy; `anyhow` only at binding/app/test edges (per PROJECT.md constraint). Note for planning: whether the `catboost-rs` facade counts as a thiserror-only "core lib" vs an "edge" was raised but not finalized — default to treating the facade as a library (thiserror) unless planning surfaces a reason otherwise.
- **D-16:** CI is split into two lanes because **ROCm cannot run in GitHub Actions** (no AMD-GPU runners — user-confirmed constraint):
  - **GitHub Actions (CPU lane, all of Phase 1):** `.github/workflows/ci.yml` runs `cargo build` (cpu feature), the clippy lint gate (`-D warnings`), the anyhow grep backstop, and the CPU oracle harness tests.
  - **ROCm GPU lane (Phase 7):** GPU/ROCm tests are defined as standalone runnable scripts (e.g. `scripts/gpu-rocm-tests.sh`) that run on local/self-hosted AMD hardware, wired to a self-hosted runner in Phase 7. Phase 1 ships only the CPU GitHub Actions workflow.
- **D-17:** Source/test separation (INFRA-06) enforced as a convention from the first commit — no inline `#[cfg(test)]` in production modules; tests in dedicated `*_test.rs` files / `tests/` dir (per CLAUDE.md mandatory rule). Do NOT replicate the inline-`mod tests` pattern of the vendored upstream rust-package.

### Claude's Discretion
- Exact dataset shapes / row counts / feature mixes of the frozen input corpus (D-11) — choose reasonable small synthetic datasets that exercise numeric, categorical, and grouped/ranking paths; planning may refine.
- `TFastRng64` bitstream-capture mechanics (INFRA-05) — how to emit the C++ generator's raw bitstream for a fixed seed for the bitstream oracle. Upstream source is at `catboost-master/util/random/fast.h` / `fast.cpp` with a unit test at `catboost-master/util/random/fast_ut.cpp`. Whether to capture via a tiny C++ harness or transcribe known test vectors is left to research/planning.
- Rust edition / MSRV / toolchain pinning specifics — apply CLAUDE.md constraints (latest stable Rust, latest crate versions) using best judgment.
- Whether `cb-backend`'s `cpu` feature uses CubeCL's CPU runtime or a plain non-CubeCL CPU path in early phases — defer to Phase 3/7 planning; Phase 1 only stubs the crate.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project & Roadmap
- `.planning/PROJECT.md` — core value, constraints, key decisions (CubeCL, PyO3, oracle strategy, thiserror/anyhow).
- `.planning/ROADMAP.md` § "Phase 1: Workspace, Lint Discipline & Oracle Harness" — goal + 5 success criteria this phase is judged against.
- `.planning/REQUIREMENTS.md` — INFRA-01 … INFRA-06 requirement text and traceability.

### Codebase Maps (vendored reference analysis)
- `.planning/codebase/TESTING.md` — test framework, mandatory source/test separation rule, `approx` usage, current vendored inline-test anti-pattern to avoid.
- `.planning/codebase/CONVENTIONS.md` — Rust lint/naming/error-handling conventions; confirms no `.clippy.toml` exists yet.
- `.planning/codebase/STACK.md` — full technology stack and build tooling of the vendored reference.

### Vendored Reference & Oracle Source (catboost-master/)
- `catboost-master/util/random/fast.h`, `catboost-master/util/random/fast.cpp` — `TFastRng64` source to port exactly (INFRA-05).
- `catboost-master/util/random/fast_ut.cpp` — upstream PRNG unit test (candidate source of bitstream test vectors).
- `catboost-master/catboost/libs/helpers/restorable_rng.{h,cpp}` — restorable RNG wrapper around the fast generator (context for seeding semantics).
- `catboost-master/catboost/python-package/catboost/version.py` — confirms vendored version `1.2.10` (the oracle pin).
- `catboost-master/catboost/python-package/` — the Python package surface used as the Python-first oracle generator.

### Process / Project Rules
- `CLAUDE.md` (project root) — constraints, naming patterns, mandatory source/test separation, CubeCL rules.
- `AGENTS.md` (project root) — CubeCL kernel rules (generics-float; manual at `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md`) — relevant to `cb-backend`/`cb-compute` only.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Vendored `catboost-master/` is the algorithmic **reference + oracle**, NOT our codebase. The Rust implementation is greenfield (no root Cargo workspace exists yet).
- `catboost-master/util/random/fast.{h,cpp}` + `fast_ut.cpp` — the exact `TFastRng64` definition and its test vectors to reproduce.
- `ndarray-npy` (Rust crate) — read `.npy` fixtures on the Rust comparison side (D-09).

### Established Patterns
- Source/test separation is mandatory (CLAUDE.md / TESTING.md): dedicated `*_test.rs` files, no inline `#[cfg(test)]` in production modules.
- `approx` crate (`abs_diff_eq!`) is the existing float-comparison idiom; the oracle harness uses absolute-error ≤ 1e-5.

### Integration Points
- The generic `R: Runtime` / `F: Float` seam introduced in `cb-compute` (stubbed P1, realized P3) is the boundary on which `cb-backend` (P7 GPU) attaches purely additively.
- `cb-oracle` is consumed by every later phase's tests; its fixture layout (D-10, D-11) and per-stage comparison API (D-06) are the contract those phases build on.

</code_context>

<specifics>
## Specific Ideas

- User explicitly wants a **single** backend crate (`cb-backend`), feature-gated — not one crate per backend. This was a direct correction and is a hard constraint.
- User explicitly confirmed **ROCm cannot be tested in GitHub Actions**; the ROCm test lane must target local/self-hosted AMD hardware (Phase 7), kept out of the GH Actions workflow.

</specifics>

<deferred>
## Deferred Ideas

- **Self-hosted AMD runner provisioning** — setting up the ROCm CI runner is deferred to Phase 7 (scripts defined now in Phase 1, automation wired later). See D-16.
- **C++ instrumentation of catboost-master** for training-internal oracle values — deferred to the specific later phase that needs it, primarily Phase 5 (ordered CTR / per-object target statistics). See D-08.
- **Facade-as-edge vs facade-as-lib for anyhow** — minor open question (D-15) to settle in planning; not blocking.

None of the above are scope creep — all are explicitly later-phase items surfaced while clarifying Phase 1 boundaries.

</deferred>

---

*Phase: 1-Workspace, Lint Discipline & Oracle Harness*
*Context gathered: 2026-06-13*
