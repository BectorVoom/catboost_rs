# Phase 1: Workspace, Lint Discipline & Oracle Harness - Research

**Researched:** 2026-06-13
**Domain:** Rust Cargo workspace architecture · clippy lint governance · numerical-parity oracle harness · exact PRNG (PCG) port
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Crate Decomposition (INFRA-01)**
- **D-01:** Day-one workspace is fine-grained — all responsibility crates stubbed now so boundaries are fixed early and never refactored. Full set: `cb-core`, `cb-data`, `cb-compute`, `cb-backend`, `cb-train`, `cb-model`, `cb-oracle`, `catboost-rs` (facade).
- **D-02:** **One** backend crate, not four. `cb-backend` is the sole owner of all CubeCL / runtime-specific code, feature-gated `cpu`/`wgpu`/`cuda`/`rocm`, resolving to a single `cfg`-gated runtime type alias (zero runtime dispatch). There is NO per-backend crate. (User explicitly corrected the initial 4-crate proposal.)
- **D-03:** `cb-compute` stays **pure generic** — owns the generic `R: Runtime` / `F: Float` algorithm boundary and must NOT depend on CubeCL. All CubeCL coupling lives in `cb-backend`. Keeps the Phase 3 generic seam clean.
- **D-04:** Naming: internal crates use the `cb-` prefix; the public/published facade crate is named **`catboost-rs`**.
- **D-05:** Crate→phase map: `cb-core` (errors/shared types, P1), `cb-data` (Pool/quantization, P2), `cb-compute` (generic boundary + kernels, P3), `cb-backend` (CubeCL runtime alias, P7), `cb-train` (boosting loop, P3), `cb-model` (serialize/SHAP, P4), `cb-oracle` (harness + fixtures, P1), `catboost-rs` (Builder facade, P4).

**Oracle Extraction Method (INFRA-03, INFRA-04)**
- **D-06:** **Python-first hybrid.** Primary oracle generator is the pinned pip `catboost` package: borders (`get_borders()`/quantization), tree splits + leaf values (`save_model(format='json')`), per-iteration approximants (`staged_predict`), SHAP / feature importance, final predictions.
- **D-07:** **Oracle version pinned to `catboost==1.2.10`** — matches vendored `catboost-master/` source (`.../version.py` = `1.2.10`). Mandatory for algorithmic parity; do not bump independently of vendored source.
- **D-08:** C++ instrumentation of `catboost-master/` is **deferred per-phase**, used ONLY where the Python API cannot reach the needed internal value (primarily Phase 5). Phase 1–4 require no C++ build.

**Fixture Storage & Scope (INFRA-03, INFRA-04)**
- **D-09:** Fixture format is a **hybrid: `config.json` (metadata — seed, version, params) + `.npy` binary arrays** (borders, leaf values, approximants, predictions). Rust reads `.npy` via `ndarray-npy`; Python writes via NumPy.
- **D-10:** Fixtures live under `cb-oracle/fixtures/`. Layout: `cb-oracle/fixtures/inputs/<dataset>/` for frozen input datasets; `cb-oracle/fixtures/<scenario>/` for per-phase expected-output fixtures that reference an input dataset.
- **D-11:** **Frozen shared input corpus.** Phase 1 generates & commits a small canonical set of synthetic INPUT datasets once (pinned seed; varied shapes). Every later phase draws inputs from this corpus and only adds its own expected-OUTPUT fixtures.
- **D-12:** Determinism: all oracle generation uses a pinned seed and `thread_count=1`; comparison tolerance is absolute error ≤ 1e-5; fixtures committed frozen (generator does NOT run in CI).

**Lint & CI Enforcement (INFRA-02, INFRA-06)**
- **D-13:** Deny-lints via a single centralized **`[workspace.lints]`** table at workspace root (`clippy::unwrap_used`, `clippy::expect_used`, `clippy::panic`, `clippy::indexing_slicing` = `"deny"`); each library crate opts in with `[lints] workspace = true`.
- **D-14:** `anyhow` ban is **structural**: core library crates (`cb-core`, `cb-data`, `cb-compute`, `cb-backend`, `cb-train`, `cb-model`) do NOT list `anyhow` in `[dependencies]`. `anyhow` permitted only as dev-dependency (tests) and at app/binding edges. CI grep (`scripts/check-no-anyhow.sh`) is the backstop.
- **D-15:** `thiserror` is the library error strategy; `anyhow` only at binding/app/test edges. Open: whether `catboost-rs` facade counts as thiserror-only "core lib" vs "edge" — default to treating the facade as a library (thiserror) unless planning surfaces a reason otherwise.
- **D-16:** CI split into two lanes (**ROCm cannot run in GitHub Actions** — no AMD-GPU runners):
  - **GitHub Actions (CPU lane, all of Phase 1):** `.github/workflows/ci.yml` runs `cargo build` (cpu feature), clippy lint gate (`-D warnings`), the anyhow grep backstop, and CPU oracle harness tests.
  - **ROCm GPU lane (Phase 7):** standalone runnable scripts (e.g. `scripts/gpu-rocm-tests.sh`) on local/self-hosted AMD hardware. Phase 1 ships ONLY the CPU GitHub Actions workflow.
- **D-17:** Source/test separation (INFRA-06) enforced as convention from first commit — no inline `#[cfg(test)]` in production modules; tests in dedicated `*_test.rs` files / `tests/` dir. Do NOT replicate the inline-`mod tests` pattern of vendored upstream rust-package.

### Claude's Discretion
- Exact dataset shapes / row counts / feature mixes of the frozen input corpus (D-11) — choose reasonable small synthetic datasets exercising numeric, categorical, and grouped/ranking paths; planning may refine.
- `TFastRng64` bitstream-capture mechanics (INFRA-05) — how to emit the C++ generator's raw bitstream for a fixed seed. (RESOLVED below — see Pitfall 4 / Code Examples: no C++ harness needed; transcribe vendored test vectors.)
- Rust edition / MSRV / toolchain pinning specifics — apply CLAUDE.md constraints (latest stable Rust, latest crate versions).
- Whether `cb-backend`'s `cpu` feature uses CubeCL's CPU runtime or a plain non-CubeCL CPU path in early phases — defer to Phase 3/7 planning; Phase 1 only stubs the crate.

### Deferred Ideas (OUT OF SCOPE)
- **Self-hosted AMD runner provisioning** — deferred to Phase 7 (scripts defined now, automation wired later). See D-16.
- **C++ instrumentation of catboost-master** for training-internal oracle values — deferred to the later phase that needs it (primarily Phase 5). See D-08.
- **Facade-as-edge vs facade-as-lib for anyhow** — minor open question (D-15) to settle in planning; not blocking.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| INFRA-01 | Modular Cargo workspace with feature-gated backend crates (`cpu`/`wgpu`/`cuda`/`rocm`) and clear separation of responsibilities | Workspace layout + `[features]` wiring in Standard Stack / Architecture Patterns; single feature-gated `cb-backend` per D-02; `[workspace.lints]` confirmed stable on Rust 1.96 |
| INFRA-02 | Lint discipline in library crates — deny `unwrap`/`expect`/`panic`/`indexing_slicing`; `thiserror` libs, `anyhow` only at edges; CI check anyhow absent from core lib | Exact clippy restriction-lint names verified; `[workspace.lints]` + `[lints] workspace = true` pattern; `check-no-anyhow.sh` grep mechanics; **test-code interaction pitfall** (Pitfall 1) |
| INFRA-03 | Oracle harness — random inputs vs upstream outputs ≤1e-5, frozen committed fixtures, pinned seed/version, single-thread determinism | Python `catboost==1.2.10` API verified (`get_borders`, `save_model(json)`, `staged_predict`, `predict`); `.npy` hybrid fixture format; `ndarray-npy` 0.10.0 read API; `thread_count=1` determinism confirmed |
| INFRA-04 | Intermediate per-stage oracle — borders, per-tree splits, leaf values, per-iteration approximants | Per-stage extraction APIs mapped to verified Python methods; **JSON `oblivious_trees` schema verified from vendored committed model** (HIGH); per-stage comparison API design |
| INFRA-05 | Exact port of `TFastRng64` PRNG, bitstream-oracle-validated for a fixed seed | **Full algorithm transcribed from vendored `fast.h`/`lcg_engine.h`/`bitops.h`** + exact test vectors from `fast_ut.cpp` (HIGH) — see Code Examples |
| INFRA-06 | Source/test code strictly separated (no inline `#[cfg(test)]` in production) | CLAUDE.md/AGENTS.md/TESTING.md mandatory rule; `*_test.rs` pattern; CI grep enforceable; lint-vs-test interaction handled |

</phase_requirements>

## Summary

Phase 1 is pure scaffolding plus two load-bearing artifacts: the **oracle harness** (`cb-oracle`) and the **exact `TFastRng64` PRNG port** (in `cb-core`). Everything in this phase is verifiable from the vendored `catboost-master/` source already on disk — no external C++ build is needed, and the most uncertain item (the PRNG bitstream) is fully resolved: the algorithm is a pair of PCG-XSH-RR 32-bit generators whose outputs are concatenated into a 64-bit word, and the vendored `fast_ut.cpp` already contains canonical test vectors (`TFastRng64(17).GenRand() == 14895365814383052362`; `TFastRng64(0,1,2,3).Uniform(100)` → a fixed 20-value sequence). **No C++ harness is required** — transcribe the vendored test vectors directly. This collapses INFRA-05 risk from MEDIUM to LOW.

The oracle side is equally de-risked: the pinned `catboost==1.2.10` Python package exposes every value the harness needs — `get_borders()` (borders), `save_model(format='json')` (per-tree splits + leaf values, schema verified against a committed vendored JSON model), `staged_predict()` (per-iteration approximants), and `predict()` (final). Determinism is achieved with a pinned seed and `thread_count=1` (the docs explicitly state `thread_count` "does not affect results", but pinning to 1 removes any summation-order risk and matches D-12). The Rust read side uses `ndarray-npy` 0.10.0 (`read_npy` / `ReadNpyExt`), which round-trips f64 bit-exactly because `.npy` stores raw little-endian IEEE-754 bytes.

The single highest-leverage planning insight is a **lint/test interaction trap**: `clippy::unwrap_used`, `expect_used`, `panic`, and `indexing_slicing` are *restriction* lints that, applied via `[workspace.lints]`, also fire inside `#[test]` code and `*_test.rs` files — where `unwrap()`/indexing are normal and desirable. Because this project mandates strict source/test separation, the clean solution is a per-crate `#![cfg_attr(test, allow(clippy::unwrap_used, ...))]` (or `--lib`-only clippy in CI plus crate-level test allows). This must be designed into the very first commit or every test will fail the lint gate.

**Primary recommendation:** Stand up the 8-crate workspace with a root `[workspace.lints]` table and a single feature-gated `cb-backend` stub; port `TFastRng64` into `cb-core` validated against the vendored `fast_ut.cpp` test vectors (no C++ build); build `cb-oracle` around a Python generator (`catboost==1.2.10`, `thread_count=1`, pinned seed) emitting `config.json`+`.npy` fixtures, and a Rust per-stage comparator (`ndarray-npy` read + absolute-error ≤1e-5). Walking skeleton = workspace builds → read one committed `.npy` → one real oracle assertion passes ≤1e-5 → clippy/anyhow gates active → GitHub Actions CPU lane green.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Shared error types, `TFastRng64` port | `cb-core` (library) | — | Errors and the deterministic PRNG are leaf dependencies every other crate uses; no upstream deps |
| Generic `R: Runtime` / `F: Float` seam | `cb-compute` (library) | — | Pure-generic algorithm boundary (D-03); must not pull CubeCL so the Phase 3 seam stays clean |
| CubeCL runtime alias, backend feature gates | `cb-backend` (library) | `cb-compute` | Sole owner of runtime-specific code (D-02); attaches additively to `cb-compute` in Phase 7; **stub-only in P1** |
| Oracle harness + fixtures + per-stage comparator | `cb-oracle` (library + test infra) | Python generator (build-time, not CI) | Generates frozen fixtures (Python) and compares (Rust); consumed by every later phase's tests |
| Lint governance | Workspace root (`Cargo.toml`) | each library crate (`[lints] workspace = true`) | Centralized DRY policy (D-13); per-crate opt-in keeps test-only crates flexible |
| CI orchestration | GitHub Actions CPU lane (`.github/workflows/ci.yml`) | local ROCm scripts (P7) | CPU-only in P1 (D-16); ROCm physically cannot run on GH-hosted runners |
| Public facade | `catboost-rs` | `cb-*` internals | Builder API realized P4; stubbed P1 (re-exports / empty) |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `thiserror` | 2.0.x (2.0.18 latest) [VERIFIED: cargo search] | Derive-based typed error enums in all `cb-*` library crates | PROJECT.md/CLAUDE.md-mandated library error strategy; zero-runtime-cost derive; the de-facto Rust library error crate |
| `ndarray` | 0.17.x (0.17.2 latest) [VERIFIED: cargo search] | f64 array type backing borders/approximants/predictions on the Rust comparison side | Required substrate for `ndarray-npy`; the standard n-d array crate |
| `ndarray-npy` | 0.10.x (0.10.0 latest) [VERIFIED: cargo search + docs.rs] | Read committed `.npy` fixtures into `Array<f64, _>` on the Rust side (D-09) | Native `.npy` reader; pairs with `ndarray`; f64 round-trips bit-exactly (raw IEEE-754 LE bytes) |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `anyhow` | 1.0.x (1.0.102 latest) [VERIFIED: cargo search] | Ergonomic error context | **Dev-dependency / test edges ONLY** (D-14/D-15). Forbidden in `[dependencies]` of the 6 core library crates |
| `approx` | 0.5.x (vendored uses 0.5.1) [ASSUMED — confirm latest at impl time] | `abs_diff_eq!` float assertions in tests | Existing project idiom (TESTING.md); use for ≤1e-5 absolute-error assertions in the comparator's tests |
| `serde` + `serde_json` | latest stable [ASSUMED — confirm at impl] | Parse fixture `config.json` metadata (seed, version, params) and (in later phases) the oracle JSON model | `config.json` half of the hybrid fixture format (D-09). Phase 1 needs only metadata parsing |
| `catboost` (PyPI) | **==1.2.10 PINNED** (D-07) [CITED: catboost-master/.../version.py] | Python oracle generator (build-time tool, not a Rust dep) | Generates all frozen fixtures. Pin is mandatory; do not bump independently of vendored source |
| `numpy` (PyPI) | latest compatible with catboost 1.2.10 [ASSUMED] | `numpy.save()` writes the `.npy` half of fixtures | Python write side of D-09 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `ndarray-npy` | hand-rolled `.npy` parser | `.npy` header parsing + dtype/shape/fortran-order handling is fiddly; the crate is the "don't hand-roll" answer |
| `[workspace.lints]` table | per-file `#![deny(...)]` attributes | Per-file drifts and is non-DRY; D-13 explicitly chose the centralized table (stable since Rust 1.74) |
| Python-first oracle | C++ instrumentation harness | C++ build is heavyweight and deferred (D-08); Python API reaches every Phase 1–4 value |
| Single `cb-backend` | four per-backend crates | User explicitly rejected (D-02); one feature-gated crate gives zero runtime dispatch via `cfg` alias |

**Installation (Rust side, illustrative — pin latest at impl time):**
```bash
# In the relevant crate manifests, not a single global install:
#   cb-oracle: ndarray, ndarray-npy, serde, serde_json; dev-deps: approx, anyhow
#   cb-core:   thiserror
cargo add thiserror              # cb-core
cargo add ndarray ndarray-npy serde serde_json   # cb-oracle
cargo add --dev approx anyhow    # cb-oracle test edges only
```

**Python side (oracle generator environment — pinned):**
```bash
python3 -m pip install "catboost==1.2.10" numpy   # numpy version per catboost 1.2.10 compat
```

**Version verification performed this session:**
- `thiserror` 2.0.18, `ndarray` 0.17.2, `ndarray-npy` 0.10.0, `anyhow` 1.0.102 — all via `cargo search` (2026-06-13).
- Rust toolchain present: `rustc 1.96.0`, `cargo 1.96.0`, `clippy 0.1.96` — `[workspace.lints]` (stable since 1.74) fully supported.
- `python3 3.12.3` present; **`catboost` and `numpy` are NOT currently installed** in this environment (see Environment Availability). Planning must include a setup step / verify the oracle-generation environment before fixture generation.

## Package Legitimacy Audit

> All Rust crates are long-established, high-download, official-org packages. Python `catboost`/`numpy` are the canonical packages.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| `thiserror` | crates.io | 6+ yrs | very high | github.com/dtolnay/thiserror | OK | Approved |
| `anyhow` | crates.io | 6+ yrs | very high | github.com/dtolnay/anyhow | OK | Approved (dev/edge only) |
| `ndarray` | crates.io | 8+ yrs | very high | github.com/rust-ndarray/ndarray | OK | Approved |
| `ndarray-npy` | crates.io | 6+ yrs | high | github.com/jturner314/ndarray-npy | OK | Approved |
| `approx` | crates.io | 7+ yrs | very high | github.com/brendanzab/approx | OK | Approved (test) |
| `serde` / `serde_json` | crates.io | 8+ yrs | very high | github.com/serde-rs | OK | Approved |
| `catboost` | PyPI | 7+ yrs | very high | github.com/catboost/catboost | OK | Approved (pinned ==1.2.10) |
| `numpy` | PyPI | 15+ yrs | very high | github.com/numpy/numpy | OK | Approved |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

> Note: The `gsd-tools query package-legitimacy check` seam was not available in this environment. Verdicts above are based on direct knowledge of these canonical, long-established packages plus `cargo search` registry confirmation. The planner should still pin exact latest versions at implementation time; ages/download figures are characterizations, not freshly-queried registry stats.

## Architecture Patterns

### System Architecture Diagram

```
                    ORACLE GENERATION (build-time tool, NOT in CI)
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Python (pinned catboost==1.2.10, seed=S, thread_count=1)            │
  │                                                                      │
  │   synthetic input gen ──► Pool ──► fit() ──► trained model           │
  │        │                                   │                         │
  │        │ (frozen INPUT corpus, D-11)       ├─ get_borders()          │
  │        ▼                                   ├─ save_model(json) ──────┐│
  │   inputs/<dataset>/{config.json,*.npy}     ├─ staged_predict()       ││
  │                                            └─ predict()              ││
  │                                                   │                  ││
  │                          numpy.save(*.npy) + json.dump(config.json)  ││
  └───────────────────────────────────┬──────────────────────┬──────────┘│
                                       ▼                      ▼           │
              cb-oracle/fixtures/inputs/<dataset>/   cb-oracle/fixtures/  │
                                                     <scenario>/          │
                                       │  (COMMITTED, FROZEN)             │
  ═════════════════════════════════════╪═══════════════════════════════════
                    RUST PARITY CHECK   ▼   (runs in CI — CPU lane)
  ┌─────────────────────────────────────────────────────────────────────┐
  │  cb-oracle (Rust)                                                     │
  │    read_npy(*.npy) ─► Array<f64> (expected)                          │
  │    serde_json(config.json) ─► metadata (seed, params)                │
  │                       │                                               │
  │    [later phases] cb-train/cb-model produce ACTUAL values            │
  │                       │                                               │
  │    per-stage comparator:  |expected - actual| ≤ 1e-5  (abs error)    │
  │       stages: borders │ splits │ leaf_values │ per-iter approx │ pred │
  └─────────────────────────────────────────────────────────────────────┘

  cb-core ── TFastRng64 port ──► bitstream test  (vs vendored fast_ut.cpp vectors)
            (no C++ build; vectors transcribed)
```

### Recommended Project Structure
```
catboost_rs/                      # workspace root
├── Cargo.toml                    # [workspace] members + [workspace.lints] (D-13) + [workspace.dependencies]
├── rust-toolchain.toml           # pin stable channel (latest)
├── .github/workflows/ci.yml      # CPU lane only (D-16): build + clippy -D warnings + anyhow grep + oracle tests
├── scripts/
│   ├── check-no-anyhow.sh        # grep backstop (D-14)
│   └── gpu-rocm-tests.sh         # P7 placeholder script, defined now, wired later (D-16)
├── crates/
│   ├── cb-core/                  # P1: errors (thiserror) + TFastRng64 port
│   │   ├── Cargo.toml            # [lints] workspace = true ; deps: thiserror
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── error.rs          # production only
│   │       ├── rng.rs            # TFastRng64 — production only
│   │       └── rng_test.rs       # bitstream vectors (separate file, D-17)
│   ├── cb-data/                  # P2 stub
│   ├── cb-compute/               # P3 — pure-generic seam (NO cubecl) (D-03), stub in P1
│   ├── cb-backend/               # single feature-gated crate (D-02), stub in P1
│   │   └── Cargo.toml            # [features] cpu, wgpu, cuda, rocm  (inert in P1)
│   ├── cb-train/                 # P3 stub
│   ├── cb-model/                 # P4 stub
│   ├── cb-oracle/                # P1: harness + comparator + fixtures
│   │   ├── Cargo.toml            # deps: ndarray, ndarray-npy, serde, serde_json; dev: approx, anyhow
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── fixture.rs        # load config.json + *.npy
│   │   │   ├── compare.rs        # per-stage abs-error ≤1e-5 comparator
│   │   │   └── *_test.rs
│   │   ├── generator/            # Python oracle generator (build-time, not a cargo target)
│   │   │   ├── requirements.txt  # catboost==1.2.10, numpy
│   │   │   ├── gen_inputs.py     # D-11 frozen input corpus
│   │   │   └── gen_fixtures.py   # borders/json/staged/predict → *.npy + config.json
│   │   └── fixtures/             # COMMITTED frozen artifacts (D-10)
│   │       ├── inputs/<dataset>/
│   │       └── <scenario>/
│   └── catboost-rs/              # P4 facade stub (re-exports)
└── ...
```

### Pattern 1: Centralized workspace lints with per-crate opt-in (D-13)
**What:** One `[workspace.lints.clippy]` table; each library crate adds `[lints] workspace = true`.
**When to use:** Every `cb-*` library crate. (A crate that is *only* tests/binaries may opt out or layer test allows.)
**Example:**
```toml
# Cargo.toml (workspace root)
[workspace]
resolver = "2"
members = ["crates/*"]

[workspace.lints.clippy]
unwrap_used      = "deny"
expect_used      = "deny"
panic            = "deny"
indexing_slicing = "deny"

# crates/cb-core/Cargo.toml
[lints]
workspace = true
```
> Constraint [VERIFIED: rust-clippy issue #13157, cargo issue #12716]: a crate using `lints.workspace = true` **cannot also add/override individual lints in the same `[lints]` table**. Per-crate exceptions must be expressed as in-code `#![allow(...)]` / `#![cfg_attr(test, allow(...))]` attributes. This is why the test-allow (Pitfall 1) lives in code, not in a per-crate manifest lint table.

### Pattern 2: Single feature-gated backend, zero runtime dispatch (D-02)
**What:** `cb-backend` exposes one `cfg`-selected runtime type alias.
**When to use:** The only place CubeCL/runtime code may live. Inert stub in Phase 1.
**Example:**
```toml
# crates/cb-backend/Cargo.toml
[features]
default = ["cpu"]
cpu  = []
wgpu = []   # inert in P1 — wired in P7
cuda = []
rocm = []
```
```rust
// crates/cb-backend/src/lib.rs  (Phase 1 stub — alias is a placeholder)
#[cfg(feature = "cpu")]
pub type SelectedRuntime = ();   // P3/P7 replaces () with the real runtime
#[cfg(all(feature = "wgpu", not(feature = "cpu")))]
pub type SelectedRuntime = ();
// ... one cfg arm per backend; compile-time selection, no runtime match
```

### Pattern 3: Per-stage oracle comparator (INFRA-04)
**What:** Comparator asserts on intermediate stages, not just final predictions.
**When to use:** The `cb-oracle` public API every later phase calls.
**Example:**
```rust
// crates/cb-oracle/src/compare.rs
pub enum Stage { Borders, Splits, LeafValues, StagedApprox, Predictions }

pub fn assert_abs_close(expected: &[f64], actual: &[f64], tol: f64) -> Result<(), OracleError> {
    if expected.len() != actual.len() { return Err(OracleError::LengthMismatch { /* .. */ }); }
    for (i, (e, a)) in expected.iter().zip(actual).enumerate() {
        let d = (e - a).abs();
        if d > tol { return Err(OracleError::Diverged { index: i, expected: *e, actual: *a, diff: d }); }
    }
    Ok(())
}
// tol = 1e-5 (D-12). Use this single primitive for every Stage.
```

### Anti-Patterns to Avoid
- **Inline `#[cfg(test)] mod tests` in production `.rs` files** — prohibited by CLAUDE.md/AGENTS.md/D-17; the vendored `rust-package/src/model.rs` does this and must NOT be copied.
- **Per-backend crates** — explicitly rejected (D-02). One feature-gated `cb-backend`.
- **CubeCL dependency in `cb-compute`** — violates D-03; keeps the generic seam impure.
- **`anyhow` in core-library `[dependencies]`** — structurally banned (D-14); dev-dep/edge only.
- **Running the oracle generator in CI** — fixtures are frozen/committed (D-12); CI only *reads* and compares.
- **Adding GPU/ROCm jobs to `.github/workflows/ci.yml`** — ROCm cannot run on GH-hosted runners (D-16).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Read `.npy` fixtures | Custom `.npy` header/dtype/shape parser | `ndarray-npy` `read_npy` / `ReadNpyExt` | Header magic, dtype strings, fortran-order, shape tuple parsing are error-prone |
| Typed errors | Manual `impl Error`/`Display` boilerplate | `thiserror` derive | CLAUDE.md-mandated; eliminates boilerplate, no runtime cost |
| Float parity assertion | Ad-hoc `(a-b).abs() < eps` scattered everywhere | One `cb-oracle` comparator primitive (+ `approx` in tests) | Single audited tolerance path; matches the later DATA-07 "single reduction utility" philosophy |
| Oracle value extraction | Re-deriving borders/splits/leaf values yourself | Pinned `catboost==1.2.10` Python API (`get_borders`, `save_model(json)`, `staged_predict`, `predict`) | The reference implementation IS the oracle; re-deriving defeats the parity test |
| PRNG bitstream capture | A custom C++ harness to emit the bitstream | Transcribe vendored `fast_ut.cpp` test vectors | Canonical vectors already committed in-repo; no toolchain needed (see Pitfall 4) |

**Key insight:** In a parity-testing project the entire point is to compare against the *unmodified* reference. Anything you hand-roll on the oracle side (borders, splits, RNG capture) reintroduces exactly the divergence the harness exists to catch. Lean on the pinned package and the vendored test vectors.

## Runtime State Inventory

> This is a greenfield phase (no root Cargo workspace exists yet — confirmed: `ls Cargo.toml` returns no root manifest). There is no prior runtime state to migrate. However, two *frozen-state* concerns are first-created here and must be treated as immutable contracts from commit one:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | **None existing** — but Phase 1 *creates* the frozen `cb-oracle/fixtures/` corpus that all 8 phases depend on (D-11). | Generate once with pinned seed/version; commit; never regenerate casually (regeneration silently shifts every downstream phase's oracle baseline) |
| Live service config | None — verified (no services; CI is the only "live" config) | The `.github/workflows/ci.yml` CPU lane is authored here; ROCm explicitly excluded (D-16) |
| OS-registered state | None — verified (no OS-level registrations) | — |
| Secrets/env vars | None — verified. (`DEBUG`/`OUT_DIR` referenced by the *vendored* package's build.rs are not part of our greenfield crates) | — |
| Build artifacts | None existing. Phase 1 establishes `target/` and the toolchain pin only | `rust-toolchain.toml` pin is the only durable build-config artifact |

**Canonical question answered:** After Phase 1, the only durable "state" that downstream phases cache/depend on is the **committed frozen fixture corpus** + the **PRNG test vectors** + the **`[workspace.lints]` policy** — all in git, all immutable contracts.

## Common Pitfalls

### Pitfall 1: Restriction lints fire inside test code
**What goes wrong:** `clippy::unwrap_used`/`expect_used`/`panic`/`indexing_slicing` are *restriction* lints that apply to ALL code in the crate, including `#[test]` functions and `*_test.rs` files where `unwrap()` and `arr[i]` are idiomatic. With `[workspace.lints]` + `-D warnings`, every test that unwraps fails the build.
**Why it happens:** `[workspace.lints]` is crate-wide; restriction lints do not auto-exempt test cfg. And because `lints.workspace = true` forbids adding per-crate manifest lint overrides, the exemption *cannot* live in `Cargo.toml`.
**How to avoid:** Add a crate-level in-code allow gated on `test`:
```rust
// top of each library crate's lib.rs
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]
```
For integration tests in `tests/` (separate compilation units), add the same `#![allow(...)]` at the top of each `tests/*.rs`, OR scope CI clippy to `cargo clippy --workspace --lib -D warnings` for the production gate while running tests separately. **Decide this in Plan 1** — it shapes every test file.
**Warning signs:** First `*_test.rs` written makes the clippy gate red; tempting (wrong) "fix" is to weaken the workspace lints.

### Pitfall 2: `thread_count` and summation-order non-determinism in the oracle
**What goes wrong:** Running the Python oracle with default `thread_count=-1` can change floating-point summation order across machines/runs, producing fixtures that differ in the last ULPs — poisoning the ≤1e-5 contract subtly.
**Why it happens:** Parallel reductions sum in non-deterministic order; CatBoost docs say `thread_count` "does not affect results" but D-12 (and the later DATA-07 reduction concern) treat single-thread as the safe contract.
**How to avoid:** Always set `thread_count=1` AND a fixed `random_seed` in every generator call. Record both in `config.json`. Generate fixtures on one canonical environment and commit them frozen (D-12) so CI never regenerates.
**Warning signs:** Oracle test passes locally, fails in CI by ~1e-7–1e-6, or fixtures change on regeneration.

### Pitfall 3: `.npy` dtype/precision mismatch (f32 vs f64)
**What goes wrong:** CatBoost predictions/approximants are computed in `double`; if the Python side saves `np.float32` arrays (or casts), the committed fixture loses precision and the ≤1e-5 comparison is measuring the wrong baseline.
**Why it happens:** NumPy defaults and accidental `astype(np.float32)` silently downcast.
**How to avoid:** Explicitly `np.asarray(x, dtype=np.float64)` before `np.save`; assert `arr.dtype == np.float64` in the generator; on the Rust side read into `Array<f64, _>` (ndarray-npy errors on dtype mismatch, which is the desired guard). `.npy` stores raw IEEE-754 LE bytes, so f64↔f64 round-trips bit-exactly.
**Warning signs:** `ndarray-npy` read returns a dtype-mismatch error, or diffs cluster at ~1e-7 (single-precision epsilon).

### Pitfall 4: Over-engineering the PRNG bitstream capture (RESOLVED)
**What goes wrong:** Building a C++ harness to emit `TFastRng64`'s bitstream — unnecessary toolchain, and a place to introduce a transcription bug.
**Why it happens:** The discretion note left "C++ harness vs transcribe vectors" open; the C++ route looks more rigorous.
**How to avoid:** The vendored `util/random/fast_ut.cpp` already contains canonical vectors (see Code Examples). Transcribe them into a Rust `*_test.rs`. The algorithm is small and fully specified in the vendored headers — port + assert against committed vectors. No C++ build. If extra coverage is wanted, add a longer `Uniform(100)` sequence or the `GenRand64`/stream-ctor vectors, all already in `fast_ut.cpp`.
**Warning signs:** Plan proposes a `build.rs`/CMake step or a C++ compiler dependency for INFRA-05 — that is scope creep; reject it.

### Pitfall 5: `LcgAdvance` overflow semantics (wrapping vs panicking)
**What goes wrong:** The C++ LCG (`x * A + C`) relies on silent unsigned wraparound (mod 2^64). In Rust, `u64` multiply/add **panics on overflow in debug builds** and the naive `*`/`+` won't reproduce the C++ value. With `panic` denied by clippy, debug-overflow panics are also a lint/runtime hazard.
**Why it happens:** Rust integer arithmetic is checked in debug; C++ unsigned is wrapping by definition.
**How to avoid:** Use `wrapping_mul`/`wrapping_add` (and `wrapping_*` throughout `Iterate`, `LcgAdvance`, the seed `(seq<<1)|1`, and the mixer). This is mandatory for bitstream parity AND avoids debug-overflow panics. The `Advance(delta)` path must port `LcgAdvance` exactly (binary-exponentiation of `A^m` and `(A^m−1)/(A−1)` accumulators) — see Code Examples.
**Warning signs:** Test1/Test2/Test3 vectors pass for small counts but `Advance` (TestAdvance) diverges, or debug builds panic with "attempt to multiply with overflow".

## Code Examples

### TFastRng64 — full algorithm (transcribed from vendored source)

The generator is **two PCG-XSH-RR 32-bit LCGs** (`R1_`, `R2_`) whose 32-bit outputs are concatenated into a 64-bit word. Sources: `util/random/fast.h`, `util/random/lcg_engine.h`, `util/generic/bitops.h`, `util/random/fast.cpp`, `util/random/common_ops.h`. [VERIFIED: vendored catboost-master source, read this session]

Key constants and operations:
- LCG multiplier `A = 6364136223846793005` (`0x5851F42D4C957F2D`), state update `X = X*A + C` (mod 2^64, **wrapping**).
- Per-stream constant `C = (seq << 1) | 1` (must be odd) — from `TLcgIterator`.
- PCG output mixer `TPCGMixer::Mix(x)`:
  ```
  xorshifted = ((x >> 18) ^ x) >> 27   // -> u32
  rot        = x >> 59                  // -> u32 (0..31)
  result     = rotate_right_u32(xorshifted, rot)
  ```
- 64-bit combine: `GenRand() = (u64(R1.GenRand()) << 32) | u64(R2.GenRand())`.
- Two-arg `TFastRng64(seed1, seq1, seed2, seq2)` with `seq2` fixed by `FixSeq`:
  ```
  mask = u32::MAX >> 1
  if (seq1 & mask) == (seq2 & mask) { seq2 = !seq2 }   // ensure distinct streams
  ```
- One-arg `TFastRng64(seed)` derives the four args via a `TReallyFastRng32(seed)`:
  ```
  rng = ReallyFastRng32(seed)   // A=6364..2D, C=1 (TFastLcgIterator addend 1)
  Seed1 = rng.GenRand64(); Seq1 = rng.GenRand(); Seed2 = rng.GenRand64(); Seq2 = rng.GenRand()
  // GenRand64 for a 32-bit rng: (u64(x)) | (u64(next GenRand()) << 32)   [common_ops.h ToRand64]
  ```
- `Advance(delta)` advances each LCG by `delta` via binary exponentiation (`lcg_engine.cpp LcgAdvance`), all **wrapping**:
  ```
  // returns seed*A^delta + C*(A^delta - 1)/(A - 1), computed iteratively
  let mut mask = 1u64;
  while mask != (1<<63) && (mask<<1) <= delta { mask <<= 1; }
  let (mut apow, mut adiv) = (1u64, 0u64);
  while mask != 0 {
      adiv = adiv.wrapping_mul(apow.wrapping_add(1));
      apow = apow.wrapping_mul(apow);
      if delta & mask != 0 { adiv = adiv.wrapping_add(apow); apow = apow.wrapping_mul(A); }
      mask >>= 1;
  }
  x = seed.wrapping_mul(apow).wrapping_add(C.wrapping_mul(adiv));
  ```

### Bitstream oracle test vectors (committed in vendored `fast_ut.cpp`)
```rust
// crates/cb-core/src/rng_test.rs  (separate file per D-17)
// Source: catboost-master/util/random/fast_ut.cpp  [VERIFIED: read this session]

#[test]
fn test3_single_seed_genrand() {
    // TFastRng64 rng(17); rng.GenRand() == 14895365814383052362
    let mut rng = TFastRng64::from_seed(17);
    assert_eq!(rng.gen_rand(), 14895365814383052362u64);
}

#[test]
fn test2_four_arg_uniform100_sequence() {
    // TFastRng64 rng(0,1,2,3); first 20 of Uniform(100):
    let expected: [u64; 20] =
        [37,43,76,17,12,87,60,4,83,47,57,81,28,45,66,74,18,17,18,75];
    let mut rng = TFastRng64::new(0, 1, 2, 3);
    for &e in &expected {
        assert_eq!(rng.uniform(100), e);
    }
}

#[test]
fn test_advance_matches_genrand_x100() {
    // 100×GenRand() == Advance(100) then GenRand()
    let mut a = TFastRng64::new(0,1,2,3);
    let mut b = TFastRng64::new(0,1,2,3);
    for _ in 0..100 { a.gen_rand(); }
    b.advance(100);
    assert_eq!(a.gen_rand(), b.gen_rand());
}
```
> `Uniform(t)` must port `GenUniform` from `common_ops.h`: rejection sampling with `randmax = RandMax() - RandMax() % t` (where `RandMax()` for `TFastRng64` is `u64::MAX`), looping while the draw `>= randmax`, returning `draw % t`.

### Oracle generator (Python, build-time) — verified API surface
```python
# crates/cb-oracle/generator/gen_fixtures.py
# Source APIs verified against catboost==1.2.10 docs this session [CITED: catboost.ai docs]
import numpy as np, json
from catboost import CatBoostRegressor, Pool

SEED = 0
model = CatBoostRegressor(iterations=10, learning_rate=0.1, depth=4,
                          random_seed=SEED, thread_count=1, verbose=False)
model.fit(X, y)                                   # X,y from frozen input corpus

borders = model.get_borders()                     # {feat_idx: [border, ...]}   (INFRA-04 borders)
model.save_model("model.json", format="json")     # oblivious_trees: splits + leaf_values (INFRA-04)
staged = np.asarray(list(model.staged_predict(X, eval_period=1)), dtype=np.float64)  # per-iter approx
final  = np.asarray(model.predict(X), dtype=np.float64)                              # final predictions

np.save("predictions.npy", final)                 # f64 — exact round-trip
np.save("staged.npy",      staged)
json.dump({"seed": SEED, "catboost_version": "1.2.10", "thread_count": 1,
           "params": model.get_params()}, open("config.json", "w"))
```

### Verified JSON model schema (the `oblivious_trees` shape for INFRA-04)
```jsonc
// Source: catboost-master/.../numeric_only_model.json [VERIFIED: read this session]
// save_model(format="json") -> top-level "oblivious_trees": [ per tree: ]
{
  "leaf_values":  [0.00749999, 0.00119999],          // f64, ordered by binary combo of split bits
  "leaf_weights": [1, 2],
  "splits": [
    { "border": -1.5, "float_feature_index": 2, "split_index": 2, "split_type": "FloatFeature" }
  ]
}
// Also present at top level: "features_info" (float_features w/ borders, nan handling),
// "scale_and_bias", "model_info". Phase 4 consumes the full schema; Phase 1 only needs to
// confirm extraction works and leaf_values/borders are f64-exact.
```

### Rust fixture read (`ndarray-npy` 0.10.0)
```rust
// crates/cb-oracle/src/fixture.rs   [CITED: docs.rs/ndarray-npy 0.10.0]
use ndarray::Array1;
use ndarray_npy::read_npy;

pub fn load_f64_vec(path: &std::path::Path) -> Result<Vec<f64>, OracleError> {
    let arr: Array1<f64> = read_npy(path)?;   // errors on dtype mismatch — desired guard
    Ok(arr.to_vec())
}
```

### CI anyhow grep backstop (D-14)
```bash
# scripts/check-no-anyhow.sh  — fails build if anyhow appears in core-lib non-test code
set -euo pipefail
CORE="crates/cb-core crates/cb-data crates/cb-compute crates/cb-backend crates/cb-train crates/cb-model"
# scan src/ excluding *_test.rs; also catch `use anyhow` / `anyhow::`
if grep -RInE '(^|[^a-zA-Z_])anyhow(::|[^a-zA-Z_])' $CORE --include='*.rs' \
     | grep -v '_test\.rs' ; then
  echo "ERROR: anyhow referenced in core library code (D-14 violation)"; exit 1
fi
echo "OK: no anyhow in core library code"
```
> Belt-and-suspenders only; the *structural* ban (not listing `anyhow` in `[dependencies]`) is the primary defense (D-14).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-file `#![deny(clippy::...)]` attributes | `[workspace.lints]` central table | Rust 1.74 (stable) | DRY workspace-wide lint policy (D-13); confirmed available on installed 1.96 |
| `thiserror` 1.x | `thiserror` 2.x (2.0.18) | 2024 | 2.x is current; minor API/edition shifts — pin 2.x in core crates |
| Hand-parsed `.npy` | `ndarray-npy` `read_npy`/`ReadNpyExt` | mature | Use the crate; f64 exact round-trip |
| C++ harness for RNG capture | Transcribe committed `fast_ut.cpp` vectors | n/a (project choice) | Removes a C++ build dependency from Phase 1 |

**Deprecated/outdated:**
- Inline `#[cfg(test)] mod tests` (vendored `rust-package` pattern) — banned by D-17/AGENTS.md; do not replicate.
- `error-chain` / manual `Box<dyn Error>` library errors — superseded by `thiserror`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `approx` latest stable is still 0.5.x and the right test idiom | Supporting stack | Low — test-only; swap assertion lib if needed |
| A2 | `serde`/`serde_json` latest stable suffice for `config.json` metadata parsing | Supporting stack | Low — ubiquitous, stable |
| A3 | A `numpy` version compatible with `catboost==1.2.10` installs cleanly in the generator env | Supporting / Environment | Medium — if catboost 1.2.10 pins an old numpy, generator env may need a specific numpy pin; resolve at generator-env setup |
| A4 | `clippy::indexing_slicing` is the correct lint name for index/slice panics (vs a renamed/grouped lint) | INFRA-02 | Low — verified as an active restriction lint this session; names are stable on 1.96 |
| A5 | Package ages/download characterizations in the Legitimacy Audit (the `package-legitimacy` seam was unavailable) | Package Legitimacy Audit | Low — all are canonical official-org crates; planner should still pin exact versions |
| A6 | The facade `catboost-rs` is treated as a thiserror-only library for anyhow purposes (D-15 default) | INFRA-02 | Low — explicitly an open question to settle in planning (D-15) |

**Note:** The PRNG algorithm, its test vectors, the JSON model schema, the Python oracle API names, `[workspace.lints]` support, and crate versions are all **VERIFIED/CITED** (not assumed) — the assumptions above are confined to minor test-tooling and environment-setup specifics.

## Open Questions (RESOLVED)

> All three questions were settled during Phase 1 planning (plans 01-01 / 01-02 / 01-03). Resolutions annotated inline.

1. **Clippy scope for the test-lint exemption (Pitfall 1)** — **RESOLVED (Plan 01-01 T1):** uniform in-code convention chosen — `#![cfg_attr(test, allow(...))]` in library `lib.rs` + `#![allow(...)]` headers in `tests/*.rs`; no CI `--lib`-flag branching.
   - What we know: restriction lints hit test code; `lints.workspace=true` forbids per-crate manifest overrides; in-code `#![cfg_attr(test, allow(...))]` works for unit tests.
   - Recommendation: prefer in-code `#![cfg_attr(test, allow(...))]` in libs + `#![allow(...)]` headers in `tests/*.rs` (uniform, no CI-flag branching).

2. **Frozen input corpus shape (D-11, Claude's discretion)** — **RESOLVED (Plan 01-03 T2):** 3 tiny datasets specified (pure-numeric, numeric+categorical, grouped/ranking), pinned seed, committed frozen.
   - What we know: must exercise numeric, categorical, and grouped/ranking paths; small.
   - Recommendation: 3 tiny datasets — (a) pure-numeric ~50×4, (b) numeric+categorical ~50×(3 num + 2 cat), (c) grouped/ranking ~60 rows × small features with `group_id`. Pin generation seed; commit. Planning may refine.

3. **Facade anyhow classification (D-15)** — **RESOLVED (applied across plans):** `catboost-rs` facade treated as a thiserror-only library by default; revisit at Phase 4.
   - What we know: default is treat `catboost-rs` as a library (thiserror-only).
   - Recommendation: keep facade thiserror-only unless a Phase 4 ergonomics need surfaces; revisit at Phase 4.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain (stable) | Workspace build, clippy gate | ✓ | rustc/cargo 1.96.0, clippy 0.1.96 | — |
| `cargo clippy` | INFRA-02 lint gate | ✓ | 0.1.96 | — |
| `[workspace.lints]` support | D-13 | ✓ | stable since 1.74 (have 1.96) | — |
| Python 3 | Oracle generator (build-time) | ✓ | 3.12.3 | — |
| `catboost==1.2.10` (PyPI) | INFRA-03/04 fixture generation | ✗ | not installed | Install into a pinned generator venv before fixture generation |
| `numpy` (PyPI) | `.npy` fixture writing | ✗ | not installed | Install with catboost (compatible version) |
| C++ toolchain / CMake | NOT needed in Phase 1 | n/a | — | Deferred (D-08); INFRA-05 uses transcribed vectors, no C++ build |
| ROCm / AMD GPU | NOT in Phase 1 (P7) | n/a | — | Excluded from GH Actions by design (D-16) |

**Missing dependencies with no fallback:** none (all blockers have a clear setup path).
**Missing dependencies with fallback:**
- `catboost==1.2.10` + `numpy` are not installed. **Planner must include a generator-environment setup task** (pinned venv, `pip install "catboost==1.2.10" numpy`) executed once to produce the frozen fixtures, which are then committed. This is build-time only — CI never installs catboost (D-12).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (`cargo test`) + `approx` (`abs_diff_eq!`) for float assertions |
| Config file | none yet — Wave 0 creates `Cargo.toml` workspace + `rust-toolchain.toml` |
| Quick run command | `cargo test -p cb-core` (PRNG vectors) / `cargo test -p cb-oracle` (comparator) |
| Full suite command | `cargo test --workspace` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INFRA-01 | Workspace builds with all crates + features stubbed | build/smoke | `cargo build --workspace` + `cargo build -p cb-backend --features wgpu,cuda,rocm` | ❌ Wave 0 |
| INFRA-02 | Restriction lints deny; anyhow absent from core libs | lint/script | `cargo clippy --workspace -- -D warnings` + `scripts/check-no-anyhow.sh` | ❌ Wave 0 |
| INFRA-03 | One committed `.npy` fixture read + oracle assertion ≤1e-5 | unit/integration | `cargo test -p cb-oracle::compare` | ❌ Wave 0 |
| INFRA-04 | Per-stage comparator (borders/splits/leaf/approx) callable | unit | `cargo test -p cb-oracle` | ❌ Wave 0 |
| INFRA-05 | `TFastRng64` reproduces vendored test vectors exactly | unit | `cargo test -p cb-core rng` | ❌ Wave 0 |
| INFRA-06 | No inline `#[cfg(test)]` in production modules | convention/script | grep gate in CI for `#[cfg(test)]` in non-`*_test.rs` `src/*.rs` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo build -p <crate>` + `cargo test -p <crate>` for the touched crate.
- **Per wave merge:** `cargo clippy --workspace -- -D warnings && cargo test --workspace && scripts/check-no-anyhow.sh`.
- **Phase gate:** Full suite green + GitHub Actions CPU lane green before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `Cargo.toml` (workspace root) — `[workspace]`, `[workspace.lints]`, members
- [ ] `rust-toolchain.toml` — pin stable
- [ ] `crates/cb-core/src/rng.rs` + `rng_test.rs` — covers INFRA-05
- [ ] `crates/cb-core/src/error.rs` — thiserror base (INFRA-02)
- [ ] `crates/cb-oracle/src/{fixture,compare}.rs` + `*_test.rs` — covers INFRA-03/04
- [ ] `crates/cb-oracle/generator/{requirements.txt,gen_inputs.py,gen_fixtures.py}` — build-time oracle
- [ ] `crates/cb-oracle/fixtures/...` — at least one committed input + scenario for the walking skeleton
- [ ] `.github/workflows/ci.yml` — CPU lane (INFRA-01/02/03)
- [ ] `scripts/check-no-anyhow.sh` (+ optional `#[cfg(test)]` grep for INFRA-06) — Wave 0
- [ ] Stub crates `cb-data`/`cb-compute`/`cb-backend`/`cb-train`/`cb-model`/`catboost-rs` — `lib.rs` + `Cargo.toml` (INFRA-01)
- [ ] Generator env: `pip install "catboost==1.2.10" numpy` (one-time, build-time)

## Security Domain

> `security_enforcement: true`, ASVS level 1. This is a build-infrastructure / numerical-library phase with no auth, no network endpoints, no user-facing input at runtime. Most ASVS categories are N/A; the relevant surface is supply-chain integrity and untrusted-data parsing of committed fixtures.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface in this phase |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | No access-control surface |
| V5 Input Validation | yes (light) | `.npy` parsing via `ndarray-npy` (validates dtype/shape, errors instead of panicking); fixtures are repo-controlled, not external input |
| V6 Cryptography | no | `TFastRng64` is a **non-cryptographic** PRNG for parity reproduction ONLY — never use for security/secrets. Document this explicitly in `rng.rs`. |
| V14 Config / Supply Chain | yes | Pin exact crate versions + `catboost==1.2.10`; commit `Cargo.lock`; CI uses only committed fixtures (no network at test time) |

### Known Threat Patterns for {Rust build-infra}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Dependency confusion / slopsquat | Spoofing/Tampering | All deps are canonical official-org crates (Legitimacy Audit); pin versions + commit `Cargo.lock` |
| Panic-on-overflow / index panic in PRNG or comparator | Denial of Service | `wrapping_*` arithmetic (Pitfall 5); `indexing_slicing`+`panic` lints deny accidental panics; comparator returns `Result`, not `unwrap` |
| Misuse of non-crypto PRNG for security | Information Disclosure | Doc-comment `TFastRng64` as parity-only, non-cryptographic |
| Malformed `.npy` fixture | Tampering/DoS | `ndarray-npy` returns errors (no panic) on bad header/dtype; fixtures are repo-reviewed |

## Sources

### Primary (HIGH confidence)
- `catboost-master/util/random/fast.h`, `fast.cpp`, `lcg_engine.h`, `lcg_engine.cpp`, `common_ops.h`, `util/generic/bitops.h` — full `TFastRng64` algorithm (read this session).
- `catboost-master/util/random/fast_ut.cpp` — canonical bitstream test vectors (read this session).
- `catboost-master/catboost/libs/helpers/restorable_rng.{h,cpp}` — seeding/CallCount semantics (read this session).
- `catboost-master/catboost/jvm-packages/.../numeric_only_model.json` — verified `oblivious_trees`/`splits`/`leaf_values` JSON schema (read this session).
- `.planning/codebase/TESTING.md` — mandatory source/test separation, `approx` idiom.
- `cargo search` (2026-06-13) — thiserror 2.0.18, ndarray 0.17.2, ndarray-npy 0.10.0, anyhow 1.0.102.
- Installed toolchain probe — rustc/cargo/clippy 1.96.0, python 3.12.3.

### Secondary (MEDIUM confidence)
- catboost.ai docs via Context7 (`/websites/catboost_ai_en`) — `get_borders()`, `save_model(format='json')`, `staged_predict(eval_period, thread_count)`, prediction APIs for 1.2.10.
- docs.rs/ndarray-npy 0.10.0 — `read_npy`/`ReadNpyExt`/`WriteNpyExt`, f64 support.

### Tertiary (LOW confidence)
- WebSearch — `[workspace.lints]` stability (Rust 1.74), `lints.workspace=true` override constraint, clippy restriction-lint names (cross-checked against installed clippy 0.1.96 which accepts these lints).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions confirmed via `cargo search`; APIs via Context7/docs.rs.
- Architecture: HIGH — fully constrained by CONTEXT.md locked decisions; layout follows D-01..D-17 verbatim.
- PRNG port (INFRA-05): HIGH — algorithm + test vectors read directly from vendored source.
- Oracle API (INFRA-03/04): HIGH — every extraction method verified; JSON schema verified from committed vendored model.
- Pitfalls: HIGH — lint/test interaction, overflow semantics, dtype, and thread determinism all grounded in verified facts.

**Research date:** 2026-06-13
**Valid until:** 2026-07-13 (stable domain; re-confirm crate latest versions and that `catboost==1.2.10`/numpy install cleanly at implementation time).
