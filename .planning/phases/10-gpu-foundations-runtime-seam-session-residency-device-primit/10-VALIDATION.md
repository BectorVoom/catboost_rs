---
phase: 10
slug: gpu-foundations-runtime-seam-session-residency-device-primit
status: approved
nyquist_compliant: true
wave_0_complete: false
created: 2026-07-03
reconciled: 2026-07-03
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution. Reconciled against the finalized 9-plan / 8-wave decomposition (10-01 … 10-09).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (workspace), SEPARATE test files per CLAUDE.md (no `#[cfg(test)] mod tests` in production). GPU self-oracles live in `cb-backend/src/kernels/*.rs`. Authoritative GPU correctness+speed oracle is the human-gated Kaggle CUDA `.ipynb`. |
| **Config file** | none — Cargo workspace; Kaggle notebook committed under `bench/` |
| **Quick run command** | `cargo test -p cb-backend --no-default-features --features rocm <primitive>` (in-env ROCm smoke) |
| **Full suite command** | `cargo test --workspace` + human-gated Kaggle CUDA notebook (`--features cuda`, correctness-gate-then-speed) |
| **Estimated runtime** | ~seconds for CPU/ROCm-smoke unit tests; Kaggle CUDA run is human-gated (out-of-band) |

---

## Sampling Rate

- **After every task commit:** Run the task's `<automated>` command (per-primitive `cargo test -p cb-backend … <name>` / `cargo test -p cb-train`)
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd-verify-work`:** Full CPU/ROCm-smoke suite green; Kaggle CUDA correctness oracle (blocking gate) discharged for every GPU deliverable (GPUT-04/08/15/16, BENCH-01/02)
- **Max feedback latency:** CPU/ROCm seconds; GPU correctness+speed sign-off is human-gated per ROADMAP validation authority (Kaggle CUDA)

---

## Per-Task Verification Map

> One row per `<task>` across the 9 finalized plans (20 tasks). GPU-kernel tasks carry a serial-CPU self-oracle (D-02, the CI-checkable part) plus a Kaggle-CUDA authoritative run (D-01/D-05, human-gated). ROCm in-env is optional smoke, NOT a gate.

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | GPU Oracle (Kaggle) | Status |
|---------|------|------|-------------|-----------|-------------------|---------------------|--------|
| 10-01-01 | 01 | 1 | GPUT-16 | unit (self-oracle) | `cargo test -p cb-backend … rocm scan` | ≤1e-4 | ⬜ pending |
| 10-01-02 | 01 | 1 | GPUT-16 | unit (self-oracle) | `cargo test -p cb-backend … rocm segmented_scan` | ≤1e-4 | ⬜ pending |
| 10-02-01 | 02 | 1 | GPUT-01 | build+unit | `cargo build -p cb-compute && cargo test -p cb-train` | N/A (CubeCL-free seam) | ⬜ pending |
| 10-03-01 | 03 | 2 | GPUT-16 | unit (self-oracle) | `cargo test -p cb-backend … rocm reduce` | ≤1e-4 | ⬜ pending |
| 10-03-02 | 03 | 2 | GPUT-16 | unit (self-oracle) | `cargo test -p cb-backend … rocm reduce` | ≤1e-4 | ⬜ pending |
| 10-03-03 | 03 | 2 | GPUT-16 | doc gate (spike) | `test -f …/SPIKE-REDUCTION.md && grep -q 'Recommendation' …` | variance+speed measured | ⬜ pending |
| 10-04-01 | 04 | 3 | GPUT-16 | unit (self-oracle) | `cargo test -p cb-backend … rocm sort` | bit-exact / ≤1e-4 | ⬜ pending |
| 10-04-02 | 04 | 3 | GPUT-16 | unit (self-oracle) | `cargo test -p cb-backend … rocm partitions` | bit-exact | ⬜ pending |
| 10-04-03 | 04 | 3 | GPUT-16 | unit (self-oracle) | `cargo test -p cb-backend … rocm fill_transform` | ≤1e-4 (transitive) | ⬜ pending |
| 10-05-01 | 05 | 4 | GPUT-16 | unit (self-oracle) | `cargo test -p cb-backend … rocm compression` | bit-exact | ⬜ pending |
| 10-05-02 | 05 | 4 | GPUT-16 | unit (self-oracle) | `cargo test -p cb-backend … rocm update_part_props` | ≤1e-4 | ⬜ pending |
| 10-06-01 | 06 | 5 | GPUT-15 | unit (self-oracle) | `cargo test -p cb-backend … rocm cindex` | bit-exact vs CPU layout | ⬜ pending |
| 10-06-02 | 06 | 5 | GPUT-15 | unit (consumer) | `cargo test -p cb-backend … rocm pointwise_hist cindex` | ≤1e-4 | ⬜ pending |
| 10-07-01 | 07 | 6 | GPUT-04/08 | unit (self-oracle) | `cargo test -p cb-backend … rocm grow_oblivious depth1` | ≤1e-5 (depth-1) | ⬜ pending |
| 10-07-02 | 07 | 6 | GPUT-02/03 | unit (residency) | `cargo test -p cb-backend … rocm session_residency` | upload-once / no der1 read-back | ⬜ pending |
| 10-07-03 | 07 | 6 | GPUT-02 | build+unit | `cargo build -p cb-backend … && cargo test … gpu_backend` | N/A (host wiring) | ⬜ pending |
| 10-08-01 | 08 | 7 | GPUT-01/04 | integration | `cargo test -p cb-train` | Ok(None) fallback + depth-1 branch | ⬜ pending |
| 10-08-02 | 08 | 7 | GPUT-01/04 | integration | `cargo test -p cb-train` | bin_id→border join | ⬜ pending |
| 10-09-01 | 09 | 8 | BENCH-01/02 | script gate | `python3 … ast.parse('bench/generator.py') … && test -f bench/fixtures/README.md` | seeded generator (D-06) | ⬜ pending |
| 10-09-02 | 09 | 8 | BENCH-01/02 | notebook gate | `python3 … json.load('bench/cuda_oracle.ipynb') … && grep 'large n' bench/RESULTS.md` | correctness-gate-then-speed (D-10-09 large-n) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky. `GPU Oracle (Kaggle)` = the human-gated ε bar / sign-off; the `Automated Command` is the CI/in-env-checkable part.*

---

## Wave 0 Requirements

Provisioned inside the owning plans (not a separate Wave 0 crate):

- [ ] Serial-CPU self-oracle harness for standalone primitives — scan/segmented-scan (10-01), reduce/reduce-by-key (10-03), radix sort + single-bit reorder + partitions (10-04), compression + `update_part_props` (10-05), cindex (10-06) — D-01/D-02
- [ ] Seeded synthetic generator `bench/generator.py` — one source for the depth-1 ≤1e-5 correctness fixture AND the large-n (~1e6×50) speed workload (10-09, D-06)
- [ ] Kaggle CUDA `bench/cuda_oracle.ipynb` scaffold — build `--features cuda` wheel, load committed seeded fixtures, correctness-first blocking gate, warm-run/JIT-excluded train-only wall-clock, structured `bench/RESULTS.md` (10-09, BENCH-01)

*`wave_0_complete` flips true once 10-01/10-09 land the self-oracle harness + generator + notebook scaffold.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Primitives + cindex ≤1e-4; depth-1 device tree ≤1e-5 vs CPU | GPUT-04/08/15/16 | Authoritative GPU run is Kaggle CUDA (human-gated external notebook); ROCm in-env is optional smoke, not a gate | Run `bench/cuda_oracle.ipynb` on a Kaggle CUDA instance; correctness report must pass at the ε bar before speed is read |
| Warm-run device-vs-CPU wall-clock at large n | BENCH-01/02 | Wall-clock speed measured off-CI on real CUDA hardware | Same notebook; read the warm-run/JIT-excluded train-only speed section. D-10-09: depth-1 device≥CPU only holds at large n (~1e5–1e6) — bar pinned to the large-n synthetic workload, NOT small n |
| Reduction-determinism winner selection | GPUT-16 (D-03/04) | Run-to-run variance + speed measured on Kaggle CUDA | Inspect `SPIKE-REDUCTION.md` recommendation; the winner is the shipped reduce primitive |

---

## Validation Sign-Off

- [x] All tasks have an `<automated>` verify (CPU/ROCm self-oracle or build+test) plus a Kaggle-CUDA authoritative check for GPU deliverables
- [x] Sampling continuity: no 3 consecutive tasks without automated verify (every task row carries a command)
- [x] Wave 0 covers all MISSING references (self-oracle harness, seeded generator, notebook scaffold) — owned by 10-01/10-03/10-04/10-05/10-06/10-09
- [x] No watch-mode flags
- [x] Feedback latency < seconds (CPU/ROCm); GPU correctness+speed sign-off human-gated by design
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-07-03 (reconciled against finalized 9-plan decomposition)
