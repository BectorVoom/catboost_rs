---
plan: 10
task_id: TASK-10
phase: device-bootstrap-parity
status: pending
order: 10
wave: F
hardware: local ROCm gfx1151 (required) + Kaggle CUDA P100 (required for sign-off)
depends_on: [TASK-08, TASK-09]
specifications: [WR01-S13]
---

# Task 10: Determinism budget + CUDA sign-off + phase close-out

## Objective

After this task the phase's ≤1e-5 claim is backed by a measured run-to-run
determinism budget (so it does not silently depend on a lucky float-atomic
schedule), confirmed on a second GPU architecture (Kaggle CUDA P100), and the
repository's own documentation of what is and is not device-covered is accurate.

## Specification references

- `WR01-S13` — the device leaf reduce's nondeterminism stays inside the ≤1e-5
  budget. Primary failure reason: repeated identical device fits differ by more
  than the sign-off tolerance.

## Context and evidence

- **The leaf reduce is explicitly nondeterministic.** `partition_update_kernel`
  merges with a NAKED float atomic (`Atomic<F>::fetch_add`), and its own doc says
  the fixed-point determinism guarantee "applies to the HISTOGRAM FILL … only, NOT
  to this kernel", that tree STRUCTURE is bit-identical but LEAF VALUES and
  PREDICTIONS "carry ulp-level run-to-run float-order variance", and that "a STRICT
  bit-reproducibility claim for predictions is out of scope here"
  `[VERIFIED: crates/cb-backend/src/kernels.rs:3800-3826, kernel body :3826-3862]`.
- The same doc names the fix: "a future fixed-point `Atomic<u64>` encode/decode of
  the `d`/`w`/`h` channels (mirroring the histogram fill) would close the gap."
- **Measured so far**: leaf deltas 5.55e-17 / 8.33e-17 device-vs-CPU on
  structurally identical trees (TASK-01's shapes). That is the *cross-implementation*
  delta, not the *run-to-run* one, which has never been measured in this repo.
- **Sampling can enlarge it**: MVS's `1/prob` weights raise the magnitudes summed by
  the atomic, and absolute float-order error scales with magnitude.
- **CUDA sign-off convention**: prior phases treat Kaggle CUDA as a human-gated
  confirmation of a locally-green ROCm result (`kernels/partitions.rs:9-11` records
  the pattern).

## Files

- Create: `crates/cb-train/tests/device_bootstrap_determinism_test.rs`
- Modify: `crates/cb-train/src/boosting.rs` — final accuracy pass on the WR-01
  comment block (`:3143-3159` region, replaced in TASK-06): confirm it names what is
  now wired (host-sampled Bayesian/Bernoulli/MVS) and what remains deferred
  (Design B′ device-resident sampling, Poisson, weighted pools WR-03,
  `random_strength != 0`, non-symmetric/Region/CTR × bootstrap).
- Modify: `.planning/plans/device-bootstrap-parity/progress.md` — final evidence.
- Modify (or delete): `bench/bootstrap_gpu/bootstrap_bench.py` — its
  `only_No_is_gpu_eligible` caveat is false after TASK-06. It is untracked; decide
  explicitly to commit the corrected version or delete it, and record which.

## TDD sequence

### 1. Red

In `device_bootstrap_determinism_test.rs` (device-gated, SKIP arm as usual):

- `repeated_bayesian_device_fit_is_structurally_identical` — run the same fit `k = 5`
  times in one process; assert every tree's `splits` vector is **equal** across all
  five runs. This must hold by construction (fixed-point histogram) and is the
  cheapest canary that the structure path did not regress.
- `repeated_bayesian_device_fit_predictions_within_1e7` — assert pairwise
  `max|Δpred| ≤ 1e-7` across the five runs. 1e-7 is two decades inside the phase's
  1e-5 sign-off bar, so leaf-reduce jitter provably cannot be what breaks parity.
  Principal failure reason: the float-atomic leaf reduce is noisier than the budget.
- `repeated_mvs_device_fit_predictions_within_1e7` — the same at
  `bootstrap_type = Mvs`, which carries the largest weight magnitudes (F-D) and is
  therefore the worst case for float-order error.
- `mvs_fixedpoint_range_headroom_is_recorded` — instrument the TASK-03 guard (or
  recompute its bound in the test) and PRINT the observed
  `n · max|s| · max(1, max|der1|)` against the `2^33` limit for the largest in-phase
  MVS fit. Assert only that it is below the limit; the printed headroom ratio is the
  evidence for whether the conservative per-fit bound needs to become per-bin.

- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_bootstrap_determinism_test -- --nocapture --test-threads 1`

### 2. Green

- No production change expected. If `≤1e-7` fails, apply the **Escalation** below
  rather than loosening the budget.
- Run: the same command.

### 3. Refactor

- Fold the repeated-fit helper into `crates/cb-train/tests/common/device_fixture.rs`
  as `fn repeat_fit(k: usize, …) -> Vec<Model>` so future device phases reuse it.
- Run: every device test target (list in Verify).

### 4. Verify — local ROCm

- Run each of these under `--no-default-features --features rocm`:
  - `--test device_oblivious_parity_test`
  - `--test device_split_tiebreak_test`
  - `--test device_bootstrap_gate_test`
  - `--test device_bootstrap_fit_test`
  - `--test bootstrap_dev_oracle_test`
  - `--test device_bootstrap_determinism_test`
  - `--test poisson_contract_test`
  - `--test device_nonsym_fit_test`
  - `--test device_region_fit_test`
  - `--test device_seam_test`
- Run: `cargo test -p cb-backend --no-default-features --features rocm --lib -- --test-threads 1`
- Run: `cargo test -p cb-train` (CPU suite; exactly one known pre-existing red)
- Run: `cargo test -p cb-train --test bootstrap_oracle_test` (**blocking**)
- Run: `cargo test --workspace` on the default (cpu) features, recording any
  pre-existing reds separately from this phase's.
- Run: `cargo clippy --workspace --all-targets --no-deps` and
  `cargo clippy -p cb-backend --no-default-features --features rocm --all-targets --no-deps`

### 5. Verify — Kaggle CUDA P100 (sign-off)

Run the same six phase-owned device targets with `--features cuda` in place of
`rocm`:

```
cargo test -p cb-train --no-default-features --features cuda --test device_oblivious_parity_test -- --nocapture
cargo test -p cb-train --no-default-features --features cuda --test device_bootstrap_fit_test -- --nocapture
cargo test -p cb-train --no-default-features --features cuda --test bootstrap_dev_oracle_test -- --nocapture
cargo test -p cb-train --no-default-features --features cuda --test device_bootstrap_determinism_test -- --nocapture
cargo test -p cb-train --no-default-features --features cuda --test device_split_tiebreak_test -- --nocapture
cargo test -p cb-backend --no-default-features --features cuda --lib -- --test-threads 1
```

Record, per target: pass/fail and every printed delta. **The CUDA numbers are
expected to differ from ROCm in the last few ulps** (different wave width, different
atomic scheduling) but must still satisfy ≤1e-5 (and ≤1e-7 for determinism). A CUDA
delta materially larger than the ROCm one is a finding, not a rounding detail —
record it.

## Escalation (only if the ≤1e-7 determinism budget fails)

Promote `partition_update_kernel`'s three channels to fixed-point `Atomic<u64>`,
mirroring the histogram fill:

- Reuse `fixedpoint_encode` / `fixedpoint_decode` (`kernels.rs:3889-3910`) — do NOT
  write new quantization math.
- The `part_stats` buffer becomes `Array<Atomic<u64>>` seeded to `0u64`; the host
  decodes after `read_part_stats_and_leaf_of`.
- The SAME `|Σ| < 2^33` range precondition then applies to the LEAF channels too —
  extend the TASK-03 guard accordingly (the leaf channel sums the UNSAMPLED
  `der1`/`weight`, so its magnitudes are bounded by the no-bootstrap case and this is
  the easier half).
- Re-run TASK-01 and TASK-08 in full: the leaf values change in the last ulps, so
  every recorded delta must be re-measured.
- This is a CubeCL change: read
  `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` first, and on
  any build error read `cubecl_error_guideline.md` before attempting a fix.

Treat the escalation as a new task (`plan11.md`) rather than absorbing it silently.

## Completion criteria

- [ ] Structure is bit-identical across 5 repeated device fits (Bayesian and MVS).
- [ ] `max|Δpred|` across repeated fits ≤ 1e-7 on ROCm and on CUDA.
- [ ] The MVS fixed-point headroom ratio is measured and recorded.
- [ ] All ten device targets green on ROCm; the six phase-owned ones green on CUDA.
- [ ] `bootstrap_oracle_test` green and unchanged; `bootstrap/` fixtures unchanged.
- [ ] The WR-01 comment in `boosting.rs` accurately states wired vs deferred.
- [ ] `bench/bootstrap_gpu/` is either corrected and committed or deleted — decided
      explicitly, recorded in `progress.md`.
- [ ] `progress.md` carries every measured number from TASK-01 … TASK-10.
- [ ] Workspace clippy clean on both cpu and rocm all-targets.

## Risks and guardrails

- **R12 leaf-reduce jitter** — the entire point; the budget is 1e-7 precisely so that
  passing it settles the question rather than deferring it.
- **CUDA-only divergence** — batching CUDA at the end means a CUDA-specific failure
  arrives late. Mitigation: the failure modes are architecture-independent by
  construction (fixed-point histogram, host-side sampling, host RNG), so the residual
  CUDA risk is confined to float-atomic scheduling — exactly what the determinism
  test measures.
- **Loosening a tolerance to close the phase** — prohibited. If ≤1e-7 fails, escalate;
  if ≤1e-5 fails anywhere, the phase does not close.
- **Untracked bench artifacts drifting into the commit** — decide explicitly.
