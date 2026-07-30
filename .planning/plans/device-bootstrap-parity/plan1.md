---
plan: 1
task_id: TASK-01
phase: device-bootstrap-parity
status: pending
order: 1
wave: A
hardware: local ROCm gfx1151 (required)
depends_on: [none]
specifications: [WR01-S11]
---

# Task 1: Promote the session probe into a committed ≤1e-5 base-grower oracle

## Objective

After this task the repository contains a running, committed proof that the device
oblivious grower reproduces the CPU grower's predictions to **≤1e-5** with
`bootstrap_type = No` — the precondition every later bootstrap claim rests on.
Today the only shipped device e2e tolerance is ε=1e-4
(`device_nonsym_fit_test.rs:167`), and the ≤1e-5 evidence exists only as an
uncommitted probe file.

## Specification references

- `WR01-S11` — the base device oblivious grower holds ≤1e-5 vs the CPU grower.
  Primary failure reason: the base grower cannot meet the phase bar.

## Context and evidence

- Uncommitted probe: `crates/cb-train/tests/device_oblivious_parity_probe_test.rs`
  (whole file read this session). It already contains the fixture generator, the
  device-eligible `BoostParams`, the `CpuRefRuntime`, and the delta computation.
  It asserts only ε=1e-4 and prints the ≤1e-5 verdict.
- Measured on gfx1151 this session:
  `512×4 d3 ×5 → 2.22e-16`, `2048×8 d6 ×10 → 2.22e-16`,
  `20000×16 d6 ×20 → 4.44e-16` max |Δpred|; split mismatches `0/5`, `0/10`, `3/20`.
- Skip / `CpuRefRuntime` pattern to mirror: `device_nonsym_fit_test.rs:92-126`
  (a `Runtime` impl overriding only `compute_gradients`, inheriting
  `begin → Ok(false)` / `grow → Ok(None)`), and the `#[cfg(not(...))]`
  `eprintln!("SKIP …")` arms at `:190-210`.
- Device eligibility clauses the fixture must satisfy: `boosting.rs:3054-3098`
  (SymmetricTree, `approx_dimension == 1`, `bootstrap_type = No`,
  `random_strength == 0.0`, unit weights, `bias == 0.0` i.e.
  `boost_from_average = false`, `LeafMethod::Gradient`).
- `GpuBackend` is not compiled under `cpu`; `CpuBackend` is not compiled under
  `rocm`/`cuda` (`crates/cb-backend/src/lib.rs:33-37`) — hence the hand-rolled
  reference runtime.

## Files

- Create: `crates/cb-train/tests/device_oblivious_parity_test.rs`
- Delete: `crates/cb-train/tests/device_oblivious_parity_probe_test.rs`
  (its content is the starting point; do not keep both — a duplicate ε=1e-4 copy is
  exactly the tolerance-creep risk R9)
- Modify: none in `src/`

## TDD sequence

### 1. Red

- Copy the probe into the new file, then tighten the single assertion from
  `max_abs <= 1e-4` to `max_abs <= 1e-5`, and add a second assertion
  `max_leaf_on_structurally_identical_trees <= 1e-5` (leaf deltas compared ONLY for
  trees whose split sequences match — the 4.35e-1 leaf delta at 20000×16 comes from
  the 3 structurally different trees and is not a numerical error).
- Add a third shape that stresses sampling-like masking without sampling:
  `probe(20_000, 16, 6, 20)` is retained; add `probe(1500, 4, 2, 3)` so the shape of
  the upstream bootstrap fixture (1500 objects, 4 features, depth 2, 3 iterations)
  is covered by the base oracle too.
- Expected failure BEFORE running on hardware: on a `cpu`/`wgpu` build the test must
  print `SKIP` and pass; on the ROCm rig it must compile and run. Confirm the RED
  state is meaningful by temporarily asserting `max_abs <= 1e-18` and observing a
  failure that reports the real measured value.
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_oblivious_parity_test -- --nocapture`

### 2. Green

- Restore ε=1e-5 and confirm all four shapes pass.
- Keep the split-mismatch count as a PRINTED diagnostic with an explicit module-doc
  sentence explaining why structure equality is not asserted (forward reference to
  TASK-02 / WR01-S12 and the fixed-point histogram, `kernels.rs:2335`,
  `:3889-3893`).
- Ensure the `#[cfg(not(any(feature = "rocm", feature = "cuda")))]` arm still
  references `fixture` and `oblivious_params` so `dead_code` stays silent under cpu.
- Run: the same command, plus
  `cargo test -p cb-train --test device_oblivious_parity_test` (cpu build → SKIP path).

### 3. Refactor

- Extract the three-way shape loop into a `const SHAPES: [(usize, usize, usize, usize); 4]`
  and one call site, so adding a shape is a one-line change.
- Move the per-shape reporting into one `fn report(...)` to remove the duplicated
  `println!`.
- Behaviour must not change: re-run both commands and compare the printed numbers.

### 4. Verify

- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_oblivious_parity_test -- --nocapture`
- Run: `cargo test -p cb-train --test device_oblivious_parity_test` (SKIP arm)
- Run: `cargo clippy -p cb-train --tests --no-deps`
- Confirm: the printed line for each shape reports `holds_1e-5=true`, and the
  20000×16 shape still reports its split-mismatch count (expected 3/20 — record the
  actual number in `progress.md`; it is the input to TASK-02).

## Implementation steps

1. `git mv crates/cb-train/tests/device_oblivious_parity_probe_test.rs crates/cb-train/tests/device_oblivious_parity_test.rs`.
2. Rewrite the module doc: this is now a GATE, not a probe; state ε=1e-5, state that
   split-structure equality is intentionally not asserted, and cite
   `kernels.rs:2335` for why.
3. Add the `(1500, 4, 2, 3)` shape.
4. Tighten the prediction assertion to 1e-5; add the structurally-matched leaf-delta
   assertion.
5. Rename the test fn to `device_oblivious_grower_matches_cpu_within_1e5`.
6. Run on the ROCm rig; record every measured number in `progress.md`.

## Completion criteria

- [ ] The Red test fails for the stated reason before the tolerance is set correctly
      (verified with the temporary 1e-18 bound).
- [ ] All four shapes pass at ≤1e-5 on gfx1151.
- [ ] The cpu build prints `SKIP` and passes.
- [ ] The probe file no longer exists (no ε=1e-4 duplicate remains).
- [ ] Measured `max|Δpred|`, `max|Δleaf|`, and split-mismatch counts are recorded in
      `progress.md`.
- [ ] `cargo clippy -p cb-train --tests --no-deps` is clean.

## Risks and guardrails

- **R9 tolerance creep** — the probe's ε=1e-4 must not survive anywhere. Guard:
  delete the probe file rather than keeping it alongside.
- **False pass on a non-device build** — the `SKIP` arm must not silently satisfy CI
  as a "device test passed". Guard: the printed message names the test and the
  missing feature, matching the shipped convention.
- **Flaky leaf deltas** — leaf values carry float-atomic jitter (F-E). Guard: only
  compare leaves for structurally identical trees, and leave the jitter budget
  itself to TASK-10.
