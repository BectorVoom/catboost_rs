---
plan: 6
task_id: TASK-06
phase: device-bootstrap-parity
status: pending
order: 6
wave: D
hardware: local ROCm gfx1151 (required) + CPU no-regression
depends_on: [TASK-04, TASK-05]
specifications: [WR01-S6, WR01-S9]
---

# Task 6: Open the gate for Bayesian / Bernoulli / MVS and build the host multiplier

## Objective

After this task a `bootstrap_type ∈ {Bayesian, Bernoulli, Mvs}` fit that satisfies
every other eligibility clause runs on the device, with the host computing the
per-tree multiplier `s[i] = control[i] ? sample_weights[i] : 0.0` from the frozen
CPU `bootstrap()` and shipping it across the TASK-04 seam.

This is the task that changes user-visible behaviour. It lands only after the
mechanism (TASK-03/04) and the stream phase (TASK-05) are proven, and it is
immediately followed by the parity oracles (TASK-08).

## Specification references

- `WR01-S9` — the eligibility gate admits Bayesian / Bernoulli / MVS and threads the
  config. Primary failure reason: the wrong set of bootstrap types reaches the
  device.
- `WR01-S6` — the host builds the per-tree sample multiplier from `bootstrap()`.
  Primary failure reason: the multiplier is not
  `control[i] ? sample_weights[i] : 0.0`.

## Context and evidence

- **The gate.** `device_host_eligible` at `crates/cb-train/src/boosting.rs:3054-3098`;
  the bootstrap clause is `&& matches!(params.bootstrap_type, EBootstrapType::No)`
  at `:3073` `[VERIFIED]`. Every other clause stays: `random_strength == 0.0`
  (`:3074`), unit weights (`:3086`), `bias == 0.0` (`:3092`),
  `LeafMethod::Gradient | Simple` (`:3098`), `approx_dimension == 1` (`:3070`).
- **The config.** `device_config` at `:3133-3160`, currently
  `..DeviceTrainConfig::default()` behind the WR-01 note at `:3143-3159` which this
  task must REPLACE with an accurate statement `[VERIFIED]`.
- **The CPU multiplier this must reproduce.**
  `score_weighted_der1` (`:3738-3751`) multiplies `weighted_der1[i]` by
  `control[i] ? sample_weights[i] : 0.0`; `score_weights` (`:3758-3770`) multiplies
  `eff_weights[i]` (== `weights[i]` for non-pairwise losses) by the same factor
  `[VERIFIED]`. At unit weight the single vector `s` is therefore the complete
  multiplier for BOTH channels.
- **MVS is the only arm that reads derivatives** (`bootstrap.rs:410-412`); the CPU
  builds `der_obj[i] = sqrt(Σ_d weighted_der1[d·n+i]²)` (`boosting.rs:3685-3695`),
  which at `approx_dimension == 1` and unit weights is `|der1[i]|`.
  On the device branch the host owns `approx`, so
  `runtime.compute_gradients(&params.loss, &approx, target, 1)` reproduces it
  exactly. Bernoulli/Bayesian must pass a length-`n` placeholder to avoid the extra
  gradient round-trip (they ignore it).
- **`sum_all_weights` / `scaled_l2` are tree-invariant and sampling-independent**
  (`boosting.rs:2376`, `:3104`, `:3802`) `[RESEARCH finding 9]` — do NOT recompute
  them per tree.
- **`EBootstrapType` → `DeviceBootstrapType`** mapping target enum at
  `crates/cb-compute/src/runtime.rs:1020-1032`.
- **Device-signature assertion precedent**: `device_nonsym_fit_test.rs:136-149`.

## Files

- Modify: `crates/cb-train/src/boosting.rs`
  - `:3073` — replace with an **exhaustive allow-list**:
    `&& matches!(params.bootstrap_type, EBootstrapType::No | EBootstrapType::Bayesian | EBootstrapType::Bernoulli | EBootstrapType::Mvs)`.
    Do NOT write `!matches!(..., Poisson)`: an allow-list makes a future variant a
    compile-time decision instead of a silent enablement (WR01-S9 acceptance
    example 3).
  - `:3133-3160` — thread `sample_from_host`, the mapped `bootstrap_type`, and
    `sample_rate = params.subsample as f32`; leave `mvs_lambda: None` and
    `rng_seed: 0` (Design A: the host owns the stream and the λ). Replace the WR-01
    note with an accurate deferral statement (Design B′, Poisson, weighted pools,
    `random_strength != 0`).
  - device branch `:3196-3212` — build `s` and pass it to `grow_tree_on_device`.
- Create: `crates/cb-train/src/device_sample.rs` +
  `crates/cb-train/src/device_sample_test.rs`
  — `pub(crate) fn build_device_sample(result: &BootstrapResult, n: usize) -> Vec<f64>`,
  the one-line multiplier, unit-tested in isolation. Keeping it out of
  `boosting.rs` is what makes WR01-S6 independently testable without a device.
- Modify: `crates/cb-train/src/lib.rs` — `mod device_sample;`

## TDD sequence

### 1. Red

Host-only unit tests in `device_sample_test.rs` (no GPU needed):

- `sample_is_weight_where_control_true_zero_otherwise` — hand-built
  `BootstrapResult` → exact expected vector.
- `sample_matches_cpu_score_weights_at_unit_weight` — for each of Bernoulli,
  Bayesian, MVS with a fixed seed, run `bootstrap()` and assert
  `build_device_sample(&r, n)` equals the CPU `score_weights` expression
  (`eff_weights[i] * sw` masked by control) computed inline with `eff_weights = 1.0`.
  Principal failure reason: the multiplier formula is wrong.
- `sample_length_and_finiteness` — `s.len() == n`, all finite, all `>= 0.0`.

Device integration test additions in
`crates/cb-train/tests/device_bootstrap_gate_test.rs` (new, small — the parity
oracles come in TASK-08):

- `bernoulli_fit_is_device_grown` — assert the fit produced `params.iterations`
  oblivious trees AND an observable device signature (mirror
  `device_nonsym_fit_test.rs:136-149`: assert the arms that only the device fold
  populates / leaves empty), so a silent CPU fallback cannot pass.
- `bayesian_fit_is_device_grown`, `mvs_fit_is_device_grown` — same.
- `random_strength_nonzero_falls_back_to_cpu` — a Bernoulli fit with
  `random_strength = 0.5` must NOT be device-grown (the other clauses stay closed).
- `weighted_pool_falls_back_to_cpu` — non-unit weights → CPU.

- Run:
  `cargo test -p cb-train --lib device_sample`
  `cargo test -p cb-train --no-default-features --features rocm --test device_bootstrap_gate_test -- --nocapture`

### 2. Green

1. `build_device_sample` (three lines).
2. The gate allow-list.
3. The config threading + WR-01 note replacement.
4. The device-branch body, in the CPU's order:
   `PRE_TREE_DRAWS` (landed in TASK-05) → `der_obj` (MVS only, via
   `runtime.compute_gradients`) → `bootstrap(...)` → `build_device_sample` →
   `runtime.grow_tree_on_device(&approx, target, &s)?` → fold → TASK-05's replay +
   `POST_TREE_EXTRA_DRAWS` + `prev_leaf_mean_l2`.
5. Keep the `Ok(None)`-after-`Ok(true)` typed error at `:3200-3212` unchanged
   (D-10-01).
- Run: the two commands above.

### 3. Refactor

- Extract the device-branch per-tree sampling preamble into one private fn
  `fn device_tree_sample(...) -> CbResult<Vec<f64>>` so the branch body stays
  readable and the MVS-only gradient round-trip is visible in one place.
- Ensure the Bernoulli/Bayesian placeholder path allocates once outside the
  iteration loop, not per tree (it is a `vec![0.0; n]` that `bootstrap()` ignores).
- Run: both commands.

### 4. Verify

- Run: `cargo test -p cb-train --lib device_sample`
- Run: `cargo test -p cb-train --test bootstrap_oracle_test` (**blocking**)
- Run: `cargo test -p cb-train --test regularization_oracle_test`
- Run: `cargo test -p cb-train` (CPU suite; one known pre-existing red)
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_bootstrap_gate_test -- --nocapture`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_oblivious_parity_test -- --nocapture`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_fit_test`
- Run: `cargo clippy -p cb-train --lib --tests --no-deps`
- Confirm: the WR-01 comment at `boosting.rs:3143-3159` no longer claims the
  sampling knobs are unwired.

## Implementation steps

1. Land `build_device_sample` + its unit tests (no gate change yet) — green.
2. Land the config threading with the gate still closed — the CPU suite must be
   byte-identical (the config is only read when the fit reaches `begin`).
3. Open the gate for **Bernoulli only** first (`s ∈ {0,1}`, bit-for-bit sampler, the
   simplest mask) and run the device gate test.
4. Add Bayesian.
5. Add MVS last (it needs TASK-05's λ carry and the extra gradient round-trip).
6. Only then proceed to TASK-08.

## Completion criteria

- [ ] `build_device_sample` matches the CPU `score_weights` expression for all three
      types at a fixed seed.
- [ ] All three types are observably device-grown under rocm.
- [ ] `random_strength != 0` and weighted pools still fall back to CPU.
- [ ] Poisson is NOT in the allow-list (TASK-09 owns its contract).
- [ ] `bootstrap_oracle_test` green and unchanged (**blocking**).
- [ ] The stale WR-01 note is replaced with an accurate one.

## Risks and guardrails

- **R6 silent CPU fallback** — the whole task is worthless if the "device" tests ran
  on CPU. Guard: an explicit device-signature assertion in every gate test, not just
  a tree count.
- **R5 fixtures not device-eligible** — this task's gate tests use synthetic bias-0
  fixtures; the upstream fixtures arrive in TASK-07.
- **Enabling all three at once** — a failure would be ambiguous. Guard: the
  one-type-at-a-time implementation order in steps 3-5.
- **MVS gradient round-trip cost** — `compute_gradients` on the device branch is an
  extra `n`-length crossing per tree for MVS only. Acceptable for correctness-first;
  note it in `progress.md` as a Design B′ motivation, do not optimise here.
- **Do not thread `mvs_lambda`** — doing so would open the device-resident MVS
  sampler arm and double-sample. WR01-S5's mutual exclusion makes this safe, but the
  config should still say `None` so intent is unambiguous.
