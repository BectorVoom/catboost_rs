---
plan: 4
task_id: TASK-04
phase: device-bootstrap-parity
status: pending
order: 4
wave: C
hardware: local ROCm gfx1151 (required) + CPU compile of all three impls
depends_on: [TASK-03]
specifications: [WR01-S4, WR01-S5]
---

# Task 4: Widen the device seam with a per-tree sample; make host sampling exclusive

## Objective

After this task `Runtime::grow_tree_on_device` carries an `n`-length (or empty)
per-tree sample multiplier across the `cb-train → cb-compute → cb-backend` seam,
validated at the boundary, and `DeviceTrainConfig::sample_from_host` makes it
structurally impossible for the host multiplier and a device-resident sampler to
both apply in one fit.

## Specification references

- `WR01-S4` — the per-tree sample crosses the seam, length-validated.
  Primary failure reason: a wrong-length or silently-ignored sample.
- `WR01-S5` — host sampling and device-resident sampling are mutually exclusive.
  Primary failure reason: both samplers run, multiplying the sample twice.

## Context and evidence

- **Exactly three impls exist** `[VERIFIED: RUN — `grep -rn "fn grow_tree_on_device" crates/`]`:
  - trait default `crates/cb-compute/src/runtime.rs:1279-1286` (binds args, returns
    `Ok(None)`),
  - `crates/cb-backend/src/gpu_backend.rs:310` (the real one),
  - `crates/cb-train/tests/device_seam_test.rs:69` (`FakeRuntime`, test-only).
  Plus exactly one caller: `crates/cb-train/src/boosting.rs:3198-3199`.
- **Length-check precedent to mirror**: `gpu_backend.rs:321-327` already rejects a
  wrong-length `approx` with `CbError::LengthMismatch { column, expected, actual }`
  (variant at `crates/cb-core/src/error.rs:46-54`).
- **`DeviceTrainConfig` is plain-host with ~20 call sites**, and every call site uses
  `..DeviceTrainConfig::default()`, so an additive field is safe
  `[RESEARCH, VERIFIED there]`. Definition + `Default` + `is_covered_regime` at
  `crates/cb-compute/src/runtime.rs:1082-1141`.
- **The session's coverage gate** for the oblivious (`None` nonsym-policy) arm is at
  `crates/cb-backend/src/gpu_runtime/session.rs:1038-1092`, with named booleans
  `exact_covered` (`:1053-1057`), `bootstrap_covered` (`:1060-1064`),
  `mvs_covered` (`:1070-1074`), `ctr_is_covered` (`:1079-1083`) and the final
  `if !config.is_covered_regime() && !exact_covered && … { return Ok(None); }`
  (`:1084-1091`).
- **The device-resident sampler arms** live at `session.rs:1546-1635`
  (`self.bootstrap` / `self.mvs`) and must become unreachable when
  `sample_from_host` is set.
- **`cb-compute` must never see a `cubecl` type** — the new field is a `bool`.

## Files

- Modify: `crates/cb-compute/src/runtime.rs`
  - `DeviceTrainConfig`: add `pub sample_from_host: bool` (documented as in SPEC
    §4.1); `Default` → `false`.
  - `is_covered_regime`: add the host-sampled arm — return `true` when
    `sample_from_host` is set and every non-bootstrap family flag is default. Keep
    the existing behaviour byte-identical when `sample_from_host == false`.
  - `grow_tree_on_device`: add `sample: &[f64]`, bind it in the default impl,
    update the doc comment (state the `{0, n}` contract and who owns the sample).
- Modify: `crates/cb-backend/src/gpu_backend.rs` — forward `sample` (`:310-341`).
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs`
  - `grow_one(&mut self, approx, target, sample)`: length check, upload via
    `upload_channel_floats` (`mod.rs:1709`), feed TASK-03's two-handle derivation.
  - `begin`: when `config.sample_from_host`, leave `self.bootstrap` / `self.mvs`
    `None` and admit the regime; keep `bootstrap_covered` / `mvs_covered`
    unchanged for the `sample_from_host == false` path.
- Modify: `crates/cb-train/src/boosting.rs:3198-3199` — pass `&[]` for now (TASK-06
  supplies the real vector).
- Modify: `crates/cb-train/tests/device_seam_test.rs:69` — `FakeRuntime` signature,
  and add an assertion that the fake observed the sample it was handed.

## TDD sequence

### 1. Red

- In `crates/cb-compute/src/runtime_test.rs` — it already exists and is mounted from
  `crates/cb-compute/src/lib.rs:112` as `mod runtime_test;` under `#[cfg(test)]`
  `[VERIFIED]`. Add to it (do not create a new file or a new mount):
  - `default_config_is_not_host_sampled` — `DeviceTrainConfig::default().sample_from_host == false`
    and `is_covered_regime()` is unchanged for every existing combination
    (table-driven over the pre-existing cases).
  - `host_sampled_config_is_covered` — `sample_from_host = true` with
    `bootstrap_type = Bayesian` is a covered regime.
  - `trait_default_grow_binds_sample` — a bare `Runtime` impl returns `Ok(None)`
    when handed a non-empty sample (proves the default binds rather than ignores).
- In `crates/cb-train/tests/device_seam_test.rs`:
  - `fake_runtime_receives_the_sample` — the `FakeRuntime` records
    `sample.len()` and the test asserts the caller passed the expected length.
- In `crates/cb-backend/src/gpu_runtime/session_host_sample_test.rs` (created in
  TASK-03):
  - `wrong_length_sample_is_length_mismatch` — `sample.len() == n - 1` →
    `Err(CbError::LengthMismatch { column: "sample", expected: n, actual: n - 1 })`.
  - `host_sampled_session_opens_without_device_sampler` — with
    `sample_from_host = true, bootstrap_type = Bayesian`, `begin` returns a session
    whose `bootstrap`/`mvs` state is `None` (assert via an existing accessor or a
    `#[cfg(test)]` accessor added on the session — NOT by making the field public).

- Run:
  `cargo test -p cb-compute --lib runtime`
  `cargo test -p cb-train --test device_seam_test`
  `cargo test -p cb-backend --no-default-features --features rocm --lib host_sample -- --test-threads 1`

### 2. Green

1. Add the field + `Default` + `is_covered_regime` arm (cb-compute).
2. Widen the trait method and its doc; bind `sample` in the default impl.
3. Update `gpu_backend.rs` to forward, mirroring the existing `approx` check.
4. Update `session.rs::grow_one` to validate, upload, and feed TASK-03's derivation;
   add the `sample_from_host` early-out that skips both sampler arms.
5. Update the single `cb-train` caller to `&[]`.
6. Update `FakeRuntime`.
- Run: the three commands above.

### 3. Refactor

- Collapse the `approx` and `sample` length checks in `grow_one` into one small
  private `fn check_len(name: &str, got: usize, want: usize) -> CbResult<()>` used
  by both, so a future third buffer cannot drift.
- Make sure the `sample_from_host` early-out reads as an explicit
  `if self.host_sampled { /* device samplers intentionally skipped */ }` with a
  comment naming WR01-S5, not as an implicit consequence of `self.bootstrap` being
  `None`.
- Run: the three commands + `cargo clippy -p cb-compute -p cb-backend --no-deps`.

### 4. Verify

- Run: `cargo test -p cb-compute`
- Run: `cargo test -p cb-train --test device_seam_test`
- Run: `cargo test -p cb-backend --no-default-features --features rocm --lib -- --test-threads 1`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_oblivious_parity_test -- --nocapture`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_fit_test`
- Run: `cargo test -p cb-train` (CPU suite; one known pre-existing red)
- Run: `cargo build -p cb-train --no-default-features --features cpu` and
  `--features wgpu` and `--features cuda` — the seam change must compile on every
  backend, not just rocm.
- Confirm: `grep -rn "fn grow_tree_on_device" crates/` still returns exactly three
  hits, all with the new signature.

## Implementation steps

1. cb-compute first (the type + trait), building alone.
2. cb-backend forwarding.
3. cb-train caller + `FakeRuntime`.
4. Session gate arm + tests.
5. Re-run the full verify set on all four backend features.

## Completion criteria

- [ ] All three `grow_tree_on_device` impls and the single caller carry `sample`.
- [ ] A wrong-length sample yields `CbError::LengthMismatch` before any launch.
- [ ] `DeviceTrainConfig::default().sample_from_host == false` and every
      pre-existing `is_covered_regime()` result is unchanged.
- [ ] A `sample_from_host` session never constructs `BootstrapState` / `MvsState`.
- [ ] The four backend feature builds compile.
- [ ] Every device oracle from TASK-01/03 still green with unchanged numbers.

## Risks and guardrails

- **Seam churn breaking an unrelated backend build.** Guard: the explicit four-way
  `cargo build` in Verify.
- **Making a session field public to test it.** Guard: add a `#[cfg(test)]`
  accessor method instead; never widen production visibility for a test.
- **`is_covered_regime` drift.** It is used by both the gate and (indirectly) the
  documentation of the covered regime. Guard: the table-driven test pinning every
  pre-existing combination before adding the new arm.
- **Double sampling.** Guard: WR01-S5's test asserts the sampler states are `None`,
  not merely that the result "looks right".
