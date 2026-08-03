---
plan: 3
task_id: TASK-03
phase: device-bootstrap-parity
status: pending
order: 3
wave: B
hardware: local ROCm gfx1151 (required)
depends_on: [TASK-01]
specifications: [WR01-S1, WR01-S2, WR01-S3, WR01-S10]
---

# Task 3: Split the device score channel from the leaf channel (+ fixed-point range guard)

## Objective

After this task `grow_oblivious_tree_resident` fills the split histogram from a
**sampled** stat pair and estimates leaves from the **unsampled** pair, and a
pre-launch guard rejects any sample that would overflow the `2^30` fixed-point
histogram encode. With an empty sample everything is byte-identical to `5a5068a`.

This is the largest, riskiest change in the phase and it fixes a **latent
correctness bug**: today the already-landed (but unwired) device bootstrap arm folds
the sample into the SINGLE `weight_ref` handed to the grow
(`session.rs:1636-1645`), so under sampling both channels would be wrong.

## Specification references

- `WR01-S1` — the split histogram consumes `(der1·s, weight·s)`.
- `WR01-S2` — the leaf estimate consumes the unsampled `(der1, weight)`.
- `WR01-S3` — an empty sample is byte-identical to today (D-04).
- `WR01-S10` — the sampled channels satisfy the fixed-point `|Σ| < 2^33`
  precondition or produce a typed error.

## Context and evidence

- **The single-pair defect.** `grow_oblivious_tree_resident` passes
  `der1_h.clone(), weight_h.clone()` to `launch_partition_hist2_resident_into`
  (`mod.rs:3836-3852`) AND to `launch_partition_update_into`
  (`mod.rs:3942-3951`), whose channels 0/1 feed
  `cb_compute::calc_average(part_stats[3l], part_stats[3l+1], scaled_l2)`
  (`mod.rs:3963-3970`) `[VERIFIED]`.
- **The CPU reference.** Sample affects the split histogram only; leaves use the
  unsampled `weighted_der1` / `weights` `[VERIFIED: boosting.rs:3723-3731,
  :3738-3770]`.
- **Routing is sample-agnostic.** `launch_partition_split_packed_into` takes
  `der1_h` for routing only (`mod.rs:3912-3925`); keep the UNSAMPLED handle there
  for clarity.
- **Next tree's der1 must stay unsampled.** `launch_der_binary_resident` at
  `mod.rs:3978` recomputes from the updated resident approx — unchanged.
- **The subtraction trick stays valid** because the score pair is used consistently
  at every level (`mod.rs:3861-3869`).
- **The multiply primitive already exists.** `fold_weights_resident` is a generic
  elementwise `f64` product over `crate::kernels::vector_mul_kernel`
  `[VERIFIED: crates/cb-backend/src/kernels/bootstrap_device.rs:417-439]` — usable
  verbatim for `der1 · s` as well as `weight · s`. **No new CubeCL kernel.**
- **The upload helper exists.** `upload_channel_floats`
  `[VERIFIED: crates/cb-backend/src/gpu_runtime/mod.rs:1709]`.
- **Fixed-point range precondition.** `|Σ_bin| < 2^33`, no in-kernel guard, silent
  sign-flip beyond it, and the doc explicitly asks for "a host-side pre-launch
  magnitude estimate (`n · max|der1| · 2^30 < i64::MAX`)"
  `[VERIFIED: kernels.rs:2325-2335, 3884-3893]`. MVS's weight channel
  `w·s = threshold / sqrt(λ + der²)` is the unbounded one (SPEC F-D).
- **Test mount convention in cb-backend**: sibling `*_test.rs` mounted from the
  parent module, e.g. `crates/cb-backend/src/kernels.rs:2941-2946` mounts
  `kernels/bootstrap_device_test.rs`. `gpu_runtime/session_depth_gt1_test.rs` is the
  session-level precedent.

## Files

- Modify: `crates/cb-backend/src/gpu_runtime/mod.rs`
  - `grow_oblivious_tree_resident` (`:3757-3781` signature): add
    `score_der1_h: &Handle`, `score_weight_h: &Handle`.
  - `:3836-3852` fill → score pair. `:3942-3951` part-stats → unsampled pair.
  - `:3912-3925` routing → keep unsampled `der1_h`.
  - Update the function doc to state the channel contract explicitly.
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs`
  - `grow_one` (`:1636-1674`): replace the single `weight_ref` fold with the
    two-handle derivation; keep the existing device-sampler arms compiling but
    routed through the same two-handle call.
  - Add the pre-launch range guard (see step 4).
  - Update the other `grow_oblivious_tree_resident` call site (CodeGraph reports 2
    callers in `session.rs` — find both before editing).
- Create: `crates/cb-backend/src/gpu_runtime/session_host_sample_test.rs`
- Modify: `crates/cb-backend/src/gpu_runtime/mod.rs` — add
  `#[cfg(test)]\nmod session_host_sample_test;` next to the existing
  `#[cfg(test)]\nmod session_depth_gt1_test;` at `mod.rs:741-742`
  `[VERIFIED: mod.rs:735-748]`. Note this directory uses the plain sibling-module
  form, NOT `#[path = "…"]` (the `#[path]` form is the `kernels.rs` convention).

## TDD sequence

### 1. Red

In `session_host_sample_test.rs` (device-gated with
`fn device_backend_active() -> bool { cfg!(any(feature = "rocm", feature = "cuda")) }`,
the `bootstrap_device_test.rs:29-31` pattern, plus the
`Err(CbError::Unsupported(msg)) if msg.contains("Atomic<u64>")` early-return
precedent from `session_depth_gt1_test.rs:660-670`):

- `empty_sample_grow_is_byte_identical` — grow one tree twice on the same session
  state, once through the no-sample path and once with an explicitly empty sample;
  assert `splits`, `leaf_of`, and `part_stats` are equal element-for-element.
  Principal failure reason: the refactor perturbed the `No` path.
- `unit_sample_grow_is_byte_identical` — same, with `s ≡ 1.0`. Principal failure
  reason: the fold path is not an identity at `s = 1`.
- `masked_sample_scores_masked_but_leaves_full` — with a hand-built `s` that zeroes
  a known object subset, assert (a) the chosen level-0 split equals a host-computed
  argmax over the MASKED histogram, and (b) `Σ_leaves part_stats[3l+1] == n` (the
  leaf weight channel is the FULL object count, not the kept count). This one test
  fails distinctly for WR01-S1 (wrong split) versus WR01-S2 (wrong leaf sum).
- `over_range_sample_is_rejected` — with `n = 1000` and one `s[i] = 1e12`, assert
  `Err(CbError::OutOfRange)` and that no kernel was launched (assert by the error
  being returned before any handle is created — structure the guard as the first
  statement after the length check).
- `non_finite_sample_is_rejected` — `s[i] = f64::NAN` → `Err(CbError::OutOfRange)`.

- Run: `cargo test -p cb-backend --no-default-features --features rocm --lib host_sample -- --nocapture --test-threads 1`

### 2. Green

Minimum implementation, in this order:

1. Widen `grow_oblivious_tree_resident` with the two extra handle parameters and
   route them: fill ← score pair, part-stats ← unsampled pair, routing ← unsampled.
2. In `grow_one`, when no sample is present, pass `der1_h` / `weight_h` for BOTH
   pairs (no extra launch, no allocation) — this is the D-04 identity path.
3. When a sample handle is present, derive
   `score_weight_h = fold_weights_resident(&client, &self.weight_h, &sample_h, n)?`
   and `score_der1_h = fold_weights_resident(&client, &der1_h, &sample_h, n)?`.
4. Add `fn check_sample_fixedpoint_range(sample: &[f64], max_abs_der1: f64, n: usize) -> CbResult<()>`
   as a private host helper in `session.rs` with its own doc comment deriving the
   bound from `REDUCE_FIXEDPOINT_SCALE_F64`. Reject when any `s[i]` is non-finite or
   negative, or when `n as f64 * max|s| * max(1.0, max_abs_der1) >= 2f64.powi(33)`.
   Emit `CbError::OutOfRange` naming the channel and both numbers.
   **Placement note:** the host `max|der1|` is cheaply available in `cb-train`
   (which owns `approx`) but NOT in the session (der1 is device-resident and must
   not be read back — D-05). Therefore compute the conservative bound from the
   sample alone plus a `max_abs_der1` value the session already holds or that
   `cb-train` supplies through the seam. **Decide in this task and record the
   decision**: preferred is a session-side bound using `max|s|` and a
   `max_abs_der1` passed alongside the sample (a scalar, no extra crossing), since
   it keeps the guard next to the launch it protects. If that requires widening the
   seam beyond `sample`, defer the scalar to TASK-04 and land the sample-only half
   of the guard here (non-finite / negative / `n · max|s| ≥ 2^33`).
5. Do NOT delete `BootstrapState` / `MvsState`; route their `sample_h` through the
   same two-handle derivation so their self-oracles stay green.
6. Replace the stale WR-01 sentence in `grow_one`'s comment block
   (`session.rs:1546-1552`) with an accurate statement.

- Run: the same command, plus
  `cargo test -p cb-backend --no-default-features --features rocm --lib bootstrap -- --test-threads 1`
  and `--lib mvs`.

### 3. Refactor

- Extract the "pick the score pair" decision into one small private fn returning
  `(Handle, Handle)` so `grow_one` has a single branch instead of two nested
  `Option` matches.
- Keep `#[allow(clippy::too_many_arguments)]` on `grow_oblivious_tree_resident`
  (already present at `mod.rs:3756`) rather than inventing a parameter struct — a
  struct would churn every call site for no behavioural gain.
- Run: the three `cb-backend` commands above.

### 4. Verify

- Run: `cargo test -p cb-backend --no-default-features --features rocm --lib -- --test-threads 1`
  (whole cb-backend lib under rocm — this DOES build, unlike the cb-train blanket)
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_oblivious_parity_test -- --nocapture`
  (the TASK-01 gate — must report the SAME numbers as before this task)
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_fit_test`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_region_fit_test`
- Run: `cargo test -p cb-train` (CPU suite; expect the one known pre-existing red)
- Run: `cargo clippy -p cb-backend --no-default-features --features rocm --lib --no-deps`
- Confirm: `git diff` shows no change to any `#[cube]` kernel body.

## Implementation steps

1. Locate BOTH `grow_oblivious_tree_resident` call sites in `session.rs`
   (`codegraph_explore "grow_oblivious_tree_resident"` reports 2).
2. Widen the signature; update both call sites to pass the unsampled pair twice.
   Run the full verify set — this intermediate state must be byte-identical.
3. Re-route fill/part-stats/routing per WR01-S1/S2. Re-run verify.
4. Add the two `fold_weights_resident` derivations behind the sample `Option`.
5. Add the range guard + its two tests.
6. Update all three doc comments (`grow_oblivious_tree_resident`, `grow_one`, the
   guard helper).

## Completion criteria

- [ ] Red tests failed for their stated single reasons before the fix.
- [ ] `Σ_leaves part_stats[3l+1] == n` under a masking sample (WR01-S2).
- [ ] The masked-histogram argmax matches a host reference (WR01-S1).
- [ ] Empty and `s ≡ 1.0` samples are byte-identical to the pre-change grow
      (WR01-S3), proven element-for-element on `splits`/`leaf_of`/`part_stats`.
- [ ] Over-range and non-finite samples produce `CbError::OutOfRange` with no launch.
- [ ] `device_oblivious_parity_test`, `device_nonsym_fit_test`,
      `device_region_fit_test`, and the `bootstrap`/`mvs` self-oracles all green with
      unchanged values.
- [ ] No new `#[cube]` kernel was written.

## Risks and guardrails

- **R1/R2 (wrong channel)** — mitigated by landing this BEFORE the gate opens, so a
  mistake cannot silently ship a wrong model. The masked-sample test is the specific
  guard.
- **Byte-identity regression on the `No` path** — mitigated by the two-step
  implementation (widen with identical handles first, verify, then re-route).
- **R11 fixed-point overflow** — the guard. Note it must be *conservative* (reject
  some safe inputs) rather than optimistic; document that trade-off.
- **Reading der1 back to compute `max|der1|` would violate D-05** (no `n`-length
  read-back per tree). Guard: the guard must use only host-side data or an O(1)
  scalar; never `client.read_one` on `der1_h`.
- **CubeCL** — if a build error appears, STOP and read
  `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md`
  before any fix. None is expected: no kernel changes.
