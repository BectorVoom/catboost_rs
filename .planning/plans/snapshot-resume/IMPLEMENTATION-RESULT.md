---
title: "ORCH-03 — Training snapshot / resume (slice 1): implementation result"
parent: ./PLAN.md
spec: ./SPEC.md
status: implemented
implemented_at: 2026-08-03
---

# ORCH-03 slice 1 — implemented

All eight tasks executed under strict TDD (a recorded Red before every Green).
The gating spike returned **GO**; its findings and the three plan amendments it
forced are in `./TASK-01-findings.md`.

## Task-by-task

| Task | Spec | Landed as | Evidence |
|---|---|---|---|
| TASK-01 | ORCH-03-S1 (audit) | `./TASK-01-findings.md` | GO + 3 amendments; every plan line anchor was stale (`train_inner` `:2259`→`:2896`) |
| TASK-02 | ORCH-03-S2 | `cb-core/src/rng.rs` `TFastRng64::from_raw_state` | `cargo test -p cb-core`: 30 passed |
| TASK-03 | ORCH-03-S1 | trailing `snapshot` param + `None` at all 4 `train_inner` call sites | full `cb-train` suite unchanged |
| TASK-04 | ORCH-03-S3 | `snapshot.rs` DTOs + `encode`/`decode` + Cargo dep fix | round-trip, bad-version, garbage, non-finite, DTO-refusal tests |
| TASK-05 | ORCH-03-S4 | `fingerprint` (FNV-1a, 16 documented fields) + `check_resume` | 23-row sensitivity table, all green |
| TASK-06 | ORCH-03-S5 | `snapshot_scope_ok` + per-tree write hook + `write_atomic` | `snapshot_write_oracle_test`: 6 passed |
| TASK-07 | ORCH-03-S6 | pre-loop resume block + `resume_from..iterations` bound | `snapshot_resume_selfconsistency_oracle_test`: 5 passed |
| TASK-08 | ORCH-03-S7 | `train_with_snapshot -> (Model, usize)`, peek-based resume point | same suite; `resumed_from` asserted 0 / K / N |

## The keystone result

`resume_equals_straight_through` passes for `(N,K)` ∈ {(4,1), (4,2), (6,3)}, with
**bit-identical** trees — split structure, leaf values and leaf weights compared as
`u64` bit patterns, not approximately.

The suite also carries an anti-false-pass test
(`a_resume_genuinely_starts_at_k_rather_than_retraining`): equality alone cannot
prove a resume happened, because a run that silently retrained from 0 would produce
the same model. That test deliberately CORRUPTS the checkpoint's first tree and
asserts the corruption survives into the final model — which it can only do if the
loop genuinely skipped the first K iterations.

## Deviations from the plan (each with its reason)

1. **`staged_out` added to the scope guard** (TASK-01 AMENDMENT 1, user-confirmed).
   `train_inner`'s `mut staged_out: Option<&mut Vec<f64>>` parameter is extended
   once per iteration at `boosting.rs:5962` — loop-carried mutable state on the
   supported path that the plan's `{approx, trees, rng}` set missed because it is a
   PARAMETER, not a local. A resumed run given `Some(out)` would emit `N-K` staged
   rows instead of `N`. Refused in the guard; `train_with_snapshot` passes `None`,
   so the shipped entry never trips it.

2. **The guard keys off `device_active`, not `device_host_eligible`**
   (AMENDMENT 3). The plan's predicate list named the latter, which is TRUE for a
   float-only SymmetricTree CPU fit — using it would have rejected exactly the
   regime the slice supports.

3. **`prev_leaf_mean_l2` reclassified** (AMENDMENT 2). The plan called it
   "written only inside an out-of-scope branch"; it is in fact written
   UNCONDITIONALLY at `:5988` and only READ out of scope (by `bootstrap`, which
   ignores it under `EBootstrapType::No`). It still needs no snapshotting, but the
   reasoning is read-gating, not write-gating — the distinction matters if the
   regime is ever widened to MVS.

4. **`CbError::Snapshot(String)` added** rather than reusing
   `CbError::Degenerate`. "degenerate training input" is not what a corrupt
   checkpoint or a fingerprint mismatch is, and the enum's own docs sanction new
   variants. No exhaustive `match` on `CbError` exists anywhere in the workspace
   (it is wrapped via `#[from]`), so the addition is non-breaking.

5. **Floats are serialized as BITS, not JSON decimals** — the one finding that
   changed a correctness outcome. See below.

6. **The `Loss` match in `fingerprint` is EXHAUSTIVE** (no `_` arm). The plan's
   sketch ended with `_ => feed_disc(...)`, and its own prose noted that this
   silent-corruption shape had already been caught three times across three review
   passes. An exhaustive match makes a newly added `Loss` variant a COMPILE ERROR
   here instead of a fourth instance. Same treatment for `EBoostingType`,
   `LeafMethod` and `EScoreFunction` tags.

7. **`bias` is VERIFIED against the checkpoint, not restored from it.** It is a
   pure function of fingerprinted inputs, so a disagreement means the fingerprint
   missed something — a bug, reported loudly, rather than papered over.

## The bit-exactness finding (TASK-07)

The keystone oracle first failed with a **one-ULP** difference in tree 0's leaf
values. Tree 0 comes from the checkpoint, so the natural suspect was the plan's
`refingerprinted_k_run` fixture assumption (that an `iterations=K` run's first K
trees equal an `iterations=N` run's).

A direct probe **disproved that**: `train(iterations=1)` and `train(iterations=4)`
produce bit-identical tree 0 — the plan's assumption is sound. The culprit was
`serde_json`: its `f64` decimal round-trip is **not bit-exact**.

`TrainSnapshot` therefore stores every float as its IEEE-754 bit pattern via
`#[serde(with = "f64_bits")]` / `f64_bits_vec`. The field types stay `f64`, so no
call site changed. `snapshot_roundtrip_is_bit_exact_for_a_value_json_decimals_perturb`
pins it using the exact offending bit patterns.

## Regression gate

* `cargo test -p cb-train --no-fail-fast` — the only failure is
  `monotone_oracle_test::monotone_non_symmetric_and_region_are_typed_errors`, the
  known-accepted pre-existing failure recorded in the F23 workspace baseline and in
  project memory. `git log` on that file shows it untouched by this work.
* `cargo clippy -p cb-train --lib --no-deps` and `-p cb-core` — zero errors, and
  zero findings referencing `snapshot.rs`.
* `cargo build -p cb-train` (the non-test LIBRARY build, PLAN-CHECK's CRITICAL-6
  requirement) — green, proving `serde`/`serde_json` really are normal
  dependencies and not left under `[dev-dependencies]`.
* `cargo build -p cb-model` — green (downstream sanity).

## Not done (recorded, out of slice 1)

* Everything the scope guard refuses: ranking, eval sets, categorical/CTR, ordered
  boosting, sampling, multi-dimension, penalties, non-symmetric grow policies,
  device training, staged buffers, custom objectives/metrics.
* `auto_learning_rate = true` resume self-consistency (fingerprint field 13 rejects
  an auto-LR change, so it is refused rather than silently wrong).
* Python-side `save_snapshot` / `snapshot_file` / `snapshot_interval` parameters —
  they remain in `catboost-rs-py`'s known-not-yet list. Exposing them is a facade
  slice, not this one.
