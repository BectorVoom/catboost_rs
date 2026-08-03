---
title: "ORCH-03 TASK-01 — Gating verification spike: findings and GO/NO-GO"
parent: ./PLAN.md
spec: ./SPEC.md
task: TASK-01
verdict: GO (with three mandatory plan amendments)
audited_at: 2026-08-03
audited_against: crates/cb-train/src/boosting.rs @ 4c28903 (6238 lines)
---

# TASK-01 — GATING VERIFICATION SPIKE: **GO**

Read-only audit, re-run from scratch against the CURRENT source. **Every line
anchor in `PLAN.md` is stale** — the plan was written 2026-07-19 against a
`boosting.rs` in which `train_inner` began at `:2259`; it now begins at **`:2896`**
and the function spans **`:2896`–`:6230`** (the file grew from ~4757 to 6238 lines
across the one-hot, CTR-engine and facade-routing waves). The audit below supplies
the current anchors. Downstream tasks must use THESE, not the plan's.

## Item 1 — Loop-carried mutable state on the scoped path

The per-tree loop is `for iter in 0..params.iterations` at **`:4127`**, closing at
**`:6091`**. Every `let mut` binding declared before it, classified:

| Local | Line | Scoped-path status |
|---|---|---|
| `approx` | 4127→`:3261` | **LOOP-CARRIED — snapshot** |
| `trees` | `:3666` | **LOOP-CARRIED — snapshot** |
| `rng` | `:3725` | **LOOP-CARRIED — snapshot** (inert here; see item 2) |
| `group_spans` | `:3045` | inert — `is_grouped_loss(&params.loss)` (ranking) |
| `yetirank_seeder` | `:3122` | inert — `is_yetirank` |
| `stochasticrank_seeder` | `:3155` | inert — `is_stochasticrank` |
| `learn_approx` | `:3286` | inert — `if is_yetirank { … } else { Vec::new() }` |
| `used_features` | `:3395` | inert — penalties |
| `non_symmetric_trees` | `:3669` | inert — pushed only at `:6043`, `grow_policy` leaf-wise |
| `region_trees` | `:3673` | inert — pushed only at `:6022`, `grow_policy=Region` |
| `detector` / `best_model` / `eval_approx` | `:3693`/`:3695`/`:3700` | inert — whole block gated `if has_test` (`:6057`) |
| `prev_leaf_mean_l2` | `:3772` | **written unconditionally — see AMENDMENT 2** |
| `fold_approxes` | `:4073` | inert — `Vec::new()` when `structure_fold_columns.is_empty()` |
| `avg_exp_approx` | `:4105` | inert — `Vec::new()` unless `exp_ctr` (needs non-empty `structure_fold_columns`) |
| `used_ctr_splits` | `:4125` | inert — pushed only from `grown.ctr_splits` (`:5254-5258`), empty with no CTR candidates |

`ordered_learning_perm` (`:3305`) is **no longer `let mut`** — it is now an
immutable `let`, so it has dropped out of the mutable set entirely.

The last five rows did not exist when the plan was written. All five are provably
inert on the scoped path, so **the snapshot state set is unchanged: `{approx,
trees, rng}`** — plus the amendment below.

No interior mutability anywhere in `train_inner`: `RefCell`/`Cell`/`Atomic*`/
`Rc`/`Arc`/`static mut` all return zero hits over `:2896`–`:6230`.

## Item 2 — `draws_active` is FALSE on the scoped path

`:3731-3732`:
```rust
let perturb_active = params.random_strength != 0.0;
let draws_active = !matches!(params.bootstrap_type, EBootstrapType::No) || perturb_active;
```
`bootstrap_type=No` + `random_strength=0.0` ⇒ both disjuncts false. The three RNG
consumers are each gated on it (`:4887` PRE draws, `:5038` `perturb`, `:5979` POST
draws), so `rng` is never advanced across the loop. Restore is a no-op for the
exact oracle but is still captured for forward-compat, as the SPEC intends.

## Item 3 — `device_active` is FALSE on the default CPU backend

`:4011` `device_active = device_host_eligible && device_n_bins > 0 && runtime.begin_device_training(…)?`.
The `Runtime` trait's DEFAULT `begin_device_training` returns `Ok(false)`
(`crates/cb-compute/src/runtime.rs:1319-1336` — it `let _ = (…)` every argument and
returns `Ok(false)`). So on `CpuBackend` the device `continue` branches (`:4382`,
`:4543`, `:4642`) are never taken, and the write hook belongs at the **end of the
CPU iteration body**.

## Item 4 — Injection points (current anchors)

- **Pre-loop resume block:** immediately before `:4127`, after `used_ctr_splits`
  (`:4125`) — the last once-only setup. `trees`/`approx`/`rng` are all in scope and
  fully initialized there.
- **Loop bound:** `:4127` `for iter in 0..params.iterations` → `resume_from..params.iterations`.
- **Per-tree write hook:** the end of the iteration body, after the `if has_test`
  block closes (`:6090`) and before the loop's `}` at `:6091`. At that point the
  tree is pushed (`:6034`), `approx` is updated, `staged_out` is extended (`:5962`)
  and `prev_leaf_mean_l2` is set (`:5988`) — i.e. iteration `k` is complete.
- **Model assembly:** `:6216`-ish (`Ok((Model { … }, baked))`), unchanged in shape.

## Item 5 — The four callers construct `train_inner(...)` positionally

`:2442` (`train_with_eval_sets`), `:2482`, `:2535` (`train_cat`), `:2587`. Each
passes the same 11 positional arguments ending
`… staged_out, eval_sets, history, RankingData::default()`. A new TRAILING param is
a one-line `None` addition per call site — mechanical, as planned.

## Item 6 — Scope-guard predicate set

The negation of the scoped regime, from gate expressions already in the source:
`is_grouped_loss(&params.loss)`, non-empty `cat_columns` / `ctr_candidates`,
`boosting_type != Plain`, `!eval_sets.is_empty()`, `bootstrap_type != No`,
`random_strength != 0.0`, `approx_dimension != 1`, `penalties_active`,
`grow_policy != SymmetricTree`, **`device_active`** (see AMENDMENT 3), plus the two
CRITICAL-3 predicates `matches!(params.loss, Loss::Custom(_))` and
`matches!(params.eval_metric, Some(EvalMetric::Custom(_)))`, plus **`staged_out.is_some()`**
(AMENDMENT 1).

---

# Verdict: **GO**

Items 1–6 all confirmed; no additional loop-carried mutable *local* on the scoped
path. Three amendments are mandatory for TASK-03/06/07 and are NOT re-litigations
of a locked design decision — each is a fact about the current source that the plan
could not have known.

### AMENDMENT 1 — `staged_out` is loop-carried external mutable state (NEW)

`train_inner`'s own parameter `mut staged_out: Option<&mut Vec<f64>>` (**`:2904`**)
is extended once per iteration at **`:5962`**:
```rust
if let Some(out) = staged_out.as_deref_mut() {
    out.extend_from_slice(&approx);
}
```
This is loop-carried mutable state on the **scoped** path — it is gated on nothing
but `Some`. The plan's state set (`{approx, trees, rng}`) covers only *locals* and
misses it. A resumed run given `Some(out)` would emit `N-K` staged rows where the
straight-through run emits `N`, silently.

**Required disposition:** add `staged_out.is_none()` to TASK-06's scope guard —
snapshotting is refused when a staged buffer is requested. This costs nothing: the
snapshot regime is a fresh-fit/resume entry (TASK-08's `train_with_snapshot`), which
passes `None`. Serializing the staged buffer instead is a strictly larger change and
is NOT in the SPEC's scope.

(`history: Option<&mut EvalMetricHistory>` is the same *shape* of out-param but is
written only inside `if has_test` (`:6076`), so `eval_sets.is_empty()` already
covers it.)

### AMENDMENT 2 — `prev_leaf_mean_l2` is written UNCONDITIONALLY

The plan classifies it as "written-only-inside-an-out-of-scope branch". That is no
longer true: **`:5988`** `prev_leaf_mean_l2 = Some(last_iter_mean_leaf_value(&leaf_values));`
executes every CPU iteration, ungated. It is only ever READ as an argument to
`bootstrap(…)` (**`:4927`**), which ignores it under `EBootstrapType::No`.

**Required disposition:** classification changes from "inert" to **"written
unconditionally, read only out-of-scope"**. It still does NOT need snapshotting
(`bootstrap_type=No` is a scope-guard predicate, so the value can never be read on
the resumable path), but the reasoning must be recorded as *read*-gating, not
*write*-gating — otherwise a future widening of the regime to MVS silently
resurrects it as unsnapshotted state.

### AMENDMENT 3 — The guard must key off `device_active`, NOT `device_host_eligible`

The plan's item-6 predicate list names `device_host_eligible` (**`:3802`**). Using it
as a rejection predicate would be **backwards**: it is TRUE for a float-only
SymmetricTree CPU fit — i.e. it would reject exactly the regime the slice supports.
The correct predicate is `device_active` (**`:4011`**), which additionally requires
`device_n_bins > 0` and a backend that actually opened a session. It is computed
before the loop, so the guard can read it at the injection point.

---

## Follow-on effect on the plan

- TASK-03/06/07 must re-anchor to the line numbers in this document.
- TASK-06's scope guard gains `staged_out.is_none()` and swaps
  `device_host_eligible` → `device_active`.
- The SPEC §1 state table gains the `prev_leaf_mean_l2` write/read distinction.
- No NO-GO condition was met: the injection points are cleanly additive and no
  unaccounted loop-carried local exists on the scoped path.

## TASK-02 status (audited in parallel, per the plan's Wave A)

`TFastRng64::from_raw_state([u64;4], u64)` implemented at
`crates/cb-core/src/rng.rs:225-247`; round-trip + anti-false-pass tests at
`crates/cb-core/src/rng_test.rs`. `cargo test -p cb-core`: **30 passed, 0 failed**;
`cargo clippy -p cb-core --lib --no-deps` clean.
