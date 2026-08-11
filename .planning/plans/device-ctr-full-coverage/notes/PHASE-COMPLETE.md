# Device CTR Coverage P1 — PHASE COMPLETION SUMMARY

Written by **T24**, the phase's tail task, against `PLAN.md` §7 (*Phase definition of done*) and
`SPEC.md` §6 (*Acceptance scenarios*).

- **Branch**: `feat/device-ctr-full-coverage`, primary checkout
  `/home/user/Documents/workspace/catboost_rs` (`planning/settings.json` —
  no leading dot — `implementation.use_worktree: false`).
- **Base of the phase**: `a0a67ec` (23/23 device PASS, clean tree).
- **State proven here**: committed `HEAD cbafe67` (wave 7) **plus** T23's and T24's uncommitted
  trees. No git write command was run by any P1 task; the coordinator commits.
- **Device**: real ROCm `gfx1151`, ROCm at `/home/user/rocm/opt/rocm`.

**Verdict: the phase definition of done is MET, with three explicitly OPEN findings that are
recorded rather than closed** (R-20, `T22-OBS-1`, `T22-OBS-2`) and a list of pre-existing failures
that are **not** phase regressions. Nothing below is presented as "no open findings".

> **POST-PHASE UPDATE (2026-08-11): all three are now CLOSED.** R-20 was closed by
> `device_ctr_eligible_max_diff_test` (see `R20-CLOSURE.md`). The user then lifted the
> RECORD-ONLY ruling on the other two and both were implemented:
>
> * **`T22-OBS-1` was a REAL device defect and is FIXED.** Not the `fused_unit_fold` correlate
>   §9.1 localised — that branch is innocent (the device `leaf_of` and the host walk agree
>   exactly on a CTR-free tree, `mism = 0/64`). The cause was one gate in
>   `gpu_runtime/session.rs`: the averaging-permutation leaf-value gather was conditioned on
>   *"this tree chose ≥1 CTR split"*, so a CTR fit's CTR-free tree returned the RESIDENT
>   learning-fold leaf estimate instead of the main/averaging one. Gate removed; every tree now
>   agrees to ≤1.4e-17. Detector: `crates/cb-train/tests/device_ctr_free_tree_leaf_test.rs`
>   (30 iterations, with a hard guard that the horizon actually reaches a CTR-free tree).
> * **`T22-OBS-2`'s remedy is implemented**:
>   `crates/cb-train/tests/device_ctr_buckets_long_horizon_diff_test.rs` makes the long-horizon
>   (20-iteration) Buckets differential over a partition-invariant projection. At prior 0.25 the
>   raw identity diverges on exactly tree 12 — T22's reported tie — while the projection holds
>   on all 20 trees; a prior-0.5 arm ships alongside it.
>
> Roster **29 → 31** binaries, **32 PASS / 0 FAIL**. Full detail, including both mutation runs
> and the `(a+v)−a` recovery trap, is the final entry of `COORDINATOR-FINDINGS.md`. §9.1 and
> §9.2 below are left as written — they are the contemporaneous record of the finding, not of
> the fix.

---

## 1. Headline

| item | result |
|---|---|
| tasks | **25** (`T00`…`T24`), all green |
| `bash ./run_device_tests.sh` | **28/28 PASS + perf lane PASS**, 0 FAIL, exit 0 |
| roster arithmetic | 23 registered at `a0a67ec` **+ 5 created by this phase = 28**, derived from the array, not aimed at |
| device CTR types on the device | `{Borders, Buckets, BinarizedTargetMeanValue, Counter}` — the **complete CPU-legal set** (`restrictions.h:18-48`) |
| device CTR projections | simple **and** ≥2-member combinations (`max_ctr_complexity = 2`) |
| the gate | one expression: `ECtrType::from_i8(col.ctr_type).is_some_and(|t| t.is_cpu_supported())` |
| fixtures | 3 new frozen directories, **zero** existing fixture files regenerated |

---

## 2. Acceptance scenarios (`SPEC.md` §6)

| # | scenario | status | evidence |
|---|---|---|---|
| 1 | Buckets CTR fit commits and matches upstream ≤1e-5 | **PASS** | `device_ctr_buckets_fit_test`, `max \|Δpred\| = 2.776e-17`, `grown = 5` |
| 2 | Counter CTR fit commits and matches ≤1e-5 | **PASS** | `device_ctr_counter_fit_test`, `1.388e-17`, `grown = 5` |
| 3 | BTMV CTR fit commits and matches ≤1e-5 **against the corrected CPU** | **PASS** | `device_ctr_btmv_fit_test`, `2.776e-17`, `grown = 5`; Track E (DCTR-04) landed at T04 **before** Track B |
| 4 | Combination CTR fit (`max_ctr_complexity = 2`) commits and matches ≤1e-5 | **PASS** | `device_ctr_combo_fit_test`, `2.082e-17`, `grown = 5`, 8 CTR splits of which **3 are ≥2-member** |
| 5 | gate has no type / arity / target-border / prior conjunct | **PASS** | `boosting.rs:2556-2580` is a single `matches!`-free delegation; `boosting_ctr_gate_tests` `13 passed` incl. the comment-stripped structural pin |
| 6 | every CTR e2e asserts device commitment via `CountingGpu` | **PASS** | 8 CTR device test files, each with a `grown.get()` assertion; measured re-run: `grown == iterations` on all five e2es |
| 7 | `run_device_tests.sh` green at its **grown** roster, every new binary registered | **PASS** | 28/28 + perf lane; registration audit in `notes/T24.md` §2 |
| 8 | `ctr_btmv_simple_oracle_test` passes **unchanged** | **PASS** | `4 passed`; `--test-threads=1` output byte-identical to T03's pre-T04 baseline modulo the wall-clock suffix |
| 9 | one-hot×CTR, multi-permutation, eval-set, cat-only, `border_count != 15` all still decline, each with a passing negative test | **PASS** | T13's 2×2 square + T21's three pins and control arm: `device_ctr_type_gate_test` `7 passed`, `device_ctr_gate_test` `2 passed`, `device_fpp_composition_test` `6 passed` |
| 10 | combination × non-Borders detector passes **before** the final gate lands | **PASS** | T22 (`cbafe67`, wave 7) landed **before** T23 (wave 8); `device_ctr_combo_types_diff_test` `3 passed`, re-run green under the final gate |

---

## 3. §7 command checklist — item by item

All run on the real ROCm device or under default cpu features as marked, at the T24 tree.

| # | command | result |
|---|---|---|
| 1 | `cargo test --workspace` | **FAILS at one target — the known pre-existing one.** `cb-backend --lib` under **default (cpu)** features: `228 passed; 59 failed; 2 ignored` in 358.81 s, all 59 in `kernels::*`, zero in `gpu_runtime::*`; cause `error: operation with block successors must terminate its parent block` / `plane_inclusive_sum … not supported on CPU` in `cubecl-cpu-0.10.0`. Cargo stops there (exit 101). §6.1. |
| 1b | `cargo test --workspace --exclude cb-backend --no-fail-fast` (the accounting run for everything #1 did not reach, because cargo stops at the first failing target) | **PASS — 196 targets, 0 FAILED, 1665 tests passed, exit 0**. So the whole workspace is accounted for: every crate is green except the one pre-existing `cb-backend --lib` cpu target. |
| 2 | `cargo test -p cb-train --lib` | **PASS — `401 passed; 0 failed`** (the CRITICAL-1 regression, green in full) |
| 3 | `cargo test -p cb-backend --no-default-features --features rocm` | **PASS** — lib `277 passed; 0 failed; 2 ignored`, + `4 passed`, + `1 passed`, + doc-tests `0` |
| 4 | `bash ./run_device_tests.sh` | **PASS — 28/28 + perf lane**, `FAIL` count 0 |
| 5 | `cargo test -p cb-oracle --test ctr_device_buckets_fixture_smoke_test` | **PASS** — `1 passed` |
| 6 | `cargo test -p cb-oracle --test ctr_device_counter_fixture_smoke_test` | **PASS** — `1 passed` |
| 7 | `cargo test -p cb-oracle --test ctr_device_btmv_fixture_smoke_test` | **PASS** — `1 passed` |
| 8 | `cargo test -p cb-train --lib boosting_ctr_gate_tests` | **PASS** — `13 passed` |
| 9 | `cargo test -p cb-train --lib device_ctr_combo_config_tests` | **PASS** — `8 passed`; the gate-state table at its final, retired form |
| 10 | `cargo test -p cb-train --lib ctr::calc_ctr_test` | **PASS** — `9 passed` |
| 11 | `cargo test -p cb-train --lib ctr::ctr_feature_test` | **PASS** — `1 passed` |
| 12 | `cargo test -p cb-train --test ctr_btmv_simple_oracle_test` | **PASS** — `4 passed`, byte-identical to T03's baseline |
| 13 | `cargo test -p cb-train --test ctr_btmv_bake_upstream_table_test` | **PASS** — `2 passed` |
| 14 | `cargo test -p cb-train --test ctr_buckets_simple_oracle_test` | **PASS** — `2 passed` |
| 15 | `cargo test -p cb-train --test ctr_counter_simple_oracle_test` | **PASS** — `4 passed` |
| 16 | `cargo test -p cb-train --test ctr_counter_full_eval_oracle_test` | **PASS** — `4 passed` |
| 17 | `cargo test -p cb-train --test ctr_borders_multiprior_oracle_test` | **PASS** — `2 passed` |
| 18 | `cargo test -p cb-train --test ctr_mixed_simple_vs_combo_oracle_test` | **PASS** — `2 passed` |
| 19 | `cargo test -p cb-train --test tensor_ctr_e2e_oracle_test` | **PASS** — `3 passed` |
| 20 | `cargo test -p cb-train --test plain_ctr_oracle_test` | **PASS** — `3 passed` |
| 21 | `cargo test -p cb-train --test ordered_ctr_oracle_test` | **PASS** — `3 passed` |
| 22 | `cargo test -p cb-train --test ctr_feature_materialize_test` | **PASS** — `6 passed` |

### §7's narrative requirements

| requirement | status |
|---|---|
| `ctr_types_are_device_covered` contains **no** type, arity, target-border or prior conjunct and delegates to `ECtrType::from_i8` / `is_cpu_supported` | **PASS** — verified by reading the current source (`boosting.rs:2556-2580`) and by `the_gate_delegates_to_the_cpu_supported_partition`, hardened to a comment-stripped exact-count assertion after MUT-B showed the raw `contains` form was satisfied by the gate's own prose |
| **every** CTR e2e asserts `grown.get() == params.iterations` via `CountingGpu` | **PASS** — §4's table |
| `cargo test -p cb-train --lib` green **in full** | **PASS** — `401 passed; 0 failed` |
| T22's DCTR-20 differential green for all three non-Borders types, landed **before** T23 | **PASS** — `cbafe67` precedes T23's tree; re-run green under the final gate |
| `run_device_tests.sh`'s `TESTS=(…)` lists **every** device binary this phase created; the count is derived | **PASS** — audit in `notes/T24.md` §2 |
| `ctr_btmv_simple_oracle_test` unchanged; **zero** fixtures regenerated | **PASS** — §7 below |
| the summary records each e2e's `max \|Δpred\|`, R-20's measured status, T05's rung, T10's leaf-gather finding, and every mutation message | **this document**, §4 / §5 / §8 |

---

## 4. Every device e2e's measured `max |Δpred|` (re-measured at the T24 tree)

The quintet has been **unmoved since it was first measured**, task after task, which is D-04's
whole point. Re-measured here with `-- --nocapture`:

| e2e | fixture | CTR content | `max \|Δpred\|` (bar 1e-5) | `grown` |
|---|---|---|---|---|
| `device_ctr_fit_test` | `ctr_device_mixed` | 1 CTR split (Borders) | **4.483e-11** | **5 / 5** |
| `device_ctr_buckets_fit_test` | `ctr_device_buckets` | 8 Buckets splits, `target_border_idx` all `0` | **2.776e-17** | **5 / 5** |
| `device_ctr_counter_fit_test` | `ctr_device_counter` | 5 Counter splits | **1.388e-17** | **5 / 5** |
| `device_ctr_btmv_fit_test` | `ctr_device_btmv` | 5 BTMV splits, max bucket occupancy 15 | **2.776e-17** | **5 / 5** |
| `device_ctr_combo_fit_test` | `ctr_device_combo` | 8 CTR splits, **3 ≥2-member combinations** | **2.082e-17** | **5 / 5** |

`grown == iterations` on all five. PLAN §6 assumption 5's *expected* `≈2.082e-17` (an expectation
carried over from a reverted spike) is now a **measurement on the landed code**.

The DCTR-20 differential is a device-vs-CPU comparison, so it reports `max |Δleaf|` rather than a
`|Δpred|` against upstream:

| `device_ctr_combo_types_diff_test` arm | config | splits (device / cpu) | result |
|---|---|---|---|
| buckets | `Prior = 0.25`, 5 iters, depth 2 | 8 (3 ≥2-member) / 8 (3) | split sequences **IDENTICAL**, `max \|Δleaf\| = 8.674e-18` (bar 1e-4), `grown 5`, cpu-arm device-grows 0 |
| counter | `Prior = 0.5`, 20 iters, depth 2 | 24 (4) / 24 (4) | **IDENTICAL**, `2.429e-17`, `grown 20`, cpu-arm 0 |
| btmv | `Prior = 0.5`, 5 iters, depth 2 | 8 (3) / 8 (3) | **IDENTICAL**, `8.674e-18`, `grown 5`, cpu-arm 0 |

### The delta is NOT a device-commitment fingerprint — in either direction

Four independent measurements say so, and this is the phase's most reusable warning:

| fixture | device `\|Δ\|` | CPU-fallback `\|Δ\|` | do they differ? |
|---|---|---|---|
| `ctr_device_mixed` (T20) | `4.483e-11` | `1.388e-17` | yes |
| `ctr_device_buckets` (T10) | `2.776e-17` | `2.776e-17` | **no** |
| `ctr_device_counter` (T12) | `1.388e-17` | `2.776e-17` | yes |
| `ctr_device_btmv` (T16) | `2.776e-17` | `2.776e-17` | **no** |

Only two signals held every single time: **`CountingGpu.grown == iterations`**, and the runtime
(≈1.6–1.9 s device vs ≈0.01 s CPU). Note the direction, from T20: **the CPU path scores BETTER
against the upstream oracle than the device path**, so no ≤1e-5 bar can ever detect a fallback.
T20's rule survives only in its one-way form: *a suspiciously tiny delta is a smell, never
reassurance.*

---

## 5. R-20 — **STILL OPEN**, measured, one hypothesis refuted

`SPEC.md` R-20: *"`ctr_device_combo` does not discriminate D-2"* (the eligibility-filtered
`eligible_max` at pass C's cat-feature-weight call site). The designated closure measurement was
**T22's mutation 1**: revert D-2's call site to the pre-T18 unfiltered
`cs.bucket_counts.iter().copied().max().unwrap_or(1).max(1)` and check whether the device split
**sequence** moves.

**It was run. It does not move. R-20 IS STILL OPEN — no task claimed closure.**

| horizon | what was run | result |
|---|---|---|
| shipped config (buckets 5 iters, counter 20, btmv 5) | MUT-1 live vs reverted | **byte-identical in every printed quantity** — same split counts (8/3, 24/4, 8/3), same `max \|Δleaf\|`, same `grown` |
| deliberately longer horizon (**all three arms at `iterations = 20, depth = 2`** — 40 level decisions per arm, **13** ≥2-member combination splits on the Buckets arm) | MUT-1 live vs reverted | still **byte-identical**. The one arm that fails at that horizon fails **identically with and without the mutation**, so the failure is `T22-OBS-2`, not D-2 |
| earlier, cheaper probe (T19 §3.4) | same reversion against `device_ctr_combo_fit_test` | `2.082e-17`, `grown = 5`, 8 splits / 3 combinations — unchanged |

Three independent negative measurements. What changed across the phase is only the *reason* R-20
stays open: at T18 the filter was the **identity** on every reachable input (the arity conjunct
still stood, so every column reaching pass C had exactly one member); after T19 that excuse is
gone — ≥2-member columns do reach pass C, the filtered and unfiltered `eligible_max` genuinely
**differ at every tree's level 0** on `ctr_device_combo` (`phantom_max == 0` there and a
combination is always ineligible at level 0; weights `(0.756, 0.707)` filtered vs `(0.894, 0.866)`
unfiltered) — **and the greedy winner still never flips.**

**One hypothesis is now REFUTED.** T19 §5 proposed *"a fixture whose combination has a much larger
bucket count than its members' simple projections"* as the plausible discriminator. On
`ctr_device_combo` that ratio is already ~3× (simple 3/4 vs combined ≤12) and nothing moves ⇒ the
ratio alone is not the missing ingredient.

**What D-2's evidence therefore IS**: `gpu_runtime::ctr_eligibility_test`'s unit tests (red-first,
`left: 40, right: 6`; every conjunct mutation-proved) plus the **source-scan** pin
`pass_c_calls_the_filtered_max_and_folds_the_phantom_outside_it`, plus MUT-W's proof that the
helper's *value* is consumed by the production cat-feature weight (`.max(1) → .max(1000)` moves
`device_ctr_fit_test` by `|Δ| = 1.188e-1`). What it is **not** is a behavioural detector for the
*filter* on any committed fixture. The open question is no longer "has anyone run the designated
detector" but **"does any reachable configuration make D-2 observable at all"**. The measured form
of all of this is written into the R-20 comment in `crates/cb-backend/src/gpu_runtime/mod.rs`.

---

## 6. Other things §7 asks to be stated explicitly

### 6.1 The pre-existing failures — stated as such, NOT phase regressions

None of these was introduced by P1; every one was verified pre-existing at `HEAD` by the task that
found it, and none was fixed inside another task (that would be scope creep).

1. **`cargo test --workspace` fails one target: `cb-backend --lib` under DEFAULT (cpu) features.**
   Measured at T24: **`228 passed; 59 failed; 2 ignored`** in 358.81 s. All 59 failures are in
   `kernels::*`; **zero** in `gpu_runtime::*`. Cause: `plane_inclusive_sum … is not supported on
   CPU` in `cubecl-cpu-0.10.0`, surfacing as `error: operation with block successors must
   terminate its parent block`. T02 proved pre-existence by restoring `runtime.rs` from
   `git show HEAD:` read-only and reproducing the identical failures.
   **The count is a RANGE, 59–60**, and that is a real flake, not sloppy bookkeeping:
   `kernels::exact_quantile_test::exact_quantile_weighted_matches_cpu` fails in some runs and
   passes in others on untouched code (T17 saw it fail three consecutive isolated runs then pass a
   fourth). In T24's run it **passed**, which is exactly why this run reads 59. Prior
   observations: T02 `211/60`, T16 `222/60`, T17 `225/59`, T18 `227/60`, T23 `227/60`, T24
   `228/59` — the *passed* count grows only because the suite grew.
   **The rest of the workspace is green and that was measured, not assumed**:
   `cargo test --workspace --exclude cb-backend --no-fail-fast` ⇒ **196 targets, 0 FAILED, 1665
   tests passed, exit 0** (`catboost-rs`, `catboost-rs-py`, `cb-compute`, `cb-core`, `cb-data`,
   `cb-model`, `cb-oracle`, `cb-train` — 179 integration targets + 9 unittest targets + doc-tests).
2. **clippy `erasing_op`** — `error: this operation will always return zero` at
   `crates/cb-backend/src/kernels/score_split.rs:374` (`cindex[0 * n + obj] = bin;`) on
   `cargo clippy -p cb-backend --no-default-features --features rocm --lib --tests`. The **only**
   `error` on that lane; warning count 18, unchanged since T17. Verified pre-existing at `HEAD`.
3. **clippy `duplicated_attributes`** — a doubled `#[allow(clippy::too_many_arguments)]` at
   `crates/cb-backend/src/gpu_runtime/mod.rs:4542` and `:4574` under default cpu features
   (verified still at those exact lines at the T24 tree).
4. **clippy `type_complexity`** — a **warning** on `load_inputs` at
   `crates/cb-train/tests/device_ctr_gate_test.rs:140`, verified pre-existing at `HEAD` (same
   line, same signature).
5. **12 `cb-train` integration-test targets fail `cargo clippy -p cb-train --lib --tests`**
   (`--keep-going` is required to see them all; without it clippy stops at whichever compiles
   first, which is why different tasks reported different "the" failing file):
   `one_hot_draw_accounting_test`, `learn_set_shuffle_oracle_test`,
   `yetirank_pairwise_tree_rng_oracle_test`, `tensor_ctr_oracle_test`,
   `device_fold_count_gate_test`, `permutation_oracle_test`, `structure_fold_cycle_oracle_test`,
   `plain_ctr_oracle_test`, `ordered_ctr_oracle_test`, `s_order_ctr_bins_oracle_test`,
   `device_seam_test`, `ordered_boost_oracle_test` — all `panic` / `expect_used` /
   `indexing_slicing` in committed `tests/*.rs`. `cargo test` is unaffected. The `cb-train`
   **lib / lib-test** target itself is clippy-clean, which is why a task whose work lands in
   `src/` can still verify itself.
   Plus two pre-existing `clippy::useless_vec` warnings inside `device_ctr_combo_config_test.rs`
   (`&vec![0usize; N]`), byte-identical at `HEAD`, only line-shifted.
6. **`crates/cb-oracle` has ~5 pre-existing clippy errors** unrelated to this phase.
7. **The workspace is NOT rustfmt-clean at `HEAD`.** `cargo fmt -p cb-train -- --check` reports
   **46 hunks in `boosting.rs`** and 6 in `device_ctr_combo_config_test.rs` on *unmodified* code
   (T00 measured 836 hunks crate-wide at the start of the phase). ⇒ **`cargo fmt --check` is not a
   repo gate** and was not used as one; tasks verified their own files by diffing hunk sets
   against extracted `HEAD` copies instead.
8. **`kernels::poisson_bootstrap_speed_test` is a known do-not-chase flake under concurrent GPU
   load (R-13).** It lives in `run_device_tests.sh`'s isolated `PERF_TESTS` lane. T24's two runs
   read **9.6×** and **7.0×** against a 5× bar — both PASS. Reported, never chased.

### 6.2 T05's escalation rung — **rung 0**, not the predicted rung 3

PLAN §6 assumption 4 predicted the `ctr_device_buckets` fixture would need **rung 3** of the
four-rung escalation ladder (widen seeds → cardinality 8 → a second cat column → 10 iterations).
It landed at **rung 0, the pinned starting point**, on its **second** seed: `CARDS = (6,)` (one cat
column), 64 rows, 5 iterations, `data_seed = 1`. `X_cat.npy` is therefore `(64,) i32`, the shape
T10's consumer contract expects, and no smoke-test shape adjustment was needed. The full ladder
remains encoded in the generator (`RUNGS`) and `config.json` records
`escalation_rung / escalation_rung_reason / escalation_attempts`.

The guard is demonstrably not vacuous: **rung 0 / seed 0 was rejected by it** — `n_ctr=1`,
`target_border_idxs=[0]` — while seed 1 gives `n_ctr=2`, `target_border_idxs=[0, 1]`. A
`[0]`-only model is a real, reachable outcome at this shape.

(For completeness: `ctr_device_btmv` and `ctr_device_counter` also landed at their starting rung.
T22's *guard-4* ladder — a different ladder, for combination coverage — needed **rung 1** for the
combination-Counter arm, and at 20 iterations rather than 10: combination Counter chooses **zero**
≥2-member splits on this corpus at 5/10 iterations, depth 2 and 3, and priors 0.0/0.25/1.0/2.0, on
**both** arms. Root cause measured: the combination Counter column carries only the JOINT
FREQUENCY of the two cat columns, `corr(joint freq, y) = 0.163`, against `simple_ctr = Borders`
columns that encode the target statistic directly, plus a `(1 + count/maxCount)^-0.5` weight
penalising the combination's larger bucket count. **General carry: a CTR type that carries no
target signal (Counter, FeatureFreq) will not produce combination splits on a small, near-uniform
categorical corpus at short horizons.** Guard 4 was never weakened.)

### 6.3 T10's leaf-gather-path finding — PLAN §6 assumption 8 is **SETTLED**

**Both** leaf paths run on a device CTR fit, and they agree **by construction**. No escalation, no
gather patch was needed or made.

| path | where | how it indexes the averaging CTR columns |
|---|---|---|
| the returned leaf **VALUES** | `GpuTrainSession::grow_one`, `session.rs` (`ctr_averaging_bins` arm) | **by CTR column POSITION** — `avg_bins.get(fu - base)`, `base = n_features - n_ctr` |
| the per-object leaf **ASSIGNMENT** (main-approx update + leaf weights) | `boosting.rs`, the `device_has_ctr_split` branch | **by FULL IDENTITY** — `assign_leaf_over_ctr_columns(&matrix, &averaging_ctr_features, …)`, keyed on `projection && ctr_type && target_border_idx && prior_num.to_bits() && prior_denom.to_bits()` |

The second path exists on a device fit precisely because `fused_unit_fold = !device_has_ctr_split
&& …` is `false` whenever a CTR split is present, so the fused device-`leaf_of` fast path is
structurally bypassed for CTR trees.

**Why position indexing is correct even for Buckets' two-columns-per-`(projection, prior)`
layout** — the failure the plan feared (a `b = 0` split paired with the `b = 1` averaging column):
`materialize_ctr_columns_for_perm` is the **SINGLE producer** of both the structure and the
averaging list, called once per permutation over the same `absolute_projections`/`ctr_candidates`;
`build_device_ctr_config`'s `build_columns` is an order-preserving `map` over each; and
`ctr_covered` rejects any config with `avg.columns.len() != ctr.columns.len()`. ⇒ device tail
position `i` ⇔ the same FULL identity on both host lists.

Measured, not argued — three investigation mutations:

* **MUT-A**, device position index mirrored ⇒ `|Δ| = 2.506e-1` (four orders past the bar) ⇒ the
  position-indexed gather is LIVE, this fixture detects a mispairing, and the `b=1` averaging
  column is genuinely different from `b=0`;
* **MUT-B**, host gather fed the STRUCTURE columns ⇒ `|Δ| = 2.236e-4` ⇒ the host full-identity
  gather over the AVERAGING columns is ALSO live;
* **MUT-C**, `target_border_idx` deleted from the host key ⇒ CPU `ctr_buckets_simple_oracle_test`
  RED (`max |diff| = 2.045e-2`) but the **device** e2e GREEN — recorded as an honest limitation:
  on this fixture the device selects only `b = 0`, the first of the two columns, so the host key's
  `target_border_idx` conjunct is pinned by the **CPU** oracle, not by the device e2e.

**The property later work must preserve**: any task that filters, reorders or de-duplicates one of
the two column lists without the other silently breaks the pairing. The detector is
`device_ctr_buckets_fit_test` (`|Δ| = 2.506e-1`), and since T19 also the executable structural pin
`combination_arity_is_structurally_bounded_and_carried_whole` (exactly ONE `DeviceCtrColumn {`
construction, exactly TWO `build_columns(` calls, exactly ONE `tensor_ctr_candidates(` call).

---

## 7. Fixtures: frozen, and provably so

- `git diff --name-status a0a67ec..HEAD -- crates/cb-oracle/fixtures/` ⇒ **28 files, every one
  `A` (added)**. **Zero `M`, zero `D`.**
- Every added path is under exactly the **three new directories** `ctr_device_buckets/`,
  `ctr_device_counter/`, `ctr_device_btmv/`.
- `git status --porcelain crates/cb-oracle/fixtures/` ⇒ **empty** (nothing uncommitted either).
- `ctr_btmv_simple_oracle_test` (the Track E no-op proof, DCTR-05) passes **unchanged**: its
  `-- --test-threads=1` output is byte-identical to T03's captured pre-T04 baseline apart from
  libtest's wall-clock suffix, verified by `diff` at T04 **and** again here at T24.
- Each new `config.json` carries the `"note": "FROZEN…"` marker and the reproducibility caveat;
  §2.6 / R-12 honoured throughout (CatBoost quantization is run-to-run nondeterministic on
  categorical routing, so regenerating any fixture would invalidate the ≤1e-5 gate).

---

## 8. §2.5 mutation-check register — every message, by task

PLAN §2.5 named **nine** mutation candidates (T03, T09-B, T13, T14-B, T15, T19 ×2, T20, T22 ×2,
T21 per pin, T23). The phase actually executed **~76 mutations across 18 tasks**, every one
isolated (live only while a single focused `--test <name>` / `--lib <filter>` ran — the MINOR-9
isolation rule), every one reverted, and every revert verified by `git diff` and a
`grep -rn "MUTATION-T<nn>"` sweep. **After T12 lost three of its own hunks to `git checkout <file>`,
every later revert was a targeted textual edit.**

### T00 — the gate-state table (3)

```
row 6 (simple/Borders/b=0/denom=2.0): expected false, got true      [delete `&& col.prior_denom == 1.0`]
row 3 (simple/Counter/b=0/denom=1.0): expected false, got true      [delete the ctr_type conjunct]
row 2 (combo[0,1]/Borders/b=0/denom=1.0): expected false, got true  [delete `col.projection.is_simple() &&`]
```

### T01 — DCTR-02 (1)

`const CTR_PRIOR_DENOM: f64 = 1.0;` → `2.0`:
```
upstream ctr_helper.cpp:50 forbids denom != 1 on the CPU task type; if this constant ever becomes
non-unit, the device gate's deleted `prior_denom == 1.0` conjunct must be RESTORED (see DCTR-02)
  left: 2.0   right: 1.0
```
Its sibling `the_device_gate_no_longer_reads_prior_denom` needed no mutation — it was genuinely
red before the deletion.

### T03 — DCTR-05 (1)

`calc_ctr.rs:62` `f64::max(1.0, prior)` → `f64::max(2.0, prior)`:
```
prior 0 is in [0,1] and must normalize to the identity (shift 0, norm 1), making DCTR-04 a no-op for it
  left: (-0.0, 2.0)   right: (0.0, 1.0)
```
Note `left: (-0.0, …)`: `shift = -min(0, p)` is **negative zero** at `p = 0`, which is why every
assertion in this family uses `PartialEq`, never `f64::to_bits` — a bit comparison fails on
*unmutated* code.

### T04 — DCTR-04, the CPU BTMV quantizer (2)

```
M1 (drop (ctr+shift)/norm):  document 1: ((1.5 + 0.0) / 2.0) * 15 = 11.25 ⇒ bin 11 … left: 15  right: 11
M2 (divide by `denom`, the shadowing trap): document 2: … ⇒ bin 7 … left: 5   right: 7
```
**M2 is why the Red column was widened from two documents to three**: at `count == 1` the CTR
denominator (2.0) coincides numerically with `calc_normalization(2.0)`'s `norm` (2.0), so
documents 0 and 1 both pass under the conflated implementation. Any device BTMV self-oracle
driving ≤2 documents per bucket inherits the same blind spot.

### T08 — DCTR-06, the Buckets numerator (2)

```
target_border_idx = 2 must be rejected at binclf, got Ok(([1, 0, 3, 4, …]))     ← payload is the N1 numerator: the silent wrong answer
a non-class-prefix ctr_type must be rejected, got Ok(([1, 0, 3, 4, …]))
```

### T09 — DCTR-07, plumbing and grouping (2)

```
the two Buckets numerator columns of one (projection, prior) must share ONE weight_group … left: 0  right: 1
Buckets: a column not binarizing to n_bins buckets must still decline (uniform-histogram invariant)
```

### T10 — DCTR-08, the Buckets e2e (5 + 3 investigation)

```
3.1  Buckets must be CPU-legal (`restrictions.h:18-48`) …
3.2  Borders emits exactly ONE column per (projection, prior) at binclf … left: 2  right: 1
3.3  the fit did not commit to the device: 0 of 5 trees were grown on device (the ≤1e-5 bar above passed regardless — R-8)  left: 0  right: 5
D    row 4 … expected true, got false + row 9 … expected true, got false   (row 10 stays GREEN)
E    row 4 … expected true, got false + row 10 … expected true, got false  (row 9 stays GREEN)
```
D and E are the answer to T00's open question: **T10's edit removed BOTH conjuncts**, each now
independently pinned. The three investigation mutations (A/B/C) are transcribed in §6.3.

### T11 — DCTR-09, the Counter kernel (2)

```
A  device Counter denominator must be the CONSTANT max bucket total for every object
     left: [20, 22, 13, 20, …]   right: [22, 22, 22, 22, …]
B  the Counter statistic must be permutation independent (ctr_type.cpp:43-56)
     left: [2, 4, 1, 2, …]       right: [2, 4, 1, 2, … 3 …]
```

### T12 — DCTR-10, Counter in production (5)

```
1  row 3 (simple/Counter/b=0/denom=1.0): expected true, got false        [row 3 and ONLY row 3]
   + `ctr_types_are_device_covered` must name `ECtrType::Counter` exactly once … Found: []
2  Counter must be permutation INDEPENDENT (`ctr_type.cpp:43-56`) …
3  Counter's default prior set is the single `0/1` … left: 3  right: 1
4  Counter: a column not binarizing to n_bins buckets must still decline (uniform-histogram invariant)
5  [prints `max |Δpred| = 2.776e-17 (bar 1e-5)` FIRST, then] the fit did not commit to the device:
   0 of 5 trees were grown on device … left: 0  right: 5
```

### T13 — DCTR-11, `counter_calc_method` + eval sets (2, plus a genuine Red)

The genuine Red first — **a learn-set-size vacuity hazard that would otherwise have shipped**:
```
[skiptest+eval] the trained model must contain ≥1 CTR split — without one this cell says nothing about CTR routing
(3 of 4 cells; the ONE cell that passed was the required NEGATIVE)
```
Then:
```
MUT-1 (delete `&& eval_sets.is_empty()`):  [full+eval]  … the fit committed 5 of 5 trees to the DEVICE … left: 5  right: 0
                                            [skiptest+eval] … left: 5  right: 0     (both COMMITTING cells stay green)
MUT-2 (`&& false` on the type gate):       [full+noeval] … 0 of 5 trees … left: 0  right: 5   in 0.02 s
                                            [skiptest+noeval] … left: 0  right: 5   (both DECLINING cells stay green)
```
The two mutations **partition** the four cells — which is what makes "the decline is at the
eval-set clause, not the type list" a measurement rather than a story.

### T14 — DCTR-12, the BTMV accumulator (4)

```
A  the device BTMV sum is NOT bit-equal to an f32-accumulating reference at divisor = 3: 38 of 96
   documents mismatch, first at doc 0: device 0x40E00000 (7) vs f32 reference 0x40E00001 (7.0000005)
B  device BTMV sum != online_mean_prefix sum (bitwise)   [doc 1 still matches — a first-in-bucket document]
C  divisor = 0 must be rejected (it is a device division by zero), got Ok(([NaN, 0.0, NaN, …]))
D  a target class above targetBorderCount must be rejected, got Ok(([7.0, 9.0, 5.0, …]))
```
**MUT-A is C-2 measured in both directions**: in the *same* f64-widened build the binclf test and
the buffer-width pin both **PASSED**, so `SPEC.md` DCTR-12's "an f64 device sum must FAIL this
test" is false at binclf and the multiclass `divisor = 3` detector is the only proof.
**MUT-C PASSED on its first run** — a sibling `class > divisor` guard rejected the input first, so
the assertion had been green without the guard it claimed to pin ever existing. Fixed *while the
mutation was live* (an all-zero class column), then it failed with the real payload: a **NaN CTR
column**, which `binarize_ctr_kernel` maps to bin 0 everywhere because `NaN > border` is false.

### T15 — DCTR-13, BTMV ≡ Borders@0 at binclf (2)

```
1 (BTMV folds the COMPLEMENT class): device BTMV (divisor = 1) and device Borders@0 emit DIFFERENT
  cindex columns at binclf … first differing object Some(0): BTMV Some(5) vs Borders Some(9)
  left: [5, 12, 6, 9, …]   right: [9, 2, 8, 5, …]      ← ALL 128 of 128 documents differ
2 (the REFERENCE arm moved off Borders@0): … BTMV Some(9) vs Borders Some(5)
  left: [9, 2, 8, 5, …]    right: [5, 12, 6, 9, …]     ← element-for-element MUT-1's right/left
```
Two one-line mutations in two different kernels reached through two different launchers produce
the identical column pair with the arms exchanged. T15 **declined** §2.5's offered
`≥2 distinct bins` substitute, and the reason generalises: for any `A == B` differential, a
non-degeneracy guard proves non-degeneracy, never **non-tautology** — only a **one-sided**
mutation proves the two arms are distinct.

### T16 — DCTR-14, BTMV in production (7)

```
1  row 5 (simple/BTMV/b=0/denom=1.0): expected true, got false   [row 5 and ONLY row 5]
   + must name `ECtrType::BinarizedTargetMeanValue` exactly once … Found: []
   + the gate must name `ECtrType::BinarizedTargetMeanValue` exactly once in its admission list
2  BinarizedTargetMeanValue must be permutation DEPENDENT (`ctr_type.cpp:43-56`) …
3  BTMV's default prior set is the `{0/1, 0.5/1, 1/1}` triple … left: 1  right: 3
4  BinarizedTargetMeanValue: a column not binarizing to n_bins buckets must still decline
5  an unimplemented AVERAGING column must decline (the averaging `.all(..)` closure needs the same conjunct)
6  device BTMV CTR train failed: Unsupported("device CTR type 2 is not implemented")     ← in 0.14 s
7  [prints the passing bar first] the fit did not commit to the device: 0 of 5 … left: 0  right: 5
```
**MUT-6 is a generally useful new shape: delete the dispatch ARM, not the gate.** Unlike `&& false`
(which proves the fit *can* commit) it proves the fit reaches **the specific new production call
site** — and it records C-14 in its strongest form: because `ctr_covered` admits the type, the
`Ok(None)` decline path is not taken and the mismatch surfaces as a **typed `CbError` out of
`train_cat`**, not even as `grown == 0`.

### T17 — DCTR-15, the D-1 eligibility gate (6)

```
1a  a single-member projection must be eligible with an empty chosen list (`AddSimpleCtrs` is unconditional)
1b  `|q| + 1 != |p|` (gap of two) must be ineligible even though `q` is a subset
1c  *** PASSED — the transcribed CPU case list is BLIND to the subset conjunct ***
    after adding case 8 (partial overlap, right arity): a PARTIALLY overlapping `q` of the right
    arity must still be ineligible — `q` must be a SUBSET of the candidate, not merely intersect it
2   (pass-C `continue` forced) the trained model must contain ≥1 CTR split
3a  (gate tightened so an ABSENT list is ineligible) — GREEN, same delta ⇒ real 1-member lists reach pass C
3b  (3a + the population blanked) the trained model must contain ≥1 CTR split ⇒ 3a's green was not vacuous
```
**MUT-1c is a live CPU-side coverage gap, not a device artifact**: the same `all` → `any` edit to
`cb_train::tree::combination_ctr_eligible` would leave all seven of `tree_test.rs:296-361`'s cases
green. T17 added the partial-overlap case on the device side; **the CPU side still lacks it.**

### T18 — DCTR-16, the D-2 filtered `eligible_max` (8)

```
1  a column with NO member-list entry must be treated as SIMPLE (eligible) … left: 6   right: 40
2  an INELIGIBLE combination's bucket count must not enter `maxCount` …      left: 40  right: 6
6  once `[0]` is chosen, `[0,1]` is eligible and its bucket count MUST enter `maxCount` … left: 6  right: 40
3  `.unwrap_or(1)` → `.unwrap_or(0)`   — PASSES
3b `.max(1)`        → `.max(0)`        — PASSES        ⇒ mutually redundant by construction, NOT test blindness
                                          (both directions were driven before concluding)
4  pass C must compute `eligible_max` through `resident_eligible_max_bucket_count` (DCTR-16 / D-2);
   the unfiltered `cs.bucket_counts.iter().copied().max()` must not come back
5  C-16: the phantom count must be folded in OUTSIDE the eligibility filter
W  obj 0: device CTR prediction 0.0974… vs upstream -0.0213… exceeds ≤1e-5 (|Δ|=1.188e-1)
```
**MUT-4 and MUT-5 left all four unit tests GREEN** — only the source scan reddened. Measured
twice, and it is the phase's fourth instance of the same lesson: **a unit test on an extracted
helper proves the helper, never the call site.**

### T19 — DCTR-17, combination CTR on device (2 mandated + 2 extras + the R-20 probe)

```
1 (the REAL hoist to fit lifetime): obj 0: device CTR prediction 0.006279… vs upstream -0.021179…
  exceeds ≤1e-5 (|Δ| = 2.746e-2)   ← bit-for-bit the control-arm number for "the arity gate simply
  opened with NO per-level eligibility gate" ⇒ a fit-lifetime list makes the gate vacuous from
  tree 1 onward, observationally identical to having no gate at all
2 (restore `col.projection.is_simple() &&`): [prints `8 CTR splits (3 ≥2-member); max |Δpred| =
  1.388e-17 (bar 1e-5)` FIRST, then] the combination-CTR fit must COMMIT to the device: expected 5
  device grows, got 0 … left: 0  right: 5
  + row 2 (combo[0,1]/Borders/b=0/denom=1.0): expected true, got false   [ONLY row 2]
  + the_device_gate_no_longer_reads_the_projection_arity … FAILED
```

### T20 — DCTR-19, `CountingGpu` on the pre-existing e2e (2 cycles)

```
Cycle A — the PRE-T20 test with the gate forced closed:  *** PASSED ***   ← the recorded, executable
  proof of the R-8 false-pass class:  max |Δpred| 4.483e-11 → 1.388e-17, wall time 1.92 s → 0.01 s,
  and the OLD test detected neither
Cycle B — same mutation, T20's assertion in place:
  [device-ctr-e2e] 1 CTR splits; max |Δpred| = 1.388e-17 (bar 1e-5)      ← printed BEFORE the panic
  the fit did not commit to the device: 0 of 5 trees were grown on device
  (the ≤1e-5 bar above passed regardless — R-8)   left: 0  right: 5
```
Cycle B is also where the **report-before-assert ordering** discipline comes from: putting the
`CountingGpu` assertion *after* the ≤1e-5 loop makes a single mutation run yield both halves of
the required evidence.

### T21 — DCTR-03 + the surviving-clause pins (9)

```
1a (SPEC-OH-26 alone disabled):  PROBE: ok=true grown=0  in 0.05 s ⇒ the RETAINED `one_hot_bins.is_empty()`
   conjunct is what holds a mixed pool off the device
   a mixed one-hot/CTR pool must be REFUSED, not trained: (Model { … one_hot_absolute: [0] }, …)
1b (BOTH disabled, the mandated joint mutation): PROBE: ok=true grown=5 in 52.81 s ⇒ the mixed pool
   reaches the DEVICE grower — the latent hazard DCTR-03 retains the conjunct against
2a (multi-permutation clause deleted):        GREEN
2b (+ the backend's view of fold_count too):  [ctr-multi-perm] expected 0 device grows … left: 5  right: 0
2c (complement: host restored, backend blind): GREEN
   ⇒ two mutually-redundant guards on the same host quantity — T18 §3's mode, not blindness
3  [borders-7] … trees=5 ctr_splits=0 grown=5/5, then: the trained model must contain ≥1 CTR split
   ⇒ the border-count shape check degrades SILENTLY: the fit commits and simply chooses zero CTR splits
4a/4b/4c (cat-only, cumulative over 3 of its 4 guards): ALL GREEN, `grown = 0` throughout;
   driving the fourth was DECLINED (it yields n_features == 0, n_bins == 0 — a red from an empty
   problem says nothing about the boundary)
```
**Phase-level lesson**: "run the mutation for the clause you CLAIM" is sometimes **impossible**,
because the code is overdetermined. T21's honest substitute — shipped — is a **control arm through
the same helper** (`unmodified_float_half_commits_to_device`, `grown = 5/5`), making each pin a
one-factor experiment. Any later decline test should check for overdetermination first and budget
for a control arm.

### T22 — DCTR-20, the combination × non-Borders differential (2 mandated + 2 complements)

```
1  (un-wire D-2 at the call site) — GREEN at the shipped config AND at iterations = 20 ⇒ R-20 OPEN (§5)
2  (restore the pre-T17 `member_bins.first()`-only bucket_counts fallback) — GREEN, because the branch
   is production-unreachable. NOT accepted at face value; both complements driven:
2b (the pre-T17 fallback FORCED onto the path): ALL THREE ARMS RED —
   device: 10 CTR splits (5 ≥2-member) | cpu: 8 (3)
   [btmv] tree 0: the FLOAT split sequence diverges between the device and CPU growers
     left: []   right: [Split { feature: 1, border: 0.7066377401 }]
   (same shape on buckets tree 0 and counter tree 10)
2c (same forcing, T17's fold-all-members arm restored): ALL THREE ARMS GREEN, byte-identical to the
   unmutated run ⇒ T17's `combine_projection_bins` fallback reproduces the production `bucket_count`
   EXACTLY. No previously-shipped test makes that statement.
```

### T23 — DCTR-18, the final gate (4)

```
A  the FINAL device CTR gate does not admit exactly the CPU-supported type set:
     FeatureFreq (discriminant 5): expected false, got true
     mixed {Borders, FeatureFreq}: expected false, got true      ← the `.all(..)` fold case is live too
B  *** the first draft of the delegation pin PASSED under a COMPLETE un-wiring *** because
   `gate_body()` returns the gate's INLINE COMMENTS and the comment spells `is_cpu_supported`.
   Hardened in place to `code_lines_mentioning` (comment-stripped, exact count 1), then:
     the gate must DELEGATE its admission decision to `crate::ctr::ECtrType::is_cpu_supported`
     (DCTR-18), in CODE and exactly once, not carry its own type list
   (the same run also caught `expected exactly ONE production use of `target_border_count` … left: 2  right: 1`)
C  (restore a BEHAVIOURALLY IDENTICAL hand-rolled list) — exactly ONE test reddens, the structural
   one; the behavioural test, the prior-denominator case and ALL TEN gate-state rows stay green
   ⇒ C-3's "do not hand-roll a second type list" is now ENFORCED, and no behavioural test could do it
D  (delete `!cols.is_empty()`): the empty column set: expected false, got true
```
**Rule that falls out of MUT-B, and it is the fifth instance of this class in the phase**: a
**positive** claim about the gate body must go through `code_lines_mentioning` (comment-stripped);
only **negative** claims ("the body must not contain X") are correct with a raw `contains`, and
those are correct precisely because a surviving comment IS a rename hazard.

### T24 (this task)

No test function is added, so there is no green-on-write test to mutate. The runner's Red (23 PASS,
the five new binaries absent) and Green (28 PASS) are the discrimination proof. See `notes/T24.md`.

### T05 / T06 / T07 — the fixtures

Not on §2.5's candidate list and changing no production code, so no production mutation was
possible. Each nevertheless demonstrated its key guard is discriminating by measuring rejected
seeds: T05 rung 0/seed 0 rejected on `target_border_idxs == [0]`; T06 **11 of 24** seeds rejected
(8 for `counter == 0`, 3 for no float split); T06 additionally showed the smoke test's explicit
prior pin rejects `["Counter"]` in place of `["Counter:Prior=0.5"]`.

---

## 9. OPEN findings — carried forward, deliberately NOT closed

### 9.1 `T22-OBS-1` — ~1e-3 device-vs-CPU leaf-value divergence on a CTR fit's **CTR-FREE trees**

**Status: OPEN. Pre-existing, NOT caused by this phase, and deliberately not patched.**
**The user was asked and ruled: RECORD ONLY, decide later** (coordinator disposition, 2026-08-10).
No bug chase, no spec, no plan, no fix in P1.

On any tree of a CTR fit whose greedy search chooses **zero** CTR splits, the device and CPU leaf
**values** diverge by ~1e-3 while their **split sequences stay identical** — against ~1e-17 on
every CTR-carrying tree.

**Reproduction — it is NOT a DCTR-20 artifact.** It reproduces on the **already-shipped
`combinations_ctr = Borders` configuration of `ctr_device_combo`**, merely run to **30 iterations**
instead of the fixture's 5:

```
tree 23   7.824e-4    (ctr_splits == 0)
tree 25   1.223e-3    (ctr_splits == 0)
tree 28   1.943e-3    (ctr_splits == 0)
tree 29   1.296e-3    (ctr_splits == 0)
```

(and, on the `combinations_ctr = Counter`, 30 iters / depth 2 probe where it was first seen:
trees 0..22 at 1e-17…2.4e-17, **tree 23 → 1.069e-3**, trees 24-26 at ~6.4e-6, **tree 27 →
1.280e-3**, trees 28-29 at 1.3e-5.)

**Why nothing has caught it**: every committed device CTR fixture stops at **5 iterations**, where
every tree still carries a CTR split. The first CTR-free tree on this corpus is **#23**.

**Correlate — stated as a correlate, not a diagnosis.** The divergent trees are exactly those with
`ctr_splits.is_empty()`, i.e. `device_has_ctr_split == false`, which makes
**`fused_unit_fold == true`** (`crates/cb-train/src/boosting.rs`, recorded by T22 at `:5665`, now
at **`:5747`** after T23's doc additions) and routes the fold down the branch that consumes the
device's own resident `dev_tree.leaf_of` instead of the host CTR-aware
`assign_leaf_over_ctr_columns` walk — §6.3's two-path split. The same trees are also the ones where
the device leaves `level_kinds` empty and the CPU does not. Root-causing beyond that correlate is a
separate bug chase.

**Containment**: no P1 acceptance test is affected — every P1 arm runs at 5–20 iterations, strictly
below the first CTR-free tree, and `device_ctr_combo_types_diff_test` **prints its CTR-free tree
count** per arm (`CTR-free trees: device 0 / cpu 0`) so a reader can see that rather than take it
on trust. No assertion was weakened to accommodate it. Its module doc says plainly: *do not read
this file's green as evidence that a CTR fit's CTR-free trees agree.*

**Recommended disposition**: a P2/P3 triage item.

### 9.2 `T22-OBS-2` — a prior ≠ 0.5 is NECESSARY BUT NOT SUFFICIENT for a Buckets differential

**Status: OPEN as a design constraint on any future long-horizon Buckets differential.** It is a
**material correction to the coordinator's own T22 guidance**, which offered two remedies (a prior
≠ 0.5, or a partition-invariant projection of the split set) and treated the first as adequate.

At `combinations_ctr = Buckets, Prior = 0.25`, 20 iters / depth 2 (a MUT-1 probe configuration,
**not** shipped), tree 12 level 1:

```
left:  CtrSig { projection: [0, 1], ctr_type: 1, prior_num: 0.25, target_border_idx: 0, border: 11.999999046325684 }
right: CtrSig { projection: [0, 1], ctr_type: 1, prior_num: 0.25, target_border_idx: 1, border:  0.9999990463256836 }
```

Same projection, same type, same prior; `target_border_idx` **0 vs 1** with roughly complementary
thresholds. **Reason**: a prior ≠ 0.5 removes the exact algebraic mirror
(`ctr(b0) + ctr(b1) = 1` becomes the total-dependent `(T + 0.5)/(T + 1)`) but **not** the ordinal
anti-monotonicity `bin(b0) + bin(b1) ≈ const`; with ~12 combination buckets over 15 CTR bins many
threshold pairs still induce the **same partition** and therefore an exact score tie, which the
greedy search then breaks by enumeration order.

**Verified independent of D-2**: the identical failure, byte-for-byte, occurs with MUT-1 live and
reverted. ⇒ **for any FUTURE long-horizon Buckets differential, only a genuinely
partition-invariant projection of the split set is robust.** T22's shipped Buckets arm sits at
5 iterations, below the tie, per the ladder's "lowest rung that satisfies guard 4" discipline; the
20-iteration run existed only as a MUT-1 probe. Recorded in full rather than quietly dropped.

### 9.3 R-20 — see §5. **Still open**, measured at two horizons, one hypothesis refuted.

### 9.4 Smaller open items (recorded, none blocking)

| item | detail |
|---|---|
| **CPU-side `combination_ctr_eligible` coverage gap** | T17's MUT-1c: the same `all` → `any` edit to `cb_train::tree::combination_ctr_eligible` leaves all seven of `tree_test.rs:296-361`'s cases green. T17 added the partial-overlap case on the **device** side only; the CPU side still lacks it. |
| **Three pre-existing unregistered device binaries** | `device_bootstrap_speed_test` (2 tests, not ignored), `device_oblivious_parity_probe_test` (1, not ignored), `device_perf_probe` (1, `#[ignore]`d) — created by earlier phases (`f663a45`, `9e92a89`, `44350a2`), outside C-8's scope for T24. Whether the first two belong in `TESTS` or `PERF_TESTS` is a real question for a later phase. |
| **Orphaned doc block** | `session.rs`'s `/// Compute the ADDITIONAL binarized-CTR cindex columns …` block (with its `# Errors` section) documents `build_ctr_cindex_columns` but is physically attached to `struct CtrSearchState`. T09 declined to move it (three later tasks were editing those lines); **still not re-attached**. |
| **`CountingGpu` is duplicated 8× in the CTR family** | `device_ctr_gate_test.rs:82` (canonical), `device_ctr_fit_test`, `device_ctr_buckets_fit_test`, `device_ctr_counter_fit_test`, `device_ctr_type_gate_test`, `device_ctr_btmv_fit_test`, `device_ctr_combo_fit_test`, `device_ctr_combo_types_diff_test` — and **19× across all `cb-train` device tests**. Consequence: any change to the `cb_compute::Runtime` methods it overrides (`compute_gradients`, `begin_device_training`, `grow_tree_on_device`, `end_device_training`) breaks N test files, not one. GLOBALS §2.2.6 mandates the verbatim copy; the cost is now measured. |

---

## 10. Structural facts worth carrying to P2/P3

1. **The device CTR type list is CLOSED at the four CPU-legal types** and the gate no longer spells
   any of them: it delegates to `ECtrType::from_i8` / `is_cpu_supported`, the same predicate
   `validate_ctr_types` rejects a fit with and `materialize_ctr_feature` refuses to build a column
   for. `FloatTargetMeanValue` and `FeatureFreq` are GPU-only upstream (`restrictions.h:20-32`) and
   must **never** be admitted.
2. **`ctr_device.rs` holds THREE launchers with three different contracts**, and that is the
   safety property: `launch_ordered_ctr_resident(perm, bins, class, prior, bucket_count, n,
   ctr_type, target_border_idx) → ResidentCtr`; `launch_counter_ctr_resident(bins, prior,
   bucket_count, n) → ResidentCtr` (no permutation, no class — structural permutation
   independence); `launch_btmv_ctr_resident(perm, bins, class, prior, divisor, bucket_count, n) →
   ResidentCtrMean` (an f32 `sum` channel, not an integer `good`). **One entry point per statistic,
   differing in exactly the arguments the statistic depends on ⇒ a copy-paste routing error is a
   COMPILE error, not a silent wrong numerator.** The ordered launcher still rejects any
   `ctr_type ∉ {0, 1}` and any `target_border_idx > 1` with `CbError::OutOfRange`; that guard was
   deliberately never widened.
3. **C-7 held for all four types**: `binarize_ctr_column_resident` and the per-column border table
   are **unchanged** per type. `build_ctr_cindex_columns`' match binds only the shared f64 `value`
   channel, which makes it visually obvious that the binarizer cannot see the CTR type at all.
4. **The gate-state table is RETIRED, not deleted**, keeping all ten rows; `flips_at` is replaced
   by `permanent: bool` with a header stating that P1 is complete and **P2/P3 will ADD rows rather
   than flip these**. The structural reason, reusable: the four exclusions P2/P3 lift
   (`learning_folds_for_cycle`, `eval_sets`, `has_any_scorable_feature`, the border-count shape
   check) are **not functions of a column's `(arity, ctr_type, target_border_idx, prior_denom)`
   shape**, so none of them CAN move a row.
5. **`boosting_ctr_gate_test.rs` carries THIRTEEN source-scan pins** (the running total in
   COORDINATOR-FINDINGS drifted to "nine" then "eleven" because T12's two were never counted; the
   file was always right). T23's disposition: **9 kept** (one hardened), **2 retired**,
   **1 rewritten**, **+2 new** ⇒ still 13, row-by-row in `notes/T23.md` §2 and in the file's own
   module doc.
6. **`counter_calc_method` is CLOSED as a P1 question**: `Full ≡ SkipTest` whenever `eval_sets` is
   empty, and eval sets never reach the device. T13's two declining cells carry
   `// P3 WILL INVERT THIS.` — P3 must **flip** them to `grown == params.iterations`, not preserve
   or delete them.
7. **DCTR-03's retention is MEASURED, not argued** (T21 §3.1): the conjunct the research called
   provably dead is **one deletion away** from being the only thing standing between a mixed
   one-hot/CTR pool and an untested device path. **Do not delete it.**
8. **`ObliviousTree::level_kinds` is exercised differently by the two growers** — on an all-float
   tree the device leaves it EMPTY (the documented single-kind fallback, SPEC-OH-31) while the CPU
   emits `[Float(0), Float(1)]`. Any future device-vs-CPU comparison must canonicalise to the
   decoded sequence or it will get a false red.
9. **Recurring test-design lessons, each measured at least once in this phase**: (a) a transcribed
   case list inherits the source suite's blind spots (T17); (b) a green mutation can mean a vacuous
   test **or** a redundant-by-construction guard — drive the complement before diagnosing
   (T14/T18/T21); (c) a unit test proves the helper, never the call site (T18); (d) a
   non-degeneracy guard never proves non-tautology (T15); (e) a source-scan `contains` pin can be
   satisfied by the code's own comments (T23); (f) report before asserting, so a mutation run
   yields the completion evidence and not just the panic (T20/T21).
