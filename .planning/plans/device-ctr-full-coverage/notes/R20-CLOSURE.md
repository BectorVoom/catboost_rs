# R-20 — CLOSED. `eligible_max`'s eligibility filter (DCTR-16 / D-2) now has a behavioural detector.

**Status: CLOSED.** Post-phase task, after `PHASE-COMPLETE`. Branch
`feat/device-ctr-full-coverage`, primary checkout, base **`0e20e16`**, tree clean at start
(`git status --short` empty), D-04 baseline captured **before the first edit**.

**Headline**: `crates/cb-train/tests/device_ctr_eligible_max_diff_test.rs` — a device-vs-CPU
split-sequence differential over a **synthetic in-test pool** whose two 5-category columns
give CTR bucket counts `[0] = 5`, `[1] = 5`, `[0,1] = 25`. Un-wire D-2 and it fails at
**tree 0** on the FLOAT split sequence; wire it back and it passes at
`max |Δleaf| = 1.518e-17`.

```
[device-ctr-eligible-max-diff] bucket_counts: [0]=5 [1]=5 [0,1]=25 | level-0 maxCount filtered=5 unfiltered=25 | cat_feature_weight 0.70711 -> 0.91287 (band x1.291 at model_size_reg=0.5)
[device-ctr-eligible-max-diff] device: 9 CTR splits (4 >=2-member) | cpu: 9 CTR splits (4 >=2-member) | device grows = 5, cpu device-grows = 0 (begins 1 / accepted 0)
[device-ctr-eligible-max-diff] split sequences IDENTICAL across 5 trees; max |Δleaf| = 1.518e-17 (bar 1e-4)
```

D-04 held exactly. `bash ./run_device_tests.sh` **30 PASS / 0 FAIL** (29 binaries + perf
lane; Poisson **8.7×**, no R-13 flake). Every `cb-backend` production edit is
**comment-only** — proved by a non-comment `git diff` filter that returns empty.

---

## 1. Why the three earlier probes failed, stated as a mechanism rather than a shrug

T18 built D-2 and proved it at unit and source level; T19 and T22 ran the designated
behavioural probes and both came back byte-identical. The reason is arithmetic, and knowing
it is what made the search converge.

The consumer is `(1 + count/maxCount)^(-model_size_reg)`, applied to every unused CTR
candidate. It is **increasing in `maxCount`**, so inflating `maxCount` raises **every** CTR
candidate's weight at once. Two consequences:

* the **CTR-vs-CTR** argmax only moves when two columns with *different* `bucket_count` sit
  near a weighted tie — rare;
* the **CTR-vs-float** comparison is a *single threshold*, so it flips whenever the best
  level-0 float gain lands inside the band `[w_filtered·g, w_unfiltered·g)`. **That is the
  reachable flip**, and the band's width is set entirely by the
  unfiltered/filtered `maxCount` ratio.

The difference is confined to levels where `float_split_count == 0` — i.e. **level 0** —
because from level 1 on, `phantom_max` is folded in outside the filter (C-16) and typically
dominates. At level 0 `chosen_ctr_projections` is empty, so every combination column is
ineligible and D-2's filter is at its most aggressive.

**On `ctr_device_combo`** the ratio is ~3× (simple 3/4 vs combined ≤12) ⇒ weights
`0.756 → 0.894`, an **18 % band**. Across T19's 5 trees, T22's 5/20/20 trees and T22's
20-iteration probe, no level-0 float candidate ever landed inside it. That is not
"D-2 is unobservable"; it is "this corpus' amplifier is too small".

**`model_size_reg`, the other amplifier, is NOT reachable.** It is hard-coded to
`model_size_reg_default() == 0.5` on both sides — `boosting.rs:2801` for the device
(`DeviceCtrConfig::model_size_reg`) and `boosting.rs:6612` for the CPU grower — and
`BoostParams` has no such field (the only other occurrence is
`catboost-rs-py/src/params.rs:196`, a name list). The task brief's "raise `model_size_reg`"
lever was therefore checked and **ruled out as unreachable from a fit**; the cardinality
ratio is the only remaining knob, and it carries the whole detector.

---

## 2. The search space actually covered

A scratch harness (`crates/cb-train/tests/r20_search_scratch_test.rs`, **deleted after the
search** — `git status --short` shows it gone) generated deterministic synthetic pools and ran
BOTH growers per configuration, reporting device-vs-CPU MATCH/DIFF. D-2 was bypassed through a
**temporary** `CB_R20_UNFILTERED` env gate at pass C's `eligible_max` call site, which was
removed before the shipped Red (§3) and is absent from the tree
(`grep -rn "MUTATION-R20\|CB_R20_UNFILTERED" crates/ --include=*.rs` ⇒ empty).

Fixed across the whole sweep: Logloss, `depth = 2`, `learning_rate = 0.1`,
`l2_leaf_reg = 3.0`, `random_strength = 0`, `bootstrap_type = No`, `one_hot_max_size = 1`,
`permutation_count = 1`, `max_ctr_complexity = 2`, `simple_ctr = Borders@0.5`,
`combinations_ctr = Borders@0.5`, `score_function = L2`, `random_seed = 0`, `n = 300`,
`nf = 3`, `model_size_reg = 0.5` (unreachable, see §1).

| axis | values swept |
|---|---|
| cat cardinality `k1 × k2` | 3×3, 4×4, 5×5, 6×6, 7×7, 8×8, 12×12, 16×16 |
| data seed | 0 … 39 |
| (float effect : cat effect) | 0.3:1, 0.5:1, 1:1, 1:2, 1:3, 0.5:2 |
| iterations | 3, 5, 6, 8, 10 |

### Measured `maxCount` deltas — the input perturbation, verified non-zero before hunting

The unfiltered/filtered `maxCount` at level 0 is `joint distinct pairs` vs `max(k1, k2)`:

| `k1 × k2` | filtered `maxCount` | unfiltered `maxCount` | ratio | weight band (`r = 0.5`) |
|---|---|---|---|---|
| 3×3 | 3 | 9 | 3.0× | 0.7071 → 0.8452 (×1.195) |
| 4×4 | 4 | 16 | 4.0× | 0.7071 → 0.8944 (×1.265) |
| **5×5** | **5** | **25** | **5.0×** | **0.70711 → 0.91287 (×1.291)** ◀ SHIPPED |
| 6×6 | 6 | 36 | 6.0× | 0.7071 → 0.9258 (×1.309) |
| 7×7 | 7 | 48–49 | ~7× | 0.7071 → 0.9354 (×1.323) |
| 8×8 | 8 | 64 | 8.0× | 0.7071 → 0.9428 (×1.333) |
| 12×12 | 12 | ~130 | ~11× | 0.7071 → 0.9578 (×1.354) |
| 16×16 | 16 | 169–185 | ~11× | 0.7071 → 0.9565 (×1.353) |
| (`ctr_device_combo`, for contrast) | 3–4 | ≤12 | ~3× | 0.756 → 0.894 (×1.18) |

The 5/5/25 row is **not** a host-side estimate in the shipped test: it is read back from the
production materializer `cb_train::materialize_ctr_feature(...).bucket_count`
(`TOnlineCtrUniqValuesCounts::Count`, `ctrs.h:50`) and asserted before either fit runs.

### Verdicts

| sweep | D-2 LIVE | D-2 BYPASSED |
|---|---|---|
| 16×16, coef 1:1, seeds 0–39, 5 iters | **0 DIFF / 40** | **13 DIFF / 40** |
| 16×16, coef {1:2, 1:3, 0.5:2}, seeds 0–19, 8 iters | — | 3 DIFF / 60 |
| {4×4, 6×6, 8×8, 12×12}, coef 1:2, seeds 0–15, 6 iters | — | 3 DIFF / 64 |
| {5×5, 6×6, 7×7} × coef {0.3:1, 0.5:1, 1:1, 1:2}, seeds 0–19, 10 iters | — | ≥58 DIFF / 240 (the captured window held 58 DIFF rows and was truncated) |
| {3×3, 4×4, 5×5} × coef {0.3:1, 1:1}, seeds 0–7, 10 iters | **0 DIFF / 48** | — |
| the SHIPPED config at iterations 3, 5, 6, 8, 10 | **0 DIFF / 5** | **5 DIFF / 5**, all at tree 0 |

**Nothing discriminated with D-2 live; a great deal discriminated with it bypassed.** That is
the shape a real detector has, and it is the shape the `ctr_device_combo` corpus could not
produce at any horizon.

### What discriminated, exactly

The shipped configuration, bypassed, at tree 0:

```
      dev: [|C[1]/t0/b0@6.0000,C[0]/t0/b0@10.0000]
      cpu: [F0@0.521169|C[0]/t0/b0@8.0000]
```

The CPU takes `Float(0) @ 0.5212` at level 0; the device, with the inflated `maxCount`, takes
a CTR split on projection `[1]` instead — the CTR-vs-float threshold flip §1 predicts, and the
whole model diverges from there.

---

## 3. Red → Green (verbatim)

### 3.1 Baseline, captured BEFORE the first edit (D-04)

```
[device-ctr-e2e]         1 CTR splits; max |Δpred| = 4.483e-11 (bar 1e-5);  device grows = 5
[device-ctr-buckets-e2e] 8 Buckets CTR splits; chosen target_border_idx values = [0]×8 (distinct [0])
[device-ctr-buckets-e2e] max |Δpred| = 2.776e-17 (bar 1e-5);                device grows = 5
[device-ctr-counter-e2e] 5 Counter CTR splits
[device-ctr-counter-e2e] max |Δpred| = 1.388e-17 (bar 1e-5);                device grows = 5
[device-ctr-btmv-e2e]    5 BTMV CTR splits; max bucket occupancy = 15
[device-ctr-btmv-e2e]    max |Δpred| = 2.776e-17 (bar 1e-5);                device grows = 5
[device-ctr-combo-e2e]   8 CTR splits (3 of them ≥2-member combinations); max |Δpred| = 2.082e-17; device grows = 5
```

### 3.2 RED — the test written while D-2 was un-wired

The env gate was replaced by the plain pre-T18 unfiltered expression
(`let eligible_max = cs.bucket_counts.iter().copied().max().unwrap_or(1).max(1);`), so the Red
is a genuine ordering and not a post-hoc mutation:

```
running 1 test
test an_ineligible_combination_must_not_inflate_the_level_0_max_bucket_count ... [device-ctr-eligible-max-diff] bucket_counts: [0]=5 [1]=5 [0,1]=25 | level-0 maxCount filtered=5 unfiltered=25 | cat_feature_weight 0.70711 -> 0.91287 (band x1.291 at model_size_reg=0.5)
[device-ctr-eligible-max-diff] device: 10 CTR splits (3 >=2-member) | cpu: 9 CTR splits (4 >=2-member) | device grows = 5, cpu device-grows = 0 (begins 1 / accepted 0)

thread 'an_ineligible_combination_must_not_inflate_the_level_0_max_bucket_count' (442640) panicked at crates/cb-train/tests/device_ctr_eligible_max_diff_test.rs:644:13:
assertion `left == right` failed: tree 0: the FLOAT split sequence diverges between the device and CPU growers. If the device chose a CTR split where the CPU chose a float, check D-2 first: an UNFILTERED `eligible_max` inflates `maxCount`, raises every CTR candidate's `(1 + count/maxCount)^-0.5` weight, and lets a CTR candidate overtake the float winner (DCTR-16 / R-20)
  left: []
 right: [Split { feature: 0, border: 0.5211687088012695 }]
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.10s
```

Note the two lines that print *before* the panic: the perturbation guard has already shown the
input difference is real (`maxCount 5 vs 25`), and the counting wrappers have already shown
the device committed (`grows = 5`) while the reference arm did not (`accepted 0`). So the Red
cannot be a routing artifact.

### 3.3 GREEN — D-2 restored by a targeted textual edit

`git checkout <file>` was never run.

```
[device-ctr-eligible-max-diff] bucket_counts: [0]=5 [1]=5 [0,1]=25 | level-0 maxCount filtered=5 unfiltered=25 | cat_feature_weight 0.70711 -> 0.91287 (band x1.291 at model_size_reg=0.5)
[device-ctr-eligible-max-diff] device: 9 CTR splits (4 >=2-member) | cpu: 9 CTR splits (4 >=2-member) | device grows = 5, cpu device-grows = 0 (begins 1 / accepted 0)
[device-ctr-eligible-max-diff] split sequences IDENTICAL across 5 trees; max |Δleaf| = 1.518e-17 (bar 1e-4)
test result: ok. 1 passed; 0 failed; …; finished in 1.96s
```

Run **twice** back to back: byte-identical output both times (the pool is generated by an
in-file 64-bit LCG, so there is no `rand`-version or platform dependence).

### 3.4 Refactor

One clippy warning was introduced and fixed inside the new file
(`clippy::slow_vector_initialization`, `vec![Vec::with_capacity(N); NF]` →
`(0..NF).map(|_| Vec::with_capacity(N)).collect()`). Afterwards the only diagnostic on
`cargo clippy -p cb-train --no-default-features --features rocm --test
device_ctr_eligible_max_diff_test` is the transitively-compiled pre-existing
`duplicated_attributes`; **zero diagnostics name the new file**.

Comment-only updates so the in-source record is not left contradicting the code: the pass-C
R-20 paragraph and `resident_eligible_max_bucket_count`'s doc in `gpu_runtime/mod.rs`, and
`gpu_runtime/ctr_eligibility_test.rs`'s module doc plus the source-scan test's doc. Each now
records R-20 as CLOSED, names the detector, and **keeps** the three negative measurements,
because they are what makes the shipped configuration load-bearing.

---

## 4. GLOBALS §2.5 mutation evidence — four mutations, two red, two diagnosed inert

Protocol for every one: mutate → run **only** `--test device_ctr_eligible_max_diff_test` →
record verbatim → **revert by targeted textual edit** → re-run green. No mutation was live
during any suite-wide command. `grep -rn "MUTATION-R20" crates/ --include=*.rs` ⇒ **empty**,
and the non-comment `git diff` filter on both `cb-backend` files returns empty.

| # | mutation | result |
|---|---|---|
| MUT-A | un-wire D-2 at the call site (the pre-T18 unfiltered `.max()`) | **FAILS** — §3.2, tree 0, float sequence `[]` vs `[Split { feature: 0, border: 0.5211687088012695 }]` |
| MUT-D | drop the `chosen` argument inside the filter (`members.len() < 2` only) — the OPPOSITE-direction defect: an *eligible* combination's count never enters `maxCount` | **FAILS** (below) |
| MUT-B | route `phantom_max` **through** the filter (a real C-16 violation) | **PASSES — inert by construction on this pool, see below** |
| MUT-C | drop `phantom_max` from the composition entirely | **PASSES — outside this detector's scope, see below** |

**MUT-D**, verbatim — and note it reddens the *vacuity* guard, i.e. the device stops choosing
combinations at all, which is a strictly visible symptom:

```
[device-ctr-eligible-max-diff] device: 5 CTR splits (0 >=2-member) | cpu: 9 CTR splits (4 >=2-member) | device grows = 5, cpu device-grows = 0 (begins 1 / accepted 0)
thread 'an_ineligible_combination_must_not_inflate_the_level_0_max_bucket_count' (449915) panicked at crates/cb-train/tests/device_ctr_eligible_max_diff_test.rs:628:9:
both arms must contain >=1 COMBINATION (>=2-member) CTR split (device 0, cpu 4); without one the combination column is materialized but never exercised. NEVER weaken this guard — raise `iterations` instead.
test result: FAILED. 0 passed; 1 failed; …
```

⇒ the `chosen` list is threaded **behaviourally**, not just structurally: under-counting
`maxCount` at the levels where a combination IS eligible suppresses every combination split
the CPU takes. Together, MUT-A and MUT-D pin the filter in **both** directions.

**Why MUT-B passes, and why that is not test blindness.** T18 §3's MUT-5 saw the same green
and this task can now say why. The mutated call passes the two-element array
`[eligible_max, phantom_max]` against `cs.projection_members`, whose entries 0 and 1 are the
two **simple** columns' member lists (`[0]` and `[1]`; the combination is column 2). Both are
eligible at every level, so the filter admits both array slots and the composed value is
unchanged — the mutation is a no-op **by construction on any pool whose first two CTR columns
are simple**, which is every pool in the repository. C-16's guard remains
`phantom_max_is_folded_in_outside_the_eligibility_filter` (unit) plus
`pass_c_calls_the_filtered_max_and_folds_the_phantom_outside_it` (source scan), both
mutation-proved by T18.

**Why MUT-C passes.** With `depth = 2` and 5 categories, `phantom_max` is at most
`2 leaves × 5 buckets = 10`, against an `eligible_max` of 5 (level 0) or up to 25 (a level
where the combination has become eligible); dropping it moves no winner on this pool. This
detector is deliberately scoped to the **level-0 filter**, which is where D-2 lives; the
phantom composition is a different invariant with its own two guards. Recorded rather than
papered over.

---

## 5. What the shipped test is, and the parameters that must not be "tidied"

`crates/cb-train/tests/device_ctr_eligible_max_diff_test.rs`, one `#[test]`:
`an_ineligible_combination_must_not_inflate_the_level_0_max_bucket_count`.

GLOBALS §2.2 in full: the five allow-attrs; every device-touching item inside
`#[cfg(any(feature = "rocm", feature = "cuda"))] mod device { … }`; **no**
`use cb_backend::CpuBackend`; SKIP by printing on cpu/wgpu (verified — `cargo test -p cb-train
--test device_ctr_eligible_max_diff_test` under default features is `1 passed` and prints the
SKIP line); no `#[ignore]`; `CountingGpu` copied **verbatim** from
`crates/cb-train/tests/device_ctr_gate_test.rs` (**NINTH** copy) with the keep-in-sync note; a
sibling `CountingCpu` wrapping `CpuRefRuntime` so `grown == 0` / `accepted_begins == 0` is an
observation; and the structure-vs-averaging permutation divergence guard.

Assertions, in order:

0. **PERTURBATION (new, and the one that keeps this file honest)** — the exact
   `bucket_count`s are read from `materialize_ctr_feature` and the combination's must
   *strictly exceed* the simple columns' max, **and** the resulting weight band must stay
   above ×1.25. Without this, a later edit that shrinks the cardinalities silently turns the
   file into another byte-identical `ctr_device_combo`-class non-detector.
1. device `grown == iterations`; cpu `grown == 0` and `accepted_begins == 0`.
2. vacuity: ≥1 CTR split and ≥1 ≥2-member CTR split on **both** arms.
3. per-tree split-sequence equality — floats, full `CtrSig` identity, one-hot.
4. leaf values within ε = 1e-4 (D-07's device-vs-CPU bar; this is a self-oracle).

The module doc carries a **"⚠ Do NOT tidy these parameters"** section naming each load-bearing
knob and *why*: `K = 5` (the 5× ratio — reusing `ctr_device_combo` re-opens R-20),
`max_ctr_complexity = 2` (no combination column otherwise), `one_hot_max_size = 1` (both cat
columns must stay CTR), `N`/`NF`/generator constants/`DATA_SEED` (swept; they place a level-0
float candidate inside the band), `iterations = 5, depth = 2` (verified equivalent at 3/5/6/8/10),
and an explicit note that `model_size_reg` is **not** a knob here because it is not a
`BoostParams` field.

**No fixture was created, read for baselining, or regenerated** (R-12). The pool is generated
in-test from an in-file LCG, so there is no frozen artifact to drift and no upstream
`predictions.npy`; the upstream half of the chain stays
`device_ctr_combo_fit_test` / `ctr_mixed_simple_vs_combo_oracle_test` /
`tensor_ctr_e2e_oracle_test`.

`run_device_tests.sh` gains the binary (roster **28 → 29**, plus the isolated perf lane).

---

## 6. Validation

| command | result |
|---|---|
| `cargo test -p cb-train --no-default-features --features rocm --test device_ctr_eligible_max_diff_test` | **PASS** — the headline block; identical on a second consecutive run |
| `cargo test -p cb-train --test device_ctr_eligible_max_diff_test` (**default cpu**, SKIP path) | **PASS** — `1 passed`, prints `SKIP …: needs rocm/cuda`, no panic, no `#[ignore]` |
| `bash ./run_device_tests.sh` | **PASS — 30 PASS / 0 FAIL** (29 binaries + perf lane; Poisson **8.7×**, no R-13 flake) |
| `… --test device_ctr_fit_test` | **PASS** — **`4.483e-11` UNCHANGED**; `grows = 5` |
| `… --test device_ctr_buckets_fit_test` | **PASS** — **`2.776e-17` UNCHANGED**; `[0]×8`; `grows = 5` |
| `… --test device_ctr_counter_fit_test` | **PASS** — **`1.388e-17` UNCHANGED**; `grows = 5` |
| `… --test device_ctr_btmv_fit_test` | **PASS** — **`2.776e-17` UNCHANGED**; occupancy 15; `grows = 5` |
| `… --test device_ctr_combo_fit_test` | **PASS** — **`2.082e-17` UNCHANGED**; `grows = 5`; 8 splits / 3 combinations |
| `cargo test -p cb-backend --no-default-features --features rocm` | **PASS** — lib `277 passed; 0 failed; 2 ignored` (**identical to T18/T19/T22**) + `4` + `1` + `0` |
| `cargo test -p cb-backend --no-default-features --features rocm --lib gpu_runtime::ctr_eligibility_test` | **PASS** — `5 passed`, incl. T18's source-scan pin over the region this task's comments edit |
| `cargo check -p cb-backend --no-default-features --features rocm` | **PASS**, zero diagnostics |
| `cargo check -p cb-backend --no-default-features --features wgpu` | **PASS**, **0 errors** |
| `cargo check -p cb-compute` / `cargo test -p cb-compute` | **PASS** — `206 + 5 + 1 + 3 + 9`, zero failures |
| `cargo test -p cb-train --lib` | **PASS** — `401 passed; 0 failed` (identical to T19/T22 — this task adds no `cb-train` lib test) |
| `cargo test -p cb-train --lib device_ctr_combo_config_tests` (V-0b) | **PASS** — `8 passed`; **no gate-state row moved** (the gate is untouched) |
| `cargo test -p cb-train --lib boosting_ctr_gate_tests` | **PASS** — `13 passed` (unchanged) |
| `cargo test -p cb-train` (all targets, default cpu) | **PASS** — **112 targets** `ok` (111 at T22 + this binary), **zero `FAILED`** |
| `cargo clippy -p cb-train --no-default-features --features rocm --test device_ctr_eligible_max_diff_test` | **PASS** — the ONLY diagnostic is the transitively-compiled pre-existing `duplicated_attributes`; **zero name the new file** |
| `cargo clippy -p cb-backend --no-default-features --features rocm --lib --tests --keep-going` | **pre-existing FAIL, unrelated** — single `error` = `erasing_op` at `kernels/score_split.rs:374`; **18 warnings**, the same count T17/T18/T19/T22 recorded |
| non-comment `git diff` filter on `gpu_runtime/{mod.rs, ctr_eligibility_test.rs}` | **empty** — both edits are comment-only |
| `grep -rn "MUTATION-R20\|CB_R20_UNFILTERED" crates/ --include=*.rs` | **empty** |

**D-04 held exactly** — §3.1's baseline is byte-identical to the post-change run.

### Pre-existing, NOT chased — nothing new discovered

* `clippy::erasing_op` at `kernels/score_split.rs:374` — still the only `error` on the rocm
  clippy lane; still 18 warnings.
* `clippy::duplicated_attributes` in `gpu_runtime/mod.rs` — unmoved (this task's edits are
  comments only, and they neither add nor remove executable lines above it).
* `clippy::type_complexity` at `device_ctr_gate_test.rs:140`, the 12 `cb-train` integration-test
  clippy targets, and the default-cpu `cb-backend --lib` 59–60 `kernels::*` failure range —
  all untouched.
* `T22-OBS-1` / `T22-OBS-2` — still unowned, still not chased (user ruling: record only). The
  shipped configuration sits at 5 iterations with **zero** CTR-free trees, i.e. strictly below
  `T22-OBS-1`'s boundary, and uses `combinations_ctr = Borders` with a single column per
  projection, so `T22-OBS-2`'s `b=0`/`b=1` tie is structurally absent.

---

## 7. Deviations and judgements

1. **The search used a temporary production env gate.** Bypassing D-2 to *search* requires
   editing production; a `CB_R20_UNFILTERED` gate was live only while the scratch harness ran
   (never during any suite-wide command), then replaced by the plain un-wired expression for
   the Red, then reverted. Both the gate and the scratch harness are gone.
2. **A synthetic in-test pool rather than a new fixture.** The task brief prefers it, R-12
   forbids regenerating frozen fixtures, and the whole point is that *no* frozen corpus has a
   large enough `maxCount` ratio. The cost is that this file is a device-vs-CPU self-oracle
   with no upstream arm — stated in its module doc, with the upstream chain named.
3. **The brief's `model_size_reg` lever was checked and rejected as unreachable** (§1), not
   silently skipped. Making it a `BoostParams` field would have been a production API change
   well outside this task.
4. **Four mutations rather than one.** MUT-D was added because a filter can be broken in two
   directions and MUT-A only covers one; MUT-B/MUT-C were driven to decide *why* they pass
   (inert-by-construction vs out-of-scope) rather than recording an ambiguous green — the
   discipline T18 §3 established.
5. **Comment-only production edits in `cb-backend`** (three doc/comment blocks). Not strictly
   required to close R-20, but leaving three in-source paragraphs asserting "R-20 is STILL
   OPEN … do not read any green in this phase as closing R-20" would have been actively
   misleading to the next reader. Executable surface unchanged, proved twice.
6. **`run_device_tests.sh` edited** — the phase's C-8 owner (T24) is finished, so the file is
   this task's, and an unregistered device binary is a DoD failure by that file's own rule.

## 8. Rollback

Delete `crates/cb-train/tests/device_ctr_eligible_max_diff_test.rs` and its
`run_device_tests.sh` entry (roster back to 28). The `cb-backend` comment updates are
independent; reverting them would restore text claiming R-20 is open, which would then be
false, so they should stay unless the detector itself is removed.
