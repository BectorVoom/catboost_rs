---
title: Device CTR Coverage P1 — CTR type and projection completeness
status: draft
format: markdown
spec_version: 1
updated_at: 2026-08-08T00:00:00Z
source_requirements:
  - "User request 2026-08-08: implement all CTR on device"
  - "User ruling Q1 2026-08-08: scope = everything (all nine surviving CTR clauses)"
  - "User ruling Q4 2026-08-08: fix the CPU BTMV normalization gap in this phase"
  - "User ruling 2026-08-08: three sequential phases; spec+plan P1 now"
  - ".planning/plans/device-ctr-full-coverage/research.md (2441 lines, two passes)"
---

# Device CTR Coverage P1 — CTR type and projection completeness

## 1. Context

`cb_train::train` decides **per fit** whether the whole boosting loop commits to
the GPU or falls back to the byte-unchanged CPU path (decision D-10-01). For fits
carrying materialized CTR features, one predicate gates that choice:
`ctr_types_are_device_covered` (`crates/cb-train/src/boosting.rs:2355-2366`),
which today admits only

```rust
col.projection.is_simple()
    && col.ctr_type == ECtrType::Borders.as_i8()
    && col.target_border_idx == 0
    && col.prior_denom == 1.0
```

Every other CTR configuration the CPU trainer supports declines to CPU. This
phase removes all four conjuncts.

Two findings from the research pass reshape the work and are load-bearing for
this spec:

**FPP-11 is root-caused and measured.** The escalated doc comment
(`boosting.rs:2317-2347`) attributed the combination-CTR divergence to
`eligible_max` and to `bucket_count` semantics. The actual primary cause was
unnamed: upstream's `AddTreeCtrs` builds combination projections only from
`{binary+one-hot tree} ∪ currentTree.GetUsedCtrs()` and skips every
`baseProj.IsEmpty()`, so **at level 0 of every tree no combination CTR candidate
exists** (`greedy_tensor_search.cpp:491-551`, called per level). This repo's CPU
mirrors it (`tree.rs:2896-2909`, `:3134-3145`); the device's pass C scores every
column at every level ungated (`gpu_runtime/mod.rs:4919`). A scratch spike
measured all three arms on `ctr_device_combo/`:

| Arm | max\|Δpred\| vs upstream | Verdict |
|---|---|---|
| Control (today's device behaviour, arity gate opened) | 2.746e-2 | FAIL |
| D-1 only (per-level eligibility gate) | 2.082e-17 | PASS |
| D-1 + D-2 (+ filtered `eligible_max`) | 2.082e-17 | PASS |

The full 23-binary device suite stayed green under the fix; the worktree was
reverted afterwards (HEAD `a0a67ec`, clean). `2.082e-17` differs from the
CPU-fallback `1.388e-17`, which is independent evidence the device path ran.
Suspect 2 (`bucket_count`) is **refuted**; suspect 1 (`eligible_max`) is a
confirmed second defect that this fixture does **not** discriminate.

**Risk R-3 is refuted.** BTMV's f32 quantization was believed inexpressible as a
`value > border` comparison. A spike over 4,504,501 `(good, total)` pairs per
prior found **0 mismatches** between the f64 border table, an f32 border table,
upstream `CalcCTR` and the repo's BTMV quantizer for every prior in `[0,1]`.
Track B therefore needs **no new quantize kernel** — only an accumulator.

### Phase position

P1 of three sequential phases (user ruling). P2 = CTR reachability
(`border_count != 15`, cat-only pools, bootstrap/MVS/sampling composition
**including** `leaf_estimation_method=Exact` per user override of R-18).
P3 = CTR × multi-permutation (gated on the *anchored* `pc=4, seed=0` cycle) and
eval sets. One-hot×CTR and CTR×non-symmetric/Region are CPU-trainer features,
not device coverage, and get their own specs after P3.

## 2. Scope and non-goals

### In scope

| ID group | Work |
|---|---|
| Seam | `DeviceCtrColumn` carries `ctr_type`, `target_border_idx`, `projection_members` |
| Track 1a | Delete the provably-dead `prior_denom == 1.0` conjunct |
| Track E | Fix the CPU BTMV `(ctr + shift)/norm` gap (must precede Track B) |
| Track A | `ECtrType::Buckets` on device, incl. `target_border_idx ∈ {0,1}` |
| Track C | `ECtrType::Counter` on device |
| Track B | `ECtrType::BinarizedTargetMeanValue` on device |
| Track D | Combination (≥2-member) projections on device (D-1 + D-2) |
| Hygiene | `CountingGpu` device-commit assertions on the two CTR e2e tests |

### Non-goals (P1)

- `border_count != 15`, cat-only pools, bootstrap/MVS/sampling/Exact composition
  → **P2**.
- Multi-permutation CTR, eval sets, Counter `Full` eval widening → **P3**.
  `counter_calc_method=Full` is *structurally moot* on device in P1 (its only
  effect flows from `eval_sets`, which `device_host_eligible` requires to be
  empty) and P1 pins that with a negative test rather than implementing it.
- One-hot × CTR (`SPEC-OH-26`) and CTR × non-symmetric/Region → separate
  **CPU-side** specs after P3. Their device clauses are dead code today.
- `FloatTargetMeanValue` / `FeatureFreq` — upstream's CPU task type rejects them
  (`restrictions.h:20-32`), so no parity surface exists.
- Multi-descriptor `simple_ctr` (e.g. `[Borders, Counter]`) — the CPU trainer
  cannot express it (SPEC-CTRT-19 raises).
- wgpu — WR-02, no f64 channel; an f32 CTR path would break the ≤1e-5 bar.
- Any change to the CPU training path other than **Track E**.

## 3. Dependencies

| Dependency | Version / location | Role |
|---|---|---|
| `catboost` | 1.2.10 (installed) + `v1.2.10` tag | Oracle ground truth. `catboost-master/` in-repo is a stale 3-file stub from a different revision and is **not** a reference. |
| CubeCL | workspace pin | Device kernels; generics-float mandatory |
| ROCm | `/home/user/rocm/opt/rocm`, `gfx1151` | Verification rig; `/dev/kfd` present |
| `cb-compute` | `crates/cb-compute` | Backend-agnostic seam types; must stay `cubecl`-free |
| `cb-backend` | `crates/cb-backend` | Kernels + `GpuTrainSession`; must never depend on `cb-train` |
| `cb-train` | `crates/cb-train` | Gate, CTR materialization, boosting loop |

## 4. Typed contracts

### 4.1 Seam extension

```rust
// crates/cb-compute/src/runtime.rs — DeviceCtrColumn (existing fields elided)
pub struct DeviceCtrColumn {
    pub member_bins: Vec<Vec<u32>>,
    pub prior: f64,
    pub borders: Vec<f64>,
    pub bucket_count: usize,
    pub weight_group: u32,
    // ADDED (P1):
    pub ctr_type: i8,              // ECtrType discriminant: 0 Borders, 1 Buckets, 2 BTMV, 4 Counter
    pub target_border_idx: u32,    // Buckets numerator selector; 0 for every other type
    pub projection_members: Vec<u32>, // SORTED cat-feature ids; len() >= 1
}
```

`prior` stays a single scalar: `prior_num / prior_denom` is already correct for
both the value and the border table, and `prior_denom` is provably always `1.0`.

### 4.2 Kernel numerator contract (Track A)

For bucket history `counts[0..C]` read **before** increment, with `total = Σ counts`:

```
Buckets @ b  =>  good = counts[b]
otherwise    =>  good = total − Σ_{c ≤ b} counts[c]
```

At binclf (`SIMPLE_CLASSES_COUNT == 2`) this yields `Borders@0 → n1`,
`Buckets@0 → n0`, `Buckets@1 → n1`.

### 4.3 Counter contract (Track C)

```
totals[b]   = #{obj : bin[obj] == b}            // whole learn set, permutation-independent
denominator = max_b totals[b]                    // CONSTANT across objects
value[obj]  = (totals[bin[obj]] + prior) / (denominator + 1)
```

### 4.4 BTMV contract (Track B)

Per bucket `(sum: f32, count: u32)`, read before adding
`class as f32 / (classes − 1).max(1)`; at binclf the added value is exactly
`class as f32 ∈ {0.0, 1.0}`. **The f32 accumulation width is load-bearing** —
`TCtrMeanHistory::Sum` is `float` upstream and the repo pins this
(`online_test::btmv_sum_is_accumulated_in_f32_not_f64`). An f64 device sum
diverges.

### 4.5 Corrected CPU BTMV quantizer (Track E)

```rust
let (shift, norm) = calc_normalization(prior_scalar);   // f64 -> cast to f32
let denom = (total as f32) + 1.0f32;                    // rename: this is NOT calc_normalization's norm
let ctr   = (good as f32 + prior_scalar as f32) / denom;
let bin_f = f64::from(((ctr + shift as f32) / norm as f32) * ctr_border_count as f32);
```

### 4.6 Combination eligibility predicate (Track D)

```
eligible(col, chosen_this_tree) =
    col.projection_members.len() < 2
    || ∃ q ∈ chosen_this_tree :
         q.len() + 1 == col.projection_members.len() && q ⊆ col.projection_members
```

`chosen_this_tree` is **tree-lifetime** (reset per tree), distinct from the
existing **fit-lifetime** `group_used` bookkeeping. Conflating the two scopes is
a specified failure mode.

### 4.7 Final gate

```rust
fn ctr_types_are_device_covered(cols: &[crate::ctr::CtrFeatureColumn]) -> bool {
    !cols.is_empty()
        && cols.iter().all(|col| {
            crate::ctr::ECtrType::from_i8(col.ctr_type)
                .is_some_and(|t| t.is_cpu_supported())
        })
}
```

`from_i8` / `is_cpu_supported` already exist (`ctr/mod.rs:96-108, 155-160`) —
reuse them; do not hand-roll a second type list.

## 5. Failure-isolated behavioral specifications

Each has one primary failure reason.

### DCTR-01 — Seam carries per-column CTR identity
**Given** a fit with materialized CTR columns, **when** `build_device_ctr_config`
runs, **then** each `DeviceCtrColumn` carries `ctr_type`, `target_border_idx` and
sorted `projection_members` copied from its `CtrFeatureColumn`.
**Invariant**: device output on the existing `ctr_device_mixed` path is
**byte-identical** to before (pure plumbing).
**Failure mode**: wrong/unsorted members ⇒ Track D's subset test is wrong.

### DCTR-02 — `prior_denom` conjunct removal is a proven no-op
**Given** the only production materialization site hard-codes
`CTR_PRIOR_DENOM = 1.0`, **when** the conjunct is deleted, **then** no fit's
routing changes. Pinned by a characterization test asserting the constant is
`1.0` and that upstream forbids `denom != 1` (`ctr_helper.cpp:50`).

### DCTR-03 — One-hot × CTR keeps declining (defense in depth)
**Given** a pool mixing one-hot columns and CTR features, **then** the fit does
**not** commit to the device.
**Spec decision, diverging from the research suggestion**: the research proposed
deleting the `one_hot_bins.is_empty()` device conjunct as dead code (SPEC-OH-26
rejects the mix earlier, in `train_inner`). This spec **retains** it and adds the
negative test instead, because deleting it would mean that the day the CPU gains
the three-way candidate union, one-hot×CTR silently reaches the device with zero
device testing. Retention costs nothing and removes a latent hazard.

### DCTR-04 — CPU BTMV applies `(ctr + shift)/norm`
**Given** a BTMV column with prior `p`, **when** the bin is computed, **then** it
equals `trunc(((ctr + shift)/norm) · bc)` with `(shift, norm) = calc_normalization(p)`,
all in f32. **Failure mode**: out-of-`[0,1]` priors bin wrongly (at `p = 2.0`,
`good=0, total=0`, the uncorrected form gives 30 where 15 is correct).

### DCTR-05 — Track E is inert for every committed artifact
**Given** `calc_normalization(p) == (0.0, 1.0)` for all `p ∈ [0,1]`, and every
prior in the repository is in `[0,1]`, **then** `ctr_btmv_simple_oracle_test`
stays green **unchanged** and no fixture is re-baselined.

### DCTR-06 — Device Buckets numerator matches CPU
Kernel self-oracle: for `(ctr_type, target_border_idx)` ∈ {(Borders,0),
(Buckets,0), (Buckets,1)}, device `good`/`total` match `online_class_prefix`
exactly. **Failure mode**: numerator selection.

### DCTR-07 — Buckets plumbing and grouping
`ctr_covered` accepts Buckets; `build_ctr_cindex_columns` dispatches on
`ctr_type`; the two columns per (projection, prior) share one `weight_group`
keyed `(ctr_type, projection)` and carry the same `bucket_count`.

### DCTR-08 — Buckets e2e ≤1e-5 with device commitment
Fixture `ctr_device_buckets`. **Both** assertions required: `max|Δpred| ≤ 1e-5`
vs upstream **and** `CountingGpu.grown.get() == params.iterations`. The fixture
must assert both `target_border_idx` values 0 and 1 appear in `model.json`.

### DCTR-09 — Device Counter statistic
Kernel self-oracle vs `online_counter_column`: whole-set tally, constant max
denominator, and **permutation independence** (shuffling the permutation must
not change any value).

### DCTR-10 — Counter e2e ≤1e-5 with device commitment
Fixture `ctr_device_counter`; prior pinned explicitly on both sides (upstream's
Counter default prior is `0/1`, not `0.5`).

### DCTR-11 — `counter_calc_method=Full` + eval set still declines
Negative test: the fit routes to CPU (via the `eval_sets.is_empty()` clause), so
P1 ships no eval-widening code. Pins the P1/P3 boundary.

### DCTR-12 — Device BTMV accumulator
Kernel self-oracle vs `online_mean_prefix`, with the **f32** sum width asserted
(an f64 sum must fail this test).

### DCTR-13 — BTMV ≡ Borders@0 at binclf
Structural cross-check: with the same prior and fixture, the device BTMV path and
the device Borders path produce identical bins. Justifies choosing the honest
accumulator (option b) over aliasing (option a) while proving they agree today.

### DCTR-14 — BTMV e2e ≤1e-5 with device commitment
Fixture `ctr_device_btmv`. Depends on **DCTR-04** having landed first.

### DCTR-15 — Per-level combination eligibility gate (D-1)
**Given** a tree at level `L` with `chosen_this_tree`, **when** pass C enumerates
CTR columns, **then** every column failing §4.6 is skipped **before scoring**, so
the strict-`>` first-wins order over survivors matches the CPU's.
**Invariant**: at level 0, `chosen_this_tree` is empty ⇒ no ≥2-member column is
scored. **Failure mode**: tree-lifetime vs fit-lifetime scope confusion.

### DCTR-16 — `eligible_max` uses the same filter (D-2)
`eligible_max` maxes `bucket_counts` over **eligible** columns only. Including an
ineligible combination's larger count inflates `maxCount`, raising
`cat_feature_weight = (1 + count/maxCount)^(-0.5)` for every unused simple
candidate — an independent way to flip the winner.
**Known gap**: the `ctr_device_combo` fixture does **not** discriminate D-2
(D-1 alone already passes). D-2 requires its own targeted test — a unit test on
the filtered max is acceptable and is the specified detector (R-20).

### DCTR-17 — Combination CTR e2e ≤1e-5 with device commitment
`device_ctr_combo_fit_test` is **un-ignored** and rewritten with `CountingGpu`.
Expected `max|Δpred| ≈ 2.082e-17` (measured), not the CPU-fallback `1.388e-17`.
Its current `#[ignore]` rationale is factually wrong and must be removed.

### DCTR-18 — Final gate admits exactly the CPU-supported types
All four conjuncts gone; the predicate delegates to `from_i8`/`is_cpu_supported`.
Positive tests per admitted type, negative tests for `FloatTargetMeanValue` /
`FeatureFreq`.

**Correction (checker pass 1, CRITICAL-1 — independently re-verified).** An
earlier draft of this spec claimed this predicate had "no covering tests found"
(a CodeGraph result). That is **false**: `crates/cb-train/src/device_ctr_combo_config_test.rs`
calls it five times (`:132, :144, :156, :163, :170`) and pins **exactly the four
conjuncts P1 removes**. CodeGraph misses it because the module is `#[path]`-mounted
as an in-crate sibling (`boosting.rs:7406-7408`). Those assertions are a deliberate
guardrail — the file says at `:124-126` that re-opening the gate "must be a
conscious act … not an accident".

⇒ This specification therefore requires a **conscious migration** of those two test
functions (`a_combination_column_set_is_NOT_device_covered_yet`,
`a_non_borders_column_still_declines`) as an explicit, ordered step that lands
**before** the first conjunct deletion — not an incidental edit. The migration must
carry a written justification citing `:124-126`, and every task that mutates the
gate expression must run `cargo test -p cb-train --lib device_ctr_combo_config_tests`.

### DCTR-20 — Combination × non-Borders types are covered before being admitted
**Given** the final gate (DCTR-18) admits any CPU-supported type on any-arity
projection, **then** the combination × {Buckets, Counter, BTMV} cross-product must
have at least one passing detector before that gate ships.

**Rationale (checker MAJOR-1)**: P1's fixtures cover simple × all four types and
combination × **Borders only**. Without this spec the finish line admits a
cross-product that nothing exercises — and the first thing it would exercise is the
`bucket_counts` fallback at `session.rs:1749-1765`, which reads `member_bins.first()`
only and whose comment at `session.rs:1748` still asserts "the gate admits only simple
projections".

**Preferred detector**: a device-grower-vs-CPU-grower **split-sequence differential**
over `ctr_device_combo` for `combinations_ctr ∈ {Buckets, Counter, BTMV}`. It needs
no new upstream fixture, is strictly stronger than a ≤1e-5 prediction comparison
(it compares chosen splits, not just outputs), and it simultaneously gives DCTR-16
(D-2) the discriminating detector that R-20 records as missing.
**Also required**: correct the false comment at `session.rs:1748`.

### DCTR-19 — Existing CTR e2e test proves device commitment
`device_ctr_fit_test` gains a `CountingGpu` assertion. Closes the false-pass class
(R-8) that let `device_ctr_combo_fit_test` pass on CPU while claiming device
coverage.

## 6. Acceptance scenarios

1. A Buckets CTR fit (`simple_ctr=["Buckets:Prior=0.5"]`) commits to the device
   and matches upstream ≤1e-5.
2. A Counter CTR fit commits and matches ≤1e-5.
3. A BTMV CTR fit commits and matches ≤1e-5, against the **corrected** CPU.
4. A combination CTR fit (`max_ctr_complexity=2`) commits and matches ≤1e-5.
5. `ctr_types_are_device_covered` contains no type/arity/target-border/prior
   conjunct.
6. Every CTR e2e test asserts device commitment via `CountingGpu`.
7. `bash ./run_device_tests.sh` is green — at its **grown** roster: the 23
   binaries green at `a0a67ec` **plus** every binary this phase adds, all
   registered in the script's `TESTS=(…)` array. (An earlier draft said "23/23",
   which would have let the phase's own new tests go unregistered — checker
   MINOR-8.)
8. `ctr_btmv_simple_oracle_test` passes **unchanged** (Track E no-op proof).
9. One-hot×CTR, multi-permutation, eval-set, cat-only and `border_count != 15`
   fits still decline, each with a passing negative test.
10. A combination × non-Borders detector passes (DCTR-20) before the final gate
    lands.

## 7. Impact scope

| Item | Class | Note |
|---|---|---|
| `boosting.rs` `ctr_types_are_device_covered` + doc block | cross-module | 1 caller; **no covering test today** |
| `cb-compute` `DeviceCtrColumn` / `DeviceCtrConfig` | cross-module | 8 / 9 callers; 3 test files construct them |
| `boosting.rs` `build_device_ctr_config` | local | populates the new fields |
| `cb-backend` `ctr_device.rs` kernels | local | numerator mode; Counter mode; BTMV accumulator |
| `cb-backend` `session.rs` `ctr_covered`, `build_ctr_cindex_columns`, `CtrSearchState` | cross-module | per-type dispatch + shape checks |
| `cb-backend` `gpu_runtime/mod.rs` `ResidentCtrSearch`, pass C, `eligible_max` | cross-module | D-1 + D-2; highest regression risk in P1 |
| `cb-train` `ctr/ctr_feature.rs` BTMV quantizer | local | Track E; CPU behaviour change (proven inert) |
| `crates/cb-oracle/fixtures/ctr_device_{buckets,btmv,counter}` | new artifacts | frozen; generated from the combo recipe |
| `device_ctr_fit_test`, `device_ctr_combo_fit_test` | test-only | `CountingGpu`; un-ignore |

Classification: **cross-module**, contained within the workspace. No public
Python surface changes (`task_type` is not Python-exposed).

## 8. Compatibility and migration

- No serialized-model format change: CTR *training* routing only; the produced
  model is a normal `.cbm`.
- No Python API change.
- Track E changes CPU BTMV binning for priors outside `[0,1]`. Proven inert for
  every committed artifact and every default prior; no fixture re-baseline.
- Rollback is per-track: each track's gate widening is one conjunct, revertible
  independently. Kernel mode parameters default to today's behaviour.

## 9. Risks and open questions

| ID | Risk | Mitigation |
|---|---|---|
| R-1 | D-1 needs projection structure across the seam + tree-lifetime state in the resident grower | DCTR-01 lands first; DCTR-15 names the scope-confusion failure mode |
| R-20 | `ctr_device_combo` does **not** discriminate D-2 | DCTR-16 mandates a dedicated unit test; **DCTR-20's differential is the stronger detector and supersedes the unit test as primary evidence** |
| R-21 | The gate's existing five covering assertions (`device_ctr_combo_config_test.rs`) are a deliberate "conscious act" guardrail that four conjunct deletions would break as a side effect | DCTR-18 mandates an ordered, justified migration **before** the first deletion, and gate-mutating tasks must run that binary |
| R-22 | CodeGraph's test attribution misses `#[path]`-mounted in-crate sibling modules, so "no covering tests found" is not evidence of absence | Any "no covering tests" claim must be confirmed with a repo-wide `grep` before it is used to justify an edit |
| R-8 | Device tests can pass on CPU | `CountingGpu` mandatory (DCTR-08/10/14/17/19) |
| R-7 | f32-vs-f64 `CalcCTR` width: 899 mismatches at `prior = −0.5` | Pre-existing CPU question, out of P1; no repo artifact uses a negative prior |
| — | Track B option (a) alias vs (b) accumulator | Spec selects **(b)**, with DCTR-13 proving they agree at binclf |
| — | Counter kernel shape: serial two-pass vs parallel atomics | Spec selects **serial two-pass** (preserves D-06, no new kernel); revisit only on profiling evidence |

**Open questions**: none blocking P1. Both first-pass blockers (Q2 BTMV
quantizer, Q5 FPP-11) were closed by measurement; Q1 and Q4 by user ruling.

Carried to P2: the user **overrode** the recommendation to exclude
`leaf_estimation_method=Exact` from CTR composition (R-18). P2's spec must carry
R-18's verification burden explicitly.

## 10. Traceability and sources

- Research: `.planning/plans/device-ctr-full-coverage/research.md` (2441 lines;
  two passes; confidence HIGH 16 / MED 4 / LOW 5 first pass plus a second-pass
  delta).
- Prior art: `.planning/plans/gpu-device-coverage-expansion/` — the phase that
  shipped today's device CTR support.
- Upstream v1.2.10 (fetched raw): `online_ctr.{h,cpp}`, `ctr_helper.{h,cpp}`,
  `greedy_tensor_search.cpp`, `restrictions.h`, `cat_feature_options.cpp`.
- Spikes (this machine, `gfx1151`): FPP-11 three-arm measurement; Q2 BTMV
  quantizer equivalence over 4.5M pairs; Q4 inertness verification.
- Baseline: `bash ./run_device_tests.sh` → 23/23 PASS, HEAD `a0a67ec`, worktree
  clean.
- **TreeFinder**: this repository has no TreeFinder-indexed specification corpus;
  its specification store is the plain-file `.planning/plans/<slug>/` set. This
  document is therefore authored directly as `SPEC.md`. No TreeFinder identifier
  is pending.

### Decisions to preserve

D-04 (byte-unchanged CPU path for anything not newly admitted), D-05, D-06
(CTR values never touch the host), D-07, D-10-01 (one per-fit all-or-nothing
decision), WR-02 (wgpu rejected), SPEC-OH-26 (one-hot × CTR stays forbidden),
SPEC-CTRB-01 (`bin_border ≤ 15`). Plus: source/test separation (no
`#[cfg(test)] mod` in production files), no `unwrap` in production,
generics-float for kernels, `cb-backend` never depends on `cb-train`,
`cb-compute` stays `cubecl`-free.

### The honest-gate discipline

When a configuration cannot be made upstream-faithful, **decline with a
documented, negatively-tested clause** rather than ship a silently-approximate
device result. That discipline produced the FPP-11 escalation; DCTR-03 and
DCTR-11 apply it here.
