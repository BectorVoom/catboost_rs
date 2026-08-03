---
title: TDD implementation plan — Part 1 waves W2–W3 (tasks E06–E17)
parent: ./PLAN.md
spec: ./SPEC.md
status: ready-for-implementation
---

# Part 1 — Engine, waves W2–W3

Continuation of `./PLAN.md`. §3 (shared conventions), §3.1 (mutation-check
protocol), §3.2 (repository-verified commands) and §4 (waves + edge list) of that
document apply verbatim to every task here and are not repeated.

---

## WAVE W2 — Type routing becomes real (still ONE prior)

> After W2, `params.simple_ctr` / `params.combinations_ctr` are genuinely honored
> and Counter / Buckets / BTMV train **and predict**. Prior selection follows
> `is_simple`. Candidate expansion (multi-prior, multi-`target_border_idx`) is
> deliberately still absent — that is W3, the highest-risk wave.

---

### E06 — Counter whole-set producer (NOT a prefix)

- **Specs:** SPEC-CTRT-08 (unit half)
- **Blocked by:** E05. **Blocks:** E07.
- **Parallelizable:** **NO** — owns `crates/cb-train/src/ctr/online.rs`.

**Goal / observable completion condition.** A pure
`online_counter_column(bins: &[u32], bucket_count: usize) -> (Vec<i64>, i64)`
returning each document's **whole-set** bucket count and the **constant** MAX
bucket total, unit-tested for **permutation invariance** — the property that
distinguishes Counter from every prefix type.

**Files**
- Modify: `crates/cb-train/src/ctr/online.rs`, `crates/cb-train/src/ctr/online_test.rs`
- Modify: `crates/cb-train/src/ctr/mod.rs` (`pub use online::{…}` at `:144-148`),
  `crates/cb-train/src/lib.rs:46-50`

**Exact verified files/symbols to touch**
- `OnlineCtrAccumulator.total_counts: Vec<i64>` already holds the whole-set
  per-bucket totals `[VERIFIED: LOCAL crates/cb-train/src/ctr/online.rs:133-135, filled at :220-222]`.
  E06 adds the **column** producer, not a new histogram.
- `build_final_ctr`'s Counter arm already computes
  `counter_denominator = acc.total_counts.iter().copied().max().unwrap_or(0)`
  `[VERIFIED: LOCAL crates/cb-train/src/ctr/final_ctr.rs:111-116]`. The ONLINE
  denominator is the SAME rule (`online_ctr.cpp:934-936`), so the producer reuses
  that expression rather than inventing a second one.
- `calc_ctr_online(cic, tot, prior) = (cic + prior) / (tot + 1)`
  `[VERIFIED: LOCAL crates/cb-train/src/ctr/calc_ctr.rs:76-79]` — **unchanged**.
  Counter supplies `(bucket_total, MAX)` as inputs; the hard `+1` is shared by
  every online arm including Counter (research §A.0/§A.3).

**CodeGraph evidence.** `calc_ctr_online` (`calc_ctr.rs:76`) has **10 callers**
with covering tests `calc_ctr_test.rs`, `ordered_ctr_oracle_test.rs`,
`plain_ctr_oracle_test.rs`, `tensor_ctr_oracle_test.rs`
`[VERIFIED: CODEGRAPH, research §F.1]`. E06 adds **no** caller and changes **no**
existing one — the quantizer is type-agnostic and stays untouched.

**Red**
- File: `crates/cb-train/src/ctr/online_test.rs`
- Test fn 1: `counter_column_is_the_whole_set_bucket_total_over_the_max_bucket`
  Input `bins = [0,0,0,1,1,2]`, `bucket_count = 3`.
  Expected: per-document totals `[3,3,3,2,2,1]` (each document's own row IS counted
  — Counter is not read-before-increment); denominator `3`.
- Test fn 2 (**the distinguishing property**, SPEC-CTRT-08's mandated assertion):
  `counter_column_is_permutation_invariant`
  Same `bins`, evaluated under permutations `[0,1,2,3,4,5]` and `[5,2,0,4,1,3]`.
  Expected `assert_eq!(col_a, col_b)` and `assert_eq!(denom_a, denom_b)` — exact
  equality, no tolerance. Failure message must read
  `"Counter is permutation-INDEPENDENT (IsPermutationDependentCtrType(Counter)==false, ctr_type.cpp:43-56); a prefix implementation would differ here"`.
- Test fn 3: `counter_column_on_empty_bins_is_empty_with_zero_denominator`
  Input `bins = []`, `bucket_count = 0`. Expected `(vec![], 0)` — no panic, no
  downstream division by zero.
- **EXPECTED INITIAL FAILURE:**
  `error[E0425]: cannot find function 'online_counter_column' in module 'super'`.
- Run: `cargo test -p cb-train --lib ctr::online_test -- counter_column`

**Green (minimal implementation intent).** One `#[must_use] pub fn` in
`online.rs` (**`pub`, not `pub(crate)`** — it is re-exported from
`crates/cb-train/src/ctr/mod.rs`; `pub use` of a `pub(crate)` item is
`error[E0365]`): tally `totals[bin] += 1` over `bins` (checked `.get_mut`), take
`max().unwrap_or(0)` as the denominator, map each document to `totals[bins[doc]]`.
**No permutation parameter at all** — the absence of the parameter makes
permutation-invariance structural rather than merely asserted. Doc comment carries
the `online_ctr.cpp:503-562,714-729,934-936` anchors and states that
`counter_calc_method` (which widens the sample range to learn+test) is deliberately
NOT a parameter here and lands in E22.

**Refactor constraints + required regression scope**
- Do NOT touch `accumulate_online`, `online_ctr_prefix_binclf`,
  `online_class_prefix`, or any `calc_ctr*` function.
- Regression scope: `cargo test -p cb-train --lib ctr::` + the 11 CTR oracles.

**Validation**
```bash
cargo test -p cb-train --lib ctr::online_test -- counter_column
cargo test -p cb-train --lib ctr::
# THE FULL 11-TARGET CTR ORACLE SCOPE (PLAN.md §3.2) — all 11, not a subset
cargo test -p cb-train --test plain_ctr_oracle_test --test ordered_ctr_oracle_test \
  --test tensor_ctr_oracle_test --test tensor_ctr_e2e_oracle_test \
  --test s_order_ctr_bins_oracle_test --test ctr_split_scoring_test \
  --test ctr_feature_materialize_test --test multi_permutation_e2e_oracle_test \
  --test multi_permutation_fold_oracle_test
cargo test -p cb-model --test ctr_data_roundtrip_test --test fstr_ctr_oracle_test
cargo clippy -p cb-train --all-targets
```

**Completion evidence.** Three named tests green (permutation invariance
explicitly); 11 CTR oracles untouched and green.

---

### E07 — BinarizedTargetMeanValue prefix producer, `Sum` accumulated in **f32**

- **Specs:** SPEC-CTRT-07 (unit half)
- **Blocked by:** E06. **Blocks:** E08.
- **Parallelizable:** **NO** — owns `crates/cb-train/src/ctr/online.rs`.

**Goal / observable completion condition.** An `online_mean_prefix(permutation,
bins, target_class, classes, prior) -> CbResult<OnlineMeanPrefix>` implementing
read-before-increment over `TCtrMeanHistory { sum: f32, count: i64 }`, with **`Sum`
accumulated in `f32`**, plus a differential test isolating the f32 factor.

**Files**
- Modify: `crates/cb-train/src/ctr/online.rs`, `crates/cb-train/src/ctr/online_test.rs`,
  `crates/cb-train/src/ctr/mod.rs`, `crates/cb-train/src/lib.rs`

**Exact verified files/symbols to touch**
- `TCtrMeanHistory { pub sum: f32, pub count: i64 }` with
  `add(&mut self, target: f32) { self.sum += target; self.count += 1; }`
  `[VERIFIED: LOCAL crates/cb-train/src/ctr/online.rs:94-112]` — **already f32,
  already correct**. This task adds the PREFIX variant; the whole-set arm at
  `online.rs:212-215` (`mean.add(class as f32 / divisor)`) is already right and
  must NOT change.
- Added value = `targetClass / targetBorderCount` where upstream passes
  **`targetClassesCount - 1`** (`online_ctr.cpp:762`) ⇒ `1` for binclf ⇒ the added
  value is exactly `targetClass ∈ {0.0f, 1.0f}` `[VERIFIED: research §A.2]`.
- `calc_ctr_online` reused verbatim for `(Sum, Count)` (`calc_ctr.rs:76-79`).

**CodeGraph evidence.** `TCtrMeanHistory` is consumed in production ONLY by
`build_final_ctr`'s two mean arms (`final_ctr.rs:101-110`) `[VERIFIED: LOCAL]`.
Adding a prefix producer alongside is additive.

**Precedent for the differential test.**
`.planning/plans/one-hot-categorical-training/instrumented-ground-truth/LEARNING_RATE_F32.md`
`[VERIFIED: LOCAL, read]` — it demonstrates the required method: measure the
CONSTANT relative factor an f32-vs-f64 width difference introduces, against real
committed upstream data, and assert the ratio rather than hand-waving a tolerance
(it records `f32(0.1)/0.1 − 1 = 1.4901161193847656e-08`, constant across all eight
leaves, verified to one ulp).

**Red**
- File: `crates/cb-train/src/ctr/online_test.rs`
- Test fn 1: `btmv_prefix_reads_sum_and_count_before_incrementing`
  Input: permutation `[0,1,2,3]`, `bins = [0,0,0,0]`, `target_class = [1,0,1,1]`,
  `classes = 2`, `prior = 0.5`.
  Expected (hand-computed, exact): `sum = [0.0, 1.0, 1.0, 2.0]`,
  `count = [0, 1, 2, 3]`, and
  `value[i] == calc_ctr_online(f64::from(sum[i]), count[i], 0.5)` asserted by
  `to_bits()` equality. Document 0 seeing `(0.0, 0)` is the no-leakage proof.
- Test fn 2 (**the mandated f32 differential**, SPEC-CTRT-07 / risk R2):
  `btmv_sum_is_accumulated_in_f32_not_f64`
  **This is a DIRECT, ALLOCATION-FREE accumulator test — NOT a document fixture.**
  A `2^24 + 1`-document fixture would allocate `sum: Vec<f32>`, `count: Vec<i64>`,
  `value: Vec<f64>` plus the caller's `permutation: Vec<i32>`, `bins: Vec<u32>` and
  `target_class: Vec<usize>` at that length — roughly **600 MB resident for one
  `#[test]`**, against a project where `target/` disk exhaustion and test-binary RSS
  are an active operational hazard. It is forbidden here.
  Setup: seed a single `TCtrMeanHistory { sum: 16_777_216.0f32, count: 16_777_216 }`
  directly (`crates/cb-train/src/ctr/online.rs:94-112`), then call `.add(1.0)` once.
  Expected:
  ```rust
  // f64 reference: what a widened accumulator would produce.
  let f64_reference: f64 = 16_777_216.0_f64 + 1.0_f64;   // == 16_777_217.0
  let f32_reference: f32 = 16_777_216.0_f32 + 1.0_f32;   // == 16_777_216.0 (f32 is saturated here)

  assert_ne!(f64_reference, f64::from(f32_reference),
      "the seed must actually discriminate f32 from f64 — otherwise this test is vacuous");
  assert_eq!(hist.sum.to_bits(), 16_777_216.0_f32.to_bits(),
      "Sum MUST accumulate in f32 to match upstream TCtrMeanHistory::Sum (online_ctr.h:373-376); \
       an f64 accumulation would give {f64_reference}");
  assert_eq!(hist.count, 16_777_217);
  ```
  The `assert_ne!` is the **anti-vacuity guard and it is KEPT**: without it an f64
  implementation passes whenever the seed is too small to discriminate.
  **Scope note (recorded in SPEC.md §7's A2 note):** for binclf the added value is
  exactly `targetClass ∈ {0.0f, 1.0f}`, so f32 and f64 are bit-identical below
  `2^24`. This accumulator test is therefore the **only** gate for the f32
  requirement; E13 test fn 3's fixture-scale differential **cannot** discriminate at
  30 rows and is a **reporting step, not a gate**.
- **EXPECTED INITIAL FAILURE:**
  `error[E0425]: cannot find function 'online_mean_prefix' in module 'super'`.
- Run: `cargo test -p cb-train --lib ctr::online_test -- btmv`

**Green (minimal implementation intent).** One `pub fn` (**`pub`, not
`pub(crate)`** — re-exported from `ctr/mod.rs`; `pub use` of `pub(crate)` is
`error[E0365]`) mirroring
`online_ctr_prefix_binclf`'s loop shape exactly (same `CbError::Degenerate` length
guards, same checked `.get`, same permutation-range errors), with per-bucket state
`Vec<TCtrMeanHistory>`: READ `(elem.sum, elem.count)` → compute
`calc_ctr_online(f64::from(sum), count, prior)` → INCREMENT
`elem.add(class as f32 / divisor)` where
`divisor = classes.saturating_sub(1).max(1) as f32`. Returns a NEW
`pub struct OnlineMeanPrefix { pub sum: Vec<f32>, pub count: Vec<i64>, pub value: Vec<f64> }`.

**Refactor constraints + required regression scope**
- **Constraint (load-bearing):** do NOT reuse `OnlineCtrPrefix` — its
  `good: Vec<i64>` would force an i64 truncation of the f32 sum, which is exactly
  the silent-widening failure this task exists to prevent. A distinct
  `OnlineMeanPrefix` is mandatory.
- Do NOT change `TCtrMeanHistory::add` or the whole-set arm at `online.rs:212-215`.
- Regression scope: `cargo test -p cb-train --lib ctr::` + the 11 CTR oracles.

**Validation**
```bash
cargo test -p cb-train --lib ctr::online_test -- btmv
cargo test -p cb-train --lib ctr::
# THE FULL 11-TARGET CTR ORACLE SCOPE (PLAN.md §3.2) — all 11, not the previous 2
cargo test -p cb-train --test plain_ctr_oracle_test --test ordered_ctr_oracle_test \
  --test tensor_ctr_oracle_test --test tensor_ctr_e2e_oracle_test \
  --test s_order_ctr_bins_oracle_test --test ctr_split_scoring_test \
  --test ctr_feature_materialize_test --test multi_permutation_e2e_oracle_test \
  --test multi_permutation_fold_oracle_test
cargo test -p cb-model --test ctr_data_roundtrip_test --test fstr_ctr_oracle_test
cargo clippy -p cb-train --all-targets
```

**Completion evidence.** Both tests green, including the `assert_ne!` anti-vacuity
guard proving the seeded `TCtrMeanHistory` genuinely discriminates f32 from f64 —
with **no** multi-hundred-megabyte fixture allocated.

---

### E08 — Buckets prefix producer via the E04 generic

- **Specs:** SPEC-CTRT-06 (unit half)
- **Blocked by:** E07. **Blocks:** E09.
- **Parallelizable:** **NO** — owns `crates/cb-train/src/ctr/online.rs`.

**Goal / observable completion condition.**
`online_class_prefix_column(permutation, bins, target_class, classes,
target_border_idx, ctr_type, prior) -> CbResult<OnlineCtrPrefix>` runs the
read-before-increment loop over `Vec<TCtrHistory>` and derives every document's
`(numerator, denominator)` **exclusively** through `online_class_prefix` (E04).

**Files**
- Modify: `crates/cb-train/src/ctr/online.rs`, `crates/cb-train/src/ctr/online_test.rs`,
  `crates/cb-train/src/ctr/mod.rs`, `crates/cb-train/src/lib.rs`

**Exact verified files/symbols to touch**
- `TCtrHistory::new(classes)` / `::increment(class)` / `::total()`
  `[VERIFIED: LOCAL crates/cb-train/src/ctr/online.rs:64-88]`.
- `online_class_prefix` (E04) — the ONLY place the numerator rule lives.
- `online_ctr_prefix_binclf` (`online.rs:263-320`) — the guard block to mirror.

**CodeGraph evidence for ordering.** E04 must precede E08 so the numerator rule
exists in exactly one place; E05 must precede E08 because the Borders-binclf
equivalence is what proves the shared producer did not move the existing oracles.

**Red**
- File: `crates/cb-train/src/ctr/online_test.rs`
- Test fn 1: `buckets_prefix_uses_class_b_numerator_over_the_prefix_total`
  Input: permutation `[0,1,2,3,4]`, `bins = [0,0,0,0,0]`,
  `target_class = [1,0,1,0,1]`, `classes = 2`, `target_border_idx = 1`,
  `ctr_type = Buckets`, `prior = 0.5`.
  Expected (hand-computed prefix `N[1]`): `good = [0,1,1,2,2]`,
  `total = [0,1,2,3,4]`, `value[i] == calc_ctr_online(good[i] as f64, total[i], 0.5)`
  bit-for-bit.
- Test fn 2: `buckets_prefix_at_border_idx_zero_differs_from_border_idx_one`
  Same input at `target_border_idx = 0` ⇒ `good = [0,0,1,1,2]`, plus
  `assert_ne!(good_b0, good_b1)` — the anti-vacuity guard proving
  `target_border_idx` is genuinely read (a hard-coded `0` makes both equal).
- Test fn 3: `class_prefix_column_at_borders_b0_equals_the_binclf_prefix`
  Asserts `online_class_prefix_column(.., 0, ECtrType::Borders, 0.5)` and
  `online_ctr_prefix_binclf(.., 0.5)` produce identical `good`, `total` and
  `value.to_bits()` — the E05 firewall extended to the column level.
- **EXPECTED INITIAL FAILURE:**
  `error[E0425]: cannot find function 'online_class_prefix_column'`.
- Run: `cargo test -p cb-train --lib ctr::online_test -- prefix_column`

**Green (minimal implementation intent).** One `pub fn` (**`pub`, not
`pub(crate)`** — re-exported from `ctr/mod.rs`; `pub use` of `pub(crate)` is
`error[E0365]`) reusing
`online_ctr_prefix_binclf`'s guard block verbatim; per-bucket state
`Vec<TCtrHistory>` sized `classes`; per document READ `hist.n` → call
`online_class_prefix(&hist.n, target_border_idx, ctr_type)` → store
`(num as i64, denom)` and `calc_ctr_online(num, denom, prior)` → INCREMENT
`hist.increment(class)`. No new value function.

**Refactor constraints + required regression scope**
- Read-before-increment order is inviolable.
- `Counter` MUST be rejected here with
  `CbError::Degenerate("Counter is not a class-prefix CTR type; use online_counter_column")`
  — a checked misuse, never a silently wrong column.
- Regression scope: `cargo test -p cb-train --lib ctr::` + **all 11 CTR oracles**.

**Validation**
```bash
cargo test -p cb-train --lib ctr::online_test
cargo test -p cb-train --test plain_ctr_oracle_test --test ordered_ctr_oracle_test \
  --test tensor_ctr_oracle_test --test tensor_ctr_e2e_oracle_test \
  --test s_order_ctr_bins_oracle_test --test ctr_split_scoring_test \
  --test ctr_feature_materialize_test --test multi_permutation_e2e_oracle_test \
  --test multi_permutation_fold_oracle_test
cargo test -p cb-model --test ctr_data_roundtrip_test --test fstr_ctr_oracle_test
cargo clippy -p cb-train --all-targets
```

**Completion evidence.** Three tests green including the `assert_ne!`
border-idx-discrimination guard and the Borders-equivalence extension; all 11 CTR
oracles green.

---

### E09 — `materialize_ctr_feature` dispatches per CTR type

- **Specs:** SPEC-CTRT-06 / -07 / -08 (wiring half)
- **Blocked by:** E08. **Blocks:** E10.
- **Parallelizable:** **NO** — owns `crates/cb-train/src/ctr/ctr_feature.rs` **and
  edits `crates/cb-train/src/boosting.rs`** (the two behavior-preserving call-site
  widenings at `:3238` / `:3274`, below).
  **E09 MUST leave `cb-train` compiling**: Rust has no default arguments, so the
  widened signature forces E09 to update **its two production call sites in
  `crates/cb-train/src/boosting.rs` (`:3238` structure folds, `:3274` averaging
  fold)** in the same task — E09 makes them compile with the pre-change constants
  (`ECtrType::Borders`, `target_border_idx: 0`); **E10 makes them per-candidate**.
  (An earlier revision of this task claimed "its two call sites in `boosting.rs`
  are E10's" — that was WRONG and is deleted: it left `cb-train` un-buildable at
  the end of E09, while E09's own Validation runs `cargo test -p cb-train`.) It
  additionally owns the FOUR
  mechanical `CtrFeatureColumn` literal fixes in `crates/cb-train/src/tree_test.rs`
  and `crates/cb-train/tests/ctr_split_scoring_test.rs`, **plus the THREE
  compile-forced `materialize_ctr_feature` argument additions at
  `crates/cb-train/tests/ctr_split_scoring_test.rs:384`, `:394` and
  `crates/cb-train/tests/s_order_ctr_bins_oracle_test.rs:70`** (see Files) —
  those are compile-forced, not behavioral. **`ctr_split_scoring_test.rs` IS
  re-edited by later tasks** — E11 (three `bake_ctr_table` call sites), E16 (five
  dropped `greedy_tensor_search_oblivious_with_ctr` arguments) and E22 (both the
  `materialize_ctr_feature` and the `bake_ctr_table` sites again) — each strictly
  mechanical and each owned by the task that forces it.

**Goal / observable completion condition.** `materialize_ctr_feature` takes
`(ctr_type: ECtrType, target_border_idx: usize)` and dispatches to E06/E07/E08's
producers; `CtrFeatureColumn` carries `target_border_idx` and the REAL
`ctr_type`, no longer the hard-coded `ECtrType::Borders.as_i8()`.

**Files**
- Modify: `crates/cb-train/src/ctr/ctr_feature.rs`
- Modify: `crates/cb-train/tests/ctr_feature_materialize_test.rs` (existing target)
- Modify: `crates/cb-train/src/boosting.rs` — **mechanical, forced by the widened
  signature, and ONLY this.** Pass the behavior-preserving `ECtrType::Borders` and
  `target_border_idx: 0` at the TWO `materialize_ctr_feature` call sites,
  `crates/cb-train/src/boosting.rs:3238` (structure folds) and `:3274` (averaging
  fold) — 7 arguments on disk today, 9 after this task
  `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:3238, :3274, read verbatim]`.
  **Per-candidate type/prior resolution is E10's and MUST NOT be done here** —
  E09 only makes these two sites compile with today's constants. Nothing else in
  `boosting.rs` may be touched by E09; the `E09->E10` edge already serializes this
  file, so there is no edit conflict.
- Modify: `crates/cb-train/tests/s_order_ctr_bins_oracle_test.rs` — **mechanical,
  forced by the widened signature.** Add the two new arguments
  (`ctr_type: ECtrType::Borders`, `target_border_idx: 0`) to the
  `materialize_ctr_feature` call at
  `crates/cb-train/tests/s_order_ctr_bins_oracle_test.rs:70` — 7 arguments on disk
  today, 9 after this task — preserving today's behavior exactly (`Borders, b=0`
  is the pre-change hard-coded path)
  `[VERIFIED: LOCAL crates/cb-train/tests/s_order_ctr_bins_oracle_test.rs:70-79]`.
  **Mechanical arity update only. CHANGE NO ASSERTION; weakening or deleting any
  assertion is FORBIDDEN** — this file is **one of the eleven SPEC-CTRT-18 oracle
  targets** and carries `assert_s_order_reproduces_bins`, the only pin on
  `Q = S∘P_avg` bit-exactness. It sits in the **MECHANICAL EDITS ONLY** row of the
  per-file diff gate (PLAN.md §3.2; the table in E15/E16), NOT in ZERO DIFF.
- Modify: `crates/cb-train/src/tree_test.rs` — **mechanical, forced by the new
  struct field.** Add `target_border_idx: 0` to the `CtrFeatureColumn` struct
  literals at `tree_test.rs:374` and `tree_test.rs:662`
  `[VERIFIED: LOCAL crates/cb-train/src/tree_test.rs:374, :662]`.
  **CHANGE NO ASSERTION.**
- Modify: `crates/cb-train/tests/ctr_split_scoring_test.rs` — **mechanical, forced
  by the new struct field AND by the new signature.** TWO permitted edit groups in
  this file, and nothing else:
  1. Add `target_border_idx: 0` to the `CtrFeatureColumn` struct literals at
     `ctr_split_scoring_test.rs:41` and `:68`
     `[VERIFIED: LOCAL crates/cb-train/tests/ctr_split_scoring_test.rs:41, :68]`.
  2. Add the two new arguments (`ctr_type: ECtrType::Borders`,
     `target_border_idx: 0`) to the `materialize_ctr_feature` calls at
     `ctr_split_scoring_test.rs:384` and `:394` — 7 arguments on disk today,
     9 after this task — **preserving today's behavior exactly** (`Borders, b=0`
     is the pre-change hard-coded path)
     `[VERIFIED: LOCAL crates/cb-train/tests/ctr_split_scoring_test.rs:384, :394]`.

  **CHANGE NO ASSERTION.** This file is **one of the eleven SPEC-CTRT-18 oracle
  targets** (PLAN.md §3.2); see the per-file diff gate table in E15/E16.
- **NOTE (all compile-forced files):** `CtrFeatureColumn`
  (`crates/cb-train/src/ctr/ctr_feature.rs:69`)
  is **not** `#[non_exhaustive]`, so all four external literals must gain the
  field. The permitted mechanical edits are exactly: **(i)** one
  `target_border_idx: 0` field initializer per `CtrFeatureColumn` literal
  (`tree_test.rs:374`, `:662`, `ctr_split_scoring_test.rs:41`, `:68`) **and (ii)**
  the two new arguments at **every** `materialize_ctr_feature` call site —
  `ctr_split_scoring_test.rs:384`, `:394`,
  `s_order_ctr_bins_oracle_test.rs:70`, and the two production sites
  `boosting.rs:3238`, `:3274` — nothing else. **Weakening or
  deleting any assertion in `ctr_split_scoring_test.rs` or in
  `s_order_ctr_bins_oracle_test.rs` is FORBIDDEN** — both are
  CTR regression oracles, and "fixing" the compile break by removing a construction
  or a call silently removes CTR-split-scoring / S-order bin-reproduction coverage
  during the highest-risk waves.

**Exact verified files/symbols to touch**
- `pub fn materialize_ctr_feature(cat_columns, projection, permutation,
  target_class, prior_num, prior_denom, ctr_border_count) -> CbResult<CtrFeatureColumn>`
  at `crates/cb-train/src/ctr/ctr_feature.rs:124-131`, already
  `#[allow(clippy::too_many_arguments)]` `[VERIFIED: LOCAL, read in full]`.
- **The two hard-codes to remove:** `ctr_type: ECtrType::Borders.as_i8()` at
  `ctr_feature.rs:232` (with the comment "Borders head — the combinations_ctr /
  simple_ctr default family") and the unconditional call to
  `online_ctr_prefix_binclf` at `ctr_feature.rs:204-205` `[VERIFIED: LOCAL]`.
- `CtrFeatureColumn { projection, ctr_type: i8, prior_num, prior_denom,
  bins: Vec<u32>, ctr_value: Vec<f64>, bucket_count: usize }` at
  `ctr_feature.rs:69-94` — gains `pub target_border_idx: usize`.
- The quantization step at `ctr_feature.rs:209-227` (`calc_ctr_online_bin` →
  `trunc()` → clamp into `[0, ctr_border_count]`) is **type-agnostic and stays
  byte-identical**; only its `(good, total)` inputs change per type.
- Counter's inputs come from `online_counter_column` and use the **constant** MAX
  denominator; BTMV's come from `online_mean_prefix` and MUST be quantized in
  **f32** (`calc_ctr_online_bin` is f64 — for the BTMV arm compute
  `(ctr as f32 + shift as f32) / norm as f32 * border_count as f32` and widen at
  the end, mirroring upstream's all-`float` `CalcCTR`, research §A.0 caveat).

**CodeGraph evidence.** `materialize_ctr_feature` is called from exactly **two**
production sites — `crates/cb-train/src/boosting.rs:3238` (structure folds) and
`:3274` (averaging fold) `[VERIFIED: LOCAL grep -n]` — plus the integration test
`crates/cb-train/tests/ctr_feature_materialize_test.rs`. `tree.rs` copies
`column.ctr_type` onto the chosen split already, so no signature change is needed
there `[VERIFIED: research §F.2]`.

**Red**
- File: `crates/cb-train/tests/ctr_feature_materialize_test.rs`
- Test fn: `materialize_emits_the_requested_ctr_type_and_border_idx`
- Setup: one cat column of 6 distinct values over 12 documents, identity
  permutation, `prior = (0.5, 1.0)`, `ctr_border_count = 15`.
- Action / expected, one case per type:
  - `Borders, b=0` → `col.ctr_type == ECtrType::Borders.as_i8()`, and `col.bins`
    and `col.ctr_value` **bit-identical** to the pre-change output (frozen literals
    transcribed from a pre-change run) — the D-04 no-op proof;
  - `Buckets, b=1` → `col.ctr_type == 1`, `col.target_border_idx == 1`, and
    `assert_ne!(bins_buckets_b1, bins_borders_b0)`;
  - `Counter, b=0` → `col.ctr_type == 4`, plus permutation invariance
    (`materialize` under two different permutations yields identical `bins`);
  - `BinarizedTargetMeanValue, b=0` → `col.ctr_type == 2`, `bins` non-constant.
- **EXPECTED INITIAL FAILURE (two distinct compile errors, in five files):**
  1. `error[E0061]: this function takes 7 arguments but 9 were supplied` on the
     `materialize_ctr_feature` call in
     `crates/cb-train/tests/ctr_feature_materialize_test.rs` (the `ctr_type` /
     `target_border_idx` parameters do not exist yet).
  1b. The **same** `error[E0061]` on the two `materialize_ctr_feature` calls in
     `crates/cb-train/tests/ctr_split_scoring_test.rs` at `:384` and `:394`, as
     soon as the widened signature lands. That target does not build until the two
     mechanical argument additions of Files item 2 land. This is compile-forced,
     not behavioral.
  1c. The **mirror** `error[E0061]: this function takes 9 arguments but 7 were
     supplied` on the `materialize_ctr_feature` call in
     `crates/cb-train/tests/s_order_ctr_bins_oracle_test.rs:70`, as soon as the
     widened signature lands. That oracle target does not build until the
     mechanical argument addition lands. Compile-forced, not behavioral.
  1d. The **same** `error[E0061]` on the TWO PRODUCTION call sites
     `crates/cb-train/src/boosting.rs:3238` and `:3274`. Until these land, the
     `cb-train` **library** does not compile and EVERY command in this task's
     Validation block fails at the build step. These two are not optional and are
     not deferrable to E10.
  2. `error[E0063]: missing field target_border_idx in initializer of
     CtrFeatureColumn` — emitted **once per external struct literal** as soon as
     the field is added to the non-`#[non_exhaustive]` `CtrFeatureColumn`
     (`crates/cb-train/src/ctr/ctr_feature.rs:69`): at
     `crates/cb-train/src/tree_test.rs:374`, `:662` and
     `crates/cb-train/tests/ctr_split_scoring_test.rs:41`, `:68`. Both the
     `cb-train` lib test build and the `ctr_split_scoring_test` target fail to
     compile until the four mechanical `target_border_idx: 0` initializers land.
  After the signature, the four initializers **and all five compile-forced argument
  additions** (`ctr_split_scoring_test.rs:384`, `:394`,
  `s_order_ctr_bins_oracle_test.rs:70`, `boosting.rs:3238`, `:3274`) land but
  before the dispatch, the failure becomes
  ``assertion `left == right` failed: left: 0, right: 4`` on the Counter case.
- Run: `cargo test -p cb-train --test ctr_feature_materialize_test`

**Green (minimal implementation intent).** Add the two parameters; `match ctr_type`
into three arms (`Borders | Buckets` → `online_class_prefix_column`;
`BinarizedTargetMeanValue` → `online_mean_prefix` + f32 quantization;
`Counter` → `online_counter_column`); `FloatTargetMeanValue | FeatureFreq` →
`Err(CbError::Unsupported(...))` mirroring E02's wording (defence in depth — E02
already rejects at `train_inner`). Set `ctr_type: ctr_type.as_i8()` and
`target_border_idx`. Keep the combined-hash fold (`:165-196`) and the quantization
clamp (`:209-227`) byte-identical.
Then, in the **same** Green step, make every caller compile again by passing the
pre-change constants `ECtrType::Borders` and `target_border_idx: 0`: the two
PRODUCTION sites `crates/cb-train/src/boosting.rs:3238` and `:3274`, the two test
sites `ctr_split_scoring_test.rs:384`/`:394`, and the oracle site
`s_order_ctr_bins_oracle_test.rs:70`. **`cb-train` MUST compile at the end of E09**
— the per-candidate values at the two `boosting.rs` sites are E10's, and E10
changes only the VALUES passed there (the arguments already exist after E09).

**Refactor constraints + required regression scope**
- Constraint: the `Borders, b=0` path must remain byte-identical — that is what
  keeps `plain_ctr`, `ordered_ctr`, `tensor_ctr`, `tensor_ctr_e2e`,
  `s_order_ctr_bins`, `ctr_split_scoring`, `multi_permutation_*` green.
- Constraint: do NOT change `TProjection::combined_hash` or the first-seen remap.
- Regression scope: **all 11 CTR oracles** + `cargo test -p cb-train --lib ctr::`.

**Validation**
```bash
cargo test -p cb-train --test ctr_feature_materialize_test
cargo test -p cb-train --test plain_ctr_oracle_test --test ordered_ctr_oracle_test \
  --test tensor_ctr_oracle_test --test tensor_ctr_e2e_oracle_test \
  --test s_order_ctr_bins_oracle_test --test ctr_split_scoring_test \
  --test multi_permutation_e2e_oracle_test --test multi_permutation_fold_oracle_test
cargo test -p cb-model --test ctr_data_roundtrip_test --test fstr_ctr_oracle_test
cargo test -p cb-train
cargo clippy -p cb-train --all-targets
```

**Completion evidence.** Four per-type cases green including the frozen
Borders byte-identity literals and the Counter permutation-invariance assertion;
all 11 CTR oracles green. **`git diff crates/cb-train/src/tree_test.rs
crates/cb-train/tests/ctr_split_scoring_test.rs
crates/cb-train/tests/s_order_ctr_bins_oracle_test.rs` shows exactly four added
`target_border_idx: 0` field-initializer lines PLUS the three widened
`materialize_ctr_feature` call sites (`ctr_split_scoring_test.rs:384`, `:394`,
`s_order_ctr_bins_oracle_test.rs:70`) and
NOTHING else** — no assertion added, removed, weakened or reworded.
**`git diff crates/cb-train/src/boosting.rs` shows exactly the two widened
`materialize_ctr_feature` call sites (`:3238`, `:3274`) passing
`ECtrType::Borders, 0` and NOTHING else** — no routing, no prior resolution
(that is E10's). `cargo build -p cb-train` succeeds at the end of this task.

---

### E10 — `ctr_type` and prior selection follow `is_simple`

- **Specs:** SPEC-CTRT-09, SPEC-CTRT-10
- **Blocked by:** E02, E03, E09. **Blocks:** E11.
- **Parallelizable:** **NO** — owns `crates/cb-train/src/boosting.rs` (and
  `tree.rs` for the `target_border_idx` argument).
- **Hand-off from E09 (explicit, so the two tasks do not collide on
  `boosting.rs`):** E09 has ALREADY widened the two `materialize_ctr_feature` call
  sites at `boosting.rs:3238` and `:3274` to the 9-argument form, passing the
  constants `ECtrType::Borders, 0` — **E09 = arity only**. E10 changes only the
  VALUES passed at those two sites (per-candidate `ECtrType` + prior) — **E10 =
  value/routing only**; it adds no argument and removes none. The `E09->E10` edge
  in the authoritative edge list already serializes `boosting.rs`, so this split
  needs no new edge.

**Goal / observable completion condition.** `params.simple_ctr` /
`params.combinations_ctr` and `params.simple_ctr_priors` /
`params.combinations_ctr_priors` are read; a simple candidate uses the simple pair
and a combination candidate the combination pair; `ctr_splits_for_tree` no longer
hard-codes Borders. Still ONE prior per candidate (`.first()`), still
`target_border_idx = 0` — expansion is W3.

**Files**
- Modify: `crates/cb-train/src/boosting.rs`
- Modify: `crates/cb-train/src/boosting_test.rs`
- Modify: `crates/cb-train/src/tree.rs` (only if the
  `greedy_tensor_search_oblivious_with_ctr` `target_border_idx` argument moves —
  in W2 it stays the literal `0`; deferred to E16)

**Exact verified files/symbols to touch**
- `let ctr_prior_num = params.combinations_ctr_priors.first().copied().unwrap_or(0.5);`
  at **`crates/cb-train/src/boosting.rs:3155`** — the site that feeds **both**
  `materialize_ctr_feature` calls (`:3238-3243` structure, `:3274-3279` averaging)
  and the bake (`:5451`) `[VERIFIED: LOCAL grep -n ctr_prior_num]`. Today the
  COMBINATION prior governs SIMPLE candidates too — the bug SPEC-CTRT-10 fixes.
- `absolute_projections` at `boosting.rs:3163-3175` is built from `ctr_candidates`
  and is index-aligned with them, so `ctr_candidates[i].is_simple` is available at
  every materialization site without a new lookup `[VERIFIED: LOCAL, read]`.
- `fn ctr_splits_for_tree(candidates, priors) -> Vec<CtrSplitSpec>` at
  `boosting.rs:1929-1949`, hard-coding `ctr_type: ECtrType::Borders.as_i8()` at
  **`:1940`**; its ONE caller is `boosting.rs:5318`
  `[VERIFIED: LOCAL + CODEGRAPH "1 caller"]`.
- `CtrCandidate { projection, is_simple }` `[VERIFIED: CODEGRAPH candidates.rs:151-157]`,
  derived from `TProjection::is_simple()`.
- `BoostParams` fields `simple_ctr` (`:261`), `simple_ctr_priors` (`:267`),
  `counter_calc_method` (`:272`), `combinations_ctr` (`:298`),
  `combinations_ctr_priors` (`:304`) `[VERIFIED: research §F, Part-2 PLAN T03/T04/T05]`.
- Stale doc comment to fix opportunistically:
  `crates/cb-train/src/boosting.rs:816-817` — "EMPTY for every path that emits no
  one-hot candidate — which is all of them until T19 populates it" is now FALSE
  (the device grower emits one-hot levels) `[VERIFIED: LOCAL sed -n '805,825p']`.

**CodeGraph evidence for ordering.** `ctr_splits_for_tree` has **⚠️ no covering
tests** `[VERIFIED: CODEGRAPH]` — E03 is a hard prerequisite. E09 is a hard
prerequisite because `materialize_ctr_feature` must already accept `ctr_type`.
E02 is a prerequisite so an illegal type is rejected before it can reach this
routing.

**Red**
- File: `crates/cb-train/src/boosting_test.rs`
- Test fn 1: `ctr_splits_for_tree_routes_type_and_prior_by_is_simple`
  Setup: `ctr_splits_for_tree` gains `(simple_ctr, simple_priors,
  combinations_ctr, combinations_priors)` (or a `&BoostParams`); candidates
  `[{proj [0], is_simple: true}, {proj [0,1], is_simple: false}]`; params
  `simple_ctr = Counter`, `simple_ctr_priors = [0.25]`,
  `combinations_ctr = Buckets`, `combinations_ctr_priors = [0.75]`.
  Expected: `specs[0].ctr_type == ECtrType::Counter.as_i8() (4)` and
  `specs[0].prior_num == 0.25`; `specs[1].ctr_type == ECtrType::Buckets.as_i8() (1)`
  and `specs[1].prior_num == 0.75`.
- Test fn 2: `ctr_splits_for_tree_defaults_are_byte_identical_to_the_e03_characterization`
  Re-asserts E03's frozen expectations under `simple_ctr = combinations_ctr =
  Borders`, `both priors = [0.25, 0.75]` ⇒ every spec `ctr_type == 0`,
  `prior_num == 0.25` — the D-04 no-op proof.
- **EXPECTED INITIAL FAILURE:** test fn 1 —
  ``assertion `left == right` failed: left: 0, right: 4`` (today `ctr_type` is the
  hard-coded `ECtrType::Borders.as_i8()` at `boosting.rs:1940`), and a second
  failure `left: 0.75, right: 0.25` on the simple candidate's prior (today both
  candidates read `combinations_ctr_priors.first()`).
- Run: `cargo test -p cb-train --lib boosting::tests -- ctr_splits_for_tree`

**Green (minimal implementation intent).**
1. Widen `ctr_splits_for_tree` to take the four values (or `&BoostParams`) and
   select per `candidate.is_simple`; update its ONE caller at `:5318`.
2. Replace the single `ctr_prior_num` at `:3155` with a per-candidate resolution
   helper `fn ctr_config_for(params, is_simple) -> (ECtrType, f64)` returning the
   `(type, head prior)` pair, called inside the two materialization loops
   (`:3237-3247`, `:3273-3282`) and at the bake site (`:5445-5455`).
   **The `ctr_type` / `target_border_idx` ARGUMENTS at those two materialization
   sites already exist** — E09 added them as the constants `ECtrType::Borders, 0`.
   E10 replaces the *values* only; it must NOT re-widen the call.
3. Keep `prior_denom = 1.0` (CPU forbids a non-unit denominator —
   `ctr_helper.cpp:50`) and `target_border_idx = 0` (W3 expands it).
4. Fix the stale `one_hot_splits` doc comment at `:816-817`.

**Refactor constraints + required regression scope**
- Constraint: **add no `BoostParams` field and change no field type** — 62 files
  pin `simple_ctr:` `[VERIFIED: LOCAL grep, 65 occurrences / 62 files]`.
- Constraint: at the default config (`Borders`, `[0.5]` both sides) every emitted
  column must be byte-identical to pre-change — that is the SPEC-CTRT-18 gate.
- Regression scope: **all 11 CTR oracles + all 3 one-hot targets** (boosting.rs is
  touched).

**Validation**
```bash
cargo test -p cb-train --lib boosting::tests
cargo test -p cb-train --test plain_ctr_oracle_test --test ordered_ctr_oracle_test \
  --test tensor_ctr_oracle_test --test tensor_ctr_e2e_oracle_test \
  --test s_order_ctr_bins_oracle_test --test ctr_split_scoring_test \
  --test ctr_feature_materialize_test --test multi_permutation_e2e_oracle_test \
  --test multi_permutation_fold_oracle_test
cargo test -p cb-model --test ctr_data_roundtrip_test --test fstr_ctr_oracle_test
cargo test -p cb-train --test one_hot_oracle_test --test one_hot_draw_accounting_test \
  --test device_one_hot_parity_test
cargo test -p cb-train
cargo clippy -p cb-train --all-targets
```

**Completion evidence.** Both routing tests green; E03's characterization test
updated in-place and still green at defaults; 11 CTR oracles + 3 one-hot targets
green; `grep -n 'params\.simple_ctr' crates/cb-train/src/boosting.rs` now returns
**non-zero** matches (it returns zero today `[VERIFIED: LOCAL]`).

---

### E11 — Per-type final tables in the bake path (+ mean threading into `cb-model`)

- **Specs:** SPEC-CTRT-13; unblocks A1/A2/A3
- **Blocked by:** E10. **Blocks:** E12, E13, E15, E18, E21.
- **Parallelizable:** **NO** — owns `crates/cb-train/src/ctr/bake.rs` and
  `crates/cb-model/src/ctr_data.rs::from_baked`.

**Goal / observable completion condition.** `bake_ctr_table` takes the CTR type,
calls `build_final_ctr(&acc, that_type, ..)`, reshapes per type (Counter/FeatureFreq
= 1 value/bucket + a real `counter_denominator`; mean types = `(Sum, Count)` pairs),
and `CtrData::from_baked` carries mean tables instead of `Vec::new()`.

**Files**
- Modify: `crates/cb-train/src/ctr/bake.rs`
- Modify: `crates/cb-train/src/ctr/final_ctr.rs` (signature only — the
  `counter_calc_skip_test` parameter its doc ALREADY documents but which does not
  exist; see below)
- Modify: `crates/cb-train/src/ctr/final_ctr_test.rs`
- Modify: `crates/cb-model/src/ctr_data.rs` (`CtrData::from_baked`)
- Modify: `crates/cb-train/src/boosting.rs` (the bake call at `:5445-5455`)
- Modify: `crates/cb-model/tests/ctr_data_roundtrip_test.rs`
- Modify: `crates/cb-train/tests/ctr_split_scoring_test.rs` — **mechanical, forced
  by the widened `bake_ctr_table` signature.** Update the THREE `bake_ctr_table`
  call sites at `:542`, `:576`, `:645` for the new arity, passing the resolved
  `ECtrType::Borders` (today's hard-coded behavior, preserved exactly)
  `[VERIFIED: LOCAL crates/cb-train/tests/ctr_split_scoring_test.rs:542, :576, :645]`.
  **CHANGE NO ASSERTION.** This file is **one of the eleven SPEC-CTRT-18 oracle
  targets** (PLAN.md §3.2); **weakening or deleting any assertion in it is
  FORBIDDEN** — "fixing" the compile break by removing a call silently removes CTR
  bake coverage. E09 already widened this file's `materialize_ctr_feature` calls at
  `:384`/`:394`; those are E09's and are not re-touched here.

**Exact verified files/symbols to touch**
- **The three hard-codes in `bake_ctr_table`:**
  `let final_table = build_final_ctr(&acc, ECtrType::Borders);` (**`bake.rs:192`**),
  `ctr_type: ECtrType::Borders.as_i8()` (**`:232`**),
  `counter_denominator: 0` (**`:236`**) `[VERIFIED: LOCAL, read verbatim]`.
- `BakedCtrTable { projection, ctr_type: i8, target_classes_count, hashes,
  int_counts: Vec<Vec<i64>>, counter_denominator: i64, shift, scale, prior_num,
  prior_denom }` at `bake.rs:61-86` — **has NO mean fields**; gains
  `pub mean: Vec<(f32, i64)>` `[VERIFIED: LOCAL]`.
- `fn build_final_ctr(acc: &OnlineCtrAccumulator, ctr_type: ECtrType) -> FinalCtrTable`
  at `crates/cb-train/src/ctr/final_ctr.rs:75`. **Its doc comment at `:70-73`
  already describes a `counter_calc_skip_test` parameter that DOES NOT EXIST in the
  signature** `[VERIFIED: LOCAL, read verbatim]` — a live documentation lie. E11
  adds the parameter (threaded as the constant **`true`** here — `SkipTest` is the
  default returned by `counter_calc_method_default()`, so `true` is the
  behavior-preserving value; E22 makes it real), or
  the doc must be corrected. **Adding it now avoids a second signature churn in
  W5** — do it here and note the semantics are inert until E22.
- `FinalCtrTable { ctr_type, target_classes_count, int_counts: Vec<i64>,
  mean_sum: Vec<f32>, mean_count: Vec<i64>, counter_denominator, bucket_count }`
  at `final_ctr.rs:44-65` — **already complete for all six types**
  `[VERIFIED: LOCAL final_ctr.rs:89-125]`. Nothing to rebuild.
- `CtrData::from_baked` hard-codes `mean: Vec::new()`
  `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs, from_baked body]`.
- `CtrValueTable::numerator_denominator` already dispatches all six types
  correctly, including `Buckets → (counts[b], Σ classes)` and
  `mean → (Sum, Count)` `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs:226-266]`
  — **the apply side needs no change**.
- Counter's wire `TargetClassesCount` is `0` and the decoder forces `width = 1`
  `[VERIFIED: LOCAL ctr_data.rs decode_one_ctr_value_table]` — the bake must emit
  `int_counts` as one value per bucket for Counter, matching that width.

**CodeGraph evidence.** `build_final_ctr` (`final_ctr.rs:75`) has **14 callers** in
`ctr/bake.rs` and `ctr/mod.rs`, covered by `crates/cb-model/tests/ctr_data_roundtrip_test.rs`
and `crates/cb-train/src/ctr/final_ctr_test.rs` `[VERIFIED: CODEGRAPH research §F.1]`.
Those covering tests are the gate for the signature change.

**Red**
- File: `crates/cb-train/src/ctr/final_ctr_test.rs` (unit) and
  `crates/cb-model/tests/ctr_data_roundtrip_test.rs` (integration)
- Test fn 1 (`final_ctr_test.rs`): `bake_emits_the_requested_type_and_denominator`
  Setup: a 12-document, 3-bucket cat column; bake once per CPU-legal type.
  Expected:
  - `Borders` → `ctr_type == 0`, `int_counts` per bucket length 2,
    `counter_denominator == 0`, `mean.is_empty()`;
  - `Buckets` → `ctr_type == 1`, same shape as Borders (the difference is at
    apply time, via `target_border_idx`);
  - `Counter` → `ctr_type == 4`, `int_counts` per bucket length **1**,
    `counter_denominator == max bucket total` (hand-computed, non-zero);
  - `BinarizedTargetMeanValue` → `ctr_type == 2`, `int_counts.is_empty()`,
    `mean.len() == bucket_count`, and `mean[b].0` is `f32`-typed.
- Test fn 2 (`ctr_data_roundtrip_test.rs`):
  `from_baked_carries_mean_tables_for_btmv`
  Expected: `CtrData::from_baked(&baked).tables[key].mean.len() == bucket_count`
  and `!mean.is_empty()`, plus `assert_ne!(mean[0].0, 0.0)` on a fixture where the
  bucket has at least one positive-class document (anti-vacuity).
- Test fn 3 (regression): `borders_bake_bytes_are_unchanged`
  Frozen literals for `hashes`, `int_counts`, `shift`, `scale` transcribed from a
  pre-change run of the existing default path.
- Test fn 4 (`ctr_data_roundtrip_test.rs`, **the de-dup-key pin**):
  `from_baked_emits_exactly_one_table_for_two_buckets_splits_at_different_border_idx`
  Setup: a `BakedCtrData` for ONE projection `{0}` reached by **two** chosen
  `Buckets` splits, one at `target_border_idx = 0` and one at `= 1`.
  Expected:
  ```rust
  let data = CtrData::from_baked(&baked);
  assert_eq!(data.tables.len(), 1,
      "target_border_idx must NOT be part of ctr_base_key or the bake key — one \
       Buckets table serves both b=0 and b=1 (crates/cb-model/src/ctr_data.rs:299)");
  assert!(data.tables.contains_key(&ctr_base_key(ECtrType::Buckets, &[0])));
  ```
  Plus the complementary case: the SAME projection with one `Buckets` split and one
  `Counter` split ⇒ `data.tables.len() == 2` (distinct `ctr_type` ⇒ distinct key) —
  proving the de-dup key is `(projection, ctr_type)` and not `projection` alone.
- **EXPECTED INITIAL FAILURE:**
  `error[E0061]: this function takes 7 arguments but 8 were supplied` on
  `bake_ctr_table`; **the same `E0061` is emitted at the three `bake_ctr_table`
  call sites in `crates/cb-train/tests/ctr_split_scoring_test.rs` (`:542`, `:576`,
  `:645`) as soon as the widened signature lands — that target does not build until
  the three mechanical argument additions in Files land; compile-forced, not
  behavioral**; after the signature lands,
  ``assertion `left == right` failed: left: 0, right: 4`` on the Counter
  `ctr_type`, and ``assertion failed: !mean.is_empty()`` on the BTMV case
  (`from_baked` hard-codes `mean: Vec::new()`).
- Run: `cargo test -p cb-train --lib ctr::final_ctr_test` and
  `cargo test -p cb-model --test ctr_data_roundtrip_test`

**Green (minimal implementation intent).**
1. `bake_ctr_table` gains `ctr_type: ECtrType`; passes it to `build_final_ctr`;
   sets `ctr_type: ctr_type.as_i8()`.
2. Reshape by type: class types keep the existing bucket-major `[N0, N1, …]`
   reshape (`bake.rs:194-219`, unchanged); Counter/FeatureFreq emit
   `vec![vec![total]]` per bucket and copy `final_table.counter_denominator`;
   mean types populate the new `mean: Vec<(f32, i64)>` from
   `final_table.mean_sum` / `.mean_count` and leave `int_counts` empty.
3. `build_final_ctr` gains `counter_calc_skip_test: bool` (inert until E22); every
   caller passes `true` (the `SkipTest` default).
4. `CtrData::from_baked` copies `t.mean.clone()` instead of `Vec::new()`.
5. `boosting.rs:5445-5455` passes the per-candidate `ctr_type` resolved by E10's
   `ctr_config_for(params, spec_is_simple)`. **The bake currently de-duplicates by
   `projection` alone (`seen: Vec<TProjection>` at `:5440-5443`); it must
   de-duplicate on `(projection, ctr_type)` — and by NOTHING ELSE.**
   **Rationale, stated accurately (do not restate it as a live hazard):** under the
   locked scalar-field design (locked decision 3) a projection determines
   `is_simple` — `CtrCandidate.is_simple` comes from `TProjection::is_simple()`
   (`crates/cb-train/src/candidates.rs:151-157, 194`) — and therefore determines its
   `ECtrType`, so **the multi-type case is not reachable today**; two different-typed
   splits on one projection cannot occur in production. `(projection, ctr_type)` is
   nonetheless the **correct** key and is what E15's copy-back lookup keys on.
   `target_border_idx` MUST NOT enter it.

   **EXPLICIT CONSTRAINT (load-bearing, do not "improve" it):**
   > **`target_border_idx` MUST NOT enter `ctr_base_key` or the bake key; it is a
   > per-split selector consumed by `CtrValueTable::numerator_denominator`. One
   > Buckets table serves both `b = 0` and `b = 1`.**

   Evidence: `ctr_base_key(ctr_type, cat_features)` at
   `crates/cb-model/src/ctr_data.rs:299` produces
   `"ctr:type=<i8>:proj=<members>"` and **carries no `target_border_idx`**;
   `CtrData::from_baked` (`:313-331`) inserts into a `BTreeMap` keyed by that
   string; `crates/cb-model/src/apply.rs:126 ctr_table_key` reconstructs the
   IDENTICAL form from `CtrSplit`; and `decode_ctr_model_parts` (`:562-566`)
   hard-errors on a duplicate key
   (`"duplicate ctr_data table key … (ctr_type/projection collision, CTR-05)"`).
   `CtrValueTable::numerator_denominator` (`:225`) already takes
   `target_border_idx` as a **parameter**. Adding the index to the key would break
   the apply-side key reconstruction, every existing CTR oracle, and every
   committed `.cbm` fixture (`ctr_load/{simple,combo}.cbm`, `tensor_ctr_e2e`,
   `fstr_ctr`) — so it is prohibited, not merely discouraged.

**Refactor constraints + required regression scope**
- Constraint: the `(shift, scale)` derivation from `calc_normalization(prior_num)`
  at `bake.rs:221-228` is type-agnostic and must not change.
- Constraint: the combined-hash key-string collision guard at `bake.rs:204-208`
  must be preserved.
- Constraint: `ctr_base_key(ctr_type, projection)` already includes the type, so
  two typed tables on one projection get distinct keys — verify, do not redesign.
  **Do NOT add `target_border_idx` to it** (see Green step 5's explicit
  constraint); test fn 4 is the pin.
- Regression scope: **all 11 CTR oracles**, `cargo test -p cb-model`, and the
  E00 `.cbm` non-mean baseline gate.

**Validation**
```bash
cargo test -p cb-train --lib ctr::final_ctr_test
cargo test -p cb-model --test ctr_data_roundtrip_test
cargo test -p cb-model --test ctr_nonmean_byte_identity_test
cargo test -p cb-train --test plain_ctr_oracle_test --test ordered_ctr_oracle_test \
  --test tensor_ctr_oracle_test --test tensor_ctr_e2e_oracle_test \
  --test s_order_ctr_bins_oracle_test --test ctr_split_scoring_test \
  --test ctr_feature_materialize_test --test multi_permutation_e2e_oracle_test \
  --test multi_permutation_fold_oracle_test
cargo test -p cb-model --test fstr_ctr_oracle_test --test cbm_oracle_test --test json_oracle_test
cargo test -p cb-train -p cb-model
cargo clippy -p cb-train -p cb-model --all-targets
```

**Completion evidence.** Four per-type bake cases green; the `from_baked` mean test
green with its anti-vacuity assertion; test fn 4's `tables.len() == 1` de-dup-key
pin (and its `== 2` complement) green; the frozen Borders bake literals unchanged;
all 11 CTR oracles + the E00 baseline gate green. **`git diff
crates/cb-train/tests/ctr_split_scoring_test.rs` shows exactly the three widened
`bake_ctr_table` call sites (`:542`, `:576`, `:645`) on top of E09's edits and
NOTHING else** — no assertion added, removed, weakened or reworded.

---

### E12 — `counter_simple` fixture + end-to-end ≤1e-5 gate

- **Specs:** SPEC-CTRT-08 (parity half); acceptance **A3**
- **Blocked by:** E11. **Blocks:** none.
- **Parallelizable:** **YES** with E13 — disjoint fixture directories and disjoint
  new test targets; neither touches production code.

**Goal / observable completion condition.** A committed, frozen
`crates/cb-oracle/fixtures/ctr_counter_simple/` produced by a **fixture-local**
generator, and a cb-train integration test asserting the repo's predictions match
upstream within **1e-5** AND that the materialized Counter column is
permutation-invariant end to end.

**Files**
- Create: `crates/cb-oracle/fixtures/ctr_counter_simple/gen_fixtures.py`
- Create + COMMIT: `crates/cb-oracle/fixtures/ctr_counter_simple/{X_cat.npy,y.npy,model.json,predictions.npy,config.json}`
- Create: `crates/cb-train/tests/ctr_counter_simple_oracle_test.rs`

**Exact verified files/symbols to touch (read-only patterns)**
- Generator shape: `crates/cb-oracle/fixtures/ctr_load/gen_fixtures.py`
  `[VERIFIED: LOCAL, head read — carries the FROZEN GENERATOR banner and the
  run-to-run-nondeterminism caveat verbatim; copy that banner]`.
- Artifact set + `config.json` key set: `crates/cb-oracle/fixtures/plain_ctr/config.json`
  (`scenario`, `requirement`, `seed`, `catboost_version`, `thread_count`, `n_rows`,
  `params`, `cardinalities`, `stages`, `npy_schema`) `[VERIFIED: LOCAL, read in full]`
  and `crates/cb-oracle/fixtures/one_hot_train/default_binary/config.json` for the
  `"NEVER regenerated in CI"` note wording `[VERIFIED: LOCAL, read in full]`.
- Test harness shape: `crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs`
  — `fn fixture(rel: &str) -> PathBuf` at `:54-60`
  (`env!("CARGO_MANIFEST_DIR")/../cb-oracle/fixtures`), the `X_cat.npy` int32 →
  string stringification at `:67-69`, and the `predictions.npy` comparison at
  `:215-222` `[VERIFIED: LOCAL grep]`.
- Training entrypoint: `cb_train::train_cat(&backend, &[], &[], &cat_cols, &y, &w,
  &params, None) -> CbResult<(Model, BakedCtrData)>`
  `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:2236-2245]`; predict via
  `cb_model::Model::from_trained(..).with_ctr_data(CtrData::from_baked(&baked))`
  then `cb_model::predict_raw_cat`.

**Fixture configuration (every value is a verified risk mitigation)**

| Choice | Value | Why |
|---|---|---|
| features | **categorical only**, 2 columns, 0 float | removes upstream float-quantization nondeterminism structurally (precedent `tensor_ctr_e2e`) |
| cardinalities | 6 and 5, `one_hot_max_size=1` | both columns are CTR-eligible (`route_categorical(card, 1) == Ctr`) `[VERIFIED: CODEGRAPH candidates.rs:92-104]` |
| `simple_ctr` | `["Counter:Prior=0.5"]` | the type under test |
| `combinations_ctr` | `[]` | disables combination CTRs (precedent `plain_ctr/config.json`) |
| `max_ctr_complexity` | 1 | simple CTRs only; clear of the ORD-06/07 combination-gating bug |
| everything else | the §3 isolating set | `permutation_count=1`, `bootstrap_type="No"`, `random_strength=0`, `random_seed=0`, `thread_count=1`, `boost_from_average=False`, `leaf_estimation_method="Gradient"`, `leaf_estimation_iterations=1`, `depth=2`, `iterations=5`, `learning_rate=0.1`, `l2_leaf_reg=3.0`, `loss_function="Logloss"`, `boosting_type="Plain"`, `fold_len_multiplier=2.0` |
| n_rows | 30 | matches `plain_ctr`'s scale; small enough to hand-inspect |

Route the config through the **low-level `CatBoost(params)` API**, not
`CatBoostClassifier(**kwargs)`, so non-sklearn keys are honored (the same reason
`gen_tensor_ctr_e2e` does).

**MANDATORY anti-false-pass guard, executed inside the generator before writing:**
```python
ctrs = model_json["features_info"]["ctrs"]
assert any(c["ctr_type"] == "Counter" for c in ctrs), (
    "no Counter CTR descriptor in model.json — the config produced ZERO Counter "
    "splits and this fixture would pass trivially")
assert predictions.std() > 1e-6, "degenerate constant predictions"
```
Plus the §3 corpus-cleanliness guard (`git status --porcelain crates/cb-oracle/fixtures`
must list only `ctr_counter_simple/*`, else `sys.exit(1)`).

**Red**
- File: `crates/cb-train/tests/ctr_counter_simple_oracle_test.rs`
- Test fn 1: `counter_simple_predictions_match_upstream_within_1e_minus_5`
  Setup: load `X_cat.npy` (int32, stringified via `cb_data::stringify_int_category`
  — the A4 form the fixture hashed), `y.npy`, `predictions.npy`; build
  `BoostParams` pinning EVERY field explicitly (Pitfall-6) with
  `simple_ctr: ECtrType::Counter`, `simple_ctr_priors: vec![0.5]`,
  `combinations_ctr_priors: vec![0.5]`, `max_ctr_complexity: 1`,
  `one_hot_max_size: 1`.
  Expected: `max |ours − upstream| <= 1e-5`, reported in the failure message; plus
  a non-degeneracy guard `assert!(ours.iter().any(|v| *v != ours[0]))`.
- Test fn 2: `counter_simple_model_actually_carries_a_counter_ctr_split`
  Expected: `baked.tables.iter().any(|t| t.ctr_type == ECtrType::Counter.as_i8())`
  AND `≥1 ModelSplit::Ctr` in some tree — the Rust-side twin of the generator's
  anti-false-pass guard.
- Test fn 3: `counter_column_is_permutation_invariant_end_to_end`
  **The assertion is on the MATERIALIZED PER-DOCUMENT COLUMN, not on the baked
  table.** The baked table comes from `bake_ctr_table` → `accumulate_online` over
  the **whole learn set** (`crates/cb-train/src/ctr/bake.rs:189-192`), which is
  permutation-independent for **every** `ECtrType` including Borders — so a
  baked-table comparison is **VACUOUS** and would still pass against a regression
  that turned the Counter *column* back into a read-before-increment prefix.
  Setup: build the same learn corpus twice and call `materialize_ctr_feature(…,
  ECtrType::Counter, 0)` under two different permutations
  (identity and a fixed shuffle).
  Expected:
  ```rust
  assert_eq!(col_a.bins, col_b.bins,
      "Counter is permutation-INDEPENDENT (IsPermutationDependentCtrType(Counter)==false, \
       ctr_type.cpp:43-56); a prefix implementation would differ here");
  for (a, b) in col_a.ctr_value.iter().zip(col_b.ctr_value.iter()) {
      assert_eq!(a.to_bits(), b.to_bits());
  }
  ```
  (Model predictions under the two permutations are an acceptable equivalent
  assertion; the column is preferred because it localizes better.)
- Test fn 3b (**the ANTI-VACUITY COMPANION — mandatory**):
  `borders_column_is_NOT_permutation_invariant_on_the_same_corpus`
  Runs the **identical** comparison on the same corpus with
  `ECtrType::Borders, b = 0` and asserts it **FAILS**:
  ```rust
  assert_ne!(borders_a.ctr_value, borders_b.ctr_value,
      "if a Borders column is ALSO permutation-invariant on this corpus then test \
       fn 3 proves nothing — widen the corpus or the permutation, do NOT weaken \
       either assertion");
  ```
  Without this companion, test fn 3 does not discriminate Counter from a prefix.
- **EXPECTED INITIAL FAILURE (before the fixture exists):**
  `No such file or directory (os error 2)` on `ctr_counter_simple/X_cat.npy`.
  **After the fixture lands but before E06–E11:** a numeric
  `max |ours − upstream| = <large>` failure, because the engine would have trained
  Borders CTRs regardless of `simple_ctr`.
- Run: `cargo test -p cb-train --test ctr_counter_simple_oracle_test`

**Green (minimal implementation intent).** **No production change** — the behavior
is delivered by E06/E09/E10/E11. This task delivers the fixture + the gate. If the
≤1e-5 gate fails, run the localization ladder below; do NOT patch `cb-model`.

**Localization ladder (STOP AND REPORT at the first hit)**
1. Compare the repo's per-object Counter column against the fixture's
   `model.json → ctr_data` bucket counts. A count mismatch ⇒ bucket-space
   divergence (research §J.3, `PerfectHash` vs `ComputeReindexHash`) —
   **STOP AND REPORT**.
2. Compare `counter_denominator`. A mismatch ⇒ MAX-bucket rule divergence.
3. Compare `shift`/`scale` against `model.json`'s `ctrs[i].{shift,scale}`.
4. Any other divergence ⇒ report the localized stage with numbers.

**Refactor constraints + required regression scope**
- Constraint: **never** invoke `crates/cb-oracle/generator/gen_fixtures.py`.
- Constraint: generate ONCE, commit, never regenerate.
- Regression scope: `cargo test -p cb-train` (no production change, so the 11 CTR
  oracles are informational here, but must be run).

**Validation**
```bash
.venv/bin/python -c "import catboost; assert catboost.__version__ == '1.2.10'"
.venv/bin/python crates/cb-oracle/fixtures/ctr_counter_simple/gen_fixtures.py
git status --porcelain crates/cb-oracle/fixtures   # ONLY ctr_counter_simple/*
cargo test -p cb-train --test ctr_counter_simple_oracle_test
# THE FULL 11-TARGET CTR ORACLE SCOPE (PLAN.md §3.2) — the stated scope, run in full
cargo test -p cb-train --test plain_ctr_oracle_test --test ordered_ctr_oracle_test \
  --test tensor_ctr_oracle_test --test tensor_ctr_e2e_oracle_test \
  --test s_order_ctr_bins_oracle_test --test ctr_split_scoring_test \
  --test ctr_feature_materialize_test --test multi_permutation_e2e_oracle_test \
  --test multi_permutation_fold_oracle_test
cargo test -p cb-model --test ctr_data_roundtrip_test --test fstr_ctr_oracle_test
cargo test -p cb-train
```

**Completion evidence.** Five committed artifacts; the generator's two assertions
passing; all four tests green (fn 1, fn 2, fn 3 **and** the fn 3b Borders
anti-vacuity companion) with the reported max-divergence number recorded;
`git status --porcelain crates/cb-oracle/fixtures` listing only the new directory.

---

### E13 — `btmv_simple` fixture + end-to-end ≤1e-5 gate

- **Specs:** SPEC-CTRT-07 (parity half); acceptance **A2**
- **Blocked by:** E11. **Blocks:** E18.
- **Parallelizable:** **YES** with E12 — disjoint fixture dir and test target.

**Goal / observable completion condition.** A committed, frozen
`crates/cb-oracle/fixtures/ctr_btmv_simple/` and an integration test asserting
≤1e-5 against upstream, with the model's mean CTR table proven non-empty.

**Files**
- Create: `crates/cb-oracle/fixtures/ctr_btmv_simple/gen_fixtures.py`
- Create + COMMIT: `crates/cb-oracle/fixtures/ctr_btmv_simple/{X_cat.npy,y.npy,model.json,model.cbm,predictions.npy,config.json}`
  — **`model.cbm` is included here** and is reused by E18/E19 as the upstream BTMV
  `.cbm` the decoder must load (A8's second half).
- Create: `crates/cb-train/tests/ctr_btmv_simple_oracle_test.rs`

**Exact verified files/symbols to touch**
- Same patterns as E12. Additionally:
  `model.json` already supports mean tables in this repo —
  `CtrData` JSON serde handles `is_mean()` with **stride 3**
  `[VERIFIED: research §F.4 against crates/cb-model/src/ctr_data.rs:366-396,428-475]`
  — so a **JSON-only** BTMV oracle is viable **before** the `.cbm` codec exists.
  That is why E13 sits in W2 and E18–E20 sit in W4.
- `CtrValueTable::numerator_denominator`'s mean arm returns
  `(f64::from(sum), count as f64)` `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs:249-256]`.

**Fixture configuration.** Identical to E12 except
`simple_ctr = ["BinarizedTargetMeanValue:Prior=0.5"]`.

**MANDATORY anti-false-pass guard (generator):**
```python
assert any(c["ctr_type"] == "BinarizedTargetMeanValue" for c in ctrs), (
    "no BTMV CTR descriptor in model.json — zero BTMV splits, fixture is vacuous")
assert predictions.std() > 1e-6
```
Plus the §3 corpus-cleanliness guard.

**Red**
- File: `crates/cb-train/tests/ctr_btmv_simple_oracle_test.rs`
- Test fn 1: `btmv_simple_predictions_match_upstream_within_1e_minus_5`
  As E12's test fn 1 with `simple_ctr: ECtrType::BinarizedTargetMeanValue`.
- Test fn 2: `btmv_baked_table_carries_a_non_empty_mean_vector`
  Expected: the baked table for the chosen projection has
  `!mean.is_empty()`, `mean.len() == bucket_count`, `int_counts.is_empty()`, and
  `mean.iter().any(|&(s, _)| s != 0.0)` (anti-vacuity).
- Test fn 3 (**the f32 differential at fixture scale — a REPORTING STEP, NOT A
  GATE**, SPEC-CTRT-07 / R2; see SPEC.md §7's A2 note):
  `btmv_f64_sum_accumulation_diverges_from_upstream_on_this_fixture`
  Computes the same predictions with an f64 `Sum` accumulation (a local test-only
  reimplementation) and asserts either (a) it diverges from upstream by more than
  the f32 version does, or (b) — **the expected outcome at binclf, since f32 and
  f64 accumulation of small integers are bit-identical below `2^24` and this
  fixture is 30 rows** — the test **records that explicitly** via a printed
  `REPORTED: f32/f64 indistinguishable at this scale (maxdiff = …)` line and defers
  to **E07 test fn 2's accumulator-level differential, which is the actual gate**.
  **A silent pass is forbidden**, but branch (b) is NOT a failure: it is the
  recorded, expected result, and the task does not stall on it.
- **EXPECTED INITIAL FAILURE:** `No such file or directory` on
  `ctr_btmv_simple/X_cat.npy`; after the fixture lands, a numeric ≤1e-5 failure
  until E07/E09/E11 are all in.
- Run: `cargo test -p cb-train --test ctr_btmv_simple_oracle_test`

**Green (minimal implementation intent).** No production change (delivered by
E07/E09/E10/E11).

**Refactor constraints + required regression scope**
- Constraint: this test must NOT call `save_cbm` on the trained BTMV model — that
  path still rejects mean tables until E20 (`ModelError::Serialize("mean/target-mean
  CTR unsupported on save (v1, MAJOR-2)")`) `[VERIFIED: LOCAL]`. Add an explicit
  test fn 4 `btmv_save_cbm_is_a_typed_rejection_until_e20` asserting that typed
  error, so the limitation is **tested**, not merely known. E20 flips this test.
- Regression scope: **the full 11-target CTR oracle scope (PLAN.md §3.2)**, plus
  `cargo test -p cb-train`, `cargo test -p cb-model`.

**Validation**
```bash
.venv/bin/python crates/cb-oracle/fixtures/ctr_btmv_simple/gen_fixtures.py
git status --porcelain crates/cb-oracle/fixtures   # ONLY ctr_btmv_simple/*
cargo test -p cb-train --test ctr_btmv_simple_oracle_test
# THE FULL 11-TARGET CTR ORACLE SCOPE (PLAN.md §3.2) — the stated scope, run in full
cargo test -p cb-train --test plain_ctr_oracle_test --test ordered_ctr_oracle_test \
  --test tensor_ctr_oracle_test --test tensor_ctr_e2e_oracle_test \
  --test s_order_ctr_bins_oracle_test --test ctr_split_scoring_test \
  --test ctr_feature_materialize_test --test multi_permutation_e2e_oracle_test \
  --test multi_permutation_fold_oracle_test
cargo test -p cb-model --test ctr_data_roundtrip_test --test fstr_ctr_oracle_test
cargo test -p cb-train -p cb-model
```

**Completion evidence.** Six committed artifacts (including `model.cbm` for E19);
four tests green including the explicit save-rejection test and the f32/f64
discrimination report.

---

## WAVE W3 — Candidate expansion (HIGHEST RISK)

> This wave **changes the scored candidate set and therefore tie-breaks**
> (SPEC.md R1). It lands only after the W1 firewall is green, each change sits
> behind its own fixture, and every one of the 11 existing single-prior CTR
> oracles is re-run at every step.

---

### E14 — `borders_multiprior` fixture (data lands BEFORE the code change)

- **Specs:** SPEC-CTRT-11 (data half); acceptance **A4**
- **Blocked by:** none. **Blocks:** E15.
- **Parallelizable:** **YES** with ALL of W0–W2 — owns only a new fixture
  directory; touches no production code and no shared test file.

**Goal / observable completion condition.** A committed, frozen
`crates/cb-oracle/fixtures/ctr_borders_multiprior/` whose `model.json` provably
contains **three** Borders CTR descriptors on the same projection (one per prior),
so E15's Red is a real, data-backed failure rather than a guess.

**Files**
- Create: `crates/cb-oracle/fixtures/ctr_borders_multiprior/gen_fixtures.py`
- Create + COMMIT: `.../{X_cat.npy,y.npy,model.json,predictions.npy,config.json}`

**Fixture configuration.** The §3 isolating set, categorical-only, with
`simple_ctr = ["Borders:Prior=0:Prior=0.5:Prior=1"]`, `combinations_ctr = []`,
`max_ctr_complexity = 1`.

**MANDATORY anti-false-pass guard (generator) — stronger than the others:**
```python
borders = [c for c in ctrs if c["ctr_type"] == "Borders"]
priors = sorted({round(c["prior_num"] / c["prior_denom"], 6) for c in borders})
assert len(priors) >= 2, (
    f"multi-prior expansion is untestable: model.json carries priors {priors}; "
    "the config produced fewer than two distinct prior columns")
assert predictions.std() > 1e-6
```
Plus the §3 corpus-cleanliness guard. **If the assertion fires, widen the corpus
(more rows / stronger cat signal) until upstream genuinely selects splits at more
than one prior — do NOT weaken the assertion.**

**Red — the falsifiability requirement for a data-only task is the
DOUBLE-GENERATION DETERMINISM CHECK (precedent: Part-2 PLAN T16):**
```bash
.venv/bin/python crates/cb-oracle/fixtures/ctr_borders_multiprior/gen_fixtures.py
cp -r crates/cb-oracle/fixtures/ctr_borders_multiprior \
      "$SCRATCH/ctr_borders_multiprior_run1"
.venv/bin/python crates/cb-oracle/fixtures/ctr_borders_multiprior/gen_fixtures.py
diff -r "$SCRATCH/ctr_borders_multiprior_run1" \
        crates/cb-oracle/fixtures/ctr_borders_multiprior
```
**Expected:** `diff` empty. If `model.json` differs (upstream quantization
nondeterminism — but this fixture is categorical-only, so it should not), commit
run 1 and record the observed instability in `config.json`'s `note`.
**If `predictions.npy` differs, STOP AND REPORT** — a nondeterministic reference
cannot be an oracle. (`$SCRATCH` = the session scratchpad directory; do NOT copy
into the repo.)

**Green.** Write the generator; run once; commit all five artifacts.

**Refactor constraints + required regression scope**
- `config.json` must carry the standard key set plus a `note` recording the
  distinct prior list the guard observed, and the `NEVER regenerate in CI`
  sentence.
- Regression scope: none (no code change); `git status --porcelain crates/cb-oracle/fixtures`
  must list only this directory.

**Validation**
```bash
.venv/bin/python -c "import catboost; assert catboost.__version__ == '1.2.10'"
.venv/bin/python crates/cb-oracle/fixtures/ctr_borders_multiprior/gen_fixtures.py
git status --porcelain crates/cb-oracle/fixtures
.venv/bin/python -c "import json,pathlib; \
  c=json.loads(pathlib.Path('crates/cb-oracle/fixtures/ctr_borders_multiprior/config.json').read_text()); \
  assert c['params']['max_ctr_complexity']==1 and c['params']['permutation_count']==1 \
     and c['params']['random_strength']==0 and c['params']['thread_count']==1"
```

**Completion evidence.** Five committed artifacts; the guard's observed prior list
recorded in `config.json`; the empty double-generation `diff`.

---

### E15 — Candidate expansion over the FULL prior list

- **Specs:** SPEC-CTRT-11; acceptance **A4**; guarded by SPEC-CTRT-18
- **Blocked by:** E05 (firewall), E11, E14. **Blocks:** E16.
- **Parallelizable:** **NO** — owns `crates/cb-train/src/boosting.rs`.
- **RISK:** **HIGHEST in Part 1.** It changes the candidate set the greedy search
  scores, hence tie-breaks, hence potentially every CTR oracle.

**Goal / observable completion condition.** ONE candidate column is emitted per
`(candidate, prior)` in upstream's order (`greedy_tensor_search.cpp:414-427`), not
`.first()` only; `ctr_borders_multiprior` passes at ≤1e-5; **every existing
single-prior CTR oracle is unchanged**.

**Files**
- Modify: `crates/cb-train/src/boosting.rs`
- Modify: `crates/cb-train/src/boosting_test.rs`
- Create: `crates/cb-train/tests/ctr_borders_multiprior_oracle_test.rs`

**Exact verified files/symbols to touch**
- The two materialization loops that currently emit ONE column per projection:
  `for proj in &absolute_projections { … materialize_ctr_feature(…) … }` at
  `crates/cb-train/src/boosting.rs:3237-3247` (structure folds) and
  `:3273-3282` (averaging fold) `[VERIFIED: LOCAL, read in full]`. Both must become
  a nested loop over the candidate's resolved prior list.
- `structure_fold_columns: Vec<Vec<CtrFeatureColumn>>` (`:3200`) and
  `averaging_ctr_features: Vec<CtrFeatureColumn>` (`:3268`) grow; they are
  **index-aligned by construction** ("Index-aligned with `materialized_ctr_features`
  (same projection order), so a chosen structure CTR split maps to the same
  averaging column" `[VERIFIED: LOCAL boosting.rs:3262-3267 comment]`).
  **That alignment invariant is the single most fragile thing this task can break**
  — both loops must iterate `(projection, prior)` in the SAME order.
- `materialized_ctr_features` = `structure_fold_columns.first().cloned()`
  (`:3253-3257`) drives the `has_ctr` gate — unchanged in shape.
- `greedy_tensor_search_oblivious_with_ctr(matrix, ctr_features, ctr_border_count,
  der1, weight, scaled_l2, depth, n_objects, target_border_idx, model_size_reg,
  score_function, cat_eligible_buckets)` at `crates/cb-train/src/tree.rs:3228-3241`,
  called at `crates/cb-train/src/boosting.rs:4653` — it consumes
  `ctr_features: &[CtrFeatureColumn]` as a flat list, so a longer list needs **no
  signature change** `[VERIFIED: LOCAL, read verbatim]`.
- `cat_eligible_buckets: &[Vec<u32>]` (the `model_size_reg` cat-feature-weight
  input, `GetCatFeatureWeight`, `greedy_tensor_search.cpp:908-932`) —
  **`cat_eligible_buckets` (`crates/cb-train/src/boosting.rs:3074`, passed at
  `:4669`) is one `perfect_hash_bins` column per CTR-eligible categorical
  feature (`eligible_absolute`), consumed by an order-insensitive `.max()` at
  `crates/cb-train/src/tree.rs:3026`. It is NOT index-aligned with `ctr_features`
  and MUST NOT grow with the `(projection, b, prior)` expansion — leave it exactly
  as built.** `[VERIFIED: LOCAL boosting.rs:3074, :4669; tree.rs:3026 and the doc
  comment at tree.rs:2984-2987]`
- **`crates/cb-train/src/boosting.rs:5437-5473` — THE BAKE BLOCK AND ITS COPY-BACK.
  E15 TAKES OWNERSHIP OF IT.** It was owned by no task before this revision and it
  is the single most dangerous omission in the plan. Three verified sub-sites:
  - the `seen: Vec<TProjection>` de-dup at **`:5440-5443`** (E11 already re-keys it
    to `(projection, ctr_type)`);
  - the `bake_ctr_table(…, ctr_prior_num, ctr_prior_denom)` call at
    **`:5445-5453`**, whose prior arguments come from the SINGLE global
    `ctr_prior_num` at `:3155`;
  - the copy-back loop at **`:5458-5472`**:
    ```rust
    for tree in &mut trees {
        for spec in &mut tree.ctr_splits {
            if let Some(table) = baked.tables.iter().find(|t| t.projection == spec.projection) {
                spec.shift       = table.shift;
                spec.scale       = table.scale;
                spec.prior_num   = table.prior_num;
                spec.prior_denom = table.prior_denom;
    ```
    `[VERIFIED: LOCAL, read verbatim]`. It is keyed on `projection` **alone**, and
    it **overwrites** the per-split prior that `crates/cb-train/src/tree.rs:3294-3295`
    already set correctly from `column.prior_num` / `column.prior_denom`.
- `calc_normalization(prior_num)` — the `(shift, scale)` derivation at
  `crates/cb-train/src/ctr/bake.rs:221-228` `[VERIFIED: LOCAL]`. After E15 it must be
  evaluated **per split** from that split's own prior, not copied off a table.

**CodeGraph evidence for ordering.** E05 must precede this task: without the
bit-equality firewall, a tie-break shift and a numerator regression are
indistinguishable in the oracle output. E14 must precede it so the Red is
data-backed.

**Red**
- File: `crates/cb-train/tests/ctr_borders_multiprior_oracle_test.rs`
- Test fn 1: `borders_multiprior_predictions_match_upstream_within_1e_minus_5`
  As E12's shape, with `simple_ctr_priors: vec![0.0, 0.5, 1.0]`.
- Test fn 2 (in `boosting_test.rs`):
  `candidate_expansion_emits_one_column_per_prior`
  **OBSERVATION CHANNEL (mandatory — channel (a), the extracted helper).**
  `materialized_ctr_features`, `structure_fold_columns` and
  `averaging_ctr_features` are `let` bindings **inside `train_inner`**
  (`crates/cb-train/src/boosting.rs:2555`) `[VERIFIED: LOCAL]`; a child-module test
  in `boosting_test.rs` can reach private *items*, **not** function locals, so this
  test **cannot** observe them as written today. Green step 0 (below) therefore
  **extracts the expansion** into
  `pub(crate) fn materialize_ctr_columns_for_perm(cat_columns: &[Vec<String>],
  absolute_projections: &[TProjection], ctr_candidates: &[CtrCandidate],
  params: &BoostParams, permutation: &[i32], target_class: &[i32],
  ctr_border_count: usize) -> CbResult<Vec<CtrFeatureColumn>>`, and **BOTH**
  `train_inner` loops (`boosting.rs:3237-3247` structure, `:3273-3282` averaging)
  call it. The test calls the SAME function — it does **not** re-implement it.
  Setup: 2 CTR-eligible cat columns, `max_ctr_complexity = 1`,
  `simple_ctr_priors = [0.0, 0.5, 1.0]`.
  Expected, over the helper's return value: `cols.len() == 2 projections *
  3 priors == 6`, the emitted `prior_num` sequence is exactly
  `[0.0, 0.5, 1.0, 0.0, 0.5, 1.0]`
  (upstream's `(ctrIdx, targetBorderIdx, priorIdx)` nesting order), and calling the
  helper a second time with the **averaging** permutation yields the SAME length
  and the SAME `(projection, prior)` sequence — the alignment invariant asserted,
  not assumed. (Because both `train_inner` loops go through this one function, the
  in-production alignment follows from the assertion rather than being restated.)
  **FORBIDDEN: a test that re-derives the expansion itself** (e.g. building its own
  `for proj { for prior { … } }` and comparing that to itself). Such a test is a
  **tautological guard** — it passes by construction no matter what `train_inner`
  does, and would leave R1 ("multi-prior expansion changes tie-breaks
  corpus-wide") completely unguarded.
- Test fn 3 (in `boosting_test.rs`, **the bake copy-back pin — CRITICAL**):
  `two_splits_on_one_projection_keep_distinct_priors_and_scales_after_the_bake`
  **OBSERVATION CHANNEL: (b), an integration-level observable** — the assertions
  read `CtrSplitSpec.{prior_num, scale}` off the **trained model returned by
  `train_inner`**, not any local, so no extraction is needed here.
  Setup: train a model on a corpus where projection `{0}` wins CTR splits at **two
  different priors** from `simple_ctr_priors = [0.0, 1.0]` (in different trees).
  Expected, after the whole `train_inner` run — i.e. **after** the
  `boosting.rs:5458-5472` copy-back has executed:
  ```rust
  let priors: Vec<f64> = all_ctr_splits_on_projection_0.iter().map(|s| s.prior_num).collect();
  let scales: Vec<f64> = all_ctr_splits_on_projection_0.iter().map(|s| s.scale).collect();
  assert!(priors.contains(&0.0) && priors.contains(&1.0),
      "the bake copy-back at boosting.rs:5458-5472 keys on `projection` alone and \
       OVERWRITES every split's prior with the first baked table's — both splits \
       must keep their OWN prior");
  assert_ne!(scales[0].to_bits(), scales[1].to_bits(),
      "shift/scale must be derived PER SPLIT from calc_normalization(spec.prior_num), \
       not copied off one shared table");
  ```
  **EXPECTED INITIAL FAILURE:** both splits report `prior_num == 0.0` and an
  identical `scale` — the exact silent defect this test exists to catch.
- **EXPECTED INITIAL FAILURE:** test fn 2 —
  ``assertion `left == right` failed: left: 2, right: 6`` (today
  `ctr_prior_num = priors.first()` yields exactly one column per projection,
  `boosting.rs:3155`). Test fn 1 fails numerically with a large max-divergence.
- Run: `cargo test -p cb-train --test ctr_borders_multiprior_oracle_test` and
  `cargo test -p cb-train --lib boosting::tests -- candidate_expansion` and
  `cargo test -p cb-train --lib boosting::tests -- distinct_priors`

**Green (minimal implementation intent).**
0. **EXTRACT THE OBSERVATION CHANNEL FIRST (it is what makes test fn 2 and E16's
   test fn 1 expressible at all).** Two `pub(crate) fn`s in
   `crates/cb-train/src/boosting.rs`, both pure and both called by `train_inner`:
   - `pub(crate) fn materialize_ctr_columns_for_perm(cat_columns: &[Vec<String>],
     absolute_projections: &[TProjection], ctr_candidates: &[CtrCandidate],
     params: &BoostParams, permutation: &[i32], target_class: &[i32],
     ctr_border_count: usize) -> CbResult<Vec<CtrFeatureColumn>>` — the body of the
     two materialization loops (`:3237-3247`, `:3273-3282`), which become one call
     each. This is the ONLY place the `(projection, prior)` — and, after E16, the
     `(projection, b, prior)` — product is built.
   - `pub(crate) fn cat_eligible_buckets_for(cat_columns: &[Vec<String>],
     eligible_absolute: &[usize]) -> Vec<Vec<u32>>` — the body of the
     `cat_eligible_buckets` build at `boosting.rs:3074`, which becomes one call.
     It takes `eligible_absolute` and NOT the expanded column list, which is
     precisely why its output cannot grow with the expansion.
   Both are behavior-preserving extractions: **no logic change in this step**, and
   the D-04 no-op proof (single-element prior list ⇒ byte-identical output) must
   still hold after step 0 alone.
1. In both materialization loops — now the single body of
   `materialize_ctr_columns_for_perm` — wrap the `materialize_ctr_feature` call in
   `for &prior in priors_for(candidate.is_simple)`, pushing one column per prior, in
   prior-list order, into both `structure_fold_columns[fold]` and
   `averaging_ctr_features`. **Leave `cat_eligible_buckets` exactly as built** — it
   is per CTR-eligible categorical FEATURE, not per column, and is consumed by an
   order-insensitive `.max()` (see above). **Do not touch the scorer** — a longer
   flat column list is all it needs.
2. **FIX THE BAKE BLOCK AT `crates/cb-train/src/boosting.rs:5437-5473` (three
   changes; the whole point of E15's ownership of it):**
   - **(a) Key the bake AND the copy-back lookup on `(projection, ctr_type)`, not
     on `projection` alone.** The `find(|t| t.projection == spec.projection)` at
     `:5464` becomes
     `find(|t| t.projection == spec.projection && t.ctr_type == spec.ctr_type)`.
     (`(projection, ctr_type)` is the same key E11 mandates for the `seen` de-dup;
     `target_border_idx` MUST NOT enter it — see E11 Green step 5.)
     **Intended role of the `find` after (b) and (c) — stated so the implementer
     has no choice to make:** the `find` **SURVIVES, purely as an existence gate**.
     Nothing is read out of `table` any more, so the lookup binds no value; write
     it as `if baked.tables.iter().any(|t| t.projection == spec.projection &&
     t.ctr_type == spec.ctr_type) { … }`. In words: **only splits with a baked
     `(projection, ctr_type)` table get a derived `shift`/`scale`; a split with no
     table keeps `0.0` / `1.0`.** Do NOT drop the lookup and derive
     unconditionally, and do NOT keep a `let Some(table)` binding that clippy will
     flag as unused.
     Consequence to record: `BakedCtrTable.{shift, scale, prior_num, prior_denom}`
     (`crates/cb-train/src/ctr/bake.rs:78-86`) become **informational-only in
     production** — `CtrData::from_baked` (`crates/cb-model/src/ctr_data.rs:313-331`)
     already ignores them. **They MUST stay** on the struct: the existing
     `crates/cb-train/tests/ctr_split_scoring_test.rs` and
     `crates/cb-train/src/ctr/final_ctr_test.rs` assertions read them.
   - **(b) STOP OVERWRITING `spec.prior_num` / `spec.prior_denom`.** Delete those
     two assignments from the copy-back loop. They are **already correct**: they
     arrive from `column.prior_num` / `column.prior_denom` via
     `crates/cb-train/src/tree.rs:3294-3295`, which carries the winning column's own
     prior. The copy-back was clobbering them with the first baked table's.
   - **(c) Compute `spec.shift` / `spec.scale` PER SPLIT** from
     `calc_normalization(spec.prior_num)` and `ctr_border_count`
     (`crates/cb-train/src/ctr/bake.rs:221-228`), rather than copying `table.shift` /
     `table.scale`. With multiple priors on one projection the table carries only
     one normalization and copying it is wrong for every other split.
   - The bake itself still passes a per-`(projection, ctr_type)` prior to
     `bake_ctr_table` at `:5445-5453`; the split-level normalization is what the
     copy-back now derives.

**Refactor constraints + required regression scope**
- **Constraint (the D-04 no-op proof):** with a single-element prior list the
  emitted column sequence must be byte-identical to pre-change. All 11 existing CTR
  oracles pin `Prior=0.5` (single) `[VERIFIED: LOCAL fixtures/*/config.json]`, so
  they are the regression gate — **not a formality; run them and read the numbers**.
- **Constraint:** the structure/averaging alignment invariant must be asserted by
  test fn 2, not merely preserved by convention.
- **Constraint (the copy-back):** after Green step 2, `crates/cb-train/src/boosting.rs`
  must contain **no** assignment of `spec.prior_num` or `spec.prior_denom` inside the
  `:5458-5472` loop, and its table lookup must test `ctr_type` as well as
  `projection`. `grep -n 'spec.prior_num =' crates/cb-train/src/boosting.rs` must
  return nothing.
- Regression scope: **all 11 CTR oracles + all 3 one-hot targets + E12 + E13**.

**Validation**
```bash
cargo test -p cb-train --lib boosting::tests
cargo test -p cb-train --test ctr_borders_multiprior_oracle_test
cargo test -p cb-train --test ctr_counter_simple_oracle_test --test ctr_btmv_simple_oracle_test
cargo test -p cb-train --test plain_ctr_oracle_test --test ordered_ctr_oracle_test \
  --test tensor_ctr_oracle_test --test tensor_ctr_e2e_oracle_test \
  --test s_order_ctr_bins_oracle_test --test ctr_split_scoring_test \
  --test ctr_feature_materialize_test --test multi_permutation_e2e_oracle_test \
  --test multi_permutation_fold_oracle_test
cargo test -p cb-model --test ctr_data_roundtrip_test --test fstr_ctr_oracle_test
cargo test -p cb-train --test one_hot_oracle_test --test one_hot_draw_accounting_test \
  --test device_one_hot_parity_test
cargo test -p cb-train
cargo clippy -p cb-train --all-targets
```

**Completion evidence.** `ctr_borders_multiprior` ≤1e-5 with the max-divergence
number recorded; the 6-column expansion + alignment test green; **test fn 3's
distinct-`prior_num`/distinct-`scale` assertion green after the copy-back fix**, and
`grep -n 'spec.prior_num =' crates/cb-train/src/boosting.rs` returning nothing;
**all 11 CTR oracles green.** The diff gate over the eleven SPEC-CTRT-18 oracle
targets (PLAN.md §3.2) is **per file**, in three categories:

| # | Oracle target file | Diff category | Owning task(s) |
|---|---|---|---|
| 1 | `crates/cb-train/tests/plain_ctr_oracle_test.rs` | **ZERO DIFF REQUIRED** | none |
| 2 | `crates/cb-train/tests/ordered_ctr_oracle_test.rs` | **ZERO DIFF REQUIRED** | none |
| 3 | `crates/cb-train/tests/tensor_ctr_oracle_test.rs` | **ZERO DIFF REQUIRED** | none |
| 4 | `crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs` | **ZERO DIFF REQUIRED** | none |
| 5 | `crates/cb-train/tests/multi_permutation_e2e_oracle_test.rs` | **ZERO DIFF REQUIRED** | none |
| 6 | `crates/cb-train/tests/multi_permutation_fold_oracle_test.rs` | **ZERO DIFF REQUIRED** | none |
| 7 | `crates/cb-model/tests/fstr_ctr_oracle_test.rs` | **ZERO DIFF REQUIRED** | none (but see the F08 `#[non_exhaustive]` note under the table) |
| 8 | `crates/cb-train/tests/s_order_ctr_bins_oracle_test.rs` | **MECHANICAL EDITS ONLY, ZERO ASSERTION CHANGES** | E09 (`materialize_ctr_feature` args at `:70` — 7 args → 9), E22 (the same site again — 9 args → 10, `extra_cat_columns`) |
| 9 | `crates/cb-train/tests/ctr_split_scoring_test.rs` | **MECHANICAL EDITS ONLY, ZERO ASSERTION CHANGES** | E09 (`target_border_idx: 0` at `:41`, `:68`; `materialize_ctr_feature` args at `:384`, `:394`), E11 (`bake_ctr_table` args at `:542`, `:576`, `:645`), E16 (five dropped args at `:99, :148, :191, :249, :305`), E22 (all five call sites again), F08 (the `cb_model::Model` literal at `:518` migrated to the `Model::new(..)` + builder form, forced by `#[non_exhaustive]`) |
| 10 | `crates/cb-train/tests/ctr_feature_materialize_test.rs` | **MECHANICAL EDITS ONLY, ZERO ASSERTION CHANGES** | E09 (widened `materialize_ctr_feature` calls + an ADDITIVE new test fn), E22 (ADDITIVE test fn 4 + the `extra_cat_columns` argument) |
| 11 | `crates/cb-model/tests/ctr_data_roundtrip_test.rs` | **MECHANICAL EDITS ONLY, ZERO ASSERTION CHANGES** | E11 (ADDITIVE test fns 2 and 4 + the compile-forced `build_final_ctr` argument at `:101`, `:138`, `:143`, `:163`) |

For rows 1–7, `git diff --stat` over those seven files must print **nothing**. For
rows 8–11 the permitted diff is **signature-driven argument/field edits and ADDITIVE
new test functions only**. **A diff that touches an EXISTING assertion in ANY of the
eleven — added, removed, weakened or reworded — is a STOP-AND-REPORT condition.**
At E15 the edits present in row 9 are E09's two `target_border_idx: 0` initializers
(`:41`, `:68`), E09's two widened `materialize_ctr_feature` calls (`:384`, `:394`)
and E11's three widened `bake_ctr_table` calls (`:542`, `:576`, `:645`); E16 adds
the five dropped `0` arguments (`:99, :148, :191, :249, :305`). Row 8 at E15 carries
E09's single widened `materialize_ctr_feature` call at `:70` and nothing else.
**F08 `#[non_exhaustive]` note (Part 2, far downstream of E15/E16):** F08 marks
`cb_model::Model` `#[non_exhaustive]`, which forbids struct-literal construction
from every *other* crate — including `crates/cb-model/tests/*.rs`, which are
separate crates. Row 7 (`fstr_ctr_oracle_test.rs`) contains ONE such literal, so
`#[non_exhaustive]` compile-forces a one-line constructor migration there. That
edit is **F08's**, is **mechanical only**, and **must change no assertion**; see
F08 in PLAN-PART2.md, which enumerates the full migration set and flags this
ZERO-DIFF collision as an OPEN item.

---

### E16 — Candidate expansion over `target_border_idx` + the `buckets_simple` gate

- **Specs:** SPEC-CTRT-12, SPEC-CTRT-06 (parity half); acceptance **A1**
- **Blocked by:** E15. **Blocks:** E17.
- **Parallelizable:** **NO** — owns `crates/cb-train/src/boosting.rs` and
  `crates/cb-train/src/tree.rs`.

**Goal / observable completion condition.** For a Buckets CTR at binclf, BOTH
`target_border_idx = 0` and `= 1` are emitted as candidate columns; the winning
column's `target_border_idx` reaches `CtrSplitSpec` (no longer the literal `0`);
`ctr_buckets_simple` passes at ≤1e-5 with both indices present.

**Files**
- Modify: `crates/cb-train/src/boosting.rs`, `crates/cb-train/src/tree.rs`,
  `crates/cb-train/src/boosting_test.rs`
- Modify: `crates/cb-train/tests/ctr_split_scoring_test.rs` — **mechanical, forced
  by the signature change in Green step 2.** Drop the `0` argument (the
  `target_border_idx` positional) from the FIVE
  `greedy_tensor_search_oblivious_with_ctr` call sites at `:99`, `:148`, `:191`,
  `:249`, `:305` `[VERIFIED: LOCAL grep -n]`. **CHANGE NO ASSERTION.** The
  `CtrFeatureColumn`s those tests build already carry `target_border_idx: 0` from
  E09, so no other edit is needed in this file.
  **NOTE:** this file is one of the eleven SPEC-CTRT-18 oracle targets; the edit is
  **purely mechanical** (delete one argument per call), and **weakening or deleting
  any assertion in `ctr_split_scoring_test.rs` is FORBIDDEN**. If a call no longer
  compiles for a reason **other than** (i) this task's dropped argument or (ii) a
  mechanical arity update that E09, E11, E22 or F08 explicitly authorizes in this
  same file — E09's `CtrFeatureColumn` initializers at `:41`/`:68` and
  `materialize_ctr_feature` arguments at `:384`/`:394`, E11's `bake_ctr_table`
  arguments at `:542`/`:576`/`:645`, E22's further widening of those same five
  sites, F08's `#[non_exhaustive]`-forced migration of the `cb_model::Model`
  literal at `:518` to the constructor form — then **STOP AND REPORT** rather than
  adjusting the test. **Any change
  that would touch an assertion is STOP AND REPORT regardless of which task
  appears to force it.**
  **The SAME exemption clause applies, word for word, to the SECOND mechanical
  oracle file `crates/cb-train/tests/s_order_ctr_bins_oracle_test.rs`:** the only
  authorized edits there are E09's widening of the `materialize_ctr_feature` call
  at `:70` (7 args → 9) and E22's further widening of that same site (9 args → 10).
  Anything else in that file is **STOP AND REPORT**.
- Create: `crates/cb-oracle/fixtures/ctr_buckets_simple/gen_fixtures.py`
- Create + COMMIT: `.../{X_cat.npy,y.npy,model.json,predictions.npy,config.json}`
- Create: `crates/cb-train/tests/ctr_buckets_simple_oracle_test.rs`

**Exact verified files/symbols to touch**
- `ECtrType::target_border_count(2)` (E01) → `2` for Buckets, `1` otherwise —
  the loop bound.
- `CtrSplitSpec.target_border_idx: usize` ("The Buckets per-class numerator
  selector (default `0`)") at `crates/cb-train/src/tree.rs:178-179`
  `[VERIFIED: LOCAL, read verbatim]`.
- `CtrSplitSpec { … target_border_idx: 0 … }` hard-coded at
  `crates/cb-train/src/boosting.rs:1943` (inside `ctr_splits_for_tree`)
  `[VERIFIED: LOCAL]`.
- **The literal `0` passed as `target_border_idx` to
  `greedy_tensor_search_oblivious_with_ctr` at `crates/cb-train/src/boosting.rs:4662`**
  (the bare `0,` between `n,` and the `model_size_reg_default()` comment)
  `[VERIFIED: LOCAL sed -n '4640,4680p'; the argument is at :4662, NOT :4658]` —
  this is the whole-tree parameter that must become **per-column**, and it is
  **DELETED** by this task (Green step 2).
- **`crates/cb-train/src/tree.rs:3296` MUST READ `column.target_border_idx`.**
  Today `CtrSplitSpec { projection: column.projection.clone(), ctr_type:
  column.ctr_type, prior_num: column.prior_num, prior_denom: column.prior_denom,
  target_border_idx, … }` at `crates/cb-train/src/tree.rs:3291-3300` sources
  `ctr_type` and both priors **from the column** but `target_border_idx` from the
  **whole-tree function parameter** `[VERIFIED: LOCAL, read verbatim]`. That is the
  inconsistency SPEC-CTRT-12 closes: line `:3296` becomes
  `target_border_idx: column.target_border_idx,`.
- `CtrFeatureColumn.target_border_idx` (added by E09) is the per-column carrier;
  `tree.rs` already copies `column.ctr_type` onto the chosen split
  `[VERIFIED: research §F.2 — "tree.rs:3291 copies column.ctr_type already"]`, so
  the same site copies `column.target_border_idx`.
- **`crates/cb-train/src/boosting.rs:5437-5473` — THE BAKE BLOCK AND ITS COPY-BACK.
  E16 SHARES OWNERSHIP OF IT WITH E15** (E15 lands the fix; E16 must not regress
  it). The lookup key stays `(projection, ctr_type)` — **`target_border_idx` MUST
  NOT be added to it** (E11 Green step 5), because one Buckets table serves both
  `b = 0` and `b = 1` and `ctr_base_key`
  (`crates/cb-model/src/ctr_data.rs:299`) carries no index. `spec.prior_num` /
  `spec.prior_denom` remain un-overwritten and `spec.shift` / `spec.scale` remain
  derived per split from `calc_normalization(spec.prior_num)`.
- `cat_eligible_buckets: &[Vec<u32>]` — the `model_size_reg` input.
  **`cat_eligible_buckets` (`crates/cb-train/src/boosting.rs:3074`, passed at
  `:4669`) is one `perfect_hash_bins` column per CTR-eligible categorical feature
  (`eligible_absolute`), consumed by an order-insensitive `.max()` at
  `crates/cb-train/src/tree.rs:3026`. It is NOT index-aligned with `ctr_features`
  and MUST NOT grow with the `(projection, b, prior)` expansion — leave it exactly
  as built.** `[VERIFIED: LOCAL boosting.rs:3074, :4669; tree.rs:3026 and the doc
  comment at tree.rs:2984-2987]` Growing it (or re-deriving it per CTR candidate)
  would change `phantom_mixed_bucket_count` and hence `model_size_reg`'s
  cat-feature weight, silently moving split choice.
- Apply side needs no change: `CtrValueTable::numerator_denominator`'s Buckets arm
  already reads `counts.get(target_border_idx)`
  `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs:242-248]`.

**Fixture configuration.** The §3 isolating set with
`simple_ctr = ["Buckets:Prior=0.5"]`, `combinations_ctr = []`,
`max_ctr_complexity = 1`.

**MANDATORY anti-false-pass guard (generator) — the strongest in the plan:**
```python
buckets = [c for c in ctrs if c["ctr_type"] == "Buckets"]
assert buckets, "no Buckets CTR descriptor — fixture is vacuous"
idxs = {c["target_border_idx"] for c in buckets}
assert idxs == {0, 1}, (
    f"buckets_simple requires BOTH target_border_idx 0 and 1; model.json has {idxs}. "
    "Without both, SPEC-CTRT-12 is untestable.")
assert predictions.std() > 1e-6
```
Plus the §3 corpus-cleanliness guard. **Empirically confirmed available:** a real
`Buckets:Prior=0.5` model emits descriptors with `target_border_idx` **0 AND 1** on
the same projection, whereas Borders/BTMV/Counter emit only `0`
`[VERIFIED: research §A.1, EXPERIMENT probe5.py]`.

**Red**
- File: `crates/cb-train/src/boosting_test.rs`
- Test fn 1: `buckets_expansion_emits_both_target_border_indices`
  **OBSERVATION CHANNEL (mandatory — channel (a), E15's extracted helpers).** Both
  `cat_eligible_buckets` (`crates/cb-train/src/boosting.rs:3074`) and the column
  list are `let` bindings **inside `train_inner`** and are unreachable from a
  child-module test `[VERIFIED: LOCAL]`. This test therefore observes them through
  the two `pub(crate) fn`s **E15 Green step 0 extracted** —
  `materialize_ctr_columns_for_perm(..)` for the columns and
  `cat_eligible_buckets_for(cat_columns, eligible_absolute)` for the bin columns.
  E16 extends the FIRST helper's body over `(projection, b, prior)`; it adds no new
  channel. The test **calls** both helpers; it re-implements neither.
  Setup: 1 CTR-eligible cat column, `simple_ctr = Buckets`,
  `simple_ctr_priors = [0.5]`, `max_ctr_complexity = 1`.
  Expected: `materialize_ctr_columns_for_perm(..)` returns 2 columns, with
  `target_border_idx` sequence `[0, 1]`, and
  `assert_ne!(cols[0].bins, cols[1].bins)` — the anti-vacuity guard proving the
  index genuinely changes the column.
  **Plus the `cat_eligible_buckets` no-growth pin, made FALSIFIABLE:** in the SAME
  test, with the SAME inputs, assert
  ```rust
  assert_eq!(cat_eligible_buckets_for(&cat_columns, &eligible_absolute).len(), 1,
      "one bin column per CTR-ELIGIBLE CATEGORICAL FEATURE — never per emitted CTR column");
  assert_eq!(cols.len(), 2,
      "…while the (projection, b, prior) product HAS grown");
  ```
  i.e. the two lengths are pinned to **different** numbers from the **same**
  fixture, so a change that made `cat_eligible_buckets` track the expansion fails
  immediately. Also assert it is **BYTE-UNCHANGED** against the un-expanded
  configuration (`simple_ctr = Borders`, which emits 1 column):
  `assert_eq!(cat_eligible_buckets_for(..) /* Buckets run */,
  cat_eligible_buckets_for(..) /* Borders run */)`, element for element.
  **FORBIDDEN: re-deriving `cat_eligible_buckets` inside the test** (e.g. rebuilding
  the `perfect_hash_bins` columns from `eligible_absolute` in the test body and
  comparing that to itself). The re-derived expression does not depend on the
  prior/border expansion at all, so the comparison is **tautological** and the pin
  can never fail — while R11 calls this "the most fragile thing W3 can break".
  **Do NOT assert
  `cat_eligible_buckets.len() == ctr_features.len()`** — that invariant is false
  (an earlier draft of this plan mandated it; it is hereby retracted), and making
  it pass would require duplicating or re-deriving the bin columns, which changes
  `phantom_mixed_bucket_count` and hence split choice.
- Test fn 2: `borders_expansion_emits_exactly_one_target_border_index`
  Same setup with `simple_ctr = Borders` ⇒ exactly 1 column,
  `target_border_idx == 0` — the D-04 no-op proof
  (`target_border_count(2) == 1` for Borders).
- File: `crates/cb-train/tests/ctr_buckets_simple_oracle_test.rs`
- Test fn 3: `buckets_simple_predictions_match_upstream_within_1e_minus_5`
- Test fn 4: `buckets_model_carries_a_split_at_target_border_idx_one`
  Expected: some tree's `ModelSplit::Ctr` has `target_border_idx == 1` — proving
  the per-column index reached the split spec rather than being reset to `0`.
- **EXPECTED INITIAL FAILURE:** test fn 1 —
  ``assertion `left == right` failed: left: 1, right: 2`` (E15 expands only over
  priors); test fn 4 — ``assertion failed: any target_border_idx == 1``, because
  `boosting.rs:4662` passes the literal `0`, `tree.rs:3296` reads that whole-tree
  parameter instead of the column, and `boosting.rs:1943` hard-codes `0`.
- Run: `cargo test -p cb-train --lib boosting::tests -- buckets_expansion` and
  `cargo test -p cb-train --test ctr_buckets_simple_oracle_test`

**Green (minimal implementation intent).**
1. Inside **`materialize_ctr_columns_for_perm`** — E15 Green step 0's `pub(crate)`
   helper, which is the single body both `train_inner` materialization loops now
   call, and the observation channel test fn 1 asserts through — nest
   `for b in 0..ctr_type.target_border_count(classes)` **outside** the prior loop,
   matching upstream's `(ctrIdx, targetBorderIdx, priorIdx)` nesting
   (`greedy_tensor_search.cpp:400-428`); pass `b` to `materialize_ctr_feature`.
   **Do not re-inline the helper** and do not add a second expansion site;
   `cat_eligible_buckets_for` is untouched by this task.
2. **The whole-tree `target_border_idx` parameter is DELETED, not made optional.**
   Concretely, and with no remaining choice for the implementer:
   - `crates/cb-train/src/tree.rs:3296` reads **`column.target_border_idx`**
     (matching how `:3294-3295` already read `column.prior_num` /
     `column.prior_denom` and `:3293` read `column.ctr_type`);
   - the `target_border_idx` **parameter** of
     `greedy_tensor_search_oblivious_with_ctr` (`crates/cb-train/src/tree.rs:3237`)
     is **DELETED** from the signature;
   - its literal `0` **argument** at `crates/cb-train/src/boosting.rs:4662` is
     **DELETED** from the call.
   There is no fallback path and no second source: after this step the value exists
   in exactly one place, `CtrFeatureColumn.target_border_idx`.
   **Compile fallout, owned by this task (see Files):** besides `boosting.rs:4653`,
   the function is called at **five sites in
   `crates/cb-train/tests/ctr_split_scoring_test.rs` (`:99, :148, :191, :249,
   :305`)** `[VERIFIED: LOCAL grep -n]`. Drop the `0` argument at each. That is the
   ONLY edit permitted in that file **by this task** — E09, E11 and E22 own the
   other mechanical arity edits there — **CHANGE NO ASSERTION.**
3. **`ctr_splits_for_tree` KEEPS `target_border_idx: 0`** at
   `crates/cb-train/src/boosting.rs:1943` — as a **deliberate, tested constant**.
   Its signature is `(candidates: &[CtrCandidate], priors: &[f64])`
   `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:1929]`, `CtrCandidate {
   projection, is_simple }` carries no index, and **no materialized column is in
   scope** — so it structurally cannot read one. It is reached only from its single
   caller at `:5318` on the `!has_ctr` branch, i.e. the **no-CTR-candidate
   fallback**, where no column exists by construction. Record that rationale as a
   doc comment on the function, and keep E03's characterization assertion
   `target_border_idx == 0` in place as the pin. (An earlier draft of this plan said
   step 3 should "take the index from the candidate column"; that is **impossible**
   and is hereby retracted.)

**Refactor constraints + required regression scope**
- Constraint: for every NON-Buckets type `target_border_count(2) == 1`, so the
  column count is unchanged and the existing oracles stay byte-identical.
- Constraint: the structure/averaging alignment invariant (E15) must hold over the
  widened `(projection, b, prior)` product — extend E15's alignment assertion.
- **Constraint (`cat_eligible_buckets`):** `cat_eligible_buckets`
  (`crates/cb-train/src/boosting.rs:3074`, passed at `:4669`) is one
  `perfect_hash_bins` column per CTR-**eligible categorical feature**
  (`eligible_absolute`), consumed by an order-insensitive `.max()` at
  `crates/cb-train/src/tree.rs:3026`. It is **NOT** index-aligned with
  `ctr_features` and **MUST NOT** grow with the `(projection, b, prior)` expansion
  — **leave it exactly as built.** Test fn 1 pins this: assert
  `cat_eligible_buckets` is **byte-unchanged** across the expansion (see Red).
- **Constraint (the bake copy-back, shared with E15):** do not regress E15's fix at
  `crates/cb-train/src/boosting.rs:5437-5473`. The lookup key stays
  `(projection, ctr_type)`; `target_border_idx` MUST NOT be added to it or to
  `ctr_base_key`; `spec.prior_num` / `spec.prior_denom` stay un-overwritten;
  `spec.shift` / `spec.scale` stay derived per split from
  `calc_normalization(spec.prior_num)`. Re-run E15's test fn 3.
- Regression scope: **all 11 CTR oracles + 3 one-hot targets + E12 + E13 + E15**.

**Validation**
```bash
cargo test -p cb-train --lib boosting::tests
.venv/bin/python crates/cb-oracle/fixtures/ctr_buckets_simple/gen_fixtures.py
git status --porcelain crates/cb-oracle/fixtures   # ONLY ctr_buckets_simple/*
cargo test -p cb-train --test ctr_buckets_simple_oracle_test
cargo test -p cb-train --test ctr_borders_multiprior_oracle_test \
  --test ctr_counter_simple_oracle_test --test ctr_btmv_simple_oracle_test
cargo test -p cb-train --test plain_ctr_oracle_test --test ordered_ctr_oracle_test \
  --test tensor_ctr_oracle_test --test tensor_ctr_e2e_oracle_test \
  --test s_order_ctr_bins_oracle_test --test ctr_split_scoring_test \
  --test ctr_feature_materialize_test --test multi_permutation_e2e_oracle_test \
  --test multi_permutation_fold_oracle_test
cargo test -p cb-model --test ctr_data_roundtrip_test --test fstr_ctr_oracle_test
cargo test -p cb-train --test one_hot_oracle_test --test one_hot_draw_accounting_test \
  --test device_one_hot_parity_test
cargo test -p cb-train -p cb-model
cargo clippy -p cb-train --all-targets
```

**Completion evidence.** The generator's `idxs == {0, 1}` assertion passing; four
tests green including the `assert_ne!` bins-differ guard, the
`cat_eligible_buckets` byte-unchanged pin and the `target_border_idx == 1` split
assertion; **all 11 CTR oracles + 3 one-hot targets green.** The diff gate over the
eleven SPEC-CTRT-18 oracle targets (PLAN.md §3.2) is **per file**, exactly as in
E15:

| # | Oracle target file | Diff category | Owning task(s) |
|---|---|---|---|
| 1 | `crates/cb-train/tests/plain_ctr_oracle_test.rs` | **ZERO DIFF REQUIRED** | none |
| 2 | `crates/cb-train/tests/ordered_ctr_oracle_test.rs` | **ZERO DIFF REQUIRED** | none |
| 3 | `crates/cb-train/tests/tensor_ctr_oracle_test.rs` | **ZERO DIFF REQUIRED** | none |
| 4 | `crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs` | **ZERO DIFF REQUIRED** | none |
| 5 | `crates/cb-train/tests/multi_permutation_e2e_oracle_test.rs` | **ZERO DIFF REQUIRED** | none |
| 6 | `crates/cb-train/tests/multi_permutation_fold_oracle_test.rs` | **ZERO DIFF REQUIRED** | none |
| 7 | `crates/cb-model/tests/fstr_ctr_oracle_test.rs` | **ZERO DIFF REQUIRED** | none (but see the F08 `#[non_exhaustive]` note under the table) |
| 8 | `crates/cb-train/tests/s_order_ctr_bins_oracle_test.rs` | **MECHANICAL EDITS ONLY, ZERO ASSERTION CHANGES** | E09 (`materialize_ctr_feature` args at `:70` — 7 args → 9), E22 (the same site again — 9 args → 10, `extra_cat_columns`) |
| 9 | `crates/cb-train/tests/ctr_split_scoring_test.rs` | **MECHANICAL EDITS ONLY, ZERO ASSERTION CHANGES** | E09 (`target_border_idx: 0` at `:41`, `:68`; `materialize_ctr_feature` args at `:384`, `:394`), E11 (`bake_ctr_table` args at `:542`, `:576`, `:645`), E16 (five dropped args at `:99, :148, :191, :249, :305`), E22 (all five call sites again), F08 (the `cb_model::Model` literal at `:518` migrated to the `Model::new(..)` + builder form, forced by `#[non_exhaustive]`) |
| 10 | `crates/cb-train/tests/ctr_feature_materialize_test.rs` | **MECHANICAL EDITS ONLY, ZERO ASSERTION CHANGES** | E09 (widened `materialize_ctr_feature` calls + an ADDITIVE new test fn), E22 (ADDITIVE test fn 4 + the `extra_cat_columns` argument) |
| 11 | `crates/cb-model/tests/ctr_data_roundtrip_test.rs` | **MECHANICAL EDITS ONLY, ZERO ASSERTION CHANGES** | E11 (ADDITIVE test fns 2 and 4 + the compile-forced `build_final_ctr` argument at `:101`, `:138`, `:143`, `:163`) |

For rows 1–7, `git diff --stat` over those seven files must print **nothing**. For
rows 8–11 the permitted diff is **signature-driven argument/field edits and ADDITIVE
new test functions only**. **A diff that touches an EXISTING assertion in ANY of the
eleven — added, removed, weakened or reworded — is a STOP-AND-REPORT condition.**
At E16 row 9 additionally carries this task's five dropped `0` arguments
(`:99, :148, :191, :249, :305`); row 8 still carries only E09's single widened
`materialize_ctr_feature` call at `:70` (E22's second widening of that site comes
later, in W5). The F08 `#[non_exhaustive]` note under E15's copy of this table
applies here unchanged.

---

### E17 — `mixed_simple_vs_combo` end-to-end gate for `is_simple` routing

- **Specs:** SPEC-CTRT-10 (parity half); acceptance **A5**
- **Blocked by:** E16. **Blocks:** none.
- **Parallelizable:** **YES** with W4/W5 tasks — owns only a new fixture directory
  and a new test target; no production change.

**Goal / observable completion condition.** A fixture whose SIMPLE and COMBINATION
CTRs use **different types and different priors** proves end to end that the
`is_simple` discriminator routes correctly.

**Files**
- Create: `crates/cb-oracle/fixtures/ctr_mixed_simple_vs_combo/gen_fixtures.py`
- Create + COMMIT: `.../{X_cat.npy,y.npy,model.json,predictions.npy,config.json}`
- Create: `crates/cb-train/tests/ctr_mixed_simple_vs_combo_oracle_test.rs`

**Exact verified files/symbols to touch (read-only)**
- The discriminator: `CtrCandidate.is_simple` from `TProjection::is_simple()`
  `[VERIFIED: CODEGRAPH candidates.rs:151-157,194]`. Upstream's rule is
  `GetCtrInfo(projection)`: single-cat projection → `SimpleCtrs`, else `TreeCtrs`
  (`ctr_helper.h:52-62`) `[VERIFIED: research §C]`.
- **Two caveats to record as doc comments on `TProjection::is_simple`
  (`crates/cb-train/src/projection.rs:144-146`), NOT to implement:**
  (a) upstream checks `PerFeatureCtrs` FIRST for a single-cat projection —
  `per_feature_ctr` is unsupported here, so the mapping is
  "`simple_ctr` unless a per-feature override exists (unsupported)";
  (b) upstream's `IsSingleCatFeature()` also requires `BinFeatures.empty() &&
  OneHotFeatures.empty()` (`projection.h:102-104`). This repo's `TProjection` holds
  ONLY `cat_features: Vec<usize>` `[VERIFIED: research §C caveat 2]`, so
  `is_simple() == (cat.len() == 1)` is currently **exactly** equivalent — but if a
  later phase adds bin/one-hot projection members the predicate MUST widen. Add the
  doc note now so the equivalence cannot be silently broken.

**Fixture configuration.** The §3 isolating set with **3** cat columns
(cardinalities 6, 5, 4, all `> one_hot_max_size = 1`),
`simple_ctr = ["Buckets:Prior=0.5"]`, `combinations_ctr = ["Counter:Prior=0.25"]`,
`max_ctr_complexity = 2`.

**MANDATORY anti-false-pass guard (generator):**
```python
simple  = [c for c in ctrs if len(c["elements"]) == 1]
combo   = [c for c in ctrs if len(c["elements"]) >= 2]
assert simple and combo, (
    f"mixed fixture needs BOTH a simple and a combination CTR; got "
    f"{len(simple)} simple / {len(combo)} combination")
assert {c["ctr_type"] for c in simple} == {"Buckets"}, \
    f"simple CTRs must be Buckets, got {{c['ctr_type'] for c in simple}}"
assert {c["ctr_type"] for c in combo} == {"Counter"}, \
    f"combination CTRs must be Counter, got {{c['ctr_type'] for c in combo}}"
assert predictions.std() > 1e-6
```
Plus the §3 corpus-cleanliness guard.

**Red**
- File: `crates/cb-train/tests/ctr_mixed_simple_vs_combo_oracle_test.rs`
- Test fn 1: `mixed_simple_vs_combo_predictions_match_upstream_within_1e_minus_5`
  Params: `simple_ctr: Buckets`, `simple_ctr_priors: vec![0.5]`,
  `combinations_ctr: Counter`, `combinations_ctr_priors: vec![0.25]`,
  `max_ctr_complexity: 2`.
- Test fn 2: `simple_projections_are_baked_as_buckets_and_combinations_as_counter`
  Expected: for every baked table, `t.projection.cat_features().len() == 1` ⇒
  `t.ctr_type == ECtrType::Buckets.as_i8()` **and** `t.prior_num == 0.5`;
  `len() >= 2` ⇒ `t.ctr_type == ECtrType::Counter.as_i8()` **and**
  `t.prior_num == 0.25`. Plus `assert!(has_simple && has_combo)` — anti-vacuity.
- **EXPECTED INITIAL FAILURE:** `No such file or directory` before generation;
  after generation and before E10, test fn 2 fails with
  ``left: 0, right: 1`` (every table baked as Borders) and
  ``left: 0.25, right: 0.5`` (the combination prior governing the simple table —
  the exact bug at `boosting.rs:3155`).
- Run: `cargo test -p cb-train --test ctr_mixed_simple_vs_combo_oracle_test`

**Green (minimal implementation intent).** No production change beyond the two
`projection.rs` doc comments (delivered by E09/E10/E11/E16).

**Localization ladder (STOP AND REPORT at the first hit).** If ≤1e-5 fails ONLY at
`max_ctr_complexity = 2`, this is the **ORD-06/ORD-07 combination-CTR candidate
gating bug**, tracked separately at `.planning/phases/24-ctr-split-search-correctness/`.
**STOP AND REPORT. DO NOT FIX IT HERE.** Confirm by re-running the same fixture
regenerated at `max_ctr_complexity = 1` (which yields no combination CTRs).

**Refactor constraints + required regression scope**
- Regression scope: all 11 CTR oracles + E12/E13/E15/E16 fixtures.

**Validation**
```bash
.venv/bin/python crates/cb-oracle/fixtures/ctr_mixed_simple_vs_combo/gen_fixtures.py
git status --porcelain crates/cb-oracle/fixtures
cargo test -p cb-train --test ctr_mixed_simple_vs_combo_oracle_test
cargo test -p cb-train --test ctr_buckets_simple_oracle_test \
  --test ctr_borders_multiprior_oracle_test --test ctr_counter_simple_oracle_test \
  --test ctr_btmv_simple_oracle_test
cargo test -p cb-train -p cb-model
```

**Completion evidence.** Three generator assertions passing; both tests green with
the recorded max-divergence; the two `projection.rs` doc caveats present.

---

> **Continues in `./PLAN-W4-W5.md`** (waves W4 `.cbm` mean codec, tasks E18–E20,
> and W5 `counter_calc_method`, tasks E21–E23) and `./PLAN-PART2.md`
> (Part 2, tasks F00–F23, plus the SPEC-ID coverage tables, risk register and
> unresolved-blocker list).
