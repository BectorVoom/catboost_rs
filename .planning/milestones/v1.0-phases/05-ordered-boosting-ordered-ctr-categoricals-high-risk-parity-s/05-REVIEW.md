---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
reviewed: 2026-06-14T00:00:00Z
depth: standard
files_reviewed: 12
files_reviewed_list:
  - crates/cb-model/src/apply.rs
  - crates/cb-model/src/ctr_data.rs
  - crates/cb-model/src/lib.rs
  - crates/cb-model/src/model.rs
  - crates/cb-train/src/boosting.rs
  - crates/cb-train/src/ctr/bake.rs
  - crates/cb-train/src/ctr/ctr_feature.rs
  - crates/cb-train/src/ctr/mod.rs
  - crates/cb-train/src/fold.rs
  - crates/cb-train/src/lib.rs
  - crates/cb-train/src/permutation.rs
  - crates/cb-train/src/tree.rs
findings:
  critical: 0
  warning: 3
  info: 5
  total: 8
status: issues_found
---

# Phase 5: Code Review Report (ORD-05 CTR leaf-value gap closure, plans 05-12/05-13/05-14)

**Reviewed:** 2026-06-14
**Depth:** standard
**Files Reviewed:** 12
**Status:** issues_found
**Scope:** diff `03e3b1f..HEAD` for the 12 listed files only.

## Summary

Reviewed the ORD-05 CTR leaf-value gap-closure diff: the identity-`Folds[0]` +
pre-averaging draw rework in `fold.rs`, the CTR-aware oblivious structure search
in `tree.rs`, the two-materialization leaf-value wiring + whole-set `ctr_data`
bake in `boosting.rs`/`bake.rs`, and the Scale/Shift threading + canonical key
sharing in `apply.rs`/`ctr_data.rs`/`model.rs`.

The parity-critical machinery is sound for the in-scope `permutation_count = 1`
config that the `tensor_ctr_e2e` hard gate validates:
- bake key reconstructs the apply key byte-for-byte — both via `ctr_base_key`
  over the sorted projection (`TProjection::from_features` sorts), both folding
  per-feature `calc_cat_feature_hash` through `fold_cat_hash` from the `0` seed;
- the Borders class-count ordering `[N0, N1]` is consistent between
  `build_final_ctr` (flattened `[bucket0_class0, bucket0_class1, …]`) and
  `calc_for_hash` (`n0=counts[0]`, `n1=counts[1]`, `(n1, n0+n1)`);
- the online-bin and inference-`Calc` spaces reconcile through `(Shift, Scale)`
  when `PriorDenom == 1` (`(ctr+shift)/norm*borderCount` == `(ctr+Shift)*Scale`
  with `Scale = borderCount/norm`);
- Scale/Shift now thread through BOTH the table-found and not-found apply
  branches (the prior hardcoded `0.0/1.0` is gone on both);
- NO `unwrap`/`expect`/`panic`/`unreachable` in any reviewed production file —
  all lookups are checked `.get`; the `l2_split_score` terms are non-negative so
  the `cat_feature_weight` multiplier cannot invert a penalty into a bonus.

Both changed crates `cargo check` clean.

No Critical defects (no new network/auth/file surface, no data-loss path, no
crash path). The findings center on one real correctness risk: the new draw-order
logic and the structure/averaging permutation selection are validated ONLY for
`permutation_count = 1`, while the production DEFAULT is `permutation_count = 4`,
and the inline draw-order contract is incorrect for that default.

## Warnings

### WR-01: Pre-averaging draw fires before the first LEARNING fold (not the averaging fold) when `permutation_count > 1` — default is 4

**File:** `crates/cb-train/src/fold.rs:259-286`
**Issue:** The learning-permutation-needed branch emits the identity for `idx ==
0` and inserts exactly ONE pre-draw (`rng.gen_rand()`) before the FIRST
non-identity shuffle, guarded by `first_real_shuffle`, then shuffles every
subsequent fold. For `permutation_count = 1` this is correct: `learning_folds =
1`, so `idx == 1` IS the averaging fold and the pre-draw precedes the averaging
shuffle (matching the "averaging shuffle starts at RNG call-count 1" upstream
claim). But the production default is `permutation_count = 4`
(`boosting.rs:227-229` `permutation_count_default()` returns `4`), giving
`learning_folds = 3`. There the pre-draw lands before `idx == 1` — the FIRST
LEARNING fold — and three learning shuffles are drawn before the averaging fold
at `idx == 4`. The doc comment at `fold.rs:264-271` asserts the draw is "between
the identity learning Folds[0] and the AveragingFold's Shuffle," which is FALSE
once intervening learning folds exist. Consequently `boosting.rs:1147-1150`
(`find(|f| f.is_averaging)`) pulls the averaging permutation at an unvalidated RNG
call-count for the default config, and the structure search uses only the FIRST
learning fold (`boosting.rs:1140-1143`, the identity), ignoring learning folds
1..3. The `tensor_ctr_e2e` gate exercises ONLY `permutation_count = 1`, so the
default path's bit-exact parity is untested and the comment is misleading.
**Fix:** Confirm against upstream whether the single pre-`GenRand` precedes the
AVERAGING shuffle specifically. If so, fire it immediately before the averaging
fold regardless of how many learning folds precede it, and correct the comment:
```rust
let is_averaging_fold = idx == learning_folds;
let permutation: Vec<i32> = if idx == 0 {
    (0..n).map(|i| i as i32).collect()
} else {
    if is_averaging_fold && needs_pre_averaging_draw {
        rng.gen_rand(); // one GenRand immediately before the averaging shuffle
    }
    shuffle_in_place(n, &mut rng)
};
```
Otherwise gate the new discipline to `permutation_count == 1` and add an oracle
covering `permutation_count > 1`, or document the restriction at the `train_cat`
boundary.

### WR-02: Whole-set bake dedups by projection only, but the apply table key is `(ctr_type, projection)`

**File:** `crates/cb-train/src/boosting.rs:1624-1648`
**Issue:** The bake loop tracks distinct chosen splits by projection alone
(`if !seen.iter().any(|p| p == &spec.projection)`, line 1630) and the
Shift/Scale/prior copy-back also matches on projection only
(`.find(|t| t.projection == spec.projection)`, line 1652). The apply-side table
key is `(ctr_type, projection)` (`ctr_base_key`, `apply.rs:129`;
`ctr_data.rs:303`). Today every `CtrSplitSpec.ctr_type` is `Borders` (the only
type `materialize_ctr_feature` emits) and `bake_ctr_table` hardcodes
`ECtrType::Borders` (`bake.rs:227`), so the mismatch is latent. But if a second
CTR type is ever scored for the same projection: (1) only ONE Borders table is
baked, (2) the second type's split gets Borders prior/Scale/Shift copied onto it,
and (3) its apply lookup (`ctr:type=<other>:proj=…`) misses the table and
silently falls to the not-found branch — a silent wrong-prediction path with no
error.
**Fix:** Key the `seen` dedup and the copy-back on the full `(ctr_type,
projection)` tuple, and bake a table per distinct `(ctr_type, projection)`,
threading `spec.ctr_type` into `bake_ctr_table` instead of the hardcoded
`ECtrType::Borders`.

### WR-03: Bake uses the global prior, ignoring (and overwriting) each split's own prior

**File:** `crates/cb-train/src/boosting.rs:1631-1640,1649-1659`
**Issue:** `bake_ctr_table(... ctr_prior_num, ctr_prior_denom)` passes the SINGLE
global prior (`ctr_prior_num = combinations_ctr_priors.first().copied().unwrap_or(0.5)`,
line 1164) for every chosen split, then copies `table.prior_num`/`table.prior_denom`
(the global prior) BACK onto each `CtrSplitSpec` (lines 1655-1656), OVERWRITING
the per-column prior the structure search recorded (`tree.rs:803-804`).
`calc_normalization` is prior-dependent, so a future multi-prior candidate set
(`combinations_ctr_priors` with >1 entry — the API accepts it) would bake the
WRONG `(Shift, Scale)` and prior for any split whose prior differs from
`priors[0]`. Inert for the in-scope single-prior fixture.
**Fix:** Bake with the split's own prior (`spec.prior_num`, `spec.prior_denom`)
and copy back only `shift`/`scale`, leaving `spec.prior_num`/`spec.prior_denom`
intact.

## Info

### IN-01: `ECtrType::from_i8(...).unwrap_or(ECtrType::Borders)` silently coerces unknown types

**File:** `crates/cb-model/src/ctr_data.rs:311`
**Issue:** `CtrData::from_baked` maps an unrecognized `t.ctr_type` i8 to `Borders`
rather than surfacing the inconsistency. Since `bake_ctr_table` only emits
`Borders.as_i8()` this cannot misfire today, but it would mask a future bake/lift
type mismatch as a silent wrong-table selection.
**Fix:** Make `from_baked` fallible and propagate a typed error, or assert the
expected discriminant rather than defaulting.

### IN-02: Hardcoded `2` target-class count in the bake call

**File:** `crates/cb-train/src/boosting.rs:1635`
**Issue:** `bake_ctr_table(..., 2, ...)` hardcodes the binclf class count
(`// binclf target-class count`). The same magic `2` and a `> 0.5` binarization
recur in the materialization path. Acceptable for the binclf-only scope, but it
pins the bake to binary classification with no compile-time tie to the actual
target arity.
**Fix:** Derive the class count from a named constant or the resolved target type
so a future multiclass path cannot silently bake a 2-class table.

### IN-03: `bake_ctr_table` passes `classes` as `target_border_count`

**File:** `crates/cb-train/src/ctr/bake.rs:191`
**Issue:** `accumulate_online(&key_refs, &target_class_n, &target_zero, classes,
classes)` supplies `classes` (2) for BOTH the `classes` and the
`target_border_count` parameters. For binclf Borders the conventional target
border count is `1`, not `2`. The `tensor_ctr_e2e` gate passes (the producer is
internally consistent), but the dual use of `classes` is non-obvious and easy to
mis-read when the producer signature is reused for another type.
**Fix:** Bind a named `target_border_count` with a justifying comment and pass it
explicitly.

### IN-04: Apply compares a continuous CTR value to an integer bin border; train truncates first

**File:** `crates/cb-model/src/apply.rs:188`, `crates/cb-train/src/ctr/ctr_feature.rs:218`, `crates/cb-train/src/tree.rs` (`passes_ctr_aware`)
**Issue:** The structure/averaging assignment compares the TRUNCATED integer bin
(`f64::from(bin) > border`; `bins` are `bin_f.trunc()`), while the apply path
compares the CONTINUOUS scaled CTR value (`ctr_value > split.border`). The two
share the same border space (verified: online-bin == inference-`Calc` when
`PriorDenom == 1`) but differ on the fractional part — e.g. a continuous `8.5`
passes border `8` at apply while a truncated bin `8` does not exceed `8` at train.
This is intentional (train/apply partitions legitimately differ — the summaries'
`[6,0,9,15]` structure vs `[10,0,0,20]` apply) and the e2e gate validates apply
against upstream, so it is NOT a defect — but the asymmetry is subtle and
undocumented at the comparison sites.
**Fix:** Add a one-line comment at `passes_ctr_split` noting that apply compares
the continuous CTR value whereas the train-side search compares the truncated
bin, so the partitions need not match.

### IN-05: Import ordering breaks the alphabetical grouping in two `pub use` blocks

**File:** `crates/cb-train/src/lib.rs:38`, `crates/cb-train/src/ctr/mod.rs:141`
**Issue:** `bake_ctr_table, BakedCtrData, BakedCtrTable,` is inserted at the TOP
of the `ctr::{...}` re-export block, before `accumulate_online` (lib.rs:38), and
`pub use bake::{...}` is appended after `calc_ctr::{...}` out of module order
(mod.rs:141). Cosmetic only.
**Fix:** Sort the re-exported names / module re-exports to match the surrounding
alphabetical convention.

---

_Reviewed: 2026-06-14_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
