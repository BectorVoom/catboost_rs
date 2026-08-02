---
title: CTR split border is persisted in bin space but compared in value space
status: draft
format: markdown
spec_version: 1
updated_at: 2026-08-02T00:00:00Z
source_requirements:
  - BUG-CTRB (defect found while executing E12 of ctr-type-engine-and-facade-routing)
  - SPEC-CTRT-08 (the Counter parity gate that exposed it)
---

# CTR split border space (BUG-CTRB)

## 1. Context

A CTR split in a trained model carries a scalar `border`. The trainer chooses it
during the structure search and every consumer — the apply path, the `.cbm`
codec, and upstream CatBoost — reads it back. **The producer and the consumers
disagree about what space that number lives in.**

- The structure search enumerates candidates as INTEGER BIN INDICES:
  `let border = border_idx as f64;` for `border_idx in 0..ctr_border_count`
  `[VERIFIED: LOCAL crates/cb-train/src/tree.rs:3163]`.
- The TRAINING split test is `f64::from(bin) > *border` — bin space, and
  internally consistent `[VERIFIED: LOCAL crates/cb-train/src/tree.rs:2600]`.
- The chosen bin index is persisted VERBATIM into `CtrSplitSpec.border`
  `[VERIFIED: LOCAL crates/cb-train/src/tree.rs:3297]`.
- The APPLY split test is `ctr_value > split.border`, where `ctr_value` is the
  SCALED inference value produced by `ctr_value_for_combined_projection` /
  `calc_inference` (shift and scale applied) — VALUE space
  `[VERIFIED: LOCAL crates/cb-model/src/apply.rs:189]`.

Upstream CatBoost stores CTR borders in VALUE space as `(bin + 1) - 2^-20`.
Verified across every committed fixture carrying CTR borders:

| fixture | borders |
|---|---|
| `ctr_counter_simple` | `8.999999046325684`, `10.999999046325684` |
| `fstr_ctr` | `3.999999`, `6.999999`, `11.999999`, `3.999999` |
| `tensor_ctr_e2e` | `2.9999990463256836`, `7.999999046325684` |

Every value satisfies `border + 2^-20 == an exact integer`;
`9 - 8.999999046325684 = 9.5367431640625e-7 = 2^-20` exactly
`[VERIFIED: LOCAL, computed over all three model.json files]`.

### Why this has gone unnoticed

A divergence manifests **only** when some document's CTR bin lands exactly on a
chosen border. For such a document `bin == b`, so training evaluates `b > b` =
false while apply evaluates `v > b` = true for any `v` in `(b, b+1)`. No
document in any pre-existing fixture happens to land on a chosen border, so the
eleven CTR oracles are green by **data-dependent coincidence, not correctness**.

This is load-bearing for the plan: **the existing 11 CTR oracles are not a
regression gate for this defect.** A new gate that pins the train/apply
agreement directly is required.

Counter exposes it because whole-set bucket totals cluster many documents onto
a small set of exact bin values.

## 2. Scope and non-goals

**In scope**
- Make the persisted `CtrSplitSpec.border` a VALUE-space threshold matching
  upstream's convention.
- Add a regression gate that fails on the train/apply disagreement directly,
  independent of any fixture's data distribution.

**Non-goals**
- Changing the structure search's candidate enumeration or its bin-space
  scoring. The search is correct as-is.
- Changing `LevelKind::Ctr`'s border. It stays bin space as a UNITS contract
  (SPEC-CTRB-03); note that converting it would be arithmetically harmless, so
  this is hygiene, not a correctness requirement.
- Changing the apply path. `apply.rs:189` is correct and is what makes upstream
  `.cbm` models score correctly today.
- Any change to float or one-hot split borders.
- Re-generating any committed fixture.

## 3. Dependencies

| Symbol | Location | Role |
|---|---|---|
| `CtrSplitSpec.border` | `crates/cb-train/src/tree.rs:181` | the persisted field |
| `LevelKind::Ctr { border }` | `crates/cb-train/src/tree.rs:286` | training-only leaf assignment |
| `assign_leaf_of_averaging` | `crates/cb-train/src/boosting.rs:1938` | consumes `LevelKind` border against `col.bins` |
| `passes_ctr_split` | `crates/cb-model/src/apply.rs:157-189` | consumes the persisted border against the scaled value |
| `build_tctr_feature` | `crates/cb-model/src/cbm.rs:390` | writes `Borders` as **f32** |
| `ctr_split_to_global_index` | `crates/cb-model/src/cbm.rs:351` | resolves a split to `(ctr_feature, border_index)` |
| decode | `crates/cb-model/src/cbm.rs:601` | `f64::from(borders.get(border_index))` |

## 4. Typed contracts

```rust
// crates/cb-train/src/tree.rs
pub struct CtrSplitSpec {
    pub border: f64, // VALUE space: the split passes when ctr_value > border
    // ...
}

pub enum LevelKind {
    Ctr { ctr_idx: usize, border: f64 }, // BIN space: passes when bin > border
    // ...
}
```

The two `border` fields are adjacent in the code and identically typed but carry
**different units**. That is the defect's root cause. It is NOT, however, a
live hazard for the fix: see SPEC-CTRB-03 — converting the `LevelKind` one too
would be arithmetically a no-op, because its consumer's left operand is an
integer bin.

## 5. Failure-isolated behavioral specifications

### SPEC-CTRB-01 — the persisted border is a value-space threshold

- **Trigger:** the oblivious CTR structure search chooses a CTR split at bin
  threshold `b`.
- **Input:** `b: usize` (the winning `border_idx`, `0 <= b < ctr_border_count`).
- **Output:** `CtrSplitSpec.border: f64` equal to `f64::from((b as f32 + 1.0) - f32::powi(2.0, -20))`.
- **Given** a chosen CTR split at bin threshold `b`,
  **when** the tree is persisted,
  **then** `CtrSplitSpec.border` is the value-space threshold, not `b`.
- **Invariant 1 (strict interval):** `b < border < b + 1`. This is what makes
  `trunc(v) > b ⟺ v > border` hold.
- **Invariant 2 (f32 fixed point):** `f64::from(border as f32) == border`. The
  `.cbm` codec stores `Borders` as f32 and decodes via `f64::from`
  `[VERIFIED: LOCAL crates/cb-model/src/cbm.rs:387,601]`, so a border that is
  not an f32 fixed point would shift on a save→load round-trip.
- **BOUNDED DOMAIN (verified numerically, plan-check pass 1).** Both invariants
  hold together only for `b <= 15`. Measured:

  | `b` | f32 form | in `(b, b+1)`? | f32 fixed point? | f64 form | f32 fixed point? |
  |---|---|---|---|---|---|
  | 0–15 | `b.999999046325684` | yes | yes | same | yes |
  | 16 | `17.0` | **NO** | yes | `16.999999046325684` | **NO** |
  | 254 | `255.0` | **NO** | yes | `254.99999904632568` | **NO** |

  So **neither** formulation is generally correct above 15: the f32 form loses
  Invariant 1, the f64 form loses Invariant 2. The f32 form is mandated because
  Invariant 2 is the one the codec enforces.
- **Reachability:** `ctr_border_count` is not configurable —
  `ctr_border_count_default()` returns `15`
  `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:529-531]`, consumed at
  `boosting.rs:3238` — so `border_idx ∈ 0..15`, i.e. `b <= 14`. The domain is
  strictly inside the safe range with one border of margin.
- **STOP CONDITION:** if `ctr_border_count` ever becomes configurable or exceeds
  16, this contract breaks and BUG-CTRB reappears at the top of the range. Any
  test must therefore sweep ONLY the reachable domain and must additionally pin
  `b = 16` as a CHARACTERIZATION of the boundary, naming
  `crates/cb-train/src/boosting.rs:3238` — never assert the formula is general.
- **Acceptance:** a unit sweep over the reachable `b`, plus the `b = 16`
  boundary characterization.
- **Out of scope:** the candidate enumeration and the scoring.

### SPEC-CTRB-02 — training and apply agree for a document on the border

- **Trigger:** a document whose CTR bin EQUALS the chosen border.
- **Given** a materialized CTR column and a chosen split at bin `b`, and a
  document whose bin is exactly `b`,
  **when** the training-side test and the apply-side test are both evaluated for
  that document,
  **then** both return the SAME boolean (specifically `false` — bin `b` does not
  exceed threshold `b`).
- **Acceptance:** a test that constructs exactly this case and asserts agreement.
  This is the PRIMARY Red. It must fail today for the stated reason.
- **Note:** this specification is the one the existing oracles do not cover.

### SPEC-CTRB-03 — `LevelKind::Ctr.border` stays bin space (a UNITS contract, not a correctness trap)

**CORRECTION (plan-check pass 1).** An earlier revision of this spec claimed that
converting `LevelKind::Ctr.border` alongside `CtrSplitSpec.border` "would
silently corrupt leaf assignment". **That claim is FALSE and is retracted.**

Its only consumer is `assign_leaf_of_averaging`
(`crates/cb-train/src/boosting.rs:1926-1938`), which evaluates
`f64::from(bin) > *border` against `pub bins: Vec<u32>`
(`crates/cb-train/src/ctr/ctr_feature.rs:86`) — an **integer** left operand.
For integer `bin` and integer `b`:

```
bin > b   ⟺   bin >= b+1   ⟺   bin > (b+1) - 2^-20
```

so the conversion is **arithmetically a no-op** over the reachable domain.
`ctr_border_count` is not configurable — `ctr_border_count_default()` is `15`
`[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:529-531 (definition),
consumed at :3238]` — so `b ∈ 0..=14`.

- **Requirement (units hygiene, not behavior):** `LevelKind::Ctr.border` remains
  bin space, because that is the unit its consumer's operand is in and mixing
  units in adjacent identically-typed fields is how this defect arose.
- **Acceptance:** ONE genuinely falsifiable assertion —
  `assert_eq!(bin_border, bin_border.trunc())` on a `LevelKind::Ctr` border.
  **The 11 CTR oracles CANNOT serve as acceptance here**: since the conversion
  is a no-op for integer operands, they stay green either way. Any plan step
  that expects an oracle to regress when `:3289` is converted is unsatisfiable
  and must not be written.
- **Consequence for the plan:** no **oracle-based** expectation can be built
  around `:3289` — converting it leaves the E12 gate, the 11 CTR oracles and the
  3 one-hot targets green. The integrality assertion is the only available
  detector. A mutation check MAY still be built around `:3289`, provided its
  expected outcome is exactly "the two integrality assertions fail AND every
  oracle stays green"; that is what proves the assertions are load-bearing.

### SPEC-CTRB-04 — the `!has_ctr` fallback is unaffected

- `ctr_splits_for_tree` (`crates/cb-train/src/boosting.rs:2007`) sets
  `border: 0.0`. It is reached only on the `!has_ctr` branch, where
  `has_ctr = !materialized_ctr_features.is_empty()`
  `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:4657]` and the candidate list
  is empty, so the function returns an EMPTY vector and the `border: 0.0`
  literal is never constructed in production. The code says so directly:
  "`ctr_splits_for_tree` is retained for the no-CTR candidate path (it returns
  empty there)" `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:~5417]`.
- **Decision:** do NOT convert it. Record the reasoning so a later reader does
  not "fix" it. `[INFERRED from the two verified facts above]`
- **Acceptance:** an assertion that the fallback returns empty for an empty
  candidate list (already covered by E03's characterization tests, which pass a
  NON-empty list deliberately and must keep passing unchanged).

### SPEC-CTRB-05 — `.cbm` round-trip is lossless for the new border

- **Given** a trained CTR model whose borders are value-space,
  **when** it is saved and re-loaded,
  **then** the decoded borders equal the in-memory borders exactly, and
  predictions are unchanged.
- **Rationale:** `.cbm` narrows borders to f32 on save. SPEC-CTRB-01's f32 fixed
  point is what makes this hold.
- **Acceptance:** a save→load→predict round-trip test asserting bit-equal
  predictions.

## 6. Acceptance scenarios

- **A-CTRB-1:** the E12 `ctr_counter_simple` gate passes at ≤1e-5 (currently
  fails at `max|diff| = 2.6874900161694987e-1`).
- **A-CTRB-2:** all 11 CTR oracles remain green.
- **A-CTRB-3:** all 3 one-hot wave targets remain green.
- **A-CTRB-4:** a document whose bin equals the border is classified identically
  by training and apply.
- **A-CTRB-5:** `.cbm` save→load→predict is bit-stable for a CTR model.

## 7. Impact scope

`cross-module`. Producer in `cb-train`, consumers in `cb-model` (apply + codec).

- **Output change:** every CTR model the trainer produces gets different
  `border` values. This changes `.cbm`/json bytes for categorical models.
- **NOT affected:** `crates/cb-oracle/fixtures/ctr_nonmean_byte_identity/`
  (E00's frozen baseline) is built from a HAND-CONSTRUCTED model whose borders
  are literals (`0.25`, `0.5`), never produced by the trainer
  `[VERIFIED: LOCAL crates/cb-model/tests/ctr_nonmean_byte_identity_test.rs]`.
  **The plan must confirm this by running that gate.**
- **NOT affected:** `float_only_byte_identity` — float-only models carry no CTR
  splits.
- Upstream `.cbm` files LOADED by us already carry value-space borders and are
  decoded correctly today; this change makes our SAVED models agree with them.

## 8. Compatibility and migration

Any `.cbm` previously written by this trainer for a categorical model carries
bin-space borders and would score incorrectly under any correct reader
(including upstream CatBoost). There is no in-repo committed artifact in that
state — every committed `.cbm` CTR fixture came from upstream. No migration
path is required; the change makes newly written models correct.

## 9. Risks and open questions

- **R1 (RETRACTED at plan-check pass 1):** an earlier revision named "converting
  `LevelKind::Ctr`'s border too" as the principal risk. It is **not** a risk —
  the conversion is arithmetically a no-op for the integer operand its consumer
  compares against (SPEC-CTRB-03). The residual concern is units hygiene only.
- **R1' (the actual principal risk):** the fix is a ONE-LINE change, so its test
  is at high risk of being tautological. The primary Red must be constructed so
  it cannot be satisfied by editing the apply side instead of the trainer.
- **R2:** the exact epsilon. `2^-20` is derived from the committed fixtures, and
  the f32 computation reproduces the observed bit patterns. `[VERIFIED: LOCAL]`
  for the three fixtures; `[INFERRED]` that upstream uses this for all CTR
  border counts.
- **OQ1 — CLOSED at plan-check pass 1.** `CtrSplitSpec` is constructed at
  **FOUR** sites, not three: `tree.rs:3291`, `boosting.rs:2007`,
  `boosting_test.rs:337` and `crates/cb-model/tests/mixed_kind_split_order_test.rs:47`
  (test-only, literal `0.9`, unaffected) `[VERIFIED: CODEGRAPH, plan-check]`.
  No non-oblivious / region grower persists CTR borders: the other five
  `GrownTree` producers all set `ctr_splits: Vec::new()` —
  `greedy_tensor_search_oblivious_perturbed` (`tree.rs:774`), `leaf_wise_grower`
  (`:2089`), `region_grower` (`:2238`), `..._ordered` (`:2541`), `..._pairwise`
  (`:3708`) `[VERIFIED: CODEGRAPH, plan-check]`. Only
  `greedy_tensor_search_oblivious_with_ctr` persists CTR borders.

## 10. Traceability and sources

- Defect discovered executing E12 of
  `.planning/plans/ctr-type-engine-and-facade-routing/`.
- Fix verified experimentally this session (applied, full regression run,
  reverted): E12 5/5 including the ≤1e-5 gate, 11 CTR oracles green, 3 one-hot
  targets green, `cb-train`+`cb-model` green except the pre-existing
  `monotone_non_symmetric_and_region_are_typed_errors` failure recorded in
  `PLAN.md` §3.
- `[UNVERIFIED]` no independent Research Agent pass was run — the user elected to
  plan from session evidence.
