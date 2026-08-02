---
title: BinarizedTargetMeanValue Sum is halved — bake passes the wrong target_border_count
status: draft
format: markdown
spec_version: 1
updated_at: 2026-08-02T00:00:00Z
source_requirements:
  - BUG-BTMV (defect found while executing E13 of ctr-type-engine-and-facade-routing)
  - SPEC-CTRT-07 (the BTMV parity gate that exposed it)
---

# BTMV target-border divisor (BUG-BTMV)

## 1. Context

E13's end-to-end ≤1e-5 gate for `simple_ctr = BinarizedTargetMeanValue` fails at
`max |diff| = 1.283e-1`, with E07/E09/E10/E11 all landed.

Localized decisively. Matching baked buckets to upstream's committed
`ctr_data` by hash (`crates/cb-oracle/fixtures/ctr_btmv_simple/model.json`,
a stride-3 `(hash, Sum, Count)` table):

| hash | upstream `(Sum, Count)` | ours `(Sum, Count)` |
|---|---|---|
| `14096670708071601218` | `(5, 7)` | `(2.5, 7)` |
| `10650234391120027977` | `(3, 7)` | `(1.5, 7)` |
| `3644720124901778394` | `(4, 6)` | `(2, 6)` |
| `15097791572046390990` | `(2, 5)` | `(1, 5)` |
| `6692239851685836511` | `(4, 5)` | `(2, 5)` |

**Every `Count` matches exactly. Every `Sum` is exactly half upstream's.**
`[VERIFIED: LOCAL, executed]`

### Root cause

`accumulate_online(column, target_class, target, classes, target_border_count)`
adds `target_class / target_border_count` to the BTMV mean history. Upstream
passes `targetClassesCount - 1` (`online_ctr.cpp:762`), which is `1` for binclf,
so `Sum` is simply the count of class-1 documents.

`crates/cb-train/src/ctr/bake.rs:196` passes **`classes`** for BOTH the
`classes` and the `target_border_count` argument:

```rust
let acc = accumulate_online(&key_refs, &target_class_n, &target_zero, classes, classes)?;
```

so the divisor is `2`, not `1`, and every BTMV `Sum` comes out halved.
`[VERIFIED: LOCAL crates/cb-train/src/ctr/bake.rs:196;
crates/cb-train/src/ctr/online.rs:163-169]`

### Why it was invisible until now

The defect is **latent, not new**. Before E11 the bake ALWAYS built a
`Borders` table (`build_final_ctr(&acc, ECtrType::Borders)` was hard-coded), and
the Borders arm reads `class_histories`, never `binarized_mean`. The wrong
divisor therefore affected a field nothing consumed. E11 made the mean path
live, and E13 is the first gate that reads it. `[INFERRED from the E11 diff]`

Note `online_mean_prefix` (E07, the PREFIX producer used during candidate
materialization) computes its divisor correctly as
`classes.saturating_sub(1).max(1)`. The defect is confined to the WHOLE-SET bake
path. This asymmetry must be verified, not assumed — see SPEC-BTMV-03.

## 2. Scope and non-goals

**In scope**
- Make the whole-set bake pass the correct `target_border_count` for the CTR
  type being baked.
- Gate it so a future regression is caught by a unit test, not only by an
  end-to-end fixture.

**Non-goals**
- Changing `accumulate_online`'s signature or semantics. It is correct; it is
  being called wrongly.
- Changing `online_mean_prefix` (E07) — verify it, do not touch it.
- Changing the Borders/Buckets/Counter bake arms. Their payloads do not read
  `binarized_mean`, so the divisor never reached them.
- Regenerating any fixture.

## 3. Dependencies

| Symbol | Location | Role |
|---|---|---|
| `accumulate_online` | `crates/cb-train/src/ctr/online.rs:163` | consumer of the divisor |
| `bake_ctr_table` | `crates/cb-train/src/ctr/bake.rs:196` | the wrong call |
| `ECtrType::target_border_count` | `crates/cb-train/src/ctr/mod.rs:123` | E01's helper — already returns the correct value |
| `online_mean_prefix` | `crates/cb-train/src/ctr/online.rs` | the prefix producer; believed correct |
| `build_final_ctr` mean arms | `crates/cb-train/src/ctr/final_ctr.rs` | reads `binarized_mean` |

**Note:** `ECtrType::target_border_count(2)` already returns `1` for
`BinarizedTargetMeanValue` and `Counter`, `2` for `Buckets`, `1` for `Borders`
(E01, SPEC-CTRT-01). The fix should route through it rather than hard-coding
`classes - 1`, so the rule lives in exactly one place.

## 4. Failure-isolated behavioral specifications

### SPEC-BTMV-01 — the bake divides by the CTR type's target border count

- **Given** a BTMV bake over a binclf corpus,
  **when** the whole-set accumulation runs,
  **then** each bucket's `Sum` equals the count of class-1 documents in that
  bucket (divisor `1`), not half of it.
- **Acceptance:** a unit test on `bake_ctr_table` asserting per-bucket
  `(Sum, Count)` against hand-computed values.

### SPEC-BTMV-02 — the baked BTMV table matches upstream bucket-for-bucket

- **Given** the frozen `ctr_btmv_simple` fixture,
  **when** the bake runs,
  **then** every `(hash, Sum, Count)` triple equals upstream's committed
  stride-3 `ctr_data` entry for that hash.
- **Acceptance:** a test comparing against `model.json` by hash, so a divisor
  error is caught at the TABLE level rather than only in predictions.
- **Anti-vacuity:** assert at least one bucket has `Sum != Count` and
  `Sum != 0`, else a degenerate corpus would satisfy it trivially.

### SPEC-BTMV-03 — the prefix producer is unaffected

- `online_mean_prefix` computes `classes.saturating_sub(1).max(1)` and is
  believed correct. **Verify with a test; do not assume.** If it is also wrong,
  this spec's scope widens and the plan must say so.

### SPEC-BTMV-04 — the non-mean bake arms are unchanged

- Borders/Buckets/Counter baked payloads must be byte-identical before and
  after. Their arms never read `binarized_mean`, so the divisor could not have
  reached them — but E11's frozen Borders bake characterization
  (`final_ctr_test::borders_bake_bytes_are_unchanged`) must still pass.

## 5. Acceptance scenarios

- **A-BTMV-1:** E13's `ctr_btmv_simple` gate passes at ≤1e-5 (currently
  `1.283e-1`).
- **A-BTMV-2:** the baked BTMV table matches upstream bucket-for-bucket.
- **A-BTMV-3:** all 11 CTR oracles + 3 one-hot targets stay green.
- **A-BTMV-4:** E12's Counter gate and the BUG-CTRB gates stay green.

## 6. Impact scope

`local` to `cb-train`'s bake, with a `cross-module` output change: BTMV models
gain correct `mean` tables. No non-mean model changes, because no non-mean arm
reads `binarized_mean`.

## 7. Risks and open questions

- **R1:** the fix may be `classes - 1` rather than
  `ctr_type.target_border_count(classes)`. These agree for Borders/BTMV/Counter
  at binclf but DIFFER for `Buckets` (helper returns `classes`). The plan must
  determine which upstream applies at the bake and justify it — a wrong choice
  silently changes the Buckets bake, which currently has no mean payload but
  does share the call.
- **OQ1:** whether `accumulate_online`'s `classes` argument is also wrong at
  this call site, or only `target_border_count`.
- **OQ2:** whether any other `accumulate_online` caller passes the same wrong
  divisor.

## 8. Traceability

- Found executing E13 of `.planning/plans/ctr-type-engine-and-facade-routing/`.
- Sibling defect: BUG-CTRB (`.planning/plans/ctr-split-border-space/`), also a
  latent defect exposed by a newly-added per-type gate.
- `[UNVERIFIED]` no Research Agent pass; planned from session evidence.
