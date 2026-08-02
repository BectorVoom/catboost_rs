---
title: BinarizedTargetMeanValue Sum is halved — bake passes the wrong target_border_count
status: complete-with-recorded-residual
completion_record: .planning/plans/btmv-target-border-divisor/COMPLETION.md
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

> **⚠ CORRECTION to the reported symptom (B07, from PLAN §1.1).** The
> `1.283e-1` figure is real, but the assertion that actually fires is **not** a
> `compare_stage` divergence panic — it is the **anti-vacuity guard** eight lines
> earlier:
>
> ```text
> panicked at crates/cb-train/tests/ctr_btmv_simple_oracle_test.rs:141:5:
> predictions are constant — the gate would be vacuous
> ```
>
> `max |diff| = 1.2830170735076996e-1` is visible only under `-- --nocapture`,
> printed by the still-passing reporting test. Our model's predictions were
> **constant across all 30 documents**: with the halved Sums, no bucket's
> apply-space CTR value reached any border the structure search (which uses the
> **correct** prefix divisor) had chosen, so the tree routed everything to one
> leaf. An executor who "cannot reproduce the reported failure" because the
> message differs should not go hunting — that *is* the reproduction.
> `[VERIFIED: LOCAL executed 2026-08-02 at c21f44a]`

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
(E01, SPEC-CTRT-01). ~~The fix should route through it rather than hard-coding
`classes - 1`, so the rule lives in exactly one place.~~

> ## ⛔ RETRACTED (BUG-BTMV close-out, B07 — 2026-08-02)
>
> The struck-through recommendation above is **WRONG** and was **not
> implemented**. It is preserved rather than deleted so a future reader can see
> that the question was asked and answered.
>
> **Upstream has TWO distinct target-border-count rules, and the helper mirrors
> the OTHER one.**
>
> | path | rule | upstream site | our mirror |
> |---|---|---|---|
> | **whole-set bake** (`CalcFinalCtrsImpl`) | `targetClassesCount - 1`, **type-INDEPENDENT** | `online_ctr.cpp:914` | `ctr::final_ctr_target_border_count` |
> | online, mean prefix | `targetClassesCount - 1`, passed as a **literal** | `online_ctr.cpp:762` | `online_mean_prefix` (`online.rs:321`) |
> | online, allocation sizing + CLASS prefix types | `GetTargetBorderCount(ctrInfo, …)`, **type-DEPENDENT** | `online_ctr.cpp:738/741`, `:777`; `ctr_helper.h:34-42` | `ECtrType::target_border_count` |
>
> `online_ctr.cpp:914` reads `int targetBorderCount = targetClassesCount - 1;`
> and sits **outside** the per-type `if/else` chain that begins at `:917`;
> `GetTargetBorderCount` is **never called anywhere inside `CalcFinalCtrsImpl`
> (`:875-939`)**.
> `[VERIFIED: UPSTREAM catboost v1.2.10, online_ctr.cpp fetched and read in full]`
>
> The two rules **agree** for Borders / `BinarizedTargetMeanValue` / Counter at
> binary classification and **DIFFER for `Buckets`** (the helper returns
> `targetClassesCount`, the bake returns `targetClassesCount - 1`) — so routing
> through the helper is **undetectable at binclf and wrong at multiclass**.
>
> **This was confirmed empirically, not just by reading.** Mutation **M2** (B04)
> implemented exactly the retracted advice and re-ran the whole gate set: the
> baked-table oracle (B02), the end-to-end `ctr_btmv_simple` fixture, all 11 CTR
> oracles, the 3 one-hot targets, the Counter gate and the BUG-CTRB gates **all
> stayed GREEN**. Only the deliberately-constructed `classes = 3` discriminators
> caught it (`final_ctr_test::bake_target_border_divisor_is_classes_minus_one_not_the_ctr_type_helper`,
> `mod_test::final_ctr_target_border_count_is_classes_minus_one`,
> `mod_test::the_two_target_border_rules_differ_for_buckets`). Those three tests
> are the **only** runtime defence against this substitution and must not be
> weakened.
>
> **The implemented rule is `classes.saturating_sub(1).max(1)`**, named as the
> private `ctr::final_ctr_target_border_count`. `ECtrType::target_border_count`
> is **correct and untouched** (body byte-unchanged; its doc block now names the
> ONLINE path). See PLAN §0 and D1/D2.

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

> **CLOSED — NOT AFFECTED (B06, verified 2026-08-02). The scope-widening branch
> did not open.** `online_mean_prefix` is upstream-exact at **every** class
> count, not merely at binclf: upstream's online mean path passes
> `targetClassesCount - 1` as a **literal** at `online_ctr.cpp:762`
> (`CalcOnlineCTRMean`'s parameter is `int targetBorderCount`, `:442`, used only
> at `elem.Add(static_cast<float>(permutedTargetClass[docId]) / targetBorderCount)`,
> `:467`) — it does **not** go through `GetTargetBorderCount`, which appears on
> that path only for allocation sizing (`:738/741`) and the CLASS prefix types
> (`:777`). B06's three tests passed on **first write**; mutation **M-B06**
> (`divisor -> classes.max(1)`) produced all four predicted outcomes, including
> the pre-existing binclf prefix test failing. `online.rs` is byte-unchanged by
> this plan.
>
> B06 Test 2 states the cross-path invariant that BUG-BTMV violated: the ONLINE
> prefix (`:762`) and the WHOLE-SET bake (`:914`) use the **same** divisor. The
> defect was exactly those two paths disagreeing — prefix right, bake wrong — so
> the trained structure and the baked table described different CTR values. The
> two expressions are nonetheless **deliberately not shared**: they coincide
> because two independent upstream code paths happen to use the same rule, and
> merging them would hide a future divergence.

### SPEC-BTMV-04 — the non-mean bake arms are unchanged

- Borders/Buckets/Counter baked payloads must be byte-identical before and
  after. Their arms never read `binarized_mean`, so the divisor could not have
  reached them — but E11's frozen Borders bake characterization
  (`final_ctr_test::borders_bake_bytes_are_unchanged`) must still pass.

## 5. Acceptance scenarios

- **A-BTMV-1:** E13's `ctr_btmv_simple` gate passes at ≤1e-5 (currently
  `1.283e-1`). — ❌ **NOT MET; recorded residual.** The fix worked and the failure
  MODE changed (the `predictions are constant` guard no longer fires, so the model
  no longer collapses to one leaf), but a **second, independent** divergence sits
  downstream of the baked table: `StageDiverged { stage: Predictions, index: 0,
  expected: 0.08766370996535935, actual: 0.049457048547128145 }`,
  `max |diff| = 1.371e-1`. It is **not** this defect — A-BTMV-2 proves the table
  itself is now byte-correct against upstream. Specced separately; not chased.
  Nothing was weakened, no fixture touched, nothing regenerated.
- **A-BTMV-2:** the baked BTMV table matches upstream bucket-for-bucket. — ✅ **MET**
  (`ctr_btmv_bake_upstream_table_test`, exact `==` on every `(hash, Sum, Count)`).
- **A-BTMV-3:** all 11 CTR oracles + 3 one-hot targets stay green. — ✅ **MET**,
  with **zero diff** on all eleven oracle files.
- **A-BTMV-4:** E12's Counter gate and the BUG-CTRB gates stay green. — ✅ **MET**.

## 6. Impact scope

`local` to `cb-train`'s bake, with a `cross-module` output change: BTMV models
gain correct `mean` tables. No non-mean model changes, because no non-mean arm
reads `binarized_mean`.

## 7. Risks and open questions

All three are **RESOLVED** (BUG-BTMV close-out, B07 — 2026-08-02).

- **R1 — RESOLVED: `classes - 1`, floored at 1. Not ambiguous.** The fix is
  `classes.saturating_sub(1).max(1)`, **not**
  `ctr_type.target_border_count(classes)`. `CalcFinalCtrsImpl` computes the
  divisor **once, outside the per-type switch** (`online_ctr.cpp:914`), so it is
  type-independent, and `GetTargetBorderCount` is never called inside it
  (`:875-939`). See the retraction block in §3.
  `[VERIFIED: UPSTREAM catboost v1.2.10 online_ctr.cpp, fetched and read in full]`

  R1's fallback criterion — *"if ambiguous, prefer the option that provably
  preserves today's non-mean payloads"* — is **also satisfied**, so the choice is
  overdetermined. The divisor reaches exactly one accumulator field,
  `binarized_mean` (`online.rs:212-214`), which exactly one arm of
  `build_final_ctr` reads (`BinarizedTargetMeanValue`); `bake_ctr_table`
  populates `BakedCtrTable.mean` only in the mean arm (`bake.rs:250-260`).
  Gated by `final_ctr_test::the_divisor_is_unreachable_from_every_non_mean_payload`
  and, at byte level, by `buckets_and_counter_bake_bytes_are_unchanged` +
  the pre-existing `borders_bake_bytes_are_unchanged`. Mutation **M-B05** proved
  the isolation's *direction*: restoring the bug leaves all three GREEN while the
  BTMV control fails.

- **OQ1 — ANSWERED: no.** `accumulate_online`'s **4th** argument (`classes`) is
  **correct and unchanged**; only the **5th** was wrong. Upstream uses
  `targetClassesCount` for exactly what our `classes` drives — the class-histogram
  width and `TargetClassesCount`:
  `result->TargetClassesCount = targetClassesCount;` and
  `AllocateBlobAndGetArrayRef<int>(leafCount * targetClassesCount)`
  (`online_ctr.cpp:909-912`), then
  `MakeArrayRef(ctrIntArray.data() + targetClassesCount * elemId, targetClassesCount);
  ++elem[targetClass[z]];` (`:930-934`). The frozen
  `borders_bake_bytes_are_unchanged` payload (`int_counts = [[0,4],[4,0],[1,3]]`)
  fails immediately if that width moves.

- **OQ2 — ANSWERED: no.** `accumulate_online` has exactly **one** production
  caller, and it was the defective one. Every other call site is a test passing a
  deliberate, explicit divisor. Full enumeration
  `[VERIFIED: LOCAL grep -rn "accumulate_online" --include=*.rs crates/]`:
  - **production:** `crates/cb-train/src/ctr/bake.rs:196` (the defect; now routed
    through `final_ctr_target_border_count`)
  - **tests:** `crates/cb-train/src/ctr/online_test.rs:20,33,46,60,74,80,85`;
    `crates/cb-train/src/ctr/final_ctr_test.rs:15`;
    `crates/cb-model/tests/ctr_data_roundtrip_test.rs:100,134,160`

### 7.1 Out-of-scope finding, recorded for a future plan

`ECtrType::target_border_count` has **ZERO production callers**. Its only call
sites workspace-wide are `crates/cb-train/src/ctr/mod_test.rs`. The reason is
structural, not accidental: the online CTR-data **allocation** path
(`online_ctr.cpp:738/741`) — the helper's actual upstream consumer — is not yet
implemented in this repository.

**⚠ Do NOT "fix" this by wiring the helper into the bake.** That is precisely the
retracted change in §3, and it is undetectable at binary classification. If the
helper's disuse is to be addressed, it belongs to the online-allocation path in a
separate plan. The warning is duplicated in the helper's own doc block so it
travels with the code.

## 8. Traceability

- Found executing E13 of `.planning/plans/ctr-type-engine-and-facade-routing/`.
- Sibling defect: BUG-CTRB (`.planning/plans/ctr-split-border-space/`), also a
  latent defect exposed by a newly-added per-type gate.
- `[UNVERIFIED]` no Research Agent pass; planned from session evidence.
