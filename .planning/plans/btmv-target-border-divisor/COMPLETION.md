---
title: Completion record — BUG-BTMV (BTMV target-border divisor)
plan_id: BUG-BTMV
status: complete
residual_resolution: BUG-SFS — fixed and gated same-day; see §6 and SPEC.md §9
format: markdown
updated_at: 2026-08-02T00:00:00Z
source_plan: .planning/plans/btmv-target-border-divisor/PLAN.md
source_spec: .planning/plans/btmv-target-border-divisor/SPEC.md
---

# BUG-BTMV — completion record

All seven tasks executed. **The defect is fixed and gated.** One acceptance
scenario (A-BTMV-1, the end-to-end `ctr_btmv_simple` ≤1e-5 gate) is **NOT met**
because a second, independent defect sits downstream of the baked table; it is
recorded in §6 and specced separately rather than chased. Every other gate is
green.

| Task | Wave | Landed | Status |
|---|---|---|---|
| B01 — primary Red + 3-way discriminator | W0 | `2c360e2` | done |
| B06 — verify the online prefix producer | W0 | `2c360e2` | done |
| B02 — upstream-anchored table Red | W0 | `2c360e2` | done |
| B03 — the one-expression fix | W1 | `2c360e2` | done |
| **B04** — name the divisor, disarm the helper trap | W2 | this commit | done |
| **B05** — non-mean isolation guard | W3 | this commit | done |
| **B07** — close-out | W4 | this commit | done |

---

## 1. The defect, in one line

`bake_ctr_table` passed `classes` for **both** the 4th (`classes`) and the 5th
(`target_border_count`) argument of `accumulate_online`, so every baked
`BinarizedTargetMeanValue` `Sum` came out **exactly half** upstream's. Upstream's
`CalcFinalCtrsImpl` sets `int targetBorderCount = targetClassesCount - 1;`
(`online_ctr.cpp:914`) **outside** the per-type dispatch.

## 2. What B04 changed (this commit)

**Behavior-preserving.** B03 already landed the fix as an inline expression; B04
gives it a name, a doc block, and a direct unit test.

- `crates/cb-train/src/ctr/mod.rs` — new **private** free function
  `final_ctr_target_border_count(target_classes_count) -> usize`, body
  `target_classes_count.saturating_sub(1).max(1)`, carrying the two-rules table,
  the `.max(1)` rationale, and the "Deliberately NOT shared with
  `online_mean_prefix`" block. Declared as a plain `fn`, not `const fn`:
  `Ord::max` is not const-stable, and PLAN B04's inline note directs dropping
  `const` rather than rewriting the expression.
- `crates/cb-train/src/ctr/mod.rs` — `ECtrType::target_border_count`'s **body is
  byte-unchanged**; its doc block gained a "Which path uses this" section naming
  the ONLINE path and the zero-production-caller fact.
- `crates/cb-train/src/ctr/bake.rs` — the call site now reads
  `super::final_ctr_target_border_count(classes)`; the 4th argument is still
  `classes`. The long inline rationale moved to the helper's doc, leaving a
  one-line `*** NOT ctr_type.target_border_count(classes) ***` pointer.
- `crates/cb-train/src/ctr/mod_test.rs` — 3 new tests (additive).

**Visibility unchanged:** no `pub`, no `pub(crate)`, no new `lib.rs` re-export.
`bake` and `mod_test` are children of `ctr` and can already see a `ctr`-private
item.

**The duplication is deliberate.** `grep -rn "saturating_sub(1).max(1)"
crates/cb-train/src/ctr/` returns **exactly two code sites** —
`mod.rs:248` (`final_ctr_target_border_count`) and `online.rs:321`
(`online_mean_prefix`). The two other grep hits are string literals inside test
messages that cite the idiom by name, not expressions. They are **not**
deduplicated: the expressions coincide because two independent upstream code
paths (`:762` and `:914`) happen to use the same rule, and merging them would
hide a future divergence.

## 3. What B05 added (this commit)

`crates/cb-train/src/ctr/final_ctr_test.rs`, additive only:

- `buckets_and_counter_bake_bytes_are_unchanged` — extends E11's frozen
  **Borders** pin to **Buckets** and **Counter**, including the same
  type-agnostic `hashes` / `shift.to_bits()` / `scale.to_bits()` triple.
- `the_divisor_is_unreachable_from_every_non_mean_payload` — the explicit
  isolation statement over all four CPU-legal types.

## 4. Mutation checks — every enumerated outcome observed

### M-B06 (B06, landed at `2c360e2`) — `online.rs:321` → `classes.max(1)`
All four predicted outcomes observed, including outcome 3 (Test 3 staying green)
and outcome 4 (the pre-existing binclf prefix test failing). Reverted manually;
`online.rs` byte-clean.

### M1 (B04) — helper body → `target_classes_count.max(1)` (restore the bug)

| # | Predicted | Observed |
|---|---|---|
| 1 | B04 Test 1 FAILS | ✅ `left: 2, right: 1` — "2 classes must give 1" |
| 2 | B01 Test 1 FAILS | ✅ `btmv_bake_sums_class_one_documents_per_bucket` FAILED |
| 3 | B01 Test 2 FAILS | ✅ `bake_target_border_divisor_…_not_the_ctr_type_helper` FAILED |
| 4 | B02 Test 1 FAILS, ratio exactly 0.5 | ✅ `bucket 3644720124901778394: Sum 2 != upstream 4. Ratio 0.500000` |
| 5 | E13 FAILS, back to `predictions are constant` | ✅ verbatim `predictions are constant — the gate would be vacuous` — the failure mode reverted **exactly** to PLAN §1.1's, confirming the divisor is what collapsed the model to one leaf |
| 6 | 11 CTR oracles, 3 one-hot targets, Counter gate, BUG-CTRB gates, `borders_bake_bytes_are_unchanged`, `ctr_nonmean_byte_identity_test` stay **GREEN** | ✅ **all green** — the required result |

**Additional observed outcome, not enumerated in the plan:** B04 Test 3
(`the_two_target_border_rules_differ_for_buckets`) also failed under M1
(`left: 3, right: 3` — under `classes.max(1)` the bake rule collides with the
Buckets helper rule at `classes = 3`). This is a natural consequence of the
mutation and strengthens rather than contradicts the check.

Reverted manually. Re-ran: 6/6, 14/14, 2/2 green.

### M2 (B04) — helper body → `ECtrType::BinarizedTargetMeanValue.target_border_count(...)` (implement the retracted advice)

| # | Predicted | Observed |
|---|---|---|
| 1 | B04 Test 1 FAILS at `3 → 1` | ✅ `left: 1, right: 2` — "3 classes must give 2" |
| 2 | B04 Test 3 FAILS at `final_ctr_target_border_count(3) == 2` | ✅ `left: 1, right: 2` |
| 3 | B01 Test 2 FAILS with `[(3.0,3),(4.0,3)]` | ✅ observed `left: [(3.0, 3), (4.0, 3)]`, `right: [(1.5, 3), (2.0, 3)]` — **the exact values the plan predicted** |
| 4 | B01 Test 1 stays **GREEN** | ✅ green — at `classes = 2` the helper also returns 1 |
| 5 | B02, E13, 11 oracles, 3 one-hot, BUG-CTRB gates stay **GREEN** | ✅ B02 2/2 green; E13 **byte-identical to its baseline failure** (same `StageDiverged`, same `max \|diff\| = 1.371207585124875e-1`, 3 passed / 1 failed); all oracles green |

> **Outcomes 4 and 5 are the empirical confirmation of PLAN §0's central claim:
> no binary-classification runtime gate can detect a helper-based "fix".** The
> only three tests that caught M2 are the deliberately-constructed `classes = 3`
> discriminators. No test was strengthened and no production code was altered to
> force a regression under M2.

Reverted manually. Re-ran: 6/6, 14/14, 2/2 green.

### M-B05 (B05) — helper body → `target_classes_count.max(1)`, **inverted expectation**

| # | Predicted | Observed |
|---|---|---|
| 1 | B05 Test 1 stays **GREEN** | ✅ `buckets_and_counter_bake_bytes_are_unchanged ... ok` |
| 2 | B05 Test 2 stays **GREEN** | ✅ `the_divisor_is_unreachable_from_every_non_mean_payload ... ok` |
| 3 | `borders_bake_bytes_are_unchanged` stays **GREEN** | ✅ ok |
| 4 | B01 Test 1 **FAILS** (the control) | ✅ `btmv_bake_sums_class_one_documents_per_bucket ... FAILED` — the mutation was live, the build was not stale |

Outcomes 1–3 staying green **is** the empirical proof of SPEC-BTMV-04's
isolation claim. Falsifiability is carried by outcome 4. Reverted manually;
`git diff crates/cb-train/src/ctr/mod.rs` shows only B04's hunks.

## 5. Final gate transcript

```text
cargo test -p cb-train --lib ctr::final_ctr_test        ok. 16 passed; 0 failed
cargo test -p cb-train --lib ctr::mod_test              ok.  6 passed; 0 failed
cargo test -p cb-train --lib ctr::online_test           ok. 28 passed; 0 failed
cargo test -p cb-train --test ctr_btmv_bake_upstream_table_test
                                                       ok.  2 passed; 0 failed
cargo test -p cb-train --test ctr_counter_simple_oracle_test
                                                       ok.  4 passed; 0 failed

── the 11 CTR oracles — ALL GREEN, ZERO DIFF on all eleven files ──────────────
ctr_feature_materialize_test        ok.  5 passed
ctr_split_scoring_test              ok. 11 passed
multi_permutation_e2e_oracle_test   ok.  1 passed
multi_permutation_fold_oracle_test  ok.  6 passed
ordered_ctr_oracle_test             ok.  3 passed
plain_ctr_oracle_test               ok.  3 passed
s_order_ctr_bins_oracle_test        ok.  2 passed
tensor_ctr_e2e_oracle_test          ok.  3 passed
tensor_ctr_oracle_test              ok.  3 passed
ctr_data_roundtrip_test  (cb-model) ok.  8 passed
fstr_ctr_oracle_test     (cb-model) ok.  3 passed

── one-hot wave, BUG-CTRB gates, byte-identity, model serde ───────────────────
one_hot_oracle_test                 ok.  6 passed; 1 ignored
one_hot_draw_accounting_test        ok.  6 passed
device_one_hot_parity_test          ok.  1 passed
ctr_border_space_test               ok.  3 passed
ctr_border_upstream_anchor_test     ok.  2 passed
ctr_border_cbm_roundtrip_test       ok.  2 passed
ctr_nonmean_byte_identity_test      ok.  2 passed; 1 ignored
cbm_oracle_test                     ok. 15 passed
json_oracle_test                    ok.  9 passed
float_only_byte_identity_test       ok.  3 passed; 1 ignored

── crate sweeps ──────────────────────────────────────────────────────────────
cargo test -p cb-train --no-fail-fast
    FAILED: btmv_simple_predictions_match_upstream_within_1e_minus_5   (§6)
    FAILED: monotone_non_symmetric_and_region_are_typed_errors         (pre-existing baseline)
    no other failure
cargo test -p cb-model --no-fail-fast
    0 failed targets

── lints ─────────────────────────────────────────────────────────────────────
cargo clippy --workspace --all-targets
    ZERO findings in any file this plan touched.
    Pre-existing, unrelated, NOT introduced here:
      crates/cb-oracle/src/bin/write_skeleton.rs:25 — approximate PI / E
        (from commit 404aa40, untouched by this branch)
      crates/cb-train/tests/ctr_btmv_simple_oracle_test.rs:25-26 — unused imports
        (PLAN §1.2: accepted, a protected untracked path, deliberately not edited)
```

### Diff gate — verified file-by-file

Changed by this plan (B04 + B05 + B07):
```
crates/cb-train/src/ctr/bake.rs             (B03 in 2c360e2, B04 here)
crates/cb-train/src/ctr/mod.rs              (B04)
crates/cb-train/src/ctr/mod_test.rs         (B04, additive)
crates/cb-train/src/ctr/final_ctr_test.rs   (B01 in 2c360e2, B05 here — additive, 0 removed lines)
crates/cb-train/src/ctr/online_test.rs      (B06 in 2c360e2)
crates/cb-train/tests/ctr_btmv_bake_upstream_table_test.rs  (B02 in 2c360e2, new)
.planning/plans/btmv-target-border-divisor/{SPEC,COMPLETION}.md
```

**Zero diff confirmed on:** all 11 CTR oracles; the 3 one-hot targets;
`ctr_counter_simple_oracle_test`; the three BUG-CTRB gates;
`ctr_nonmean_byte_identity_test`; `crates/cb-train/src/ctr/online.rs`;
`crates/cb-train/src/ctr/final_ctr.rs`; `crates/cb-train/src/ctr/ctr_feature.rs`;
`crates/cb-train/src/boosting.rs`; everything in `crates/cb-model/src/`.

**Protected untracked path intact:**
`crates/cb-train/tests/ctr_btmv_simple_oracle_test.rs` is still untracked and
unmodified. (`crates/cb-oracle/fixtures/ctr_btmv_simple/` was committed by B03.)
No fixture was regenerated; neither `gen_fixtures.py` was invoked. No
`git checkout --`, `git stash` or `git clean` was run at any point.

## 6. Acceptance scenarios

| Scenario | Result |
|---|---|
| **A-BTMV-1** — E13's `ctr_btmv_simple` gate at ≤1e-5 | ✅ **MET after BUG-SFS** (❌ at plan close-out; the residual below was then localized and fixed — see the resolution block at the end of this section) |
| **A-BTMV-2** — baked BTMV table matches upstream bucket-for-bucket | ✅ **MET** — B02 green, exact `==` on every `(hash, Sum, Count)` |
| **A-BTMV-3** — 11 CTR oracles + 3 one-hot targets green | ✅ **MET**, with zero diff on all eleven |
| **A-BTMV-4** — Counter gate + BUG-CTRB gates green | ✅ **MET** |

### The recorded residual (PLAN §7 OQ-A, resolved NEGATIVELY)

OQ-A — *that fixing the divisor alone turns E13 green* — was flagged
`[INFERRED]`, not measured, with an explicit STOP-AND-REPORT branch. It resolves
**negatively**, and B03's STOP branch was executed rather than chased:

- **The failure MODE changed**, which is itself evidence the fix worked: the
  anti-vacuity guard (`predictions are constant`) no longer fires, so the model
  no longer collapses to a single leaf.
- The failure is now a genuine divergence:
  `StageDiverged { stage: Predictions, index: 0, expected: 0.08766370996535935,
  actual: 0.049457048547128145 }`, `max |diff| = 1.371207585124875e-1`.
- **The residual is NOT this defect.** B02 proves the baked table is now
  byte-correct against catboost 1.2.10's own committed `ctr_data`, by hash, with
  exact equality. A second independent divergence sits downstream of the table.
- Nothing was weakened, no fixture was touched, nothing was regenerated.

**This is specced separately.** It is not a blocker for B04, B05 or B07.

### RESOLUTION (same-day follow-on): BUG-SFS

The residual was localized and fixed after this record was first written. Full
write-up: **SPEC.md §9**. In one paragraph: the STRUCTURE-search CTR column was
materialized under the raw identity permutation, but upstream builds `Folds[0]`
on the **already-S-shuffled** learn data (`ShuffleLearnDataIfNeeded`,
`preprocess.cpp:183`; `shuffle = foldIdx != 0`, `learn_context.cpp:526-529`), so
the structure fold's prefix order in original-object coordinates is `S` itself.
The averaging fold already composed `S` (plan 05-19); the two folds disagreed
about the CTR feature space, and on this fixture the structure search chose
bins `(6, 12)` instead of upstream's `(7, 10)` — bin 12 degenerate at apply
time. Fix: `boosting.rs` `structure_fold_columns` fold-0 branch, identity →
`S` under `need_shuffle`. Gate:
`crates/cb-train/tests/ctr_structure_fold_shuffle_test.rs` pins all 5 trees'
persisted CTR borders to upstream's committed pair; mutation M-SFS reproduced
the original residual byte-identically (`max |diff| = 1.371207585124875e-1`)
and was reverted. After the fix: `ctr_btmv_simple_oracle_test` **4/4**, the 11
CTR oracles + 3 one-hot targets + Counter + BUG-CTRB gates + cb-model suite all
green, cb-train sweep clean except the pre-existing monotone failure. Every
existing e2e oracle passes under both permutations (their structure argmax is
not near a tie), which is why this fixture was the first to see it. A-BTMV-1 is
now **MET**.

## 7. SPEC resolutions recorded in SPEC.md

1. **§3's recommendation RETRACTED in place** (struck through, not deleted), with
   the two-rules table, the `online_ctr.cpp` line citations, and the M2 evidence
   showing the retracted change is undetectable at binclf.
2. **R1 — RESOLVED:** `classes - 1`, floored. Not ambiguous; the
   preserves-non-mean-payloads fallback is satisfied too, so the choice is
   overdetermined.
3. **OQ1 — ANSWERED: no.** The 4th argument (`classes`) is correct
   (`online_ctr.cpp:909-912, 930-934`).
4. **OQ2 — ANSWERED: no.** `accumulate_online` has exactly one production caller;
   full call-site enumeration reproduced in SPEC.md §7.
5. **SPEC-BTMV-03 — CLOSED, not affected.** `online_mean_prefix` is
   upstream-exact at every class count; the scope-widening branch did not open.
6. **Out-of-scope finding recorded:** `ECtrType::target_border_count` has zero
   production callers because the online CTR-data allocation path
   (`online_ctr.cpp:738/741`) is not implemented here — with an explicit
   "do NOT wire it into the bake" warning, duplicated in the helper's own doc so
   it travels with the code.
7. **§1's symptom corrected:** the reported `1.283e-1` was accompanied by the
   anti-vacuity guard `predictions are constant — the gate would be vacuous`, not
   a `compare_stage` panic; the number is visible only under `-- --nocapture`.
