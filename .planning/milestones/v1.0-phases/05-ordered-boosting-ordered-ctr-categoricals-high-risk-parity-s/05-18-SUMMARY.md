---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
plan: 18
subsystem: cb-train (multi-permutation AveragingFold cat-CTR) + live-trainer instrumentation
gap_closure: true
tags: [ORD-01, SC-1, parity, ctr, fold, instrumentation, live-trainer, spike-001, fallback]
requires:
  - "catboost 1.2.10 in .venv (offline oracle)"
  - "Conan 2.29 + Ninja 1.13 + clang-18/lld-18 + Python 3.13 (persisted from 05-17 in /tmp)"
provides:
  - "SELF-CONSISTENT live-trainer oracle (live_trainer_self_consistent.json) — Spike-001 inconsistency RESOLVED"
  - "Precise bar-(c) root cause: data-provider storage reorder S (Q = S o LearnPermutation)"
affects:
  - "Bar (c) pc=4 e2e prediction oracle (DEFERRED via user-chosen FALLBACK — requires porting S)"
outcome: FALLBACK (bar (c) deferred; Spike-001 resolved; production untouched)
key-files:
  created:
    - crates/cb-train/tests/fixtures/multi_permutation_fold/live_trainer_self_consistent.json
  modified:
    - crates/cb-train/tests/fixtures/multi_permutation_fold/live_trainer_ctr_bins_blocker.json  # annotated superseded
    - crates/cb-oracle/generator/instrument_live_trainer_README.md  # second cycle + S root cause
    - .planning/phases/05-.../deferred-items.md
  instrumentation_untracked:
    - catboost-master/catboost/private/libs/algo/train.cpp        # self_consistent_ctr event (vendored, untracked)
    - catboost-master/catboost/private/libs/algo/online_ctr.cpp   # online_ctr_inputs event (vendored, untracked)
  untouched_by_fallback:
    - crates/cb-train/src/fold.rs
    - crates/cb-train/src/boosting.rs
    - crates/cb-train/tests/multi_permutation_e2e_oracle_test.rs  # pc=4 e2e, stays UNCOMMITTED
---

# Plan 05-18 SUMMARY — bar (c) re-plan: Spike-001 RESOLVED, root cause localized to storage reorder S, FALLBACK taken

## Outcome

The re-authorized second live-trainer instrumentation cycle **succeeded** and
derived a **self-consistent oracle**, RESOLVING the Spike-001 inconsistency. It
also **precisely localized** the remaining bar-(c) blocker — which turns out to be
deeper and more specific than any prior analysis. Per the user's decision
(2026-06-15), the documented **FALLBACK** was taken: cb-train production is
untouched, the pc=4 e2e oracle stays uncommitted, no oracle was weakened. Bars
(a),(b),(d),(e) remain green; bar (c) is deferred to a future plan.

## Tasks 1–2 (executed)

- **Task 1 (toolchain).** Disk gate NOT tripped (50G free > 40G). The 05-17
  toolchain + instrumented build PERSIST in `/tmp` (`/tmp/clang18_prefix`,
  `/tmp/cb_build313`) — no multi-hour rebuild needed; the existing `.so` reproduced
  `predictions_pc4.npy` bit-identically (0.0). STATE.md's "ephemeral prefix gone"
  was wrong.
- **Task 2 (re-instrument + capture).** Added two atomic `CB_INSTRUMENT_LOG` events:
  `train.cpp` `self_consistent_ctr` (per-fold projection + `LearnPermutationFeaturesSubset`
  + `LearnTargetClass` + `GetData` bins) and `online_ctr.cpp` `online_ctr_inputs`
  (the LITERAL `CalcOnlineCTRSimple` inputs: `perm_subset`, reindexed `enumerated`,
  every classifier's `target_classes`). Incremental rebuilds (~6s each). Predictions
  stayed bit-identical (0.0) to `predictions_pc4.npy` — instrumentation faithful.

## The Spike-001 inconsistency, explained (the breakthrough)

The averaging online-CTR ui8 bins (`GetData`/`Feature[docIdx]`) are stored in the
**CTR materialization order Q**, where **Q = S ∘ LearnPermutation** and `S` is the
catboost quantized data-provider's internal object STORAGE reorder. The first cycle
paired those bins with `GetLearnPermutationArray()` = `[11,18,15,29,…]` (the
leaf-index *iteration* order, a DIFFERENT order) and never logged `LearnTargetClass`
— so `(perm, bins)` were mutually inconsistent (Spike-001 VERDICT).

With the atomic capture, the bins ARE the single-cat-0 Borders online prefix under
order **Q** with target `LearnTargetClass[1]` (= binarized y; `LearnTargetClass[0]`
is all-zeros / unused — a second prior red herring). **Q reproduces all five tree
partitions `[6,0,10,14],[8,8,0,14],[6,0,10,14],[8,8,0,14],[8,8,0,14]` bit-exact**,
and the structure-fold cycle is `[0,2,0,2,2]`. Committed as
`live_trainer_self_consistent.json`; the inconsistent blocker is annotated
`superseded_by` it (kept for the evidence trail).

## Why bar (c) is DEFERRED (the precise, decisive blocker)

cb-train's online-CTR math is ALREADY correct (Spike-001 proof). The gap is `S`.
pc=4 tree-B borders `[3,7]` **split the mixed cat buckets** (cat3, cat4), so the
leaf VALUES depend on the exact per-mixed-bucket bin→object assignment, which is
fixed by `S`. Evidence (tree-B):

| avg permutation | partition | leaf1 (sum_y,count) | leaf3 |
|---|---|---|---|
| upstream Q = S∘LearnPerm | [8,8,0,14] | **(5,8)** | **(12,14)** |
| must_haves `[11,18,15,…]` | [8,8,0,14] ✓ | (6,8) ✗ | (11,14) ✗ |
| cb-train current avg-perm | [9,7,0,14] ✗ | (4,7) ✗ | (12,14) |

The must_haves' `[11,18,15,…]` reproduces the PARTITION but the wrong leaf VALUES
(off by one object's gradient, ≫1e-5). cb-train materializes CTRs on object-order
`X_cat` WITHOUT `S`; no `create_folds`-generable permutation reproduces upstream's
pc=4 leaf values, and re-pinning to a lucky permutation is forbidden (invariant #2).
**pc=1 / `tensor_ctr_e2e` (green ≤1e-5) is green only because its borders do not
split the mixed buckets** (leaf composition is order-invariant there) — NOT because
cb-train reproduces the CTR order. `S` itself is an arbitrary
hash/quantization-driven data-provider internal order (verified: not a sort of any
column).

Closing bar (c) requires porting `S` (the data-provider quantized-object storage
order) into cb-train so its averaging-CTR materialization interleaves mixed buckets
exactly as upstream — a research-grade subsystem, out of this plan's scope, and a
new plan.

## Invariants honored

- cb-train production (`fold.rs`/`boosting.rs`) byte-unchanged (`git status` clean).
- pc=4 e2e oracle (`multi_permutation_e2e_oracle_test.rs`) UNCOMMITTED.
- No oracle weakened / `#[ignore]`d / `assert_ne`d / tolerance-loosened. No re-pin to
  a cb-train value or to the superseded blocker fixture.
- Committed green oracle re-checked: `multi_permutation_fold_oracle_test` 4/4.

## Next

A future plan to port the data-provider storage order `S` into cb-train's CTR
materialization (the only remaining bar-(c) gap), or accept bar (c) as a documented
parity limitation and finalize Phase 5 on bars (a),(b),(d),(e).
