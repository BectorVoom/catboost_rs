---
spike: 001
name: online-ctr-averaging-fold-offline
type: standard
validates: "Given the committed inputs (col0, y, upstream_avg_perm, prior 0.5, border_count 15), when upstream catboost 1.2.10's ComputeOnlineCTRs(AveragingFold) is reimplemented offline, then the reconstructed ui8 bins equal upstream_avg_ctr_bins_avg_order bit-exact"
verdict: NOT-ACHIEVABLE
related: []
tags: [ctr, parity, online-ctr, averaging-fold, phase-05, ord-01, pc4, bar-c]
---

# Spike 001: Online-CTR AveragingFold — Offline Reproducibility

## What This Validates

Given the committed inputs — cat column 0, target `y`, the instrumented averaging
permutation `upstream_avg_perm`, prior 0.5, ctr_border_count 15 — **can
catboost 1.2.10's training-time `ComputeOnlineCTRs(AveragingFold)` ui8 bins
(`upstream_avg_ctr_bins_avg_order`) be reproduced BIT-EXACT in pure offline
code, or are they irreducibly a live-training-time artifact?**

This is the feasibility gate for the Phase-5 bar-(c) (pc=4 SC-1/ORD-01) re-plan.
Outcome decides: re-plan 05-18 for a real offline CTR port (if ACHIEVABLE) vs.
a different path (if NOT).

## VERDICT: NOT-ACHIEVABLE — and the committed oracle is internally inconsistent

This is a **mathematical impossibility proof**, not a "couldn't find the rule"
result. Two load-bearing conclusions:

1. **cb-train's online-CTR code is ALREADY bit-exact to the upstream C++
   algorithm.** `materialize_ctr_feature` / `ctr/online.rs` / `calc_ctr.rs`
   correctly implement `CalcOnlineCTRSimple → CalcQuantizedCtrs` with
   `CalcCTR = (good+0.5)/(total+1)`, shift 0 / norm 1 for prior 0.5, ui8
   truncation. The bar-(c) gap is **NOT** in our CTR materialization.

2. **The committed pair (`upstream_avg_perm`, `upstream_avg_ctr_bins_avg_order`)
   is mutually inconsistent under the documented upstream algorithm.** Fed the
   committed inputs, the deterministic upstream algorithm provably yields a
   *different* bin sequence than the committed bins. So the committed bins do
   not correspond to the committed permutation — they encode live per-tree fold
   state (the effective `LearnPermutation`/`LearnTargetClass` in force when
   `ComputeOnlineCTRs` ran for tree-B) that no committed input captures.

## How to Run

```bash
python3 .planning/spikes/001-online-ctr-averaging-fold-offline/reconstruct.py
```

## What to Expect

`H1 global online prefix (good=class1), avg order: 6/30 match`, first divergence
at position 4 (recon 7, target 3) — exactly the divergence recorded in
`live_trainer_ctr_bins_blocker.json`. No faithful variant exceeds 8/30.

## The Proof (decisive, not partial)

For a single-cat Borders online-prefix CTR on `cat_feature 0` under a *fixed*
permutation, each cat-value bucket's bin sequence is **fully determined** with
zero free parameters: in bucket-appearance order, `total = 0,1,2,…` and
`good` = running count of the good class among prior appearances.

- **cat0 has 6 docs, ALL y=0** (confirmed by raw data and the model's baked
  whole-set count `[N0=6, N1=0]`). Its upstream target bins are
  `[11, 12, 11, 7, 13, 5]` — high and **non-monotone**.
- good=class1 on an all-y=0 bucket forces `good≡0`, giving the strictly
  determined `[7,3,2,1,1,1]`. good=class0 gives `[7,11,12,13,13,13]`. Neither is
  the target, and **no ordering or block structure can change a fully-determined
  sequence**.
- Mixed bucket cat3 target `[7,11,3,7,9,10,1,13]` jumps ctr 0.1→0.85 (the `1`→`13`
  step) while `total` only grows — impossible for any function of a
  monotonically-growing `(good,total)` prefix.

Best offline match across **all** faithful interpretations (good=class1/class0;
order = avg_perm / inverse / natural; block counts 1–30 with faithful
`SumCtrsFromBlocks` pre-seeding; col0 / col1 / combination keys): **8/30**.

## Investigation Trail

1. **H0 — reproduce the documented divergence.** Implemented `CalcCTR`/ui8
   quantization bit-exact (`online_ctr.h:128–131`, `online_ctr.cpp:102–111`),
   global read-before-increment prefix in `upstream_avg_perm` order, good=class1.
   Result: 6/30, divergence at pos 4 — matches the blocker JSON. Confirms our
   model is faithful to the documented algorithm.
2. **H1 — block structure / `SumCtrsFromBlocks` pre-seeding.** Read
   `online_ctr.cpp:242` (`SumCtrsFromBlocks`), `:271` (`CalcQuantizedCtrs`),
   `:344` (`CalcOnlineCTRSimple`). With `thread_count=1`, `GetThreadCount()+1 ⇒ 1`
   block ⇒ pre-seeding is a no-op; the path reduces to the global prefix. Swept
   block counts 1–30 anyway. Result: block structure is NOT the missing
   ingredient.
3. **H2 — good-class / target-classifier choice.** Read `fold.cpp:322–353`
   (`AssignTarget`): `permutedTargetClass = fold.LearnTargetClass` is fixed at
   fold construction on raw `y`, NOT iteration/approx-dependent. So fixed-y
   reordered by the permutation is exactly the right input — and it provably
   cannot produce the target (per-bucket determinism proof above).
4. **H3 — wrong column / combination key.** Tested col1 and the {0,1}
   combination hash. None reach the target; max 8/30.
5. **Per-bucket determinism proof.** Generalized: since every bucket's sequence
   is forced once the good-class is chosen, and the target bins violate
   monotonicity-of-prefix within a bucket, NO offline `(order, block, key,
   good-class)` choice can match. Impossibility, not search exhaustion.
6. **Consistency cross-checks (guard against false negative).** (a) The baked
   whole-set `ctr_data.hash_map` counts `[6,0],[4,0],[2,6],[1,6],[0,5]` ARE
   reproduced offline — our whole-set statistic is correct; only the per-doc
   online sequence is unreachable. (b) `target bin → leaf` over
   `upstream_leaf_indices_treeB_avg_order` is clean/monotone (bin≤3→leaf0,
   bin∈[5,7]→leaf1, bin≥8→leaf3; splits at borders 3 and 8, matching model.json
   `2.9999/7.9999`), so the target really is the cat0 simple-Borders online CTR —
   the impossibility result is load-bearing, not a mislabeled-column artifact.

## C++ ground truth read

- `catboost-master/catboost/private/libs/algo/online_ctr.cpp`:
  `CalcOnlineCTRSimple` (344), `CalcQuantizedCtrs` (271), `SumCtrsFromBlocks`
  (242), `ComputeOnlineCTRs` (600/849).
- `online_ctr.h:128–131` (`CalcCTR`), `online_ctr.cpp:102–111` (shift/norm).
- `fold.cpp:222` (`BuildPlainFold`), `:322–353` (`AssignTarget`).
- `greedy_tensor_search.cpp:675/809/1104` (`ComputeOnlineCTRs` call sites).
- Our (correct) impl: `crates/cb-train/src/ctr/online.rs`, `calc_ctr.rs`,
  `ctr_feature.rs` — bit-exact to the C++; the gap is NOT here.

## Results / Signal for the re-plan

- **Bar (c) cannot be closed by changing `materialize_ctr_feature`** or any
  online-prefix / border-quantization rule — those are already correct. The
  05-18 plan's hypothesized "fix #3" was aimed at code that is not wrong.
- **The committed `upstream_avg_perm` + `upstream_avg_ctr_bins_avg_order` cannot
  serve as an offline oracle** — they are internally inconsistent under the
  upstream algorithm and should be treated as SUSPECT, not as a parity target.
- **To proceed, the re-plan needs LIVE-trainer instrumentation** (recipe in
  `.planning/phases/05-.../05-17-SUMMARY.md`) to capture the *actual* per-tree
  fold state — the effective `LearnPermutation` and `LearnTargetClass` in force
  for tree-B — and re-derive a *self-consistent* `(permutation, bins)` pair.
  Only then is an offline oracle reconstructable.
- This raises the broader question whether the 05-17 "compensating errors"
  conclusion itself partly rests on the inconsistent committed pair — worth a
  re-examination during re-plan.
