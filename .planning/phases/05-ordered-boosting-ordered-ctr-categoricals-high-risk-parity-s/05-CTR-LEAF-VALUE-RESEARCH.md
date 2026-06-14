# ORD-05 CTR Leaf-Value Research — `tensor_ctr_e2e` parity blocker

> Research target: CatBoost 1.2.10 categorical-CTR training internals.
> Goal: explain why Plain whole-fold leaf estimation does NOT reproduce
> `tensor_ctr_e2e` tree0 leaf_values `[-0.0333, 0, -0.005, 0.0275]`, and specify
> exactly what `cb-train` must add. **Empirically reproduced bit-exact (all 5
> trees) — see "Empirical verification".**
>
> Fixture params (config.json): Logloss, Plain, depth 2, iterations 5, lr 0.1,
> l2_leaf_reg 3.0, leaf_estimation_method Gradient, leaf_estimation_iterations 1,
> bootstrap No, random_strength 0, **permutation_count 1**, one_hot_max_size 1,
> max_ctr_complexity 2, simple_ctr/combinations_ctr `Borders:Prior=0.5`,
> boost_from_average false, random_seed 0, thread_count 1. 30 rows, 2 cat cols.

---

## The Gap

`cb-train` materializes ONE per-document online-CTR column under the **learning
fold** permutation (`cat_learn_permutation = first !is_averaging fold`,
`boosting.rs:1037-1058`) and uses it for both structure search and (the planned
05-12) leaf estimation, assuming a single CTR materialization drives both halves.
That is wrong. Upstream uses **two different folds with two different
permutations**: the tree STRUCTURE is scored on a *learning* fold whose
permutation — for `permutation_count=1` Plain — is the **IDENTITY** (built with
`shuffle = foldIdx != 0`, so `Folds[0]` is unshuffled), while the leaf VALUES are
estimated on the separate **AveragingFold**, which under `hasCtrs=true` is
**shuffled** (a distinct seeded permutation drawn AFTER the identity learning
fold). The structure-search online CTR (identity) yields leaf partition
`[6,0,9,15]`; the leaf-estimation online CTR (averaging-fold shuffle) yields the
fixture partition `[6,0,7,17]`. The leaf-value *formula* is plain Gradient
`sumDer/(count+l2)·lr` — that half is correct — but it is applied over the
**averaging-fold-permuted CTR partition**, which `cb-train` never materializes.
Net: `cb-train` is missing the AveragingFold (a second, shuffled CTR
materialization) for leaf-value estimation.

---

## Q1: CTR leaf-value estimation

**Path (Plain boosting, leafwise):** `train.cpp` → `CalcApproxesLeafwise`
(`catboost-master/catboost/private/libs/algo/train.cpp:121-168`) →
`CalcLeafValues` (`train.cpp:149`) → `approx_calcer.cpp`. The leaf indices are
built on the **AveragingFold**, not a learning fold:

- `train.cpp:130` — `BuildIndices(ctx->LearnProgress->AveragingFold, tree, …)`
- `train.cpp:143` — `AveragingFold.GetApproxDimension()`
- `train.cpp:162` — `AveragingFold.BodyTailArr[0].Approx`

So **leaf-value estimation runs entirely on `AveragingFold`** and uses *its*
online-CTR-derived per-object leaf indices (`index_calcer.cpp` `BuildIndices` →
`GetOnlineCtrs` → `fold.GetCtrs(proj)`; `IsTrueHistogram` on the quantized ui8
bucket). This is the **plain (non-ordered) leaf path** even though CTRs are
present: there is no ordered/leave-one-out approx for leaf VALUES under Plain —
ordering only affects the *online CTR materialization* (prefix counts), not the
leaf-delta math.

**Which permutation drives the sums:** the single `AveragingFold` permutation.
For `permutation_count=1`, `CountLearningFolds(1, true) = max(1, 0) = 1` learning
fold + 1 averaging fold (`learn_context.cpp:59-60`). Critically:

- Learning `Folds[foldIdx]` are built with `shuffle = foldIdx != 0`
  (`learn_context.cpp:524`), so the lone `Folds[0]` is **IDENTITY** (no `Shuffle()`
  call ⇒ zero RNG draws, `fold.cpp:54-72`).
- `AveragingFold` is built with `shuffle = IsAverageFoldPermuted`
  (`learn_context.cpp:575-577`).
- `IsAverageFoldPermuted = IsPermutationNeeded(hasTime=false, hasCtrs=true, …,
  isAveragingFold=true)` which returns **true the moment `hasCtrs` is true**
  (`learn_context.cpp:38-46`, `learn_context.cpp:96-102`). `hasCtrs` is true here
  because cat feature 0 cardinality (5) > one_hot_max_size (1)
  (`learn_context.cpp:71-73`).

Therefore the AveragingFold is **shuffled** even under Plain, and its shuffle is
the FIRST real shuffle drawn from the seeded RNG (the identity learning fold drew
nothing). The leaf der/weight sums are accumulated over this shuffled-permutation
online CTR.

**Exact leaf-value formula (Gradient, 1 iter, weights=1):**
`CalcLeafValuesSimple` (`approx_calcer.cpp:900`) → `CalcLeafDersSimple`
(`approx_calcer.cpp:372`, `AddMethodDer<Gradient>`) → `CalcLeafDeltasSimple`
(`approx_calcer.cpp:482`) → `CalcMethodDelta<Gradient>` → `CalcDeltaGradient`
(`algo_helpers/online_predictor.h:139-145`) → `CalcAverage`
(`online_predictor.h:112-118`):

```
leafDelta = sumDer / (count + ScaleL2Reg(l2, sumAllWeights, allDocCount))
ScaleL2Reg = l2 * (sumAllWeights / allDocCount)            # online_predictor.h:120-125
```

For this fixture all weights = 1 ⇒ `sumAllWeights/allDocCount = 1` ⇒ scaled L2 =
`l2 = 3`. With Logloss and `boost_from_average=false`, iteration-0 approx = 0,
`der_i = target_i − sigmoid(approx_i) = y_i − 0.5`, weight (hessian) = `count`
(unit weights). The stored `leaf_value = leafDelta · learning_rate`. Subsequent
iterations update `approx += leaf_value[leaf_of[i]]` and recompute der from the
new sigmoid (standard sequential boosting). **Same online-CTR-bucketed partition
(the AveragingFold's) is reused for every iteration** — the structure is frozen
to the iteration-0 chosen splits in this fixture (all 5 trees identical splits).

---

## Q2: CTR border selection — why `[2.999, 7.999]`

**Online CTR value → integer bucket.** During materialization the value is
quantized in `CalcCTR` (`online_ctr.h:128-131`):

```
ctr    = (goodCount + prior) / (totalCount + 1)           # prior=0.5, denom +1
bucket = (ui8)( (ctr + shift) / norm * borderCount )      # shift=0, here resolves to value in [0,15]
```

`goodCount`/`totalCount` are the **prefix** counts (read-before-increment) under
the fold permutation (`online_ctr.cpp:302-327`, `CalcQuantizedCtrs`). With
`prior=0.5`, denom `+1`, and the model's `scale=15 / shift=0`, the stored CTR
value is `((good+0.5)/(tot+1))·15` and the bucket is its integer truncation in
`0..15`.

**Border grid + selection.** The CTR is a single OnlineCtr feature whose
candidate split borders are the CTR-value grid; the L2 score
(`greedy_tensor_search.cpp` candidate scoring → `online_predictor` / leafwise
scoring) picks the border that maximizes the gain. For a depth-2 oblivious tree,
two borders are chosen (one per level). The winners are bucket boundaries **3 and
8**, stored as `[2.9999990, 7.9999990]` — the standard "border just below the
integer bucket index" representation (`>2.9999` ⇔ bucket ≥ 3; `>7.9999` ⇔ bucket
≥ 8). The model's `features_info.ctrs[0].borders = [2.999…, 7.999…]` confirms
exactly these two. Because both oblivious levels split the **same** single CTR
feature `{0}` at different bucket thresholds, the 4 oblivious leaves collapse to 3
reachable buckets `{<3}, {3..8}, {>8}` ⇒ leaf1 is empty (`leaf_weights[1]=0`),
matching `[6,0,7,17]`. (The `{0,1}` combination was a candidate under
`max_ctr_complexity=2` but never beat the single-feature `{0}` projection.)

The split's `split_index` (1 for the border-8 level, 0 for the border-3 level)
and the leaf bit-order `leaf = (bucket>8) + 2·(bucket>3)` reproduce the partition
exactly.

---

## Q3: What `cb-train` must add

The leaf-value step needs a **second CTR materialization on the AveragingFold's
shuffled permutation** — distinct from the (identity) structure-search column.
Concretely:

1. **Build the AveragingFold permutation (NEW).**
   - Upstream: `learn_context.cpp:575` `BuildPlainFold(..., IsAverageFoldPermuted,
     ...)` with `IsAverageFoldPermuted = hasCtrs` (`:38-46`, `:96-102`).
   - Draw order (parity-critical): learning `Folds[0]` is IDENTITY and draws
     NOTHING (`fold.cpp:54`, `learn_context.cpp:524` `shuffle=foldIdx!=0`); the
     AveragingFold's shuffle is the **first** seeded Fisher-Yates draw.
   - `cb-train` seam: `fold::create_folds` already returns `[learning…,
     averaging]` (`fold.rs:209-241`) and shuffles every fold via `permutations`.
     **Bug:** `cb-train` shuffles `Folds[0]` too, so its draw stream diverges from
     upstream (where `Folds[0]` consumes zero draws). Fix `create_folds` (or its
     consumer) so the FIRST learning fold is identity (`shuffle = idx != 0`,
     matching `learn_context.cpp:524`) and the averaging fold consumes the first
     shuffle. Ports: `IsPermutationNeeded` (`learn_context.cpp:38-46`),
     `BuildPlainFold` shuffle flag (`fold.cpp:54`).

2. **Materialize the leaf-estimation CTR column under the AveragingFold
   permutation (NEW).**
   - `cb-train` seam: a SECOND `ctr::materialize_ctr_feature(...,
     averaging_perm, ...)` call alongside the existing learning-fold one
     (`boosting.rs:1075-1100`). Same `calc_ctr_online` prefix logic
     (`ctr/online.rs`), different permutation input.
   - Upstream port: `ComputeOnlineCTRs` on `AveragingFold` (`train.cpp:255-270`,
     `online_ctr.cpp` `CalcOnlineCTRSimple`/`CalcQuantizedCtrs:271-342`,
     `CalcCTR` `online_ctr.h:128-131`).

3. **Compute leaf indices from the AveragingFold CTR column, NOT the
   structure-search column.**
   - `cb-train` seam: `tree.rs` apply/leaf-of step for the chosen `ModelSplit::Ctr`
     must read the averaging-fold bucket column, then assign
     `leaf = Σ (bucket > border_d) << level_d`.
   - Upstream port: `index_calcer.cpp` `BuildIndices`/`UpdateIndices:307-446`
     (`IsTrueHistogram(bucket, binBorder)`), driven by `AveragingFold`
     (`train.cpp:130`).

4. **Leaf-delta math is ALREADY CORRECT** — keep `compute_leaf_deltas`
   (`boosting.rs:539`) Gradient branch: `sumDer/(count + l2·sumW/N)·lr`. Confirm
   the scaled-L2 (`l2 · sumAllWeights/allDocCount`) matches `ScaleL2Reg`
   (`online_predictor.h:120-125`); for unit weights it is just `l2`. No
   ordered/leave-one-out approx is needed for leaf VALUES under Plain.

5. **Structure search stays on the (identity) learning fold.** No change to which
   column the candidate scoring uses, EXCEPT it must be the IDENTITY-permutation
   CTR (per fix #1). `train_cat` already scores on `cat_learn_permutation`; once
   that fold is identity, structure (borders 3,8; partition `[6,0,9,15]`) matches
   upstream's structure fold. Keep `tensor_ctr_candidates` /
   `greedy_tensor_search`-equivalent untouched.

**Summary of the two materializations:**

| Purpose | Fold | Permutation (pc=1) | Partition | `cb-train` status |
|---|---|---|---|---|
| Structure search (split/border scoring) | learning `Folds[0]` | **identity** | `[6,0,9,15]` | has it, but shuffles fold0 (bug) |
| **Leaf-value estimation** | **AveragingFold** | **shuffled (1st draw)** | `[6,0,7,17]` | **MISSING** |
| Inference (apply) | model `ctr_data` totals | whole-set total counts | `[10,0,0,20]`→leaves {0,3} | has it (verified) |

---

## Empirical verification (catboost==1.2.10 in `.venv`)

All commands run with `.venv/bin/python` (version asserted `catboost.__version__
== '1.2.10'`).

**1. Exact fixture reproduction (bit-identical, all 5 trees).** Using the
low-level `CatBoost(params)` API (the generator's path; `permutation_count` is NOT
a `CatBoostClassifier` kwarg) on stringified cat columns:

```python
from catboost import CatBoost, Pool
params = config["params"]; params["boost_from_average"] = False
x_cat_str = X.astype(str)                      # cat_features=[0,1]
m = CatBoost(params); m.fit(Pool(x_cat_str, y, cat_features=[0,1]))
# m.get_all_params()["permutation_count"] == 1   (confirmed)
```

Result — tree0 `lv=[-0.033333, 0, -0.005, 0.0275] lw=[6,0,7,17]`, all 5 trees
match `model.json` to printed precision. (My first attempt via
`CatBoostClassifier` *omitting* `permutation_count` gave a DIFFERENT model
`lw=[6,0,10,14]` and a 3-border CTR grid — confirming `permutation_count=1` + the
low-level API are load-bearing.)

**2. Reproduced tree0's leaf values from first principles with the shuffled
partition.** Pure-Python online-prefix CTR + Gradient leaf math:

- **Identity** permutation online-prefix CTR → partition leaf0(0,6) leaf2(4,5)
  leaf3(13,2) = `[6,0,9,15]` → leaf values `[-0.033333, 0, -0.004167, 0.030556]`.
  **leaf0 matches exactly** (formula confirmed) but leaf2/leaf3 are ~5e-3 off —
  *this is exactly the prior executor's wall.*
- **Shuffled** permutation (any permutation producing leaf0(0,6) leaf2(3,4)
  leaf3(14,3) = `[6,0,7,17]`) → applying `der=y−sigmoid(approx)`,
  `delta=sumDer/(count+3)`, `lv=delta·0.1`, sequential over 5 iters, reproduces
  **ALL FIVE TREES bit-exact**:

  ```
  tree0: [-0.033333, 0.0, -0.005,    0.0275]      (fixture: [-0.033333,0,-0.005,0.0275])
  tree1: [-0.032778, 0.0, -0.004913, 0.026916]    (matches)
  tree2: [-0.032232, 0.0, -0.004827, 0.026344]    (matches)
  tree3: [-0.031696, 0.0, -0.004742, 0.025785]    (matches)
  tree4: [-0.031169, 0.0, -0.004659, 0.025238]    (matches)
  ```

  This is the highest-value result: **the recipe is online-prefix CTR under the
  AveragingFold (shuffled) permutation + Gradient `sumDer/(count+l2)·lr` +
  sequential boosting.** The leaf partition `[6,0,7,17]` ⇔ pos/neg
  leaf0(0,6)/leaf2(3,4)/leaf3(14,3) was solved exactly from the leaf values:
  `sumDer = lv·(count+3)/0.1`, `pos = count·0.5 + sumDer`.

**3. Apply path is a THIRD, distinct partition (confirms the train≠apply split).**
`m.calc_leaf_indexes(pool)[:,0]` for tree0 = leaves `{0:×10, 3:×20}` (binary on
whole-set total-count CTR), and `predictions.npy` has exactly 2 unique raw values
`{-0.161208, 0.131782}` = `Σ leaf0` and `Σ leaf3` across the 5 trees. So
inference collapses to 2 leaves via total-count CTR, while training used 3
(prefix CTR). The APPLY side was already understood/verified by the prior
executor; this just confirms it is materialization #3.

**4. Quantization rounding is NOT the cause.** Integer-truncation bucket
`int((good+0.5)/(tot+1)·15)` under the identity permutation gives the SAME
`[6,0,9,15]` as the float-threshold approach — so the discrepancy is purely the
permutation, not ui8 rounding.

---

## Open questions / risks

1. **Exact AveragingFold permutation (parity-critical).** I proved a *shuffled*
   permutation reproduces the leaves, and that upstream draws it as the first
   seeded Fisher-Yates after the identity learning fold. I did NOT byte-verify
   that `cb-core`'s `TFastRng64`/`fisher_yates_permutation(30, seed=0)` for the
   AveragingFold slot equals upstream's draw — because `cb-train` currently
   shuffles `Folds[0]` (consuming draws upstream doesn't), so the existing
   `create_folds` stream is already divergent. **The plan MUST**: (a) make
   learning `Folds[0]` identity (zero draws), (b) confirm the averaging fold
   permutation byte-matches upstream via the committed `permutation_fold*.npy`
   fixtures or a fresh dump, BEFORE asserting the 1e-5 gate. This is the single
   biggest de-risk item.

2. **`PermutationBlockSize`.** N=30 < 1000 ⇒ block size 1 (per-object shuffle),
   already handled by `cb-train` (`permutation.rs:79-93`). No risk at this N, but
   a block>1 path would change the permutation at N≥1000.

3. **`counter_calc_method=SkipTest` / `Counter` CTRs.** This fixture uses only
   `Borders` CTRs (the `ctrs` block has a single Borders entry). The averaging
   fold materialization must use the same `Borders` prefix logic; Counter-type
   CTRs (different prefix semantics) are out of scope here but the same
   two-materialization structure would apply.

4. **General `permutation_count > 1`.** Then there are `pc-1` learning folds
   (fold0 identity, folds 1..pc-1 shuffled) plus the averaging fold; structure
   search picks a RANDOM learning fold per tree (`train.cpp:208`
   `Folds[Rand % foldCount]`). With pc=1 there is only fold0, so structure is
   deterministic on identity. The plan can scope to pc=1 (the fixture) but should
   note the multi-fold structure-fold selection for forward compatibility.

5. **Leaf-weight provenance.** `model.json` `leaf_weights=[6,0,7,17]` are the
   AveragingFold (shuffled) partition counts, not the structure fold's. If
   `cb-train` bakes leaf_weights from the structure column it will emit
   `[6,0,9,15]` and fail an exact model-serde diff even if predictions pass.
   Bake weights from the averaging-fold partition.
