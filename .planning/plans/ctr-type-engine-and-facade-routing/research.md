# Research: CTR-type engine (Part 1) + `cat_features` facade routing (Part 2)

**Date:** 2026-08-01
**Repo:** `/home/user/Documents/workspace/catboost_rs`, branch `fix/bootstrap-rng-draw-accounting`
**Parity bar:** ≤1e-5 vs `catboost==1.2.10`
**Upstream reference:** GitHub tag `v1.2.10` (fetched live this session; see §11)

Evidence tags: `[VERIFIED: LOCAL <path:line>]`, `[VERIFIED: WEB <url>]`,
`[VERIFIED: CODEGRAPH <symbol>]`, `[VERIFIED: EXPERIMENT <script>]`,
`[INFERRED: …]`, `[UNVERIFIED: …]`.

---

## 0. Executive summary — the four findings that change the plan

1. **The scope of Part 1 is SMALLER than the brief assumes: three new online arms, not five.**
   CatBoost **CPU rejects `FloatTargetMeanValue` and `FeatureFreq` outright** at options
   validation. Only `Borders`, `Buckets`, `BinarizedTargetMeanValue`, `Counter` are legal on CPU.
   `[VERIFIED: WEB restrictions.h:18-48]` `[VERIFIED: EXPERIMENT probe1.py — both types raise
   "Ctr type X is not implemented on CPU yet" from catboost_options.cpp:509]`

2. **The scope of Part 1 is BIGGER than the brief assumes in a different dimension:
   `simple_ctr` is not one type with one prior — upstream's default is a LIST of TWO
   descriptions × multiple priors × multiple target borders.** The default CPU
   `simple_ctr` is `[Borders(priors 0/1, 0.5/1, 1/1), Counter(prior 0/1)]`, and each
   `(description, targetBorderIdx, prior)` triple becomes its **own** candidate CTR feature
   column. The repo models a single `ECtrType` + a single prior list and only ever uses
   `priors.first()`. `[VERIFIED: EXPERIMENT probe2.py get_all_params()]`
   `[VERIFIED: WEB catboost_options.cpp:439-472]` `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:3155]`

3. **`counter_calc_method` is unobservable without an eval set — and it only affects `Counter`.**
   With no eval set, `Full` and `SkipTest` produce **bit-identical** predictions. With an eval set
   they differ (counts + `CounterDenominator` diverge). So it cannot be oracle-tested by a
   learn-only fixture. `[VERIFIED: EXPERIMENT probe6.py/probe7.py]`

4. **The oracle path is far better than feared: `catboost==1.2.10` is importable in this repo's
   `.venv`, and all four CPU-legal CTR types train and emit fully-typed model JSON.** A new
   fixture family can be produced end-to-end today. The frozen-corpus hazard is real and the
   `gen_fixtures.py` fallthrough claim is **CONFIRMED**, but there is an established safe pattern.
   `[VERIFIED: EXPERIMENT .venv/bin/python -c "import catboost" → 1.2.10]`
   `[VERIFIED: LOCAL crates/cb-oracle/generator/gen_fixtures.py:3861 __main__ block]`

---

## A. The ONLINE (training-time) formula for each non-Borders CTR type

### A.0 The single shared quantizer — there is no per-type shift/norm difference

Every online arm calls the **same** `CalcCTR`:

```cpp
// catboost/private/libs/algo/online_ctr.h:128-131  (v1.2.10 — line numbers match the
// in-repo transcription citations exactly)
inline ui8 CalcCTR(float countInClass, int totalCount, float prior,
                   float shift, float norm, int borderCount) {
    float ctr = (countInClass + prior) / (totalCount + 1);
    return (ctr + shift) / norm * borderCount;
}
```
`[VERIFIED: WEB https://raw.githubusercontent.com/catboost/catboost/v1.2.10/catboost/private/libs/algo/online_ctr.h]`

and the **same** normalization:

```cpp
// online_ctr.cpp:102-111
float left = Min(0.0f, prior); float right = Max(1.0f, prior);
shift = -left; norm = right - left;
```
`[VERIFIED: WEB online_ctr.cpp:102-111]`

**Answer to "any per-type shift/norm difference": NO.** `CalcNormalization` depends only on the
prior numerator, never on `ECtrType`. The repo's `calc_normalization`
`[VERIFIED: LOCAL crates/cb-train/src/ctr/calc_ctr.rs:60-66]` is already type-agnostic and correct
for all arms. The hard `+1` denominator in `CalcCTR` is likewise shared by **all** online arms —
including `Counter`, where the "+1" is added to `CounterDenominator`, not to a per-bucket total.

**Therefore each new arm is ONLY a new `(numerator, denominator)` producer.**
`calc_ctr_online` / `calc_ctr_online_bin` `[VERIFIED: LOCAL crates/cb-train/src/ctr/calc_ctr.rs:76,89]`
**do not need per-type variants** — they need per-type *inputs*. This is the single most
plan-simplifying fact in this document.

> **Float-width caveat `[INFERRED, medium risk]`:** upstream `CalcCTR` is entirely `float` (f32):
> `countInClass`, `prior`, `shift`, `norm` are `float`; the division and the `(ctr+shift)/norm*borderCount`
> product are `float`; the return is an implicit `float → ui8` truncation. The repo computes in `f64`
> `[VERIFIED: LOCAL calc_ctr.rs:76-101]`. For `Borders`/`Buckets`/`Counter` the numerator/denominator
> are exact small integers so f32 and f64 agree well before the truncation boundary. For
> **`BinarizedTargetMeanValue` the numerator is a running f32 `Sum`**, so an f64 division of an f32
> sum can land on the other side of a bin boundary in principle. Recommendation: compute the
> BTMV arm's quantization in `f32` to mirror upstream exactly, and add a differential test.

### A.1 Buckets (`ECtrType::Buckets = 1`)

Online entrypoint: `CalcOnlineCTRClasses` (also serves `Borders` when `targetClassesCount > 2`).
`[VERIFIED: WEB online_ctr.cpp:144-215; dispatch at :769-770]`

Per-bucket state: `TBucketsView bv(leafCount, targetClassesCount)` holding, per bucket,
one `Total` int plus a `targetClassesCount`-wide class-count array `[VERIFIED: WEB online_ctr.cpp:70-99,166]`.

Per document, **in this exact order** `[VERIFIED: WEB online_ctr.cpp:168-185]`:

```cpp
int goodCount = totalCountByDoc[d] = bv.GetTotal(elemId);      // READ total (prefix)
auto bordersData = bv.GetBorders(elemId);                       // READ class counts (prefix)
for (border = 0; border < targetBorderCount; ++border) {
    UpdateGoodCount(bordersData[border], ctrType, &goodCount);  // see below
    goodCountByBorderByDoc[border][d] = goodCount;
}
if (datasetIdx == 0) {                 // learn only — test docs never increment
    ++bordersData[permutedTargetClass[docId]];
    ++bv.GetTotal(elemId);
}
```

with `[VERIFIED: WEB online_ctr.cpp:115-121]`:

```cpp
static void UpdateGoodCount(int curCount, ECtrType ctrType, int* goodCount) {
    if (ctrType == ECtrType::Buckets) { *goodCount = curCount; }
    else                              { *goodCount -= curCount; }
}
```

**Buckets numerator/denominator for target border `b`:**
- numerator = `N[b]` — the prefix count of documents in class **exactly** `b`
- denominator = `Total` = Σ over all classes (the prefix total)
- value = `CalcCTR(N[b], Total, prior, shift, norm, ctrBorderCount)`

**Borders-multiclass numerator (for completeness, `targetClassesCount > 2`):**
`goodCount = Total − Σ_{c ≤ b} N[c]` = the prefix count of documents with class **strictly greater
than** `b`. For binclf (`targetClassesCount == 2`, `targetBorderCount == 1`) this collapses to
`Total − N[0] = N[1]`, exactly the existing `CalcOnlineCTRSimple` fast path
`[VERIFIED: WEB online_ctr.cpp:300-307]` and the repo's `online_ctr_prefix_binclf`
`[VERIFIED: LOCAL crates/cb-train/src/ctr/online.rs:299-310]`. **The existing binclf Borders path is
therefore provably a special case of the generic classes path** — a useful regression anchor.

**TargetBorderCount interaction (the structural surprise):**

```cpp
// catboost/private/libs/algo/ctr_helper.h:34-42
inline int GetTargetBorderCount(const TCtrInfo& ctrInfo, ui32 targetClassesCount) {
    if (Type == BinarizedTargetMeanValue || Type == Counter) return 1;
    return Type == Buckets ? targetClassesCount : targetClassesCount - 1;
}
```
`[VERIFIED: WEB ctr_helper.h:34-42]`

For binclf (`targetClassesCount == 2`):

| type | targetBorderCount | candidate columns per prior |
|---|---|---|
| Borders | 1 | 1 |
| **Buckets** | **2** | **2** (`target_border_idx` 0 and 1) |
| BinarizedTargetMeanValue | 1 | 1 |
| Counter | 1 | 1 |

`AddCtrsToCandList` emits one `TSplitCandidate` per `(ctrIdx, targetBorderIdx, priorIdx)`
`[VERIFIED: WEB greedy_tensor_search.cpp:400-428]`. **Empirically confirmed:** a `Buckets:Prior=0.5`
model emits CTR descriptors with `target_border_idx` **0 AND 1** on the same projection, whereas
Borders/BTMV/Counter emit only `0` `[VERIFIED: EXPERIMENT probe5.py]`.

⇒ Honoring `Buckets` requires the CTR-candidate materialization to produce **two** columns per
projection per prior, and `CtrSplitSpec.target_border_idx` (currently pinned to the literal `0`
at `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:1943]` and passed as the literal `0` argument to
`greedy_tensor_search_oblivious_with_ctr` at `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:4656]`)
must become per-column.

### A.2 BinarizedTargetMeanValue (`= 2`)

Online entrypoint: `CalcOnlineCTRMean` `[VERIFIED: WEB online_ctr.cpp:437-501; dispatch at :756-767]`.

Per-bucket state: one `TCtrMeanHistory { float Sum; int Count; }`
`[VERIFIED: WEB catboost/libs/model/online_ctr.h:380-401]` — already mirrored in-repo at
`[VERIFIED: LOCAL crates/cb-train/src/ctr/online.rs:94-112]`.

Per document, in this order `[VERIFIED: WEB online_ctr.cpp:460-470]`:

```cpp
TCtrMeanHistory& elem = ctrArrMean[elemId];
sum[d]   = elem.Sum;      // READ (prefix)
count[d] = elem.Count;    // READ (prefix)
if (docOffset == 0) { elem.Add(static_cast<float>(permutedTargetClass[docId]) / targetBorderCount); }
```

then `[VERIFIED: WEB online_ctr.cpp:483-489]`:

```cpp
featureData[d] = CalcCTR(sum[d], count[d], priorX, shiftX, normX, ctrBorderCount);
```

- numerator = prefix `Sum` (a **float**, not an int)
- denominator = prefix `Count`
- value = `(Sum + prior) / (Count + 1)` then the shared `(ctr+shift)/norm*borderCount`
- the divisor passed as `targetBorderCount` is **`targetClassesCount - 1`**, supplied by the caller
  at `[VERIFIED: WEB online_ctr.cpp:762]` (`targetClassesCount - 1`), i.e. `1` for binclf — **not**
  `GetTargetBorderCount(...)`'s `1`-by-special-case, though for binclf both are 1.
  For binclf this makes the added value exactly `targetClass ∈ {0.0, 1.0}`.

Repo delta: `accumulate_online` already does `mean.add(class as f32 / divisor)`
`[VERIFIED: LOCAL crates/cb-train/src/ctr/online.rs:212-215]` — the accumulation is already right;
only the **prefix (read-before-increment) variant** and the per-object value path are missing.

### A.3 Counter (`= 4`) — NOT a prefix statistic

Online entrypoint: `CalcOnlineCTRCounter` `[VERIFIED: WEB online_ctr.cpp:503-562; dispatch at :784-796]`.

**Counter is NOT read-before-increment.** Its per-bucket totals are computed once, over the whole
dataset, before any per-document loop `[VERIFIED: WEB online_ctr.cpp:714-729]`:

```cpp
if (AnyOf(ctrInfo, [](auto& i){ return i.Type == ECtrType::Counter; })) {
    counterCTRTotal.resize(leafCount);
    int sampleCount = learnSampleCount;
    if (catFeatureParams.CounterCalcMethod == ECounterCalc::Full) {
        uniqValuesCounts.CounterCount = leafCount;
        sampleCount = hashArr.size();               // learn + ALL test sets
    }
    CountOnlineCTRTotal(hashArr, sampleCount, &counterCTRTotal);   // ++total[elemId]
    counterCTRDenominator = *MaxElement(counterCTRTotal.begin(), counterCTRTotal.end());
}
```

Then per document `[VERIFIED: WEB online_ctr.cpp:523-551]`:

```cpp
ctrTotal[d] = counterCTRTotal[elemId];                  // whole-set bucket count
featureData[d] = CalcCTR(ctrTotal[d], denominator, prior, shift, norm, borderCount);
                                        // denominator == counterCTRDenominator (a CONSTANT)
```

- numerator = the bucket's **whole-set** count (includes the document's own row)
- **ONLINE denominator = `counterCTRDenominator` = MAX bucket total** — the *same* rule as the
  final-CTR path `[VERIFIED: WEB online_ctr.cpp:934-936]`. **Online does NOT differ from final for
  Counter**, other than the `CounterCalcMethod` sample range.
- the shared `+1` still applies: `(count + prior) / (maxBucketTotal + 1)`
- Counter is permutation-independent: `IsPermutationDependentCtrType(Counter) == false`
  `[VERIFIED: WEB ctr_type.cpp:43-56]`. There is therefore **no leakage concern and no per-fold
  recomputation** — one Counter column serves every fold.
- Counter needs **no target classifier** (`NeedTarget(Counter) == false`,
  `NeedTargetClassifier(Counter) == false`) `[VERIFIED: WEB ctr_type.cpp:8-41]`.

### A.4 FloatTargetMeanValue (`= 3`) — **CONFIRMED final-CTR-only, and CPU-illegal**

The in-repo doc comment at `[VERIFIED: LOCAL crates/cb-train/src/ctr/final_ctr.rs:106-109]` claiming
"final-CTR path ONLY" is **CONFIRMED and is in fact stronger than stated**:

- It is absent from the online dispatch chain: the `else` branch asserts
  `Y_ASSERT(ctrType == ECtrType::Counter)` `[VERIFIED: WEB online_ctr.cpp:785]`.
- It is **rejected at options validation on CPU**:
  `IsSupportedCtrType(CPU, …)` returns true only for `{Borders, Buckets, BinarizedTargetMeanValue, Counter}`
  `[VERIFIED: WEB restrictions.h:18-32]`, enforced by
  `CB_ENSURE(IsSupportedCtrType(CPU, ctrType), "Ctr type … is not implemented on CPU yet")`
  `[VERIFIED: WEB catboost_options.cpp:504-509]`.
- Empirically: `simple_ctr=["FloatTargetMeanValue:Prior=0.5"]` raises
  `catboost/private/libs/options/catboost_options.cpp:509: Ctr type FloatTargetMeanValue is not
  implemented on CPU yet` `[VERIFIED: EXPERIMENT probe1.py]`.

It IS implemented in `CalcFinalCtrsImpl` (`elem.Add(targets[z])`, raw target)
`[VERIFIED: WEB online_ctr.cpp:923-925]` and in the apply path
`[VERIFIED: WEB static_ctr_provider.cpp:50-60]` — because **GPU-trained models** can carry it and
must be scorable on CPU.

**⇒ This changes Part 1's scope: `FloatTargetMeanValue` needs NO online arm.**

### A.5 FeatureFreq (`= 5`) — **also CPU-illegal, GPU-only**

`ctr_type.h` carries the upstream comment `// TODO(kirillovs): only for cuda models`
`[VERIFIED: WEB catboost/private/libs/ctr_description/ctr_type.h]`. `IsSupportedCtrType(CPU, FeatureFreq)
== false` `[VERIFIED: WEB restrictions.h:21-30]`; empirically rejected `[VERIFIED: EXPERIMENT probe1.py]`.
It is the GPU replacement for `Counter` (`CreateDefaultCounter` returns `FeatureFreq` on GPU)
`[VERIFIED: WEB catboost_options.cpp:393-415]`. **No online arm needed.**

The final-CTR `Counter` vs `FeatureFreq` denominator distinction the repo documents
`[VERIFIED: LOCAL crates/cb-train/src/ctr/final_ctr.rs:111-124]` is correct and matches
`[VERIFIED: WEB online_ctr.cpp:934-939]` (`Counter → *MaxElement`, `FeatureFreq → totalSampleCount`).
Keep it for the model-load/apply side.

### A.6 Summary table — what Part 1 must actually build

| ECtrType | CPU legal | Online arm needed | Prefix? | Numerator | Denominator | targetBorderCount (binclf) |
|---|---|---|---|---|---|---|
| Borders (0) | ✅ | already exists | yes | `N[1]` (binclf) / `Total − Σ_{c≤b}N[c]` | prefix `Total` | 1 |
| **Buckets (1)** | ✅ | **NEW** | yes | prefix `N[b]` | prefix `Total` | **2** |
| **BinarizedTargetMeanValue (2)** | ✅ | **NEW** | yes | prefix `Sum` (f32) | prefix `Count` | 1 |
| FloatTargetMeanValue (3) | ❌ CPU | **none** | — | — | — | — |
| **Counter (4)** | ✅ | **NEW** | **no** (whole-set) | whole-set bucket count | `MAX` bucket count (constant) | 1 |
| FeatureFreq (5) | ❌ CPU | **none** | — | — | — | — |

---

## B. `counter_calc_method` — exactly what it changes, and whether it is testable

`ECounterCalc { Full, SkipTest }`, default `SkipTest`
`[VERIFIED: WEB enums.h:248-251; cat_feature_options.cpp:234]`.

It has **exactly two effect sites**, both `Counter`-only:

1. **Online path** — the sample range for `CountOnlineCTRTotal`:
   learn-only (`SkipTest`) vs `hashArr.size()` = learn + all test sets (`Full`); also sets
   `uniqValuesCounts.CounterCount = leafCount` under `Full`
   `[VERIFIED: WEB online_ctr.cpp:716-729]`.
2. **Final-CTR bake** — `totalSampleCount += Data.GetTestSampleCount()` **only when**
   `ctrType == Counter && counterCalcMethod == Full`
   `[VERIFIED: WEB online_ctr.cpp:956-960]` (plus a matching RAM estimate at `:1013-1016`).

It does **not** affect `FeatureFreq`, `Borders`, `Buckets`, or `BinarizedTargetMeanValue`.

**Observability — measured, not inferred:**

| configuration | `Full` vs `SkipTest` |
|---|---|
| learn set only (no `eval_set`) | **predictions bit-identical**, `maxdiff = 0.000e+00` |
| with `eval_set` (40 rows) | **predictions differ**, `maxdiff = 4.010e-01` |

`[VERIFIED: EXPERIMENT probe6.py]`

And the baked `Counter` table under an eval set:

```
Full     → counts [18, 19, 18, 20, 22, 21], CounterDenominator = 22   (Σ non-sentinel = 100 = 60 learn + 40 test)
SkipTest → counts [ 8, 14,  8, 14, 13, 11], CounterDenominator = 14   (Σ non-sentinel =  60 = learn only)
```
`[VERIFIED: EXPERIMENT probe7.py]`

**⇒ `final_ctr.rs:70-73`'s claim is CORRECT: in a whole-learn-set build the flag changes nothing.**
**⇒ `counter_calc_method` can only be oracle-tested with a fixture that has an `eval_set` AND a
Counter CTR that actually wins a split.** The repo's `train_with_eval_sets` seam exists, but no CTR
fixture currently carries an eval set. This is a distinct, larger sub-scope.

**Planning recommendation:** thread `counter_calc_method` through (it is 3 lines), but scope its
*oracle* separately, or explicitly document it as "recorded and threaded, verifiable only with an
eval set — no fixture yet" rather than claiming it is parity-verified.

---

## C. simple vs combination CTR — the discriminator, confirmed with one caveat

```cpp
// catboost/private/libs/algo/ctr_helper.h:52-62
const TVector<TCtrInfo>& GetCtrInfo(const TProjection& projection) const {
    if (projection.IsSingleCatFeature()) {
        const int featureId = projection.CatFeatures[0];
        if (PerFeatureCtrs.contains(featureId)) return PerFeatureCtrs.at(featureId);
        return SimpleCtrs;
    }
    return TreeCtrs;   // == CatFeatureParams.CombinationCtrs
}
```
`[VERIFIED: WEB ctr_helper.h:52-62; population at ctr_helper.cpp:66-113]`

and the single consumer:

```cpp
// greedy_tensor_search.cpp:400-428  AddCtrsToCandList
const auto& ctrInfo = ctrsHelper.GetCtrInfo(proj);
```
`[VERIFIED: WEB greedy_tensor_search.cpp:400-428]`

**CONFIRMED:** single-feature projection → `simple_ctr`; multi-feature projection →
`combinations_ctr`. `CtrCandidate.is_simple`
`[VERIFIED: LOCAL crates/cb-train/src/candidates.rs:156,194]` derived from
`TProjection::is_simple()` `[VERIFIED: LOCAL crates/cb-train/src/projection.rs:144-146]` is the
correct discriminator **for the repo's current projection model**.

**Two caveats the planner must record:**

1. **`per_feature_ctr` overrides `simple_ctr` for a specific feature.** `PerFeatureCtrs` is checked
   *first* for a single-cat projection. The repo has no `per_feature_ctr` support at all, so this is
   a latent gap, not a Part-1 blocker — but the `is_simple → simple_ctr` mapping must be documented
   as "simple_ctr unless a per-feature override exists (unsupported here)".

2. **`IsSingleCatFeature()` is stricter than "one cat feature".**
   ```cpp
   // projection.h:102-104
   bool IsSingleCatFeature() const { return BinFeatures.empty() && OneHotFeatures.empty() && CatFeatures.ysize() == 1; }
   ```
   `[VERIFIED: WEB projection.h:102-104]`
   Upstream projections can carry **bin/one-hot members** (`AddTreeCtrs` seeds from
   `binAndOneHotFeaturesTree` `[VERIFIED: WEB greedy_tensor_search.cpp:506-511]`), so a
   `{1 cat feature + 1 float bin}` projection routes to **combinations_ctr**. The repo's
   `TProjection` holds **only** `cat_features: Vec<usize>`
   `[VERIFIED: LOCAL crates/cb-train/src/projection.rs:105]`, so `is_simple() == cat.len() == 1`
   is currently *exactly* equivalent. **If a later phase adds bin/one-hot projection members,
   `is_simple` must be widened to `cat.len()==1 && bin.is_empty() && one_hot.is_empty()`** — add a
   doc comment now so the equivalence is not silently broken.

---

## D. Defaults — the repo's `*_default()` functions do NOT match upstream

### D.1 What catboost 1.2.10 actually defaults to (CPU)

```cpp
// catboost_options.cpp:429-472  SetCtrDefaults()
switch (lossFunction) {
  case PairLogit: case PairLogitPairwise:
    defaultSimpleCtrs = { Counter(prior 0/1) };
    defaultTreeCtrs   = { Counter(prior 0/1) };
    break;
  default:
    defaultSimpleCtrs = { Borders(priors 0/1, 0.5/1, 1/1), Counter(prior 0/1) };
    defaultTreeCtrs   = { Borders(priors 0/1, 0.5/1, 1/1), Counter(prior 0/1) };
}
```
`[VERIFIED: WEB catboost_options.cpp:439-453; GetDefaultPriors at cat_feature_options.cpp:118-138]`

Measured on a real fit `[VERIFIED: EXPERIMENT probe2.py `get_all_params()`]`:

```
simple_ctr       = ['Borders:CtrBorderCount=15:CtrBorderType=Uniform:TargetBorderCount=1:
                    TargetBorderType=MinEntropy:Prior=0/1:Prior=0.5/1:Prior=1/1',
                    'Counter:CtrBorderCount=15:CtrBorderType=Uniform:Prior=0/1']
combinations_ctr = <identical>
counter_calc_method = SkipTest
ctr_target_border_count = 1
```

Other verified defaults:
- `max_ctr_complexity` (`MaxTensorComplexity`) = **4** `[VERIFIED: WEB cat_feature_options.cpp:231]`
- `one_hot_max_size` = **2** `[VERIFIED: WEB cat_feature_options.cpp:232]`
- CTR binarization = `Uniform, 15 borders` `[VERIFIED: WEB cat_feature_options.cpp:169]`
- target binarization = `MinEntropy, border_count 1` `[VERIFIED: WEB cat_feature_options.cpp:230]`
- `counter_calc_method` = `SkipTest` `[VERIFIED: WEB cat_feature_options.cpp:234]`
- CPU forbids a non-unit `PriorDenom`: `CB_ENSURE(denom == 1.0, "Error: CPU could use only 1 as
  denom for ctrs currently")` `[VERIFIED: WEB ctr_helper.cpp:50]` — **this vindicates the repo's
  `prior_denom: 1.0` pin** `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:1942]`.
- CPU forbids non-Uniform CTR binarization: `CB_ENSURE(BorderSelectionType == Uniform, "Error:
  custom ctr binarization is not supported on CPU yet")` `[VERIFIED: WEB catboost_options.cpp:517-520]`.

### D.2 What the repo defaults to

| repo function | value | upstream CPU default | verdict |
|---|---|---|---|
| `simple_ctr_default()` `[LOCAL boosting.rs:464]` | `ECtrType::Borders` | list `[Borders, Counter]` | **mismatch (model, not value)** |
| `simple_ctr_priors_default()` `[LOCAL boosting.rs:473]` | `vec![0.5]` | `[0.0, 0.5, 1.0]` (Borders) | **mismatch** |
| `combinations_ctr_default()` `[LOCAL boosting.rs:509]` | `ECtrType::Borders` | list `[Borders, Counter]` | **mismatch (model)** |
| `combinations_ctr_priors_default()` `[LOCAL boosting.rs:518]` | `vec![0.5]` | `[0.0, 0.5, 1.0]` | **mismatch** |
| `counter_calc_method_default()` `[LOCAL boosting.rs:480]` | `SkipTest` | `SkipTest` | ✅ match |
| `max_ctr_complexity_default()` `[LOCAL boosting.rs:496-499]` | `4` (via `projection::max_ctr_complexity_default`) | `4` | ✅ match |
| `ctr_border_count_default()` `[LOCAL boosting.rs:529]` | `15` | `15` | ✅ match (closes PLAN-CHECK "Unverified Item") |
| `one_hot_max_size_default()` `[LOCAL candidates.rs:78-80]` | `2` | `2` | ✅ match |

### D.3 Is the mismatch a parity bug? — **Nuanced. It is a deliberate, documented simplification
whose blast radius makes "fixing it" a phase of its own.**

Every existing CTR fixture **explicitly overrides** `simple_ctr` / `combinations_ctr` to a single
`Borders:Prior=0.5` description:
`plain_ctr` `[VERIFIED: LOCAL crates/cb-oracle/fixtures/plain_ctr/config.json]`,
`ordered_ctr`, `tensor_ctr`, `tensor_ctr_e2e`, `fstr_ctr` — all identical in this respect.
So the repo is parity-correct **for every configuration it has ever been tested at**, and
parity-*wrong* for the **default** configuration (which is the one the Part-2 facade will expose to
users who set nothing).

**The blast radius of changing the field TYPE is the decisive planning constraint:**
`grep -rn "simple_ctr:" crates/ --include=*.rs` → **65 occurrences across 62 files**, because the
project's Pitfall-6 discipline pins every `BoostParams` field explicitly at every construction site
`[VERIFIED: LOCAL command output, §11]`. Changing `simple_ctr: ECtrType` to
`simple_ctr: Vec<CtrDescription>` forces 62 file edits with no behavioral benefit to any of them.

**Recommendation (see §6):** keep the scalar `ECtrType` field shape in Part 1. Honor the *type* and
the *full prior list*. Defer the multi-description list (`[Borders, Counter]` simultaneously) to a
follow-on phase, and record the default mismatch explicitly as a known parity gap.

---

## E. ORACLE STRATEGY — the most important section

### E.1 What the existing CTR fixtures pin

| fixture | producer | what it pins | CTR types exercised |
|---|---|---|---|
| `plain_ctr/` | `generator/ordered_oracle.cpp` (transcription harness) | per-object `ctr_good_count`/`ctr_total_count`/`ctr_value` + permutation | Borders only, `Prior=0.5`, `combinations_ctr: []` |
| `ordered_ctr/` | same | fold-0/1 permutations + ordered per-object CTR | Borders only, `Prior=0.5` |
| `tensor_ctr/` | same | per-object CTR over a 2-feature combination projection | Borders only, `Prior=0.5`, `max_ctr_complexity=2` |
| `tensor_ctr_e2e/` | `generator/gen_fixtures.py::gen_tensor_ctr_e2e` (real catboost) | `X_cat`, `y`, `model.json` **with baked ctr_data**, `predictions.npy` | Borders only, `Prior=0.5` |
| `fstr_ctr/` | fixture-local `fixtures/fstr_ctr/gen_fixtures.py` (real catboost) | mixed float+cat model, interaction / PVC | Borders only, `Prior=0.5` |
| `ctr_load/` | fixture-local `fixtures/ctr_load/gen_fixtures.py` (real catboost) | **default-params** upstream models, `.cbm` + `.json` | **`ctr_types_in_splits: ["Borders","Counter"]`** ← the only Counter evidence in the corpus |
| `multi_permutation_fold/` (under `crates/cb-train/tests/fixtures/`) | `generator/gen_multi_permutation_fold.py` | `predictions_pc{1,2,4}.npy` on the tensor_ctr_e2e corpus | Borders |

`[VERIFIED: LOCAL crates/cb-oracle/fixtures/*/config.json]`
`[VERIFIED: LOCAL crates/cb-oracle/generator/ordered_oracle.cpp:1-40]`

**Key gap:** no fixture exercises `Buckets`, `BinarizedTargetMeanValue`, or a Counter that *trains*
(the `ctr_load` Counter tables are only *loaded*). No fixture uses a non-0.5 prior, a multi-prior
list, or `target_border_idx != 0`.

### E.2 The `gen_fixtures.py` fallthrough claim — **VERIFIED TRUE**

```python
if   "--wave1-only"        in sys.argv: gen_wave1_only()
elif "--wave2-only"        in sys.argv: gen_wave2_only()
elif "--wave3-only"        in sys.argv: gen_wave3_only()
elif "--multiclass-only"   in sys.argv: gen_multiclass_only()
elif "--multilabel-only"   in sys.argv: gen_multilabel_only()
elif "--mvs-seeds-only"    in sys.argv: gen_mvs_seeds_only()
elif "--bootstrap-dev-only" in sys.argv: gen_bootstrap_dev_only()
elif "--one-hot-only"      in sys.argv: gen_one_hot_only()
elif "--multiquantile-only" in sys.argv: gen_multiquantile_only()
else: main()          # <-- regenerates the ENTIRE frozen corpus
```
`[VERIFIED: LOCAL crates/cb-oracle/generator/gen_fixtures.py:3861 and the __main__ block above it]`

There is **no positional-scenario dispatch**. `gen_fixtures.py ctr_types` → `main()` → everything.
PLAN-CHECK's BLOCKER-1 is **confirmed correct**.

**Two safe patterns already exist in-repo:**

- **Pattern A (in-file guarded flag)** — `gen_one_hot_only()`
  `[VERIFIED: LOCAL crates/cb-oracle/generator/gen_fixtures.py:1398-1424]` snapshots
  `_one_hot_fixture_dirty_paths()` before and after, and **hard-aborts (`sys.exit(1)`) if any path
  outside `one_hot_train/` was dirtied**, printing the offenders. This is the reference
  implementation for a scenario-scoped generation and should be copied verbatim in shape.
- **Pattern B (fixture-local generator, preferred)** — `fixtures/<name>/gen_fixtures.py` as used by
  `ctr_load`, `fstr_ctr`, `partial_dependence`, `coreml_export`, `model_sum`, `staged_predict`,
  `advanced_fstr`, `feature_selection`, `fstr_loss_change`
  `[VERIFIED: LOCAL find crates/cb-oracle/fixtures -name gen_fixtures.py]`. This is
  **structurally incapable** of touching the frozen corpus and is the newer convention.

**Recommendation: use Pattern B** (`crates/cb-oracle/fixtures/ctr_types/gen_fixtures.py`) for the new
family, and do **not** touch `generator/gen_fixtures.py` at all. This dissolves BLOCKER-1 entirely
rather than mitigating it.

### E.3 The nondeterminism constraint

Project memory (`ctr-model-loading`) records that catboost quantization is run-to-run
nondeterministic, which is why CTR fixtures are frozen. This applies to **float feature borders**.
The proposed `ctr_types/` family should be **categorical-only** (no float columns) — as
`tensor_ctr_e2e` already is `[VERIFIED: LOCAL fixtures/tensor_ctr_e2e/config.json: n_cat_features=2,
no X_float]` — which removes float quantization from the picture entirely. Even so: **generate once,
commit, never regenerate in CI** (SPEC-CATF-17 discipline).

### E.4 Recommended oracle ladder for Part 1

Three rungs, in dependency order. Each new arm should clear rung 1 before rung 2 is attempted.

**Rung 1 — per-object online-CTR ground truth (localisation).**
Extend `crates/cb-oracle/generator/ordered_oracle.cpp` with the three new arms, transcribing
`CalcOnlineCTRClasses` (Buckets), `CalcOnlineCTRMean` (BTMV) and `CalcOnlineCTRCounter` (Counter)
from the v1.2.10 sources cited in §A. This harness already transcribes `CalcCTR`,
`CalcNormalization` and the read-before-increment prefix `[VERIFIED: LOCAL
crates/cb-oracle/generator/ordered_oracle.cpp:20-32]`, so the three arms are a ~60-line addition and
reuse the existing self-oracle anchors. Emits `ctr_good_count`/`ctr_total_count`/`ctr_value` `.npy`
in the same D-02 schema as `plain_ctr/`.
*Strength:* pinpoints a wrong numerator/denominator to one document.
*Weakness:* it is a transcription, not an independent implementation — a shared misreading of the
C++ would pass. **Mitigate by cross-checking rung 1 against rung 2.**

**Rung 2 — end-to-end train→predict against real catboost (authoritative).**
`catboost==1.2.10` is installed and **all four CPU-legal types train cleanly**
`[VERIFIED: EXPERIMENT probe1.py]`. Generate, per type, the `tensor_ctr_e2e` artifact set
(`X_cat.npy`, `y.npy`, `model.json` with baked `ctr_data`, `predictions.npy`) with the isolating
config (`boosting_type=Plain`, `one_hot_max_size=1`, `permutation_count=1`,
`fold_len_multiplier=2.0`, `bootstrap_type=No`, `random_strength=0`, `random_seed=0`,
`thread_count=1`, `boost_from_average=False`, `verbose=False`, `leaf_estimation_method=Gradient`,
`leaf_estimation_iterations=1`). Suggested scenarios:

| scenario | `simple_ctr` / `combinations_ctr` | `max_ctr_complexity` | purpose |
|---|---|---|---|
| `buckets_simple` | `["Buckets:Prior=0.5"]` / `[]` | 1 | Buckets numerator + **both** `target_border_idx` |
| `btmv_simple` | `["BinarizedTargetMeanValue:Prior=0.5"]` / `[]` | 1 | f32 mean prefix |
| `counter_simple` | `["Counter:Prior=0.5"]` / `[]` | 1 | whole-set count + MAX denominator |
| `borders_multiprior` | `["Borders:Prior=0:Prior=0.5:Prior=1"]` / `[]` | 1 | **multi-prior candidate expansion** (the highest-risk sub-scope, §6) |
| `mixed_simple_vs_combo` | `["Buckets:Prior=0.5"]` / `["Counter:Prior=0.5"]` | 2 | proves the `is_simple` routing (item C) |
| `counter_full_eval` | `["Counter:Prior=0.5"]`, `counter_calc_method="Full"`, **with `eval_set`** | 1 | the only way to test item B |

**Anti-false-pass guard (mandatory):** each scenario's generator must assert that
`model.json → features_info.ctrs` contains **≥1 descriptor of the intended `ctr_type`**, and for
`buckets_simple` that both `target_border_idx: 0` and `1` appear. Without this, "the model trained"
is satisfiable by a config that produced zero CTR splits (both sides then agree trivially).
The probe already demonstrates exactly this JSON is available `[VERIFIED: EXPERIMENT probe5.py]`.

**Rung 3 — `.cbm`/`.json` round-trip.** For each new type, save the upstream model as `.cbm` and
assert the repo's loader reproduces the tables. **Blocked for BTMV — see §F.4.**

### E.5 Non-obvious oracle traps

- **Pin `random_strength=0` explicitly.** The raw dict API's default differs from the builder's
  (project memory `cv-orch01-random-strength-fixture`). All existing CTR fixture configs already do
  `[VERIFIED: LOCAL fixtures/*/config.json]`.
- **`combinations_ctr: []`** is used by `plain_ctr/config.json` to disable combination CTRs
  `[VERIFIED: LOCAL fixtures/plain_ctr/config.json]`. Reuse it for the simple-only scenarios.
- **`ctr_target_border_count` default is 1** ⇒ `targetClassesCount == 2` for binclf
  `[VERIFIED: EXPERIMENT probe2.py]`. Do not vary it in Part 1; it changes `targetBorderCount`
  for *every* type and multiplies the candidate set.
- **Uniform CTR binarization is mandatory on CPU** `[VERIFIED: WEB catboost_options.cpp:517-520]`,
  so `ctr_border_count=15` + Uniform is the only legal CTR binarization — no fixture variation needed.

---

## F. Impact scope for Part 1

### F.1 Blast radius (CodeGraph)

`[VERIFIED: CODEGRAPH codegraph_explore "calc_ctr_online calc_ctr_online_bin accumulate_online build_final_ctr ctr_splits_for_tree"]`

| symbol | callers | covering tests |
|---|---|---|
| `build_final_ctr` (`ctr/final_ctr.rs:75`) | 14 in `ctr/bake.rs`, `ctr/mod.rs` | `cb-model/tests/ctr_data_roundtrip_test.rs`, `ctr/final_ctr_test.rs` |
| `accumulate_online` (`ctr/online.rs:163`) | 14 in `ctr/bake.rs`, `ctr/mod.rs` | `ctr_data_roundtrip_test.rs`, `final_ctr_test.rs`, `ctr/online_test.rs` |
| `calc_ctr_online` (`ctr/calc_ctr.rs:76`) | 10 in `calc_ctr.rs`, `online.rs`, `mod.rs` | `calc_ctr_test.rs`, `ordered_ctr_oracle_test.rs`, `plain_ctr_oracle_test.rs`, `tensor_ctr_oracle_test.rs` |
| `calc_ctr_online_bin` (`ctr/calc_ctr.rs:89`) | 7 in `ctr/ctr_feature.rs`, `mod.rs` | `calc_ctr_test.rs`, `multi_permutation_fold_oracle_test.rs` |
| `ctr_splits_for_tree` (`boosting.rs:1929`) | 1 in `boosting.rs:5318` | ⚠️ **no covering tests** |

### F.2 MUST CHANGE

| file:symbol | change | downstream |
|---|---|---|
| `crates/cb-train/src/ctr/mod.rs::ECtrType` | add helpers: `target_border_count(classes)`, `is_permutation_dependent()`, `needs_target_classifier()`, `is_cpu_supported()` (mirroring `ctr_helper.h:34-42` and `ctr_type.cpp`) | new API, additive |
| `crates/cb-train/src/ctr/online.rs` | add prefix producers for Buckets (generic classes loop, `UpdateGoodCount` semantics), BTMV (`(Sum, Count)` prefix), and a whole-set Counter total + MAX denominator. Keep `online_ctr_prefix_binclf` as the Borders binclf fast path (it is provably the `b=0`, `classes=2` case). | `bake.rs`, `ctr_feature.rs` |
| `crates/cb-train/src/ctr/ctr_feature.rs:213,232` | `materialize_ctr_feature` must take `(ctr_type, target_border_idx, prior)` and dispatch to the right prefix producer; **stop hard-coding `ECtrType::Borders.as_i8()`**; add `target_border_idx` to `CtrFeatureColumn` | `boosting.rs:3238,3274`; `tree.rs:3291` copies `column.ctr_type` already |
| `crates/cb-train/src/boosting.rs:3155` | pick `simple_ctr_priors` vs `combinations_ctr_priors` per candidate's `is_simple`; **and emit one column per prior** (currently `.first()`) | scoring candidate set grows |
| `crates/cb-train/src/boosting.rs:3238,3274` | loop over `(ctr_type, target_border_idx, prior)` instead of one column per projection | `structure_fold_columns`, `averaging_ctr_features` shape |
| `crates/cb-train/src/boosting.rs:4656` | the literal `0` `target_border_idx` argument must come from the winning column | `tree.rs::greedy_tensor_search_oblivious_with_ctr` signature |
| `crates/cb-train/src/boosting.rs:1929-1949` | `ctr_splits_for_tree` — pick type/prior by `is_simple`; **note this is the no-CTR fallback only** (`has_ctr` false ⇒ empty candidates), so it is cosmetic but must not lie | none (dead path) |
| `crates/cb-train/src/ctr/bake.rs:192,232,236` | `bake_ctr_table` must take `ctr_type`; call `build_final_ctr(&acc, ctr_type)`; reshape per type (Counter/FeatureFreq = 1 value/bucket, mean types = `(Sum,Count)`), and set `counter_denominator` (currently hard-coded `0`) | `BakedCtrTable` gains `mean_sum`/`mean_count`; `boosting.rs:5445` |
| `crates/cb-train/src/ctr/final_ctr.rs:70-75` | thread `counter_calc_skip_test` from `params.counter_calc_method` (record-only for a learn-only build — see §B) | none behaviourally |
| `crates/cb-model/src/ctr_data.rs:313-331` `CtrData::from_baked` | **`mean: Vec::new()` is hard-coded** — must carry mean tables from `BakedCtrTable` | apply path for BTMV |
| `crates/cb-train/src/tree.rs:3237,3291-3300` | `target_border_idx` becomes per-column, not a whole-tree parameter | `boosting.rs:4656` |

### F.3 MAY CHANGE

- `crates/cb-train/src/candidates.rs` — if `tensor_ctr_candidates` is asked to expand
  `(type, borderIdx, prior)` rather than leaving that to `boosting.rs`. Prefer **not** to; keep
  `CtrCandidate` = projection + `is_simple` and expand at materialization.
- `crates/cb-train/src/projection.rs:144` — add the doc note from §C caveat 2.
- `crates/cb-model/src/ctr_data.rs::CtrValueTable::numerator_denominator:226-258` — **Borders
  ignores `target_border_idx`** and always uses `(N[1], N[0]+N[1])`
  `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs:228-242]`. Upstream's Borders arm has a
  `targetClassesCount > 2` branch computing `good = Σ_{c > b} N[c]`, `total = Σ_all`
  `[VERIFIED: WEB static_ctr_provider.cpp:90-110]`. Correct for binclf, **wrong for multiclass** —
  out of Part 1's binclf scope but must be recorded as a known limitation, not silently left.
- `crates/catboost-rs/src/builder.rs` — Part 2 setters (see §G).

### F.4 VERIFICATION ONLY / BLOCKING CONSTRAINTS discovered

- **`.cbm` save AND load REJECT mean-type CTRs.**
  `decode_ctr_model_parts` returns `ModelError::Deserialize("mean/target-mean CTR unsupported
  (v1, MAJOR-2)")` when `ctr_type.is_mean()` `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs:584-588]`,
  and the encoder documents "Mean tables and marker-valued hashes are rejected (v1)"
  `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs:756-758]`.
  ⇒ **If `BinarizedTargetMeanValue` becomes trainable, `save_model(.cbm)` on the resulting model
  will fail.** The planner must choose: (a) lift the v1 mean restriction (adds a `TCtrMeanHistory`
  blob codec — the FlatBuffers `CTRBlob` is a raw byte array, so it is `f32 Sum` + `i32 Count` pairs),
  or (b) keep BTMV out of Part 1, or (c) ship BTMV training with a documented, tested `.cbm`
  rejection. **`model.json` already supports mean tables** (`to_json`/`from_json` handle
  `is_mean()` with stride 3) `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs:366-396,428-475]`,
  so a JSON-only oracle for BTMV is viable.
- **`ctr_splits_for_tree` has no covering test** `[VERIFIED: CODEGRAPH]` — any change there is
  unguarded. Add one.
- **62 files pin `simple_ctr:` in a `BoostParams` literal** `[VERIFIED: LOCAL grep]`. Adding a
  *field* is a 62-file edit; changing a field's *type* is a 62-file edit. **Adding no field and
  changing no type is free.** Design accordingly.
- Existing CTR oracles that must stay green:
  `crates/cb-train/tests/{plain_ctr,ordered_ctr,tensor_ctr,tensor_ctr_e2e,s_order_ctr_bins,
  ctr_split_scoring,ctr_feature_materialize,multi_permutation_e2e}*.rs`,
  `crates/cb-model/tests/{ctr_data_roundtrip,fstr_ctr_oracle}_test.rs`,
  `crates/cb-train/src/ctr/{online_test,calc_ctr_test,final_ctr_test}.rs`
  `[VERIFIED: LOCAL ls]`. All pin Borders + `Prior=0.5`, so a correctly-defaulted change is a no-op
  for them — **that is the regression gate**.
- **Uncommitted one-hot device wave (T23–T29b) is live in the working tree** and touches
  `boosting.rs`, `tree.rs`, `cb-model/src/{apply,model,cbm,json}.rs`, `cb-backend/*`
  `[VERIFIED: LOCAL git status]`. `ModelSplit::OneHot`, `ObliviousTree.one_hot_splits` and
  `level_kinds` now exist and the device grower emits them `[VERIFIED: CODEGRAPH ModelSplit,
  ObliviousTree, passes_one_hot_split; LOCAL boosting.rs:3908-4077]`. **Do not revert.** Part 1
  edits `boosting.rs` in the same region — plan for merge friction and re-run
  `crates/cb-train/tests/{one_hot_oracle,one_hot_draw_accounting,device_one_hot_parity}_test.rs`.
- **The stale doc comment at `crates/cb-train/src/boosting.rs:815-816`** ("EMPTY for every path
  that emits no one-hot candidate — which is all of them until T19 populates it") is now false;
  fix opportunistically.
- **`catboost-master/` is a 3-file stub**, and its `greedy_tensor_search.cpp` is **1997 lines vs
  v1.2.10's 1955** — i.e. it is a *different* (later) revision than the pinned parity target
  `[VERIFIED: LOCAL wc -l + diff]`. **Do not use it as the parity reference.** CLAUDE.md's
  description of a full vendored tree is stale. All upstream citations in this document come from
  the live `v1.2.10` tag.

---

## G. Part 2 — triage of the 18 existing tasks

Source: `.planning/plans/catboost-builder-cat-features-routing/{SPEC.md,PLAN.md,PLAN-CHECK.md}`
`[VERIFIED: LOCAL, all three read]`. PLAN-CHECK verdict was **ISSUES_FOUND** with 2 BLOCKERs,
2 CRITICALs, 4 MAJORs, 4 MINORs.

### G.1 Verification of PLAN-CHECK's own claims (re-derived this session)

| claim | status |
|---|---|
| BLOCKER-1: `gen_fixtures.py` has no positional dispatch; unknown argv → `main()` | ✅ **CONFIRMED** `[LOCAL gen_fixtures.py __main__]` |
| BLOCKER-2: 4 params never read by `train_inner` | ✅ **CONFIRMED** `[LOCAL grep — only `combinations_ctr_priors` and `max_ctr_complexity`/`one_hot_max_size` are read]` |
| CRITICAL-4: one-hot columns silently dropped; `grow_one_hot_tree` has no production caller | ❌ **NOW OBSOLETE** — `ModelSplit::OneHot` exists, `ObliviousTree.one_hot_splits` + `level_kinds` exist, `passes_one_hot_split` exists, and the device grower emits one-hot levels `[CODEGRAPH + LOCAL boosting.rs:3908-4077]` |
| MAJOR-5: 4 un-listed `data_to_pool` sites | ✅ **CONFIRMED, exactly 14 sites total.** The 4 missing are `search.rs:304`, `search.rs:389`, `cv.rs:61`, `cv.rs:64` `[VERIFIED: LOCAL grep -rn data_to_pool crates/catboost-rs-py/src/]`. Full list: `classifier.rs:68,100,126,280`; `regressor.rs:68,149,254`; `ranker.rs:65,100`; `estimator.rs:270`; `search.rs:304,389`; `cv.rs:61,64` |
| "Unverified: `ctr_border_count_default()` = 15" | ✅ **NOW VERIFIED** `[WEB cat_feature_options.cpp:169]` |
| "Unverified: upstream domains for `one_hot_max_size` / `max_ctr_complexity`" | ✅ **NOW VERIFIED**: `one_hot_max_size` default 2, upper bound `OneHotMaxSizeLimit = GetMaxBinCount()` on CPU `[WEB cat_feature_options.cpp:232-233,267-268]`; `max_ctr_complexity` default 4, `CB_ENSURE(MaxTensorComplexity < GetMaxTreeDepth())` `[WEB cat_feature_options.cpp:231,269-271]` |

### G.2 Task-by-task triage

| Task | Verdict | Reason |
|---|---|---|
| T01 `one_hot_max_size` setter | **SURVIVES (upgraded)** | CRITICAL-4 dissolved: one-hot columns now genuinely train, so the setter is no longer semantically hollow. Drop the "hollow" caveat; **add** a regression test with a cardinality-2 column + default `one_hot_max_size=2` proving the column now contributes |
| T02 `max_ctr_complexity` setter | **SURVIVES unchanged** | already live in the engine `[LOCAL boosting.rs:3061]` |
| T03 `simple_ctr` + priors setters | **REVISE** | after Part 1 the type IS honored. Delete the BLOCKER-2 "not consumed" doc-comment requirement; **add** a note that only ONE description (not upstream's list) is representable, and that `Prior=<n>/<d>` with `d != 1` is illegal on CPU anyway `[WEB ctr_helper.cpp:50]` |
| T04 `combinations_ctr` + priors | **REVISE** | same as T03 |
| T05 `counter_calc_method` setter | **REVISE** | Part 1 threads it, but it is **only observable with an eval set** (§B). Doc-comment must say so; do not claim parity coverage |
| T06 default-equivalence guard | **SURVIVES + MINOR-9 fix** | add the second mutation on a `boost_params()` **read** |
| T07 re-export `ECtrType`/`CounterCalcMethod` | **SURVIVES unchanged** | |
| T08 `fit()` → `train_cat` | **SURVIVES + strengthen** | keep the `ctr_data.is_some()` + `≥1 ModelSplit::Ctr` guard; **add** a `ModelSplit::OneHot` assertion for the one-hot arm |
| T09 `cat_columns()` width validation | **REVISE (CRITICAL-3 still open)** | one-hot no longer being dropped narrows failure scenario #1 but does **not** close it: `max(projection member)+1` is still a lower bound whenever a trailing cat column is one-hot-routed or unused by any winning split. Prefer PLAN-CHECK option 1: store an explicit trained cat width on `cb_model::Model`. Note `ObliviousTree.one_hot_splits` now gives a second source of cat indices — the derived bound must union CTR projections **and** one-hot split `cat_feature`s |
| T10 `predict_with` CTR-aware | **SURVIVES + widen** | must ALSO route one-hot-only models (no `ctr_data`, but `ModelSplit::OneHot` present) to `predict_raw_cat`. `predict_raw = predict_raw_cat(.., &[])` `[LOCAL apply.rs]` ⇒ a one-hot model scored via `predict_raw` reads `cat_values.get(i) → None → false` `[LOCAL apply.rs:205-210]`, i.e. **every one-hot split fails** — a NEW silent-wrongness class the original plan did not know about. `is_ctr_model()` must become `needs_cat_columns()` |
| T11 zero-cat-column guard | **SURVIVES + widen** | same predicate change as T10 |
| T12 Python `one_hot_max_size` / `max_ctr_complexity` / `counter_calc_method` | **REVISE** | ranges now verified (§G.1). `counter_calc_method` honesty note per T05 |
| T13 Python `simple_ctr` / `combinations_ctr` grammar | **REVISE, materially** | BLOCKER-2 dissolves *partially*. After Part 1 the accepted domain is **`{Borders, Buckets, BinarizedTargetMeanValue, Counter}`** — `FloatTargetMeanValue`/`FeatureFreq` must be **rejected**, matching upstream's own CPU rejection message `[WEB catboost_options.cpp:504-509]`. The single-description limit and `Prior=n/d` (d≠1) rejection remain. `combinations_ctr=[]` needs a defined meaning (PLAN-CHECK "Potential Bugs") |
| T14 `cat_features` fit-kwarg | **REVISE (MAJOR-5)** | add `search.rs` + `cv.rs` to the file list; **14** call sites, not 10; MINOR-10 (`list[int]`) and MINOR-11 (de-dup) stand |
| T15 registry truthfulness | **SURVIVES unchanged** | |
| T16 fixture generation + freeze | **OBSOLETE AS WRITTEN → REPLACE** | do **not** add a flag to `generator/gen_fixtures.py`. Use the fixture-local generator pattern (`fixtures/<name>/gen_fixtures.py`, §E.2 Pattern B), which cannot dirty the corpus. Add a `git status --short crates/cb-oracle/fixtures` guard anyway |
| T17 Python-API categorical oracle ≤1e-5 | **SURVIVES + cite** | ladder step 1 should cite `crates/cb-train/tests/multi_permutation_e2e_oracle_test.rs` (already proves `permutation_count=4` + `max_ctr_complexity=2` at ≤1e-5) instead of proposing a scratch run |
| T18 float-only no-regression gate | **SURVIVES unchanged** | |

### G.3 New tasks Part 2 needs that the 18 do not contain

- **T-new-A (from MAJOR-6):** `cv()` / `grid_search()` on a categorical pool. `select_rows`
  preserves `cat_features` `[LOCAL cb-data/src/pool.rs:232-236]` and `staged_predict` →
  `ensure_scalar_oblivious` rejects `ctr_data.is_some()` `[LOCAL catboost-rs/src/model.rs:164-169]`.
  Decide: fail fast, or convert to `error_score`. Add a test.
- **T-new-B (from MAJOR-7):** `shap_values`, `partial_dependence`, `feature_importance_with_data`
  are silently wrong on CTR **and now also on one-hot** models. Add typed `UnsupportedModel`
  rejections (cheap, additive, precedent = `staged_predict`).
- **T-new-C (from MAJOR-8):** fix `cargo test -p X --test <target>::<filter>` →
  `cargo test -p X --test <target> -- <filter>` in three commands.
- **T-new-D:** the one-hot arm of `fit()` needs its own oracle (the frozen `one_hot_train/` family
  exists `[LOCAL fixtures/one_hot_train/]` and can be driven through the facade).

---

## H. Recommended approach for Part 1 (prescriptive)

**Design decision 1 — keep the scalar `BoostParams` field shape.**
Do not turn `simple_ctr: ECtrType` into a list. Rationale: 62 construction sites
`[VERIFIED: LOCAL grep]`, zero behavioral benefit to any of them, and every existing CTR fixture
pins a single description anyway. Record the "upstream default is a 2-description list" mismatch as
an explicit, documented parity gap with the `catboost_options.cpp:439-453` anchor.

**Design decision 2 — honor the FULL prior list, not `.first()`.**
This is the one place where the repo's data model is already right (`Vec<f64>`) and the code is
wrong. Upstream emits one candidate column per `(targetBorderIdx, priorIdx)`
`[VERIFIED: WEB greedy_tensor_search.cpp:414-427]`. This is the **highest-risk sub-scope** because
it changes the candidate set the greedy search scores, which changes tie-breaks. Gate it behind its
own `borders_multiprior` fixture (§E.4) **before** touching the type dispatch.

**Design decision 3 — no new value/quantizer function.**
`calc_ctr_online` / `calc_ctr_online_bin` / `calc_normalization` are already type-agnostic and
correct (§A.0). Add only per-type `(numerator, denominator)` producers in `online.rs`.

**Design decision 4 — implement the generic classes loop, then prove Borders-binclf is its special
case.** `CalcOnlineCTRClasses` with `UpdateGoodCount` serves BOTH `Buckets` and multiclass
`Borders`, and at `classes=2, b=0` reduces to the existing `online_ctr_prefix_binclf`
`[VERIFIED: WEB online_ctr.cpp:115-121,168-185,300-307]`. Write an equivalence test asserting
`generic(classes=2, b=0, Borders) == online_ctr_prefix_binclf` bit-for-bit — that single test
protects every existing CTR oracle from the refactor.

**Suggested wave ordering:**

- **W0** — `ECtrType` helpers (`target_border_count`, `is_cpu_supported`, …) + a CPU-legality guard
  in `BoostParams` validation that rejects `FloatTargetMeanValue`/`FeatureFreq` with upstream's own
  message. Zero behavior change; unblocks everything.
- **W1** — generic classes prefix producer + the Borders-binclf equivalence test. **No behavior
  change** (still only Borders, still `b=0`).
- **W2** — thread `ctr_type` + `is_simple` prior selection (still one prior). `simple_ctr` and
  `combinations_ctr` become real. Fixtures: `counter_simple`, `buckets_simple`, `btmv_simple`.
- **W3** — multi-prior + multi-`target_border_idx` candidate expansion. Fixture:
  `borders_multiprior`, and the `buckets_simple` `target_border_idx=1` assertion.
- **W4** — `bake.rs` per-type final tables + `CtrData::from_baked` mean threading + the `.cbm` mean
  decision (§F.4).
- **W5** — `counter_calc_method` threading + the eval-set fixture (or an explicit deferral).

**Backend trap to respect:** `--features rocm` alone does not switch backends (`cpu` is a default
feature that wins the cfg chain); use `--no-default-features --features rocm`.
**Test baseline:** `.planning/plans/one-hot-categorical-training/baseline/` — 297 passed / 59 failed,
**all 59 in the `cb_backend` lib target** under the CubeCL cpu backend; the accepted gate is "no
target that passes here may fail later" `[VERIFIED: LOCAL baseline/README.md]`.
`exact_quantile_weighted_matches_cpu` is flaky.

---

## I. Risks

| # | Risk | Trigger | Consequence | Prevention |
|---|---|---|---|---|
| R1 | Multi-prior expansion changes float-only tie-breaks | W3 lands before its fixture | silent regression across the whole CTR corpus | `plain_ctr`/`tensor_ctr_e2e`/`fstr_ctr` all pin one prior ⇒ they are the regression gate; run them at every wave |
| R2 | f64 vs upstream f32 in the BTMV quantizer | boundary bin | `≤1e-5` failure only at specific data | compute the BTMV arm in `f32`; add a differential test (precedent: `LEARNING_RATE_F32.md`) |
| R3 | `.cbm` save fails for BTMV models | user calls `save_model` | typed error at the worst moment | decide in W4 (lift / defer / documented rejection) |
| R4 | `gen_fixtures.py` corpus wipe | any argv typo | frozen corpus destroyed | use fixture-local generator (Pattern B); never invoke `generator/gen_fixtures.py` |
| R5 | `counter_calc_method` "verified" without an eval set | a learn-only fixture "passes" trivially | false confidence | §B proves `maxdiff = 0` without eval; require an eval-set fixture or an explicit deferral |
| R6 | Merge friction with the uncommitted one-hot device wave | Part 1 edits the same `boosting.rs` regions | conflict / silent behavior change | land Part 1 on top of the wave, not beside it; re-run the one-hot oracles |
| R7 | `ctr_splits_for_tree` change unguarded | it has no covering test | silent breakage of the no-CTR fallback | add a test before touching it |
| R8 | Multiclass Borders apply is wrong (`target_border_idx` ignored) | a multiclass CTR model | wrong predictions | out of scope; record as a known limitation with the `static_ctr_provider.cpp:90-110` anchor |

---

## J. Open questions / UNVERIFIED items

1. **`[UNVERIFIED]` Does `PrepareScratchBlob<int>` zero-initialize `TBucketsView`'s backing array?**
   Correctness of `CalcOnlineCTRClasses` requires it. `[INFERRED: yes, from upstream correctness]`.
   *Settled by:* reading `catboost/private/libs/algo_helpers/scratch_cache.h` (not fetched).
   *Blocking?* **No** — the repo allocates fresh zeroed vectors regardless.
2. **`[UNVERIFIED]` Whether the online `Counter` bucket space (`ComputeReindexHash` with
   `topSize = CtrLeafCountLimit`) can differ from the final-CTR bucket space for large-cardinality
   columns.** `ctr_leaf_count_limit` defaults to `u64::MAX` `[VERIFIED: EXPERIMENT probe2.py]`, so
   they coincide for every realistic fixture. *Blocking?* **No** at default settings; would matter
   only if `ctr_leaf_count_limit` is ever exposed.
3. **`[UNVERIFIED]` Whether the repo's `PerfectHash` first-seen bin ordering matches upstream's
   `ComputeReindexHash` bin ordering.** It has never needed to (lookups are by hash, and
   `bake_ctr_table` documents the ordering as internally-consistent-only
   `[VERIFIED: LOCAL crates/cb-train/src/ctr/bake.rs:165-169]`). **But `Counter`'s
   `CounterDenominator = MAX bucket total` is order-independent, and `Buckets`' numerator is
   per-bucket, so this remains non-blocking.** *Blocking?* **No.**
4. **`[UNVERIFIED]` Whether the `buckets_simple`/`btmv_simple` isolating configs actually yield a
   winning split of the intended type on a 30-row corpus.** Data-dependent. *Settled by:* running
   the generator with the §E.4 anti-false-pass assertion. *Blocking?* **No, but it must be checked
   at generation time, not discovered at test time.**
5. **`[OPEN DECISION — needs the maintainer]` `.cbm` mean-table support (§F.4).** Lift the v1
   restriction, defer BTMV, or ship BTMV with a tested `.cbm` rejection. This is a scope decision,
   not a research gap.
6. **`[OPEN DECISION]` Whether Part 1 should also close the "upstream default is a
   2-description list" gap.** Recommended: **no** (§D.3, §H decision 1) — but it must be recorded
   as a known parity gap rather than left implicit, because Part 2 exposes the default to users.

**Nothing in this list blocks planning.** Items 5 and 6 are decisions the planner must make and
record; items 1–4 are verifiable during implementation without changing the plan's shape.

---

## K. Sources

**Upstream (fetched live, 2026-08-01, tag `v1.2.10`)** — base
`https://raw.githubusercontent.com/catboost/catboost/v1.2.10/`:
- `catboost/private/libs/algo/online_ctr.h` (185 lines; `CalcCTR` at **128-131** — matches the
  in-repo citation exactly)
- `catboost/private/libs/algo/online_ctr.cpp` (1144 lines; `CalcNormalization` 102-111,
  `UpdateGoodCount` 115-121, `CalcOnlineCTRClasses` 144-215, `CalcQuantizedCtrs` 271-342,
  `CalcOnlineCTRSimple` 344-435, `CalcOnlineCTRMean` 437-501, `CalcOnlineCTRCounter` 503-562,
  `CountOnlineCTRTotal` 564-573, `ComputeOnlineCTRs` 600-801, `CalcFinalCtrsImpl` 875-940,
  `CalcFinalCtrs` 943-1002)
- `catboost/private/libs/ctr_description/ctr_type.h` + `.cpp` (`ECtrType` discriminants;
  `NeedTarget` / `NeedTargetClassifier` / `IsPermutationDependentCtrType`)
- `catboost/private/libs/algo/ctr_helper.h` + `.cpp` (`GetTargetBorderCount` 34-42,
  `GetCtrInfo` 52-62, `MakeCtrInfo` / CPU denom==1 guard)
- `catboost/private/libs/options/cat_feature_options.cpp` (`GetDefaultPriors` 118-138,
  `TCatFeatureParams` ctor 226-240 incl. 231/232/234, `TCtrDescription` ctors 140-177)
- `catboost/private/libs/options/catboost_options.cpp` (`CreateDefaultCounter` 393-415,
  `SetCtrDefaults` 429-477, `ValidateCtr` 494-541, `SetDefaultPriorsIfNeeded` 1209-1215)
- `catboost/private/libs/options/restrictions.h` (`IsSupportedCtrType` 18-48)
- `catboost/private/libs/options/enums.h` (`ECounterCalc` 248-251)
- `catboost/private/libs/algo/greedy_tensor_search.cpp` (`AddCtrsToCandList` 400-428,
  `AddSimpleCtrs` 457-486, `AddTreeCtrs` 491-551)
- `catboost/private/libs/algo/projection.h` (`IsSingleCatFeature` 102-104,
  `GetFullProjectionLength` 138-144)
- `catboost/libs/model/online_ctr.h` (`TModelCtr::Calc` **289-292**, `TCtrHistory` 369-378,
  `TCtrMeanHistory` 380-401 — again matching the in-repo citations)
- `catboost/libs/model/static_ctr_provider.cpp` (`TStaticCtrProvider::CalcCtrs` 14-122)

**Local repository (read this session):**
`crates/cb-train/src/ctr/{mod,online,calc_ctr,final_ctr,ctr_feature,bake}.rs`;
`crates/cb-train/src/{candidates,projection}.rs`;
`crates/cb-train/src/boosting.rs` (440-530, 1440-2002, 3040-3300, 4630-4680, 5300-5480, 775-843);
`crates/cb-train/src/tree.rs` (120-170, 3198-3300, 3448-3473);
`crates/cb-model/src/ctr_data.rs` (190-360, 575-600, 749-800);
`crates/cb-model/src/{apply,model,cbm}.rs` (via CodeGraph);
`crates/cb-oracle/generator/gen_fixtures.py` (`__main__`, `gen_one_hot_only` 1398-1424,
`gen_tensor_ctr_e2e` 2468-2540);
`crates/cb-oracle/generator/{ordered_oracle.cpp,instrumented_trainer_STATUS.md}`;
`crates/cb-oracle/fixtures/{plain_ctr,ordered_ctr,tensor_ctr,tensor_ctr_e2e,ctr_load,fstr_ctr}/config.json`;
`.planning/plans/catboost-builder-cat-features-routing/{SPEC.md,PLAN.md,PLAN-CHECK.md}`;
`.planning/plans/one-hot-categorical-training/baseline/README.md`.

**CodeGraph queries:**
`codegraph_explore "calc_ctr_online calc_ctr_online_bin accumulate_online build_final_ctr ctr_splits_for_tree"`;
`codegraph_explore "ModelSplit OneHot variant one_hot_splits ObliviousTree passes_split grow_one_hot_tree production caller"`.

**TreeFinder:** `search_hierarchy("CTR type simple_ctr combinations_ctr counter_calc_method
categorical training")` → **no relevant indexed document** (index holds only
`snapshot-resume/SPEC.md`, `snapshot-resume/PLAN.md`, `xgboost-rust-rewrite/SPEC.md`). Consistent
with the Part-2 SPEC's own note that TreeFinder does not cover this feature. Nothing to reconcile.

**Experiments** (all with `.venv/bin/python`, `catboost 1.2.10`, Python 3.12.13; scripts under the
session scratchpad `…/scratchpad/probe{1..7}.py`):
- `probe1.py` — CPU legality of all six `ECtrType` values
- `probe2.py` — `get_all_params()` default expansion of `simple_ctr` / `combinations_ctr`
- `probe4.py`/`probe5.py` — `features_info.ctrs` + `ctr_data` shape per CTR type;
  Buckets `target_border_idx ∈ {0,1}`
- `probe6.py` — `counter_calc_method` observability with/without `eval_set`
- `probe7.py` — Counter table counts + `CounterDenominator` under `Full` vs `SkipTest`

**Local commands:** `wc -l`/`diff` on `catboost-master/` vs upstream;
`grep -rn "data_to_pool" crates/catboost-rs-py/src/` (14 sites);
`grep -rn "simple_ctr:" crates/ --include=*.rs` (65 occurrences / 62 files);
`git status --short`.

---

## L. Confidence

**HIGH (directly verified by upstream source, live experiment, or repository evidence) — 27:**
A.0 shared quantizer / no per-type shift-norm; A.1 Buckets numerator + `UpdateGoodCount` +
`targetBorderCount = targetClassesCount`; A.1 Borders-multiclass cumulative numerator; A.1
Borders-binclf is the classes special case; A.2 BTMV `(Sum, Count)` prefix + divisor
`targetClassesCount-1`; A.3 Counter is whole-set (not a prefix) + MAX denominator + `+1`; A.3
Counter is permutation-independent; A.4 FloatTargetMeanValue absent from online dispatch AND
CPU-rejected; A.5 FeatureFreq CPU-rejected; B `counter_calc_method` affects only Counter, only via
sample range, and only two sites; B unobservable without eval set (measured `maxdiff=0`); B
observable with eval set (measured counts + denominators); C `GetCtrInfo` routing by
`IsSingleCatFeature`; C `per_feature_ctr` override exists; C `IsSingleCatFeature` includes
bin/one-hot emptiness; D upstream default `simple_ctr` is a 2-description list (measured); D
`counter_calc_method`/`max_ctr_complexity`/`ctr_border_count`/`one_hot_max_size` defaults match; D
CPU requires `PriorDenom == 1` and Uniform CTR binarization; E `gen_fixtures.py` fallthrough to
`main()`; E fixture-local generator pattern exists; E all four CPU-legal types train + emit typed
JSON; F `.cbm` rejects mean tables (encode and decode); F `CtrData::from_baked` hard-codes
`mean: Vec::new()`; F `ctr_splits_for_tree` has no covering test; F 62-file `BoostParams` blast
radius; F `catboost-master/` is a stale stub at a different revision; G MAJOR-5's four missing
`data_to_pool` sites; G CRITICAL-4 is obsolete (one-hot now trains).

**MEDIUM (multiple reliable sources, not exercised locally) — 5:**
The f32-vs-f64 divergence risk for BTMV (formula verified, magnitude not measured); the exact
candidate-count multiplication from multi-prior expansion (formula verified from
`AddCtrsToCandList`, not run); the one-hot-model `predict_raw` silent-wrongness in T10/T11
(derived from `passes_one_hot_split` + `predict_raw = predict_raw_cat(.., &[])`, not executed);
the multiclass-Borders apply limitation (upstream branch verified, repo branch verified, no test);
`bake_ctr_table`'s per-type reshape requirements (derived from `CalcFinalCtrsImpl`, not implemented).

**LOW (incomplete / requires validation) — 4:**
`PrepareScratchBlob` zero-initialization (J.1); online-vs-final bucket-space equality under a
non-default `ctr_leaf_count_limit` (J.2); `PerfectHash` vs `ComputeReindexHash` bin-order
equivalence (J.3); whether the proposed 30-row isolating configs yield a winning split of each
intended type (J.4).
