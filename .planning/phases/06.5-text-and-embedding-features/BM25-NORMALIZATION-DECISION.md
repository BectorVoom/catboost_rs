# BM25 estimated-feature split-border normalization — source location + instrumented dump + A/B decision

**Plan:** 06.5-08 (gap-closure investigation, gates 06.5-09)
**Question (open since 06.5-04):** `crates/cb-oracle/fixtures/text_calcers/BM25/splits.npy` stores the BM25
estimated-feature split borders as the pair `(-0.550486, 1.2439)` repeated across 5 trees, while the raw
BM25 calcer scores for this 16-row corpus are provably `O(1e-3)` (max ≈ `0.0064`). Where does the ~200×
scale come from? Plan 06.5-04 proved exhaustively that no single learn-permutation strict prefix and no
offline whole-set estimate reproduces this scale, but never located the mechanism in source.

---

## Source mechanism

The estimated-feature value→border path was read end-to-end in upstream catboost 1.2.10. The chain from
the raw calcer output to the stored split border has **no value transform anywhere in it**; the stored
border is exactly `BestSplit(rawCalcerValues)`:

1. **Raw calcer output (O(1e-3), no normalization).**
   `catboost/private/libs/feature_estimator/base_text_feature_estimator.h:74-88` —
   `TTextBaseEstimator::ComputeOnlineFeatures` runs the read-before-update loop over `learnPermutation`,
   writing `learnFeatures[f*samplesCount+line] = featureCalcer.Compute(text)` directly (line 77 `Compute`,
   line 81-87 `learnVisitor(f, TConstArrayRef<float>(learnFeatures.data()+f*samplesCount, …))`). The
   per-doc value written is the raw BM25 score. **No mean-shift / z-score / scale is applied here** — the
   already-shipped `cb_instr_calcer_encoding` hook (line 178) dumps exactly this array.
   `catboost/private/libs/text_features/bm25.cpp:33-83` — `Score()` (saturation, `k=1.5,b=0.75`) and
   `CalcTruncatedInvClassFreq()` (`truncate=1e-3`) produce the raw scores; this math was re-derived
   bit-exact (≤1e-5) in 06.5-04 and re-confirmed `O(1e-3)` independently for this corpus (see Instrumented
   dump below, reference column).

2. **Border selection — directly on the raw `values`, no transform.**
   `catboost/private/libs/algo/estimated_features.cpp:204-250` — `CreateSingleFeatureWriter`'s visitor
   lambda receives `TConstArrayRef<float> values` (the raw calcer column from step 1) and, when the
   feature has no borders yet (line 216 `!HasBorders`), calls
   `NSplitSelection::BestSplit(NSplitSelection::TFeatureValues(std::move(valuesForQuantization)), …,
   binarizationOptions.BorderCount, binarizationOptions.BorderSelectionType, …)` (lines 225-232) and stores
   the result verbatim: `quantizedFeaturesInfo->SetBorders(TFloatFeatureIdx(featureIdx),
   std::move(quantization.Borders))` (lines 234-237). `valuesForQuantization` is just a row-subset of
   `values` (`GetSubset`, lines 217-221) — **no standardization**. `library/cpp/grid_creator/binarization.cpp`
   (`BestSplit` / GreedyLogSum) was grepped for `Mean|StdDev|Standard|Normaliz|Scale` — **none present**;
   `BestSplit` returns thresholds strictly inside the input value range.

3. **Stored model border = the quantizer border, verbatim.**
   `catboost/private/libs/algo/split.cpp:33-46` — for `ESplitType::EstimatedFeature`,
   `TEstimatedFeatureSplit{ …, estimatedFeaturesInfo.QuantizedFeaturesInfo->GetBorders(TFloatFeatureIdx(
   FeatureIdx))[BinBorder] }` (lines 45-46). The split threshold is the quantizer border from step 2.
   `catboost/libs/model/model.cpp:196-209` — `EstimatedFeatures.back().Borders.push_back(
   estimatedFeatureSplit.Split)` writes that same value into the serialized model (no export-time scaling).

**Candidate verdict (per D-04 transcribe-first; all three explicitly investigated):**
- **(a) per-feature value transform (standardization / z-score / mean-shift+scale)** — **ABSENT** from the
  located source. Searched `estimated_features.cpp`, `base_text_feature_estimator.h`, `bm25.cpp`,
  `library/cpp/grid_creator/binarization.{h,cpp}`, `catboost/private/libs/quantization/utils.cpp`,
  `grid_creator.cpp` for `Mean|StdDev|Standard|Normaliz|Scale|ZScore` — no hit on float feature values.
- **(b) different border-selection algorithm/grid for estimated columns** — borders ARE selected by
  `BestSplit` with `binarizationOptions.BorderSelectionType` (estimated_features.cpp:229), same family the
  Rust seam uses; but `BestSplit` thresholds are bounded by the input range, so a different selection type
  still cannot map `O(1e-3)` inputs to `±1.24` outputs. Border-algorithm choice is **not** the scale source.
- **(c) multi-permutation ordered averaging** — `ComputeOnlineFeatures` runs once per the SINGLE
  `learnPermutation` passed in (estimated_features.cpp:448-472 dispatch; `fold.cpp` constructs the fold's
  permutation). Averaging across permutations would change WHICH values appear (and can yield the genuine
  depth-2 `[7,2,0,7]` structure 06.5-04 noted) but is still an average of `O(1e-3)` scores — it **cannot
  raise the magnitude to `±1.24`** either.

**Source conclusion:** none of (a)/(b)/(c) — as located in source — can turn an `O(1e-3)` raw BM25 column
into `±1.24` borders, because every step in the chain (`base_text_feature_estimator.h:74-88` →
`estimated_features.cpp:204-250` → `split.cpp:45-46` → `model.cpp:209`) preserves the value scale and
`BestSplit` borders are bounded by the input range. Therefore EITHER the `values` array actually fed into
`BestSplit` for this fixture is NOT the `O(1e-3)` raw column my reference + 06.5-04's closed-form computed
(i.e. the real online calcer column differs), OR the `splits.npy` fixture was produced by a path the source
reading above does not cover. Source reading alone cannot discriminate these; the instrumented RUN-ONCE
dump below feeds the EXACT array into the recorded decision.

<!-- gsd:task1-source-mechanism complete; Instrumented dump + DECISION appended by Task 2 -->
