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

---

## Instrumented dump

The instrumented catboost 1.2.10 trainer was rebuilt sudo-free (`/tmp` had been cleared; clang-18/lld-18
toolchain survived in `/tmp/clang18_prefix`, the build tree `/tmp/cb_build313` was re-provisioned via
`crates/cb-oracle/generator/build_instrumented_trainer.sh`, full Release build `[4300/4300]`, staged
`/tmp/cb_build313/instr_pkg/catboost/_catboost.so`). A NEW env-gated hook `cb_instr_estimated_borders`
(06.5-08) was added at `estimated_features.cpp:234-237` (immediately after the `SetBorders(...)` that stores
the `BestSplit` result) to dump the per-estimated-feature selected borders. The vendored
`catboost-master/` patch stays **UNCOMMITTED** (the entire `catboost-master/` subtree is untracked in this
repo; D-09/D-12 satisfied) — only the driver-script hook block + this DECISION.md are committed.

**RUN-ONCE replay** (`/tmp/cb065_08/replay_bm25_instr.py`, `PYTHONPATH=/tmp/cb_build313/instr_pkg`,
`CB_INSTRUMENT_LOG=/tmp/cb065_08/instr_log.jsonl`): the EXACT committed `params.json` (text-only BM25)
over the frozen 16-row corpus (`texts.json` + `labels.npy`). No fabricated fixtures, no weakened tolerance,
no `#[ignore]`.

### Dumped values (decisive)

1. **BM25 estimated-feature borders selected by upstream `BestSplit` — O(1e-3), NOT ±1.24:**
   ```
   {"event":"estimated_borders","feature":0,"borders":[0.000465116…,0.000947044…,0.001251158…,…,0.005428576…]}
   {"event":"estimated_borders","feature":1,"borders":[0.000540540…,0.001270467…,0.001660159…,…,0.005288986…]}
   ```
   (`values:[]` because `valuesForQuantization` was already `std::move`-d into `TFeatureValues` at
   `estimated_features.cpp:226` before the hook ran; the **borders** — the load-bearing output — are
   captured verbatim.) These borders are the SAME O(1e-3) scale as the raw BM25 column.

2. **Raw BM25 calcer column (`calcer_encoding`) — O(1e-3):** 32 events, value range `min=0  max=0.00637875`
   (e.g. doc2 `[0.000930…,0.001860…]`). Matches the independent closed-form re-derivation (06.5-04) and the
   Rust seam column reproduced here (`feature 0: max≈0.0062; feature 1: max≈0.0064`).

3. **The committed `splits.npy` (±1.24 / -0.550486) — belongs to a DIFFERENT feature.** Re-running the
   identical `params.json` fit and reading `model._get_tree_splits()` (the exact API the fixture generator
   `_tree_split_borders`/`gen_text_embedding_fixtures.py:249-263` parses for `splits.npy`):
   - **text-only pool** → `bin=0.00248965 / 0.00127047 / 0.00125116 / 0.00322683` (O(1e-3),
     `calcer_id=4559D4B0…` = the BM25 text calcer) — and the committed `model.cbm` **cannot even load
     against a text-only pool**: `"Feature emb0 is present in model but not in pool"`. The committed BM25
     fixture model was trained WITH an `emb0` embedding column.
   - **embedding-present pool** (`_make_pool` adds `embedding_features=["emb0"]`, the generator's default)
     → `bin=-0.550486 / 1.2439`, **bit-identical to the committed `splits.npy`**, and EVERY split carries
     `calcer_id=96AE6D4D92B01A43B254591E5FCD8467` — the **default EMBEDDING-feature calcer** on `emb0`, NOT
     the BM25 text calcer (`4559D4B0…`). The depth-2 `[7,2,0,7]` structure and the ±1.24 borders are the
     embedding clouds' (centers ±1.0) splits; the model never splits on the BM25 text feature because the
     well-separated embedding signal dominates the noisy O(1e-3) BM25 column.

### Cross-check verdict

The ±1.24 / -0.550486 scale is **not a BM25 normalization at all** — it is the split border of the
**embedding calcer** (`96AE6D4D…` on `emb0`), which the BM25 fixture's pool inadvertently included. The
genuine BM25 estimated-feature borders are O(1e-3), exactly the raw calcer scale, with NO transform —
confirming the source reading (no value-transform / no averaging-rescale / no estimated-column standardization).
The 06.5-04 "±1.24 normalized BM25 border" deferral rested on a **fixture mislabel**: `splits.npy` was frozen
from a text+embedding pool, so it records the EMBEDDING feature's borders, attributed to BM25.

---

## DECISION: PATH-A

**The "BM25 estimated-feature border normalization" is reproducible because it does not exist** — the
±1.24 / -0.550486 borders in `crates/cb-oracle/fixtures/text_calcers/BM25/splits.npy` are the borders of the
**default embedding calcer on the `emb0` column**, not the BM25 text calcer. The real BM25 estimated-feature
borders are O(1e-3) and are already reproduced ≤1e-5 by the existing Rust seam
(`select_borders_greedy_logsum` on the online read-before-update O(1e-3) BM25 column —
`crates/cb-train/src/estimated/estimated_features.rs:185-294`). There is NO upstream transform/averaging/
standardization to implement (source: `base_text_feature_estimator.h:74-88` → `estimated_features.cpp:204-250`
→ `split.cpp:45-46` → `model.cpp:209`, all scale-preserving; instrumented `estimated_borders` borders are
O(1e-3)).

### Numeric justification

| Quantity | Value (dumped) | Scale |
|---|---|---|
| BM25 raw calcer column (`calcer_encoding`) | `max = 0.00637875` | O(1e-3) |
| BM25 estimated-feature borders (`estimated_borders`, upstream `BestSplit`) | `[0.000465…, … 0.005428…]` | O(1e-3) — **reproducible** |
| BM25 text-calcer model splits (`_get_tree_splits`, text-only fit, `calcer_id=4559D4B0…`) | `0.00248965, 0.00127047, …` | O(1e-3) |
| Committed `splits.npy` / embedding-present fit (`calcer_id=96AE6D4D…` on `emb0`) | `-0.550486, 1.2439` | O(1) — **embedding feature, NOT BM25** |

### What 06.5-09 must now do (PATH-A reproduction recipe)

This is a **fixture-correctness** fix, not a trainer-normalization implementation:

1. **Regenerate the BM25 (and NaiveBayes/BoW) text-calcer fixtures from a TEXT-ONLY pool** — drop the
   `embedding_features=["emb0"]` from `_make_pool` for the *text* calcer fixtures in
   `crates/cb-oracle/generator/gen_text_embedding_fixtures.py` (the `gen_text_calcer` path), so `splits.npy`/
   `leaf_*`/`staged`/`predictions` record the BM25 **text** feature, not the embedding feature. Re-freeze with
   the instrumented single-thread trainer (RUN-ONCE, thread_count=1). The regenerated BM25 `splits.npy` will be
   O(1e-3) (e.g. `0.00248965, 0.00127047, …`).
2. **Add the BM25 per-stage oracle** (`crates/cb-oracle/tests/bm25_oracle_test.rs`) gating Splits/LeafValues/
   StagedApprox/Predictions ≤1e-5 against the regenerated text-only fixtures — the same per-stage gate
   NaiveBayes already passes (06.5-04). The Rust seam already produces the O(1e-3) borders, so this should pass
   without any production-code change; if a residual remains it is now a *real* calcer/quantizer bug to fix
   under Rules 1-3, no longer a "normalization" excuse.
3. **HARD ≤1e-5 parity contract for 06.5-09** (necessary-and-sufficient): BM25 text-only per-stage
   Splits/LeafValues/StagedApprox/Predictions ≤1e-5. No stage is deferred; the ±1.24 "normalization" gap is
   **dissolved**, not deferred. FEAT-01 closes when the regenerated text-only BM25 per-stage oracle is green.

### Path-B (not taken)

PATH-B (upstream-irreducible) is explicitly **rejected**: the dump proves the BM25 borders are O(1e-3) and
already reproduced; the irreducibility hypothesis (internal float standardization / non-recoverable
multi-permutation averaging) is falsified — the ±1.24 values are simply a different feature's borders.

<!-- gsd:06.5-08 complete: PATH-A (fixture mislabel; BM25 borders O(1e-3), reproducible) -->
