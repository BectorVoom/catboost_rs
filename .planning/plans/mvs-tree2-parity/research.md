# Research: MVS tree-2 upstream-parity bug (`mvs-tree2-parity`)

> Research only. No production code or test was left changed by this document — every
> experiment below was run against a temporarily patched working tree and reverted;
> `git status` is clean at HEAD `2c14d7f` [VERIFIED: RUN `git status --short` → empty,
> `git diff --quiet` → clean].

---

## 0. Research Summary

**Root cause — FOUND and PROVEN, not hypothesised.**

`crates/cb-train/src/bootstrap.rs:413-423` makes the MVS arm consume **three** draws on
the persistent training RNG per tree (`1` real `rand_seed` + `2` fabricated
"compensation" draws). Instrumented upstream CatBoost 1.2.10 consumes **exactly one**.
The resulting **+2-draws-per-tree phase drift** means tree 0 samples correctly, tree 1
onward samples from the wrong RNG phase, and the wrong 20 % dropout set eventually flips
a split — which is why divergence surfaces at tree 1 or 2 and is seed/bias dependent.

Deleting the two draws (a 10-line deletion, no algorithmic change):

| configuration | before | after |
|---|---|---|
| `boost_from_average=true`, seeds 0–4, 3 trees, ≤1e-5 | **3 / 5** (0, 2, 3) | **5 / 5** |
| `boost_from_average=false`, seeds 0–4, 3 trees, ≤1e-5 | **0 / 5** | **5 / 5** |
| frozen `bootstrap/mvs` oracle (3 trees) | pass | pass (still) |
| `bootstrap_dev/mvs` at `MVS_GATED_TREES = 3` | **fail @ split 5** | pass |
| whole `cb-train` suite | 503 pass / 1 known red | 503 pass / 1 known red |

[VERIFIED: RUN — full transcripts in §3 and §8]

**Most important constraints**

- `crates/cb-oracle/fixtures/bootstrap/**` + `fixtures/inputs/**` are FROZEN; they stay
  byte-unchanged and green (verified: they do).
- CatBoost quantization borders are **not** stable across configurations. Every
  comparison must use each model's own `float_feature_borders()`. §8.
- The instrumented upstream build already exists, is up to date, and reproduces its
  committed traces byte-for-byte. §4.

**Highest-risk findings**

1. The fix's *only* real risk is that it silently invalidates the phase accounting the
   device branch shares (`boosting.rs` reuses `bootstrap()` verbatim on the GPU path).
   Verified inert on CPU; the rocm re-verification list is §6.
2. Two further genuine (but currently *unobservable*) upstream divergences exist in the
   same function: an `f32`-vs-`f64` `sampleSize` and `f32`-narrowed sample weights. Both
   were spiked and are safe; neither is required. §2.6, §2.7.
3. `MVS_GATED_TREES = 2` and its 30-line doc comment in
   `crates/cb-train/tests/bootstrap_dev_oracle_test.rs:121-155` encode the *wrong*
   diagnosis and must be rewritten, not just bumped.

---

## 1. Our MVS implementation, line by line

File: `crates/cb-train/src/bootstrap.rs` (439 lines) [VERIFIED: LOCAL, read in full].

### 1.1 Constants and dispatch

| ours | line | value | upstream |
|---|---|---|---|
| `BAYESIAN_BLOCK_SIZE` | `bootstrap.rs:61` | `1000` | `tensor_search_helpers.cpp` block size |
| `MVS_BLOCK_SIZE` | `bootstrap.rs:64` | `8192` | `mvs.h:48` `const ui32 BlockSize = 8192` ✅ |
| `EBootstrapType` | `bootstrap.rs:69-80` | 5 arms | `EBootstrapType` ✅ |
| `BootstrapResult` | `bootstrap.rs:88-93` | `sample_weights: Vec<f64>`, `control: Vec<bool>` | `fold->SampleWeights` (`TVector<float>`!) + `TCalcScoreFold::Control` — see §2.7 |

### 1.2 `single_probability` — `bootstrap.rs:192-201`

```rust
192	fn single_probability(derivative_abs: f64, threshold: f64) -> f64 {
194	    if derivative_abs > threshold { 1.0 }
196	    else if threshold > 0.0 { derivative_abs / threshold }
198	    else { 0.0 }
```
Upstream (`mvs.cpp:17-19`) has **no** `threshold > 0` guard: `return (der > threshold) ?
1.0 : (der / threshold);` — a zero threshold yields `inf`/`NaN` upstream and `0.0` here.
Divergent only for an all-zero-gradient block (degenerate); see §7 risk R4.

### 1.3 `calculate_threshold` — `bootstrap.rs:209-273`

Structure-for-structure port of `TMvsSampler::CalculateThreshold` (`mvs.cpp:81-118`):

| ours | upstream | match |
|---|---|---|
| `215-218` pivot `= candidates.first()` (empty ⇒ `0.0`) | `mvs.cpp:88` `threshold = *candidatesBegin` (no empty guard) | ✅ semantics; ours adds a defensive empty case |
| `221-232` build `small` / `middle` / `large` vectors by `< / <= / >` | `mvs.cpp:89-94` **two in-place `std::partition`s** | ⚠️ same 3-way split, but ours preserves relative order inside each part; `std::partition` physically permutes (Hoare-style swaps) |
| `233` `sum_of_small_update = sum_f64(&small)` | `mvs.cpp:96` `Accumulate(candidatesBegin, middleBegin, 0.0)` | ⚠️ same *naive left-to-right* algorithm (`cb-core/src/reduction.rs:32-38` is an uncompensated `acc += v` loop — NOT Kahan) but a different *element order* (see §2.5) |
| `234-236` `n_large`, `n_middle`, `sum_of_middle = n_middle * threshold` | `mvs.cpp:97-99` | ✅ |
| `238-245` `estimated_sample_size`; guards `threshold != 0.0 ⇒ INFINITY` | `mvs.cpp:101-102` (unguarded division) | ⚠️ ours guards; upstream divides by zero |
| `247-259` `> sampleSize` branch: recurse on `large`, else closed form | `mvs.cpp:103-109` (`middleEnd != candidatesEnd`) | ✅ (`!large.is_empty()` ≡ `middleEnd != candidatesEnd`) |
| `260-272` else branch: recurse on `small`, else closed form | `mvs.cpp:110-116` (`middleBegin != candidatesBegin`) | ✅ |
| `254/267` extra `denom != 0.0` guards returning `threshold` | none upstream | ⚠️ ours guards |

It is a **recursive quickselect/partition** search, **not** bisection, with **no
tolerance and no iteration cap** — it terminates when one side is empty. Its iteration
count *is* data dependent (that was one of the task's hypotheses); see §3 for why it is
nevertheless not the defect.

### 1.4 `mvs_sample_weights` — `bootstrap.rs:281-331`

| ours | what | upstream |
|---|---|---|
| `288-290` | `sample_rate >= 1.0` ⇒ all `1.0`, **no draw** | `mvs.cpp:127-128` `if (SampleRate == 1.0f) Fill(..., 1.0f)` ✅ |
| `294` | `sample_rate = f64::from(sample_rate as f32)` | `TMvsSampler::SampleRate` is `float` (`mvs.h:47`) ✅ |
| `295` | `let rand_seed = rng.gen_rand();` — **the one and only** main-stream draw | `mvs.cpp:174` `const ui64 randSeed = rand->GenRand();` ✅ |
| `297` | `block_count = n.div_ceil(8192)` | `mvs.cpp:172-173` `blockParams.SetBlockSize(BlockSize)` ✅ |
| `299-300` | `TFastRng64::from_seed(rand_seed + block_idx)`, `.advance(10)` | `mvs.cpp:177-178` ✅ |
| `307` | `candidates[i] = (lambda + d*d).sqrt()` | `mvs.cpp:186-196` (`TVector(blockSize, lambda)`, `+= der*der` per dim, then `sqrt`) ✅ at `dim == 1` |
| `308-313` | `calculate_threshold(&mut candidates, 0.0, 0.0, sample_rate * block_size as f64)` — **per block**, seeded `sumSmall=0`, `nLarge=0` | `mvs.cpp:197-202` `CalculateThreshold(begin, end, 0, 0, SampleRate * blockSize)` — ⚠️ **`SampleRate * blockSize` is a `float` expression upstream**, `f64` here (§2.6) |
| `315-328` | reweight loop: `p = single_probability(sqrt(grad2+λ), threshold)`; `if p > f64::EPSILON { w = 1/p; r = block_rng.gen_rand_real1(); slot = w * (r<p) } else { 0 }` | `mvs.cpp:203-217` ✅ including the **conditional** draw (a `p ≤ ε` object consumes NO draw) |
| `323` | stores full `f64` weight | `mvs.cpp:213` stores into `TVector<float>` (`fold.h:217`) — ⚠️ §2.7 |

The threshold **is computed per block**, never globally — matching upstream. With the
fixture's `n = 1500` there is exactly **one** block, so all per-block/parallel-order
concerns are moot for the oracle.

### 1.5 `mean_grad_value` — `bootstrap.rs:336-342`

`sum_f64(|d| (d*d).sqrt()) / n`, one flat ordered sum. Upstream
`CalculateMeanGradValue` (`mvs.cpp:37-65`) computes the **same mathematical value** but
through a **blocked reduction**: `blockParams.SetBlockCount(CB_THREAD_LIMIT)` with
`CB_THREAD_LIMIT = 128` (`catboost/private/libs/options/restrictions.h:59`
[VERIFIED: LOCAL]), i.e. for `cnt = 1500`: `BlockSize = ceil(1500/128) = 12`,
`BlockCount = ceil(1500/12) = 125` (`local_executor.h:81-90` `SetBlockCount` →
`BlockSize = CeilDiv(range, blockCount)`, `BlockCount = CeilDiv(range, BlockSize)`
[VERIFIED: LOCAL]), then `Accumulate` over the 125 partial sums (`mvs.cpp:63`).
**The decomposition is independent of `thread_count`**, so `thread_count=1` does NOT
collapse it to a flat sum. Residual order-of-summation difference ≈ 1 ulp scale; affects
**tree 0's λ only** (`leafValues.empty()`). Not the defect (§3), but a real
transcription gap worth a comment.

### 1.6 `mvs_lambda` — `bootstrap.rs:348-356` and `last_iter_mean_leaf_value` — `bootstrap.rs:363-369`

- `mvs_lambda`: `prev_leaf_mean_l2.map(m → m*m)` else `mean_grad_value(d)²` ≡
  `GetLambda` (`mvs.cpp:67-79`) ✅. `TMaybe<float> Lambda` (`mvs.h:49`, the `mvs_reg`
  user override) is **not modelled** — correct while `mvs_reg` is unset, a gap if it is
  ever exposed.
- `last_iter_mean_leaf_value`: `mean(|v|)` over leaf values ≡
  `CalculateLastIterMeanLeafValue` (`mvs.cpp:21-35`) at `approxDimension == 1` ✅.
- **The λ input is the learning-rate-SCALED leaf values, and that is correct**:
  `train.cpp:591` calls `NormalizeLeafValues(..., LearningRate, ..., &treeValues)` which
  does `leafValue *= learningRate`
  (`catboost/private/libs/algo_helpers/approx_updater_helpers.cpp:33-37`) **before**
  `LearnProgress->LeafValues.push_back(treeValues)` at `train.cpp:639`
  [VERIFIED: LOCAL]. `boosting.rs:4723` feeds the stored (scaled) values. ✅

### 1.7 The `Mvs` dispatch arm — `bootstrap.rs:410-434` — **THE BUG**

```rust
410	        EBootstrapType::Mvs => {
411	            let lambda = mvs_lambda(derivatives, prev_leaf_mean_l2);
412	            let sample_weights = mvs_sample_weights(derivatives, lambda, subsample, rng);
413	            if subsample < 1.0 {
414	                // MVS uses `performRandomChoice=false` (calc_score_cache.cpp:752),
415	                // so its `sampledDocs->Sample` keeps the full doc set and the score
416	                // path consumes two additional `GenRand()` draws on the main stream
417	                // relative to the Bernoulli/`SetSampledControl` path. ...
421	                rng.gen_rand();
422	                rng.gen_rand();
423	            }
426	            let control: Vec<bool> = sample_weights.iter()
428	                .map(|&w| w > f64::from(f32::EPSILON)).collect();
```
- Lines `426-429` are **correct**: `SetControlNoZeroWeighted` is
  `Control[i] = sampleWeights[i] > std::numeric_limits<float>::epsilon()`
  (`calc_score_cache.cpp:1196-1203`) ✅.
- Lines `413-423` are **fabricated**. The cited justification is wrong on its own
  source: with `performRandomChoice == false`, `TCalcScoreFold::Sample`
  (`calc_score_cache.cpp:730-747`) takes the `else` branch — it sets
  `BernoulliSampleRate = 0.0f` and calls `SetControlNoZeroWeighted`, and **never touches
  `rand`**. Nothing else in `Bootstrap` (`tensor_search_helpers.cpp:487-566`) draws for
  MVS: `CalcWeightedData` (`:442-486`) is draw-free, and `Langevin` is off.
  [VERIFIED: LOCAL] and confirmed numerically in §2.4/§4.2.

---

## 2. Upstream source of record — function by function

Root: `/home/user/cb_instrumented_build/catboost-src/catboost/private/libs/algo/`
(upstream tag `v1.2.10`, commit `b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084`, with this
repo's three already-instrumented files dropped in
[PROJECT: `.planning/plans/bayesian-rng-draw-accounting/instrumented-ground-truth/GROUND_TRUTH.md:1-30`]).

### 2.1 `mvs.h:19-50` — the sampler object

```cpp
19	class TMvsSampler {
21	    TMvsSampler(ui32 sampleCount, float sampleRate, const TMaybe<float>& lambda)
46	    ui32 SampleCount;
47	    float SampleRate;          // <-- f32
48	    const ui32 BlockSize = 8192;
49	    TMaybe<float> Lambda;      // <-- mvs_reg override, unmodelled in Rust
```

### 2.2 `GetSingleProbability` (`mvs.cpp:17-19`) and `CalculateThreshold` (`mvs.cpp:81-118`)

Quoted in §1.2/§1.3. Key upstream facts:
- **Not** bisection. A recursive 3-way `std::partition` quickselect. **No tolerance, no
  iteration budget**; termination is "one partition became empty".
- `std::partition` is **not stable** and mutates the range; libstdc++ uses a two-pointer
  swap loop. Consequently `Accumulate(candidatesBegin, middleBegin, 0.0)` at `mvs.cpp:96`
  sums the small part in *permuted* order, and the recursive call operates on a permuted
  subrange. Our port sums in original order. Difference: ~ulp-level, **not** structural.
- Upstream has no divide-by-zero guards (`mvs.cpp:102, 108, 115`).

### 2.3 `CalculateMeanGradValue` (`mvs.cpp:37-65`) / `CalculateLastIterMeanLeafValue` (`mvs.cpp:21-35`) / `GetLambda` (`mvs.cpp:67-79`)

Covered in §1.5/§1.6. The **only** order-sensitive upstream reduction in the MVS path is
`CalculateMeanGradValue`'s 125-block decomposition. `GenSampleWeights` is blocked at
8192 (one block at `n = 1500`), so no parallel-order concern there.

### 2.4 `GenSampleWeights` (`mvs.cpp:120-224`) — the RNG contract

Exactly one main-stream draw: `mvs.cpp:174 const ui64 randSeed = rand->GenRand();`
Everything else runs off the per-block child RNG
(`mvs.cpp:177-178 TRestorableFastRng64 prng(randSeed + blockId); prng.Advance(10);`).
The reweight draw (`mvs.cpp:212 double r = prng.GenRandReal1();`) is **conditional** on
`probability > double-epsilon` (`mvs.cpp:210`) — ours matches (`bootstrap.rs:319-321`).

Call chain, per tree, for `SamplingFrequency = PerTree` (the default):
`GreedyTensorSearch` → `DoBootstrap` (`greedy_tensor_search.cpp:1959`, before the level
loop) → `Bootstrap` (`tensor_search_helpers.cpp:487`) → `TMvsSampler::GenSampleWeights`
→ `CalcWeightedData` → `sampledDocs->Sample(..., performRandomChoice=false, ...)`.
`greedy_tensor_search.cpp:1222` (`if (!isSamplingPerTree) DoBootstrap(...)`) is not taken.

### 2.5 `SetControlNoZeroWeighted` / `Sample` (`calc_score_cache.cpp`)

```cpp
730	void TCalcScoreFold::Sample( ... TRestorableFastRng64* rand, ...
742	    if (performRandomChoice) {
743	        SetSampledControl(objectCount, samplingUnit, fold.LearnQueriesInfo, rand);
744	    } else {
745	        BernoulliSampleRate = 0.0f;
747	        SetControlNoZeroWeighted(objectCount, fold.SampleWeights.data());
748	    }
...
1196	void TCalcScoreFold::SetControlNoZeroWeighted(int docCount, const float* sampleWeights) {
1197	    constexpr float EPS = std::numeric_limits<float>::epsilon();
1199	        Control[docIdx] = sampleWeights[docIdx] > EPS;
```
Note `BernoulliSampleRate` is 1.0f for MVS at construction
(`calc_score_cache.h:55-62 GetBernoulliSampleRate` returns `1.0f` for non-Bernoulli)
and is clobbered to `0.0f` on the first MVS `Sample` — before any consumer reads it in
that tree, so it introduces no cross-tree state divergence. [VERIFIED: LOCAL]

### 2.6 `float` vs `double` narrowings — real and quantified

| # | upstream | ours | delta at fixture |
|---|---|---|---|
| N1 | `SampleRate * blockSize` is `float * ui32` ⇒ **float** arithmetic, then widened into the `double sampleSize` parameter (`mvs.cpp:202`) | `sample_rate * block_size as f64` in `f64` (`bootstrap.rs:312`) | upstream `1200.0` exactly vs ours `1200.0000178813934` ⇒ **+1.788e-5** absolute on `sampleSize` [VERIFIED: RUN numpy, §8.4] |
| N2 | `fold->SampleWeights` is `TVector<float>` (`fold.h:217`); `mvs.cpp:213` narrows `1/probability` to `f32`; `CalcWeightedData` reads `const float*` (`tensor_search_helpers.cpp:457, 470-472`) | full `f64` (`bootstrap.rs:90, 323`) | ~6e-8 relative per non-unit weight |
| N3 | `powf`, `FastLogf` (Bayesian) | already transcribed in f32 (`bootstrap.rs:117-149`) | ✅ no gap |
| N4 | `sqrt` (double) everywhere in MVS | `f64::sqrt` | ✅ |

Both N1 and N2 were **spiked** (§8.5, §8.6): each keeps all oracles green and changes
the measured residuals by **nothing at the printed precision**. Reason: MVS sample
weights reach the model *only* through the discrete split choice (leaf values are
estimated on the un-sampled averaging fold), so a ≤1e-7 relative weight/threshold
perturbation is invisible unless it flips a near-tied split. This is exactly why these
were not the bug and why the RNG phase was.

### 2.7 Other differences worth recording

- Our multi-dimension path pre-aggregates `der_obj[i] = sqrt(Σ_d wd²)`
  (`boosting.rs:3819-3829`) then squares it again in `mvs_sample_weights`; upstream sums
  `der*der` per dim directly into the λ-seeded accumulator (`mvs.cpp:187-193`). A
  round-trip ulp difference at `dim > 1`. No MVS multi-dim oracle exists (only the smoke
  test `multiclass_onevsall_mvs_trains_per_object`,
  `crates/cb-train/tests/multidim_sampling_regression_test.rs:131-144`).
- `CalcWeightedData` also does `ff.SampleWeights[i] *= learnWeights[i]`
  (`tensor_search_helpers.cpp:481-485`) before the control mask is derived. Our port
  never multiplies by learn weights. Inert while all learn weights are `1.0` (all
  in-scope fixtures) — a real gap for a weighted pool.

---

## 3. Why tree 1/2 and not tree 0 — hypotheses, ranked, one confirmed

Per-tree draw budget (MVS, `depth=2`, 4 listed float features, `random_strength=0`),
from the committed instrumented trace [VERIFIED: LOCAL
`.planning/plans/bayesian-rng-draw-accounting/instrumented-ground-truth/mvs.jsonl`]:

```
tree 0: rng_start cc=0  pre_gts cc=2   level0 cc_start=7 (+1 bootstrap +4 RSM)
        level0: calcscores 1, selectbest 10 -> cc 18 ; gap 18->22 = 4 RSM
        level1: calcscores 1, selectbest 10 -> cc 33 ; pre_leaf 33 ; end 35
tree 1: 35 -> 68   (33 draws)      tree 2: 68 -> 101  (33 draws)
```
`pre_gts(2) → level0 cc_start(7)` is a gap of **5** = `Bootstrap` + 4 RSM
`GenRandReal1`s (`SelectCandidatesAndCleanupStatsFromPrevTree`,
`greedy_tensor_search.cpp:329, 343, 352`, one per candidate sublist, drawn even at
`Rsm = 1.0`). ⇒ **MVS `Bootstrap` = 1 draw.** Our code takes 3.

| # | hypothesis | verdict |
|---|---|---|
| **H1** | **Cumulative RNG phase drift: MVS consumes +2 draws/tree, so tree 1's `rand_seed` is 2 draws late, tree 2's is ≥4 late; the wrong 80 % subsample eventually flips a split.** | ✅ **CONFIRMED ROOT CAUSE.** Removing the 2 draws turns 3/5 + 0/5 into 5/5 + 5/5 and keeps every frozen oracle green (§8.2, §8.3). |
| H2 | λ magnitude shrinks into a regime where `CalculateThreshold` behaves differently | ❌ Rejected. λ is *already* carried at tree 1 (which matched), and with H1 fixed all 10 configs pass with **unchanged** λ code. |
| H3 | Threshold-search iteration count is data dependent and diverges once λ crosses a boundary | ❌ Rejected by the same evidence; the search is a deterministic partition recursion with no tolerance, and the port is structurally identical. |
| H4 | `sampleSize` f32/f64 narrowing (N1) | ❌ Not the cause (real, but invisible — §8.5). |
| H5 | f32 `SampleWeights` narrowing (N2) | ❌ Not the cause (real, but invisible — §8.6). |
| H6 | `CalculateMeanGradValue` blocked-reduction order | ❌ Not the cause (tree-0-only, ulp-level). |
| H7 | `last_iter_mean_leaf_value` / `CalculateLastIterMeanLeafValue` | ❌ Already ruled out by the task; independently confirmed (§1.6 verifies the lr-scaling is right). |

**Why H1 explains both observations**

- *Onset*: tree 0's `rand_seed` is drawn at call-count 2 in **both** implementations, so
  tree 0 is always right. Drift starts at tree 1. Whether the wrong subsample actually
  changes the *argmax* split is luck; on this clean, strongly separable synthetic set
  (`y = X·[1.5,−2,0.5,3] + 0.1·N(0,1)`) it usually does not at tree 1 and usually does
  by tree 2. The drift also **compounds non-uniformly**, because
  `SelectBestCandidate`'s `std_normal` draw count is itself phase dependent (Marsaglia
  polar rejection — the trace shows 8 and 10 for different levels), so by tree 2 the
  offset is not a clean `+4`.
- *`boost_from_average` asymmetry*: with bias 0 the first-tree gradients are `−y`
  (large, single-signed, wide spread), which puts many more objects strictly below the
  MVS threshold (heterogeneous `p < 1`) and makes the *selected subset* — not just a
  reweighting — decide the split. The fit is therefore far more sensitive to a wrong
  phase: 0/5 vs 3/5. [INFERRED: mechanism; the 0/5→5/5 flip under the H1 fix is
  VERIFIED]
- *Correction to the recorded evidence*: the divergence is **not** always at flat split
  index 4 or 5. Measured first divergent index across the 7 failing configs:
  `{3, 4, 4, 4, 5, 5, 4}` — `boost_from_average=true, seed=4` first diverges at index
  **3**, i.e. **tree 1, split 1** [VERIFIED: RUN §8.2]. That is precisely what H1
  predicts (drift begins at tree 1) and is inconsistent with any "tree-2-specific"
  mechanism. The doc comment at `bootstrap_dev_oracle_test.rs:140-144` ("never trees 0
  or 1") is therefore wrong and must be rewritten.

---

## 4. The instrumented-upstream ground-truth workflow

### 4.1 Build state — **compiled and usable, no rebuild needed**

- Binary: `/home/user/cb_instrumented_build/build/catboost/app/catboost`, 411 533 960 B,
  mtime 2026-07-30 06:47 [VERIFIED: RUN `ls -la`].
- Build system: CMake + Ninja, generator dir `/home/user/cb_instrumented_build/build`,
  `ninja` at `/home/user/cb_instrumented_build/build_venv/bin/ninja`.
- `ninja -n catboost/app/catboost` in that dir prints **"ninja: no work to do."**
  [VERIFIED: RUN] ⇒ the tree is up to date.
- Source tree: `/home/user/cb_instrumented_build/catboost-src` (upstream `v1.2.10` +
  this repo's 3 instrumented files). Build caveats (clang 22, `ld.bfd` not `lld`,
  `--start-group` around clapack, OpenSSL `no_fips`, ad-hoc Perl modules) are recorded in
  [PROJECT: `.../instrumented-ground-truth/GROUND_TRUTH.md:1-30`].

### 4.2 Existing trace points (all env-gated on `CB_INSTRUMENT_LOG`, strict no-op when unset)

Sinks: `train.cpp:185-198 CbInstrumentLog`, `greedy_tensor_search.cpp:50-57
CbGtsInstrumentLog`, `yetirank_helpers.cpp:24-37`. Emission sites (JSONL, one object per
line, appended):

| `event` | file:line | payload relevant here |
|---|---|---|
| `tree_rng_start` | `train.cpp:292` | `iter`, `fold_count`, `cc` |
| `structure_fold` | `train.cpp:331` | `callcount_before`, `draw_raw`, `taken_fold` |
| `tree_rng_pre_gts` | `train.cpp:366` | `cc` (after the 2 PRE draws) |
| `tree_rng_post_gts` | `train.cpp:381` | `cc` |
| `tree_rng_pre_leaf` | `train.cpp:447` | `cc` |
| `tree_struct` | `train.cpp:495` | per-split `bin_border` (quantized bin index) |
| `leaf_partition` | `train.cpp:508` | `sum_leaf_weights`, `leaf_values_raw` |
| `leaf_indices` | `train.cpp:530` | per-object leaf, averaging-fold permutation |
| `avg_ctr_bins`, `self_consistent_ctr` | `train.cpp:554, 223` | CTR-only |
| `tree_rng_end` | `train.cpp:647` | `cc` |
| `gts_level_rng` | `greedy_tensor_search.cpp:1245` | `depth`, `cc_start`, `calcscores_draws`, `selectbest_draws` |
| `cand_score_rng` | `greedy_tensor_search.cpp:976` | per-candidate `dist`, `stdev`, `draws` |

**`mvs.cpp` has ZERO instrumentation** [VERIFIED: RUN `grep -rn CB_INSTRUMENT_LOG
catboost/private/libs/algo/` → only train.cpp / greedy_tensor_search.cpp /
yetirank_helpers.cpp].

### 4.3 Verified invocation (reproduces the committed trace byte-for-byte)

```bash
D=/tmp/scratch/cbrun; mkdir -p $D
CB_INSTRUMENT_LOG=$D/probe.jsonl \
/home/user/cb_instrumented_build/build/catboost/app/catboost fit \
  --learn-set /home/user/cb_instrumented_build/run/pool.tsv \
  --column-description /home/user/cb_instrumented_build/run/pool.cd \
  --loss-function RMSE -i 3 --depth 2 --learning-rate 0.1 --l2-leaf-reg 3 \
  --random-strength 0 --leaf-estimation-iterations 1 --score-function L2 \
  --leaf-estimation-method Gradient --random-seed 0 --thread-count 1 \
  --bootstrap-type MVS --subsample 0.8 --boost-from-average true \
  -m $D/model.bin --train-dir $D/tdir
```
[VERIFIED: RUN] → 63 lines, `diff probe.jsonl
/home/user/cb_instrumented_build/run/instr_mvs.jsonl` ⇒ **identical**; and
`instr_mvs.jsonl` is byte-identical to the committed
`.planning/plans/bayesian-rng-draw-accounting/instrumented-ground-truth/mvs.jsonl`
[VERIFIED: RUN `diff`]. The dataset is `run/pool.tsv` (1500×4 + label, `pool.cd` =
`0<TAB>Label`), the TSV export of the frozen `inputs/bootstrap_multiblock`.

### 4.4 New trace points, IF the planner wants MVS internals (probably unnecessary)

Not needed for the fix in §5 — the existing `tree_rng_*` / `gts_level_rng` fences already
pin the bootstrap draw count to 1 by arithmetic, and the CPU oracle passes 10/10 after
the fix. If deeper evidence is ever wanted:

1. Copy the 8-line env-gated sink from `train.cpp:185-198` into `mvs.cpp` (add
   `<cstdio>`, `<cstdlib>`, `<mutex>`); name it `CbMvsInstrumentLog`.
2. `mvs.cpp:170`, right after `double lambda = GetLambda(...)`: log
   `{"event":"mvs_lambda","tree":leafValues.size(),"lambda":lambda}` — `leafValues.size()`
   *is* the tree index (`GetLambda` uses `!leafValues.empty()` for the same purpose).
3. `mvs.cpp:174`: fence `rand->GetCallCount()` before/after the `GenRand()` to pin the
   bootstrap draw count **directly** rather than by subtraction.
4. `mvs.cpp:202`, after `CalculateThreshold`: log `blockId`, `blockSize`,
   `SampleRate*blockSize` and `threshold`.
5. `mvs.cpp:203-217`: log the first K `(probability, r, SampleWeights[i])` triples and
   the kept count, plus (critically) the **`f32`** value actually stored.

Rebuild cost: one TU recompile + one relink of the 411 MB static binary via
`/home/user/cb_instrumented_build/build_venv/bin/ninja -C
/home/user/cb_instrumented_build/build catboost/app/catboost`. The 466-step
partial rebuild in `build3.log` completed inside a 46-minute window; a 1-TU change is
dominated by the single `ld.bfd` link step — order of minutes [INFERRED: from
`build3.log` step list and log mtimes].

---

## 5. Prescribed fix

### 5.1 Required (this is the whole bug)

Delete `crates/cb-train/src/bootstrap.rs:413-423` — the `if subsample < 1.0 { … two
rng.gen_rand() … }` block — and rewrite the surrounding doc so it states the verified
contract:

> MVS consumes **exactly one** main-stream draw per `Bootstrap()` call (`mvs.cpp:174`
> `randSeed = rand->GenRand()`). `performRandomChoice = false` sends
> `TCalcScoreFold::Sample` down the `SetControlNoZeroWeighted` branch
> (`calc_score_cache.cpp:742-748`), which never touches `rand`;
> `CalcWeightedData` (`tensor_search_helpers.cpp:442-486`) is draw-free. Verified
> against the instrumented 1.2.10 trace: `tree_rng_pre_gts.cc = 2` → level-0
> `cc_start = 7` = 1 bootstrap draw + 4 RSM draws.

Also update the module-level MVS bullet at `bootstrap.rs:35-39` (it currently advertises
`performRandomChoice = false` without the draw claim — safe, but should cite the trace).

### 5.2 Required test-surface changes

- `crates/cb-train/tests/bootstrap_dev_oracle_test.rs:155` `MVS_GATED_TREES: 2 → 3`, and
  **rewrite** the 30-line doc comment at `:121-154`: its diagnosis ("never trees 0 or 1",
  "the divergence enters when tree 2's sample is drawn from that λ", "Raise this to 3
  once the MVS tree-2 sampling gap is fixed") is superseded. Consider deleting the
  constant entirely and folding `mvs` back into `SCENARIOS`
  (`:112-116`) — that removes the `gated_trees` parameter of `gate_against_upstream`
  (`:163-220`) and the `.chain(std::iter::once((MVS_SCENARIO, MVS_GATED_TREES)))` at
  `:234-237` and `:344-347`.
- `crates/cb-train/src/bootstrap_test.rs` has **no** MVS draw-count or draw-sequence test
  (Bayesian has one at `:60-83`, Bernoulli at `:107-137`, and `No`/MVS-`subsample=1`
  have zero-draw probes at `:141-151`/`:157-166`). The natural Red is a
  **draw-count test**: after `bootstrap(Mvs, ders, 0.8, …, &mut rng)` on a fresh
  `TFastRng64::from_seed(s)`, `rng` must be in the phase of a probe that consumed
  **exactly one** `gen_rand()` — e.g. `let mut probe = TFastRng64::from_seed(s);
  probe.gen_rand(); assert_eq!(rng.gen_rand(), probe.gen_rand());`. This fails today and
  passes after the fix, and it pins the contract at unit level where the oracle cannot.

### 5.3 Recommended (cheap, verified-safe, latent correctness)

`bootstrap.rs:312` — make the block sample-size target reproduce upstream's **float**
expression `SampleRate * blockSize`:
`f64::from((sample_rate as f32) * block_size as f32)`. `block_size ≤ 8192 < 2^24` so the
`f32` cast of the count is exact, making this a faithful transcription of
`float * ui32`. Spiked: all oracles green, 10/10 seeds, residuals unchanged (§8.5).

### 5.4 Optional / defer with rationale

`bootstrap.rs:323` — narrowing the stored weight to `f32` to mirror
`TVector<float> SampleWeights` (`fold.h:217`). Spiked green with **zero** measurable
effect (§8.6). Arguments both ways: exactness vs. adding a narrowing that no oracle can
observe and that would have to be mirrored in `cb-backend`'s inline copies (§6). Suggest
documenting it as a known, deliberate deviation instead of implementing it, unless the
phase's spec demands bit-level transcription.

### 5.5 Must NOT be hand-rolled / changed

- Do **not** reimplement `calculate_threshold` as bisection to "match" the device kernel;
  the CPU function is the upstream-faithful one and the device kernel deliberately
  matches *semantics* (`mvs_device.rs:16-22`).
- Do **not** touch `PRE_TREE_DRAWS` (`boosting.rs:59`), `POST_TREE_EXTRA_DRAWS`
  (`boosting.rs:69`), `replay_grow_draws` (`device_draw_replay.rs:64-85`) or
  `select_level_perturbed`'s draw shape (`tree.rs:596-614, 1141-1218`). All four are
  independently verified by the instrumented trace and by the value-sensitive
  `bootstrap_oracle_bayesian` (3 trees) — changing any of them re-breaks Bayesian.
- Do **not** regenerate `crates/cb-oracle/fixtures/bootstrap/**` or
  `fixtures/inputs/**`. Use `--bootstrap-dev-only` (§8.7) if `bootstrap_dev/` ever needs
  a refresh — it does **not** for this fix (the existing fixtures already prove it).
- All draws must keep going through `cb_core::TFastRng64`; all sums through
  `cb_core::sum_f64` / `scatter_add_f64` (D-08; `scripts/check-no-raw-float-sum.sh`).

---

## 6. Blast radius

### 6.1 Callers of the changed code [VERIFIED: CODEGRAPH + grep]

```
bootstrap()  (crates/cb-train/src/bootstrap.rs:383, pub)
 ├─ mvs_lambda:348 → mean_grad_value:336
 ├─ mvs_sample_weights:281 → calculate_threshold:209, single_probability:192
 └─ callers:
    ├─ boosting.rs:3833  (CPU branch)
    ├─ boosting.rs:3262  (DEVICE branch — same function, Design A)
    └─ bootstrap_test.rs (unit tests)
last_iter_mean_leaf_value (pub) ← boosting.rs:4723
```
`bootstrap` / `EBootstrapType` / `BootstrapResult` / `BAYESIAN_BLOCK_SIZE` /
`MVS_BLOCK_SIZE` are re-exported from `crates/cb-train/src/lib.rs`; `EBootstrapType` is
parsed from Python at `crates/catboost-rs-py/src/params.rs:395`. No signature changes are
needed, so the public surface is untouched.

### 6.2 Does the GPU/device path depend on this? **Yes — through the host `bootstrap()`.**

`boosting.rs:3218-3291` (the `device_active` branch) does `PRE_TREE_DRAWS` →
**`bootstrap(...)` (line 3262, the very same CPU function)** → `grow_tree_on_device` →
`replay_grow_draws` (`:3607`) → `POST_TREE_EXTRA_DRAWS`. The comment at `:3221-3231`
states this explicitly: "the device branch keeps the ENTIRE sampler on the host". So the
fix **moves the device numbers identically to the CPU numbers** — which is the desired
outcome (they must agree) but means every device test must be re-run.

The *device-resident* MVS sampler is **dead code on the live path**:
`crates/cb-backend/src/gpu_runtime/session.rs:1434-1440` only builds `MvsState` when
`config.mvs_lambda.is_some()`, and `cb-train` deliberately never sets it
(`boosting.rs:3172-3181` "STILL NOT WIRED … `mvs_lambda`"; grep shows no `mvs_lambda:
Some(..)` anywhere outside `session.rs`) [VERIFIED: RUN grep]. So
`launch_mvs_weights_resident` (`crates/cb-backend/src/kernels/mvs_device.rs:281`) and
`mvs_sample_kernel` are exercised **only** by the self-oracle
`crates/cb-backend/src/kernels/mvs_device_test.rs`.

### 6.3 Duplicated transcriptions that must stay consistent

`crates/cb-backend/src/kernels/mvs_device_test.rs` contains **verbatim inline copies** of
the CPU sampler (`single_probability:42-50`, `calculate_threshold:55-115`,
`mvs_lambda_iter0:119-127`, `cpu_block_threshold:130-133`, `cpu_mvs_sample:138-172`) —
deliberately, to avoid a `cb-train` dep. `cpu_block_threshold:132` carries the **same**
`sample_rate * block.len() as f64` expression as `bootstrap.rs:312`, and
`cpu_mvs_sample:161` the same full-`f64` weight store. §5.3/§5.4 changes must be mirrored
there or the device self-oracle diverges from the CPU. §5.1 (the draw fix) does **not**
touch these copies, because `cpu_mvs_sample` models only `GenSampleWeights`, not the
`Bootstrap()` wrapper.

### 6.4 Classification

**Must change**
| item | why |
|---|---|
| `crates/cb-train/src/bootstrap.rs:413-423` (+ doc `:35-39`, `:414-420`) | the defect |
| `crates/cb-train/tests/bootstrap_dev_oracle_test.rs:121-155` (+ `:163-220`, `:234-237`, `:344-347` if the carve-out is removed) | `MVS_GATED_TREES 2→3`; the doc encodes the wrong diagnosis |
| `crates/cb-train/src/bootstrap_test.rs` | add the MVS draw-count Red (no coverage today) |

**May change**
| item | why |
|---|---|
| `bootstrap.rs:312` | §5.3 recommended f32 `sampleSize` |
| `bootstrap.rs:323`, `mvs_device_test.rs:132,161` | §5.4 optional f32 weight narrowing (mirror required) |
| `bootstrap.rs:336-342` | doc-only note that upstream uses a 125-block reduction |

**Verification only (no edit expected)**
- `crates/cb-train/tests/bootstrap_oracle_test.rs` — frozen `bootstrap/**` family; must
  stay green (it does).
- `crates/cb-train/tests/device_bootstrap_parity_test.rs` — device-vs-**CPU**, so both
  sides move together; expected green (rocm/cuda only).
- `crates/cb-train/tests/device_bootstrap_speed_test.rs`,
  `device_oblivious_parity_probe_test.rs`, `device_seam_test.rs`,
  `device_nonsym_fit_test.rs`, `device_region_fit_test.rs` (rocm).
- `crates/cb-backend/src/kernels/mvs_device_test.rs` (rocm) — untouched unless §5.4.
- `crates/cb-train/tests/multidim_sampling_regression_test.rs:131-144` — MVS smoke, no
  numeric expectation.
- `crates/cb-train/tests/regularization_oracle_test.rs` — Bayesian/random_strength draw
  path; must not move.
- `crates/cb-oracle/fixtures/bootstrap/**`, `fixtures/bootstrap_dev/**`,
  `fixtures/inputs/**` — byte-unchanged.

**Explicitly out of scope**
- Wiring the device-resident MVS sampler (`config.mvs_lambda`) — Design B′, separate perf
  phase.
- `mvs_reg` / `TMaybe<float> Lambda` support, Ordered-boosting MVS
  (`mvs.cpp:136-168` tail-derivative copy), group-unit sampling (upstream rejects),
  pairwise MVS (upstream skips the sampler), Poisson on CPU.
- Multi-dimension MVS exactness (§2.7) and learn-weight multiplication (§2.7).

---

## 7. Risks and pitfalls

| # | trigger | consequence | prevention | verification |
|---|---|---|---|---|
| R1 | Comparing our fit against upstream fixtures generated under a *different* config while reusing one border set | Bogus divergence/agreement — CatBoost quantization borders shift with unrelated params (observed varying with `subsample`) | Always `load_model_json(that config).float_feature_borders()` per config, as `bootstrap_dev_oracle_test.rs:240-241` and `bootstrap_oracle_test.rs:60-62` already do | any multi-config comparison harness |
| R2 | "Fixing" the draw count by re-tuning `POST_TREE_EXTRA_DRAWS` or the RSM/normal replay instead of deleting the 2 MVS draws | Breaks `bootstrap_oracle_bayesian` (the value-sensitive 3-tree lock) and every YetiRank RNG oracle | The MVS arm is the *only* place with a wrong count; leave the shared accounting alone | `cargo test -p cb-train --test bootstrap_oracle_test` + `yetirank_pairwise_tree_rng_oracle_test` |
| R3 | Editing `calculate_threshold` / `mvs_sample_weights` without mirroring `mvs_device_test.rs:55-172` | device MVS self-oracle fails on rocm only — invisible on the default CPU CI | Mirror or don't touch | rocm run of `cb-backend --lib` |
| R4 | An all-zero-gradient block (perfect fit / empty leaf) | ours returns `p = 0` ⇒ weight 0 ⇒ every object dropped ⇒ `CB_ENSURE(sampledDocs->GetDocCount() > 0)` analogue; upstream would produce `inf`/`NaN` | Out of scope, but do not "harmonise" to upstream's UB | no fixture exercises it |
| R5 | Assuming `thread_count=1` makes upstream reductions flat | `CalculateMeanGradValue` blocks by `CB_THREAD_LIMIT=128`, not by thread count | Note it; don't try to match it unless a tree-0 λ oracle demands it | §1.5 |
| R6 | Chasing pre-existing reds | wasted cycles | see §8.8 | — |
| R7 | Believing `MvsState` / `launch_mvs_weights_resident` participates in a real fit | wrong blast-radius model, wrong perf conclusions | `config.mvs_lambda` is never `Some` from `cb-train` | §6.2 |
| R8 | The residual `max|Δleaf| ≈ 6e-9` / `max|Δstaged| ≈ 2e-8` being read as "still broken" | needless scope creep | That residual is present in **all** four bootstrap arms and is the ordinary leaf/border arithmetic floor, **not** sampler-related — it is byte-identical with and without §5.3/§5.4 | §8.5, §8.6 |

---

## 8. Reproduction, verification, and the exact commands used

### 8.1 Red (reproduce the bug, no fixture regeneration)

```bash
# crates/cb-train/tests/bootstrap_dev_oracle_test.rs:155 -> MVS_GATED_TREES = 3
cargo test -p cb-train --test bootstrap_dev_oracle_test
```
[VERIFIED: RUN] →
```
[cpu] bootstrap_dev/no|bayesian|bernoulli: ... 3/3 trees   (ok)
panicked ... [cpu] bootstrap_dev/mvs: splits diverged from upstream:
  StageDiverged { stage: Splits, index: 5,
                  expected: -0.025514747947454453,
                  actual: -0.2692405581474304, diff: 0.2437258101999760 }
```
This is a complete, self-contained TDD Red against **already-committed** fixtures — no
Python, no new fixture, no GPU.

A second, cheaper Red (unit level, §5.2): assert `bootstrap(Mvs, …, subsample=0.8, …)`
advances the RNG by exactly one `gen_rand`.

### 8.2 Green + generalisation (the fix)

Patch applied for the spike: `bootstrap.rs:413` `if subsample < 1.0 {` → `if false {`
(equivalent to deleting the block).

```bash
cargo test -p cb-train --test bootstrap_dev_oracle_test --test bootstrap_oracle_test
```
[VERIFIED: RUN] → `bootstrap_dev_cpu_matches_upstream ok` (with `MVS_GATED_TREES = 3`)
and all 5 `bootstrap_oracle_*` ok.

Multi-seed × bias probe (throwaway upstream fixtures per §8.7, throwaway
`crates/cb-train/examples/mvs_seed_probe.rs`, both deleted afterwards):

| | before the fix | after the fix |
|---|---|---|
| bias=true seed 0 | ok | ok |
| bias=true seed 1 | **fail, first_bad_split=4** | ok |
| bias=true seed 2 | ok | ok |
| bias=true seed 3 | ok | ok |
| bias=true seed 4 | **fail, first_bad_split=3** | ok |
| bias=false seeds 0..4 | **fail, first_bad_split = 5,4,5,4,4** | ok (all) |
| totals | **3/5 and 0/5** | **5/5 and 5/5** |

After the fix, every config: `max|Δleaf| ∈ [5.9e-9, 6.9e-9]`,
`max|Δstaged| ∈ [1.6e-8, 2.4e-8]` [VERIFIED: RUN]. The "before" column reproduces the
task's recorded measurement exactly (bias=true passes on seeds 0, 2, 3).

### 8.3 No regression

```bash
cargo test -p cb-train --no-fail-fast
```
[VERIFIED: RUN] Baseline at HEAD: **503 passed**, 1 failed
(`monotone_non_symmetric_and_region_are_typed_errors`,
`crates/cb-train/tests/monotone_oracle_test.rs:286`). With the fix applied:
**503 passed**, same single failure. Identical.

### 8.4 The `sampleSize` narrowing, quantified

```
blockSize  upstream float(0.8f*bs)   ours f64(0.8f)*bs        ours-upstream
1500       1200.0                    1200.0000178813934       +1.788e-05
8192       6553.60009765625          6553.60009765625          0
3616       2892.800048828125         2892.800043106079        -5.722e-06
20000      16000.0                   16000.00023841858        +2.384e-04
```
[VERIFIED: RUN numpy]. Note the 8192 case is exact (scaling by a power of two), so a
*full* 8192-block is unaffected; only partial blocks and small `n` diverge.

### 8.5 / 8.6 Optional-fix spikes

- **A + B** (draw fix + f32 `sampleSize` at `bootstrap.rs:312`): all 5
  `bootstrap_oracle_*` ok, `bootstrap_dev` ok at 3 trees, probe **5/5 and 5/5**,
  residuals **byte-identical** to A alone [VERIFIED: RUN].
- **A + B + C** (additionally narrowing the stored weight to f32 at `bootstrap.rs:323`):
  identical results again — every printed residual unchanged [VERIFIED: RUN]. Confirms
  MVS weights influence the model only via the discrete split argmax.

### 8.7 Throwaway upstream fixtures (never touch the frozen ones)

`catboost==1.2.10` is importable both from system python 3.14.6 and `./.venv`
[VERIFIED: RUN `python3 -c "import catboost;print(catboost.__version__)"` → `1.2.10`].
The generator's convention is a dedicated `--<name>-only` entrypoint
(`crates/cb-oracle/generator/gen_fixtures.py:3352-3382`); `gen_bootstrap_dev_only()`
(`:951-955`) regenerates **only** `bootstrap_dev/`, and `gen_bootstrap()` (`:710`) is
reachable only from `main()`. For research/TDD, write to a scratch dir instead:

```python
# scratch script — writes NOTHING into crates/
x = np.load("crates/cb-oracle/fixtures/inputs/bootstrap_multiblock/X.npy")
y = np.load("crates/cb-oracle/fixtures/inputs/bootstrap_multiblock/y.npy")
BASE = dict(iterations=3, learning_rate=0.1, depth=2, l2_leaf_reg=3.0,
            bootstrap_type="MVS", subsample=0.8, random_strength=0,
            leaf_estimation_iterations=1, score_function="L2",
            leaf_estimation_method="Gradient", thread_count=1, verbose=False)
for bias in (True, False):
    for seed in range(5):
        m = CatBoostRegressor(random_seed=seed, boost_from_average=bias, **BASE)
        m.fit(x, y); m.save_model(f"{OUT}/bias{int(bias)}_seed{seed}/model.json",
                                  format="json")
        # + np.save staged from m.staged_predict(x)
```
[VERIFIED: RUN — this exact script produced the 10 fixture dirs used in §8.2]. Mirrors
`gen_bootstrap_dev()` (`gen_fixtures.py:858-949`) knob for knob (every default catboost's
raw dict API differs on — notably `random_strength=0` — is pinned on both sides).

If the phase decides to *commit* a multi-seed MVS family, follow the existing pattern: a
new `gen_mvs_seeds()` + `--mvs-seeds-only` entrypoint writing a new
`fixtures/mvs_seeds/…` directory, leaving `bootstrap/` and `bootstrap_dev/` alone.

### 8.8 Verification commands and known pre-existing reds

CPU (default `cpu` feature):
```bash
cargo test -p cb-train --test bootstrap_oracle_test          # 5 tests, must be green
cargo test -p cb-train --test bootstrap_dev_oracle_test      # 1 test, must be green at MVS_GATED_TREES=3
cargo test -p cb-train bootstrap                             # unit tests in bootstrap_test.rs
cargo test -p cb-train --no-fail-fast                        # 503 pass / 1 known red
cargo test -p cb-backend --lib --no-fail-fast                # 173 pass / 60 known reds
cargo clippy -p cb-train --all-targets
cargo clippy -p cb-backend --all-targets                     # 4 lib + 2 lib-test known errors
bash scripts/check-source-test-separation.sh
bash scripts/check-no-raw-float-sum.sh
bash scripts/check-no-anyhow.sh
```
Device (project convention: **always** `--no-default-features --features rocm`, **always**
`--test <target>`; taken from the prior phase's verified plan
[PROJECT: `.planning/plans/device-bootstrap-parity/plan10.md:84-117`, `plan2.md:98-127`,
`plan6.md:116-149`]):
```bash
cargo test -p cb-train --no-default-features --features rocm --test device_bootstrap_parity_test -- --nocapture --test-threads 1
cargo test -p cb-train --no-default-features --features rocm --test bootstrap_dev_oracle_test -- --nocapture --test-threads 1
cargo test -p cb-train --no-default-features --features rocm --test device_oblivious_parity_probe_test -- --nocapture
cargo test -p cb-train --no-default-features --features rocm --test device_seam_test
cargo test -p cb-backend --no-default-features --features rocm --lib -- --test-threads 1
cargo clippy -p cb-backend --no-default-features --features rocm --all-targets --no-deps
```
(substitute `cuda` for `cuda` hosts; `--features wgpu` is build-only in CI,
`.github/workflows/ci.yml:31-34`.)

**Known pre-existing reds — do NOT chase** [all VERIFIED: RUN at HEAD `2c14d7f`]:
1. `cb-train`: `monotone_non_symmetric_and_region_are_typed_errors`
   (`crates/cb-train/tests/monotone_oracle_test.rs:286`). 503 pass / 1 fail.
2. `cb-backend --lib` under the default `cpu` feature: **60 failures / 173 passes**
   (338 s), the CubeCL-CPU MLIR limitation (`plane_inclusive_sum` unsupported).
3. `cargo clippy -p cb-backend --all-targets`: **4 errors in the lib target** —
   `kernels/exact_quantile.rs:178:8` (indexing may panic),
   `kernels/bootstrap_device.rs:230:28` (approximate value of `f32::consts::LN_2`),
   `cpu_runtime.rs:696:13` (indexing may panic), `cpu_runtime.rs:1025:29` (slicing may
   panic) — **plus 2 more in the lib *test* target** the task did not list:
   `kernels/gradient.rs:18:18` (approximate `LN_2`) and `kernels/score_split.rs:374:16`
   ("this operation will always return zero"). Restrict to `--lib` to see exactly 4.
4. `cb-backend` build warnings: `kernels.rs:645:30` and `:673:38`
   `float_literal_f32_fallback` (future-incompat, not errors).

---

## 9. Planning guidance

**Ordering**

1. **Red-1 (unit):** MVS draw-count test in `crates/cb-train/src/bootstrap_test.rs`.
2. **Red-2 (oracle):** `MVS_GATED_TREES = 3` in
   `crates/cb-train/tests/bootstrap_dev_oracle_test.rs` → fails at split index 5.
3. **Green:** delete `bootstrap.rs:413-423`; rewrite the two doc blocks with the
   trace-derived contract and the `.../instrumented-ground-truth/mvs.jsonl` citation.
4. **Doc debt:** rewrite `MVS_GATED_TREES`'s comment (or remove the carve-out entirely
   and fold `mvs` into `SCENARIOS`).
5. **Optional:** §5.3 f32 `sampleSize` (+ mirror in `mvs_device_test.rs:132` if taken).
6. **Regression:** `cargo test -p cb-train --no-fail-fast` (expect 503/1) and
   `cargo clippy -p cb-train --all-targets`.
7. **Device re-verification** on the rocm rig (§8.8 device list). Steps 1–6 are
   backend-independent and can land without a GPU; step 7 is the only device-gated gate.

**Dependencies between tasks**

- Red-2 requires no fixture work — `bootstrap_dev/mvs` is already committed
  (`2c14d7f`) and is *already* the bias-0, worst-case config.
- The optional multi-seed committed fixture family (§8.7) is independent of the fix and
  can be deferred; the throwaway probe already establishes 10/10.

**Decisions to preserve**

- One draw per MVS `Bootstrap()`; two for `PRE_TREE_DRAWS`; two for
  `POST_TREE_EXTRA_DRAWS`; `n_features` RSM + 1 `CalcScores` + `n_features` `std_normal`
  per level, border-less features included. All four verified; do not "simplify".
- Design A (host-side sampler shared by the device branch) — the fix depends on it and
  keeps device and CPU numerically identical by construction.
- `sum_f64` / `TFastRng64` exclusivity; source/test separation (no `mod tests` inline).

**No spike or user decision is required.** The root cause is proven and the fix is a
deletion; §5.3/§5.4 are the only judgement calls, both spiked safe.

---

## 10. Open questions (none blocking)

1. Should §5.4 (f32 `SampleWeights` narrowing) be implemented for transcription
   exactness, given it is provably unobservable here and must be mirrored in
   `cb-backend`'s inline copies? — a spec/style call, not a correctness one.
2. Should a committed multi-seed MVS oracle family be added (§8.7) so a future phase
   cannot regress the sampler on the 7 configurations that were silently broken? —
   recommended, but scope.
3. `CalculateMeanGradValue`'s 125-block reduction order (§1.5) is unmatched. No oracle
   can currently see it (tree-0 λ only). Leave as a documented deviation? [UNVERIFIED:
   whether any realistic dataset makes it observable.]
4. Learn-weight multiplication into `SampleWeights` (§2.7) is unimplemented; no weighted
   MVS fixture exists to prove or disprove a gap.

---

## 11. Sources

**Local repository** (all read this session)
`crates/cb-train/src/bootstrap.rs` (full), `bootstrap_test.rs` (full),
`crates/cb-train/src/boosting.rs:42-69, 3150-3310, 3599-3612, 3780-4000, 4080-4180,
4690-4730`, `crates/cb-train/src/tree.rs:574-654, 1130-1224`,
`crates/cb-train/src/device_draw_replay.rs` (full),
`crates/cb-train/tests/bootstrap_oracle_test.rs` (full),
`crates/cb-train/tests/bootstrap_dev_oracle_test.rs` (full),
`crates/cb-train/tests/device_bootstrap_parity_test.rs:255-440`,
`crates/cb-train/tests/multidim_sampling_regression_test.rs:90-175`,
`crates/cb-backend/src/kernels/mvs_device.rs:1-130, 264-347`,
`crates/cb-backend/src/kernels/mvs_device_test.rs:1-300`,
`crates/cb-backend/src/gpu_runtime/session.rs` (grep: `mvs`),
`crates/cb-core/src/reduction.rs:32-58`, `crates/cb-core/src/rng.rs:232-235`,
`crates/cb-core/src/normal.rs:50-64`,
`crates/cb-oracle/generator/gen_fixtures.py:120-175, 700-830, 858-975, 3352-3382`,
`crates/cb-oracle/src/compare.rs:10-120`, `crates/cb-backend/Cargo.toml`,
`crates/cb-train/Cargo.toml`, `.github/workflows/ci.yml`, `CLAUDE.md`.

**Project planning documents**
`.planning/plans/bayesian-rng-draw-accounting/instrumented-ground-truth/GROUND_TRUTH.md`
(full) and `mvs.jsonl`; `.planning/plans/device-bootstrap-parity/plan2.md`,
`plan6.md`, `plan10.md` (verified command conventions).

**Upstream source of record** — `/home/user/cb_instrumented_build/catboost-src/catboost/`
`private/libs/algo/mvs.cpp` (full), `mvs.h` (full),
`private/libs/algo/tensor_search_helpers.cpp:442-566`,
`private/libs/algo/calc_score_cache.cpp:245-270, 725-830, 1170-1215`,
`private/libs/algo/calc_score_cache.h:50-70, 242-314, 405`,
`private/libs/algo/fold.h:217`, `private/libs/algo/fold.cpp:119, 246`,
`private/libs/algo/greedy_tensor_search.cpp:40-58, 329-400, 755-800, 960-990,
1195-1260, 1950-1975`, `private/libs/algo/train.cpp:180-300, 430-560, 630-660`,
`private/libs/algo_helpers/approx_updater_helpers.cpp:8-38` and `.h:145-150`,
`private/libs/options/restrictions.h:59`,
`library/cpp/threading/local_executor/local_executor.h:70-115, 145-161`.

**Commands run** (selection; all exit codes/outputs quoted above)
`git status --short`, `git log --oneline -3`,
`cargo test -p cb-train --test bootstrap_oracle_test --test bootstrap_dev_oracle_test`,
`cargo test -p cb-train --no-fail-fast`,
`cargo test -p cb-backend --lib --no-fail-fast`,
`cargo clippy -p cb-backend --all-targets`,
`cargo run -p cb-train --example mvs_seed_probe` (throwaway, deleted),
`.venv/bin/python <throwaway fixture generator>`,
`CB_INSTRUMENT_LOG=… catboost fit …` (§4.3),
`ninja -n -C /home/user/cb_instrumented_build/build catboost/app/catboost`,
`python3 -c "…numpy float vs double sampleSize…"`.

**Not used**: Context7 CLI (no third-party library question arose — the only
"dependency" here is the vendored upstream C++ source, which is present locally and is a
higher-priority source than any docs site). No web research was required; every claim is
backed by local source, local execution, or the instrumented upstream binary.

---

## 12. Confidence assessment

**HIGH** (directly verified by execution or by reading the source of record)
1. Root cause = the 2 fabricated MVS draws (`bootstrap.rs:413-423`).
2. Upstream MVS `Bootstrap` consumes exactly 1 main-stream draw.
3. Deleting them yields 5/5 + 5/5 across seeds 0–4 × both bias settings, keeps the
   frozen `bootstrap/**` oracles green, and leaves `cb-train` at 503/1.
4. The `bootstrap_dev_oracle_test` Red at `MVS_GATED_TREES = 3`, split index 5.
5. First-divergence index set `{3,4,5}` — including a **tree-1** case, correcting the
   "never trees 0 or 1" note.
6. The instrumented build is compiled, up to date, and reproduces `mvs.jsonl` byte-exactly
   with the §4.3 command; `mvs.cpp` has no trace points.
7. `sampleSize` is a `float` expression upstream; the 1500-block delta is +1.788e-5.
8. `fold->SampleWeights` is `TVector<float>`.
9. `CalculateThreshold` is a partition quickselect with no tolerance/iteration cap;
   `CalculateMeanGradValue` uses a `CB_THREAD_LIMIT = 128` blocked reduction.
10. `sum_f64` is an uncompensated left-to-right fold ≡ `Accumulate`.
11. The device branch calls the same host `bootstrap()`; the device-resident MVS sampler
    is unreachable from `cb-train`.
12. Stored `LeafValues` are learning-rate-scaled, so the λ carry is correct.
13. All pre-existing reds and their exact counts/locations (503/1; 173/60; 4 lib +
    2 lib-test clippy errors).
14. `catboost==1.2.10` importable; `--bootstrap-dev-only` is the only bootstrap-family
    generator entrypoint reachable without touching frozen fixtures.

**MEDIUM** (reasoned from verified facts, not directly exercised)
1. The `boost_from_average=false` sensitivity mechanism (single-signed, wide-spread first
   gradients ⇒ more heterogeneous keep-probabilities ⇒ subset choice dominates).
2. That the drift compounds non-uniformly because `std_normal`'s draw count is
   phase dependent.
3. Instrumented-build relink cost "order of minutes" for a 1-TU change.
4. That the rocm device tests will pass unchanged after the fix (device-vs-CPU
   comparisons move together; device-vs-upstream improves). **Not run — no GPU run was
   performed in this session.**

**LOW** (unresolved / needs validation)
1. Whether `std::partition`'s permutation-order difference in `calculate_threshold`
   (§2.2) could ever be observable on some dataset. Believed ulp-level.
2. Whether the unmatched `CalculateMeanGradValue` block order, multi-dim MVS
   aggregation, or learn-weight multiplication is observable anywhere. No fixture.
3. Whether §5.4 (f32 weights) should be adopted — a spec judgement, no evidence either
   way beyond "no measurable effect here".
