# Perf-critical parameter wave — upstream research

Scope (user-selected, 2026-08-15): `rsm`, `thread_count`, and the Langevin trio
(`langevin` / `diffusion_temperature` / `posterior_sampling`). Deep CPU + GPU
coverage with oracle tests and measured benchmarks, rather than broad-but-shallow.

Sources: the `v1.2.10` tag at github.com/catboost/catboost (fetched via `gh api`,
never the stale `catboost-master/` stub — see CLAUDE.md's correction block) plus
empirical probes against the installed `catboost==1.2.10` wheel.

---

## `rsm` (alias `colsample_bylevel`)

**Legal range.** `(0, 1]`. `0`, negative and `> 1` all raise
`oblivious_tree_options.cpp:125: Rsm should be in (0, 1]` (measured).

**`rsm = 1.0` is bit-identical to leaving it unset** (measured, max|diff| = 0),
so the default path stays byte-identical — the draw still happens but
`GenRandReal1() <= 1.0` always holds. NOTE: the draws are still CONSUMED at
`rsm = 1.0`, so a naive "skip the loop when rsm == 1" shortcut is only safe
because upstream's own default also takes the same number of draws... it does
NOT: see "Draw accounting" below.

**Accepted for every grow policy on CPU** (SymmetricTree / Depthwise /
Lossguide all measured OK at rsm ∈ {1.0, 0.5, 0.3}).

**Algorithm** (`greedy_tensor_search.cpp`). Per DEPTH LEVEL — hence
"colsample_by*level*" — `GreedyTensorSearch`'s depth loop calls
`SelectFeaturesForScoring` FIRST, before the per-level bootstrap and before
scoring:

```
for curDepth in 0 .. MaxDepth:
    candidatesContexts = SelectFeaturesForScoring(...)   // <-- rsm draws here
    if (!isSamplingPerTree) DoBootstrap(...)             // per-LEVEL bootstrap
    CalcScores(...)                                      // random_strength draws
```

`SelectDatasetFeaturesForScoring` (line 973) builds the candidate list then
filters it:

```
AddFloatFeatures(...)      // one candidate sublist per available float feature, feature order
AddOneHotFeatures(...)     // then the one-hot cat features
CompressCandidates(...)    // binary features get packed into TBinaryFeaturesPack
SelectCandidatesAndCleanupStatsFromPrevTree(...)   // <-- the rsm filter
AddSimpleCtrs(...)         // one draw per HIGH-CARDINALITY cat feature
AddTreeCtrs(...)           // one draw per (baseProj, non-one-hot cat feature)
```

The filter itself (`SelectCandidatesAndCleanupStatsFromPrevTree`, line 311), for
the `OneFeature` ensemble type this port produces:

```cpp
const double rsm = ctx->Params.ObliviousTreeOptions->Rsm;
auto& rand = ctx->LearnProgress->Rand;          // the PERSISTENT learn stream
for (auto& candSubList : candList) {
    addCandSubListToResult = rand.GenRandReal1() <= rsm;   // ONE draw per sublist
    ...
}
```

**Draw accounting (the parity-critical part).** For a float-only dataset with no
binary-packed features, that is exactly `n_float_features` `GenRandReal1()`
draws per depth level, in feature-index order, taken from the same persistent
learn RNG that feeds `random_strength` and the bootstrap. The two estimated-feature
`SelectDatasetFeaturesForScoring` calls return early on a null `learnData`, so
they take NO draws.

Landmine: `CompressCandidates` packs features that have a SINGLE border into
`TBinaryFeaturesPack`s, and the packed case draws once per BIT rather than once
per sublist. Fixtures must therefore use features with > 1 border (continuous
data at the default `border_count`) so every candidate stays `OneFeature`.

**Seeded by `random_seed`** (measured: seeds 42 and 43 give different models).

`colsample_bylevel` is bit-identical to `rsm` (measured, max|diff| = 0).

Elsewhere: `CheckSubtractTrickAllowed` (line 1462) disables the subtraction
trick unless `Rsm == 1.0f` — a pure optimization gate, no numeric effect.

---

## `thread_count`

**Numerically INERT.** Measured over `thread_count` ∈ {1, 2, 4, 8, 16, -1}:
max|diff| = `0.000e+00` against `thread_count = 1` in every case, and also with
`rsm = 0.5` active (so the feature draw is thread-independent too).

That invariance is structural, not incidental: every parallelised upstream loop
blocks on a CONSTANT `CB_THREAD_LIMIT = 128` block size
(`private/libs/options/restrictions.h:59`) via
`TSimpleIndexRangesGenerator(fullRange, blockSize)`, never on the runtime thread
count. So the RNG stream and the reduction order are fixed regardless of how many
threads execute the blocks.

**Consequence for this port:** `thread_count` is a pure performance knob and its
oracle is an INVARIANCE test (the model must not move), plus a benchmark. Any
implementation that changes results across thread counts is wrong by
construction.

Prior wave's headroom measurement (see the `string-param-wave-progress` memory):
our CPU `GreedyLogSum` baseline was 0.62 s vs official 0.17 s, but official at
`thread_count=1` is 0.58 s — i.e. essentially the entire 3.6x gap is threading.

---

## Langevin: `langevin` / `diffusion_temperature` / `posterior_sampling`

Source: `catboost/private/libs/algo_helpers/langevin_utils.{h,cpp}` @ v1.2.10.

### Noise rate

```cpp
double CalcLangevinNoiseRate(float diffusionTemperature, float learningRate) {
    return sqrt(2.0 / learningRate / diffusionTemperature);
}
```

Independently confirmed empirically BEFORE reading the source: rms(leaf delta) /
`sqrt(2/(dt*lr))` is constant at 0.033592 across dt ∈ {1, 4, 100, 10⁴} and at
0.11197 across lr ∈ {0.1, 0.3, 0.9}. Note the noise DECREASES with rising
temperature.

`diffusionTemperature == 0.0f` returns early from every entry point — no noise,
and (critically) no RNG draws.

### Two injection points

**(1) Object derivatives** — `greedy_tensor_search.cpp:760`, before the tree is
grown, so the noise perturbs SPLIT SCORING:

```cpp
if (ctx->Params.BoostingOptions->Langevin) {
    for (auto& bodyTail : fold->BodyTailArr) {
        AddLangevinNoiseToDerivatives(
            DiffusionTemperature, LearningRate,
            ctx->LearnProgress->Rand.GenRand(),   // ONE main-stream draw per body/tail
            &bodyTail.WeightedDerivatives, LocalExecutor);
    }
}
```

with the block-seeded stream:

```cpp
TSimpleIndexRangesGenerator rangesGenerator(TIndexRange(objectCount), CB_THREAD_LIMIT);
// block b covers [b*128, min((b+1)*128, n)); count = ceil(n / 128)
localExecutor->ExecRange([&](int blockIdx) {
    TFastRng64 blockRng(randomSeed + blockIdx);
    for (auto idx : rangesGenerator.GetRange(blockIdx).Iter())
        dersData[idx] += coef * StdNormalDistribution<double>(blockRng);
}, ...);
```

**The seed is per BLOCK of 128, not per element**, and within a block ONE rng is
drawn from sequentially in index order. `CB_THREAD_LIMIT` is a compile-time
constant block SIZE, which is exactly why the result is thread-count independent.

> ⚠️ The already-landed device kernel
> (`cb-backend/src/kernels/langevin.rs`) reseeds PER ELEMENT
> (`from_seed(rand_seed + i).advance(10)`). That is a DIFFERENT stream from
> upstream's per-block-of-128 sequential draw, and it also carries an
> `advance(10)` that `AddLangevinNoiseToDerivatives` does not do. The kernel's
> self-oracle only checks it against a CPU replica of its own rule, so the
> divergence is invisible there. Reconcile before claiming device parity.

**(2) Leaf derivative sums** — `approx_calcer.cpp:768`, inside `leafUpdaterFunc`
(so once per leaf-estimation iteration), NOT guarded by `if (Langevin)` because
the helper early-returns at `dt == 0`:

```cpp
const double scaledL2Regularizer = ScaleL2Reg(L2Reg, fold.GetSumWeight(), fold.GetLearnSampleCount());
if (estimationMethod == Gradient)
    AddLangevinNoiseToLeafDerivativesSum(dt, lr, scaledL2Regularizer, randomSeed, &leafDers);
else if (estimationMethod == Newton)
    AddLangevinNoiseToLeafNewtonSum(dt, lr, scaledL2Regularizer, randomSeed, &leafDers);
```

```cpp
TFastRng64 rng(randomSeed);                       // ONE rng for ALL leaves
for (TSum& sum : *leafDersSum) {
    if (sum.SumWeights < 1e-9) continue;          // skipped leaves take NO draw
    double scaledCoef = coef * sqrt(sum.SumWeights + scaledL2Regularizer);   // Gradient
    // Newton variant: coef * sqrt(fabs(sum.SumDer2) + scaledL2Regularizer)
    sum.SumDer += scaledCoef * StdNormalDistribution<double>(rng);
}
```

`randomSeed` here is `CalcApproxDeltaSimple`'s own parameter — the SAME value is
reused for `CalcLeafDersSimple` and for the noise, and it does not advance
between leaf-estimation iterations.

### The option-resolution rules (all measured against the wheel)

| supplied | resolves to |
|---|---|
| `langevin=True` (alone) | `diffusion_temperature = 10000`, `model_shrink_rate = 0.001` |
| `diffusion_temperature = X` (langevin unset) | `langevin = True` — **setting the temperature implicitly turns Langevin ON** |
| `langevin=False, diffusion_temperature=X` | Langevin stays OFF and the fit is bit-identical to the default (max\|diff\| = 0) |
| `posterior_sampling=True` | `langevin = True`, `diffusion_temperature = n_learn`, `model_shrink_rate = 1/(2·n_learn)`, `model_shrink_mode = Constant` |
| `posterior_sampling=True, langevin=False` | RAISES `catboost_options.cpp:746: Posterior Sampling requires Langevin boosting.` |

Verified for `posterior_sampling` at n ∈ {50, 200, 777}: temperature = n exactly and
shrink rate = 1/(2n) exactly.

**`langevin=True` implies `model_shrink_rate = 0.001`, and that is the whole
explanation for the `dt = 0` case being non-identical to the default.** Proven:
replaying `model_shrink_rate=0.001` alone reproduces `langevin=True,
diffusion_temperature=0` at max|diff| = `0.0`. Model shrinkage already ships in
this port (`ExtraBoostParams::model_shrink_rate`), so the trio must set it as a
DEFAULT that an explicit user value still overrides.

`posterior_sampling` does NOT refuse `leaf_estimation_method=Newton`,
`model_shrink_rate=0.5`, `bootstrap_type=Bayesian` or `boosting_type=Ordered`
(all measured accepted) — the only refusal is `langevin=False`.
