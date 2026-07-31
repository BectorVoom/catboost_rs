# One-hot RSM draw-order ground truth (SPEC-OH-27 / T01a)

## Verdict

```
RSM_RULE: n_float + n_one_hot
```

**Method: SOURCE-DERIVED from the vendored upstream CatBoost 1.2.10 tree
(`catboost-master/`), NOT from an instrumented binary run.** See "Evidence
grade" below — this distinction is load-bearing and must not be dropped when
citing this file.

One `GenRandReal1()` draw is consumed per *candidate sub-list* of ensemble type
`OneFeature`, and one-hot-eligible categorical features each contribute exactly
one such sub-list — the same as a float feature. The draw is consumed
**unconditionally, even at the default `rsm = 1.0`**, so it affects RNG stream
alignment in every configuration, not only when feature sampling is enabled.

## The evidence chain (three sites, all in
`catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp`)

**1. One-hot candidates enter the list before selection.**
`SelectDatasetFeaturesForScoring` (`:1000`) calls, in order:

```
AddFloatFeatures(...)                              // :1020
AddOneHotFeatures(*learnData, ctx, &candidatesContext.CandidateList);   // :1021
CompressCandidates(...)                            // :1022
SelectCandidatesAndCleanupStatsFromPrevTree(...)   // :1023  <- the draw site
```

**2. `AddOneHotFeatures` adds ONE candidate per CAT FEATURE, not per value**, and
its eligibility predicate is character-for-character our `route_categorical`:

```cpp
auto onLearnOnlyCount = quantizedFeaturesInfo.GetUniqueValuesCounts(catFeatureIdx).OnLearnOnly;
if ((onLearnOnlyCount > oneHotMaxSize) || (onLearnOnlyCount <= 1)) {
    return;                      // skip: CTR-routed, or constant
}
...
TSplitEnsemble splitEnsemble(std::move(splitCandidate));   // single candidate => OneFeature
candList->emplace_back(std::move(candidate));              // exactly one sub-list
```

i.e. eligible iff `1 < cardinality <= one_hot_max_size` — identical to
`crates/cb-train/src/candidates.rs:92-104`'s `EncodingPath::OneHot` arm.

**3. The draw site charges one draw per `OneFeature` sub-list, unconditionally:**

```cpp
const double rsm = ctx->Params.ObliviousTreeOptions->Rsm;
auto& rand = ctx->LearnProgress->Rand;
for (auto& candSubList : candList) {
    switch (splitEnsemble.Type) {
        case ESplitEnsembleType::OneFeature:
            addCandSubListToResult = rand.GenRandReal1() <= rsm;   // ALWAYS draws
            break;
        ...
```

Note `GenRandReal1()` is called before the `<= rsm` comparison, so `rsm = 1.0`
(the default) still consumes the draw. This matches our own unconditional loop at
`crates/cb-train/src/tree.rs:610-614`
(`for _ in 0..matrix.n_features() { p.rng.gen_rand_real1(); }`) — which today
counts **float features only** (`n_features()` is float-only, `tree.rs:355-357`;
`n_cat_features()` is separate at `:361-363`).

## What this means for the implementation

Our current per-level RSM loop under-draws by exactly the number of one-hot-eligible
categorical columns. Once one-hot columns become candidates (T16/T17/T18), the loop
bound must become `n_float + n_one_hot_eligible`, or every subsequent tree's
bootstrap sample desynchronises — the same defect class as the two fabricated MVS
draws fixed in `d7676b5`.

## CAVEAT — the compression path is NOT covered by this derivation

`CompressCandidates` (`:1022`) runs **between** `AddOneHotFeatures` and the draw
site, and can re-bundle `OneFeature` candidates into `BinarySplits`,
`ExclusiveBundle`, or `FeaturesGroup` ensembles. Those arms have *different* draw
arithmetic — `BinarySplits` draws once per set bit in the pack mask, the bundle and
group arms once per selected feature in the bundle/group — so a configuration where
compression engages does **not** necessarily obey `n_float + n_one_hot`.

A binary (cardinality-2) categorical column is exactly the shape most likely to be
packed. This caveat is therefore *not* hypothetical for the `default_binary`
scenario.

**Consequence for T01b:** Branch A (consume the rule) is justified only for
configurations where every candidate remains `OneFeature`. T01b must either
(a) assert that no bundling occurred for the fixture configs it enables, or
(b) keep one-hot × (bootstrap ≠ `No` OR `random_strength ≠ 0`) typed-rejected until
an instrumented run settles the compressed case. Do not silently assume the
un-bundled arithmetic holds under compression.

## Evidence grade

- **Rule for the un-bundled `OneFeature` path — HIGH.** Read directly from the
  selection loop, the candidate constructor, and the call ordering; three
  independent sites agree, and the eligibility predicate matches ours exactly.
- **Behaviour under `CompressCandidates` — NOT ESTABLISHED.** Requires the
  instrumented run.

## Why no instrumented run was produced here

The recipe in
`.planning/plans/bayesian-rng-draw-accounting/instrumented-ground-truth/GROUND_TRUTH.md`
requires a full upstream C++ build (clang 22 + GNU `ld.bfd`, conan, OpenSSL
`no_fips`, plus several Perl core modules fetched ad hoc because this Fedora install
lacks `perl-core`), and no built `catboost` CLI binary is present in this
environment. The three instrumented sources ARE still in the tree
(`CB_INSTRUMENT_LOG` in `train.cpp`, `greedy_tensor_search.cpp`,
`yetirank_helpers.cpp`), so the run remains reproducible on demand.

The source derivation above was pursued first because the question is narrow and
statically decidable for the `OneFeature` path — and it is, unambiguously. The
instrumented run remains the correct way to close the compression caveat, and is
the recommended corroboration before enabling one-hot × bootstrap.

**No draw count was guessed.** Where the source does not settle the behaviour
(compression), this file says so rather than extrapolating.
