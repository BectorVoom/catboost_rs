# Real upstream CatBoost 1.2.10 RNG-draw ground truth (TRAIN-04 bootstrap oracle)

**How this was produced (2026-07-30):** built a real, unmodified upstream CatBoost
1.2.10 (tag `v1.2.10`, commit `b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084`) CLI
binary (`catboost/app` target only, not the Python package) from a fresh clone,
with this repo's THREE already-instrumented files
(`crates/../catboost-master/catboost/private/libs/algo/{train.cpp,
greedy_tensor_search.cpp,yetirank_helpers.cpp}`) dropped in verbatim over the
pristine equivalents — no new C++ code was written for this. Built with system
clang 22 + GNU `ld.bfd` (no `lld` available; `-fuse-ld=lld` dropped, `-lstdc++`
added as a link fallback for `__cxa_demangle`, `--start-group`/`--end-group`
added to work around a `clapack part1`/`part2` static-archive ordering issue —
none of these affect program behavior, only linking). OpenSSL built with
`no_fips=True` to avoid needing a full `perl-devel`. Toolchain: `cmake`/`ninja`/
`conan`/`cython` via pip into a venv; several missing Perl core modules
(`FindBin`, `IPC::Cmd`, `ExtUtils::MakeMaker`, `ExtUtils::ParseXS`, ...) fetched
ad hoc from metacpan/GitHub since this Fedora install lacks `perl-core`.

Ran `catboost fit` with `CB_INSTRUMENT_LOG` set, on the EXACT
`bootstrap_multiblock` dataset (1500×4, exported to TSV) and the EXACT
`ISOLATING_PARAMS` + per-scenario overrides from
`crates/cb-oracle/generator/gen_fixtures.py::gen_bootstrap()`: RMSE,
`iterations=3`, `depth=2`, `learning_rate=0.1`, `l2_leaf_reg=3.0`,
`random_strength=0`, `leaf_estimation_iterations=1`, `score_function=L2`,
`leaf_estimation_method=Gradient`, `random_seed=0`, `thread_count=1`,
`boost_from_average=True`, plus `bootstrap_type`/`bagging_temperature`/
`subsample` per scenario (Bayesian: `bagging_temperature=1.0`; Bernoulli/MVS:
`subsample=0.8`; No: nothing extra). Raw JSONL logs for all 4 scenarios are
committed alongside this file (`{bayesian,bernoulli,mvs,no}.jsonl`).

## The reconstructed per-tree draw model (verified self-consistent across all 12 tree/scenario combinations, byte-exact call-count reconciliation)

Per tree, on the single persistent `LearnProgress->Rand` stream:

1. **`PRE_TREE_DRAWS = 2`** — structure-fold pick (`Rand.GenRand() % foldCount`)
   + derivative-recalc seed draw. **Matches Rust's existing `PRE_TREE_DRAWS`
   constant exactly** (`boosting.rs:58`) — confirmed correct, no change needed.
2. **`Bootstrap()` — once per tree, type-dependent, BEFORE any level loop runs:**
   - `No`: 0 draws.
   - `Bayesian`: 1 draw (`rand_seed = Rand.GenRand()`, matches
     `bootstrap.rs::generate_random_weights` exactly — no change needed).
   - `Bernoulli` (subsample<1): exactly `n_objects` (1500) sequential draws
     (`SetSampledControl`, matches `bootstrap.rs::set_sampled_control` exactly
     — no change needed).
   - `MVS`: consistent with Rust's existing 1 (rand_seed) + 2 (extra) = 3 model
     (not separately isolated in this pass, but the observed cc deltas are
     consistent with it).
3. **`RSM_SELECT_DRAWS = n_features` per level (`SelectFeaturesForScoring`),
   COMPLETELY ABSENT from Rust today, for EVERY bootstrap type including `No`.**
   One `Rand.GenRandReal1()` per LISTED float feature (here 4, INCLUDING the
   one border-less/unused-but-quantized feature), drawn UNCONDITIONALLY
   regardless of RSM=1.0 (default, no actual subsampling occurs — the draw
   still happens and is discarded). Confirmed exactly 4 draws before level 0
   (folded into the 5-draw gap alongside Bayesian's 1 bootstrap draw) and
   exactly 4 draws between level 0's end and level 1's start, in ALL 4
   scenarios, ALL 3 trees (12/12 confirmations).
4. **`CalcScores` = 1 draw per level** — already correctly modeled in Rust's
   old bulk formula's `depth` component (`greedy_tensor_search.cpp:884`'s
   per-level `randSeed`).
5. **`SelectBestCandidate` = variable Marsaglia-polar (`std_normal`) draws, ONE
   PER FEATURE, for ALL `n_features` (4) features — INCLUDING the border-less
   one.** This directly resolves Open Question #3 from `research.md`: the
   plain (non-ranking) dispatch uses the SAME "count every listed float
   feature, even border-less ones" rule as the already-fixed YetiRank
   `WR-02` accounting — `tree.rs::select_level_perturbed`'s current border-less
   SKIP (`if borders.is_empty() { feature_best.push(None); continue; }`, no
   draw) is CONFIRMED WRONG for real upstream behavior. Draw count per level is
   genuinely variable (8, 10, 12, 14 observed across different levels/trees/
   scenarios) — a `std_normal` Box-Muller rejection-sampling artifact, not a
   fixed per-feature constant; must be produced by literally calling the real
   draw function 4 times, not computed from a formula.
6. **Leaf-estimation seed = 2 draws per tree (once, after both levels)**, not
   1 as `POST_TREE_EXTRA_DRAWS` currently assumes. Confirmed identically
   across ALL 12 tree/scenario combinations (always exactly `tree_rng_end -
   tree_rng_pre_leaf == 2`).

Per-tree total reconciles exactly: `PRE(2) + Bootstrap(type) + sum over levels
of [RSM(4) + CalcScores(1) + SelectBestCandidate(variable)] + Leaf(2)` matches
`tree_rng_end.cc - tree_rng_start.cc` to the byte for all 12 tree/scenario
combinations captured.

## Why this resolves the Bernoulli/MVS/No "pass despite being wrong" mystery (Open Question #1 from `research.md`)

The `No`-bootstrap trace is the decisive data point: `bootstrap_type=No` +
`random_strength=0` means `draws_active=false` in Rust today — Rust consumes
**zero** draws for tree growth in this case. The REAL upstream trace shows
upstream consumes ~30 draws/tree in this exact configuration (RSM + CalcScores
+ SelectBestCandidate + leaf-seed) — yet `bootstrap_oracle_no` passes bit-exact
in Rust. This is only possible because **none of those draws' VALUES affect
the outcome**: RSM's `GenRandReal1() <= 1.0` is always true regardless of the
drawn value (no real subsampling), and `SelectBestCandidate`'s
`random_score_instance(raw, std_dev=0, ...)` = `raw + std_normal()*0 = raw`
regardless of the drawn normal's value. The draws are real but functionally
inert at `random_strength=0`/`RSM=1.0`.
Bernoulli/MVS are the same story at one more remove: their bootstrap draws
(1500 sequential Bernoulli comparisons; MVS block sampling) DO depend on the
actual values drawn, so a wrong phase genuinely produces a different random
80% subsample/weight set each time — but for THIS specific clean, strongly
linearly-separable synthetic dataset (`y = x @ [1.5,-2.0,0.5,3.0] +
0.1*N(0,1)`), the optimal split/leaf outcome is apparently robust to exactly
which ~20% of objects get dropped, so a "wrong" phase still coincidentally
lands on the same discrete best-split choice at the ≤1e-5 tolerance. Bayesian
is the odd one out because its per-object weight is a CONTINUOUS multiplier
feeding directly into the weighted histogram sums used for score comparison —
the specific VALUE, not just an on/off gate, so any phase drift visibly moves
which candidate border wins.

**Conclusion:** Bernoulli/MVS/No currently passing is NOT evidence that
Rust's model is correct for them — it is evidence that this specific fixture
happens to be insensitive to the (real, present) RNG-phase gap for those
scenarios. A future fixture (larger dataset, closer split margins, different
seed) could plausibly expose the exact same gap for Bernoulli/MVS too. The fix
below should be verified against ALL FOUR committed scenarios to ensure it
does not regress the ones that currently pass "by luck."

## What the fix needs to do (informational only — not implemented in this session)

Replace `boosting.rs`'s bulk `depth + POST_TREE_EXTRA_DRAWS` branch with real,
literal replays of items 3-6 above (RSM `GenRandReal1` × n_features per level,
`CalcScores` 1 draw per level, `SelectBestCandidate` `std_normal` × n_features
per level counting ALL listed float features including border-less ones, then
2 leaf-estimation draws once per tree) — driven by actually calling the real
draw primitives, not a closed-form count, since item 5's draw count is
data-dependent (Marsaglia-polar rejection sampling). This also requires fixing
`tree.rs::select_level_perturbed`'s border-less-feature skip to count (and
draw for) ALL listed float features, matching the already-correct
`YetiRankTreeSeeder` WR-02 rule. Re-verify all 4 bootstrap-type oracle
scenarios (not just Bayesian) plus `regularization_oracle_random_strength`/
`regularization_oracle_bagging_temp` after the change.
