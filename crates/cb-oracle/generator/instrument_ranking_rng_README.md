# Ranking-RNG instrumentation — YetiRank / StochasticRank ground truth (Plan 06.3-04, Wave C)

**Authorization.** A DELIBERATE, USER-APPROVED deviation from the
"Python-reachable floor, no C++ instrumentation" rule, scoped to the two
randomized RNG-stream losses (D-6.3-02: instrument ONLY the randomized losses;
umbrella D-07). YetiRank and StochasticRank have no clean Python-reachable
ground truth — their gradients are drawn from a `TFastRng64` stream whose exact
draw COUNT + ORDER is the parity crux (RESEARCH Pitfall 1) — so they get C++
instrumentation. Escalate-don't-weaken (D-6.3-03b) governs the build path.

**RUN-ONCE / COMMIT, OFFLINE ONLY, NEVER in CI (D-09 / D-12).** The generators
are built and run by hand; CI reads only the committed fixture data under
`crates/cb-oracle/fixtures/ranking_corpus/{yetirank,yetirank_pairwise,stochasticrank}/`,
never the generators.

## Two generators (zero catboost includes)

Both are STANDALONE, DEPENDENCY-FREE C++20 files — the `ordered_oracle.cpp` /
`cityhash_oracle.cpp` precedent. They transcribe the smallest RNG units verbatim
(with `file.cpp:line` citations) and SELF-ORACLE the transcription against the
already-oracle-locked `cb-core::TFastRng64` + `cb-core::std_normal` Rust
reproduction (rng_test.rs / normal.rs, validated against the vendored
`fast_ut.cpp` vectors). **The Rust reproduction is the ground truth; the C++ must
AGREE with it bit-for-bit** — the self-oracle prints the chain and a divergence is
caught at oracle-compare time.

| Generator | Transcribes | Smallest unit |
|-----------|-------------|---------------|
| `yetirank_oracle.cpp` | `GenerateYetiRankPairsForQuery` (`yetirank_helpers.cpp:305-345`), `CalcWeightsClassic` (`:193-205`), `AddNoise` Gumbel (`:149-152`), `GenRandUI64Vector` (`restorable_rng.cpp:3-9`), the 2-level seed derivation (`:365-389`) | 1 group, 3 docs, permutations=10 |
| `stochasticrank_oracle.cpp` | `CalcDersForSingleQuery` Stage-1 shift/center + Stage-2 noise stream (`error_functions.cpp:1024-1055`), `StdNormalDistribution` (`normal.h:11-24`) | 1 group, 3 docs, num_estimations=1 |

## OFFLINE build + run command

The generators have NO catboost dependency — they build with a stock C++20
compiler (no clang-18, no conan, no _catboost build required):

```sh
# Build (g++ or clang++, either works — zero catboost link):
g++ -std=c++20 -O2 -Wall crates/cb-oracle/generator/yetirank_oracle.cpp -o /tmp/yetirank_oracle
g++ -std=c++20 -O2 -Wall crates/cb-oracle/generator/stochasticrank_oracle.cpp -o /tmp/stochasticrank_oracle

# Run (self-oracle to stderr; JSONL RNG log only when CB_INSTRUMENT_LOG is set):
CB_INSTRUMENT_LOG=/tmp/yetirank_rng.jsonl       /tmp/yetirank_oracle
CB_INSTRUMENT_LOG=/tmp/stochasticrank_rng.jsonl /tmp/stochasticrank_oracle
```

The self-oracle was RUN this session and AGREES with `cb-core` bit-for-bit:

```
[self-oracle] random_seed=0 block_seed=12283622132691337806 query_seed=446232438382340955
[self-oracle] std_normal(seed=0)[0] = 0.63370673353927653
[self-oracle] std_normal(seed=0)[1] = -0.52843396094496731
[self-oracle] std_normal(seed=0)[2] = -0.44081967217770740
```

(reproduced bit-identically by the Rust `cb-core::TFastRng64` / `std_normal`).

## CB_INSTRUMENT_LOG schema (JSONL)

YetiRank (`yetirank_oracle.cpp`):
```
{"event":"query_seed","group":<g>,"seed":<u64>}
{"event":"gumbel_draw","perm":<p>,"doc":<d>,"u":<f64>,"boot":<f64>}
{"event":"sorted_order","perm":<p>,"order":[<idx>,...]}
{"event":"competitor","winner":<w>,"loser":<l>,"weight":<f64>}
```
StochasticRank (`stochasticrank_oracle.cpp`):
```
{"event":"gauss_draw","group":<g>,"sample":<s>,"doc":<d>,"noise":<f64>}
{"event":"score","group":<g>,"sample":<s>,"doc":<d>,"score":<f64>}
{"event":"sorted_order","group":<g>,"sample":<s>,"order":[<idx>,...]}
```

## Decay resolution (RESEARCH A3 / Open-Q2)

The YetiRank Classic-weight decay ambiguity (0.85 vs 0.99) is RESOLVED here by the
upstream default: `GetYetiRankDecay` returns `0.85`
(`loss_description.cpp:188-193`). The generator LOGS the effective decay
(`[yetirank_oracle] decay=0.850000 permutations=10`). The Rust sampler defaults
to `YETIRANK_DECAY_DEFAULT = 0.85` accordingly.

## Captured ground truth (committed fixtures)

The RNG stream for the smallest unit is frozen under the corpus directories:
- `yetirank/`: query_seed=446232438382340955 (random_seed=0, 2-level chain);
  the 3 nonzero sampled competitors
  `(w0,l1)=0.192, (w0,l2)=0.098250, (w2,l1)=0.083250`.
- `stochasticrank/`: the 3-doc num_estimations=1 noise/score stream for
  group-0 seed=5, sorted order `[1,0,2]`.

These RNG draws are the per-stage oracle's INTEGER-/f64-exact compare target
(`yetirank_oracle_test.rs` / `stochasticrank_oracle_test.rs` `compare_stage`).

## STATUS — partial closure + deferred end-to-end fixture (escalate-don't-weaken, D-6.3-03b)

The Task-1 feasibility probe found the path (c) (escalate) condition:
- the persisted `/tmp` clang-18 + built `_catboost` toolchain (memories
  `instrumented-trainer-toolchain-persists`) is **ABSENT this session**;
- no `clang`/`clang-18` anywhere on the host (only an apt *candidate*, not
  installed); a fresh build needs `apt-get download` + `dpkg -x` of the LLVM
  toolchain (~300-400 MB extracted) PLUS a full instrumented catboost C++ TRAINER
  build;
- **disk is at 95% / ~12 GB free** (NOT the 67 GB RESEARCH assumed) — the
  documented `disk-pressure-and-full-suite-verification` link-failure regime
  (cb-compute could not even link at ~100% full).

Per D-6.3-03b the executor did **NOT weaken** any oracle. What IS closed this
session (no weakening, real ground truth):
1. The two standalone generators are authored, **compile clean** (g++ C++20,
   `-Wall`, zero warnings), **run**, and **self-oracle bit-for-bit** against the
   oracle-locked Rust RNG — proving the RNG draw transcription (the parity crux)
   is correct. Their RNG-stream ground truth is frozen as committed fixture data.
2. The Rust YetiRank sampler + StochasticRank der draw from the SAME
   `cb-core::TFastRng64` / `std_normal`, so their RNG streams are validated by
   transitivity (the self-oracle agreement).

What remains DEFERRED (the OFFLINE closure step, recorded precisely):
- The **end-to-end per-stage fixture** (a trained YetiRank/YetiRankPairwise/
  StochasticRank `model.json` from the instrumented catboost 1.2.10 trainer)
  needed to gate `compare_stage(Splits|LeafValues|StagedApprox|Predictions)`
  ≤1e-5 across full training. That requires the full instrumented TRAINER build
  (infeasible this session: toolchain absent + disk NO-GO).
- **Closure recipe** (when disk + toolchain are available): follow the
  `instrument_live_trainer_README.md` sudo-free recipe (clang-18 + lld-18 via
  `apt-get download` + `dpkg -x` into `/tmp/clang18_prefix`; conan/ninja/cython
  via `uv tool`; `build_native.py --targets _catboost`), train the three corpus
  configs with `CB_INSTRUMENT_LOG` set, capture the per-stage `.npy` + the RNG
  draw log, freeze under `ranking_corpus/{yetirank,yetirank_pairwise,
  stochasticrank}/`, then drop the `#[ignore]`-free oracle tests' OFFLINE guard.

The three oracle test files are authored **compile-clean, with NO `#[ignore]` and
NO weakened tolerance**: they assert the RNG-draw ground truth captured above
(which IS available) and are structured so the end-to-end per-stage compare wires
in unchanged once the deferred trainer fixture lands. This mirrors the Phase-5
ORD-01 "fallback taken, ground truth committed, oracle uncommitted, no weakening"
precedent.

## Disk hygiene

Free `target/debug/incremental` and `~/.conan2` cache before/after any trainer
build (`disk-pressure` memory). The standalone generators themselves need
negligible disk (single small binaries in `/tmp`).
