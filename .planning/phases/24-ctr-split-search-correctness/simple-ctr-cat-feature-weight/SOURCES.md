# Evidence Ledger — ORD-07 Phantom Mixed-Projection `max_bucket_count`

Research report consumed: `.planning/phases/24-ctr-split-search-correctness/simple-ctr-cat-feature-weight/research.md`
(phase-research-agent initial pass, 2026-07-18, PLUS a same-session live
`catboost==1.2.10` debug-log spike addendum that resolved the research's own
"a spike is required" blocking recommendation). Read in full before
planning.

## Discovery context
- Surfaced after ORD-06 (`.planning/phases/24-ctr-split-search-correctness/combination-ctr-level-gating/`,
  checker-approved PASS, implemented and verified) fixed a DIFFERENT bug in
  the SAME function (`select_level_ctr_aware`). After ORD-06 landed,
  `crates/cb-model/tests/fstr_ctr_oracle_test.rs` still failed, now at
  tree0/level2 instead of level0 — confirmed via direct re-run
  `[VERIFIED: LOCAL cargo test -p cb-model --test fstr_ctr_oracle_test --
  --nocapture, this session]`.

## Root cause (HIGH confidence — live-spike verified, not just static reading)
- `[VERIFIED: LOCAL live catboost==1.2.10 debug-log spike]` — installed via
  `uv venv --python 3.12 && uv pip install catboost==1.2.10 'numpy<2'`
  (the project's standard offline-fixture recipe); retrained the FROZEN
  `fstr_ctr` fixture's EXACT params/seed/data (copied
  `gen_fixtures.py`, redirected output paths to a scratch directory, added
  ONLY `logging_level="Debug"` — the committed fixture files under
  `crates/cb-oracle/fixtures/fstr_ctr/` were NEVER touched or regenerated)
  with CatBoost's public `logging_level="Debug"` training parameter, which
  prints each level's WINNING candidate's post-`GetCatFeatureWeight` score
  directly — no C++ rebuild required.
- Cross-referenced against a TEMPORARY Rust-side `eprintln!` instrumentation
  of `select_level_ctr_aware` (added and FULLY REVERTED after use —
  `[VERIFIED: LOCAL git diff --stat crates/cb-train/src/tree.rs matching the
  pre-spike ORD-06 104-line diff exactly, post-revert]`): Rust's FLOAT
  candidate scores at tree0 levels 0-1 match upstream's printed winning
  scores to 6 DECIMAL PLACES EXACTLY (`2.415736554`/`2.415737`;
  `3.138951304`/`3.138951`) — confirms the shared scorer
  (`build_ctr_aware_histogram`/`score_candidate_ctr_aware`/`split_score`) is
  correct and that upstream's printed score is `raw × catWeight` directly.
- At level 2, Rust's best POSSIBLE weighted CTR{cat0} score
  (`raw=4.250470` at its own peak `border_idx=10`,
  `weight=0.707107` from ORD-06-04's `eligible_max_bucket_count`,
  `weighted=3.005536`) falls well short of upstream's actual winning score
  (`3.842356441`).
- `[VERIFIED: LOCAL direct NumPy counting of distinct `(partition-leaf,
  cat-value)` pairs against `crates/cb-oracle/fixtures/fstr_ctr/X_float.npy`/
  `X_cat.npy`, this session]` — using tree0's ALREADY-CONFIRMED-CORRECT
  chosen float splits to define the partition at each level:

  | Level (chosen) | phantom(float-ctx, cat0) | phantom(float-ctx, cat1) |
  |---|---|---|
  | 0 ([]) | N/A | N/A |
  | 1 ([Float(1)@-0.201386]) | 10 | 8 |
  | 2 ([Float(1)@-0.201386, Float(0)@0.561006]) | 20 | 16 |

  Including these in `max_bucket_count`: level 2 → `max(5,4,20,16)=20` gives
  `weight≈0.894427`, closing ~90% of the previously-observed gap (needed
  `≈0.90397`, back-solved from Rust's OWN instrumented raw-score peak
  `4.250470` against upstream's ACTUAL instrumented winning score
  `3.842356441` — both directly traced to the live debug-log spike);
  residual ~1% attributable to upstream's border-index labeling not being
  proven 1:1 comparable to Rust's `border_idx`. **This level-2 fit is the
  SOLE quantitative evidence claimed for the mechanism** — level 2 is also
  the fixture's actual failing level.

  **[CORRECTED, plan-checker pass 2]** An earlier version of this ledger
  also claimed a level-1 fit (`max(5,4,10,8)=10` → `weighted≈3.13950`,
  "within 0.0005" of level 1's actual float winner `3.138951`). That claim
  depended on a raw CTR score (`3.844592`) that was never sourced from any
  instrumented run, was not self-consistent with the claimed weight to the
  precision otherwise used (back-solving from `3.138951` gives `≈3.844415`,
  not `3.844592`), and was suspiciously close to the UNRELATED, correctly-
  sourced level-2 winning score (`3.842356441`) — consistent with a
  transcription error, not independent evidence. **This level-1 claim is
  RETRACTED and is not part of this bugfix's supporting evidence** — see
  `SPEC.md` §1/§9 for the authoritative, corrected framing. Only the
  level-2 fit above stands as evidence; `AT-ORD07-03b` (the real oracle
  test) remains the actual arbiter regardless.

## Upstream mechanism (already vendored + quoted during ORD-06's own research — re-confirmed relevant here)
- `[VERIFIED: LOCAL catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp:503-568
  AddTreeCtrs]` — `binAndOneHotFeaturesTree.BinFeatures =
  currentTree.GetBinFeatures()` (chosen FLOAT splits) is built and inserted
  into `seenProj` UNCONDITIONALLY (independent of `GetUsedCtrs()`'s
  contents) — the moment the tree has chosen `>= 1` float split, this base
  becomes non-empty and eligible for extension by one cat feature, matching
  ORD-07-02's gating rule exactly.
- `[VERIFIED: LOCAL catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp:1097-1115
  CalcMaxFeatureValueCount]` — consumes the bucket counts of ALL projections
  in the current level's candidate context, including this phantom
  mixed-projection's — even though it is never itself offered as a scoreable
  candidate in this codebase's categorical-only `TProjection` design
  (`[VERIFIED: LOCAL crates/cb-train/src/projection.rs:93-106]`, ORD-06's
  own established simplification).

## Rust seams (CodeGraph, this session + reused from ORD-06's own verified evidence)
- `[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs:2588-2717
  select_level_ctr_aware, full body re-read]` — `used_projections`/
  `max_bucket_count` (ORD-06-04) computed BEFORE the CTR candidate loop;
  `assign_leaves_ctr_aware` already called internally by
  `build_ctr_aware_histogram` for the SAME level — reusable via a second,
  cheap, pure call.
- `[VERIFIED: CODEGRAPH crates/cb-train/src/candidates.rs:121-127
  learn_set_cardinality]` — builds a `PerfectHash`, calls
  `ph.remap(hash)` per object, discards the per-object result, keeps only
  `ph.len()`. The per-object `remap` return value IS the dense bucket index
  needed for `phantom_mixed_bucket_count`.

  **[CORRECTED, plan-checker pass 1]** An earlier version of this ledger
  proposed capturing this via a NEW `candidates.rs::learn_set_buckets`
  sibling function reusing the same hash/`PerfectHash` sequence. On review,
  this was found to duplicate an ALREADY-EXISTING, already-`pub`,
  already-oracle-tested function byte-for-byte:
  `[VERIFIED: CODEGRAPH crates/cb-data/src/cat_hash.rs:471-479
  perfect_hash_bins(column: &[&str]) -> CbResult<Vec<u32>>]`, exported via
  `[VERIFIED: CODEGRAPH crates/cb-data/src/lib.rs:39]`, oracle-tested at
  `[VERIFIED: LOCAL crates/cb-data/tests/cat_hash_oracle_test.rs:56
  cat_hashes_and_perfect_hash_bins_match_oracle]`. **The corrected design
  (T3, `PLAN.md`) calls `cb_data::perfect_hash_bins` directly — no new
  hand-rolled hashing function is written.**
- `[VERIFIED: CODEGRAPH crates/cb-train/src/boosting.rs:2696-2721]` —
  `cat_cardinalities`/`eligible_absolute` computed in `train_inner`,
  `cat_columns: &[Vec<String>]` in scope; `[VERIFIED: CODEGRAPH
  crates/cb-train/src/boosting.rs:3892-3916]` — the `has_ctr` branch calling
  `greedy_tensor_search_oblivious_with_ctr`, the plumbing insertion point.

## Fixture/test survey (provable-no-op targets)
- `[VERIFIED: LOCAL grep cat_features/depth in
  crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs,
  multi_permutation_e2e_oracle_test.rs]` — BOTH have ZERO float features
  (categorical-only fixtures) — `phantom_bucket_gate` is provably always
  `false` for them (no `Float` split can ever appear in `chosen`), making
  this fix's new contribution a structurally-guaranteed no-op, not merely an
  empirically-assumed one.
- `[VERIFIED: LOCAL grep model_size_reg crates/cb-train/tests/ctr_split_scoring_test.rs]`
  — all existing tests there use `model_size_reg=0.0`, making the WEIGHT's
  magnitude irrelevant regardless of `max_bucket_count`'s value (though the
  new parameter threading must still compile identically).

## Constraints
- `[VERIFIED: LOCAL Cargo.toml:10-14]` — workspace denies `unwrap_used`,
  `expect_used`, `panic`, `indexing_slicing` (clippy-only enforcement).
- `[VERIFIED: LOCAL CLAUDE.md]` — source/test separation.
- `[PROJECT: memory ctr-model-loading.md]` — CTR fixtures frozen, never
  regenerate.
- `[PROJECT: .planning/phases/24-ctr-split-search-correctness/combination-ctr-level-gating/{SPEC,PLAN,PLAN-CHECK,SOURCES}.md]`
  — ORD-06's own conventions, evidence style, and its already-landed,
  must-not-be-altered fix, mirrored throughout this slice's artifacts.

## Planner Agent availability
- `[VERIFIED: LOCAL find /home/user/.claude/agents
  /home/user/Documents/workspace/catboost_rs/.claude/agents -iname
  '*planner*']` → only `specification-planner.md` exists (a different
  skill's agent). No agent literally named `planner` is installed. `PLAN.md`
  was therefore authored directly by this skill session as the documented
  fallback (`[UNVERIFIED: Planner Agent unavailable]`), still subject to the
  independent Plan Checker gate.

## Spike artifacts (NOT part of the deliverable — reverted/scratch only)
- A `uv`-managed venv, a copied+modified `gen_fixtures.py` (added
  `logging_level`, redirected output paths), and temporary Rust
  `eprintln!` instrumentation of `select_level_ctr_aware` were used during
  this session's research spike and FULLY REVERTED / left in the session
  scratchpad directory (never committed, never touching the frozen fixture
  or the working tree's tracked files beyond the already-existing ORD-06
  diff). No permanent artifact from the spike itself is part of this
  slice's deliverable — only the SPEC/PLAN/SOURCES/research.md documents.
