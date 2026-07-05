# Phase 21: CPU Split-Finding Histogram Rewrite - Research

**Researched:** 2026-07-05
**Domain:** CPU gradient-boosting split-finding (histogram / `TBucketStats` algorithm, deterministic float summation, rayon parallelism) — pure host Rust in `cb-train` / `cb-compute`
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Root cause (Spike 003 — do not re-investigate):** The CPU path `select_level_plain`/`score_candidate` (`crates/cb-train/src/tree.rs`) re-scans ALL n objects and re-reduces them from scratch for EVERY `(feature,border)` candidate → `O(n·n_features·n_bins·depth)` per tree. No `TBucketStats` histogram, no subtraction trick.

**Algorithm (PERF-01):**
- Build per-feature bin histograms once per level: a single `O(n)` pass accumulates `(feature, bin) → {Σder1, Σweight}` for the current leaf partition; score all of a feature's borders from its histogram in `O(n_bins)` — the `O(n)` factor is paid ONCE per level, not once per candidate.
- Implement the **subtraction trick**: derive the larger child's histogram by subtracting the smaller sibling's from the parent's (mirrors upstream `TStatsForSubtractionTrick` / `PrevTreeLevelStats`).
- **Mirror the existing device reference:** `crates/cb-backend/src/kernels/pointwise_hist.rs` (Phase 11 `pointwise_hist2` + subtraction trick) is the algorithmic template — transcribe its logic onto the host.

**Parity (PERF-02) — the crux:**
- EVERY shipped ≤10⁻⁵ CPU oracle fixture MUST stay bit-exact after the rewrite.
- The rewrite MUST preserve the bit-exact `cb_core::sum_f64` canonical-order summation (D-05/D-08) while dropping the rescan — accumulate bins with a deterministic ordered sum (fixed-point-u64, already proven in Phase 10/11, OR per-bin ordered `sum_f64`). Parity is the acceptance gate, not an afterthought.

**Parallelism (PERF-03) — SECOND, after the algorithm:**
- Fix the algorithm FIRST; parallelize SECOND.
- Parallelize over features/candidates with `rayon`; keep the final per-bin reduction deterministic (per-feature independent, deterministic merge).
- Introduce reusable scratch buffers (a `TLearnContext` analogue): one incrementally updated `leaf_of`, fixed-size histogram arrays — eliminate the per-candidate allocation storm.

**Hard constraints:**
- **NEVER add a `cb-train` dependency on `cb-backend`** (feature-unification landmine). Transcribe histogram logic inline into `cb-train`/`cb-compute`; do NOT reach across the seam to reuse the device kernel.
- CPU-only: the `cubecl`-free `cb-compute` generic boundary (D-03) stays cubecl-free.
- Kernels/`#[cube]` code are NOT touched by this phase (pure host Rust).

### Claude's Discretion

- Choice between fixed-point-u64 vs per-bin ordered `sum_f64` for the parity-preserving accumulation (both explicitly sanctioned by CONTEXT — this research recommends a primary and a fallback).
- Scratch-buffer struct shape (the `TLearnContext` analogue).
- Wave sequencing within the suggested (1) oblivious → (2) Depthwise/Lossguide+CTR → (3) rayon+scratch ordering.

### Deferred Ideas (OUT OF SCOPE)

- CTR-path CPU-perf on categorical-heavy real datasets (Spike 002 was numeric-only) — a Phase 22 validation nuance, not a blocker here.
- Pairwise-scorer histogram unification (fold in ONLY if it falls out cleanly; otherwise leave `greedy_tensor_search_oblivious_pairwise` on its dedicated scorer and note the exclusion).
- The GPU/device grow path (`pointwise_hist.rs`, `grow_tree_on_device`) — untouched, do NOT regress.
- Any change to model format, leaf-value calculation, or inference.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PERF-01 | Histogram + subtraction trick; per-tree time no longer scales with `n_bins` | §"Histogram data structure on the host" + §"Subtraction trick"; upstream `TBucketStats` (`calc_score_cache.h:72`), prefix scan (`scoring.cpp` `CalcScoresForLeaf`), device template (`pointwise_hist.rs:106-163`). Acceptance = flat `n_bins` sweep (Spike 002 harness). |
| PERF-02 | All CPU policies + CTR path use the histogram scorer; every ≤1e-5 oracle stays bit-exact | §"Parity-preserving summation" + §"Coverage across grow policies + CTR". Score math (`score.rs`) is a pure function of `LeafStats` and stays UNCHANGED; only how `LeafStats` are produced changes. Gate = the 60+ oracle suite in `crates/cb-train/tests/`. |
| PERF-03 | rayon parallelism + reusable scratch buffers | §"Parallelism + scratch buffers"; `rayon` 1.12.0 (verdict OK) is a new `cb-train`/`cb-compute` dep. Determinism preserved via per-feature-independent work + fixed-order merge. |
</phase_requirements>

## Summary

The CPU split search is algorithmically wrong, not merely un-tuned: for every `(feature, border)` candidate it rebuilds the whole-dataset leaf partition (`assign_leaves`, `tree.rs:396`) and re-reduces every object (`reduce_leaf_stats`, `cb-compute/src/histogram.rs:49`), giving `O(n · n_features · n_bins · depth)` per tree (Spike 002/003, verdict ROOT-CAUSE-CONFIRMED). The fix is the standard gradient-boosting **histogram / `TBucketStats`** algorithm that the project's own design doc already mandates (`docs/CATBOOST_CORE_DESIGN.md` §"The Tree-Growing Pipeline" step 3c) and that already exists on the GPU path (`cb-backend/src/kernels/pointwise_hist.rs`): bin every object ONCE per level into per-`(feature, bin)` accumulators, then score all of a feature's borders by a `O(n_bins)` prefix scan over its buckets — collapsing the `n_bins`/`n_features` linear blow-up.

The score arithmetic does not change. `l2_split_score` / `cosine_split_score` / `multi_dim_split_score` (`cb-compute/src/score.rs`) are pure functions of `&[LeafStats]` in canonical leaf order. The rewrite changes only HOW the `LeafStats` (`{sum_weighted_delta = Σder1, sum_weight = Σweight}`) are produced — from a per-candidate object-order gather to a per-level histogram + prefix scan. This is exactly what upstream CatBoost does (`scoring.cpp` `CalcScoresForLeaf` walks buckets maintaining running `trueStats`/`falseStats`, calling `AddLeafPlain(falseStats, trueStats)`), so the histogram approach is if anything MORE parity-faithful than the current object-order-within-leaf sum, which already differs from upstream's bucket-order sum at the ULP level yet passes ≤1e-5.

**Primary recommendation:** Build a 2-channel per-`(feature, bin)` histogram (Σder1, Σweight) once per level, each bin accumulated via `cb_core::sum_f64` over its member objects in ascending object order (transcribe `host_reference_hist2`, `pointwise_hist.rs:106-163`, into `cb-compute`); score borders by a prefix scan combining buckets via `sum_f64` in ascending bin order; feed the resulting `LeafStats` into the UNCHANGED `split_score`. Keep leaf-VALUE estimation (`reduce_leaf_stats`/`reduce_leaf_der2`/`collect_leaf_residuals`, run ONCE per tree — not the bottleneck) on the existing object-order path so leaf values stay byte-identical. The parity gate is the full oracle suite; the sole real hazard is strict-`>` tie-flips in split selection, mitigated by matching upstream's bucket/prefix order. Do PERF-01 first, then rayon (PERF-03).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Split-candidate scoring (histogram build + prefix scan) | `cb-train` grow loop (`tree.rs`) | `cb-compute` (new histogram + `LeafStats` derivation) | The grow policy owns candidate enumeration/ordering; the pure-generic, cubecl-free reduction primitive belongs in `cb-compute` next to `reduce_leaf_stats` (D-03). |
| Deterministic float summation | `cb-core` (`sum_f64`) | — | Single sanctioned primitive (D-07/D-08); histogram accumulation routes through it, unchanged. |
| Score math (L2 / Cosine / multi-dim) | `cb-compute` (`score.rs`) | — | Pure function of `LeafStats`; UNCHANGED by this phase. |
| Leaf-value estimation | `cb-compute` (`leaf.rs`) via `reduce_leaf_stats`/`reduce_leaf_der2` | — | Runs once per tree (not the hot loop); stays on the object-order path so leaf values are byte-identical. |
| Parallelism / scratch reuse | `cb-train` grow loop | `rayon` (new dep) | Feature-independent histogram build/scoring parallelizes cleanly; scratch buffers live in the grow context. |
| Device histogram (GPU) | `cb-backend` (`pointwise_hist.rs`) | — | Out of scope; READ-ONLY template. `cb-train` must NOT depend on `cb-backend` for it. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `rayon` | 1.12.0 | Data-parallel iterators for the per-feature histogram build/scoring (PERF-03) | The de-facto Rust work-stealing parallelism crate; `par_iter` over independent features maps directly to CatBoost's `LocalExecutor` parallel-over-candidates model (`docs/CATBOOST_CORE_DESIGN.md` §step 3c, `TLearnContext` owns the thread pool). `[VERIFIED: crates.io]` |
| `cb-core::sum_f64` | (in-repo) | Deterministic sequential f64 fold; the parity primitive | Already the single sanctioned summation (D-07/D-08, `reduction.rs:32`). Histogram bins route through it — no new summation primitive. `[VERIFIED: crates/cb-core/src/reduction.rs:32]` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| (none new) | — | — | Fixed-point-u64 accumulation, if chosen for the parallel merge (Q2 fallback), is plain `u64` integer arithmetic — no crate needed. Phase 10/11 proved the pattern (`cb-backend/src/kernels/reduce.rs:654` `round(v*2^30) → i64 → u64`). |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `rayon` `par_iter` over features | `std::thread` manual pool | rayon gives deterministic-order `collect` into a pre-sized `Vec` for free; hand-rolled threads reintroduce the ordering/merge bugs rayon already solves. Use rayon. |
| Prefix-scan per-bin `sum_f64` (Q2 primary) | Fixed-point-u64 order-invariant accumulation (Q2 fallback) | Fixed-point is exact + order-independent (makes the subtraction trick and parallel merge trivially deterministic) but changes exact bits vs the sanctioned f64 fold → higher tie-flip risk on the tight 1e-5 CPU bar. Prefer `sum_f64` first; adopt fixed-point only for the parallel merge if it passes the oracle suite. |

**Installation (workspace Cargo.toml `[workspace.dependencies]`, then `[dependencies]` in cb-train/cb-compute):**
```toml
# workspace root Cargo.toml — pin latest stable (CLAUDE.md: always latest crate versions)
rayon = "1.12.0"
```

**Version verification:** `cargo search rayon` → `rayon = "1.12.0"` (confirmed 2026-07-05). Published 2015-12-10, 7.48M weekly downloads, repo `github.com/rayon-rs/rayon`, not deprecated.

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| `rayon` | crates.io | ~10 yrs (2015-12-10) | 7.48M/wk | github.com/rayon-rs/rayon | OK | Approved |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

`rayon` verdict via `gsd-tools query package-legitimacy check --ecosystem crates rayon` → `OK`, no reasons. `[VERIFIED: crates.io]`

## Architecture Patterns

### System Architecture Diagram (target CPU split-finding, per level)

```text
                    per-object der1[], weight[]  (from compute_gradients)
                    quantized feature columns    (matrix.feature_values → bin via feature_borders)
                    current partition leaf_of[]  (incrementally maintained scratch)
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  ONE O(n) binning pass per level  (rayon par_iter over features)│
   │  for each object o:  bin = search(feature_borders[f], value)    │
   │  hist[(leaf, feature, bin)].{Σder1, Σweight} += (der1[o], w[o])  │  ← via cb_core::sum_f64
   │                                                                 │     (per-bin gather in object order)
   └──────────────────────────────────────────────────────────────┘
                              │  per-(leaf,feature,bin) TBucketStats
                              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  O(n_bins) PREFIX SCAN per feature  (upstream CalcScoresForLeaf)│
   │  running trueStats/falseStats over buckets → LeafStats per split│  ← combine buckets via sum_f64
   │  border b:  falseStats = Σ bins ≤ b ; trueStats = Σ bins > b     │     (ascending bin order)
   └──────────────────────────────────────────────────────────────┘
                              │  &[LeafStats] in canonical leaf order
                              ▼
        UNCHANGED score math:  split_score(fn, &leaves, scaled_l2)   (cb-compute/src/score.rs)
                              │  candidate score
                              ▼
        strict `>` first-wins select_best_candidate  (tree.rs:303)  → chosen Split
                              │
                              ▼
        SUBTRACTION TRICK across level transition (FixUpStats):
        build the SMALLER child partition's histogram; larger = parent.Remove(smaller)
        (upstream scoring.cpp:315 FixUpStats; TStatsForSubtractionTrick leafwise_scoring.h:11)
```

Leaf-VALUE estimation is a SEPARATE once-per-tree path (`reduce_leaf_stats`/`reduce_leaf_der2` → `leaf.rs`) that stays UNCHANGED (byte-identical leaf values).

### Recommended Project Structure

```
crates/cb-compute/src/
├── histogram.rs      # ADD: per-(feature,bin) TBucketStats build + prefix-scan → LeafStats.
│                     #      Keep reduce_leaf_stats / reduce_leaf_der2 / collect_leaf_residuals
│                     #      UNCHANGED (leaf-value path). New fns are pure, cubecl-free (D-03).
├── score.rs          # UNCHANGED (score math is a pure fn of LeafStats).
crates/cb-train/src/
├── tree.rs           # REWRITE the candidate scoring inside select_level_plain / _perturbed /
│                     #      best_split_for_leaf / select_level_ctr_aware to consume the histogram.
│                     #      Candidate ENUMERATION ORDER + strict `>` tie-break UNCHANGED.
│                     # ADD: a GrowScratch struct (leaf_of, histogram arrays) reused across levels.
```

### Pattern 1: 2-channel `(feature, bin)` histogram — the `TBucketStats` analogue on host
**What:** For the current leaf partition, one `O(n)` pass accumulates, per `(leaf, feature, bin)`, `{SumWeightedDelta = Σder1, SumWeight = Σweight}`.
**When to use:** Once per level (oblivious) / once per leaf's doc subset (leaf-wise), replacing the per-candidate rescan.
**Source layout (transcribe the FROZEN device layout, `pointwise_hist.rs:44-49`):**
```rust
// index(leaf, feature, bin, channel) collapses (single tree, one feature group) to:
//   flat[(leaf * n_features * n_bins + feature * n_bins + bin) * 2 + channel]
//   channel 0 = Σ der1 ("weighted delta"), channel 1 = Σ weight
// Mirrors upstream TBucketStats { SumWeightedDelta, SumWeight } (calc_score_cache.h:72-95).
```
**Accumulation (transcribe `host_reference_hist2`, `pointwise_hist.rs:126-162`):**
```rust
// Gather each (leaf, feature, bin) cell's contributions in ASCENDING OBJECT ORDER,
// then fold through the single sanctioned primitive — never a raw .sum() (D-05/D-08).
let mut delta_members: Vec<Vec<f64>> = vec![Vec::new(); n_cells];   // scratch, reused across levels
// ... for obj in 0..n { let bin = bin_of(feature, obj); delta_members[cell].push(der1[obj]); }
out[base]     = cb_core::sum_f64(&delta_members[cell]);  // Σ der1
out[base + 1] = cb_core::sum_f64(&weight_members[cell]); // Σ weight
```

### Pattern 2: prefix scan over buckets → `LeafStats` per split threshold
**What:** For a feature's buckets, maintain a running `falseStats` (bins ≤ b) and `trueStats` (bins > b); every border b yields the 2-leaf partition's `[falseStats, trueStats]` fed to `split_score`.
**When to use:** Scoring all of a feature's borders in `O(n_bins)` from its histogram row — the core `n_bins` collapse.
**Source (upstream `scoring.cpp` `CalcScoresForLeaf` → `UpdateSplitScore` → `AddLeafPlain(falseStats, trueStats)`, `scoring.cpp:576-583`):**
```rust
// leaf order is canonical (bit i = split i outcome), matching the existing assign_leaves
// leaf_index() forward-bit convention (tree.rs:24). Combine bucket TBucketStats via sum_f64
// in ascending bin order so the fold order is fixed and reproducible.
```

### Pattern 3: subtraction trick across the level transition (`FixUpStats`)
**What:** After a split is chosen, the level's leaves subdivide. Build the SMALLER child partition's histogram directly; derive the larger child = `parent.Remove(smaller)`.
**Source (upstream `scoring.cpp:315` `FixUpStats`; `calc_score_cache.h:88` `TBucketStats::Remove` = plain `-=`; `TStatsForSubtractionTrick` `leafwise_scoring.h:11`; device mirror in `pointwise_hist.rs` Phase 11):**
```rust
// stats[i].Remove(stats[i + halfOfStats])  — and a swap when the selected side is `false`.
// f64 subtraction is a rounding hazard on the 1e-5 bar (see Pitfall 2); upstream uses it too,
// so matching its order is the parity-faithful choice. Fixed-point-u64 makes Remove EXACT.
```

### Anti-Patterns to Avoid
- **Re-deriving `leaf_of` per candidate** (`score_candidate → assign_leaves`, `tree.rs:432`): this IS the root cause. Maintain `leaf_of` as reused scratch, updated once per chosen split.
- **Nested `Vec<Vec<f64>>` allocated per candidate** (`reduce_leaf_stats`, `histogram.rs:58-59`): the ~10⁸ allocs/tree OOM (Spike 004). Allocate fixed-size histogram scratch ONCE and clear-reuse.
- **Changing the score math** to "help" parity: forbidden — score functions consume `LeafStats` unchanged (CONTEXT).
- **Reaching into `cb-backend`** to reuse the device kernel: feature-unification landmine (CONTEXT hard constraint, memory `backend-crate-and-rocm-ci-constraints`). Transcribe inline.
- **Raw iterator `.sum()` / `.fold(0.0, +)`** over floats: banned by `scripts/check-no-raw-float-sum.sh` (D-08). Route every sum through `cb_core::sum_f64`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Deterministic float summation | A new Kahan/pairwise/fixed-point summer for the histogram | `cb_core::sum_f64` (`reduction.rs:32`) | Single sanctioned primitive; matches upstream `thread_count==1` single-block fold. A different summer breaks the ≤1e-5 gate everywhere (D-07/D-08). |
| Work-stealing thread pool | `std::thread` + channels + manual merge | `rayon` `par_iter` + ordered `collect` | rayon gives deterministic collect-into-`Vec` and mirrors CatBoost's `LocalExecutor`. |
| Bucket-stats struct | Ad-hoc `(f64,f64)` tuples scattered across the loop | A `TBucketStats`-shaped `LeafStats`-adjacent type in `cb-compute` | Keeps the `Add`/`Remove` (subtraction trick) semantics in one place, matching `calc_score_cache.h:72`. |
| Score/gain formula | New L2/Cosine math to fit the histogram | `l2_split_score`/`cosine_split_score`/`multi_dim_split_score` (`score.rs`) UNCHANGED | They are pure functions of `LeafStats`; the phase only changes how `LeafStats` are produced. |

**Key insight:** The entire phase is a *data-production* change (per-candidate object-order gather → per-level histogram + prefix scan) behind an UNCHANGED score interface (`&[LeafStats] → f64`). Everything hard about parity is already solved in `cb_core::sum_f64` and `cb-compute/src/score.rs`; the risk is confined to summation ORDER and split-SELECTION, not new math.

## Runtime State Inventory

> This is a pure in-process algorithm refactor (no rename, no persisted data, no external service). All five categories verified empty.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no on-disk datastore keys/collections touched; model format explicitly OUT of scope (CONTEXT). Verified: change is confined to the in-memory grow loop in `cb-train`/`cb-compute`. | none |
| Live service config | None — no external service. Verified: no network/service code in `cb-train`. | none |
| OS-registered state | None — no OS registrations. | none |
| Secrets/env vars | Only `CB_PERF` (test-gate for `perf_baseline_test.rs`, `bench_grow_speed_test.rs`) — read, never written; unchanged. | none |
| Build artifacts | Adding `rayon` changes `Cargo.lock` (new transitive deps: `rayon-core`, `crossbeam-*`). Expected + committed. | commit updated `Cargo.lock` |

**Nothing found in categories 1–4:** verified by scoping the change to `crates/cb-train/src/tree.rs` + `crates/cb-compute/src/histogram.rs` and the `[dependencies]` tables; no serialization, service, or OS surface is in the diff.

## Common Pitfalls

### Pitfall 1: Split-selection tie-flip under the new summation order (THE parity risk)
**What goes wrong:** The strict `>` first-wins tie-break (`select_best_candidate`, `tree.rs:303-313`; `select_level_perturbed` instances, `tree.rs:719,745`) compares candidate scores by exact float value. A histogram/prefix-scan sum differs from the current object-order-within-leaf sum at the ULP level; if two candidates are near-tied, the winner can flip → a different split → divergent tree structure → oracle failure far above 1e-5.
**Why it happens:** Grouping objects by bucket then combining reorders float additions vs a pure object-order fold; f64 addition is non-associative.
**How to avoid:** Match upstream's accumulation ORDER as closely as possible — per-bin `sum_f64` in object order, buckets combined via `sum_f64` in ascending bin order (this is structurally what upstream `CalcScoresForLeaf` does, so the scores land closer to upstream's than the current code does). Run the FULL oracle suite as the gate; a flip surfaces as a specific fixture failing, not a silent regression.
**Warning signs:** An oracle test that currently passes fails with a large (not ~1e-5) delta → suspect a structure divergence, i.e. a tie-flip.

### Pitfall 2: Subtraction-trick rounding on the tight 1e-5 CPU bar
**What goes wrong:** `sibling = parent.Remove(child)` (`scoring.cpp:315`, `calc_score_cache.h:88` plain `-=`) introduces cancellation/rounding not present in a fresh sum, potentially pushing a score across a tie boundary.
**Why it happens:** f64 `parent - child ≠` direct sum of the sibling's objects.
**How to avoid:** Two options. (a) Match upstream exactly (upstream ALSO subtracts, so its bucket values already carry this rounding — matching it is parity-faithful). (b) If a fixture reveals subtraction-induced divergence, either fall back to a fresh rebuild of both children (still `O(n)` per level — the target complexity, since the `O(n)` is per-level not per-candidate) OR adopt fixed-point-u64 for the histogram (integer `Remove` is EXACT — this is why Phase 11 chose fixed-point for the device accumulator, memory `phase10-reduce-determinism-spike`). Recommend implementing (a), gate on the suite, keep (b) ready.
**Warning signs:** A fixture passes with fresh-rebuild scoring but fails once the subtraction trick is enabled.

### Pitfall 3: Perturbed-search RNG draw order must not change
**What goes wrong:** `select_level_perturbed` (`tree.rs:662-754`) reseeds a fresh RNG per candidate FEATURE (`taskIdx` ascending, `tree.rs:698`) and draws one `std_normal` per border, then one main-RNG `GetInstance` per feature. If the histogram rewrite changes the candidate ENUMERATION ORDER (feature ascending × border ascending) or how many draws happen, the RNG stream desyncs → different perturbed scores → different tree (memory: Pitfall 3, TRAIN-05).
**How to avoid:** The histogram changes only how each candidate's RAW score is COMPUTED, not the enumeration. Keep the exact `for feature { for border in borders { ... } }` loop shape and the per-feature reseed/draw sequence byte-for-byte; feed the histogram-derived raw score into `random_score_instance` exactly where `score_candidate` was called (`tree.rs:706-717`).
**Warning signs:** `random_strength`-enabled oracle fixtures (e.g. any fixture with non-zero `random_strength`) diverge while unperturbed fixtures pass.

### Pitfall 4: Binning objects vs the existing `value > border` split semantics
**What goes wrong:** The current split test is `f64::from(value) > border` (`FeatureMatrix::passes_float`, `tree.rs:360-365`) with borders ASCENDING per feature (`feature_borders`, `tree.rs:322-324`). A histogram must bin each object into the bucket consistent with this: bin b = number of borders the value exceeds, so "border b true" ⇔ bins > b. Getting the prefix/suffix boundary off-by-one silently mis-scores.
**How to avoid:** Define `bin_of(f, obj) = count of feature_borders[f] strictly less than value` (equivalently upper-bound), so `falseStats(border_k) = Σ bins ≤ k` and `trueStats = Σ bins > k`, matching `passes_float`'s strict `>`. Add a targeted unit test asserting the histogram-scored raw score equals the old `score_candidate` raw score on a small fixture BEFORE wiring selection.
**Warning signs:** Off-by-one manifests as consistently-wrong first-level splits.

### Pitfall 5: rayon nondeterminism in the merge (PERF-03)
**What goes wrong:** Parallel accumulation into a shared histogram with unordered merges reorders float additions → nondeterministic bins → parity breaks run-to-run.
**How to avoid:** Parallelize over INDEPENDENT features (each feature owns disjoint histogram rows), `collect` results into a pre-sized `Vec` indexed by feature (rayon preserves index order on `collect`), and keep each feature's per-bin fold as the sequential `sum_f64`. No cross-feature reduction, so no merge-order hazard. If a global reduction is ever needed, use fixed-point-u64 (order-independent). This mirrors CONTEXT's "per-feature independent, deterministic merge."
**Warning signs:** A fixture that passes single-threaded fails or flickers under `--features` with rayon enabled.

### Pitfall 6: `indexing_slicing` deny-lint in production histogram code
**What goes wrong:** The workspace denies `indexing_slicing`, `unwrap_used`, `expect_used`, `panic` (`Cargo.toml [workspace.lints.clippy]`). Hot-loop histogram code naturally reaches for `hist[cell]`.
**How to avoid:** Use `.get()`/`.get_mut()` with defensive fallbacks (the existing `reduce_leaf_stats` does exactly this, `histogram.rs:69-74`), or localized `#[allow(clippy::indexing_slicing)]` ONLY where a bound is provably established — but the deny is workspace-wide and manifest overrides are forbidden when `lints.workspace = true`, so prefer `.get()`. Test code is exempt via `#![cfg_attr(test, allow(...))]`.
**Warning signs:** `cargo clippy` fails the build on the new hot loop.

## Code Examples

### The score interface that stays UNCHANGED (produce these, feed them in)
```rust
// Source: crates/cb-compute/src/score.rs:49 (l2_split_score) — pure fn of LeafStats.
pub fn l2_split_score(leaves: &[LeafStats], scaled_l2: f64) -> f64 { /* Σ avg·SumWeightedDelta via sum_f64 */ }
// Source: crates/cb-compute/src/histogram.rs:29 (the struct the histogram must yield per leaf)
pub struct LeafStats { pub sum_weighted_delta: f64, pub sum_weight: f64 }
```

### Upstream `TBucketStats` + subtraction semantics to transcribe
```cpp
// Source: catboost/private/libs/algo/calc_score_cache.h:72-95
struct TBucketStats { double SumWeightedDelta; double SumWeight; double SumDelta; double Count;
    inline void Add(const TBucketStats& o)   { SumWeightedDelta += o.SumWeightedDelta; SumWeight += o.SumWeight; /*…*/ }
    inline void Remove(const TBucketStats& o){ SumWeightedDelta -= o.SumWeightedDelta; SumWeight -= o.SumWeight; /*…*/ } };
// Source: catboost/private/libs/algo/scoring.cpp:315 FixUpStats — subtraction trick + swap
// Source: catboost/private/libs/algo/scoring.cpp:576 UpdateSplitScore → AddLeafPlain(falseStats, trueStats)
```
(cb-train scoring needs only the 2 channels `SumWeightedDelta`, `SumWeight`; `SumDelta`/`Count` and the second-order der2 belong to the leaf-value / Newton path which stays on `reduce_leaf_der2`, `histogram.rs:100`.)

### Host histogram accumulation to mirror (READ-ONLY reference)
```rust
// Source: crates/cb-backend/src/kernels/pointwise_hist.rs:106-162 (host_reference_hist2)
// Generalizes reduce_leaf_stats from `leaf→bin` to `(feature,bin)` cells; gathers each cell in
// ascending object order, folds via cb_core::sum_f64, writes the frozen (feature*n_bins+bin)*2+channel layout.
// TRANSCRIBE this into cb-compute (cb-train must NOT depend on cb-backend).
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-candidate full-dataset rescan (`score_candidate`+`assign_leaves`+`reduce_leaf_stats`) | Per-level `TBucketStats` histogram + `O(n_bins)` prefix scan + subtraction trick | This phase | `O(n·nf·nbins·depth)` → `O(n·nf·depth) + O(nf·nbins·leaves)`; the `n_bins`/`n_features` linear blow-up (Spike 002 "smoking gun") collapses. |
| 100% single-threaded grow loop | rayon parallel-over-features histogram build/scoring | This phase (PERF-03, after algorithm) | Recovers the ~core-count factor CatBoost gets from `LocalExecutor` (Spike 004: 3.9× at n=20k, 16 threads). |
| Per-candidate nested `Vec<Vec<f64>>` + `Vec<bool>`/object allocation storm | Reusable fixed-size histogram scratch + incremental `leaf_of` (`TLearnContext` analogue) | This phase | Eliminates the ~10⁸ allocs/tree that OOM-killed the full Spike-002 grid. |

**Deprecated/outdated:**
- `assign_leaves` per candidate (`tree.rs:432`) as the scoring mechanism — retained only for the once-per-tree final `leaf_of` (`tree.rs:583`) and leaf-value estimation, not for candidate scoring.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | "bit-exact" (CONTEXT) means the shipped ≤1e-5 oracle suite keeps PASSING (not that new Rust output is byte-identical to old Rust output) — because the current object-order sum already differs from upstream's bucket-order sum yet passes. | §Parity-preserving summation | If the intent is strict byte-identity to the OLD Rust output, then only fixed-point-u64 with an unchanged combination order could approach it, and even then subtraction differs — planner must confirm the acceptance definition with the user before locking the summation strategy. |
| A2 | The ordered-boosting path (`greedy_tensor_search_oblivious_ordered` → `score_candidate_ordered`, `tree.rs:1523`) is NOT in the explicit CONTEXT scope list (oblivious/nonsym/CTR) and may remain on its per-segment rescan for this phase. | §Coverage / Open Questions | If ordered boosting is expected to also get the histogram treatment, scope grows (per-segment prefix histograms are feasible but non-trivial). Flag for the planner. |
| A3 | Only 2 histogram channels (Σder1, Σweight) are needed for CPU scoring; der2/NewtonL2/NewtonCosine are rejected at CPU train time (`split_score` doc, `tree.rs:49-54`; `validate_score_function`). | §Histogram data structure | If a CPU Newton score path were reachable, a 3rd channel (Σweighted_der2) would be required. Confirmed not reachable in current code. |
| A4 | rayon parallel-over-features with `collect` into a pre-sized `Vec` is deterministic and does not perturb per-bin folds. | §Parallelism | If a future global cross-feature reduction is added, order-independence (fixed-point-u64) becomes necessary. |

**All other claims are `[VERIFIED]` against the vendored upstream C++, in-repo source, or the spike evidence.**

## Open Questions

1. **Ordered-boosting path (A2).**
   - What we know: `greedy_tensor_search_oblivious_ordered` uses `score_candidate_ordered` (`tree.rs:1523`), which does `assign_leaves` + per-segment `reduce_leaf_stats` — the same rescan pathology, multiplied by segment count.
   - What's unclear: whether PERF-02 ("all policies … use it") is intended to include the ordered path or only the CONTEXT-listed oblivious/nonsym/CTR set.
   - Recommendation: planner asks the user; default to leaving ordered on its current path this phase (note the residual slowness for Phase 22), since per-segment prefix histograms are additional scope.

2. **Summation-strategy lock: `sum_f64`-prefix vs fixed-point-u64 (A1, Pitfalls 1–2).**
   - What we know: both are CONTEXT-sanctioned; `sum_f64`-prefix is the closest structural match to upstream; fixed-point-u64 makes subtraction/parallel-merge exact but changes exact bits.
   - What's unclear: whether any shipped fixture has a near-tie that flips under `sum_f64`-prefix.
   - Recommendation: implement `sum_f64`-prefix + upstream-order subtraction FIRST; run the full oracle suite; only if a specific fixture flips, evaluate fixed-point-u64 or fresh-rebuild for that path. Treat the suite as the empirical arbiter.

3. **Pairwise path fold-in (CONTEXT deferred).**
   - What we know: `greedy_tensor_search_oblivious_pairwise` (`boosting.rs:3837`) has a dedicated `TPairwiseScoreCalcer` (`scoring.cpp` `CalcStatsPairwise`, a 4-channel weight histogram — memory `phase74-pairwise-histogram-outcome`).
   - What's unclear: whether it "falls out cleanly" into the same 2-channel design (it does not — it is a distinct pairwise weight-only histogram).
   - Recommendation: EXCLUDE pairwise from this phase (per CONTEXT), note it retains its scorer.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `rayon` crate | PERF-03 parallelism | ✓ (crates.io) | 1.12.0 | Serial fallback (algorithm alone hits PERF-01) |
| Rust stable toolchain | build | ✓ | (workspace) | — |
| Official CatBoost 1.2.10 (Python, `.venv`) | PERF-01/03 acceptance benchmark (`catboost_grid.py`) | ✓ (used in Spike 002) | — | — |
| Vendored upstream C++ (`catboost-master/…/algo/`) | algorithm reference | ✓ | vendored | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** `rayon` — if parallelism is deferred, PERF-01 (the dominant win) still lands serially; PERF-03 then follows.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `approx` 0.5 (float assertions); oracle fixtures under `crates/cb-train/tests/fixtures/` |
| Config file | none (cargo test); perf harness gated by `CB_PERF` env var |
| Quick run command | `cargo test -p cb-compute histogram && cargo test -p cb-train tree::` |
| Full suite command | `cargo test -p cb-train` (60+ oracle test files) + `cargo test -p cb-compute` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PERF-01 | Per-tree time flat across `border_count` 32→254 (histogram fingerprint) | perf/bench | `CB_PERF=1 cargo test --release -p cb-train --test perf_baseline_test -- --nocapture` (re-run the n_bins sweep) | ✅ `perf_baseline_test.rs` |
| PERF-01 | Histogram raw score == old `score_candidate` raw score on a fixture (correctness of the binning/prefix) | unit | `cargo test -p cb-compute histogram` | ❌ Wave 0 (new equivalence unit test) |
| PERF-02 | Oblivious `SymmetricTree` parity preserved | oracle | `cargo test -p cb-train --test loss_oracle_test --test overfit_oracle_test --test regularization_oracle_test` | ✅ |
| PERF-02 | Depthwise/Lossguide parity | oracle | `cargo test -p cb-train --test non_symmetric_grower_oracle_test` | ✅ |
| PERF-02 | CTR-path parity | oracle | `cargo test -p cb-train --test plain_ctr_oracle_test --test tensor_ctr_oracle_test --test ctr_split_scoring_test` | ✅ |
| PERF-02 | Perturbed (`random_strength`) RNG-order parity | oracle | `cargo test -p cb-train --test penalty_oracle_test` + tie-break unit `cargo test -p cb-train tree::tie_break` | ✅ |
| PERF-02 | Multi-dim / multiclass parity (multi-channel `LeafStats`) | oracle | `cargo test -p cb-train --test multiclass_oracle_test --test multilabel_oracle_test` | ✅ |
| PERF-02 | Full regression gate (all shipped ≤1e-5 fixtures) | oracle | `cargo test -p cb-train` (whole suite green) | ✅ |
| PERF-03 | Before/after speedup documented on the Spike-002 grid | bench | `CB_PERF=1 cargo test --release -p cb-train --test bench_grow_speed_test -- --nocapture` | ✅ `bench_grow_speed_test.rs` |
| PERF-03 | Determinism under rayon (identical output single- vs multi-threaded) | unit/oracle | full oracle suite (run twice) — bins must not flicker | ✅ (reuse suite) |

### Sampling Rate
- **Per task commit:** `cargo test -p cb-compute histogram && cargo test -p cb-train tree::` (fast subset — histogram build + tie-break)
- **Per wave merge:** `cargo test -p cb-train` full oracle suite green (the PERF-02 gate)
- **Phase gate:** full suite green + PERF-01 flat-`n_bins` sweep + PERF-03 speedup number recorded, before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `crates/cb-compute/src/histogram_test.rs` — new: histogram raw-score equals old `score_candidate` raw-score on a small fixture (correctness before wiring selection; guards Pitfall 4).
- [ ] Determinism test: run a representative oracle twice with rayon enabled, assert byte-identical model (guards Pitfall 5).
- [ ] Framework install: `rayon = "1.12.0"` added to workspace + `cb-train`/`cb-compute` `[dependencies]`.

*(Existing 60+ oracle fixtures already cover every loss/policy/CTR combination — no new oracle fixtures needed; the gap is the equivalence + determinism unit tests.)*

## Security Domain

> `security_enforcement` is enabled (ASVS L1). This phase is a pure in-process numerical refactor with no external input, network, auth, storage, or serialization surface.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (no auth surface) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | partial | Histogram indices (bin, leaf, object) are internal; existing typed guards (`CbError::OutOfRange`/`LengthMismatch`, mirrored in `pointwise_hist.rs:635`) + `.get()` defensive access (deny-lint `indexing_slicing`) prevent OOB. New code must use the same pattern (Pitfall 6). |
| V6 Cryptography | no | — |

### Known Threat Patterns for CPU histogram code
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-range bin/leaf index → OOB read/write | Tampering / DoS | `.get()`/`.get_mut()` with defensive fallback (existing `reduce_leaf_stats` pattern, `histogram.rs:62-74`); no raw indexing (workspace deny-lint). |
| Integer overflow in histogram flat index (`n_features·n_bins·2^depth`) | DoS | `depth` is capped at `MAX_DEPTH=16` (`tree.rs:100`, `check_depth`); allocation bounded before the `2^depth` alloc (`tree.rs:28-30`). |
| Nondeterministic parallel accumulation (parity break, not a classic vuln but a correctness threat) | Tampering (integrity) | Feature-independent rayon work + fixed-order merge (Pitfall 5). |

## Sources

### Primary (HIGH confidence)
- Vendored upstream C++ `catboost-master/catboost/private/libs/algo/` — `calc_score_cache.h:72-95` (`TBucketStats` + `Add`/`Remove`), `scoring.cpp:315` (`FixUpStats` subtraction trick), `scoring.cpp:576-583` (`UpdateSplitScore`/`AddLeafPlain`), `CalcScoresForLeaf` prefix scan, `leafwise_scoring.h:11` (`TStatsForSubtractionTrick`), `score_calcers.cpp`/`.h` (L2/Cosine calcer math).
- In-repo source (read this session): `crates/cb-train/src/tree.rs` (`score_candidate:419`, `assign_leaves:396`, `select_level_plain:608`, `select_level_perturbed:662`, `best_split_for_leaf:819`, `leaf_wise_grower:923`, `score_candidate_ctr_aware:1768`, `score_candidate_ordered:1523`, `FeatureMatrix:319`, `Split:108`), `crates/cb-compute/src/histogram.rs` (`LeafStats:29`, `reduce_leaf_stats:49`, `reduce_leaf_der2:100`), `crates/cb-compute/src/score.rs` (`l2_split_score:49`, `cosine_split_score:73`, `multi_dim_split_score:110`), `crates/cb-core/src/reduction.rs:32` (`sum_f64`), `crates/cb-backend/src/kernels/pointwise_hist.rs:44-163` (frozen layout + `host_reference_hist2`), `crates/cb-backend/src/kernels/reduce.rs:654` (fixed-point-u64 pattern).
- Spike evidence: `.planning/spikes/002/003/004/README.md` (root cause + scaling + allocation).
- `docs/CATBOOST_CORE_DESIGN.md` §"The Tree-Growing Pipeline" step 3c + `TBucketStats`/`TLearnContext` (~lines 858–1023).
- `gsd-tools query package-legitimacy check --ecosystem crates rayon` → OK.

### Secondary (MEDIUM confidence)
- `cargo search rayon` → `rayon = "1.12.0"` (registry version confirmation).

### Tertiary (LOW confidence)
- none.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — `rayon` verified on crates.io; `sum_f64` is existing in-repo primitive.
- Architecture: HIGH — the target algorithm is grounded cell-for-cell in vendored upstream C++ and an existing in-repo device implementation (`pointwise_hist.rs`) that CONTEXT mandates mirroring.
- Pitfalls: HIGH — derived from the spike evidence, the existing parity discipline (D-05/D-08), and upstream's own subtraction/prefix behavior.
- Parity strategy (Q2): MEDIUM — the correct-in-principle approach is clear; the empirical tie-flip risk can only be settled by running the oracle suite (flagged as Open Question 2 / A1).

**Research date:** 2026-07-05
**Valid until:** ~2026-08-05 (stable domain; upstream C++ vendored and pinned, `rayon` API stable).

## RESEARCH COMPLETE

**Phase:** 21 - CPU Split-Finding Histogram Rewrite
**Confidence:** HIGH

### Key Findings
- The rewrite is a **data-production change behind an unchanged score interface**: `l2/cosine/multi_dim_split_score` (`cb-compute/src/score.rs`) are pure functions of `&[LeafStats]`; only the production of `LeafStats` moves from per-candidate object-order gather (`score_candidate`→`assign_leaves`→`reduce_leaf_stats`) to a per-level 2-channel `TBucketStats` histogram + `O(n_bins)` prefix scan. Score math, model format, and the leaf-value path stay untouched.
- The algorithm and the FROZEN bin layout `(feature*n_bins+bin)*2+channel` are already implemented on-device in `cb-backend/src/kernels/pointwise_hist.rs` (`host_reference_hist2`, lines 106-163) and match vendored upstream `TBucketStats` (`calc_score_cache.h:72`) + `FixUpStats` subtraction (`scoring.cpp:315`) — transcribe onto host into `cb-compute` (never depend on `cb-backend`).
- **The crux (Q2):** the current object-order sum already differs from upstream's bucket-order sum yet passes ≤1e-5, so a histogram+prefix scan is MORE parity-faithful, not less. Recommend per-bin `sum_f64` (object order) + buckets combined via `sum_f64` (ascending bin order) as primary; fixed-point-u64 as the fallback that makes subtraction/parallel-merge exact. The genuine risk is strict-`>` tie-flips in split selection — gated empirically by the full oracle suite.
- Only 2 histogram channels needed (Σder1, Σweight); NewtonL2/Cosine are rejected on CPU so no der2 channel; der2 stays on the unchanged `reduce_leaf_der2` leaf-value path.
- `rayon` 1.12.0 (verdict OK) is the parallelism dep; parallelize over INDEPENDENT features with ordered `collect` for determinism (PERF-03, after PERF-01). Ordered-boosting and pairwise paths flagged as out-of-scope/deferred (Open Questions 1 & 3).

### File Created
`.planning/phases/21-cpu-split-finding-histogram-rewrite/21-RESEARCH.md`

### Confidence Assessment
| Area | Level | Reason |
|------|-------|--------|
| Standard Stack | HIGH | rayon verified; sum_f64 existing primitive |
| Architecture | HIGH | grounded in vendored upstream C++ + existing device impl |
| Pitfalls | HIGH | from spikes + parity discipline + upstream behavior |
| Parity strategy | MEDIUM | tie-flip risk resolvable only by running the oracle suite |

### Open Questions
1. Is the ordered-boosting path (`score_candidate_ordered`, `tree.rs:1523`) in PERF-02 scope, or does it stay on its per-segment rescan this phase?
2. Lock the summation strategy (`sum_f64`-prefix vs fixed-point-u64) after running the suite — near-tie flips are the deciding empirical signal.
3. Pairwise scorer stays excluded (CONTEXT deferred) — confirmed.

### Ready for Planning
Research complete. Planner can now create PLAN.md files (suggested waves: (1) oblivious histogram+subtraction+parity, (2) Depthwise/Lossguide + CTR, (3) rayon + scratch buffers).
