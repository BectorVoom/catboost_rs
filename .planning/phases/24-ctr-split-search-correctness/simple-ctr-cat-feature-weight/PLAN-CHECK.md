## Plan Check Result

**Verdict:** PASS
**Goal:** ORD-07 — make `select_level_ctr_aware`'s `max_bucket_count` include
upstream's "phantom" mixed float-partition + categorical-feature projection
bucket count (gated on `chosen` containing `>= 1` `Float` split), matching
`CalcMaxFeatureValueCount`'s real behavior, so `fstr_ctr_oracle_test.rs`'s
tree0/level2 divergence resolves, without perturbing ORD-06's already-landed
`combination_ctr_eligible`/`eligible_max_bucket_count` fix or any currently
green CTR/float regression fixture.
**Plan:** `.planning/phases/24-ctr-split-search-correctness/simple-ctr-cat-feature-weight/PLAN.md`
(spec: `SPEC.md`, evidence: `SOURCES.md`, research: `research.md`) — this is
**pass #3, the final allowed re-review pass.**

---

### Prior-pass history (retained for context; not re-litigated except where noted)

**Pass #1 (`ISSUES_FOUND`, 2 CRITICAL + 2 MAJOR + 2 MINOR):**
1. [CRITICAL] T4 claimed a nonexistent "7th call site in a numeric path" for
   `greedy_tensor_search_oblivious_with_ctr` → corrected to the true 6 real
   call sites (1 production + 5 in `ctr_split_scoring_test.rs`).
2. [CRITICAL] SPEC.md §8 wrongly claimed `greedy_tensor_search_oblivious_with_ctr`
   was private → corrected to document its true `pub`/re-exported/6-caller
   status.
3. [MAJOR] SPEC.md's level-1 "almost exact tie" claim rested on an unsourced
   `3.844592` raw score → downgraded/retracted in SPEC.md §1/§9.
4. [MAJOR] T3 originally proposed hand-rolling a new
   `candidates.rs::learn_set_buckets` duplicating `cb_data::perfect_hash_bins`
   byte-for-byte → rewritten to reuse `perfect_hash_bins` directly.
5. [MINOR] A type mismatch in the (now-deleted) hand-rolled function's unit
   test → moot after finding #4's fix.
6. [MINOR] T4 lacked a performance acknowledgment for a doubled per-level
   `assign_leaves_ctr_aware` call → added.

**Pass #2 (`ISSUES_FOUND`, 2 MAJOR + 2 MINOR — all documentation-consistency,
not core-logic, gaps):**
1. [MAJOR] `PLAN.md`'s top-of-file "Regression discipline" callout and
   `SPEC.md` §7's "Must change" bullet still described the discarded
   hand-rolled-`PerfectHash`-reuse design (pass #1 finding #4's fix wasn't
   propagated to these two passages) → **now rewritten** to state the
   `cb_data::perfect_hash_bins`-reuse design.
2. [MAJOR] `SOURCES.md` and `research.md` still asserted the retracted
   `3.844592`-based level-1 claim and the discarded hand-rolled-`learn_set_buckets`
   T3 design as solid/current (pass #1 finding #3/#4's fixes weren't
   propagated to these sibling documents) → **now updated** with explicit
   `[CORRECTED, plan-checker pass 2]` / `[UNVERIFIED — RETRACTED]` markers.
3. [MINOR] SPEC.md §7's "May change" bullet claimed `perfect_hash_bins` was
   "already imported" at `candidates.rs:43`/`boosting.rs:35` — those lines
   import different `cb_data` symbols → **now reworded** to say only the
   `cb_data` module is already a dependency there.
4. [MINOR] `PLAN.md`'s traceability table cited a dangling `AT-T3c` that
   T3's body never defines → **now changed** to `AT-T3a (+ optional AT-T3b)`.

---

### Pass #3 — independent re-verification (this session)

All four required revisions from pass #2 were independently re-checked
against the **current on-disk text** of all four documents (not the prior
review's summary), plus fresh CodeGraph/grep verification of every
structural claim. Findings below.

#### (a) PLAN.md top-of-file callout / SPEC.md §7 "Must change" — CONFIRMED FIXED

- `PLAN.md`'s "Regression discipline" paragraph (lines 50-60) now reads:
  *"the new per-object bucket data must NOT come from a new hand-rolled
  `PerfectHash`/`calc_cat_feature_hash` loop either (plan-checker pass 1
  MAJOR finding): it comes from calling the ALREADY-EXISTING, already-`pub`,
  already-oracle-tested `cb_data::perfect_hash_bins` DIRECTLY (T3), never a
  second, independently-maintained hashing implementation."* No remaining
  "NEW sibling function reusing the SAME PerfectHash primitives" language.
- `SPEC.md` §7 "Must change" (lines 425-432) now reads: *"`train_inner`'s
  `has_ctr` call site supplies the new raw per-object cat-bucket data by
  calling the ALREADY-EXISTING `cb_data::perfect_hash_bins` directly — see
  'May change' below; do NOT hand-roll a new `PerfectHash`-based hashing
  loop, per plan-checker pass 1's MAJOR finding."*
- Grepped both documents for any other "sibling function"/"reusing
  learn_set_cardinality's existing PerfectHash machinery" phrasing — none
  found outside the explicitly-historical/retraction context (SOURCES.md's
  "an earlier version... on review this was found to duplicate" passage,
  and PLAN.md's own T3 "revised per plan-checker pass 1" framing, and the
  optional thin-wrapper's legitimate `learn_set_buckets` delegating stub in
  T3 — which is the *allowed* design, not the anti-pattern).
- **Confirmed: no remaining language suggesting a new hand-rolled `PerfectHash`
  loop.**

#### (b) SOURCES.md / research.md level-1 retraction — CONFIRMED FIXED

- `grep -rn "3.844592"` across all 4 documents returns 9 occurrences total
  (research.md ×3, SOURCES.md ×2, SPEC.md ×2, PLAN-CHECK.md's own historical
  recap ×2 — now superseded by this rewrite). Read each in context:
  - `SPEC.md:114-118` — inside the explicit `[UNVERIFIED — plan-checker pass
    1 finding, downgraded...]` retraction paragraph.
  - `SOURCES.md:67,70` — inside the `[CORRECTED, plan-checker pass 2]`
    paragraph explicitly retracting the claim.
  - `research.md:676,678,682` — inside the `[UNVERIFIED — RETRACTED,
    plan-checker pass 2 finding]` paragraph in the Addendum.
  - **Every occurrence of `3.844592` in the live document set is inside an
    explicit retraction/correction note; none is asserted as live evidence.**
- Separately checked research.md's "STRONG, coherent fit" framing
  (`research.md:666`, `"This table's LEVEL-2 entry is a STRONG, coherent,
  fully-sourced fit..."`) — this phrase is scoped explicitly to the
  **level-2 entry only** (confirmed by re-reading the full paragraph: the
  header sentence is immediately followed by a **per-level** breakdown where
  the level-1 bullet is the retraction paragraph itself, and the level-2
  bullet is the standalone supporting claim). This is not a stale overclaim
  — it is accurately scoped.

#### (c) T3-design descriptions in SOURCES.md / research.md — CONFIRMED FIXED

- `SOURCES.md:107-118` ("Rust seams" section): now reads *"**[CORRECTED,
  plan-checker pass 1]** An earlier version of this ledger proposed
  capturing this via a NEW `candidates.rs::learn_set_buckets` sibling
  function... On review, this was found to duplicate an ALREADY-EXISTING...
  **The corrected design (T3, `PLAN.md`) calls `cb_data::perfect_hash_bins`
  directly — no new hand-rolled hashing function is written.**"*
- `research.md`'s Addendum (lines 703-717) describes the NEW quantity as
  "PARTITION-SCOPED... exactly mirroring `CalcMaxFeatureValueCount`'s real
  upstream behavior" without asserting a hand-rolled implementation
  mechanism — consistent with (not contradicting) the corrected T3 design.
- **Confirmed: both documents now match the `cb_data::perfect_hash_bins`-reuse
  approach, not the discarded hand-rolled `learn_set_buckets` duplicate.**

#### (d) SPEC.md §7 import-location wording — CONFIRMED ACCURATE (CodeGraph-verified)

Re-verified via direct read (not just CodeGraph) of both cited files:
- `crates/cb-train/src/candidates.rs:43` → `use cb_core::CbResult;` is line
  42; line 43 is `use cb_data::{calc_cat_feature_hash, PerfectHash};` —
  matches SPEC.md's claim exactly (imports `calc_cat_feature_hash`/
  `PerfectHash`, NOT `perfect_hash_bins`).
- `crates/cb-train/src/boosting.rs:35` → `use cb_data::Pair;` — matches
  SPEC.md's claim exactly.
- `grep -rn "perfect_hash_bins" crates/cb-train/` returns **zero** matches
  anywhere in `cb-train` today — confirming SPEC.md's claim that
  `perfect_hash_bins` itself is "not yet imported anywhere in `cb-train` and
  needs a new `use` statement at the call site."
- Confirmed `perfect_hash_bins` is defined at `crates/cb-data/src/cat_hash.rs:471-479`
  (`pub fn perfect_hash_bins(column: &[&str]) -> CbResult<Vec<u32>>`) and
  exported at `crates/cb-data/src/lib.rs:39` (exact line, confirmed via
  `grep -n`) inside a `pub use cat_hash::{...}` block spanning lines 38-40.
- **Confirmed: SPEC.md §7's wording is now accurate** — it correctly states
  only that the `cb_data` MODULE is already a dependency of `cb-train` at
  those two files (for other symbols), and that `perfect_hash_bins` itself
  needs a new `use` statement.

#### (e) Traceability table `AT-T3c` — CONFIRMED FIXED

- `PLAN.md`'s traceability table (line 463) now reads:
  `| T3 | (enabler for ORD-07-03) | AT-T3a (+ optional AT-T3b) | unit |`.
- `grep -rn "AT-T3c" PLAN.md SPEC.md SOURCES.md research.md` returns zero
  matches in any of the four live documents (the only remaining `AT-T3c`
  occurrences are inside this PLAN-CHECK.md's own pass-#2 historical recap
  of the finding, which is expected).
- **Confirmed fixed.**

#### (f) Fresh full-sweep for other inconsistencies — no material findings

Re-derived, from first principles against the live `crates/cb-train/src/tree.rs`
source (not the plan's prose), the full T0-T6 call/data-flow and cross-checked
every numeric/behavioral claim:

- **`CtrAwareSplit` enum** (`tree.rs:2238-2243`): `Float(Split)` /
  `Ctr { col: usize, border: f64 }` — matches ORD-07-02's Given/When/Then and
  T2's `matches!(s, CtrAwareSplit::Float(_))` implementation exactly.
- **`select_level_ctr_aware`** (`tree.rs:2588-2717`, full body read): private
  `fn`, no `pub`, exactly 1 caller (`greedy_tensor_search_oblivious_with_ctr`
  at `tree.rs:2764`) — matches SPEC §8/PLAN T0's visibility claim exactly.
  `eligible_max_bucket_count(ctr_features, &used_projections)` is computed at
  line 2653, immediately BEFORE the CTR candidate loop (line 2659) that
  consumes `max_bucket_count` in `cat_feature_weight(...)` at line 2683 — T4's
  instruction to insert the new phantom-contribution code "AFTER ORD-06-04's
  `eligible_max_bucket_count` call" places it correctly between lines 2653
  and 2659, i.e. before the value is ever consumed. No ordering bug.
- **`greedy_tensor_search_oblivious_with_ctr`** (`tree.rs:2747`): confirmed
  `pub fn`, re-exported at `crates/cb-train/src/lib.rs:104`. `grep -rn
  "greedy_tensor_search_oblivious_with_ctr" crates/` independently re-run
  this session returns exactly the same 6 real call sites the plan
  enumerates (`boosting.rs:3900` production; `ctr_split_scoring_test.rs:99,
  147,189,246,301`) plus 2 non-call references (an import line and a
  comment) — the plan's "6 real call sites, not 7" framing is exactly
  correct; CodeGraph's raw "8 callers" blast-radius count includes the
  import/re-export lines, which the plan correctly excludes from the
  call-site-update requirement.
- **`forward_bit_leaf_index_mixed_float_and_ctr`** (`ctr_split_scoring_test.rs:172-210`,
  read in full): confirmed it chooses `Float` at level 0 then `Ctr` at level
  1 (`assert_eq!(grown.splits.len(), 1); assert_eq!(grown.ctr_splits.len(), 1);`
  with the comment explicitly walking through the L0/L1 score arithmetic) —
  exactly matching PLAN.md T4's claim that `phantom_bucket_gate` evaluates
  `true` for this specific existing test at level 1.
- **Arithmetic self-consistency of the 3-level worked example** (re-derived
  independently, not just checked against the plan's own table): at level 0,
  `chosen=[]` → gate `false` → `max_bucket_count = eligible_max_bucket_count = max(5,4) = 5` (unchanged). At level 1, `chosen=[Float(1)]`, no `Ctr` chosen yet
  so `used_projections` stays empty → `eligible_max_bucket_count` still `=
  max(5,4) = 5`; phantom counts `{10,8}` → combined `max(5,10) = 10`. At level
  2, `chosen=[Float(1),Float(0)]`, still no `Ctr` chosen → `eligible_max_bucket_count`
  still `5`; phantom counts `{20,16}` → combined `max(5,20) = 20`. This
  independently reproduces SPEC §5/PLAN T4's claimed values (5, 10, 20)
  exactly — the "Combined `max_bucket_count`" formula in SPEC §4 and its 3
  worked Given/When/Then scenarios in ORD-07-03 are internally consistent.
  `cat_feature_weight`'s formula (`tree.rs:2416-2422`,
  `(1+count/max_count)^(-model_size_reg)`) applied at level 2 with
  `count=5, max_count=20, model_size_reg=0.5` gives `(1.25)^-0.5 = 0.894427`,
  matching SPEC §1/§9's cited value exactly.
- **`cb_data::perfect_hash_bins` oracle test**: confirmed
  `crates/cb-data/tests/cat_hash_oracle_test.rs:56`
  (`cat_hashes_and_perfect_hash_bins_match_oracle`) exists, calls
  `perfect_hash_bins` (line 91), and the fixture
  `crates/cb-oracle/fixtures/cat_hash/perfect_hash_bins.npy` exists on disk —
  T3's AT-T3a is a real, currently-passing acceptance artifact, not a
  fabricated citation.
- **`fstr_ctr` oracle fixture and test**: confirmed all cited fixture files
  exist (`X_float.npy`, `X_cat.npy`, `model.json`, etc.) and
  `crates/cb-model/tests/fstr_ctr_oracle_test.rs` contains exactly the 3
  named tests (`fstr_ctr_predictions_sanity_gate`,
  `interaction_matches_upstream_on_mixed_ctr_model`,
  `pvc_matches_upstream_on_mixed_ctr_model`) cited throughout SPEC/PLAN.
- **`boosting.rs` plumbing origin**: `cat_cardinalities`/`eligible_absolute`
  computed at lines 2698-2725 (close to, and consistent with, the plan's
  "~2696-2721" citation — T0 already instructs re-confirming exact lines
  before starting, an appropriate safeguard for citation drift); `has_ctr`
  branch calling `greedy_tensor_search_oblivious_with_ctr` at line 3900,
  confirmed.
- One (non-blocking) observation not raised in prior passes: T4's Green
  step 1 prose illustrates the new per-object bucket assembly with
  `cb_data::perfect_hash_bins(&cat_columns[abs_idx]...)` — literal
  bracket-indexing notation, whereas the plan's own header mandates "No
  `unwrap`/`expect`/`panic`/`indexing_slicing` in production" and the
  existing pattern it says to reuse (`cat_columns.iter().map(...)` at
  `boosting.rs:2698-2701`) itself uses iterator-based access, never raw
  indexing. This is inline descriptive prose, not a literal code block, and
  `abs_idx` values are guaranteed in-range (derived via `enumerate()` over
  `cat_cardinalities`, which has the same length as `cat_columns` by
  construction), so this could not actually panic even if implemented
  literally — but an implementer copying the bracket notation verbatim
  would trip the `cargo clippy -p cb-train --all-targets` restriction-lint
  gate T4/T6 both name as the authoritative acceptance bar. See Potential
  Bugs below; downgraded to MINOR since the clippy gate is the plan's own
  named enforcement mechanism and would catch it before T6 completes.

No other cross-document contradictions, dangling references, stale line
citations, or scope mismatches were found. The corrected documents read
coherently end-to-end: an implementer starting fresh at T0 and proceeding
through T1-T6 would encounter a single, consistent design (reuse
`cb_data::perfect_hash_bins`, gate on `>= 1` chosen `Float` split, extend
`max_bucket_count` via an additive outer `max(...)`, update exactly 6 real
call sites) with no passage anywhere in the 4-document set pointing back at
either of the two now-discarded approaches (hand-rolled hashing duplicate;
overstated level-1 arithmetic).

---

### Specification Coverage

- [x] ORD-07-01 (`phantom_mixed_bucket_count` distinct-pair counting): T1
  maps directly; pure function, `.zip`-based (no indexing/panic), 4
  Given/When/Then unit scenarios all mapped to named tests.
- [x] ORD-07-02 (gating on `chosen` containing `>= 1` Float): T2 maps
  directly; `CtrAwareSplit::Float`/`::Ctr` discriminant confirmed to exist
  exactly as used.
- [x] ORD-07-03 (`max_bucket_count` extension + plumbing): T4 wires T1/T2's
  primitives into `select_level_ctr_aware`, correctly enumerates and updates
  all 6 real call sites (independently re-verified via grep this session),
  and the worked 3-level arithmetic (5/10/20) is internally consistent and
  independently re-derivable from the formula in SPEC §4.
- [x] Oracle re-verification (AT-ORD07-03b/c) and provable-no-op regression
  targets: T5 correctly identifies `tensor_ctr_e2e`/`multi_permutation_e2e`
  (zero float features) and `ctr_split_scoring_test.rs` (`model_size_reg=0.0`
  in all pre-existing tests, one test with a Float-then-Ctr sequence that
  still numerically no-ops) as structurally-guaranteed no-op targets.
- [x] Full-slice regression + clippy gate: T6.
- [x] Documentation/evidence-chain internal consistency (pass #2's own
  finding category): re-verified fixed in items (a)-(e) above.

### CodeGraph Evidence

- `select_level_ctr_aware` in `crates/cb-train/src/tree.rs:2588`
  - Definition: private `fn` (no `pub`), full body re-read this session.
  - Callers: exactly 1 (`greedy_tensor_search_oblivious_with_ctr`).
  - Callees: `build_ctr_aware_histogram`, `score_candidate_ctr_aware`,
    `eligible_max_bucket_count`, `cat_feature_weight`, `combination_ctr_eligible`
    (unchanged by this fix, per SPEC's non-goals).
  - Impact assessment: signature gains one new parameter; low risk given
    single, private caller.

- `greedy_tensor_search_oblivious_with_ctr` in `crates/cb-train/src/tree.rs:2747`
  - Definition: `pub fn`, re-exported `crates/cb-train/src/lib.rs:104`.
  - Callers (re-verified via grep this session, exact match to plan): 1
    production (`crates/cb-train/src/boosting.rs:3900`) + 5 test call sites
    (`crates/cb-train/tests/ctr_split_scoring_test.rs:99,147,189,246,301`).
  - Impact assessment: PUBLIC API signature change with a small, fully
    enumerated, verified blast radius; all 6 sites are named in T4.

- `cb_data::perfect_hash_bins` in `crates/cb-data/src/cat_hash.rs:471-479`
  - Definition: `pub fn perfect_hash_bins(column: &[&str]) -> CbResult<Vec<u32>>`.
  - Export: `crates/cb-data/src/lib.rs:39`, confirmed exact line via `grep -n`.
  - Callers today: 0 in `cb-train` (confirmed via `grep -rn "perfect_hash_bins"
    crates/cb-train/` — zero matches), consistent with SPEC's claim that a
    new `use` is required.
  - Test coverage: `crates/cb-data/tests/cat_hash_oracle_test.rs:56`
    (`cat_hashes_and_perfect_hash_bins_match_oracle`), against
    `crates/cb-oracle/fixtures/cat_hash/perfect_hash_bins.npy` (confirmed on
    disk).
  - Impact assessment: safe, already-proven reuse target; T3's design is
    sound and non-duplicative.

- `crates/cb-train/src/candidates.rs:43` / `crates/cb-train/src/boosting.rs:35`
  - Confirmed via direct read: `use cb_data::{calc_cat_feature_hash,
    PerfectHash};` and `use cb_data::Pair;` respectively — SPEC.md §7's
    corrected wording (module already a dependency, symbol not yet imported)
    is accurate.

- `CtrAwareSplit` in `crates/cb-train/src/tree.rs:2238-2243`
  - `Float(Split)` / `Ctr { col: usize, border: f64 }` — matches ORD-07-02's
    contract and T2's implementation exactly.

- `cat_feature_weight` in `crates/cb-train/src/tree.rs:2416-2422`
  - `(1.0 + ratio).powf(-model_size_reg)`, `ratio = count/max_count` —
    independently re-applied to the plan's level-2 worked example
    (`count=5, max_count=20, model_size_reg=0.5`) and confirmed to yield
    `0.894427`, matching SPEC §1/§9's cited figure exactly.

### Issues

No BLOCKER, CRITICAL, or MAJOR issues found this pass. See "Potential Bugs"
below for one MINOR, non-blocking observation.

### Implementation Order Review

Unchanged from passes #1/#2 and re-confirmed sound: T0 (source
re-verification, catches citation drift before any edit) before T1/T2/T3
(parallel, independent pure-function/reuse-verification tasks — T1/T2 land
in the same file and should be serialized at the file-edit level per the
plan's own note, even though logically independent) before T4 (integration
wiring — correctly gated on all three predecessors, correctly enumerates
all 6 real call sites, correctly inserts the new contribution between the
existing `eligible_max_bucket_count` computation and its first consumption)
before T5 (oracle verification, strictly gated on T4) before T6 (full
regression + clippy gate, strictly gated on T5). No circular dependencies,
no task requires an artifact not yet produced by an earlier task, and no
intermediate state leaves the crate unable to build (T4's own validation
step explicitly runs `cargo build -p cb-train` to catch any missed call
site before proceeding).

### Potential Bugs

- **T4 Green step 1's pseudocode uses bracket-indexing notation
  (`cat_columns[abs_idx]`)** rather than the project's established
  checked-access idiom (`.get(abs_idx)`), inconsistent with the plan's own
  stated "no `indexing_slicing`" mandate and with the exact existing pattern
  (`cat_columns.iter().map(...)`, `boosting.rs:2698-2701`) it instructs the
  implementer to reuse. **Trigger:** an implementer copies the bracket
  notation literally. **Failure mode:** `cargo clippy -p cb-train
  --all-targets` (the plan's own named restriction-lint gate, run in T4's
  validation block and again in T6) fails, blocking task completion — not a
  silent runtime defect, since `abs_idx` is always in-range by construction
  (`eligible_absolute` values come from `enumerate()` over `cat_cardinalities`,
  which has the same length as `cat_columns`). **Impact:** at most a
  clippy-gate failure requiring a one-line fix (`.get(abs_idx)` +
  `CbResult`/`.ok_or_else` handling, mirroring the file's existing idiom
  throughout `build_ctr_aware_histogram`); does not affect correctness of
  the shipped fix if caught, and the plan's own T4/T6 validation commands
  are positioned to catch it before completion is claimed. **Required
  mitigation (non-blocking for this review's verdict, but should be applied
  during implementation):** when writing T4's actual code, use
  `cat_columns.get(abs_idx)` with a checked fallback/error path (matching
  `tree.rs`'s existing `.get(...)`-based idiom throughout `build_ctr_aware_histogram`),
  not direct bracket indexing.
- T1/T2's pure-function designs remain sound: `.zip`-based truncation (no
  panic on length mismatch), `HashSet`-based distinct counting (no overflow
  risk at this scale), `.iter().any(...)` gating (empty-slice-safe).
- T4's `cat_eligible_buckets.iter().map(...).max().unwrap_or(0)` is safe for
  an empty `cat_eligible_buckets` (the 5 external test call sites passing
  `&[]`) — falls back to `0`, correctly matching "no CTR-eligible cat
  features, no phantom contribution" semantics; no panic.
- The doubled per-level `assign_leaves_ctr_aware` call (once inside
  `build_ctr_aware_histogram`, once directly in the new phantom-contribution
  code) is confirmed pure/deterministic (both calls receive byte-identical
  `matrix`/`ctr_features`/`chosen`/`n_objects` arguments at the same call
  site), and is additionally gated to zero cost at level 0 via
  `phantom_bucket_gate` — acceptable, already acknowledged in the plan.

### Required Plan Revisions

None required to reach PASS. The one MINOR observation above (T4's
pseudocode indexing notation) is a documentation-clarity nit that the
plan's own clippy gate would catch during implementation; it does not need
a plan-text change to proceed safely, but implementers should default to
`.get(...)` per the file's established idiom rather than the plan's
shorthand bracket notation.

### Unverified Items

None. Every structural claim examined this pass (6 real call sites for
`greedy_tensor_search_oblivious_with_ctr`; `select_level_ctr_aware`'s
privacy/single-caller status; `perfect_hash_bins`'s definition, export,
zero current importers in `cb-train`, and oracle-test coverage;
`candidates.rs:43`/`boosting.rs:35`'s actual imports; the `CtrAwareSplit`
enum shape; `cat_feature_weight`'s formula and the level-2 worked-example
arithmetic; the `AT-T3c` table fix; the `3.844592` retraction's scope
across all 4 documents; the `fstr_ctr` fixture/test file existence and the
3 test names) was independently re-verified this session via CodeGraph,
direct file reads, and targeted `grep`, and matched the plan's claims.
