# Phase Research: Complete Device (GPU) CTR Coverage

**Worktree**: `/home/user/Documents/workspace/catboost_rs/.claude/worktrees/gpu-borders-shared-sample`
(branch `worktree-gpu-borders-shared-sample`, HEAD `a0a67ec`) `[VERIFIED: git log --oneline -3]`
**Research date**: 2026-08-08. **Upstream parity target**: `catboost==1.2.10`
(installed at `/home/user/.local/lib/python3.14/site-packages/catboost`,
`[VERIFIED: python3 -c "import catboost; print(catboost.__version__)"` → `1.2.10]`)
and the `v1.2.10` git tag at github.com/catboost/catboost (fetched raw this
session; see **Sources**).

---

## ⚠️ SECOND PASS — Coordinator Rulings + Measured Spike Results (2026-08-08)

This block supersedes the first-pass conclusions it names. Everything else below
stands.

### Rulings received

- **Q1 (scope) → "Everything — all nine clauses."** The phase is the five tracks
  **plus** every one of the nine surviving CTR-conditioned declines. Researched
  in the new section **"Newly In-Scope Clauses (Q1)"** below. **Verdict summary:
  6 IN, 3 OUT-with-evidence, and 2 of the 6 turn out to be provably dead
  clauses.**
- **Q4 (CPU BTMV normalization) → "Fix it in this phase."** Now **Track E**.
  Measured verdict: **the fix is a zero-risk no-op for every artifact in the
  repository** — no fixture re-baseline needed.

### Measured spike results (both run on the gfx1151 rig this session)

**Spike Q2 — BTMV quantizer. `[VERIFIED: python3 q2_spike_fast.py`, 4,504,501
`(good, total)` pairs per prior, `total ∈ [0, 3000]`, `border_count = 15]`**

| prior | shift, norm | f64 border table vs repo BTMV bin | f32 border table vs repo BTMV bin | upstream `CalcCTR` vs repo BTMV bin | upstream `CalcCTR` vs repo f64 bin |
|---|---|---|---|---|---|
| 0.0 | 0, 1 | **0** | **0** | **0** | **0** |
| 0.5 | 0, 1 | **0** | **0** | **0** | **0** |
| 1.0 | 0, 1 | **0** | **0** | **0** | **0** |
| 2.0 | 0, 2 | 4 208 787 | 4 208 787 | 4 208 787 | **0** |
| −0.5 | 0.5, 1.5 | 4 057 196 | 4 057 196 | 4 057 196 | 899 |

**Q2 is ANSWERED: BTMV needs NO dedicated f32 quantize kernel.** For every prior
with `shift = 0 ∧ norm = 1` — i.e. every prior in `[0, 1]`, which is every prior
reachable in this repository — the shipped f64 `binarize_ctr_kernel` + border
table, an f32 border table, upstream's `CalcCTR`, and the repo's BTMV f32
quantizer **all agree bit-for-bit over 4.5M pairs**. **Risk R-3 is REFUTED.**
Track B collapses from "new quantize kernel" to "new accumulator only".

The prior=2.0 / prior=−0.5 columns are **not** a rounding effect — they are
exactly **R-6** (the repo's BTMV arm omits `(ctr + shift)/norm`), which is
numerically invisible iff `shift = 0 ∧ norm = 1`. The prior=−0.5 row's
`899` in the last column is **R-7** (f32-vs-f64 `CalcCTR` width), which is
**zero for every prior in `[0, 2]`** and only appears at negative priors.

**Every prior in the entire repository is in `[0, 1]`**:
`{Prior=0, Prior=0.25, Prior=0.5, Prior=1}` across all fixture `config.json`s,
and `{vec![0.25], vec![0.5]}` across all Rust tests
`[VERIFIED: grep -rho "Prior=[-0-9./]*" crates/cb-oracle/fixtures/*/config.json | sort -u`
and the `ctr_priors: vec![…]` grep`]`. ⇒ **Track E (the R-6 fix) changes no
committed artifact; no fixture re-freeze task is needed.** R-7 is likewise inert
today and is downgraded to a documented hazard for user-supplied negative priors.

**Spike Q5 — FPP-11 root cause. `[VERIFIED: scratch edit + 3 measured arms on
gfx1151, then reverted]`**

| Arm | Change | `ctr_device_combo` e2e result |
|---|---|---|
| **Control** | arity gate opened, **no** eligibility gate (today's device behaviour) | **FAIL** — obj 0 `\|Δ\|=2.746e-2` vs the 1e-5 bar |
| **D-1 only** | + per-level combination-eligibility gate on the candidate loop | **PASS — max \|Δpred\| = 2.082e-17** |
| **D-1 + D-2** | + `eligible_max` filtered by the same predicate | **PASS — max \|Δpred\| = 2.082e-17** |
| Reverted | worktree restored | back to `1.388e-17` (the CPU number) |

**FPP-11 is CONFIRMED FIXED, measured not hypothesized.** The control arm
reproduces the documented ~3.3e-2 failure class; adding the gate closes it to
`2.082e-17`. Note the fixed number `2.082e-17` **differs** from the CPU fallback
number `1.388e-17`, which is independent evidence the device path actually ran.
The **full 23-binary device suite stayed green with the fix applied**
`[VERIFIED: bash ./run_device_tests.sh under the spike → 23/23 PASS + perf 10.7×]`.

**Nuance the planner must not lose**: **D-1 alone was sufficient on this
fixture**; D-2 was not discriminated by it. D-2 is still **required** for CPU
parity (`eligible_max_bucket_count`, `tree.rs:2920-2933`, applies the same
filter), so it must ship — but `ctr_device_combo` is **not its detector**. A
D-2-discriminating fixture needs a combination whose `bucket_count` exceeds every
simple column's *and* a level-0 decision on a thin margin. Treat "D-2 has no
detector" as an explicit test-coverage gap, not as licence to skip D-2.

**Scratch-edit disclosure.** The Q5 spike touched four production files —
`crates/cb-compute/src/runtime.rs` (added `DeviceCtrColumn.projection_members`),
`crates/cb-train/src/boosting.rs` (populated it; removed the
`col.projection.is_simple()` conjunct), `crates/cb-backend/src/gpu_runtime/session.rs`
(threaded it through `CtrSearchState` → `ResidentCtrSearch`), and
`crates/cb-backend/src/gpu_runtime/mod.rs` (added `spike_combination_eligible`,
the per-tree `chosen_ctr_projections` list, the pass-C skip, and the
`eligible_max` filter). **All four were reverted with `git checkout --`.**
Worktree state now: `git status --short` reports **only** the untracked
`?? .planning/plans/device-ctr-full-coverage/` (this report); HEAD is still
`a0a67ec` `[VERIFIED]`. A post-revert re-run of the combo test returns
`1.388e-17`, confirming the pre-spike behaviour is restored `[VERIFIED]`.

### First-pass claims this block supersedes

- **R-3 (BTMV border table cannot work) — REFUTED.** Track B needs no new
  quantize kernel; reuse `binarize_ctr_kernel` + the existing border table.
- **Q2 — CLOSED.** No longer an open question.
- **Q5 / Track D — CLOSED and measured.** No longer a hypothesis.
- **Q4 / R-6 — now Track E**, and verified inert for every repo artifact.
- **"`border_count == 15` is an architecture question"** — **downgraded**. The
  padding infrastructure that supports catboost's default 254 borders **already
  exists** for float features (`pad_hist_line_bins`, `session.rs:1010-1018`, whose
  own comment names "the CatBoost default 254 borders → 255 bins"). See C-1.

---

## Research Summary

- **Phase goal.** Remove every CTR clause from the CPU-decline path so that every
  CTR configuration *this repository's CPU trainer supports* commits to the
  rocm/cuda device grower at ≤1e-5 vs upstream `catboost==1.2.10`. The single
  gate is `ctr_types_are_device_covered`
  (`crates/cb-train/src/boosting.rs:2355-2366`), which today admits only
  `projection.is_simple() && ctr_type == Borders && target_border_idx == 0 &&
  prior_denom == 1.0` `[VERIFIED: read boosting.rs:2355-2366]`.

- **Recommended approach — and the single biggest planning correction.**
  The five requested tracks do **not** have equal weight, and two of them are
  *already satisfied or structurally unreachable*:

  | Track (as requested) | Reality established this session | Recommended treatment |
  |---|---|---|
  | 1a. Non-unit prior **denominator** | **DEAD CLAUSE.** `materialize_ctr_columns_for_perm` hard-codes `CTR_PRIOR_DENOM = 1.0` at the *only* production materialization site (`boosting.rs:2237-2238`, const at `:2257`), so `CtrFeatureColumn.prior_denom` is **always** `1.0`. Upstream forbids it too: `CB_ENSURE(denom == 1.0, "Error: CPU could use only 1 as denom for ctrs currently")` (`ctr_helper.cpp:50`, v1.2.10). The Python surface rejects it (`catboost-rs-py/src/params.rs:750-756`). And `build_device_ctr_config` *already* divides `prior_num/prior_denom` and derives `(shift,norm)` from that scalar (`boosting.rs:2503-2510`), so even a hypothetical non-unit denom is already correct on the device. | **Delete the conjunct as a proven no-op**, with a characterization test. **Zero kernel work.** |
  | 1b. Multi-**target-border** | Only `Buckets` ever produces `target_border_idx > 0`: `ECtrType::target_border_count(2)` returns `2` for Buckets and `1` for every other CPU-legal type (`ctr/mod.rs:137-146`), and the candidate expansion loops `0..target_border_count` (`boosting.rs:2228`). | **Merge into Track 2** — it is not a separate work item. |
  | 2. **Buckets** | Small, purely additive kernel delta: the existing serial prefix kernel already keeps a 2-class `[N0,N1]` bucket history; Buckets@b is `counts[b]`, Borders@b is `total − Σ_{c≤b} counts[c]`. No new buffer, no f64 change. | **Track A** — smallest real delta. Do first. |
  | 3. **BinarizedTargetMeanValue** | ~~Needs a new quantize kernel~~ — **REFUTED by spike Q2** (0 mismatches over 4.5M pairs for every reachable prior). Needs only a new *accumulator* (`f32 Sum` + `i64 Count`); the existing border table is bit-exact. **And at binclf the statistic is IDENTICAL to Borders@b=0** — see the "BTMV ≡ Borders at binclf" finding below. | **Track B** — new accumulator only; reuse `binarize_ctr_kernel`. |
  | 4. **Counter** | Permutation-**independent** whole-set tally with a *constant* MAX-bucket denominator. `counter_calc_method=Full` is **structurally moot on the device**: `Full`'s only effect is `counter_full_eval_columns`, built solely from `eval_sets` (`boosting.rs:4093-4107`), and `device_host_eligible` requires `eval_sets.is_empty()` (`boosting.rs:4480`). So `Full ≡ SkipTest` on any device-eligible fit. | **Track C** — no permutation, cheapest math, but a *different* device code path (no prefix scan). |
  | 5. **FPP-11 combination CTR** | **Root-caused this session, and the doc comment's named suspects are only half the story.** See below. | **Track D** — highest value, now well-specified. |

- **FPP-11 root cause (NEW, evidence-backed).** The device's CTR pass C scores
  **every** CTR column at **every** level with no combination-eligibility gate
  (`crates/cb-backend/src/gpu_runtime/mod.rs:4919 for c in 0..cs.n_ctr`), and
  its `eligible_max` maxes over **all** columns
  (`mod.rs:4895`). Upstream and this repo's CPU both restrict the CTR candidate
  set **per level**:
  - Upstream `AddTreeCtrs` builds combination projections only from `seenProj`
    = `{binAndOneHotFeaturesTree} ∪ currentTree.GetUsedCtrs()`, and skips every
    `baseProj.IsEmpty()` — so at **level 0 the current tree is empty and NO
    combination candidate exists at all** (`greedy_tensor_search.cpp:491-551`,
    called per level from `GreedyTensorSearchOblivious` at `:1189-1190` with the
    running `currentSplitTree`). `CalcMaxFeatureValueCount` is likewise
    recomputed **per level** over that gated candidate list (`:1070-1088, :1201`).
  - This repo's CPU mirrors it exactly: `combination_ctr_eligible`
    (`crates/cb-train/src/tree.rs:2896-2909`) returns `false` for every
    combination when `used_projections` is empty, `select_level_ctr_aware`
    `continue`s past ineligible combinations **before scoring**
    (`tree.rs:3134-3145`), and `eligible_max_bucket_count` applies the *same*
    filter (`tree.rs:2920-2933`).
  - **Empirical confirmation on the fixture**: upstream's
    `ctr_device_combo/model.json` uses exactly two CTR descriptors — simple `[0]`
    and combination `[0,1]` — and in **all five trees** level 0 is the *simple*
    `[0] @ 5.999999` and the combination `[0,1] @ 3.999999` appears **only at
    level 1** `[VERIFIED: python3 dump of model.json oblivious_trees]`. The
    device diverges by picking `[0,1] @ 8.0` at **tree 3 level 0** — a candidate
    upstream structurally cannot have there.
  ⇒ The doc comment's **suspect 1 (`eligible_max`) is confirmed as a real
  second-order defect**; **suspect 2 (`bucket_count` semantics) is REFUTED**
  (see Track D); and the **primary** defect — a missing per-level candidate
  eligibility gate on the device — was not named at all.

- **BTMV ≡ Borders at binary classification (NEW, second pass).** Upstream's
  `CalcOnlineCTRMean` accumulates `elem.Add(float(permutedTargetClass[docId]) /
  targetBorderCount)` with `targetBorderCount = targetClassesCount - 1 = 1`
  (`online_ctr.cpp:467`, called at `:757-767`), so `Sum` is exactly the count of
  class-1 documents seen and `Count` is the number of documents seen. Upstream's
  binclf Borders path (`CalcQuantizedCtrs`, `online_ctr.cpp:298-307`) uses
  `goodCount = elem[1]`, `totalCount = elem[0] + elem[1]` — **the same two
  numbers** — and both feed the **same** `CalcCTR(cic, tot, prior, shift, norm,
  bc)`. ⇒ **In upstream, a BinarizedTargetMeanValue column and a Borders column
  are bit-identical at binclf.** `[VERIFIED: read both upstream functions]`
  Consequences: (a) the device already computes exactly what BTMV needs, in the
  *existing* kernel — Track B's "new accumulator" is only needed if the planner
  wants a structurally-honest BTMV path rather than an aliasing one; (b) it
  explains why spike Q2 found zero mismatches; (c) it gives Track B a free,
  extremely strong oracle: **a device BTMV fit and a device Borders fit on the
  same fixture must produce identical models**. The repo's CPU currently makes
  them differ only by the f32-vs-f64 quantization width and the missing
  `shift/norm` — both of which spike Q2 shows are inert for every reachable
  prior, i.e. **the repo's CPU BTMV and Borders columns are also already
  identical in practice.**

- **`device_ctr_combo_fit_test` is currently a FALSE PASS if un-ignored.**
  `[VERIFIED: cargo test -p cb-train --no-default-features --features rocm --test
  device_ctr_combo_fit_test -- --ignored --nocapture]` →
  `[device-ctr-combo-e2e] 8 CTR splits; max |Δpred| = 1.388e-17 (bar 1e-5) … ok`.
  Because the gate declines combinations, the fit runs on the **CPU** grower, and
  the test's only routing assertion (`oblivious_trees.len() == iterations`,
  `non_symmetric_trees.is_empty()`) is satisfied by the CPU oblivious grower too.
  It does **not** use the `CountingGpu` wrapper that every real device-commit
  test uses. The `1.388e-17` it prints is the *CPU* number quoted in the gate's
  own doc comment. The test's `#[ignore]` reason ("this fit runs on the CPU
  grower and the arm-routing assertion below would fail") is **factually wrong**.

- **Most important constraints.**
  1. A real ROCm GPU **is present on this machine**: `gfx1151` / AMD Radeon 860M,
     ROCm at `/home/user/rocm/opt/rocm` (not `/opt/rocm`), `/dev/kfd` present
     `[VERIFIED: /home/user/rocm/opt/rocm/bin/rocminfo | grep gfx → gfx1151]`.
     The **entire existing device suite is GREEN** here
     `[VERIFIED: bash ./run_device_tests.sh → 23/23 PASS + perf lane PASS]`.
  2. Device CTR fits require the float quantization to produce **exactly 16
     bins** (`border_count == 15`), because `ctr_covered` demands
     `col.borders.len() + 1 == n_bins` for every CTR column
     (`session.rs:150-162`) while `n_bins = max_f(borders_f + 1)`
     (`boosting.rs:4589-4592`) and `ctr_border_count_default() == 15`.
     **NOW IN SCOPE as C-1** (Q1 ruling), and much smaller than first assessed:
     the histogram-line padding that supports catboost's default 254 borders
     already ships for float features (`pad_hist_line_bins`,
     `session.rs:1010-1018`).
  3. Cat-**only** pools can never reach the device: `has_any_scorable_feature`
     needs `n_float > 0 || n_one_hot > 0` (`boosting.rs:3284-3286`), and CTR
     requires `one_hot_bins.is_empty()`. Every new fixture must carry float
     columns + a frozen `borders.npy`.
  4. `simple_ctr` / `combinations_ctr` are a **single** `ECtrType` each, not
     upstream's list of descriptions (SPEC-CTRT-19, `catboost-rs-py/src/params.rs:41-52`).
     The upstream CPU default `[Borders(0/1,0.5/1,1/1), Counter(0/1)]` is **not
     representable**, so this phase cannot close that gap.
  5. Upstream's **GPU** task type does NOT support `BinarizedTargetMeanValue` or
     `Counter` (`restrictions.h:18-48`, v1.2.10). Tracks B and C therefore make
     catboost-rs's device path a *superset* of upstream GPU. That is fine — the
     oracle is upstream **CPU** predictions, which is what every existing CTR
     fixture already uses — but it means there is no upstream GPU reference to
     cross-check against.

- **Highest-risk findings.**
  - **R-1 (Track D)**: the device has no per-level CTR candidate gate at all;
    adding one requires shipping per-column projection member sets across the
    `cb-compute` seam and threading a per-tree "chosen CTR projections" list into
    `ResidentCtrSearch`. This is a genuine seam widening, not a one-liner.
  - **R-3 (Track B)**: BTMV's f32 quantization cannot be expressed as a
    `value > border` test. A border-table implementation will pass small
    self-oracles and silently miss at boundaries.
  - **R-6**: the CPU BTMV quantizer **omits** upstream's `(ctr + shift)/norm`
    normalization (`ctr_feature.rs:327-333` computes `ctr * border_count`
    directly). It is invisible for every prior in `[0, 1]` (where `shift = 0`,
    `norm = 1`) but is a latent CPU divergence a device transcription would
    faithfully inherit.
  - **R-8**: `device_ctr_fit_test` and `device_ctr_combo_fit_test` lack
    device-commit assertions; every new e2e test in this phase **must** use the
    `CountingGpu` pattern or it can pass on CPU.

---

## Phase Requirements

### In Scope (user-confirmed: all five tracks PLUS six of the nine additional clauses)

**A. The original five tracks** (+ Track E, added by the Q4 ruling):

1. **Track 1 (as requested) — non-unit prior denominator + multi-target-border.**
   Resolves to: (a) delete the provably-dead `col.prior_denom == 1.0` conjunct;
   (b) fold multi-target-border into Track 2, since only Buckets produces it.
2. **Track 2 — `ECtrType::Buckets` on device**, including `target_border_idx ∈
   {0, 1}` at binclf.
3. **Track 3 — `ECtrType::BinarizedTargetMeanValue` on device**, mirroring the
   CPU's f32 accumulation and f32 quantization exactly.
4. **Track 4 — `ECtrType::Counter` on device**, whole-set tally, constant MAX
   denominator, no permutation dependence.
5. **Track 5 — FPP-11**: make combination (`≥2`-member) CTR projections
   device-eligible with a passing e2e oracle at ≤1e-5 on
   `crates/cb-oracle/fixtures/ctr_device_combo/`. **Measured working this
   session** (spike Q5: `2.082e-17`).
6. **Track E (Q4 ruling) — fix the CPU BTMV `(ctr + shift)/norm` gap**
   (`ctr_feature.rs:327-333`). Proven a no-op for every committed artifact.

**B. Six of the nine additional clauses** (Q1 ruling), each researched in
"Newly In-Scope Clauses (Q1)": **C-1** (`border_count != 15`), **C-2**
(multi-permutation, parity-capped), **C-4** (eval sets ⇒ Counter `Full`),
**C-5** (cat-only pools), **C-6** (bootstrap / MVS / host sampling / Lossguide
cap — *excluding* exact-leaf), **C-7** (non-symmetric / Region — recommended as
its own phase).

**C. Two provably-dead conjuncts to delete with characterization tests**:
`col.prior_denom == 1.0` (Track 1a) and `one_hot_bins.is_empty()` (C-3-dead).

### Acceptance Criteria (derived — no locked spec exists yet for this phase)

- Every device CTR fit matches upstream `catboost==1.2.10` `predictions.npy` at
  **≤1e-5** (the project-wide oracle bar; every existing `*_oracle_test.rs` uses
  it, e.g. `device_ctr_fit_test.rs:119+`).
- Every new e2e test asserts **device commitment** via the `CountingGpu`
  `Runtime` wrapper (`device_ctr_gate_test.rs:60-170` is the canonical shape),
  i.e. `grown.get() == params.iterations` — never merely `oblivious_trees.len()`.
- `ctr_types_are_device_covered` ends with **no CTR-type / projection-arity /
  target-border / prior-denominator conjunct left**, or each remaining conjunct
  is justified by a passing negative test.
- **(Q1 ruling)** For each of C-1, C-2, C-4, C-5, C-6: the clause is either
  removed with a positive `CountingGpu` commit test, or retained with a
  *documented, negatively-tested* justification. A silently-surviving clause is a
  phase failure.
- **(Q1 ruling)** For C-8, C-9 and `FloatTargetMeanValue`/`FeatureFreq`: the
  decline stays, and the doc comment cites the upstream/CPU evidence for why no
  parity surface exists.
- **(Q4 ruling)** `ctr_btmv_simple_oracle_test` stays green **unchanged** after
  Track E — that is the no-op proof — and a new `prior = 2.0` unit test pins the
  corrected bin.
- The 23 currently-green device test binaries stay green
  `[VERIFIED baseline: ./run_device_tests.sh, 2026-08-08]`.
- `device_ctr_combo_fit_test` is **un-ignored** and rewritten with a
  `CountingGpu` device-commit assertion.

### The nine additional clauses — IN/OUT verdicts (Q1 ruling: "everything")

Researched in full in **"Newly In-Scope Clauses (Q1)"** below. Verdicts:

| # | Clause | Site | Verdict |
|---|---|---|---|
| C-1 | `col.borders.len() + 1 == n_bins` (float `border_count` must be 15) | `session.rs:150-162` | **IN** — and far smaller than first assessed; the float padding infrastructure already exists |
| C-2 | `learning_folds_for_cycle == 1` (multi-permutation CTR) | `boosting.rs:4425` | **IN, but parity-capped** — the CPU's own `structure_fold_cycle` is anchored only at `pc=4, seed=0` |
| C-3 | `one_hot_bins.is_empty()` (one-hot x CTR) | `boosting.rs:4426` | **DEAD CLAUSE on device; the real blocker is a CPU feature.** SPEC-OH-26 rejects the mixed pool at `train_inner`, before any device consideration |
| C-4 | `eval_sets.is_empty()` | `boosting.rs:4480` | **IN**, but it is an *eval-set device feature*, not a CTR feature; it unlocks `counter_calc_method=Full` (Track C) |
| C-5 | `has_any_scorable_feature` (cat-only pools) | `boosting.rs:4491`, `3284-3286` | **IN** — needs `n_bins` derived from CTR columns, not float borders alone |
| C-6 | CTR x bootstrap / MVS / exact-leaf / Lossguide cap / host sampling | `session.rs:1699-1709` | **IN (partially)** — bootstrap/MVS/Lossguide are compositional; exact-leaf x CTR is genuinely contradictory |
| C-7 | CTR x non-symmetric / Region grow policies | `session.rs:1699-1700` | **IN, largest remaining device build** — `ResidentCtrSearch` exists only on the oblivious resident path |
| C-8 | Multi-descriptor `simple_ctr` (`[Borders, Counter]`) | `catboost-rs-py/src/params.rs:41-52` | **OUT — no CPU parity surface.** `BoostParams.simple_ctr` is one `ECtrType`; the CPU trainer cannot express the config at all |
| C-9 | wgpu backend | `ctr_device.rs:198-206`, `session.rs:135-137` | **OUT — permanent.** WGSL has neither f64 nor u64 (WR-02) |
| — | `FloatTargetMeanValue` / `FeatureFreq` | `boosting.rs:1372-1387` | **OUT — upstream CPU rejects them** (`restrictions.h:20-32`) |

**IN: C-1, C-2, C-4, C-5, C-6, C-7 (six). OUT with evidence: C-3 (as a device
item), C-8, C-9, and FloatTargetMeanValue/FeatureFreq.**

### Out of Scope — evidence-backed

The "everything" ruling is about ambition; it cannot create parity surfaces that
do not exist. These three must stay out, and here is the evidence for each.

- **C-8 multi-descriptor `simple_ctr` — OUT.** `BoostParams.simple_ctr: ECtrType`
  and `combinations_ctr: ECtrType` are **single** types
  (`catboost-rs/src/builder.rs:97-115`) `[VERIFIED: read]`, and the Python layer
  documents the gap and **raises** on a longer list: *"a simultaneous
  [Borders, Counter] configuration is not representable … `simple_ctr` /
  `combinations_ctr` accept exactly one CTR description each (SPEC-CTRT-19)"*
  (`catboost-rs-py/src/params.rs:41-52, 826-840`) `[VERIFIED: read]`. The CPU
  trainer has no such configuration, so there is nothing for a device arm to
  match. **Recommendation: OUT of this phase; file as the highest-value CPU CTR
  follow-on** — it is what would close the gap to catboost's actual CPU default
  `[Borders(0/1, 0.5/1, 1/1), Counter(0/1)]` (`cat_feature_options.cpp:118-138`).
- **C-9 wgpu — OUT, permanently.** The CTR statistic is
  `(good + prior)/(total + 1)` in **f64** and the border table is `Vec<f64>`;
  WGSL has **neither f64 nor u64**, which is exactly why `ctr_device.rs:198-206`
  returns a typed `CbError::OutOfRange` instead of an opaque JIT crash and
  `ctr_covered` short-circuits on `cfg!(feature = "wgpu")` (`session.rs:135-137`)
  `[VERIFIED: read]`. An f32 CTR path is **not** a safe substitute: spike Q2's
  prior=-0.5 row shows the f32 and f64 quantizations are **not** bin-identical in
  general (601-1200 mismatches), so an f32 wgpu path would break the ≤1e-5 bar
  rather than extend coverage. **Do not invent one.**
- **`FloatTargetMeanValue` / `FeatureFreq` — OUT.**
  `IsSupportedCtrType(ETaskType::CPU, …)` admits exactly
  `{Borders, Buckets, BinarizedTargetMeanValue, Counter}` (`restrictions.h:20-32`)
  `[VERIFIED]`; `validate_ctr_types` (`boosting.rs:1372-1387`) and
  `materialize_ctr_feature` (`ctr_feature.rs:311-318`) both reject them. No CPU
  oracle can exist.
- `catboost-master/` — a stale 3-file stub from a different upstream revision;
  never a parity reference.

### Open or Conflicting Requirements

- ~~The phase title says "complete device CTR coverage" but the five tracks leave
  nine clauses standing; needs a user ruling.~~ **RULED (Q1): "everything".** All
  nine were researched; six are IN, three are OUT with evidence. The remaining
  conflict is one of *size*, not of scope: eleven admitted work items do not fit
  one buildable phase. See **Planning Guidance → Phase-size assessment** for the
  recommended three-phase staging (P1 type/projection completeness, P2 CTR
  reachability, P3 CTR × remaining families), with C-3 and C-7 called out as
  **CPU features** the ruling buys rather than device-coverage items.
- Track 1's "non-unit prior denominators" premise in the task brief is
  contradicted by the code and by upstream (`ctr_helper.cpp:50`). The brief's
  statement that "the device kernel hard-codes the `+1` denominator" is true but
  **not a divergence**: upstream's `CalcCTR` also hard-codes `+1`
  (`online_ctr.h:128-131`), and the prior *denominator* enters only through the
  scalar `prior_num/prior_denom`, which the device already carries.

---

## Project Constraints

- **CLAUDE.md (worktree copy, authoritative sections)**: source/test separation
  is **mandatory** — no `#[cfg(test)] mod tests` in production files; tests in
  dedicated `*_test.rs` (mounted via `#[path]`) or `tests/*.rs`. `thiserror` in
  libraries, `anyhow` in applications; `unwrap()` prohibited in production.
  Kernels must use `generics-float`. Latest crate versions. GPU tests on `rocm`
  only. `catboost-master/` is a **stale 3-file stub from a different revision** —
  never cite it as a parity reference.
- **CubeCL manual**: read before any kernel-shape claim —
  `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md`
  `[VERIFIED: read this session]`.
  **Correction for the planner**: CLAUDE.md/AGENTS.md point at
  `.../Cubecl/cubecl_error_guideline.md`, which **does not exist**
  `[VERIFIED: ls → No such file or directory]`. The real error guidance is the
  directory `.../Cubecl/cubecl_error_solution_guide/` containing
  `mismatched types.md` and
  `calling a "normal" Rust function from inside a cube macro function fails in CubeCL.md`.
  Consult those on any CubeCL build error.
- **`cb-backend` must never depend on `cb-train`** (the T-10-04 feature-unification
  landmine, restated at `ctr_device.rs:11-14` and `mod.rs:4381-4382`). Every CPU
  reference formula the device needs is **transcribed inline** — e.g.
  `resident_cat_feature_weight`, `resident_phantom_bucket_count`,
  `calc_hash`/`fold_cat_hash`. New device semantics must follow the same rule.
- **`cb-compute` types crossing the seam must be plain host types** (no `cubecl`):
  `DeviceCtrConfig`, `DeviceCtrColumn`, `DeviceCtrAveraging`
  (`cb-compute/src/runtime.rs:1104-1190`).
- **No `unwrap`/`expect`/`panic`/raw indexing** anywhere in `cb-backend/src/**`
  (workspace lints + D-13).
- **Decision IDs to preserve**: D-04 (byte-unchanged CPU path when not covered),
  D-05 (only O(1) per-level decisions cross the seam), D-06 (CTR residency —
  values never touch the host), D-07 (ε=1e-4 device-vs-CPU self-oracle vs ≤1e-5
  upstream), D-10-01 (per-fit all-or-nothing), WR-02 (wgpu has no f64 channel),
  SPEC-OH-26 (one-hot × CTR forbidden), SPEC-CTRB-01 (`ctr_bin_border_to_value_space`
  domain `bin_border ≤ 15`, `tree.rs:3260-3290`).
- **Frozen-fixture rule**: catboost quantization is run-to-run nondeterministic,
  so every CTR fixture is generated once and **never** regenerated in CI
  (`ctr_device_combo/gen_fixtures.py` docstring "Reproducibility caveat";
  every `config.json` `"note": "FROZEN…"`).

---

## Current Project Architecture

### 1. The CPU reference: what the device must reproduce

`crates/cb-train/src/ctr/ctr_feature.rs:146-363`
`materialize_ctr_feature(cat_columns, projection, permutation, target_class,
prior_num, prior_denom, ctr_border_count, ctr_type, target_border_idx,
extra_cat_columns) -> CbResult<CtrFeatureColumn>` — the **single** per-column
materializer. Steps `[VERIFIED: read in full]`:

1. Per-object combined key via `TProjection::combined_hash` over **all** cat
   columns' `calc_cat_feature_hash` values (`:194-205`).
2. First-seen dense remap → `combined_bins`; `bucket_count = remap.len()`
   (`:207-221, :266`).
3. `extra_bins` — Counter-only `counter_calc_method=Full` eval widening
   (`:223-254`); **empty on any device-eligible fit** (see Track C).
4. `prior_scalar = prior_num / prior_denom` (`:257`).
5. Per-type dispatch (`:267-319`):
   - `Borders | Buckets` → `online_class_prefix_column(permutation,
     &combined_bins, target_class_n, SIMPLE_CLASSES_COUNT, target_border_idx,
     ctr_type, prior_scalar)` (`ctr/online.rs:374-462`).
     `nums = prefix.good as f64`, `denoms = prefix.total`, `quantize_in_f32 = false`.
   - `BinarizedTargetMeanValue` → `online_mean_prefix(permutation,
     &combined_bins, target_class_n, SIMPLE_CLASSES_COUNT, prior_scalar)`
     (`online.rs:298-360`). `nums = f64::from(prefix.sum)` (an **f32** running
     sum), `denoms = prefix.count`, **`quantize_in_f32 = true`**.
   - `Counter` → `online_counter_column(&combined_bins, &extra_bins,
     bucket_count)` (`online.rs:493-521`) → `(per_doc_bucket_total,
     max_bucket_total)`; `denoms = vec![denominator; n]` (a **constant**);
     `quantize_in_f32 = false`.
   - `FloatTargetMeanValue | FeatureFreq` → `CbError::Unsupported` (`:311-318`).
6. Quantize (`:321-349`):
   ```rust
   let bin_f = if quantize_in_f32 {
       let norm = (total as f32) + 1.0f32;
       let ctr  = (good as f32 + prior_scalar as f32) / norm;
       f64::from(ctr * ctr_border_count as f32)          // NOTE: no shift/norm
   } else {
       calc_ctr_online_bin(good, total, prior_scalar, ctr_border_count)
   };
   let clamped = bin_f.trunc().clamp(0, ctr_border_count);
   ```

Supporting primitives (`crates/cb-train/src/ctr/`) — all `[VERIFIED: read]`:

| Symbol | Location | Contract |
|---|---|---|
| `SIMPLE_CLASSES_COUNT: usize = 2` | `online.rs:52` | binclf `N[0]`/`N[1]` |
| `online_class_prefix(counts, b, ctr_type) -> (f64, i64)` | `online.rs:552-570` | `Buckets → counts[b]`; else `total − Σ_{c≤b} counts[c]`; denom = bucket total |
| `online_class_prefix_column(perm, bins, class, classes, b, ty, prior)` | `online.rs:374-462` | read-before-increment prefix; flat `bucket_count*classes` `i64` counts; **rejects `Counter`** |
| `online_mean_prefix(perm, bins, class, classes, prior)` | `online.rs:298-360` | `TCtrMeanHistory{sum: f32, count: i64}`; adds `class as f32 / (classes-1).max(1) as f32` |
| `online_counter_column(bins, extra_bins, bucket_count)` | `online.rs:493-521` | whole-set totals + `max` denominator; **no permutation parameter at all** |
| `calc_ctr_online(cic, total, prior)` | `calc_ctr.rs:76-79` | `(cic + prior) / (total + 1)` — hard `+1` |
| `calc_ctr_online_bin(cic, total, prior, bc)` | `calc_ctr.rs:89-101` | `(ctr + shift)/norm * bc` |
| `calc_normalization(prior) -> (shift, norm)` | `calc_ctr.rs:60-66` | `left=min(0,p); right=max(1,p); shift=-left; norm=right-left` |
| `ECtrType::target_border_count(k)` | `ctr/mod.rs:137-146` | Buckets→`k`; BTMV/Counter→1; else `k-1` |
| `ECtrType::is_cpu_supported()` | `ctr/mod.rs:155-160` | `{Borders,Buckets,BTMV,Counter}` |
| `ECtrType::is_online_prefix()` | `ctr/mod.rs:176-183` | `Counter`/`FeatureFreq` → `false` |

Candidate expansion (`boosting.rs:2204-2250`
`materialize_ctr_columns_for_perm`): for each projection, `for target_border_idx
in 0..ctr_type.target_border_count(2) { for prior_num in priors { … } }` — the
upstream `(ctrIdx, targetBorderIdx, priorIdx)` nesting. `prior_denom` is the
constant `CTR_PRIOR_DENOM = 1.0` (`boosting.rs:2257`).

### 2. The device seam

- `crates/cb-compute/src/runtime.rs:1104-1190`:
  - `DeviceCtrColumn { member_bins: Vec<Vec<u32>>, prior: f64, borders: Vec<f64>,
    bucket_count: usize, weight_group: u32 }` — **no `ctr_type`, no
    `target_border_idx`, no prior pair**.
  - `DeviceCtrConfig { permutation, target_class, columns, averaging:
    Option<DeviceCtrAveraging>, cat_eligible_buckets, model_size_reg }`.
  - `DeviceCtrAveraging { permutation, target_class, columns }`.
  - `DeviceTrainConfig::is_covered_regime()` (`:1344-1356`) requires
    `self.ctr.is_none()` — CTR has its own gate arm.
- `crates/cb-train/src/boosting.rs:2429-2544` `build_device_ctr_config(...)`:
  per column, resolves each projection member to its `cat_eligible_buckets`
  column, computes `prior = prior_num/prior_denom`, `(shift, norm) =
  calc_normalization(prior)`, and the border table
  `borders[k] = ((k+1)·norm/bc − shift).next_down()`; assigns
  `weight_group` by `(ctr_type, projection)` identity, shared across the
  structure and averaging halves.
- `crates/cb-backend/src/kernels/ctr_device.rs` (428 lines):
  - `combine_projection_bins(member_bins, n) -> (Vec<u32>, usize)` (`:89-117`) —
    host fold of member codes via `fold_cat_hash`/`calc_hash` (MAGIC_MULT
    `0x4906ba494954cb65`) + first-seen remap.
  - `ordered_ctr_prefix_kernel(perm, bins, class, prior: &Array<f64>,
    counts: &mut Array<u32>, good, total, value: &mut Array<f64>)` (`:134-167`) —
    **serial, unit 0 only** (`if ABSOLUTE_POS == 0`), `counts` layout
    `2*bucket + class`, `value[doc] = (g + pr)/(t + 1)` in f64. **Not** generic
    over float (concrete `f64` — a pre-existing deviation from the generics-float
    mandate).
  - `binarize_ctr_kernel<F: Float>(values, borders, out_bins)` (`:174-190`) —
    `bin = #{ borders[j] : v > borders[j] }`; generic-float, launched as `f64`
    (`:334-342`).
  - `launch_ordered_ctr_resident` (`:231+`), `binarize_ctr_column_resident`
    (`:321-343`), plus wgpu rejection stubs (`:198-206, :348-357`) and host
    read-back oracle wrappers (`:367+, :408+`).
- `crates/cb-backend/src/gpu_runtime/session.rs`:
  - `ctr_covered(config, n, n_bins)` (`:134-163`) — wgpu reject; both
    permutations present; `permutation.len()==n`; `target_class.len()==n`;
    every column non-empty with `borders.len()+1 == n_bins`; averaging mirrors.
  - `build_ctr_cindex_columns(client, perm, target_class, columns, n)`
    (`:201-234`) — per column: single-member fast path (`buckets = max(bins)+1`)
    or `combine_projection_bins`; `launch_ordered_ctr_resident`;
    `binarize_ctr_column_resident`; read back **only** the `u32` bin column.
  - The CTR arm in `begin` (`:1699-1789`): `ctr_is_covered` ANDs
    `nonsym_policy.is_none() && !region_active && ctr_covered && bootstrap_arm ==
    NoDraw && !exact_leaf && mvs_lambda.is_none() && max_leaves.is_none() &&
    !sample_from_host`; builds structure columns into the augmented cindex tail,
    averaging columns into `ctr_averaging_bins`, and constructs `CtrSearchState`.
  - `CtrSearchState { n_ctr, bucket_counts, weight_groups, group_used,
    cat_eligible_buckets, model_size_reg, float_bins }` (`:175-198`) — the
    session-owned, **fit-lifetime** state (`group_used` persists across trees).
  - The averaging leaf-value gather (`:2723-2784`): re-derives only the
    CTR-level bits of `leaf_of` from `avg_bins`, keeps float bits from the
    structure `leaf_of`, then `calc_average(Σ(w·der1), Σw, scaled_l2)`.
- `crates/cb-backend/src/gpu_runtime/mod.rs`:
  - `ResidentCtrSearch<'a>` (`:4360-4378`) — the per-tree borrowed view; its doc
    says **"DEVICE SCOPE: simple (single-feature) projections only — the cb-train
    gate declines combination-CTR fits, so no combination-eligibility logic
    exists here."**
  - `resident_cat_feature_weight(count, max_count, model_size_reg)` (`:4383-4388`).
  - `resident_phantom_bucket_count(float_leaf_of, cat_bucket)` (`:4392-4398`).
  - Pass C, the per-column CTR scorer loop (`:4890-4944`): `eligible_max =
    cs.bucket_counts.iter().max()`; `phantom_max` when `float_split_count > 0`;
    `max_bucket_count = eligible_max.max(phantom_max).max(1)`; then
    `for c in 0..cs.n_ctr { score_partition_over_binsums(window [c, c+1)); weight
    = group_used[g] ? 1.0 : resident_cat_feature_weight(bucket_counts[c],
    max_bucket_count, model_size_reg); … }`, strict `>` first-wins, and the
    float/one-hot winner keeps ties (`:4938-4944`).
  - Bookkeeping (`:4980-5005`): a CTR winner lifts `group_used[group]`; a float
    winner appends a bit to the host `float_leaf_of` replay.
  - `ctr_base = n_features - n_ctr` (`:4752-4754`); float/one-hot passes clamp to
    `[0, ctr_base)`.
- `crates/cb-train/src/boosting.rs:5284-5332` — device split → `CtrSplitSpec`
  translation. It already copies `ctr_type`, `prior_num`, `prior_denom` and
  `target_border_idx` **off the materialized column**, so **no change is needed
  here** for new CTR types. Likewise the whole-set bake
  (`boosting.rs:7275-7360`) and `cb-model` apply/`ctr_data` already support all
  four types (three CPU oracle fixtures prove it).

### 3. The gate under change

```rust
// crates/cb-train/src/boosting.rs:2355-2366  [VERIFIED]
fn ctr_types_are_device_covered(cols: &[crate::ctr::CtrFeatureColumn]) -> bool {
    !cols.is_empty()
        && cols.iter().all(|col| {
            col.projection.is_simple()
                && col.ctr_type == crate::ctr::ECtrType::Borders.as_i8()
                && col.target_border_idx == 0
                && col.prior_denom == 1.0
        })
}
```
Consumed at `boosting.rs:4423-4430` inside `device_host_eligible`:
```
&& ( (materialized_ctr_features.is_empty() && structure_fold_columns.iter().all(Vec::is_empty))
   || (learning_folds_for_cycle == 1
       && one_hot_bins.is_empty()
       && ctr_types_are_device_covered(&materialized_ctr_features)) )
```

---

## Upstream v1.2.10 Semantics (the parity ground truth)

All line numbers below are from files fetched this session from
`https://raw.githubusercontent.com/catboost/catboost/v1.2.10/…` (accessed
2026-08-08) `[WEB]`; each is also `[VERIFIED: grep/sed of the fetched file]`.

### The one quantizer — all `float`

`catboost/private/libs/algo/online_ctr.h:128-131`:
```cpp
inline ui8 CalcCTR(float countInClass, int totalCount, float prior, float shift, float norm, int borderCount) {
    float ctr = (countInClass + prior) / (totalCount + 1);
    return (ctr + shift) / norm * borderCount;
}
```
- The denominator is a **hard `+1`**, never `+PriorDenom` — matching
  `calc_ctr.rs:76-79`.
- **Everything is `float` (f32)**, and the return is an implicit `float → ui8`
  truncation. This repo's CPU computes the class/Counter types in **f64**
  (`calc_ctr_online_bin`) and only BTMV in f32. That is a **pre-existing**
  CPU-vs-upstream width divergence that has so far been invisible at the tested
  cardinalities; it becomes more exposed as Buckets/Counter reach the device with
  new fixtures. See Risk R-7.
- `CalcNormalization` (`online_ctr.cpp:102-112`): `left = Min(0.f, prior);
  right = Max(1.f, prior); shift = -left; norm = right - left`.

### Class types (Borders multiclass + Buckets): `CalcOnlineCTRClasses`

`online_ctr.cpp:144-212` `[VERIFIED]`:
```cpp
static void UpdateGoodCount(int curCount, ECtrType ctrType, int* goodCount) {   // :115-121
    if (ctrType == ECtrType::Buckets) { *goodCount = curCount; } else { *goodCount -= curCount; }
}
...
int goodCount = totalCountByDoc[…] = bv.GetTotal(elemId);
auto bordersData = bv.GetBorders(elemId);
for (int border = 0; border < targetBorderCount; ++border) {
    UpdateGoodCount(bordersData[border], ctrType, &goodCount);
    goodCountByBorderByDoc[border][…] = goodCount;
}
if (datasetIdx == 0) { ++bordersData[permutedTargetClass[docId]]; ++bv.GetTotal(elemId); }   // READ-then-INCREMENT
```
⇒ `online_class_prefix` (`online.rs:552-570`) is a faithful collapse of this for a
single border index. `[VERIFIED by inspection of both]`

### Dispatch and `targetBorderCount`

`online_ctr.cpp:730-795` `ComputeOnlineCTRs` `[VERIFIED]`:
- `Borders && targetClassesCount == SIMPLE_CLASSES_COUNT` → `CalcOnlineCTRSimple`
  (the binclf fast path).
- `BinarizedTargetMeanValue` → `CalcOnlineCTRMean(..., targetClassesCount - 1, ...)`.
- `Buckets || (Borders && targetClassesCount > 2)` → `CalcOnlineCTRClasses(...,
  GetTargetBorderCount(ctrInfo, targetClassesCount), ...)`.
- else `Y_ASSERT(Counter)` → `CalcOnlineCTRCounter`.

`ctr_helper.h:35-42` `GetTargetBorderCount(ctrInfo, k)`: `BTMV|Counter → 1`;
`Buckets → k`; else `k-1`. ⇒ at binclf, Buckets alone gets **two** target
borders. Mirrored exactly by `ECtrType::target_border_count`.

### Counter

`online_ctr.cpp:503-568` `CalcOnlineCTRCounter` `[VERIFIED]`:
`featureData[docId] = CalcCTR(ctrTotal[docId], /*totalCount=*/denominator, priorX,
shiftX, normX, borderCount)` — the denominator is the **same constant for every
document**, and `writer->GetDataBuffer(ctrIdx, /*targetBorderIdx*/ 0, prior, …)`.

`online_ctr.cpp:713-729` `[VERIFIED]`:
```cpp
counterCTRTotal.resize(leafCount);
int sampleCount = learnSampleCount;
if (catFeatureParams.CounterCalcMethod == ECounterCalc::Full) {
    uniqValuesCounts.CounterCount = leafCount;      // leafCount now includes test buckets
    sampleCount = hashArr.size();
}
CountOnlineCTRTotal(hashArr, sampleCount, &counterCTRTotal);
counterCTRDenominator = *MaxElement(counterCTRTotal.begin(), counterCTRTotal.end());
```
⇒ `online_counter_column` is faithful; `extra_bins` is the `Full` widening.

### `TOnlineCtrUniqValuesCounts` — refutes FPP-11 suspect 2

`catboost/libs/data/ctrs.h:49-70` `[VERIFIED]`:
```cpp
struct TOnlineCtrUniqValuesCounts {
    i32 Count = 0;
    i32 CounterCount = 0;   // Counter ctrs could have more values when counter_calc_method == Full
    i32 GetMaxUniqueValueCount() const { return Max(Count, CounterCount); }
    i32 GetUniqueValueCountForType(ECtrType t) const { return t == Counter ? CounterCount : Count; }
};
```
`online_ctr.cpp:696-700`:
```cpp
auto leafCount = ComputeReindexHash(topSize, rehashHashVal.Get(), hashArr.begin(), hashArr.begin() + learnSampleCount);
TOnlineCtrUniqValuesCounts uniqValuesCounts;
uniqValuesCounts.CounterCount = uniqValuesCounts.Count = leafCount;
```
with `topSize = catFeatureParams.CtrLeafCountLimit`, whose **default is
`Max<ui64>()`** (`cat_feature_options.cpp:236`) `[VERIFIED]`. ⇒ At defaults
`Count` **is** the observed distinct combined-hash count on the learn set —
exactly what `combine_projection_bins` / `materialize_ctr_feature` compute.
**Suspect 2 is refuted at default parameters.** (It could only bite if
`ctr_leaf_count_limit` were set, or `store_all_simple_ctr` interacted — neither is
exposed by this repo's `BoostParams`.)

### `GetCatFeatureWeight` and `CalcMaxFeatureValueCount` — the FPP-11 crux

`greedy_tensor_search.cpp:908-932` `[VERIFIED]`:
```cpp
if (!ctx.LearnProgress->UsedCtrSplits.contains(std::make_pair(ctrType, projection))) {
    return pow(1 + (uniqValuesCounts.GetUniqueValueCountForType(ctrType) / (double)maxFeatureValueCount),
               -ctx.Params.ObliviousTreeOptions->ModelSizeReg.Get());
}
return 1.0;
```
`ModelSizeReg` default `0.5f` (`oblivious_tree_options.cpp:28`) `[VERIFIED]`,
matching `model_size_reg_default()` (`boosting.rs:586-588`).

`greedy_tensor_search.cpp:1070-1088` `[VERIFIED]`:
```cpp
static size_t CalcMaxFeatureValueCount(const TFold& fold, TConstArrayRef<TCandidatesContext> candidatesContexts) {
    i32 maxFeatureValueCount = 1;
    for (const auto& candidatesContext : candidatesContexts)
        for (const auto& candidate : candidatesContext.CandidateList)
            if (candidate.Candidates[0].SplitEnsemble.IsSplitOfType(ESplitType::OnlineCtr))
                maxFeatureValueCount = Max(maxFeatureValueCount,
                    fold.GetCtrs(proj).GetUniqValuesCounts(proj).GetMaxUniqueValueCount());
    return …;
}
```
`candidatesContexts` is rebuilt **per level** by `SelectFeaturesForScoring(data,
currentSplitTree, fold, ctx)` (`:1189-1190`), i.e. the max is over the level's
**gated** candidate set.

`greedy_tensor_search.cpp:491-551` `AddTreeCtrs` `[VERIFIED]`:
```cpp
TProjection binAndOneHotFeaturesTree;
binAndOneHotFeaturesTree.BinFeatures  = currentTree.GetBinFeatures();
binAndOneHotFeaturesTree.OneHotFeatures = currentTree.GetOneHotFeatures();
seenProj.insert(binAndOneHotFeaturesTree);
for (const auto& ctr : currentTree.GetUsedCtrs()) seenProj.insert(ctr.Projection);
for (const auto& baseProj : seenProj) {
    if (baseProj.IsEmpty()) continue;                    // <<<<<< LEVEL-0 SKIP
    … proj = baseProj; proj.AddCatFeature(cat); … AddCtrsToCandList(proj) …
}
```
and `AddTreeCtrs` is only called at all when `currentSplitTree.Defined()`
(`:998-1006`). ⇒ **No combination CTR candidate exists at level 0 of any tree.**

`greedy_tensor_search.cpp:400-429` `AddCtrsToCandList` emits candidates in
`(ctrIdx, targetBorderIdx, priorIdx)` nesting — matching
`materialize_ctr_columns_for_perm`'s emission order.

### Prior denominator

`ctr_helper.cpp:43-52` `[VERIFIED]`:
```cpp
const auto num   = prior[0];
const auto denom = prior.size() > 1 ? prior[1] : 1.0;
CB_ENSURE(denom == 1.0, "Error: CPU could use only 1 as denom for ctrs currently");
ctrInfo.Priors.push_back(num);
```

### CPU vs GPU supported CTR types

`restrictions.h:18-48` `[VERIFIED]`: CPU `{Borders, Buckets, BinarizedTargetMeanValue,
Counter}`; GPU `{Borders, Buckets, FloatTargetMeanValue, FeatureFreq}`.

---

## Newly In-Scope Clauses (Q1 ruling: "everything")

Each clause below gets the same treatment as the five tracks: upstream semantics
→ CPU reference → device delta → config/plumbing → search interaction → gate
change → blast radius → fixtures → risks.

---

### C-1 — `col.borders.len() + 1 == n_bins` (float `border_count` must be 15) — **IN**

This was the item flagged as needing the most attention. **It needs the least of
the six**, because the hard infrastructure already exists.

**What actually assumes 16.** Tracing the chain `[VERIFIED: read each site]`:
- `ctr_border_count_default() == 15` (`boosting.rs:568-572`), consumed at exactly
  one site; CTR bins are therefore `0..=15` (16 values).
- `build_device_ctr_config` emits `borders: Vec<f64>` of length
  `ctr_border_count` = 15 (`boosting.rs:2505-2511`).
- `n_bins = max_f(borders_f.len() + 1).max(1)` — **float features only**
  (`boosting.rs:4589-4592`).
- `ctr_covered` requires `col.borders.len() + 1 == n_bins` **for every CTR
  column** (`session.rs:150-162`). With `ctr_border_count = 15` this forces
  `n_bins == 16`, i.e. every float feature must have exactly 15 borders.

**Do CTR bins and float bins have to share a width? NO — and the machinery to
prove it is already shipped.** `pad_hist_line_bins` (`session.rs:1010-1018`) maps
`n_bins` into the dispatched line families `{32, 64, 128, 256}`, and its own
comment names the target case explicitly:

> *"Any OTHER `n_bins` — e.g. the real-world `border_count=32` → 33 bins, or the
> CatBoost default 254 borders → 255 bins — is handled by PADDING the histogram
> LINE width up to the next dispatched family width (`n_bins_line`): the padding
> cells stay zero … and their phantom borders are excluded from the split argmin
> (`n_bins_used` …), so the scored candidate set — and therefore the chosen
> splits — stay bit-identical to the unpadded CPU enumeration."*
> `[VERIFIED: session.rs:1559-1572]`

So a 16-valued CTR column inside a 255-bin cindex is **already representable
losslessly** (the packed field is `n_bins_line`-wide and 15 < 255 < 256), and the
"narrower feature inside a wider line" case is **already the normal state** for
float features: `quantize_feature_major_with_one_hot` produces a per-feature
`real_folds[f] = borders[f].len() + 1` while `n_bins` is the max over features.

**The one real hazard.** The split argmin's threshold-arm upper bound is the
**global** scalar `n_bins_used`, not the per-feature `real_folds`:

```rust
// crates/cb-backend/src/gpu_runtime/mod.rs:3438-3450  [VERIFIED]
if one_hot {
    let fold = real_folds.get(cand_feature).copied().unwrap_or(0) as usize;
    if cand_border >= fold { continue; }
} else if cand_border >= n_bins_used - 1 { continue; }
```

With `n_bins = 255`, a CTR column would be enumerated over borders `0..=253`
instead of `0..=14`. Borders `15..=253` are **phantom**: no object has a CTR bin
above 15, so `bin > b` is false for everyone and the split is all-left. The
danger is not the extra candidates *per se* — it is that a phantom winner would
carry `bin_id >= 16` into `ctr_bin_border_to_value_space`, whose domain is
**hard-bounded at 15**:

> *"They hold together ONLY while `bin_border + 1` has an f32 ulp `<= 2^-20`,
> i.e. `bin_border <= 15`. At `bin_border == 16` the f32 subtraction rounds to
> exactly `17.0` and property 1 is LOST … NEITHER form is correct above 15."*
> `[VERIFIED: tree.rs:3260-3290, with a `b = 16` characterization test in
> tree_test.rs]`

**Why a phantom border can never actually win** (the load-bearing argument):
an all-left split leaves one child empty, so its score equals the un-split parent
score; every real split weakly improves it, and the argmin uses strict `>`.
⇒ A phantom candidate can tie but never displace. `[UNVERIFIED: this is an
analytic argument about the L2/Cosine score functions; it is NOT exercised by any
current fixture, because every device fixture has uniform per-feature border
counts. It must be tested, not assumed — see the spike below.]`

**Minimal correct design (recommended), in ascending cost:**
1. **Relax `ctr_covered` from `==` to `<=`**: `col.borders.len() + 1 <= n_bins`.
   One-line change; makes `border_count = 254` + CTR reachable immediately.
2. **Set `real_folds` for the CTR tail columns** to `ctr_border_count + 1` (= 16)
   instead of the current `n_bins_line` padding
   (`session.rs:1804-1817` currently pads CTR columns with `n_bins_line` and
   documents it as "inert because a CTR column is never one-hot and the scorer
   reads this array only on the one-hot arm") — a correct, cheap belt.
3. **Give the THRESHOLD arm a per-feature bound too**: replace
   `cand_border >= n_bins_used - 1` with
   `cand_border >= real_folds[cand_feature] - 1`. This is strictly more faithful
   to the CPU, which enumerates `matrix.feature_borders[feature]` **per feature**
   (`tree.rs:3078-3086`), and it makes the SPEC-CTRB-01 domain hold **structurally**
   rather than by the "phantom can never win" argument. It also removes a latent
   (currently benign) float-feature discrepancy. **Recommended.**
4. Mirror the same bound in the kernel-side eligibility (the host belt at
   `mod.rs:3438-3450` explicitly "mirrors the kernel's eligibility"), so host and
   kernel do not drift.

**Sub-tracks.** C-1 decomposes into: **C-1a** the `ctr_covered` relaxation;
**C-1b** the per-feature threshold bound (host belt + kernel); **C-1c** a
`border_count = 254` device CTR fixture + e2e oracle; **C-1d** a *negative-control*
unit test proving a phantom border never wins (construct a histogram where a real
candidate and a phantom candidate tie, and assert the real one is chosen).

**Blast radius.** `ctr_covered` (1 site); `real_folds` construction
(`session.rs:1804-1817`); the host belt + kernel eligibility in
`score_partition_over_binsums` — which is shared by **every** device oblivious
fit, so C-1b touches the hottest shared path in the backend and every existing
device oracle is its regression suite.

**Fixture.** `ctr_device_default_borders` — the `ctr_device_combo` recipe with
`"border_count": 254` and the generator's border-count assertion updated. This
is the fixture that finally proves a **default-parameter** CatBoost CTR fit runs
on the device.

**Risks.** (i) C-1b changes a shared scorer bound — regression risk across all
23 device binaries, mitigated by running the full suite; (ii) if the "phantom
never wins" argument is wrong for some score function (SolarL2 / LOOL2 / SatL2
have `adjust` factors that could behave differently at zero weight), C-1a alone
would be unsafe — which is precisely why C-1b is recommended over C-1a alone.

---

### C-2 — `learning_folds_for_cycle == 1` (multi-permutation CTR) — **IN, parity-capped**

**Upstream.** Each learning fold is a `TFold` with its own permutation and its
own approx trajectory; `ComputeOnlineCTRs` is invoked per fold
(`online_ctr.h:136-155` takes `foldLearnPermutationFeaturesSubset` and
`foldLearnTargetClass`), and the structure search consumes the **taken** fold.

**CPU reference.** `learning_fold_count(permutation_count, has_ctrs) =
max(1, permutation_count - 1)` (`fold.rs:168-174`) `[VERIFIED]`. So
`permutation_count = 4` ⇒ **3** learning folds. Per fold, `structure_fold_columns[fold]`
is a full CTR column set materialized under that fold's permutation
(`boosting.rs:4153-4180`), and `fold_approxes: Vec<Vec<f64>>` holds **one approx
per learning fold**, each advanced every iteration over that fold's own CTR-bin
leaf assignment (`boosting.rs:4884-4888`, `:6851-6900`, upstream
`UpdateLearningFold` / `train.cpp:585`). The per-iteration selection is
`taken_fold = struct_fold_cycle[iter]` (`boosting.rs:5848`).

**The parity cap — the decisive finding.** `structure_fold_cycle`
(`boosting.rs:661-690`) is instrument-derived and **anchored for exactly one
config**:

> *"Instrument-derived anchor for the production-default pc=4, seed=0 family …
> `[0,2,0,2,2]` … Other `learning_folds>1` configs are not yet anchored. …
> Unverified `learning_folds>1` config: keep the fixed `Folds[0]` (prior
> behavior) rather than ship an un-instrumented guess (parity discipline — do not
> fit)."* `[VERIFIED: read]`

⇒ **The CPU itself is upstream-parity-correct for `permutation_count > 1` only at
`pc = 4, seed = 0`.** Lifting the device clause for any other config would commit
a fit to the device whose *own CPU reference* is a documented approximation. The
device can be made to match **the CPU** everywhere, but it can only be shown to
match **upstream** at the anchored config.

**Device delta.** The session must hold, per learning fold: one materialized CTR
column set (structure), one resident approx, and one der1 stream; plus the
averaging set (already present). Concretely:
- `DeviceCtrConfig.columns` becomes `Vec<Vec<DeviceCtrColumn>>` (per fold) or a
  parallel `fold_columns` field; `build_ctr_cindex_columns` runs `folds + 1`
  times instead of 2.
- The resident cindex must carry **all folds'** CTR tails (memory = `folds ×
  n_ctr × n` u32) or re-bind the tail per iteration; the former is simpler and
  matches D-06 residency, the latter is cheaper.
- The grow seam needs the per-iteration `taken_fold` — a **new O(1) per-tree**
  value crossing the seam. `grow_tree_on_device` currently carries
  `(approx, target, sample, family)` (`runtime.rs`, per
  `device_ctr_gate_test.rs:122-127`'s wrapper signature) `[VERIFIED: read]`, so
  this is an additive seam parameter, not a redesign.
- `fold_approxes` must live device-side (one resident approx per fold), each
  updated over **that fold's** leaf assignment — this is the real work, and it is
  the same shape as the existing structure/averaging two-trajectory split
  generalized from 2 to `folds + 1`.

**Search interaction.** `group_used` (`UsedCtrSplits`) is model-lifetime and
fold-independent — unchanged. `bucket_count` is per (fold, column) because each
fold's remap is over the same data in a different order — but the *distinct*
count is order-independent, so `bucket_count` is fold-invariant. `weight_group`
likewise. **No search-path change beyond indexing by fold.**

**Gate change.** Drop `learning_folds_for_cycle == 1` from `boosting.rs:4425`;
`ctr_covered`'s `fold_count != 1` decline in the session must be relaxed in
lockstep (the two were deliberately kept in lockstep by GDC-01,
`boosting.rs:4822-4828`).

**Blast radius.** `device_ctr_gate_test::multi_permutation_ctr_declines_to_device`
**must be inverted** to a positive commit assertion; `device_fold_count_gate_test`
must be re-read for the same reason. `DeviceCtrConfig` shape change hits its
9 callers.

**Fixture.** A `pc=4, seed=0` device CTR fixture — the only config where an
upstream oracle is meaningful. The existing `multi_permutation_fold` generator
(`crates/cb-oracle/generator/gen_multi_permutation_fold.py`) `[VERIFIED: ls]` is
the starting point; it must gain float columns + `border_count` + frozen borders
to become device-reachable.

**Risks.** (i) Memory: `folds × n_ctr` extra resident columns; (ii) shipping the
clause relaxation for **unanchored** configs silently trains against a CPU
reference that is itself a documented fallback — **recommend gating the device
relaxation on `structure_fold_cycle` being anchored**, i.e. `pc == 4 && seed == 0`,
rather than on `folds == 1`. That keeps the honest-gate discipline the codebase
already uses.

---

### C-3 — `one_hot_bins.is_empty()` (one-hot × CTR) — **DEAD CLAUSE; the blocker is a CPU feature**

**This is not a device item at all**, and the device clause is provably dead.

**What SPEC-OH-26 actually is.** A pool spanning both encoding routes is
**typed-rejected in `train_inner`** — long before any device consideration:

```rust
// crates/cb-train/src/boosting.rs:3925-3946  [VERIFIED: read in full]
// SPEC-OH-26 — a pool spanning BOTH encoding routes is typed-rejected.
//
// The level search has no three-way candidate union: `has_ctr` selects
// `greedy_tensor_search_oblivious_with_ctr` (which takes no `cat_bins` and
// therefore enumerates no one-hot candidates), otherwise the plain perturbed
// arm runs (which sees no CTR columns). A mixed pool would silently take one
// branch and drop the OTHER encoding's columns entirely — exactly the
// class of bug this whole plan exists to fix. Device-side CTR co-existence is
// deferred (SPEC §9 R12), so the honest gate ships instead of a silent drop.
if !one_hot_absolute.is_empty() && !eligible_absolute.is_empty() {
    return Err(CbError::Unsupported(...));
}
```

**Was it a correctness finding or a scoping decision?** **A scoping decision** —
but one taken *because* the alternative was a **known-wrong silent drop**, not
because a mixed pool was measured to diverge. The rationale is structural ("no
three-way candidate union"), the remedy shipped was an honest error, and the
comment names the deferral explicitly. So lifting it does **not** contradict a
measured correctness result. `[VERIFIED: read]`

**Why the device clause is dead.** If a pool is all-one-hot, `eligible_absolute`
is empty ⇒ no CTR columns ⇒ the `materialized_ctr_features.is_empty()` arm of the
disjunct is taken and `one_hot_bins.is_empty()` is never evaluated. If a pool is
all-CTR, `one_hot_absolute` is empty ⇒ `one_hot_bins.is_empty()` is trivially
true. If a pool is mixed, `train_inner` **errored out at line 3937**. ⇒
`one_hot_bins.is_empty()` is **structurally always true whenever
`materialized_ctr_features` is non-empty** — a second provably-dead conjunct,
exactly like `prior_denom == 1.0`. `[VERIFIED by case analysis over
`partition_cat_columns`'s two outputs]`

**What it takes to lift it.** A **CPU** feature: a three-way candidate union in
the level search — float ∪ one-hot ∪ CTR — inside
`select_level_ctr_aware`/`greedy_tensor_search_oblivious_with_ctr`, with
upstream's enumeration order (`AddFloatFeatures` → `AddOneHotFeatures` →
`AddSimpleCtrs` → `AddTreeCtrs`, `greedy_tensor_search.cpp:992-1006`) and the
correct interaction with `phantom_mixed_bucket_count` (whose
`binAndOneHotFeaturesTree` half is *currently unrepresentable* in this codebase's
categorical-only `TProjection` — see `tree.rs:2952-2975`'s ORD-07 note). Note the
device side is comparatively easy: the device already concatenates
`float | one-hot | CTR` on one feature axis and the fused one-hot pass B already
coexists with pass C mechanically (`mod.rs:4838-4944`).

**Recommendation.** Delete the dead device conjunct as a proven no-op (with a
characterization test), and **treat "one-hot × CTR" as a CPU-side feature with
its own spec** — it is the second-largest item in the whole expanded scope after
C-7, and it is not a device coverage item. Flag to the user that the "everything"
ruling, applied literally here, buys a *CPU* feature.

---

### C-4 — `eval_sets.is_empty()` — **IN, but it is an eval-set device feature**

**Why it exists.** `device_host_eligible` requires `eval_sets.is_empty()`
(`boosting.rs:4480`) because the device grow seam returns a `DeviceGrownTree` and
nothing scores the held-out sets per iteration — the overfitting detector
(`od_type`/`od_pval`/`od_wait`) and `use_best_model` both need per-iteration eval
predictions. That is orthogonal to CTR.

**The CTR interaction (Track C's dependency).** `counter_calc_method = Full`'s
*only* effect is `counter_full_eval_columns`, assembled purely from
`eval_sets[*].cat_columns` (`boosting.rs:4093-4107`) and threaded into
`materialize_ctr_feature` as `extra_cat_columns` (`boosting.rs:2242-2244`), which
widens the Counter bucket space and the MAX denominator
(`ctr_feature.rs:223-254`, `online.rs:493-521`; upstream
`online_ctr.cpp:713-729`) `[VERIFIED]`. With eval sets empty, `Full ≡ SkipTest`.
⇒ **Lifting C-4 is what makes Track C's `Full` arm real.**

**Device delta.** Two independent pieces:
1. **Eval scoring** (the bulk): per-iteration application of the grown tree to the
   eval matrices. Cheapest correct design: keep eval scoring on the **host** (the
   CPU path already computes it) and have the device return `leaf_of` — which it
   already does (`DeviceGrownTree.leaf_of`) — so the host applies the tree to the
   eval rows exactly as the CPU arm does. No new kernel is strictly required.
2. **Counter `Full` widening on device**: `DeviceCtrColumn` gains
   `extra_member_bins: Vec<Vec<u32>>` (the eval rows' member bins). The Counter
   tally kernel counts learn ++ extra, the denominator is the max over the
   combined tally, but the **output column stays learn-indexed** — exactly
   `online_counter_column`'s contract.

**Gate/blast radius.** Drop `eval_sets.is_empty()`; every existing device test
passes empty eval sets so all are byte-unchanged (D-04). The overfitting detector
+ `use_best_model` truncation (`boosting.rs:7265-7274`) must be verified to work
off host-computed eval predictions on the device arm.

**Fixture.** `ctr_device_counter_full` — the `ctr_counter_full_eval` recipe
(which already ships `model_full.json` / `model_skiptest.json` and both
prediction sets) re-cut with float columns + frozen 15-border quantization.

**Risk.** Scope creep: C-4 is really "device eval-set support", a feature whose
main beneficiary is *not* CTR. **Recommend splitting it**: land the Counter
`Full` device semantics behind C-4 only if device eval-set support lands anyway;
otherwise keep `Full ≡ SkipTest` (correct today) and document it.

---

### C-5 — `has_any_scorable_feature` (cat-only pools) — **IN**

**Why it exists.** `has_any_scorable_feature(matrix) = matrix.n_features() > 0 ||
matrix.n_cat_features() > 0` (`boosting.rs:3284-3286`) `[VERIFIED]`, where
`n_cat_features()` counts **one-hot** columns. A pure-CTR pool has zero float and
zero one-hot columns ⇒ `false` ⇒ CPU. The session independently declines
`n_features == 0` (`session.rs:1555-1557`).

**Why it matters.** Six of the repo's nine CTR fixtures are cat-only
(`plain_ctr`, `ordered_ctr`, `tensor_ctr_e2e`, `ctr_buckets_simple`,
`ctr_btmv_simple`, `ctr_counter_simple`, `ctr_mixed_simple_vs_combo`,
`ctr_borders_multiprior`), deliberately so — *"Categorical-ONLY feature matrix …
so upstream float-quantization nondeterminism is structurally excluded"*
`[VERIFIED: their config.json descriptions]`. **None of them can ever reach the
device today**, which is why Tracks A–C each need a brand-new fixture. Lifting
C-5 would make the **existing, already-frozen** CTR oracle corpus device-testable
— a large testing win.

**Device delta.** The blocker is that `n_bins` is derived from float borders
alone (`boosting.rs:4589-4592`), so a cat-only pool yields `n_bins = 0` (`.max(1)`
⇒ 1) and `n_features = 0`, and `begin` declines before the CTR augmentation ever
runs. The fix:
- Derive `device_n_bins` as `max(float-derived n_bins, ctr_border_count + 1)`
  when CTR columns are present — a host-side change in `boosting.rs`.
- Move the session's `n_features == 0` decline to **after** the CTR tail is
  appended, i.e. test `eff_n_features == 0` instead of `n_features == 0`
  (`session.rs:1555` vs the `eff_n_features` computed at `:1725-1789`). This is a
  reordering, and it must not weaken the guard for genuinely empty pools.
- `bins_feature_major` is an empty slice for a cat-only pool; the augmented
  buffer is then just the CTR tail — already the code's shape
  (`augmented = bins_feature_major.to_vec(); augmented.extend(ctr_columns)`).
- The **phantom** `float_leaf_of` replay degenerates correctly:
  `float_split_count` stays 0, so `phantom_max = 0` — matching the CPU, whose
  `phantom_bucket_gate(chosen)` requires ≥1 Float split (`tree.rs:2977-2984`).

**Composition with C-1.** These two interact: C-5 sets `n_bins` from the CTR
width, C-1 lets the CTR width differ from the float width. Doing C-1 first makes
C-5 mechanical.

**Gate change.** Relax `has_any_scorable_feature` to also admit "≥1 CTR column",
or (cleaner) evaluate it against the **effective** device feature axis.

**Fixtures.** **None new needed** — this is the clause that unlocks the eight
existing cat-only CTR fixtures for device oracles. That is C-5's main payoff and
a strong argument for prioritizing it.

**Risk.** A cat-only pool has `n_bins = 16` ⇒ `n_bins_line = 32`; the resident
fill/scorer path is exercised at its narrowest. Existing coverage for
`n_bins_line = 32` exists (`session.rs:1804-1817` cites "a cardinality-2 column
pads to a legal `n_bins_line == 32`"), so this is low-risk.

---

### C-6 — CTR × bootstrap / MVS / exact-leaf / Lossguide cap / host sampling — **IN (partially)**

The five sub-clauses at `session.rs:1699-1709` are **not** one decision:

| Sub-clause | Verdict | Reasoning |
|---|---|---|
| `bootstrap_arm == NoDraw` (device Poisson) | **IN** | Compositional. The sample is a per-object multiplier folded into `score_der1`/`score_weight` (`mod.rs:4068-4089` / WR-01), which is orthogonal to *which* feature columns the histogram reads. CTR columns are just more columns. |
| `!config.sample_from_host` (Bernoulli/Bayesian/MVS host sampling) | **IN** | Same argument; the host computes the multiplier and the device folds it. WR-01's design (only the multiplier crosses the seam) is exactly what makes this compose. |
| `mvs_lambda.is_none()` | **IN** | MVS is a sampling arm like the others. |
| `max_leaves.is_none()` (Lossguide cap) | **IN via C-7** | Meaningless without the non-symmetric grow, which is C-7. Do not lift independently. |
| `!config.exact_leaf` | **OUT — genuinely contradictory** | See below. |

**Why exact-leaf × CTR is contradictory.** The exact (order-statistic) leaf
computes leaf values from `(approx, target)` over the tree's leaf assignment
(`compute_exact_leaf_values(&tree.leaf_of, …)`, `session.rs:2698-2701`). The CTR
arm **replaces** the returned leaf values with an averaging-permutation gather
(`session.rs:2723-2784`), and the code already states the mutual exclusion:
*"Exact-leaf × CTR cannot co-occur (the gate declines the combination), so this
and the override above are mutually exclusive."* `[VERIFIED: read]` Making them
co-exist requires an *exact* leaf computed over the **averaging** partition —
which is well-defined in principle (re-derive `leaf_of` over `avg_bins`, then run
the order statistic) but has **no CPU reference**: cb-train's CPU CTR path uses
`assign_leaf_over_ctr_columns` + `calc_average`, and `LeafMethod::Exact` is only
validated for `{LogCosh, Mae, Quantile, MultiQuantile}` (`boosting.rs:1389-1396`).
**Recommendation: keep `!exact_leaf`, and add a negative test.** If the user
insists, the CPU must gain exact-leaf-over-averaging-CTR-partition first.

**Device delta for the IN sub-clauses.** Largely *removing* conjuncts and
verifying. The one substantive check: the sampled `score_der1_h`/`score_weight_h`
must be what the CTR pass-C scorer reads. It already is — pass C calls
`score_partition_over_binsums` over the **same** `bin_sums` the sampled fill
produced (`mod.rs:4790-4944`), so the composition is structural. `[VERIFIED: read]`

**Gate change / blast radius.** `session.rs:1699-1709` ANDs; and
`device_host_eligible`'s bootstrap × grow-policy cross-product
(`boosting.rs:4457-4475`) must be re-read so Poisson's SymmetricTree restriction
still holds. `device_fpp_composition_test` (6 tests) and
`device_gate_composition_test` (3 tests) are the regression suites and will need
new positive cases.

**Fixtures.** One composition fixture per admitted pair is overkill; recommend
**one** `ctr_device_bootstrap` fixture (CTR + Bernoulli, the cheapest to
generate) plus device-vs-CPU self-oracles at ε=1e-4 (D-07) for the rest.

**Risk.** Each newly-admitted combination is a new numeric path; without a
fixture each, the ≤1e-5 claim rests on composition arguments. Be explicit about
which combinations get an upstream oracle and which get only a device-vs-CPU
self-oracle.

---

### C-7 — CTR × non-symmetric (Depthwise / Lossguide) and Region — **IN, the largest remaining build**

**Why it exists.** `ctr_is_covered` starts with `nonsym_policy.is_none() &&
!region_active` (`session.rs:1699-1700`) `[VERIFIED]`. `ResidentCtrSearch` and
the pass-C scorer live **only** inside `grow_oblivious_tree_resident`
(`mod.rs:4690-5010`); the non-symmetric grower (`kernels/nonsym_grow.rs`) and the
Region grower score via a different per-node `pointwise_hist2` path that "keeps
its own dispatch" (`session.rs:1571-1572`) and holds host bin copies
(`session.rs:1585-1595`).

**What a non-symmetric CTR search requires.** Everything pass C does, re-done in
a leaf-wise idiom:
1. The CTR tail columns must reach the non-sym/Region grow state (they currently
   are appended to `eff_bins` before packing, so the *cindex* already carries them
   — the gap is the **search**, not the data).
2. Per-node (not per-level) candidate scoring with the **per-column** cat-feature
   weight applied host-side to each column's O(1) best — pass C's structure, but
   invoked per node.
3. The per-level `AddTreeCtrs` eligibility gate (Track D's D-1) restated for a
   leaf-wise tree: upstream's `seenProj` is built from `currentTree.GetUsedCtrs()`,
   which for a non-symmetric tree is the set of CTR splits on the **path** to the
   node being split, not the whole tree. **This is a genuine semantic question
   with no in-repo answer** — upstream's non-symmetric CTR search lives in
   `GreedyTensorSearchDepthwise`/`Lossguide` variants that were **not fetched or
   read this session**. `[UNVERIFIED: the exact upstream non-symmetric CTR
   candidate-set semantics. This must be read from
   `greedy_tensor_search.cpp`'s depthwise/lossguide entry points before design.]`
4. The averaging-permutation leaf gather (`session.rs:2723-2784`) is written
   against the **oblivious** `leaf_of` bit-decomposition (`(structure_leaf >> l) &
   1`); a non-symmetric tree has a node graph, so the gather must be rewritten
   against `device_leaf_of_nonsym` / the Region path walk.
5. The CPU reference must exist too: `select_level_ctr_aware` is oblivious-only;
   `grow_nonsym_tree` and the Region grower take no `ctr_features` argument
   `[VERIFIED: their signatures carry no CTR parameter]`. **So the CPU cannot
   train CTR × Depthwise/Lossguide/Region either.** Like C-3, the blocker is
   CPU-side first.

**Recommendation.** C-7 is a **multi-wave feature in its own right**, and — as
with C-3 — the CPU is the blocker, not the device. It should be its **own phase**.
Admitting it into this phase would roughly double the phase and make the finish
line unclear.

---

### Track E — fix the CPU BTMV `(ctr + shift)/norm` gap (Q4 ruling)

**Upstream formula and site.** `CalcOnlineCTRMean` calls
`CalcCTR(sum[…], count[…], priorX, shiftX, normX, ctrBorderCount)`
(`online_ctr.cpp:483-489`), with `shift`/`norm` from
`CalcNormalization(priors, &shift, &norm)` at `:449-451`, and
`CalcCTR` is `(ctr + shift)/norm * borderCount` (`online_ctr.h:128-131`).
`[VERIFIED: read both]`

**The exact CPU site to change.** `crates/cb-train/src/ctr/ctr_feature.rs:327-333`:
```rust
let bin_f = if quantize_in_f32 {
    let norm = (total as f32) + 1.0f32;                        // <-- the CTR denominator,
    let ctr  = (good as f32 + prior_scalar as f32) / norm;     //     NOT CalcNormalization's norm
    f64::from(ctr * ctr_border_count as f32)                   // <-- missing (ctr + shift)/norm
} else { ... };
```
Note the local `norm` here is the **`total + 1` denominator**, shadowing the name
of `calc_normalization`'s `norm` — a readability trap the fix should also clear.
The corrected form is `f64::from(((ctr + shift_f32) / norm_f32) * bc_f32)` with
`(shift, norm) = calc_normalization(prior_scalar)` cast to f32, mirroring
upstream's all-float chain.

**Which fixtures/tests change value: NONE. Verified hard, as requested.**
- Spike Q2: for `prior ∈ {0.0, 0.5, 1.0}` the corrected and uncorrected forms give
  **0 mismatches over 4,504,501 `(good, total)` pairs each**. The reason is
  structural: `calc_normalization(p) = (-min(0,p), max(1,p) - min(0,p))`, so for
  every `p ∈ [0, 1]` it returns exactly `(0.0, 1.0)` and `(ctr + 0)/1` is the
  identity `[VERIFIED: calc_ctr.rs:60-66 + the spike]`.
- Every prior used anywhere in the repository is in `[0, 1]`:
  `{Prior=0, Prior=0.25, Prior=0.5, Prior=1}` across all fixture `config.json`s and
  `{vec![0.25], vec![0.5]}` across all Rust tests
  `[VERIFIED: grep -rho "Prior=[-0-9./]*" crates/cb-oracle/fixtures/*/config.json | sort -u]`.
- `ECtrType::default_priors()` returns `{0, 0.5, 1}` / `{0}` — all in `[0, 1]`
  (`ctr/mod.rs:191-206`) `[VERIFIED]`.

⇒ **No fixture re-baseline task is needed.** The fix is a pure correctness
extension for out-of-`[0,1]` priors, which a user can supply
(`parse_ctr_description` accepts any `Prior=<num>`) but no committed artifact
uses.

**Ordering.** Land Track E **before** the device BTMV work (Track B), so
"device == CPU" is provable against the *corrected* CPU at every point. Landing
it after would mean the device either mirrors a known-wrong CPU and then changes,
or diverges from CPU for a window.

**Tests.** (a) A characterization test asserting `calc_normalization(p) == (0.0,
1.0)` for `p ∈ {0, 0.25, 0.5, 1}`, making the no-op provable in-repo; (b) a unit
test at `prior = 2.0` asserting the corrected bin (the uncorrected form gives
`bin = 30` where the correct answer is `15` — from the spike's first mismatch,
`good=0, total=0`); (c) the existing `ctr_btmv_simple_oracle_test` must stay
green **unchanged** — that is the no-op proof.

**Bonus.** Because BTMV ≡ Borders at binclf upstream (see Research Summary), a
second corrected-CPU test is available for free: a CPU BTMV fit and a CPU Borders
fit on the same fixture with the same prior must produce **identical** models.

---

## Standard Stack

| Component | Version | Existing/Proposed | Purpose in this phase | Notes |
|---|---|---|---|---|
| `cubecl` | `0.10.0` (workspace pin, `Cargo.toml:38`) `[VERIFIED]` | Existing | Kernel authoring/launch | Backend selection via `cb-backend` features only: `default=["cpu"]`, `cpu=["cubecl/cpu"]`, `wgpu=["cubecl/wgpu"]`, `cuda=["cubecl/cuda"]`, `rocm=["cubecl/hip"]` (`cb-backend/Cargo.toml:38-42`) `[VERIFIED]`. Do **not** add `cubecl-hip-sys/hip_xxx`. |
| `cubecl-hip` | `0.10.0` (resolved) `[VERIFIED: build log]` | Existing | The rocm runtime actually exercised here | gfx1151 present |
| `bytemuck` | workspace | Existing | `cast_slice` on CTR bin read-back (`session.rs:230`) | |
| `rayon` | workspace | Existing | host prep only | |
| `cb-compute::Runtime` | in-repo | Existing | `begin_device_training` / `grow_tree_on_device` seam | The `DeviceCtrConfig` shape changes here |
| `cb_core::sum_f64` | in-repo | Existing | the sanctioned ordered reduction | Used by the averaging leaf gather |
| `catboost` (Python) | `1.2.10` | Existing (dev/oracle only) | Fixture generation | `numpy`, `Pool.quantize()`, `save_quantization_borders` |

**No new external crate is needed for any admitted track** — the original five,
Track E, or any of C-1/C-2/C-4/C-5/C-6/C-7. All work is in-repo kernel + seam +
gate + fixture work.

---

## Dependency Analysis

- **Direct**: `cb-train` → `cb-backend` (feature-gated `Runtime` impl) →
  `cb-compute` (plain host seam types) → `cubecl`.
- **Direction constraint**: `cb-backend` must not depend on `cb-train`; new
  formulas are transcribed inline. `cb-compute` must stay `cubecl`-free.
- **Runtime/system**: a rocm (or cuda) device with `Atomic<u64>` add for the
  resident partition histogram; `/dev/kfd` + ROCm at
  `/home/user/rocm/opt/rocm` on this machine. `wgpu` is permanently excluded
  from CTR (WR-02, no f64).
- **Build**: full `--no-default-features --features rocm` rebuild of the
  cb-train test target takes **≈52 s** on this machine
  `[VERIFIED: build log "Finished `test` profile … in 52.18s"]`. Disk: 631 G
  free, `target/` 6.1 G `[VERIFIED: df -h /home; du -sh target]` — the known
  `target/`-exhaustion trap is not active right now, but re-check `df -h` if a
  background job dies.
- **No dependency additions or removals anticipated.**

---

## Recommended Architecture and Implementation Pattern

### Cross-track seam change (do this once, before Tracks A–D)

Extend `cb_compute::DeviceCtrColumn` (`runtime.rs:1104-1128`) with the fields the
device needs to distinguish types — all plain host scalars:

```
ctr_type:          i8      // ECtrType discriminant (0 Borders, 1 Buckets, 2 BTMV, 4 Counter)
target_border_idx: u32     // the Buckets per-class numerator selector; 0 elsewhere
projection_members: Vec<u32>   // absolute (or eligible-position) cat-feature ids, SORTED — Track D
```
and populate them in `build_device_ctr_config` (`boosting.rs:2429-2544`) from
`CtrFeatureColumn.{ctr_type, target_border_idx, projection.cat_features()}`.

Do **not** split `prior` into a num/denom pair: `prior_num/prior_denom` is
already the correct scalar for both the value and the border table, and
`prior_denom` is provably always `1.0` (see Track 1a).

Downstream consumers to update: `ctr_covered` (`session.rs:134-163` — add
per-type shape checks), `build_ctr_cindex_columns` (`session.rs:201-234` — per-type
dispatch), `CtrSearchState` (`session.rs:175-198`), `ResidentCtrSearch`
(`mod.rs:4360-4378`), the two `#[cfg(test)]` construction sites
(`crates/cb-train/src/device_ctr_combo_config_test.rs`,
`crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs`) and
`crates/cb-compute/src/runtime_ctr_averaging_test.rs`
`[CODEGRAPH: DeviceCtrColumn — 8 callers; DeviceCtrConfig — 9 callers]`.

### Track 1a — delete the dead `prior_denom` conjunct (no kernel work)

1. Remove `&& col.prior_denom == 1.0` from `ctr_types_are_device_covered`.
2. Add a characterization test asserting `CTR_PRIOR_DENOM == 1.0` and that
   `materialize_ctr_columns_for_perm` is the only production materialization
   site, so the removal is a provable no-op.
3. Record in the doc comment: upstream `ctr_helper.cpp:50` forbids `denom != 1`
   on the CPU task type, so there is no parity surface to gain.

### Track A (= requested Tracks 1b + 2) — Buckets + multi-target-border

**Kernel delta (smallest of the four).** `ordered_ctr_prefix_kernel`
(`ctr_device.rs:134-167`) already maintains the full 2-class `[N0, N1]` bucket
history and reads it before incrementing. Only the *numerator selection* changes:

```
total = n0 + n1                              (unchanged)
good  = match (ctr_type, target_border_idx):
          (Borders, 0) => n1                 (today's hard-coded behaviour)
          (Buckets, 0) => n0
          (Buckets, 1) => n1
```
Generalized form (matching `online_class_prefix`, `online.rs:552-570`):
`Buckets → counts[b]`; else `total − Σ_{c≤b} counts[c]`.

**Recommended kernel shape** (CubeCL manual consulted:
`Cubecl_conditionals.md` — use `if`/`else` statements, never `if`-expressions;
`Cubecl_loop_control.md` — `while` with an explicit counter; `comptime_macro.md`
/ `Cubecl_comptime_specialization.md` — hoist a *runtime-constant* branch out of
the inner loop):
- **Absorb into the existing kernel**, do **not** fork it. Add one scalar
  parameter, e.g. `mode: &Array<u32>` (length 1), and select `good` with a plain
  `if/else` inside the serial `while` body. The kernel is already serial on unit
  0, so a per-iteration branch costs nothing measurable and divergence is not a
  concern (a single unit).
- Keep `prior: &Array<f64>` and `value: &mut Array<f64>` unchanged — Buckets uses
  the **same** `calc_ctr_online` + border-table quantization as Borders, so the
  existing `binarize_ctr_kernel<F: Float>` border trick
  (`borders[k] = ((k+1)·norm/bc − shift).next_down()`) remains exactly correct.
- **No new buffer, no f64 change, no wgpu implication beyond the existing WR-02
  rejection.**

**Config/plumbing**: `ctr_type` + `target_border_idx` on `DeviceCtrColumn`;
`build_ctr_cindex_columns` passes them through; `ctr_covered` accepts
`{Borders, Buckets}`.

**Search path**: Buckets at binclf emits **two** columns per (projection, prior)
— `b=0` and `b=1` — that share one `weight_group` (`build_device_ctr_config`
already keys groups on `(ctr_type, projection)`, `boosting.rs:2509-2517`), which
matches upstream's `UsedCtrSplits` key `(ctrType, projection)`
(`greedy_tensor_search.cpp:920`). `bucket_count` is the projection cardinality
and is identical for both borders — correct, since upstream's
`GetUniqueValueCountForType` is per (ctr type, projection), not per border.

**Fixture**: `ctr_device_buckets` — a **new** device-reachable fixture. Recipe:
copy `ctr_device_combo/gen_fixtures.py` verbatim and change
`simple_ctr → ["Buckets:Prior=0.5"]`, `combinations_ctr → []`,
`max_ctr_complexity → 1`; keep `border_count: 15`, `one_hot_max_size: 1`,
2 float + 1–2 cat columns, `boost_from_average: False`, `bootstrap_type: "No"`,
`random_strength: 0`, `score_function: "L2"`, `thread_count: 1`. Add
`ctr_buckets_simple`'s anti-false-pass guard: assert **both**
`ctr_target_border_idx` values 0 and 1 appear in the committed `model.json`
(`ctr_buckets_simple/config.json` records `"observed_target_border_idxs": [0, 1]`).
Freeze `borders.npy` via `Pool.quantize(border_count=15)` +
`save_quantization_borders`, and assert the quantized-pool fit is **bit-identical**
to the raw-pool fit (the combo generator's guard,
`ctr_device_combo/gen_fixtures.py` `main()`).

### Track B (= requested Track 3) — BinarizedTargetMeanValue

**This is the only track that needs a genuinely new device data path.**

**Accumulator**: per bucket `(sum: f32, count: u32)` instead of `(N0, N1): u32`.
Per document, read `(s, c)` **before** adding `class as f32 / (classes-1).max(1)`
— for binclf `classes = 2`, so the added value is exactly `class as f32 ∈ {0.0, 1.0}`
(`online.rs:298-360`, mirroring `online_ctr.cpp:762`'s
`targetClassesCount - 1`).

**The f32 accumulation width is load-bearing.** `TCtrMeanHistory::Sum` is `float`
(`online_ctr.h:373`), and this repo pins that with a dedicated test
(`online_test::btmv_sum_is_accumulated_in_f32_not_f64`, cited at
`online.rs:294`). A device `f64` sum would diverge.

**Quantization — the trap (R-3).** The CPU/upstream bin is
```
ctr   = (sum_f32 + prior_f32) / (count_f32 + 1.0f32)      // all f32
bin_f = f64::from(ctr * ctr_border_count as f32)          // the MULTIPLY is f32
bin   = trunc(bin_f) clamped to [0, bc]
```
`[VERIFIED: ctr_feature.rs:327-335]`.

**SECOND PASS — this concern is REFUTED; the recommendation below is REVISED.**
The worry was that the f32 rounding of `ctr * bc` could cross an integer boundary
a `value > border` comparison never crosses. **Spike Q2 measured it: 0 mismatches
over 4,504,501 `(good, total)` pairs at each of `prior` in `{0, 0.5, 1}`** — the
f64 border table, an f32 border table, upstream `CalcCTR` and the repo BTMV
quantizer **all agree bit-for-bit** whenever `shift = 0` and `norm = 1`, which is
every prior in `[0, 1]`, which is every prior reachable in this repository.
`[VERIFIED: python3 q2_spike_fast.py]`

**REVISED recommendation: do NOT add a dedicated BTMV quantize kernel.** Reuse
`binarize_ctr_kernel` + the existing `build_device_ctr_config` border table
unchanged. Track B therefore reduces to **the accumulator only**.

**And it may reduce further still.** At binary classification, upstream's BTMV
statistic is *identical* to Borders@b=0: `CalcOnlineCTRMean` adds
`float(targetClass) / targetBorderCount` with `targetBorderCount =
targetClassesCount - 1 = 1` (`online_ctr.cpp:467`, `:762`), so `Sum` = count of
class-1 docs and `Count` = docs seen — exactly the `good`/`total` the existing
`ordered_ctr_prefix_kernel` already produces, and both feed the same `CalcCTR`
`[VERIFIED: read online_ctr.cpp:437-501 and :271-330]`. Two implementation
options; the planner should choose deliberately:

  * **(a) Alias** — route `ctr_type == BTMV` through the *existing* Borders path
    at binclf. Zero new kernel code. Correct today, but would become silently
    wrong the day multiclass CTR lands (`SIMPLE_CLASSES_COUNT` is hard-coded 2 at
    `online.rs:52`, so that day is not near).
  * **(b) Honest accumulator** — a real `(f32 Sum, u32 Count)` bucket history in
    the kernel, selected by `ctr_type`. Slightly more code; survives multiclass;
    gives a *structural* device-vs-CPU oracle rather than an aliasing one.
    **Recommended**, with a test asserting (a) and (b) agree at binclf.

**R-6 is now Track E — FIX the CPU, do not mirror it** (Q4 ruling). The corrected
CPU form is `f64::from(((ctr + shift_f32) / norm_f32) * bc_f32)` with
`(shift, norm) = calc_normalization(prior_scalar)`; upstream does exactly this
(`CalcOnlineCTRMean` -> `CalcCTR(sum, count, priorX, shiftX, normX, bc)`,
`online_ctr.cpp:483-489`). Spike Q2 proves the change is a **no-op for every
prior in `[0, 1]`**, and every prior in the repository is in `[0, 1]`, so **no
fixture re-baseline is required**. See the **Track E** subsection under "Newly
In-Scope Clauses (Q1)" for the full analysis and test plan.

**Config/plumbing**: `ctr_type == BTMV` selects the mean accumulator in
`build_ctr_cindex_columns`; the border table stays live and unchanged (it is
bit-exact per spike Q2), so the `borders.len()+1` shape check needs **no**
per-type special case.

**Fixture**: `ctr_device_btmv`, same recipe with
`simple_ctr → ["BinarizedTargetMeanValue:Prior=0.5"]`. Note
`ECtrType::target_border_count(BTMV) == 1`, so exactly one column per prior.

### Track C (= requested Track 4) — Counter

**No permutation, no prefix, no read-before-increment.** For each column:
`totals[b] = #{obj : bin[obj] == b}` over the learn set; `denominator =
max_b totals[b]`; `value[obj] = (totals[bin[obj]] + prior) / (denominator + 1)`
(`online.rs:493-521`, `online_ctr.cpp:503-568`).

- **`counter_calc_method` is moot on the device**: `Full`'s only effect is
  `extra_bins`, sourced from `counter_full_eval_columns` which is built purely
  from `eval_sets` (`boosting.rs:4093-4107`), and `device_host_eligible` requires
  `eval_sets.is_empty()` (`boosting.rs:4480`). ⇒ `Full ≡ SkipTest` on device.
  **Do not implement eval widening.** Add a *negative* test pinning that
  `counter_calc_method = Full` + a non-empty eval set still declines to CPU
  (it does today, via the `eval_sets.is_empty()` clause).
- **Recommended shape**: the whole Counter column is a trivially parallel
  histogram + max + elementwise map. Two acceptable designs:
  - **(preferred) reuse the existing serial kernel shape** with a `ctr_type ==
    Counter` mode that runs a two-pass serial scan on unit 0 (tally, then map).
    It preserves D-06 residency, adds no new kernel, and the serial cost is the
    same order as the existing prefix. Then quantize with the **existing**
    `binarize_ctr_kernel` + border table — which is exactly correct here, because
    Counter uses the same f64 `calc_ctr_online_bin` path as Borders/Buckets
    (`ctr_feature.rs:296-309, 334`).
  - (alternative) a parallel `Atomic<u32>` tally + a max reduce + an elementwise
    map — more kernels, more launch overhead, and it introduces atomics where
    none are needed. Only worth it if profiling shows the serial tally matters.
- **Search path**: Counter's `bucket_count` is `GetUniqueValueCountForType(Counter)
  = CounterCount`, which equals `Count` under SkipTest
  (`online_ctr.cpp:699-724`, `ctrs.h:63-68`), so no new weight input is needed.
  `weight_group` keys on `(Counter, projection)` as usual.
- **Fixture**: `ctr_device_counter`, `simple_ctr → ["Counter:Prior=0.5"]`.
  Note the default Counter prior upstream is `0/1` (`ctr/mod.rs:198-203`); pin
  the prior explicitly in the fixture config and in the Rust `BoostParams`.

### Track D (= requested Track 5) — FPP-11 combination CTR

**Two defects, in order of magnitude.**

**D-1 (primary, previously unnamed): the device has no per-level combination
eligibility gate.** Add it, mirroring `select_level_ctr_aware`
(`tree.rs:3134-3145`) and `combination_ctr_eligible` (`tree.rs:2896-2909`):

- Carry `projection_members: Vec<u32>` (sorted) per column across the seam.
- In `grow_oblivious_tree_resident`, maintain a per-**tree** list
  `chosen_ctr_projections: Vec<&[u32]>` (reset each tree, appended when a CTR
  split wins — the existing `group_used` bookkeeping site, `mod.rs:4980-4990`, is
  the right hook, but note `group_used` is *fit-lifetime* whereas this list must
  be *tree-lifetime*: they are different scopes and must not be conflated).
- In pass C, `continue` past any column whose `projection_members.len() >= 2`
  and for which no already-chosen projection `q` in this tree satisfies
  `q.len() + 1 == members.len() && q ⊆ members`. Skip **before** scoring, so the
  strict-`>` first-wins enumeration order over the surviving columns is identical
  to the CPU's.

**D-2 (the doc comment's suspect 1, confirmed): `eligible_max` must use the same
filter.** `mod.rs:4895` currently maxes `cs.bucket_counts` over **all** columns;
it must max only over **eligible** columns, exactly as
`eligible_max_bucket_count` (`tree.rs:2920-2933`) does. Including an ineligible
combination's (larger) bucket count inflates `max_bucket_count`, which *raises*
`cat_feature_weight` for every unused simple candidate
(`(1 + count/maxCount)^(-0.5)` is increasing in `maxCount`) — a second,
independent way to flip the greedy winner.

**Suspect 2 (`bucket_count` semantics) is REFUTED** at default parameters:
`TOnlineCtrUniqValuesCounts::Count = ComputeReindexHash(topSize=Max<ui64>(), …,
learn range)` = the observed distinct combined-hash count
(`online_ctr.cpp:696-700`, `cat_feature_options.cpp:236`) — the same quantity
`combine_projection_bins` and `materialize_ctr_feature` produce.
**Evidence that would refute this refutation**: a fixture with
`ctr_leaf_count_limit` set, or `store_all_simple_ctr=True` — neither is
expressible through this repo's `BoostParams`, so the risk is closed.

**Empirical evidence for D-1 (already collected):**
`ctr_device_combo/model.json` has exactly two CTR descriptors — simple `[0]` and
combination `[0,1]` — and every one of the five trees has the **simple** CTR at
level 0; the combination appears only at level 1 (trees 2–4), which is precisely
the `AddTreeCtrs` `baseProj.IsEmpty()` level-0 skip
`[VERIFIED: python3 json dump of oblivious_trees; greedy_tensor_search.cpp:491-551]`.

**⚠ THIS EXPERIMENT WAS RUN — see the SECOND PASS block at the top.** Measured:
control (no gate) **FAILS at 2.746e-2**; D-1 alone **PASSES at 2.082e-17**;
D-1 + D-2 **PASSES at 2.082e-17**; the full 23-binary device suite stayed green.
`[VERIFIED: scratch edit on gfx1151, then reverted]` D-2 was **not** discriminated
by this fixture but must still ship for CPU parity (R-20). The original
instructions are retained below for reproduction.

**Confirmation experiment (RUN; reproduce with):**
temporarily restore the combination-arity conjunct removal *plus* D-1 + D-2, then
run
`cargo test -p cb-train --no-default-features --features rocm --test
device_ctr_combo_fit_test -- --ignored --nocapture`
with a `CountingGpu` wrapper added. If the max |Δpred| collapses from `3.3e-2` to
`≤1e-5` **and** `grown.get() == 5`, D-1+D-2 is the whole story. If it improves but
does not reach the bar, the next suspects (in order) are: the per-level phantom
`float_leaf_of` replay under combination columns; the single-member fast path in
`build_ctr_cindex_columns` (`session.rs:213-219`) vs `combine_projection_bins` for
1-member projections; and the f32-vs-f64 `CalcCTR` width (R-7).

**Do this last within P1** — it is the only track that needs the seam to carry
projection structure, and it composes with Tracks A–C: a combination
Buckets/BTMV/Counter column must be gated by D-1 too, which is exactly why D-1
cannot land before the types it has to gate.

### Gate change and blast radius

Final form (all four conjuncts gone):
```rust
fn ctr_types_are_device_covered(cols: &[crate::ctr::CtrFeatureColumn]) -> bool {
    !cols.is_empty()
        && cols.iter().all(|col| {
            crate::ctr::ECtrType::from_i8(col.ctr_type)
                .is_some_and(|t| t.is_cpu_supported())     // {Borders,Buckets,BTMV,Counter}
        })
}
```
`from_i8`/`is_cpu_supported` already exist (`ctr/mod.rs:96-108, 155-160`) —
**reuse them; do not hand-roll a second type list.** Blast radius:
`ctr_types_are_device_covered` has exactly **1 caller** and, per CodeGraph,
**no covering test** `[CODEGRAPH: ctr_types_are_device_covered — 1 caller in
crates/cb-train/src/boosting.rs; ⚠️ no covering tests found]`. That gap must be
closed in this phase.

---

## Project Impact Scope

### Must Change

| Item | Classification | Reason | Downstream effect |
|---|---|---|---|
| `crates/cb-train/src/boosting.rs:2298-2366` `ctr_types_are_device_covered` + doc comment | must change | The gate itself; the FPP-11 doc block is now partly wrong (suspect 2 refuted, primary cause unnamed) | Every CTR fit's device/CPU routing |
| `crates/cb-compute/src/runtime.rs:1104-1128` `DeviceCtrColumn` | must change | Needs `ctr_type`, `target_border_idx`, `projection_members` | 8 callers `[CODEGRAPH]`; `PartialEq`/`Default` derives; 3 test files construct it |
| `crates/cb-train/src/boosting.rs:2429-2544` `build_device_ctr_config` | must change | Populate the new fields | 3 callers; `device_ctr_combo_config_test.rs` |
| `crates/cb-backend/src/kernels/ctr_device.rs` | must change | Buckets numerator mode; BTMV accumulator + f32 quantize kernel; Counter tally path | `ctr_device_test.rs` (6 tests) must grow per-type oracles |
| `crates/cb-backend/src/gpu_runtime/session.rs:134-163` `ctr_covered` | must change | Per-type shape validation (BTMV border table inert) | `session_depth_gt1_test.rs` CTR gate tests |
| `crates/cb-backend/src/gpu_runtime/session.rs:201-234` `build_ctr_cindex_columns` | must change | Per-type dispatch | Called twice per fit (structure + averaging) |
| `crates/cb-backend/src/gpu_runtime/session.rs:175-198` `CtrSearchState` | must change | Carry `projection_members` (Track D) | Constructed in `begin` |
| `crates/cb-backend/src/gpu_runtime/mod.rs:4360-4378` `ResidentCtrSearch` + doc | must change | Add eligibility inputs; the "simple projections only" doc claim becomes false | Per-tree borrowed view |
| `crates/cb-backend/src/gpu_runtime/mod.rs:4890-4944` pass C + `:4980-5005` bookkeeping | must change | D-1 candidate gate, D-2 `eligible_max` filter, per-tree chosen-projection list | The core FPP-11 fix |
| `crates/cb-train/tests/device_ctr_combo_fit_test.rs` | must change | Un-ignore; add `CountingGpu` device-commit assertion; fix the wrong `#[ignore]` rationale | Currently a false pass |
| `crates/cb-oracle/fixtures/ctr_device_{buckets,btmv,counter}/` | must change (new) | No device-reachable fixture exists for these three types | Each needs `gen_fixtures.py`, `config.json`, `X.npy`, `X_cat.npy`, `y.npy`, `borders.npy`, `model.json`, `predictions.npy` |
| `run_device_tests.sh` | must change | Register the new test binaries | The repo's device DoD runner |

### May Change

- `crates/cb-train/src/device_ctr_combo_config_test.rs` — extend for the new
  `DeviceCtrColumn` fields and per-type coverage.
- `crates/cb-compute/src/runtime_ctr_averaging_test.rs`,
  `crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs` — construction
  sites of `DeviceCtrConfig`/`DeviceCtrColumn` `[CODEGRAPH]`.
- `crates/cb-train/tests/device_ctr_fit_test.rs` — add the `CountingGpu`
  wrapper (it currently only asserts `oblivious_trees.len()`, R-8).
- `crates/cb-train/src/ctr/ctr_feature.rs:327-333` — **only** if the planner
  decides to also fix the missing BTMV `(ctr+shift)/norm` normalization (R-6).
  That is a CPU behaviour change and would invalidate frozen fixtures if any
  used a prior outside `[0, 1]` — none do today.

### Verification Only

- The 23 device test binaries in `run_device_tests.sh` — green baseline captured
  `[VERIFIED 2026-08-08]`; all must stay green.
- `crates/cb-backend/src/kernels/ctr_device_test.rs` (6 tests, all pass
  `[VERIFIED: cargo test -p cb-backend --no-default-features --features rocm
  --lib kernels::ctr_device_test]`).
- CPU CTR oracles that pin the parity target the device must match:
  `ctr_buckets_simple_oracle_test.rs`, `ctr_btmv_simple_oracle_test.rs`,
  `ctr_counter_simple_oracle_test.rs`, `ctr_counter_full_eval_oracle_test.rs`,
  `ctr_borders_multiprior_oracle_test.rs`, `ctr_mixed_simple_vs_combo_oracle_test.rs`,
  `tensor_ctr_e2e_oracle_test.rs`, `plain_ctr_oracle_test.rs`,
  `ordered_ctr_oracle_test.rs`, `s_order_ctr_bins_oracle_test.rs`,
  `ctr_feature_materialize_test.rs`, `ctr_split_scoring_test.rs`,
  `ctr_border_space_test.rs`, `ctr_border_upstream_anchor_test.rs`,
  `ctr_btmv_bake_upstream_table_test.rs`, `ctr_structure_fold_shuffle_test.rs`.
  These run under the **default (cpu)** feature set, i.e. plain
  `cargo test -p cb-train --test <name>`.
- `crates/cb-train/src/tree_test.rs` (`phantom_mixed_bucket_count`,
  `cat_feature_weight` coverage), `crates/cb-train/src/candidates_test.rs`.
- `crates/cb-model` bake/apply — unchanged; already type-complete.

### Explicitly Out of Scope

- `catboost-master/` — a stale 3-file stub from a **different** upstream
  revision; never a parity reference (worktree `CLAUDE.md` correction block).
- `FloatTargetMeanValue` / `FeatureFreq` — CPU-illegal upstream
  (`restrictions.h:20-32`); rejected by `validate_ctr_types`
  (`boosting.rs:1372-1387`) and `materialize_ctr_feature` (`ctr_feature.rs:311-318`).
- The nine surviving CTR clauses listed under **Out of Scope** above, unless the
  planner escalates Q1.
- The QPACK-01 raw device channel for CTR fits: `raw_device_channel` requires
  `materialized_ctr_features.is_empty()` (`boosting.rs:4567-4570`), so CTR fits
  stay on the host quantize channel. A perf consideration, not a correctness one.

---

## Do Not Hand-Roll

- **The CTR type list**: reuse `ECtrType::from_i8` + `ECtrType::is_cpu_supported`
  (`ctr/mod.rs:96-108, 155-160`). A second inline list will drift.
- **The target-border count**: `ECtrType::target_border_count`
  (`ctr/mod.rs:137-146`). **Do not** substitute `final_ctr_target_border_count`
  (`ctr/mod.rs:246-249`) — they differ for Buckets and swapping them is
  **BUG-BTMV**, documented in that same file.
- **The online CTR value**: `calc_ctr_online` / `calc_ctr_online_bin` /
  `calc_normalization` (`ctr/calc_ctr.rs`) on the CPU side; on the device side
  the existing inline transcription in `ordered_ctr_prefix_kernel`.
- **The combined-projection fold**: `combine_projection_bins` +
  `fold_cat_hash` + `calc_hash` (`ctr_device.rs:68-117`). Do not re-derive
  `MAGIC_MULT` or the sign-extension.
- **The border table**: `((k+1)·norm/bc − shift).next_down()` in
  `build_device_ctr_config` (`boosting.rs:2503-2510`) — proven equivalent to
  `calc_ctr_online_bin`'s truncation under strict `>`. Reuse for Borders,
  Buckets and Counter. **Do not** use it for BTMV (R-3).
- **The cat-feature weight**: `resident_cat_feature_weight` (`mod.rs:4383-4388`)
  and `resident_phantom_bucket_count` (`mod.rs:4392-4398`) already transcribe
  `cat_feature_weight` / `phantom_mixed_bucket_count`. Extend, don't duplicate.
- **The combination eligibility rule**: transcribe `combination_ctr_eligible`
  (`tree.rs:2896-2909`) **once** into cb-backend and have **both** the candidate
  skip and the `eligible_max` filter read it — that lockstep is exactly what
  `eligible_max_bucket_count`'s doc comment demands on the CPU side.
- **The ordered reduction**: `cb_core::sum_f64` for any new host-side float sum.
- **The device-commit assertion**: the `CountingGpu` `Runtime` wrapper from
  `device_ctr_gate_test.rs:60-170`.
- **The fixture recipe**: `ctr_device_combo/gen_fixtures.py` — seed search with
  anti-false-pass guards, `Pool.quantize(border_count=15)` +
  `save_quantization_borders`, and the quantized-vs-raw bit-identity assertion.

---

## Common Pitfalls and Risks

**R-1 — No per-level CTR candidate gate on device (Track D primary).**
*Trigger*: relaxing the projection-arity conjunct without adding the gate.
*Consequence*: combination candidates are scored at level 0, where upstream has
none; the greedy winner flips; 3.3e-2 vs the 1e-5 bar.
*Prevention*: implement D-1 + D-2 together (they share one eligibility predicate).
*Verification*: `device_ctr_combo_fit_test` with a `CountingGpu` wrapper must
reach ≤1e-5 **and** `grown.get() == iterations`.

**R-2 — Fit-lifetime vs tree-lifetime state confusion.**
*Trigger*: reusing `CtrSearchState.group_used` (fit-lifetime, correct for the
`UsedCtrSplits` weight) as the combination-eligibility source.
*Consequence*: a combination becomes eligible at level 0 of tree N because tree
N−1 used its base — upstream resets `seenProj` from `currentTree` each tree
(`greedy_tensor_search.cpp:503-510`), and the CPU resets from `chosen`
(`tree.rs:3090-3100`).
*Prevention*: a **separate**, per-tree `chosen_ctr_projections` list.
*Verification*: a multi-tree unit test where tree 0 uses `[0]` and tree 1's level
0 must still exclude `[0,1]`.

**R-3 — ~~BTMV f32 quantization is not a border comparison~~ — REFUTED.**
*Measured*: spike Q2, 4,504,501 pairs per prior, `border_count = 15`. For every
prior with `shift = 0` and `norm = 1` (i.e. every prior in `[0, 1]`, i.e. every
prior in this repository) the f64 border table, an f32 border table, upstream
`CalcCTR` and the repo BTMV quantizer are **bit-identical**.
`[VERIFIED: python3 q2_spike_fast.py]` **Reuse `binarize_ctr_kernel`; do not add
a BTMV quantize kernel.** The residual (real) hazard is the *opposite* of the
original one: **a prior outside `[0, 1]`** makes the border table and the repo's
uncorrected BTMV quantizer disagree on **93%** of pairs (4.2M/4.5M at prior=2.0)
— which is R-6, closed by **Track E**, not a quantizer problem.
*Prevention*: land Track E first; pin every new fixture's prior in `[0, 1]`.
*Verification*: the characterization test in Track E, plus a device-vs-CPU
bit-exact bin-column test at `prior = 0.5`.


**R-4 — BTMV `Sum` widened to f64 on device.**
*Trigger*: using an `f64` accumulator "for precision".
*Consequence*: silent divergence from `TCtrMeanHistory::Sum` (`online_ctr.h:373`)
and from the CPU's `online_mean_prefix`.
*Prevention*: `f32` accumulator, widened only at the final `f64::from`.
*Verification*: mirror `online_test::btmv_sum_is_accumulated_in_f32_not_f64` on
the device side.

**R-5 — Assuming `counter_calc_method=Full` needs device work.**
*Trigger*: implementing eval-set bin widening on device.
*Consequence*: dead code plus a false sense of coverage.
*Prevention*: rely on the `eval_sets.is_empty()` clause (`boosting.rs:4480`) and
pin it with a negative test.
*Verification*: a gate test asserting `counter_calc_method=Full` + a non-empty
eval set yields `grown.get() == 0`.

**R-6 — ~~Mirroring the CPU's missing BTMV normalization~~ → now Track E (FIX it).**
*Ruling*: the coordinator directed the CPU gap be **fixed** in this phase, not
mirrored-and-filed. *Trigger for the remaining risk*: fixing it without first
proving the no-op. *Consequence*: an unnecessary fixture re-baseline task, or —
worse — a silent value change nobody checked for. *Prevention*: the proof is
already done — `calc_normalization(p) == (0.0, 1.0)` for every `p` in `[0, 1]`
(`calc_ctr.rs:60-66`), spike Q2 measures 0 mismatches at `p` in `{0, 0.5, 1}`
over 4.5M pairs each, and every prior in the repo is `{0, 0.25, 0.5, 1}`
`[VERIFIED: grep over all fixture config.json + Rust tests]`.
*Verification*: `ctr_btmv_simple_oracle_test` must stay green **unchanged** —
that is the no-op proof — plus a new `prior = 2.0` unit test pinning the
corrected bin (uncorrected gives 30, correct is 15 at `good=0, total=0`).

**R-7 — f32-vs-f64 `CalcCTR` width (pre-existing) — QUANTIFIED, inert today.**
*Measured*: upstream's all-`float` `CalcCTR` (`online_ctr.h:128-131`) vs the
repo's f64 `calc_ctr_online_bin` gives **0 mismatches** at `prior` in
`{0, 0.5, 1, 2}` over 4.5M pairs each, and **899** mismatches only at
`prior = -0.5` `[VERIFIED: python3 q2_spike_fast.py]`.
*Trigger*: a user-supplied **negative** prior. *Consequence*: the repo's CPU bins
diverge from upstream's, and the device border table diverges from the repo's CPU
(1200 mismatches at `prior = -0.5`) — a three-way disagreement with no oracle.
*Prevention*: out of scope for this phase (no fixture uses a negative prior), but
**worth a validation guard**: reject or warn on `prior < 0` at the parameter
surface, mirroring how `parse_ctr_description` already rejects `denom != 1`.
*Verification*: a characterization test pinning the current behaviour at
`prior = -0.5` so a future change is visible.


**R-8 — Device e2e tests that pass on CPU.**
*Trigger*: asserting only `oblivious_trees.len() == iterations`.
*Consequence*: exactly the current state of `device_ctr_combo_fit_test`
`[VERIFIED: it passes with `--ignored` at 1.388e-17 — the CPU number]`.
*Prevention*: `CountingGpu` in every new/edited device e2e test.
*Verification*: temporarily force the gate closed and confirm the new test FAILS.

**R-9 — `--features rocm` without `--no-default-features`.**
*Trigger*: a bare `cargo test -p cb-backend --features rocm`.
*Consequence*: `cb-backend`'s `default = ["cpu"]` stays on, `SelectedRuntime`
resolves to `cubecl-cpu`, and "device" tests compare cpu-vs-cpu — a false pass.
*Prevention*: always pair the two flags.
*Verification*: none automated; call it out in every task's verify block.

**R-10 — Package-wide `cargo test -p cb-train --no-default-features`.**
*Trigger*: omitting `--test <name>`.
*Consequence*: ~52 test files import `cb_backend::CpuBackend` at module scope and
fail `E0432`, blocking the whole run (documented in `run_device_tests.sh:2-4`).
*Prevention*: one `--test <name>` per invocation; extend `run_device_tests.sh`.
*Verification*: `run_device_tests.sh` exit code.

**R-11 — The `border_count == 15` invariant silently declining a fixture.**
*Trigger*: a new fixture generated with catboost's default `border_count` (254).
*Consequence*: `ctr_covered`'s `borders.len() + 1 == n_bins` fails, the fit runs
on CPU, and (absent `CountingGpu`) the test still passes.
*Prevention*: pin `"border_count": 15` in every device CTR fixture's params and
assert `len(per_feature[fi]) == 15` in the generator (the combo generator already
does).
*Verification*: `CountingGpu`.

**R-12 — Frozen-fixture regeneration.**
*Trigger*: re-running any `gen_fixtures.py` after a change.
*Consequence*: catboost's quantization is run-to-run nondeterministic on
categorical routing; regenerating invalidates the ≤1e-5 gate for **all**
downstream tests.
*Prevention*: generate once, commit, never regenerate in CI. Every new fixture
must carry the `"note": "FROZEN…"` marker and the reproducibility caveat
docstring.

**R-13 — `poisson_parallel_draw_outpaces_the_serial_stream_draw` flakiness.**
*Trigger*: running the perf lane inside the package-wide cb-backend suite.
*Consequence*: 3.3–3.8× under contention vs a 5× bar; a pre-existing, known,
**do-not-chase** flake. It read **9.6×** in the isolated lane this session
`[VERIFIED: run_device_tests.sh perf lane]`.
*Prevention*: keep it in `run_device_tests.sh`'s isolated `PERF_TESTS` lane.

**R-14 — A purely categorical ±1 target hides leaf-assignment bugs.**
*Trigger*: a fixture whose target is a pure function of the cat columns.
*Consequence*: structure-vs-averaging leaf swaps become numerically invisible.
*Prevention*: every new device CTR fixture must include a **per-object float
ramp** in the target — the combo generator's `logit = 3.0*(pair - 0.5) + 2.0*(x[:,0] - x[:,1])`
is the working template. Also keep the existing
structure-vs-averaging-permutation divergence guard
(`assert_ne!(structure, averaging)`, `device_ctr_combo_fit_test.rs:112-121`).

**R-15 — `model.json` float borders are the PRUNED used subset.**
*Trigger*: feeding the Rust trainer borders read out of `model.json`.
*Consequence*: a different quantization than upstream trained with.
*Prevention*: freeze the **full** set via `Pool.quantize()` +
`save_quantization_borders` into `borders.npy` (the combo generator's recipe).

---

**R-16 (C-1) — the "phantom border can never win" assumption.**
*Trigger*: shipping C-1a (`ctr_covered` `==` -> `<=`) without C-1b (the
per-feature threshold bound). *Consequence*: with `border_count = 254` a CTR
column is enumerated over borders `0..=253`; a phantom winner at `bin_id >= 16`
would be fed to `ctr_bin_border_to_value_space`, whose domain is hard-bounded at
15 (`tree.rs:3260-3290`) — the persisted border would be silently wrong.
*Prevention*: ship C-1b (bound the THRESHOLD arm by `real_folds[feature]`, as the
one-hot arm already does). *Verification*: a unit test constructing a histogram
where a real candidate and a phantom candidate tie, asserting the real one wins;
plus a `b = 16` guard test mirroring the existing `tree_test.rs` characterization.

**R-17 (C-2) — committing an UNANCHORED multi-permutation config to the device.**
*Trigger*: relaxing `learning_folds_for_cycle == 1` unconditionally.
*Consequence*: for any `(permutation_count, seed)` other than `(4, 0)`,
`structure_fold_cycle` returns the constant `Folds[0]` fallback, which the code
itself labels "rather than ship an un-instrumented guess"
(`boosting.rs:678-689`) — the device would faithfully reproduce a CPU reference
that is a documented approximation of upstream. *Prevention*: gate the device
relaxation on the cycle being **anchored** (`pc == 4 && seed == 0`) rather than on
`folds == 1`. *Verification*: a negative test asserting `pc = 3` still declines.

**R-18 (C-6) — admitting exact-leaf x CTR.**
*Trigger*: dropping `!config.exact_leaf` along with the other four sub-clauses.
*Consequence*: `compute_exact_leaf_values` and the averaging-permutation gather
are written as mutually exclusive overrides of the same `leaf_values`
(`session.rs:2698-2784`); admitting both would apply one and silently discard the
other. *Prevention*: keep `!exact_leaf`; lift the other four independently.
*Verification*: a negative composition test.

**R-19 (C-3, C-7) — mistaking a CPU feature for a device clause.**
*Trigger*: planning C-3 (one-hot x CTR) or C-7 (non-symmetric x CTR) as device
work. *Consequence*: the device clause is lifted, the fit reaches
`train_inner`... which errors out at `boosting.rs:3937` (C-3) or has no
`ctr_features` parameter to pass at all (C-7) — wasted work with no observable
change. *Prevention*: recognise both as CPU-side features first
`[VERIFIED: SPEC-OH-26 rejection site; `grow_nonsym_tree`/Region signatures carry
no CTR parameter]`. *Verification*: before any device work on either, a CPU test
must first train the configuration successfully.

**R-20 — losing the D-2 coverage gap.**
*Trigger*: skipping D-2 because spike Q5 showed D-1 alone closes the combo
fixture. *Consequence*: `eligible_max` keeps maxing over ineligible combinations,
diverging from `eligible_max_bucket_count` (`tree.rs:2920-2933`) on any fixture
where the combination's `bucket_count` dominates and a level-0 decision is close.
*Prevention*: ship D-2 regardless; record explicitly that **no fixture currently
detects it**. *Verification*: build a D-2-discriminating fixture, or accept the
gap knowingly and document it in the code.


## Testing and Verification Strategy

### Verified-working commands on this machine (2026-08-08)

| Purpose | Command | Status |
|---|---|---|
| Full device DoD suite | `bash ./run_device_tests.sh` | `[VERIFIED: 23/23 PASS + perf PASS]` |
| One device e2e | `cargo test -p cb-train --no-default-features --features rocm --test device_ctr_combo_fit_test` | `[VERIFIED: ok, 1 ignored; cold build 52.18s]` |
| An `#[ignore]`d device e2e | `cargo test -p cb-train --no-default-features --features rocm --test device_ctr_combo_fit_test -- --ignored --nocapture` | `[VERIFIED: 1 passed — and this is the R-8 false pass]` |
| Backend CTR kernel self-oracles | `cargo test -p cb-backend --no-default-features --features rocm --lib kernels::ctr_device_test` | `[VERIFIED: 6 passed]` |
| Isolated perf lane | `cargo test -p cb-backend --no-default-features --features rocm --lib kernels::poisson_bootstrap_speed_test -- --ignored --nocapture --test-threads=1` | `[VERIFIED: 9.6×, bar 5×]` |
| CPU CTR oracles (default features) | `cargo test -p cb-train --test ctr_buckets_simple_oracle_test` (etc.) | `[ASSUMED — not run this session; these are default-feature tests and were green at HEAD per the repo's own CI discipline]` |
| Build-only check without a GPU | `cargo check -p cb-train --no-default-features --features rocm --test <name>` | `[ASSUMED — documented in the sibling gpu-device-coverage-expansion research; not re-run this session because a real GPU is present]` |
| GPU present? | `/home/user/rocm/opt/rocm/bin/rocminfo \| grep -E "Name:\|gfx"` | `[VERIFIED: gfx1151, AMD Radeon 860M]` |
| **Reproduce spike Q2** | `python3 <scratchpad>/q2_spike_fast.py` (self-contained numpy; ~40 s) | `[VERIFIED: 4,504,501 pairs x 5 priors]` |
| **Reproduce spike Q5 control** | scratch-edit `spike_combination_eligible` to `return true`, then `cargo test -p cb-train --no-default-features --features rocm --test device_ctr_combo_fit_test -- --ignored --nocapture` | `[VERIFIED: FAIL, obj 0 \|Δ\|=2.746e-2]` |
| **Reproduce spike Q5 fixed** | same with the real predicate | `[VERIFIED: PASS, max \|Δpred\| = 2.082e-17]` |
| Incremental rocm rebuild after a 4-file edit | (implicit in the above) | `[VERIFIED: 4.92 s]` |

**Do NOT** run a bare `cargo test -p cb-train --no-default-features --features
rocm` (R-10) or a bare `--features rocm` without `--no-default-features` (R-9).

### Unit / kernel self-oracles (per track)

Extend `crates/cb-backend/src/kernels/ctr_device_test.rs` (currently 6 tests,
all passing) with, per new type, a device-vs-CPU-reference pair mirroring
`ordered_ts_matches_cpu_reference`:
- **Buckets**: `good`/`total` integer equality for `target_border_idx ∈ {0, 1}`,
  plus a bit-exact bin column (`ctr_binarized_cindex_column_bit_exact` template).
- **BTMV**: an `f32`-sum equality test (bit-level on the `f32`), plus a
  **bit-exact** bin test over inputs deliberately placed on quantization
  boundaries (R-3).
- **Counter**: whole-set totals + the MAX denominator, plus a
  **permutation-invariance** test (run the same column under two different
  permutations and assert identical bins — the structural property
  `IsPermutationDependentCtrType(Counter) == false`, `ctr_type.cpp:43-56`).
- **Track D**: a pure-host unit test of the transcribed
  `combination_ctr_eligible` (empty chosen ⇒ every combination ineligible;
  `q ⊆ p` with `|q|+1 == |p|` ⇒ eligible), and an `eligible_max` test showing
  the ineligible combination is excluded from the max.

The CPU reference inside these tests must be an **inline transcription** (the
existing `cpu_ordered_ctr` helper at `ctr_device_test.rs:55-84` is the template),
because cb-backend cannot depend on cb-train.

### Integration / gate tests

New `crates/cb-train/tests/device_ctr_type_gate_test.rs` (or extend
`device_ctr_gate_test.rs`), all with `CountingGpu`:
- Each of Buckets / BTMV / Counter / combination **commits** (`grown == iterations`).
- `counter_calc_method = Full` + a non-empty eval set still **declines** (R-5).
- `permutation_count > 1` + any CTR type still **declines** (preserve
  `multi_permutation_ctr_declines_to_device`).
- A pool with `border_count != 15` (i.e. float borders ≠ 15) still **declines**
  (R-11) — this pins the surviving invariant explicitly rather than by accident.
- `FloatTargetMeanValue` / `FeatureFreq` still error out on CPU
  (`validate_ctr_types`) — a `cargo test -p cb-train` default-feature test.

### End-to-end oracles (the ≤1e-5 bar)

One per track, mirroring `device_ctr_fit_test.rs` **plus** `CountingGpu`:
- `device_ctr_buckets_fit_test` → new `ctr_device_buckets/` fixture.
- `device_ctr_btmv_fit_test` → new `ctr_device_btmv/` fixture.
- `device_ctr_counter_fit_test` → new `ctr_device_counter/` fixture.
- `device_ctr_combo_fit_test` (existing, un-ignored + `CountingGpu`) →
  existing `ctr_device_combo/` fixture — **no new fixture needed for Track D**.

Each must keep the structure-vs-averaging permutation divergence guard
(`assert_ne!(create_shuffled_indices(n, seed), averaging_ctr_permutation(n, 1, seed))`)
so a structure-only leaf gather cannot pass.

### Fixture generation checklist (per new fixture)

1. Copy `crates/cb-oracle/fixtures/ctr_device_combo/gen_fixtures.py`.
2. Change only `SCENARIO`, `simple_ctr`, `max_ctr_complexity`,
   `combinations_ctr`, and the type-specific anti-false-pass guard.
3. Keep: `border_count: 15`, `one_hot_max_size: 1`, `boosting_type: "Plain"`,
   `permutation_count: 1`, `bootstrap_type: "No"`, `random_strength: 0`,
   `score_function: "L2"`, `leaf_estimation_method: "Gradient"`,
   `leaf_estimation_iterations: 1`, `boost_from_average: False`,
   `thread_count: 1`, `verbose: False`, and **no `task_type`** (CPU oracle).
4. Keep the seed search with a real anti-false-pass guard (≥1 float split AND the
   type-specific property, e.g. both `ctr_target_border_idx` values for Buckets).
5. Freeze the full float border set (`Pool.quantize` + `save_quantization_borders`
   → `borders.npy`) and assert the quantized-pool fit is **bit-identical** to the
   raw-pool fit.
6. Record the fixed-point overflow margin in the docstring
   (`n · max(w) · max(|der1|) ≪ 2^33`) — mandatory per SPEC §9; with Logloss and
   unit weights at `n = 64` the margin is > 1.3e8×.
7. Commit `config.json` with `"note": "FROZEN…"` and the FPP/track requirement id.

### Ordering / regression

Run `bash ./run_device_tests.sh` after every track lands, plus the CPU CTR oracle
set under default features. The green baseline captured this session is the
regression reference.

---

## Planning Guidance (REVISED for the full admitted scope)

### Phase-size assessment and staging recommendation — READ THIS FIRST

The admitted scope is now **eleven** work items: Tracks 1a, A, B, C, D, E plus
C-1, C-2, C-4, C-5, C-6. That is **not** one buildable phase, and shipping it as
one would leave the user without a finish line for a long time. But the good news
is that the spikes collapsed the two scariest items (Track B's kernel, Track D's
root cause) and two clauses turned out to be **provably dead** (Track 1a's
`prior_denom`, C-3's `one_hot_bins`), so the remaining work partitions unusually
cleanly.

**Recommendation: three sequential phases, with a real finish line each.** This
delivers the user's "everything" ruling in full, but as three reviewable units
rather than one unbounded one.

| Phase | Contents | Finish line | Rough size |
|---|---|---|---|
| **P1 — CTR type + projection completeness** | Track 1a (dead clause), Track E (CPU BTMV fix), seam widening, Track A (Buckets), Track C (Counter), Track B (BTMV), Track D (combination) | `ctr_types_are_device_covered` has **no type/arity/border/prior conjunct left**; four new device e2e oracles green; `device_ctr_combo_fit_test` un-ignored with a `CountingGpu` assertion | Medium. Track D is **already measured working**; Track B collapsed to an accumulator; Track 1a is a deletion. |
| **P2 — CTR reachability** | C-1 (`border_count != 15`), C-5 (cat-only pools), C-6 (bootstrap/MVS/host-sampling/Lossguide-cap composition, **excluding** exact-leaf) | A **default-parameter** (`border_count = 254`) CTR fit commits to the device; the eight existing cat-only CTR fixtures become device oracles | Medium. C-1b touches the shared scorer bound — highest regression risk in the whole scope. |
| **P3 — CTR × the remaining families** | C-2 (multi-permutation), C-4 (eval sets + Counter `Full`) | A `pc=4, seed=0` CTR fit and a `counter_calc_method=Full` fit both commit | Large. C-2 needs `folds+1` resident trajectories; C-4 needs device eval-set support. |

**Deliberately NOT in the three phases, each with evidence** (see "Out of Scope —
evidence-backed"): C-8 (no CPU parity surface), C-9 (wgpu has no f64),
`FloatTargetMeanValue`/`FeatureFreq` (upstream CPU rejects them).

**And two items the "everything" ruling buys that are CPU features, not device
coverage** — recommend each get **its own** spec, sequenced after P3:
- **C-3 one-hot × CTR** — needs a three-way candidate union in the CPU level
  search. The device clause is dead; lifting it changes nothing until the CPU can
  train the configuration.
- **C-7 CTR × non-symmetric / Region** — the CPU growers take no `ctr_features`
  argument at all, and the upstream non-symmetric CTR candidate-set semantics were
  **not read this session** `[UNVERIFIED]`. Largest single item in the whole
  scope.

If the user wants one phase rather than three, **P1 alone is the right one**: it
is the only unit that fully closes the original five tracks, it has the highest
value per unit of risk, and every one of its hard questions is now measured.

### Suggested ordering across ALL admitted tracks, with dependency edges

```
P1
 0. Seam widening  (DeviceCtrColumn += ctr_type, target_border_idx, projection_members)
      | (blocks everything below in P1)
 1. Track 1a  — delete the dead `prior_denom == 1.0` conjunct  [independent, do first]
 1'. C-3-dead — delete the dead `one_hot_bins.is_empty()` conjunct  [independent]
 2. Track E   — fix the CPU BTMV (ctr+shift)/norm            [MUST precede Track B]
      |
 3. Track A   — Buckets + multi-target-border                [needs 0]
 4. Track C   — Counter                                      [needs 0; independent of A]
 5. Track B   — BTMV accumulator                             [needs 0 and 2]
      |
 6. Track D   — combination CTR (D-1 + D-2)                  [needs 0; MUST be last in P1,
                                                              because it must gate the
                                                              types added by 3/4/5]
 7. Fixtures  — ctr_device_{buckets,btmv,counter}            [parallel with 3/4/5;
                                                              blocks their e2e tests]
 8. Test fix  — CountingGpu in device_ctr_{fit,combo_fit}_test

P2  (needs P1's seam)
 9. C-1a  ctr_covered  `==` -> `<=`
10. C-1b  per-feature THRESHOLD bound (host belt + kernel)   [MUST accompany 9; R-16]
11. C-1c  border_count=254 device CTR fixture + e2e
12. C-1d  phantom-border negative-control unit test
13. C-5   cat-only pools                                     [needs 9/10: n_bins from CTR width]
14. C-5'  re-point the 8 existing cat-only CTR fixtures at device oracles  [needs 13]
15. C-6   bootstrap / MVS / host-sampling / Lossguide-cap composition
                                                             [independent of 9-14;
                                                              exact-leaf stays OUT, R-18]

P3
16. C-2   multi-permutation (gated on the ANCHORED pc=4/seed=0 cycle, R-17)
17. C-4a  device eval-set support (host-side eval scoring off DeviceGrownTree.leaf_of)
18. C-4b  Counter `Full` eval widening on device             [needs 17]
```

**Critical edges to preserve:**
- **0 before everything** — the seam carries `ctr_type`/`target_border_idx`
  (Tracks A/B/C) and `projection_members` (Track D).
- **2 (Track E) before 5 (Track B)** — so "device == CPU" is provable against the
  *corrected* CPU at every point, never against a known-wrong one.
- **6 (Track D) last in P1** — D-1's eligibility gate must cover the Buckets /
  BTMV / Counter columns added by 3/4/5, not just Borders.
- **10 with 9, never 9 alone** — R-16.
- **13 after 9/10** — C-5 derives `n_bins` from the CTR width, which only makes
  sense once the CTR and float widths are allowed to differ.

### Decisions the planner must preserve

D-04, D-05, D-06, D-07, D-10-01, WR-02, SPEC-CTRB-01 (`bin_border <= 15` — now
load-bearing for C-1, see R-16), and the honest-gate discipline: when a
configuration cannot be made upstream-faithful, **decline with a documented,
negatively-tested clause** rather than ship a silently-approximate device result.
That discipline is what produced the FPP-11 escalation, and it is what R-17
applies to C-2.

Also preserve: source/test separation; no `unwrap` in production; generics-float
for new kernels; `cb-backend` never depends on `cb-train`; `cb-compute` stays
`cubecl`-free.

### Items still needing a user decision

1. **Three phases or one?** (recommendation above). If one, take P1.
2. **Track B (a) alias vs (b) honest accumulator** — recommendation (b).
3. **C-6 exact-leaf** — recommend it stays OUT (R-18); confirm.
4. **C-2 anchoring gate** — recommend gating on `pc == 4 && seed == 0` rather
   than on `folds == 1` (R-17); confirm, because it means "multi-permutation CTR
   on device" ships for exactly one config family.
5. **C-3 and C-7 as separate CPU-side specs** — confirm they are accepted as
   follow-on phases rather than being forced into this one.


## Open Questions

**All five first-pass open questions are now CLOSED.** Q1 and Q4 were ruled on by
the user; Q2, Q3 and Q5 were answered by measurement or by a recommendation.

| First-pass Q | Status | Resolution |
|---|---|---|
| Q1 scope boundary | **CLOSED (ruling)** | "Everything." 6 of the 9 clauses IN, 3 OUT with evidence. See the verdict table. |
| Q2 BTMV quantizer | **CLOSED (measured)** | No dedicated kernel. 0 mismatches over 4.5M pairs for every reachable prior. |
| Q3 Counter kernel shape | **CLOSED (recommendation)** | Serial two-pass tally on unit 0, reusing the existing kernel shape and the existing border table. No profiling spike run: the tally is O(n) integer work on the same serial unit the prefix already uses, and Counter columns are built **once per fit** inside `begin` (`session.rs:1725-1789`), not per tree — so it cannot be a hot path. A parallel `Atomic<u32>` tally would add kernels and atomics for no measurable gain at fixture scale. `[UNVERIFIED at large n: not profiled. If a 1M-row Counter fit ever shows up in the SPD bench, re-measure.]` |
| Q4 CPU BTMV normalization | **CLOSED (ruling + measured)** | Fix it (Track E). Proven a no-op for every repo artifact; no fixture re-baseline. |
| Q5 Track D confirmation | **CLOSED (measured)** | Control 2.746e-2 -> fixed 2.082e-17. D-1 alone sufficed on this fixture; D-2 still required for CPU parity but has no detector (R-20). |

### Genuinely open

1. **The upstream non-symmetric CTR candidate-set semantics (C-7).** Upstream's
   `AddTreeCtrs` builds `seenProj` from `currentTree.GetUsedCtrs()` for the
   *oblivious* search. The depthwise / lossguide entry points in
   `greedy_tensor_search.cpp` were **not fetched or read this session**, so
   whether a leaf-wise tree's eligibility set is the whole tree's used CTRs or
   only the path to the node being split is unknown. `[UNVERIFIED — must be read
   before any C-7 design.]` Blocks C-7 only, which is recommended as its own
   phase anyway.
2. **Does a phantom border ever win? (C-1, R-16.)** The analytic argument (an
   all-left split leaves one child empty, so its score equals the un-split parent
   and strict `>` cannot displace a real candidate) is sound for L2/Cosine but
   was **not** checked against the GPU-only score functions (`SolarL2`, `LOOL2`,
   `SatL2`), whose `adjust` factors behave differently at zero weight
   (`runtime.rs:842-863`). `[UNVERIFIED]` Resolution path: the C-1d
   negative-control unit test. This is why C-1b is recommended over C-1a alone —
   it makes the question moot.
3. **Is there a D-2-discriminating fixture? (R-20.)** Needs a combination whose
   `bucket_count` dominates every simple column's *and* a level-0 decision on a
   thin margin. `[UNVERIFIED whether such a fixture is constructible at the small
   `n` the frozen-fixture discipline prefers.]` Resolution path: a seed search
   like `ctr_device_combo`'s, with the discriminating property as the
   anti-false-pass guard.
4. **Phase staging** — three phases or one? A recommendation, not a fact; needs
   the user's confirmation (Planning Guidance, decision 1).

### Feasibility floor (revised)

No admitted track now looks infeasible at ≤1e-5, and two that previously did have
been measured working:
- **Track D is measured green** (2.082e-17).
- **Track B's quantizer risk is refuted** (0 mismatches).
- **Track A and Track C** reuse the existing, oracle-verified border table
  wholesale, because their statistics go through the same f64
  `calc_ctr_online_bin` path as Borders.
- **C-1** is bounded by infrastructure that already ships for float features.
- **C-2** is the only admitted item whose *parity ceiling* is externally capped
  (by the CPU's own `structure_fold_cycle` anchoring), which is why R-17
  recommends gating rather than relaxing.

**Fallback if any track cannot reach ≤1e-5**: leave that single clause in the
gate as an explicit, doc-commented, **negatively-tested** decline — the existing
FPP-11 pattern — but insist the negative test carry a `CountingGpu` assertion so
the decline is *proven*. Today's FPP-11 test does not, which is precisely how it
became a false pass.
---

## Sources

### Project documents / plans (plain files; this repo has no TreeFinder-indexed corpus)
- `/home/user/Documents/workspace/catboost_rs/.claude/worktrees/gpu-borders-shared-sample/CLAUDE.md`
  (worktree copy, incl. the stale-`catboost-master/` correction block).
- `.planning/plans/gpu-device-coverage-expansion/research.md` (1090 lines, read in
  full — the closest prior art; its phase shipped the current device CTR support).
- `.planning/plans/` directory inventory (21 sibling plans).
- **TreeFinder MCP was not used**: this repository's specification store is the
  plain-file `.planning/plans/<slug>/` corpus, not a TreeFinder-indexed document
  set (same conclusion the sibling report reached). Files were read directly.

### CodeGraph MCP (`mcp__codegraph__codegraph_explore`)
Queries: `ctr_types_are_device_covered build_device_ctr_config DeviceCtrConfig
DeviceCtrColumn materialize_ctr_feature`; `select_level_ctr_aware
phantom_mixed_bucket_count cat_feature_weight UsedCtrSplits eligible_max maxCount
tensor_ctr_candidates`.
**Staleness caveat (important)**: CodeGraph's index is rooted at the **main
checkout** (`/home/user/Documents/workspace/catboost_rs`), not this worktree — it
warned so explicitly on every call. Every load-bearing CodeGraph result above was
re-verified against the worktree with `sed`/`grep`/`Read`. CodeGraph was used for
**blast radius only** (caller/dependent counts), and those counts are marked
`[CODEGRAPH]`.

### Upstream v1.2.10 (fetched raw 2026-08-08 from
`https://raw.githubusercontent.com/catboost/catboost/v1.2.10/…`) `[WEB]`
- `catboost/private/libs/algo/online_ctr.h` — `CalcCTR` (`:128-131`),
  `TOnlineCtrBase::GetUniqValuesCounts`.
- `catboost/private/libs/algo/online_ctr.cpp` — `CalcNormalization` (`:102-112`),
  `UpdateGoodCount` (`:115-121`), `CalcOnlineCTRClasses` (`:144-212`),
  `CalcOnlineCTRCounter` (`:503-568`), `ComputeOnlineCTRs` (`:680-795`),
  `CalcFinalCtrsImpl` (`:900-940`).
- `catboost/private/libs/algo/ctr_helper.h` — `GetTargetBorderCount` (`:35-42`).
- `catboost/private/libs/algo/ctr_helper.cpp` — `CB_ENSURE(denom == 1.0, …)` (`:50`).
- `catboost/private/libs/algo/greedy_tensor_search.cpp` — `AddCtrsToCandList`
  (`:400-429`), `AddSimpleCtrs` (`:457-487`), `AddTreeCtrs` (`:491-551`),
  `GetCatFeatureWeight` (`:908-932`), `SelectBestCandidate` (`:934-970`),
  `SelectDatasetFeaturesForScoring` (`:973-1025`), `CalcMaxFeatureValueCount`
  (`:1070-1088`), `GreedyTensorSearchOblivious` (`:1171-1230`).
- `catboost/private/libs/options/cat_feature_options.cpp` — `MaxTensorComplexity`
  default 4 (`:231`), `StoreAllSimpleCtrs` false (`:235`), `CtrLeafCountLimit`
  `Max<ui64>()` (`:236`).
- `catboost/private/libs/options/oblivious_tree_options.cpp` — `ScoreFunction`
  Cosine (`:22`), `ModelSizeReg` `0.5f` (`:28`).
- `catboost/private/libs/options/restrictions.h` — `IsSupportedCtrType` (`:18-48`).
- `catboost/libs/data/ctrs.h` — `TOnlineCtrUniqValuesCounts` (`:49-70`).
- `catboost/private/libs/ctr_description/ctr_type.{h,cpp}`,
  `catboost/private/libs/algo/index_hash_calcer.h`,
  `catboost/libs/model/ctr_value_table.h`, `catboost/private/libs/algo/projection.h`
  (fetched; consulted for cross-checks).

### CubeCL documentation
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` (read in
  full). Relevant entries for this phase: `Cubecl_conditionals.md`,
  `Cubecl_loop_control.md`, `Cubecl_generics.md`, `comptime_macro.md`,
  `Cubecl_comptime_specialization.md`, `08_atomic_contention.md`,
  `09_fixedpoint_atomics.md`, `11_launch_overhead_and_transfers.md`.
- `.../Cubecl/cubecl_error_solution_guide/` — the **actual** error guidance
  (2 files). `cubecl_error_guideline.md` referenced by CLAUDE.md/AGENTS.md does
  **not** exist `[VERIFIED: ls]`.
- **Context7 CLI was not invoked**: no external library/framework question arose
  — every dependency (`cubecl 0.10.0`, `bytemuck`, `rayon`) is already adopted
  in-repo and the authoritative CubeCL reference for this project is the local
  manual mandated by AGENTS.md.

### Local files read in the worktree (all `[VERIFIED]`)
`crates/cb-train/src/boosting.rs` (lines 1360-1395, 2150-2296, 2298-2366,
2429-2650, 3276-3310, 4070-4110, 4380-4560, 4630-4700, 5270-5350, 7270-7300);
`crates/cb-train/src/ctr/{mod.rs, calc_ctr.rs, online.rs, ctr_feature.rs}`;
`crates/cb-train/src/tree.rs` (2760-3300);
`crates/cb-train/src/candidates.rs`;
`crates/cb-compute/src/runtime.rs` (1095-1200, 1344-1356);
`crates/cb-backend/src/kernels/ctr_device.rs` (full);
`crates/cb-backend/src/gpu_runtime/session.rs` (110-260, 1690-1810, 2700-2800);
`crates/cb-backend/src/gpu_runtime/mod.rs` (4330-4420, 4740-4790, 4860-5010);
`crates/cb-backend/Cargo.toml`; `Cargo.toml`;
`crates/catboost-rs/src/builder.rs` (grep); `crates/catboost-rs-py/src/params.rs`
(lines 41-52, 745-775, greps);
`crates/cb-oracle/fixtures/ctr_device_combo/gen_fixtures.py` (full);
`crates/cb-oracle/fixtures/{ctr_buckets_simple,ctr_btmv_simple,ctr_counter_simple}/config.json`;
`crates/cb-oracle/fixtures/ctr_device_combo/model.json` (python dump);
`crates/cb-train/tests/{device_ctr_combo_fit_test.rs, device_ctr_gate_test.rs,
device_ctr_fit_test.rs}`; `crates/cb-backend/src/kernels/ctr_device_test.rs` (grep);
`crates/cb-train/src/device_ctr_combo_config_test.rs`; `run_device_tests.sh`.

### SECOND PASS — additional sources and commands (2026-08-08)

**Upstream v1.2.10, additionally read this pass** `[WEB]` (same raw-fetch base):
- `online_ctr.cpp:437-501` `CalcOnlineCTRMean` — the BTMV accumulation AND the
  `CalcCTR(sum, count, priorX, shiftX, normX, bc)` call that carries `shift`/`norm`
  (Track E's ground truth), plus `:449-451` `CalcNormalization`.
- `online_ctr.cpp:271-330` `CalcQuantizedCtrs` — the binclf Borders fast path
  (`goodCount = elem[1]`, `totalCount = elem[0]+elem[1]`), the comparison that
  establishes **BTMV ≡ Borders at binclf**.
- `online_ctr.cpp:757-767` — the `targetClassesCount - 1` argument to
  `CalcOnlineCTRMean`.
- `cat_feature_options.cpp:118-138` — the per-type default priors (the
  `[Borders(0/1,0.5/1,1/1), Counter(0/1)]` CPU default C-8 cannot express).

**Worktree files additionally read this pass** `[VERIFIED]`:
`crates/cb-train/src/boosting.rs` (645-690 `structure_fold_cycle`, 3918-3960
SPEC-OH-26 rejection, 4148-4180 `learning_folds_for_cycle` +
`structure_fold_columns`, 4870-4935 `fold_approxes` + `exp_ctr`, 4770-4840 the
`begin_device_training` fold_count threading, 4567-4570 `raw_device_channel`);
`crates/cb-train/src/fold.rs:168-200` `learning_fold_count`;
`crates/cb-backend/src/gpu_runtime/session.rs` (1010-1018 `pad_hist_line_bins`,
1555-1600 the `n_bins_line` padding rationale + the QPACK raw-channel decline,
1794-1820 `real_folds` / `n_buckets_per_feature`, 2588-2600 the
`ResidentCtrSearch` view construction, 2698-2701 the exact-leaf override);
`crates/cb-backend/src/gpu_runtime/mod.rs:3420-3470` (the host belt mirroring the
kernel's border eligibility);
`crates/cb-train/tests/device_ctr_gate_test.rs:60-170` (the `CountingGpu` pattern
and the `grow_tree_on_device` seam signature);
`crates/catboost-rs/src/builder.rs:97-122` (the single-`ECtrType` fields);
`crates/catboost-rs-py/src/params.rs:41-52, 826-840` (SPEC-CTRT-19 and the raise).

**Spike artifacts (scratchpad, not in the repo)**:
`q2_spike.py` (scalar reference implementation) and `q2_spike_fast.py`
(vectorised, the one that produced the reported table).

**Commands run this pass** `[VERIFIED]`:
- `python3 q2_spike_fast.py` — the BTMV/normalization/width table (4,504,501
  pairs per prior, 5 priors).
- `grep -rho "Prior=[-0-9./]*" crates/cb-oracle/fixtures/*/config.json | sort -u`
  → `{Prior=0, Prior=0.25, Prior=0.5, Prior=1}`; and the `ctr_priors: vec![…]`
  grep over `crates/cb-train/tests/*.rs` → `{vec![0.25], vec![0.5]}`.
- Scratch spike Q5: four production files edited, built
  (`cargo build -p cb-train --no-default-features --features rocm` → clean), and
  three measured arms run; then `git checkout --` on all four.
- `bash ./run_device_tests.sh` **under the spike** → 23/23 PASS + perf 10.7×.
- Post-revert `git status --short` → only `?? .planning/plans/device-ctr-full-coverage/`;
  `git log --oneline -1` → `a0a67ec`; post-revert combo test → `1.388e-17`.
- `python3` dump of `ctr_device_combo/model.json` oblivious_trees (level-0 split
  is the simple projection in all five trees).


### Commands run this session (`[VERIFIED]`)
- `git log --oneline -3` → HEAD `a0a67ec`; `git status --short` → clean.
- `/home/user/rocm/opt/rocm/bin/rocminfo | grep -E "Name:|gfx"` → `gfx1151`,
  AMD Radeon 860M; `ls -l /dev/kfd` → present; `ls -d /opt/rocm*` → **absent**
  (ROCm lives under `/home/user/rocm/opt/rocm`).
- `df -h /home` → 631 G avail; `du -sh target` → 6.1 G.
- `python3 -c "import catboost; print(catboost.__version__, …)"` → `1.2.10`.
- `cargo test -p cb-train --no-default-features --features rocm --test
  device_ctr_combo_fit_test` → ok, 1 ignored (cold build 52.18 s).
- same `-- --ignored --nocapture` → **1 passed**, `max |Δpred| = 1.388e-17`
  (the R-8 false pass).
- `cargo test -p cb-backend --no-default-features --features rocm --lib
  kernels::ctr_device_test` → 6 passed.
- `bash ./run_device_tests.sh` → 23/23 PASS + perf lane PASS (Poisson 9.6×).
- `python3` dump of `ctr_device_combo/model.json` → 2 CTR descriptors; level-0
  split is the simple projection in all 5 trees.

---

## Confidence Assessment — SECOND PASS DELTA

Read this before the first-pass assessment below; it supersedes the entries it
names.

**Promoted to HIGH by measurement this pass:**
- **FPP-11 / Track D root cause AND sufficiency.** Was MEDIUM ("sufficiency is
  inferred"). Now measured: control `2.746e-2` FAIL → fixed `2.082e-17` PASS,
  full 23-binary device suite green under the fix.
- **BTMV needs no dedicated quantize kernel** (R-3 refuted): 0 mismatches over
  4,504,501 pairs at each of `prior ∈ {0, 0.5, 1}`.
- **Track E is a no-op for every repo artifact**: `calc_normalization(p) ==
  (0, 1)` for all `p ∈ [0, 1]`, and every prior in the repo is in `{0, 0.25,
  0.5, 1}`.
- **R-7 (f32-vs-f64 width) is inert today**: 0 mismatches at `prior ∈
  {0, 0.5, 1, 2}`; 899 only at `prior = -0.5`.
- **BTMV ≡ Borders at binclf in upstream** — read directly from both upstream
  functions.
- **C-3 (`one_hot_bins.is_empty()`) is a provably dead device clause** and
  SPEC-OH-26 is a scoping decision, not a measured correctness result — read from
  the rejection site and its rationale comment.
- **C-2's parity cap**: `structure_fold_cycle` is anchored only at
  `pc=4, seed=0` — read from the function and its doc comment.
- **C-1's padding infrastructure already exists** for float features — read from
  `pad_hist_line_bins` and its rationale comment naming `border_count=254`.
- **C-8 has no CPU parity surface** and **C-9 (wgpu) has no f64 channel** — read
  from the parameter surface and the kernel rejection sites.
- **The worktree is clean and the spike fully reverted** — `git status --short`,
  `git log --oneline -1`, and a post-revert behavioural re-run.

**Still MEDIUM:**
- **D-2's necessity.** It is required for CPU parity (`eligible_max_bucket_count`
  applies the same filter) but **no fixture detects it** — D-1 alone passed the
  combo oracle. Ship it; know it is untested (R-20).
- **The Track B (b) "honest accumulator" recommendation over (a) "alias".** Both
  are correct at binclf today; the choice is about future multiclass, which is
  not near.
- **C-6's compositionality argument** (sampling is a per-object multiplier
  orthogonal to which columns the histogram reads). Structurally sound from the
  code, but no CTR × bootstrap fixture exists yet.
- **C-5's claim that `n_bins_line = 32` is well-covered** — inferred from a code
  comment, not from a cat-only device run (none is possible today).

**LOW / UNVERIFIED — carry these into the plan as spikes, not assumptions:**
- **Upstream's non-symmetric CTR candidate-set semantics (C-7)** — the depthwise
  / lossguide entry points were **not read**. Blocks C-7 design.
- **Whether a phantom border can win under `SolarL2` / `LOOL2` / `SatL2`
  (C-1, R-16)** — the analytic argument was made for L2/Cosine only.
- **Whether a D-2-discriminating fixture is constructible** at the small `n` the
  frozen-fixture discipline prefers.
- **Q3's Counter kernel shape at large `n`** — not profiled; the recommendation
  rests on Counter columns being built once per fit, not per tree.
- **Phase staging (three vs one)** — a recommendation requiring user
  confirmation, not a fact.

---

## Confidence Assessment (first pass — see the delta above for supersessions)

**HIGH** (directly verified by reading current worktree source, fetched upstream
v1.2.10 source, or running commands this session):
- The exact text, line numbers and semantics of `ctr_types_are_device_covered`,
  `build_device_ctr_config`, `materialize_ctr_feature`, `online_class_prefix*`,
  `online_mean_prefix`, `online_counter_column`, `calc_ctr_online*`,
  `calc_normalization`, `SIMPLE_CLASSES_COUNT`, `ECtrType::*`.
- `prior_denom` is structurally always `1.0` in this codebase, and upstream
  forbids `denom != 1` on CPU (`ctr_helper.cpp:50`). Track 1a is a proven no-op.
- Only `Buckets` produces `target_border_idx > 0` at binclf.
- `counter_calc_method=Full` is moot on the device (`eval_sets.is_empty()`).
- Upstream `CalcCTR` is all-`float` with a hard `+1` denominator.
- Upstream `AddTreeCtrs` adds **no** combination candidate at level 0, and
  `CalcMaxFeatureValueCount` is per-level over the gated candidate list.
- The `ctr_device_combo` upstream model places the simple CTR at level 0 in all
  five trees and the combination only at level 1.
- The device's pass C has **no** eligibility gate and its `eligible_max` maxes
  over all columns.
- `TOnlineCtrUniqValuesCounts::Count` is the observed learn-set distinct count at
  default `ctr_leaf_count_limit` (refuting FPP-11 suspect 2).
- Upstream GPU does **not** support BTMV/Counter (`restrictions.h`).
- `device_ctr_combo_fit_test` currently passes on the **CPU** path with
  `--ignored` and has no device-commit assertion.
- A real gfx1151 ROCm GPU is present; the full device suite is green; the exact
  working command forms.
- `ctr_covered`'s `borders.len()+1 == n_bins` forces float `border_count == 15`.
- No device-reachable fixture exists for Buckets / BTMV / Counter.
- `cubecl 0.10.0`; `cb-backend` feature map; the CubeCL error-guideline path
  correction.

**MEDIUM** (well-supported by two independent code sources — upstream C++ and the
repo's own CPU transcription — but not yet exercised on device):
- ~~That D-1 + D-2 together are the complete FPP-11 fix; sufficiency is inferred~~
  — **NOW MEASURED AND HIGH** (spike Q5: control 2.746e-2 FAIL → fixed 2.082e-17
  PASS, suite green). The residual MEDIUM is only that **D-1 alone sufficed on
  this fixture**, so D-2 ships untested (R-20).
- The precise kernel deltas for Buckets (mode selector, no new buffer) and
  Counter (serial two-pass on unit 0) being adequate — the shapes follow directly
  from the existing serial kernel, but were not prototyped.
- That the existing border-table trick remains exactly correct for Buckets and
  Counter (they use the same f64 `calc_ctr_online_bin` path as Borders, so the
  proof carries over). **Partially upgraded**: spike Q2's "san f64 border table vs
  repo f64 bin" column measured **0 mismatches** at every prior in `[0, 2]`, which
  is a direct test of exactly this trick.
- ~~The R-7 f32-vs-f64 `CalcCTR` width divergence being currently invisible … is
  inferred~~ — **NOW MEASURED AND HIGH**: 0 mismatches at `prior ∈ {0, 0.5, 1, 2}`
  over 4.5M pairs each; 899 only at `prior = -0.5`, which no repo artifact uses.

**LOW** (requires a decision or a measurement before it can be treated as a
constraint):
- Whether the phase closes at five tracks or must also take the surviving nine
  CTR clauses (Q1) — a user decision, not a fact.
- ~~Whether BTMV needs a dedicated f32 quantize kernel (Q2)~~ — **MEASURED: it
  does not.** The spike was run; 0 mismatches over 4.5M pairs per reachable prior.
- ~~Whether the CPU's missing BTMV `(ctr+shift)/norm` should be fixed in this
  phase (Q4)~~ — **RULED: fix it (Track E)**, and measured to be a no-op for every
  committed artifact.
- CPU CTR oracle green-ness under default features was **not** re-run this
  session `[ASSUMED]`; only the rocm device lane and the backend CTR kernel lane
  were executed.
- `cargo check` without a GPU as a build-verification path was not re-exercised
  here (a real GPU is present); the claim is carried over from the sibling
  research report `[ASSUMED]`.
