---
plan: 7
task_id: TASK-07
phase: mvs-tree2-parity
status: pending
order: 8
wave: serial (v2 — runs AFTER TASK-08, PLAN-CHECK MINOR-3)
hardware: none (CPU only)
depends_on: [TASK-02, TASK-05, TASK-08]
blocks: [TASK-09]
specifications: [MVS-S7, MVS-S10]
parallelizable: false
parallel_with: []
plan_file_note: >
  This file stays `plan7.md` (bound to TASK-07) while its execution rank is 8; TASK-08 in
  `plan8.md` runs before it. See PLAN.md §1's "Plan-file numbering note" — renumbering
  just-reviewed artifacts by name would invalidate PLAN-CHECK.md's citations.
revision_note: >
  v2: sequenced AFTER TASK-08 (MINOR-3 — v1 claimed Wave E was "disjoint by construction",
  but this task writes `crates/cb-train/src/bootstrap.rs` while TASK-08 compiles that crate
  seven times under rocm in the same working tree; target-dir lock contention, though the
  correctness risk is nil since a doc-only edit cannot move numerics). Also: the grep gate
  is scoped to exclude this phase's own artifacts (MINOR-2, now also fixed in SPEC.md), and
  the citation set is pinned to SPEC.md §10.1 (MINOR-5).
---

# Task 7: Remove the superseded tree-2 diagnosis; record the remaining deviations

## Objective

Two documentation outcomes, both falsifiable:

1. **`MVS-S7`** — every surviving statement about this defect says: the cause was
   **two fabricated draws** in the MVS arm producing a per-tree phase drift; drift
   begins at **tree 1** (not tree 2); the observed first-divergent-split set is
   `{3, 4, 4, 4, 5, 5, 4}`; upstream takes exactly one draw, with the trace citation.
   Nothing **outside `.planning/plans/mvs-tree2-parity/`** still claims "never trees 0 or
   1", "the divergence enters when tree 2's sample is drawn from that λ", or "Raise this to
   3 once the MVS tree-2 sampling gap is fixed".

   **The scope exclusion is a SPEC correction, not a convenience relaxation** (MINOR-2).
   As originally written, `MVS-S7`'s postcondition ("anywhere in the repo") and `AC-7` were
   literally unsatisfiable: a measured **23 of the 26** repo-wide hits are inside this
   phase's own artifacts, where the phrasings appear only as explicitly-refuted quotations
   — 14 in `plan7.md`, 4 in `research.md`, 2 in `plan9.md`, 2 in `SPEC.md`, 1 in
   `progress.md`; only 1 in `crates/` and 2 in
   `.planning/plans/device-bootstrap-parity/progress.md`
   `[VERIFIED: RUN — grep -rn over crates/ and .planning/ at HEAD 2c14d7f]`. `SPEC.md` v2
   has been amended to match, so the spec and this executable gate now agree.
2. **`MVS-S10`** — the four remaining KNOWN deviations are recorded as deviations,
   with their reason and their observability, so no future reader mistakes the MVS
   port for a complete transcription.

**Observable completion condition:** the three-string grep in Verify returns nothing,
and the four deviations are present in `crates/cb-train/src/bootstrap.rs`'s doc
comments.

## Specification references

- `MVS-S7` — the superseded tree-2 diagnosis is removed from the tree. Principal
  failure reason: *documentation asserts a tree-2-specific mechanism and a λ-derived
  cause that the evidence refutes, misleading the next reader.*
- `MVS-S10` — remaining known deviations are documented, not silently carried.
  Principal failure reason: *a future reader assumes the MVS port is now a complete
  transcription and is surprised by an unmodelled difference.*

## Prerequisites and blocking

- Prerequisites: **TASK-02** (the 30-line superseded comment lives in the file
  TASK-02 rewrites — it must be deleted there first, or the grep in this task cannot
  pass) and **TASK-05** (so the docs describe the FINAL numerics rather than being
  written twice).
- Blocks TASK-09 (which re-runs the grep as a phase gate).
- **Sequenced AFTER TASK-08 (revised, MINOR-3).** v1 put this task in a parallel Wave E and
  claimed it and TASK-08 were "disjoint by construction". That claim was wrong in kind, not
  in correctness: this task writes `crates/cb-train/src/bootstrap.rs`, and TASK-08 runs
  seven `cargo test -p cb-train --no-default-features --features rocm --test …` invocations
  that COMPILE that crate — in the same working tree, alternating with `cpu`-feature
  builds. A mid-edit save forces a rebuild and the two cargo processes contend for the same
  target-dir lock. The *correctness* risk is nil (a doc-only edit cannot move numerics),
  which is why the remedy is simply ordering, at zero cost: TASK-08 writes nothing, so
  running it first is strictly free.

## Context and evidence

### The corrected finding (this is what the docs must say)

- The originating report claimed divergence begins at **tree 2, never trees 0 or 1**.
  That is **false as a general claim**: with `boost_from_average=true, seed=4` the
  first divergent split is flat index **3** — tree 1, split 1. The measured
  first-bad-split set across the seven failing configurations is
  `{3, 4, 4, 4, 5, 5, 4}` `[VERIFIED: research.md §3, §8.2]`.
- The mechanism is a **+2-draws-per-tree phase drift**: tree 0's `rand_seed` lands at
  call-count 2 on both sides, so tree 0 is always right; from tree 1 the sampler draws
  a different 80 % subset, and a wrong subset eventually flips a split argmax. Whether
  it flips at tree 1 or tree 2 is luck, which is exactly why the failure is seed- and
  bias-dependent `[VERIFIED: research.md §3]`.
- The λ carry was NOT the cause: `last_iter_mean_leaf_value` agrees with upstream to
  ~3e-9 and the learning-rate scaling of the stored leaf values is correct
  (`train.cpp:591` → `NormalizeLeafValues` → `approx_updater_helpers.cpp:33-37` runs
  BEFORE `LearnProgress->LeafValues.push_back` at `train.cpp:639`)
  `[VERIFIED: research.md §1.6]`.

### The superseded claims and where they live — all re-verified at HEAD

`[VERIFIED: grep -rniE "tree.2 (sampling|gap)|never trees 0 or 1|tree-2|MVS tree" --include="*.rs" --include="*.md" --include="*.py" crates/ .planning/plans/device-bootstrap-parity/]`

| location | claim | disposition |
|---|---|---|
| `crates/cb-train/tests/bootstrap_dev_oracle_test.rs:141` | "**tree 2**, never trees 0 or 1" | **deleted by TASK-02** (the whole `:121-155` block). Verify it is gone. |
| `crates/cb-train/tests/bootstrap_dev_oracle_test.rs:143-144` | "the divergence enters when tree 2's sample is drawn from that λ" | same |
| `crates/cb-train/tests/bootstrap_dev_oracle_test.rs:153-154` | "Raise this to 3 once the MVS tree-2 sampling gap is fixed" | same |
| `.planning/plans/device-bootstrap-parity/progress.md:47-79` | residual **R-1**, incl. `:67` "never trees 0 or 1" and `:69-70` the λ claim and `:73-76` the `MVS_GATED_TREES = 2` handling | **this task**: mark RESOLVED with the real root cause + a pointer to this phase |
| `crates/cb-train/src/bootstrap.rs:35-39` | the module MVS bullet — advertises `performRandomChoice = false` but makes **no draw claim**; not wrong, just incomplete | **this task**: add the one-draw contract + the trace citation |
| `crates/cb-train/src/bootstrap.rs:414-420` | the deleted block's own justification comment | **deleted by TASK-01**. Verify it is gone. |
| `.planning/plans/device-bootstrap-parity/SPEC.md:746` | "MVS gated to trees 0–1, see progress.md R-1" | **annotate** (planner finding `P2`) |
| `.planning/plans/device-bootstrap-parity/SPEC.md:793` | "mvs over trees 0–1 (R-1)" | **annotate** (planner finding `P2`) |
| `.planning/plans/device-bootstrap-parity/progress.md:137` | "`WR01-S15` carries the MVS carve-out described in R-1" | **annotate** |

`.planning/plans/device-bootstrap-parity/research.md:300` also matches the grep
("Required for MVS trees ≥ 1") but is a *correct* statement about the λ carry, not a
causal claim about tree 2 — **leave it alone**.

### The four `MVS-S10` deviations, with their evidence

| # | deviation | reason it is unmodelled | observability |
|---|---|---|---|
| (a) | `CalculateMeanGradValue`'s blocked reduction order. Upstream (`mvs.cpp:37-65`) uses `blockParams.SetBlockCount(CB_THREAD_LIMIT)` with `CB_THREAD_LIMIT = 128` (`restrictions.h:59`), giving `BlockSize = ceil(1500/128) = 12` and `BlockCount = ceil(1500/12) = 125` (`local_executor.h:81-90`), then `Accumulate`s the 125 partials. Ours (`bootstrap.rs:336-342`) is a flat ordered `sum_f64`. | The decomposition is independent of `thread_count`, so `thread_count=1` does NOT collapse it to a flat sum. Matching it would mean reimplementing an executor block split for a ~1 ulp effect. | Tree-0 λ **only** (`GetLambda` uses `leafValues.empty()`); **no current oracle can see it** `[VERIFIED: research.md §1.5]` |
| (b) | `mvs_reg` / upstream's `TMaybe<float> Lambda` override (`mvs.h:49`) is not modelled. | Correct while `mvs_reg` is unset — which it always is today. | Invisible until `mvs_reg` is exposed `[VERIFIED: research.md §1.6]` |
| (c) | Learn-weight multiplication into `SampleWeights`. `CalcWeightedData` does `ff.SampleWeights[i] *= learnWeights[i]` at **`tensor_search_helpers.cpp:482`** (guarded block `:479-484`; v1 said `:481-485`) before the control mask is derived; our port never multiplies. | Inert while all learn weights are `1.0` (every in-scope fixture). | A real gap for a weighted pool; **no weighted-MVS fixture exists** `[VERIFIED: research.md §2.7; line range re-read — MINOR-5]` |
| (d) | Ordered-boosting MVS tail-derivative handling (`mvs.cpp:136-168`). | Out of scope for this phase (SPEC §2). | Not exercised `[VERIFIED: research.md §2.7, §6.4]` |

`MVS-S10`'s scope names `mean_grad_value` and the module MVS bullet as the places to
record them: put (a) on `mean_grad_value` (`bootstrap.rs:333-335`) and (b)(c)(d) in
the module MVS bullet (`:35-39`).

**Use `SPEC.md` §10.1's verified citation set for every upstream line reference written
here** (MINOR-5). Notably: `catboost/private/libs/algo/fold.h:217` (there is no
`algo_helpers/fold.h`), the `const float*` read at `tensor_search_helpers.cpp:456`, the
learn-weight multiply at `:482`, `CalcWeightedData` = `:442-485`, and
`SetControlNoZeroWeighted` = `:1196-1204` with the mask at `:1202`. These land in
production doc comments, so a wrong path ships.

Also worth recording while in the same doc (both already verified, both genuinely
divergent-but-out-of-scope): our `single_probability` has a `threshold > 0.0` guard
upstream lacks — `mvs.cpp:17-19` would yield `inf`/`NaN` on an all-zero-gradient block
(SPEC `R6`, deliberately NOT harmonised) — and `calculate_threshold` sums the small
part in ORIGINAL order where upstream's `std::partition` permutes it (ulp-level)
`[VERIFIED: research.md §1.2, §1.3, §2.2]`. Adding these is optional; if added, mark
them clearly as the same class of documented deviation.

## Files

- Modify: `crates/cb-train/src/bootstrap.rs`
  - the module MVS bullet (`:35-39`) — add the one-draw contract with the trace
    citation, plus deviations (b), (c), (d);
  - `mean_grad_value`'s doc (`:333-335`) — deviation (a).
- Modify: `.planning/plans/device-bootstrap-parity/progress.md` — residual **R-1**
  (`:47-79`) marked RESOLVED; the `:137` sentence annotated.
- Modify (annotate only): `.planning/plans/device-bootstrap-parity/SPEC.md` — `:746`
  and `:793`.
- Do NOT touch: `.planning/plans/device-bootstrap-parity/research.md` (a historical
  research record; its `:300` line is correct anyway), any other sibling phase's
  artifacts, `crates/cb-train/tests/bootstrap_dev_oracle_test.rs` (TASK-02 owns it),
  or any code.

## TDD sequence

This is a documentation task, so the "test" is an executable grep gate rather than a
Rust test. Keep it Red/Green/Refactor/Verify all the same.

### 1. Red

**The naive line-based greps CANNOT work (C2-9).** All three claims WRAP across lines in
`crates/cb-train/tests/bootstrap_dev_oracle_test.rs` — claim 1 at `:141`, claim 2 across
`:143-144`, claim 3 across `:153-154` — and in a `.rs` file the wrap inserts a `///`
comment leader mid-phrase. Measured at HEAD `[VERIFIED: RUN]`: `grep -rn "never trees 0 or 1" crates/`
→ **1 file**, but `grep -rn "divergence enters when tree 2" crates/` → **0** and
`grep -rn "MVS tree-2 sampling gap" crates/` → **0**. So v2's "if the third still hits,
TASK-02 is incomplete" could never fire.

Use the normalising gate — it strips comment leaders (`///`, `//!`, `//`, `*`, `>`) and
squashes all whitespace, so a wrapped phrase still matches:

```bash
# MVS-S7 / AC-7 gate. Hits are permitted ONLY under .planning/plans/mvs-tree2-parity/.
ROOTS=(crates .planning)
PHRASES=(
  "never trees 0 or 1"
  "the divergence enters when tree 2's sample is drawn from that"
  "Raise this to 3 once the MVS tree-2 sampling gap is fixed"
)
hits=0
for p in "${PHRASES[@]}"; do
  while IFS= read -r f; do
    [[ "$f" =~ ^\.planning/plans/mvs-tree2-parity/ ]] && continue
    if sed -E 's,^[[:space:]]*(///|//!|//|\*|>)+[[:space:]]?,,' "$f" \
       | tr -s '[:space:]' ' ' | grep -qF "$p"; then
      echo "HIT [$p] -> $f"; hits=$((hits+1))
    fi
  done < <(grep -rIl "" "${ROOTS[@]}" 2>/dev/null)
done
echo "TOTAL_HITS=$hits"; exit $(( hits > 0 ))
```

- **Expected BEFORE this task, MEASURED at HEAD** `[VERIFIED: RUN — this exact script,
  `TOTAL_HITS=5`, exit 1]`:

  ```
  HIT [never trees 0 or 1] -> crates/cb-train/tests/bootstrap_dev_oracle_test.rs
  HIT [never trees 0 or 1] -> .planning/plans/device-bootstrap-parity/progress.md
  HIT [the divergence enters when tree 2's sample is drawn from that] -> crates/cb-train/tests/bootstrap_dev_oracle_test.rs
  HIT [the divergence enters when tree 2's sample is drawn from that] -> .planning/plans/device-bootstrap-parity/progress.md
  HIT [Raise this to 3 once the MVS tree-2 sampling gap is fixed] -> crates/cb-train/tests/bootstrap_dev_oracle_test.rs
  ```

  **All three phrases are now detective**, including claim 3 which the naive form missed
  entirely. Note claim 3 exists ONLY in the test file — it has no counterpart in the
  sibling `progress.md`.
- **After TASK-02** (which deletes `bootstrap_dev_oracle_test.rs:121-155`) the three
  `crates/` hits must be gone, leaving the two `device-bootstrap-parity/progress.md` hits
  for this task to resolve. **If any `crates/` hit survives, TASK-02 is incomplete** — go
  back and finish it rather than editing the test file here. That check is now real, where
  v2's was vacuous.
- Cross-check with `grep -rn "MVS_GATED_TREES\|MVS_SCENARIO\|gated_trees" crates/` → no
  output; that grep is line-safe and remains the independent detector.
- Also run
  `grep -n "performRandomChoice" crates/cb-train/src/bootstrap.rs` and confirm the
  only surviving occurrence is the module bullet at `:35` and the control-mask comment
  near `:424` (the deleted block's `:414` comment must be gone from TASK-01).
- Record all three grep outputs.

### 2. Green

**2a. `crates/cb-train/src/bootstrap.rs` module MVS bullet (`:35-39`).** Extend it so
it states, in substance:

> MVS consumes **exactly one** main-stream draw per `Bootstrap()` call
> (`mvs.cpp:174` `randSeed = rand->GenRand()`); `performRandomChoice = false` sends
> `TCalcScoreFold::Sample` down the draw-free `SetControlNoZeroWeighted` branch
> (`calc_score_cache.cpp:742-748`) and `CalcWeightedData`
> (`tensor_search_helpers.cpp:442-485`) is draw-free. Verified against the
> instrumented 1.2.10 trace
> (`.planning/plans/bayesian-rng-draw-accounting/instrumented-ground-truth/mvs.jsonl`):
> `tree_rng_pre_gts.cc = 2` → level-0 `cc_start = 7` = 1 bootstrap draw + 4 RSM draws.
> A prior version of this port fabricated two extra draws here, producing a
> +2-draws-per-tree phase drift whose first divergent split was observed at flat
> indices `{3, 4, 4, 4, 5, 5, 4}` — i.e. drift begins at **tree 1**, and any claim of
> a tree-2-specific mechanism is superseded.

Then add, in the same bullet or immediately after it, the KNOWN DEVIATIONS (b), (c),
(d) from the table above, each with its reason AND its observability.

**2b. `mean_grad_value`'s doc (`:333-335`).** Add deviation (a): upstream's
`CB_THREAD_LIMIT = 128`-blocked reduction (125 partials at `n = 1500`), the fact that
it is independent of `thread_count`, that ours is a flat ordered `sum_f64` (D-05 /
`cb_core::sum_f64` is the only sanctioned reduction), that the difference is ~1 ulp,
and that it is observable **only** through tree-0 λ where no oracle currently sees it.

> **HARD CONSTRAINT (C2-6) — this is the single most likely place in the phase to trip
> its own gate.** Deviation (a) is *about summation order*, so the natural sentence is
> "ours is a flat ordered `sum_f64`, never a naive `.sum()`". **Do not write that.**
> `scripts/check-no-raw-float-sum.sh` applies
> `SUM_PATTERN = \.sum\(\)|\.fold\(0\.0|\.fold\(0_f|\.fold\(0f` with `grep -RIlE` to every
> non-`*_test.rs` source file — **comments included** — which is why 12 of `B8`'s 15
> baseline violations are doc comments *describing* the ban, e.g.
> `crates/cb-train/src/boosting.rs:1649`. `crates/cb-train/src/bootstrap.rs` is currently
> **clean** of the pattern `[VERIFIED: RUN — grep -nE '\.sum\(\)|\.fold\(0\.0|\.fold\(0_f|\.fold\(0f'
> crates/cb-train/src/bootstrap.rs → exit 1]`, and that is the ONLY reason the phase's
> diff-scoped `B8` gate is empty. Writing the literal `.sum()` here would newly name
> `bootstrap.rs` in the D-08 output and turn this phase's own gate red — forcing either a
> weakened `MVS-S10` doc or an accepted red gate.
> **Permitted phrasings:** `sum_f64`, "a raw iterator summation", "a naive iterator sum",
> "an uncompensated left-to-right fold". **Forbidden literals:** `.sum()`, `.fold(0.0`,
> `.fold(0_f`, `.fold(0f`.
> **Verify immediately after the edit:**
> `bash scripts/check-no-raw-float-sum.sh 2>&1 | grep -E "src/bootstrap\.rs"` → must be
> EMPTY, and the script's header count must still be **15**.

**2c. `.planning/plans/device-bootstrap-parity/progress.md` R-1.** Rewrite the
heading to something like
`### R-1 — MVS diverged from upstream (pre-existing, CPU-side) — RESOLVED 2026-07-31`
and add a resolution block that: names the real root cause (two fabricated draws at
the old `bootstrap.rs:413-423`); corrects "tree 2, never trees 0 or 1" to "drift
begins at tree 1; first-bad-split set `{3,4,4,4,5,5,4}`"; states the before/after
(3/5 + 0/5 → 5/5 + 5/5); and points to `.planning/plans/mvs-tree2-parity/`
(`SPEC.md` `MVS-S1`, `PLAN.md`, `plan1.md`).
**Preserve the original measurement table** (the 3/5 and 0/5 rows) as historical
evidence — mark the WRONG causal sentences as superseded rather than deleting the
measurements. Do not restate the refuted claims as if current; if a quoted form is
kept for history, prefix it unmistakably (e.g. `SUPERSEDED (was): …`) so the
three-string grep in Verify still passes — i.e. the banned phrasings must not survive
verbatim.
Also annotate `:137` ("`WR01-S15` carries the MVS carve-out described in R-1") with
"— carve-out removed by `mvs-tree2-parity`; MVS is now gated 3/3".

**2d. `.planning/plans/device-bootstrap-parity/SPEC.md` — annotate only.** Append to
`:746` and `:793` a short clause such as
"(superseded 2026-07-31: RESOLVED by `mvs-tree2-parity`; MVS now matches upstream over
all 3 trees)". **Do not rewrite or delete the completed phase's measured evidence** —
that phase's specifications are approved and its measurements are the record.

### 3. Refactor

- Read the two `bootstrap.rs` doc blocks end to end and check they do not now
  contradict each other or repeat the trace citation three times. One canonical
  statement of the one-draw contract lives at the draw site (TASK-01); the module
  bullet should summarise and cite, not duplicate.
- Keep every deviation entry in the same shape: *what* / *why unmodelled* /
  *observability*. A deviation without an observability claim is not `MVS-S10`
  compliant.
- Run: `cargo doc -p cb-train --no-deps` → the doc comments must build (intra-doc
  links to `[`calculate_threshold`]` etc. already exist in this file; a broken link is
  a warning worth fixing).
- **Clippy — ERROR-attributed and diff-scoped** (CRITICAL-1 / C2-1). A `-->` grep would be
  red at HEAD on `bootstrap.rs:134`'s pre-existing `excessive_precision` **warning**:

  ```bash
  clippy_error_files -p cb-train --all-targets | grep "src/bootstrap\.rs"   # must be EMPTY
  ```

  A doc-only change can still introduce `broken_intra_doc_links` (which IS an error under
  the workspace config), so this is a real check, not a formality.
- **Re-run the D-08 diff-scoped grep** (C2-6) — this task is the one most likely to trip it:
  `bash scripts/check-no-raw-float-sum.sh 2>&1 | grep -E "src/bootstrap\.rs"` → EMPTY, and
  the script's `D-08 violation:` header count must still be **15**.

### 4. Verify

- Re-run the **normalising gate script from step 1** → `TOTAL_HITS=0`, exit 0. Do NOT
  substitute the naive line-based greps: two of the three phrases cannot match a wrapped
  claim (C2-9, measured).
- Then run the script WITHOUT the `mvs-tree2-parity` exclusion and confirm by READING that
  every remaining hit sits in an explicit "this claim is false / superseded" context inside
  this phase's own artifacts; record that judgement. Hit counts inside this phase drift as
  the artifacts themselves quote the phrases (v2 recorded 23 of 26; the checker measured 25
  of 28) — the count is **not** a gate, only the exclusion-scoped `TOTAL_HITS=0` is.
- Run: `grep -rn "tree-2\|tree 2" crates/` → any surviving hit must not be an MVS
  causal claim. Read each and record the verdict.
- Run: `grep -n "mvs.jsonl\|cc_start = 7\|randSeed = rand->GenRand" crates/cb-train/src/bootstrap.rs`
  → the trace citation is present.
- Run: `grep -n "CB_THREAD_LIMIT\|125" crates/cb-train/src/bootstrap.rs` → deviation
  (a) is recorded on `mean_grad_value`.
- Run: `grep -n "mvs_reg\|TMaybe<float>\|learn weight\|learnWeights\|Ordered" crates/cb-train/src/bootstrap.rs`
  → deviations (b), (c), (d) are recorded.
- Run: `cargo test -p cb-train --lib bootstrap` → still green (doc-only change).
- Run: `cargo test -p cb-train --test bootstrap_dev_oracle_test` → still green.
- Run: `git diff --stat` → only `crates/cb-train/src/bootstrap.rs` and the two
  `.planning/plans/device-bootstrap-parity/` Markdown files.
- Run: `git diff .planning/plans/device-bootstrap-parity/SPEC.md` → **additions only**,
  no deletion of any measured evidence.

## Completion criteria

- [ ] The pre-edit **normalising gate** output was captured (the Red): `TOTAL_HITS=5` at
      HEAD, with all three phrases hitting — **not** the naive line greps, two of which
      cannot match a wrapped claim (C2-9).
- [ ] The gate now reports `TOTAL_HITS=0` (exit 0): none of the three banned phrasings
      survives outside `.planning/plans/mvs-tree2-parity/` (and inside it only as an
      explicitly refuted quotation).
- [ ] **No `.sum()` / `.fold(0.0` literal was written into `bootstrap.rs`** (C2-6), and the
      D-08 diff-scoped grep for `src/bootstrap.rs` is still EMPTY with the header count
      still 15.
- [ ] `bootstrap.rs`'s module MVS bullet states the one-draw contract with the
      `mvs.cpp:174` / `calc_score_cache.cpp:742-748` /
      **`tensor_search_helpers.cpp:442-485`** / `mvs.jsonl` citations AND the corrected
      "drift begins at tree 1, first-bad-split set `{3,4,4,4,5,5,4}`" finding. Every
      upstream line reference matches `SPEC.md` §10.1 exactly (MINOR-5).
- [ ] All four `MVS-S10` deviations appear with reason AND observability —
      (a) on `mean_grad_value`, (b)(c)(d) in the module bullet.
- [ ] `progress.md` R-1 is RESOLVED, with the real root cause, the corrected onset,
      the before/after table preserved, and a pointer to this phase.
- [ ] The two `device-bootstrap-parity/SPEC.md` cross-references are ANNOTATED, with
      no measured evidence deleted.
- [ ] `cargo doc -p cb-train --no-deps` builds; the diff-scoped clippy grep for
      `src/bootstrap.rs` is EMPTY (**not** "clippy clean" — `PLAN.md` §4.11 `B3`/`B4`); the
      two MVS test targets still green.
- [ ] This task ran AFTER TASK-08 (MINOR-3), so no rocm compile of `cb-train` was in
      flight while `bootstrap.rs` was being edited.

## Completion evidence to record in `progress.md` (this phase's)

- The pre- and post-edit grep transcripts for all three banned strings.
- The list of surviving `tree 2` hits with the verdict for each.
- Confirmation that the sibling `SPEC.md` diff is additions-only.

## Risks and guardrails

- **Overwriting a completed phase's approved record.** `device-bootstrap-parity` is
  `status: implemented` with measured evidence per specification. This task
  **annotates**; it never rewrites a measurement or flips an
  `implementation_state`. If a change feels like it needs to rewrite that phase's
  claims, stop and report instead.
- **Re-tuning instead of removing.** `MVS-S7` explicitly requires the superseded
  diagnosis to be *rewritten or removed*, not merely re-tuned (e.g. bumping
  `MVS_GATED_TREES` from 2 to 3 and leaving the comment). TASK-02 does the removal;
  this task must verify it happened.
- **Documenting a deviation as parity.** `MVS-S10`'s postcondition is that none of the
  four is *presented as parity*. Each entry needs an explicit "unmodelled / not
  matched / out of scope" word.
- **Scope creep into implementing a deviation.** `MVS-S10`'s non-goal is explicit:
  implement none of them. In particular do not attempt to reproduce
  `CalculateMeanGradValue`'s 125-block split, and do not "harmonise" the
  `threshold > 0.0` guard to upstream's UB (SPEC `R6`).
- **A grep that passes for the wrong reason.** If a banned phrase was merely reworded
  while the false claim survives in substance, the gate is defeated. The Verify step's
  manual read of every surviving `tree 2` hit is the backstop.
