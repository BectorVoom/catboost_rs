# Coordinator findings — carried forward between tasks

Findings raised by a completed task that a LATER task must act on. Each executor is
handed this file. Append; never rewrite another task's entry.

## From T03 → **T04**

`calc_normalization(0.0)` returns shift `-0.0`, not `0.0` (`shift = -min(0.0, p)`).
⇒ T04's assertions must compare with `==` (PartialEq), **never** `f64::to_bits` — a bit
comparison fails on *unmutated* code.

T03 also captured the deterministic `--test-threads=1` output of
`cargo test -p cb-train --test ctr_btmv_simple_oracle_test` in `notes/T03.md` as the
**pre-T04 byte-comparison baseline**. T04 must diff against it.

## From T07 → **T05** (already resolved) and **T10/T16**

`model.json`'s `ctrs` array is the *pruned/used* set, so **descriptor count alone does not
discriminate CTR type** — `Buckets:Prior=0.5` on the same data also yields a single
descriptor at `target_border_idx=0`. Assert on `ctr_type` explicitly.

## From T05 → **T10**

`ctr_device_buckets` landed at **escalation rung 0** (`CARDS=(6,)`, 64 rows, 5 iterations,
`data_seed=1`), not the predicted rung 3. `X_cat.npy` is `(64,) i32` (one cat column) —
the shape T10's consumer contract expects. `model.json` carries **2 Buckets descriptors
with `target_border_idx` `{0, 1}`**, which is exactly the two-columns-per-`(projection,
prior)` layout PLAN §6 assumption 8 flags as T10's most likely failure site.

## From T06 → **T12** (raised explicitly by T06; act on it)

On `ctr_device_counter`'s data the Counter prior is **prediction-neutral upstream**:
`Prior=0.5`, the default `0/1`, and `Prior=3` all give **bit-identical** predictions
(upstream compensates by shifting the CTR border — 8.999999 at prior 0.5/0, 3.999999 at
prior 3). ⇒ **the ≤1e-5 e2e cannot police a Rust-side prior mismatch on this fixture.**
The only guard is the explicit pin on both sides: `Counter:Prior=0.5` in the fixture
(asserted by the smoke test) and `simple_ctr_priors = vec![0.5]` in T12's `BoostParams`.
T12 must set that explicitly and say so in its note. See `notes/T06.md` §6.

## From T00 → **T10** (and any task editing the gate-state table)

T00 declined the optional rows 9/10 (`simple/Buckets/b=0`, `simple/Borders/b=1`) to avoid
renumbering rows later tasks are specified against. Cost, documented in-source: **row 4
(`simple/Buckets/b=1`) is the one row with no single-conjunct mutation proof** — it varies
both `ctr_type` and `target_border_idx` vs row 1. T10, which flips row 4, should state in
its note which of the two conjuncts its edit actually removed and how it knows.

T00 also recorded, verbatim, the mutation output that **is T01's predicted Red**:

```
row 6 (simple/Borders/b=0/denom=2.0): expected false, got true
```

## From T02 → **T17 and T19** (MATERIAL — correction to PLAN §6 C-12)

C-12's *form* column is wrong. The **site set** is right (the grep returns the same 9
lines, so C-12's STOP condition correctly did not trigger), but **all 7 test-side
`DeviceCtrColumn` literals end in `..DeviceCtrColumn::default()`** — not just the 3 C-12
lists. Only the production literal (`boosting.rs:2519`, now `:2559`) is a full literal.
⇒ T02 correctly edited **1** literal, not 5, and both `cargo check`s pass.

**Consequence T17/T19 must act on:** the `projection_members: vec![]` re-review set is
**8 sites, not 3**. The dangerous one is
`crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs:630`, a genuine **2-member
combination** (`member_bins: vec![cat0, cat1]`) that now defaults to an **empty** member
list — once T17's eligibility gate reads `projection_members`, that fixture will look
*simple*. Production is safe: T02 added a `!projection_members.is_empty()` →
`CbError::Degenerate` guardrail making an empty list unreachable from
`build_device_ctr_config`.

## From T02 → **T24** (pre-existing failure to record, do not chase)

`cargo test --workspace` fails **one** target: `cb-backend --lib` under **default (cpu)**
features, 211 passed / 60 failed. Cause is `plane_inclusive_sum … is not supported on CPU`
in `cubecl-cpu-0.10.0`. T02 proved pre-existence by restoring `runtime.rs` from
`git show HEAD:` read-only and re-running `kernels::sort::` to the identical failures.
Unrelated to this phase. The rocm path (`bash ./run_device_tests.sh`) was **23/23 PASS +
perf lane** at the end of Wave 0, Poisson at 10.5× (no R-13 flake).

DCTR-01 byte identity held: `device_ctr_fit_test` `max |Δpred| = 4.483e-11` before **and**
after T02. That is the number later device tasks should expect to stay put unless they
intentionally move it.

## From T01 → **T10, T12, T16, T19, T22, T23** (the gate covering-test home)

T01 created `crates/cb-train/src/boosting_ctr_gate_test.rs`, mounted at the end of
`boosting.rs`. Its module doc records the **extension contract** every later gate task
follows: **two tests per conjunct** — a structural pin, plus a `gate_body()`-based
"the gate no longer reads X" pin — using the reusable `production_source` /
`code_lines_mentioning` / `gate_body` helpers already in the file. Extend it; do not
start a new file.

**Fragility to know about**: those helpers work by `include_str!("boosting.rs")` and
scanning the source text of `ctr_types_are_device_covered`. They are fail-loud on rename.
**T23 rewrites the gate entirely** (delegating to `ECtrType::from_i8` / `is_cpu_supported`)
— T23 must therefore update or retire these source-scan pins as part of its own change,
not leave them asserting against a function shape that no longer exists.

## From T04 → **T14, T15, T16** (Track B — the device BTMV mirror target)

The CPU BTMV quantizer is now (`ctr_feature.rs`, the `quantize_in_f32` arm):

```rust
let (shift, norm) = crate::ctr::calc_normalization(prior_scalar);   // norm, NOT the denominator
let denom = (total as f32) + 1.0f32;                                // the CTR denominator
let ctr   = (good as f32 + prior_scalar as f32) / denom;
f64::from(((ctr + shift as f32) / norm as f32) * ctr_border_count as f32)
```

Track B's "device == CPU" claims are now provable against the **corrected** CPU. Two things
to carry:

1. **`norm` and `denom` are different quantities and they COINCIDE at `count == 1`.** The
   plan's two-document T04 Red construction reaches `(good = 1.0, total = 1)` as predicted
   — but at that state `denom == 2.0 == norm`, so it cannot distinguish "divide by
   `calc_normalization`'s norm" from "divide by the CTR denominator". T04 widened its
   column to **three** documents (`(sum = 1.0, count = 2)` ⇒ correct bin 7, uncorrected 15,
   conflated 5) and only then did the mutation fail. **Any device BTMV self-oracle that
   drives ≤ 2 documents per bucket inherits the same blind spot.** Drive ≥ 3 documents in at
   least one bucket.
2. **C-17 was applied as "omit/mark dead", not as a live guard.** The device side must not
   introduce a `norm == 0.0` branch either — `norm = max(1, p) - min(0, p) >= 1` always.

DCTR-05 held exactly: `ctr_btmv_simple_oracle_test`'s printed output is byte-identical to
T03's captured baseline across DCTR-04, and **no fixture changed**. `cb-train` as a whole is
**106 targets, all ok** after the change.

## From T04 → **T24** (pre-existing clippy scope is WIDER than recorded)

The "Pre-existing, out of scope" bullet below names only `device_seam_test.rs`. Measured in
T04 with `cargo clippy -p cb-train --lib --tests --keep-going`: **12** `cb-train`
integration-test targets fail the lint gate, not one — `one_hot_draw_accounting_test`,
`learn_set_shuffle_oracle_test`, `yetirank_pairwise_tree_rng_oracle_test`,
`tensor_ctr_oracle_test`, `device_fold_count_gate_test`, `permutation_oracle_test`,
`structure_fold_cycle_oracle_test`, `plain_ctr_oracle_test`, `ordered_ctr_oracle_test`,
`s_order_ctr_bins_oracle_test`, `device_seam_test`, `ordered_boost_oracle_test`. Same class
(`panic`/`expect_used`/`indexing_slicing` in committed `tests/*.rs`), same cause,
`cargo test` unaffected. The **`cb-train` lib / lib-test target itself is clippy-clean**, so
a task whose work lands in `src/` can still verify itself. Without `--keep-going` clippy
stops at whichever target compiles first, which is why different tasks report different
"the" failing file.

## Environment / process facts for every executor

- Settings live at `planning/settings.json` (**no leading dot**); `implementation.use_worktree`
  is `false`. Plans live at `.planning/plans/` (**with** the dot). PLAN.md's frontmatter
  names a `gpu-borders-shared-sample` worktree that **no longer exists** — ignore it. All
  work happens in the primary checkout on branch `feat/device-ctr-full-coverage`.
- ROCm `gfx1151` is present and device tests really run.
- **Pre-existing, out of scope**: `cargo clippy -p cb-train --lib --tests` fails to compile
  the committed `crates/cb-train/tests/device_seam_test.rs` (23 `indexing_slicing` /
  `panic` / `expect_used` errors, from commit `de81a16`). `cargo test` is unaffected. Do
  not fix it inside another task; T24 records it.
- `crates/cb-oracle` has ~5 pre-existing clippy errors unrelated to this phase.
- Wave-parallel artifact: fixture generators carry a corpus-contamination check that
  reports sibling `ctr_device_*` directories created by concurrent tasks. Harmless.

## From T20 → **T10, T12, T16, T19** (every remaining `CountingGpu` e2e) and **T24**

**1. There is now a measured CPU-fallback fingerprint, and it is cheap to read.** On
`ctr_device_mixed`, forcing the gate closed (`&& false`) flips the printed
`max |Δpred|` from **`4.483e-11` (device)** to **`1.388e-17` (CPU)** — the exact
CPU-fallback value `SPEC.md` DCTR-17 predicts — and collapses the run from **1.9s to
0.01s**. ⇒ *An unexpectedly tiny `max |Δpred|` or an instant device e2e is a
CPU-fallback smell.* Note the direction: **the CPU path scores BETTER against the
upstream oracle than the device path.** No ≤1e-5 bar can ever detect a fallback,
because falling back makes the number improve. Do not read a very small delta as
reassurance.

**2. Put the `CountingGpu` assertion AFTER the ≤1e-5 loop, not right after the fit.**
T20 did, deliberately. Under the §2.5 mutation the run then prints the passing
`max |Δpred|` line *before* panicking on `grown`, so a **single** mutation run yields
both halves of the required completion evidence ("the assertion failed AND the ≤1e-5
bar still passed"). Fail-fast ordering hides exactly the evidence DCTR-08/10/14/17
completion criteria ask for. In-source comments in `device_ctr_fit_test.rs` explain
this so a later reader does not "fix" the ordering.

**3. `&& false` on `ctr_types_are_device_covered` is the universal, cheap mutation for
any device-commitment assertion.** Rebuild + run is ~2.3s + ~2s; it needs no fixture and
no per-task bespoke mutation. Predicted failure shape is always
``assertion `left == right` failed … left: 0, right: <iterations>``. It edits the gate
expression, so the isolation rule applies — focused `--test <name>` only while live.

**4. The `CountingGpu` wrapper now exists in TWO copies** —
`device_ctr_gate_test.rs:82-138` (canonical) and `device_ctr_fit_test.rs` (copied
verbatim per GLOBALS §2.2.6, with a keep-in-sync note). Each new e2e adds a third,
fourth, … copy. Consequence: **any change to the `cb_compute::Runtime` method
signatures this wrapper overrides (`compute_gradients`, `begin_device_training`,
`grow_tree_on_device`, `end_device_training`) now breaks N test files, not one.** No
task in this phase is scheduled to change them, but T24 should record the duplication
count at DoD.

**5. Parallel-wave build hazard (process).** Any `cargo {check,test,clippy} -p cb-train`
transitively compiles `cb-backend`. During Wave 2, T08's in-flight edit left
`crates/cb-backend/src/kernels/ctr_device.rs` non-compiling (`E0061` at `:294`, kernel
signature mid-change) and that surfaced as a **red build in T20's commands** with
nothing wrong in `cb-train`. ⇒ **Before chasing a `cb-backend` compile error, run
`git diff --name-only` and check whether the broken file is yours.** If it is another
task's, record and move on; do not "fix" it.

## From T08 → **T09, T11, T12, T14, T16** (the ordered-prefix kernel seam after DCTR-06)

T08 landed the `(ctr_type, target_border_idx)` numerator selector on
`ordered_ctr_prefix_kernel` (`ctr_device.rs`, `fn` now at `:164`) and on
`launch_ordered_ctr_resident` (`:307`). Four facts later kernel/plumbing tasks must act on:

**1. `launch_ordered_ctr_resident` now REJECTS any `ctr_type` outside `{0 Borders, 1 Buckets}`
with `CbError::OutOfRange`.** This is deliberate and beyond the letter of T08's task text: the
alternative (accept `Counter` / `BTMV` and fall through to the Borders numerator) is a silent
wrong answer, not a worse one. ⇒ **T11 (Counter) and T14 (BTMV) must not route their columns
through this function.** If a later task genuinely needs to, widening the guard is a deliberate,
test-visible act — `ctr_device_test::out_of_range_ctr_mode_is_rejected` pins it and will go red.
`target_border_idx > 1` is rejected the same way (`SIMPLE_CLASSES_COUNT == 2`).

**2. Reuse `CTR_TYPE_BORDERS` / `CTR_TYPE_BUCKETS`** (`pub(crate)` at `ctr_device.rs:277`/`:279`)
rather than re-transcribing the `ECtrType` discriminants. C-3 demands a citation on every
cb-backend transcription; these already carry the full list plus the `restrictions.h` reference.

**3. T09's production edit is two literals, at `session.rs:230-231`** — replacing the
`CTR_TYPE_BORDERS, 0` that T08 pinned there (with an in-source comment naming T09) by
`col.ctr_type, col.target_border_idx`. **No conversion is needed**: `DeviceCtrColumn.ctr_type` is
already `i8` and `.target_border_idx` already `u32` (T02's seam). Combined with finding 1 and
C-14, a cb-train-gate / cb-backend-list mismatch surfaces as a loud `CbError` → `grown == 0`,
never a silent numerator swap.

**4. The `#[cube]` parameter order is now
`perm, bins, class, prior, mode, counts, good, total, value`** — `mode` is the **5th** positional
argument (`Array<u32>`, length 2, `[is_buckets, target_border_idx]`). There is still **exactly
one** `ordered_ctr_prefix_kernel::launch` site (`ctr_device.rs:391`).

**5. Test-side: `cpu_ordered_ctr` (`ctr_device_test.rs:69`) gained
`is_buckets: bool, target_border_idx: usize`**; all 5 pre-existing call sites pass `(false, 0)`,
which reduces to the previous `good = N1` exactly. It is deliberately kept in
`online_class_prefix`'s **generic loop-over-classes** form while the kernel implements the
2-class collapse — that asymmetry is what makes the oracle a cross-check rather than a copy.
`compute_ordered_ctr_host` kept its old signature and delegates to the new
`compute_ordered_ctr_host_mode` (`:496`), so the 4 pre-existing device oracle call sites are
source-unchanged.

**6. Kernel-style note for T11/T14 (cost-free, avoids a warning):** writing the CubeCL
`let mut x = <default>;` + `if`/`else` pattern with an explicit assignment in *every* arm makes
the initializer dead and emits `warning: value assigned to 'x' is never read`. Leaving the
"unreachable/pinned" case to the initializer (and documenting it in a comment) keeps the manual's
pattern and the build warning-free. T08 did this for `Borders@1`.

**D-04 held at T08:** `device_ctr_fit_test` `max |Δpred| = 4.483e-11`, unchanged; 23/23 device
PASS; `cb-backend --lib` under rocm `263 passed / 0 failed`.

**Pre-existing clippy failure to add to T24's list (cb-BACKEND, not cb-train):**
`cargo clippy -p cb-backend --no-default-features --features rocm --lib --tests` fails with
`error: this operation will always return zero` (`clippy::erasing_op`) at
`crates/cb-backend/src/kernels/score_split.rs:374` — `cindex[0 * n + obj] = bin;`. Verified
pre-existing at `HEAD` via `git show HEAD:…/score_split.rs | sed -n '374p'`. `cargo test` and
`cargo check` are unaffected. Do not fix inside another task.

## From T09 → **T10, T12, T16** (the backend admission list is now a live red-maker)

T09 added the per-type admission conjunct
`matches!(col.ctr_type, CTR_TYPE_BORDERS | CTR_TYPE_BUCKETS)` to **both** `.all(..)` closures
of `ctr_covered` (`session.rs`, structure half `:180`, averaging half `:192`) and turned
`build_ctr_cindex_columns`' launch into a `match col.ctr_type` (`:262-282`) whose `_` arm is
`CbError::Unsupported("device CTR type {other} is not implemented")`. Four things to act on:

**1. There is now a covering test that WILL go red for T12 and T16, deliberately.**
`crates/cb-backend/src/gpu_runtime/session_ctr_type_test.rs::ctr_covered_declines_unimplemented_ctr_types`
asserts today that discriminants `{2 BTMV, 3 FloatTMV, 4 Counter, 5 FeatureFreq}` **decline**
and that `{-1, 6, i8::MIN, i8::MAX}` decline too. When T12 admits Counter (or T16 admits BTMV)
it must, in the SAME hunk, (a) add the discriminant to both `ctr_covered` closures, (b) add its
arm to the `build_ctr_cindex_columns` match, and (c) move that discriminant from the test's
decline loop to its admit loop. **That red is the expected Red of those tasks, not a
regression.** The out-of-enum strays and `{3, 5}` must stay in the decline loop forever
(GPU-only upstream, `restrictions.h:20-32`).

**2. The `cb-train` gate and this backend list are still allowed to disagree transiently.**
C-14 is now written into `ctr_covered`'s doc comment in-source: the predicate's first caller
feeds the coverage disjunction whose failure path declines the **whole fit** (`Ok(None)`), so a
gate-admits / backend-declines mismatch surfaces as `grown == 0`, never a silent CTR drop.
T10 may therefore open the cb-train gate for Buckets without touching `session.rs` at all —
the backend already admits it.

**3. `launch_ordered_ctr_resident` is now reached with REAL `(col.ctr_type,
col.target_border_idx)`**, not T08's pinned `(CTR_TYPE_BORDERS, 0)`. Combined with T08's host
guard (which rejects any `ctr_type ∉ {0,1}` and any `target_border_idx > 1` with
`CbError::OutOfRange`), an unimplemented type now has **two** loud stops before it can reach a
wrong numerator. T11/T14 still must not route Counter/BTMV through this function.

**4. Pre-existing doc bug in the region T12/T16/T17 all edit.** The block at
`session.rs:200-215` beginning `/// Compute the ADDITIONAL binarized-CTR cindex columns …`
(including its `# Errors` section) documents `build_ctr_cindex_columns`, but is physically
attached to `struct CtrSearchState`, which was inserted between them at some point. T09 did
not move it (out of scope; three later tasks edit those lines) and instead put the new
`CbError::Unsupported` `# Errors` note on `build_ctr_cindex_columns`' own GDC-09 doc comment.
Whoever next restructures this region should re-attach the orphan.

**D-04 held at T09:** `device_ctr_fit_test` `max |Δpred| = 4.483e-11`, unchanged; `device
grows = 5`; 23/23 device PASS (+ perf lane, Poisson 10.2–10.5×, no R-13 flake); `cb-backend
--lib` under rocm `265 passed / 0 failed` (263 at T08 + T09's 2 new tests); `cb-train --lib`
`390 passed / 0 failed`.

## From T10 → **T12, T16, T19, T22, T23** and **T24** (assumption 8 is SETTLED)

**1. PLAN §6 assumption 8 is resolved: BOTH leaf paths run on a device CTR fit, and they
agree by construction. No escalation, no gather patch.**

* the returned leaf **VALUES** come from the device session's **position-indexed** gather
  (`session.rs:2782-2839`, `avg_bins.get(fu - base)`);
* the per-object leaf **ASSIGNMENT** (main-approx update + leaf weights) comes from the host
  **full-identity** `assign_leaf_over_ctr_columns(&matrix, &averaging_ctr_features, …)`
  (`boosting.rs:5609`; reached because `fused_unit_fold` is `false` whenever a CTR split
  exists, `boosting.rs:5511`).

Position indexing is correct for Buckets' two-columns-per-`(projection, prior)` layout
because `materialize_ctr_columns_for_perm` is the **single** producer of both the structure
and the averaging list (called at `boosting.rs:4254` and `:4284` over the same
`absolute_projections`/`ctr_candidates`), `build_device_ctr_config`'s `build_columns` is an
order-preserving `map` over each, and `ctr_covered` rejects `avg.columns.len() !=
ctr.columns.len()`. ⇒ device tail position `i` ⇔ the same FULL identity on both host lists.
**T17/T19 must preserve that single-producer property**: any task that filters, reorders or
de-duplicates one of the two lists without the other silently breaks the pairing, and the
detector is `device_ctr_buckets_fit_test` (mispairing measured at `|Δ| = 2.506e-1`).

**2. The `ctr_device_buckets` fit selects only `b = 0` splits on the DEVICE** (`[0]×8`),
while the CPU fallback selects `[0,1,0,0,1,1,0,1]` — and **both reproduce upstream at
`max |Δpred| = 2.776e-17`**. Cause, exact: at `Prior = 0.5` and binclf,
`ctr(Buckets@0) + ctr(Buckets@1) = (total + 1)/(total + 1) = 1`, and
`calc_normalization(0.5) = (0, 1)`, so the two columns binarize to mirrored bins and
`bin(b=0) > k` is the complement of `bin(b=1) > 14−k` — the same oblivious partition with one
level bit flipped, hence identical predictions. Two consequences:
  * **a `Prior = 0.5` Buckets fixture can never distinguish "b=0 chosen" from "b=1 chosen" by
    predictions.** A task needing a b-discriminating e2e must use a prior ≠ 0.5.
  * this e2e does **not** exercise the `b = 1` numerator through the *scoring* path. The
    `b = 1` evidence is T08's kernel self-oracle plus T10's MUT-A. Do not cite "the Buckets
    e2e is green" as proof that `b = 1` scoring is right.

**3. T20's CPU-fallback fingerprint has a counterexample — MATERIAL for T12/T16/T19.**
On `ctr_device_buckets` the device and CPU paths print the **same** `max |Δpred| =
2.776e-17`. So "device delta ≠ CPU delta" is NOT a reliable device-commitment signal either.
The two that held: `CountingGpu.grown.get() == iterations`, and runtime (1.7 s device vs
0.01 s CPU). T20's rule survives in its *original* one-way form ("a tiny delta is a smell,
never reassurance"), not as a fingerprint match.

**4. `boosting_ctr_gate_test.rs` now has FOUR more pins, two of them source scans over the
gate body — and one is a trap for the next editor.**
`the_device_gate_no_longer_reads_the_buckets_numerator_selector` asserts the RAW body text
(inline comments included) does not contain `target_border_idx`; the sibling type pin asserts
it does not contain `ECtrType::Borders.as_i8()` and DOES contain `ECtrType::from_i8`.
⇒ **T12/T16 must not spell either phrase in a comment inside the gate body**, and **T23,
which rewrites the predicate to delegate to `from_i8`/`is_cpu_supported`, must update or
retire all six pins in that file** (T01's two + T10's four) as part of its own change — the
`from_i8` conjunct of T10's pin is the one T23 can keep.

**5. The gate-state table gained rows 9 and 10 (APPENDED — rows 1-8 keep their numbers)** and
`gate_admits_exactly_the_current_wave` now reports **all** mismatching rows instead of failing
at the first. Later gate tasks get a complete admitted-set diff in one run; the message format
is `row N (label): expected X, got Y` + the row's `flips_at`, one block per mismatch.

**6. T00's question answered.** T10's edit removed **both** conjuncts (`ctr_type == Borders`
and `target_border_idx == 0`), proved by two single-conjunct restorations: restoring the type
conjunct alone reddens rows 4+9 and leaves row 10 green; restoring the target-border conjunct
alone reddens rows 4+10 and leaves row 9 green. Verbatim output in `notes/T10.md` §3.4.

**7. Pre-existing clippy item for T24's list (NEW — cb-backend, DEFAULT cpu features).**
`cargo clippy -p cb-backend --lib` emits `clippy::duplicated_attributes`: a doubled
`#[allow(clippy::too_many_arguments)]` at `crates/cb-backend/src/gpu_runtime/mod.rs:4401` and
`:4433`. Verified pre-existing at `HEAD` (`git show HEAD:…/mod.rs`); `mod.rs` is in no P1
task's diff so far. Distinct from T08's `erasing_op` (rocm, `score_split.rs:374`) and T04's 12
`cb-train` test targets.

**8. `CountingGpu` is now duplicated THREE times** (`device_ctr_gate_test.rs:82-138` canonical,
`device_ctr_fit_test.rs`, `device_ctr_buckets_fit_test.rs`). T24's DoD count should expect one
more per remaining e2e (T12, T16, T19).

**D-04 held at T10:** `device_ctr_fit_test` `max |Δpred| = 4.483e-11`, unchanged; `device
grows = 5`; 23/23 device PASS (+ perf lane, Poisson 10.8×, no R-13 flake); `cb-backend --lib`
under rocm `265 passed / 0 failed` (identical to T09); `cb-train --lib` `394 passed / 0 failed`;
`cargo test -p cb-train` **107 targets, all ok**.

## From the coordinator, off T10's finding #2 → **T22** (READ BEFORE DESIGNING THE DIFFERENTIAL)

T22 is specified as a device-vs-CPU **chosen-split-sequence** differential, deliberately
comparing splits rather than predictions. T10 measured a case where **the chosen splits
legitimately diverge while the predictions are identical**: on `ctr_device_buckets`
(`Prior = 0.5`, binclf) the device selects `[0]×8` and the CPU selects `[0,1,0,0,1,1,0,1]`,
both at `max |Δpred| = 2.776e-17`, because `Buckets@0` and `Buckets@1` binarize to mirrored
bins and the two split sets describe the **same oblivious partition with one level bit
flipped**.

⇒ A naive "device split sequence == CPU split sequence" assertion **will fail for a benign
reason** on any `Prior = 0.5` Buckets configuration. T22 must either

* choose a prior ≠ 0.5 for its Buckets arm (T10 §2 shows `0.5` is the degenerate point), or
* compare a **partition-invariant** projection of the split set rather than the raw
  `(feature, border)` sequence,

and must say in its note which it did and why. Do **not** discover this as a red test and
"fix" it by loosening the assertion after the fact — that is exactly the class of change
§2.5 exists to prevent. Note this cuts the other way too: because the partitions coincide,
a raw-sequence comparison that happens to pass is not evidence the two paths agree.

## From T11 → **T12** (primary), **T14/T16** (the Counter kernel sibling landed)

T11 added `counter_ctr_kernel<F: Float>` (`ctr_device.rs`, `#[cube]` at `:262`) and
`launch_counter_ctr_resident` (`:524`), plus two host-readback oracle seams
(`compute_counter_ctr_host` `:739`, `binarize_counter_column_host` `:781`). **No production call
site exists** — `grep -rn "launch_counter_ctr_resident\|counter_ctr_kernel::launch" crates/
--include=*.rs` returns only definitions and the two in-file oracle callers. Five things later
tasks must act on:

**1. T12 must call `launch_counter_ctr_resident`, NOT `launch_ordered_ctr_resident`.** The
ordered launcher still rejects `ctr_type == 4` with `CbError::OutOfRange` (T08 finding #1) and
that guard **stays** — widening it was considered and deliberately not done. The new entry point
takes `(client, bins, prior, bucket_count, n)`: **no `permutation`, no `target_class`, no
`target_border_idx`** (structural permutation independence, `ctr_type.cpp:43-56`). A Counter arm
that threads a permutation in will not type-check, which is the point.

**2. There is no `CTR_TYPE_COUNTER` const yet.** T12 should add it beside `CTR_TYPE_BORDERS` /
`CTR_TYPE_BUCKETS` (`ctr_device.rs:277`/`:279`) with the same C-3 discriminant citation, rather
than spelling a bare `4` in `session.rs`. T09's finding #1 already specifies the rest of T12's
hunk (both `ctr_covered` closures + the `build_ctr_cindex_columns` match + moving `4` from the
decline loop to the admit loop in `session_ctr_type_test`).

**3. C-7 confirmed empirically.** Counter returns the same `ResidentCtr` triple as the ordered
path, with `total[obj]` = the CONSTANT max-bucket denominator repeated per object (mirroring
`ctr_feature.rs:304`'s `denoms = vec![denominator; n]`), so `binarize_ctr_column_resident` and
the existing per-column border table apply **unchanged** — `quantize_in_f32 == false` for
Counter, the same f64 quantizer as Borders/Buckets. `binarize_counter_column_host`'s bit-exact
oracle is the proof. No per-type border handling is needed anywhere.

**4. Kernel-style note for T14 (BTMV), cost-free.** Under a generic `<F: Float>`, `F::new(1.0)`
emits `warning: falling back to f32 as the trait bound f32: From<f64> is not satisfied`
(`float_literal_f32_fallback`, a **future hard error**) — `Float::new` takes an `f32`. Use
`F::from_int(1)` for integer constants inside a generic `#[cube]` body. T14's BTMV kernel does
the same `/(count + 1)` division and will hit this on its first draft. **NOTE T14's contract is
the opposite of T11's on width**: BTMV's accumulator is a *parity contract* `Array<f32>`
(`TCtrMeanHistory::Sum` is `float` upstream) and must stay concrete with the C-2/§2.4 comment,
whereas T11's Counter tally is an exact integer `Array<u32>` and its value channel is generic.

**5. The device suite now exercises the ctr_device module on BOTH backends.**
`cargo test -p cb-backend --lib kernels::ctr_device_test` (**default cpu** features) is
`10 passed; 0 failed`, same as under rocm. Useful cheap smoke for any later kernel edit in this
file — it catches a broken `#[cube]` body without a GPU. It is **not** a substitute for the rocm
run (R-9 still applies to anything claiming device evidence).

**D-04 held at T11:** `device_ctr_fit_test` `max |Δpred| = 4.483e-11`, unchanged; `device grows
= 5`; 23/23 device PASS (+ perf lane, Poisson 10.7×, no R-13 flake); `cb-backend --lib` under
rocm `267 passed / 0 failed; 2 ignored` (265 at T09/T10 + T11's 2 new tests). Zero clippy
diagnostics originate from `ctr_device.rs` / `ctr_device_test.rs`; the pre-existing `erasing_op`
at `score_split.rs:374` is still the only `error` on that lane.

## From T12 → **T13** (primary), **T16/T19** (the gate chain), **T24**

T12 wired T11's Counter kernel into production: `CTR_TYPE_COUNTER` (`ctr_device.rs`, beside the
other two consts), `4` added to **both** `ctr_covered` closures, a
`CTR_TYPE_COUNTER => launch_counter_ctr_resident(client, &bins, col.prior, buckets, n)` arm in
`build_ctr_cindex_columns`, and `ECtrType::Counter` added to the `cb-train` gate's `from_i8`
admission list. `device_ctr_counter_fit_test`: **`grown = 5`, `max |Δpred| = 1.388e-17`, 1.73 s**.
Gate-state **row 3 flipped to `true`** in the same edit; rows 1-2 and 4-10 untouched.

**1. `counter_calc_method` is structurally moot on a device fit — T13's decline must come from
somewhere else.** Nothing in the device path reads that field, and the widening it performs
(learn + every eval-set tally, `online_ctr.cpp:714-729`) is reachable ONLY through
`train_cat_with_eval_sets` (PLAN C-1). Since T12 the CTR **type** list admits Counter, so a T13
negative test that leaves `eval_sets` empty now proves nothing at all — it would decline for no
reason and pass vacuously. T13 must drive a genuinely non-empty `eval_sets` slice and show the
decline comes from the eval-set/fold gate. This is recorded in-source in the gate's new
"Counter IS covered" doc section, which names T13 as the owner of that boundary.

**2. `session_ctr_type_test`'s mixed structure/averaging pair now uses
`BINARIZED_TARGET_MEAN_VALUE`, and T16 must swap it again.** Those two assertions
(`config_with(BORDERS, X)` / `config_with(X, BORDERS)`) are what pin that BOTH `.all(..)`
closures carry the type conjunct; they need an `X` that still DECLINES. T09 used `Counter`, T12
implemented it and moved the pair to BTMV. **When T16 admits BTMV it must move the pair to an
out-of-enum stray (e.g. `6`) — not delete the two assertions.** An in-source note says so.
T12 also added `COUNTER` to `buckets_keeps_the_borders_shape_checks`' loop (C-7 for the newly
admitted type, mutation-proved); T16 should add BTMV there the same way.

**3. The delta is not a fingerprint in EITHER direction — third data point.** On
`ctr_device_counter` the device prints `1.388e-17` and the CPU fallback `2.776e-17` (they
differ); on `ctr_device_buckets` (T10) both print `2.776e-17` (they coincide); on
`ctr_device_mixed` (T20) device `4.483e-11` vs CPU `1.388e-17`. Only `grown == iterations` and
the runtime (≈1.7-1.9 s device vs 0.01 s CPU) held every time.

**4. A prior mismatch is guarded by CONSTRUCTION, never by the fixture (T06's finding, closed).**
Three layers now: `simple_ctr_priors: vec![0.5]` in the e2e's `BoostParams`; an in-test
`assert_eq!` on it before the fit; and
`boosting_ctr_gate_tests::counter_is_a_cpu_legal_whole_set_tally_not_a_class_prefix`, which pins
`ECtrType::Counter.default_priors() == [0/1]` against Borders/Buckets' three-prior default
(mutation-proved). Any later Counter/BTMV fixture inherits the same hazard class: **check the
default-prior arm of `ECtrType::default_priors` before assuming the params list may be left
implicit.**

**5. PROCESS — never `git checkout <file>` to revert a §2.5 mutation in a file your task also
edits.** T12 did that once after MUT-1 and silently lost all three of its `boosting.rs` hunks;
caught by a `grep` for the admitted discriminant, reapplied, and every later mutation used a
targeted textual revert. T10's use of `git checkout` was safe only because it had mutated a file
(`ctr/mod.rs`) it had not otherwise edited. Verify with `git diff <file>` after every revert.

**6. `CountingGpu` is now duplicated FOUR times** (`device_ctr_gate_test.rs:82-138` canonical,
`device_ctr_fit_test.rs`, `device_ctr_buckets_fit_test.rs`, `device_ctr_counter_fit_test.rs`);
expect one more per remaining e2e (T16, T19) at T24's DoD count.

**D-04 held at T12:** `device_ctr_fit_test` `max |Δpred| = 4.483e-11`, unchanged; `device grows
= 5`; `device_ctr_buckets_fit_test` unchanged at `2.776e-17`; 23/23 device PASS (+ perf lane,
Poisson 10.7×, no R-13 flake); `cb-backend --lib` under rocm `267 passed / 0 failed; 2 ignored`
(identical to T11); `cb-train --lib` `396 passed / 0 failed`; `cargo test -p cb-train` **108
targets, all ok**. No new pre-existing-failure item was discovered.

## From T13 → **T21** (file co-owner), **T16/T19** (every remaining e2e), **T23/T24**

T13 created `crates/cb-train/tests/device_ctr_type_gate_test.rs` with the
`{SkipTest, Full} × {no eval set, eval set}` square (4 tests, 1.77 s, all green). Production
change: **doc comment only** on `ctr_types_are_device_covered`. D-04 held (`4.483e-11`);
`run_device_tests.sh` **24 PASS / 0 FAIL**; `cargo test -p cb-train` **109 targets, all ok**.

**1. The eval-set decline is at `device_host_eligible`, not at the CTR type list — and both
mutations were needed to say so.** MUT-1 (delete only `&& eval_sets.is_empty()`,
`boosting.rs:4620`) flips **both** declining cells to `grown = 5` (`left: 5, right: 0`,
exactly as predicted) and leaves both committing cells green. MUT-2 (T20's universal
`&& false` on the type gate) reddens **only** the two committing cells (`left: 0, right: 5`,
run collapses to 0.02 s). The two mutations partition the four cells; on its own a
`grown == 0` assertion cannot distinguish "declined at the eval-set clause" from "declined at
the type clause". **Any later task asserting a decline should run the mutation for the clause
it CLAIMS, not merely a mutation that reddens the test.**

**2. LEARN-SET SIZE IS A VACUITY HAZARD — read before designing any new CTR e2e.** T13's first
draft followed the task text literally (32-row learn half of `ctr_device_counter`) and **three
of four cells grew models with ZERO CTR splits**: at 32 rows this recipe's float splits beat
every Counter CTR candidate. The cell that *passed* was the required negative — i.e. without
a `Σ ctr_splits >= 1` guard the test would have shipped green, vacuous and CTR-free. Two
carries: (a) **every CTR routing test must assert ≥1 chosen CTR split AND assert the chosen
splits' `ctr_type`**, not just `grown`; (b) **do not shrink or resample a CTR fixture's learn
set** — T13's fix was to make the learn side the whole frozen fixture (byte-identical to
DCTR-10's proven-committing configuration) and take only the eval slice from it. T16/T19
should reuse their fixture's full learn set for the same reason.

**3. `counter_calc_method` is now CLOSED as a P1 question.** `Full ≡ SkipTest` whenever
`eval_sets` is empty (`counter_full_eval_columns` is assembled purely out of `eval_sets`,
`boosting.rs:4231-4249`), and eval sets never reach the device. The gate's doc block now says
so with the citations and the layer. **The two declining cells carry `// P3 WILL INVERT
THIS.`** — P3 must FLIP them to `grown == params.iterations`, not preserve or delete them.

**4. File ownership (T21).** `device_ctr_type_gate_test.rs` is shared with **T21**, which adds
the DCTR-03 surviving-clause pins to the same file. T21 can reuse `device::assert_route(tag,
method, with_eval_set, expect)` and `device::Route`; `load_split()` is `ctr_device_counter`
specific. T22 does **not** touch this file (it owns `device_ctr_combo_types_diff_test.rs`).

**5. `CountingGpu` is now duplicated FIVE times** (`device_ctr_gate_test.rs:82-138` canonical,
`device_ctr_fit_test.rs`, `device_ctr_buckets_fit_test.rs`, `device_ctr_counter_fit_test.rs`,
`device_ctr_type_gate_test.rs`); expect one more per remaining e2e (T16, T19) at T24's DoD
count.

**6. T23 must add `device_ctr_type_gate_test` to `run_device_tests.sh`'s `TESTS=(…)`** (C-8 —
T13 did not touch that file; the binary was verified with explicit `--test` invocations).

**7. Process (confirms T12 §6).** Both mutations were reverted by **targeted textual edit**
and `git diff crates/cb-train/src/boosting.rs` re-verified after each (40 ins / 4 del,
T12's three hunks, byte-identical before and after). `grep -rn "MUTATION-T13" crates/` is
empty. No new pre-existing-failure item was discovered; T02/T04/T08/T10's list is unchanged.

## From the coordinator → **T23 and T24** (ownership of `run_device_tests.sh` — settle this now)

T10 and T13 each recorded that "T23 owns `run_device_tests.sh`". That is **wrong**. PLAN §6
**C-8** names **T24** as the single owner, precisely to avoid concurrent edits to the
`TESTS=(…)` array, and §7's DoD makes registering every new device binary a T24
deliverable. **T23 must not touch the script; T24 registers all of the new binaries in one
edit.**

As of the end of Wave 4 the script is byte-unchanged from `HEAD` (23 test names + the
isolated perf lane). New device binaries created so far that T24 must register:

* `device_ctr_buckets_fit_test`   (T10)
* `device_ctr_counter_fit_test`   (T12)
* `device_ctr_type_gate_test`     (T13)

…plus whatever T16, T19 and T22 add. `device_ctr_combo_fit_test` is **already** listed
(line 13) and is currently `#[ignore]`d — T19 un-ignores it, which changes that entry from
a vacuous pass to a real one without changing the array.

## From T14 → **T15** (primary), **T16** (the wiring), **T24**

T14 added `btmv_ctr_prefix_kernel<F: Float>` (`ctr_device.rs`, `#[cube]` at `:353`, `fn` at `:354`),
`ResidentCtrMean` (`:454`), `launch_btmv_ctr_resident` (`:731`) and three host-readback oracle seams
(`compute_btmv_ctr_host` `:1057`, `read_btmv_sum_bytes` `:1110`, `binarize_btmv_column_host` `:1149`).
**No production call site exists** — `grep -rn "launch_btmv_ctr_resident\|btmv_ctr_prefix_kernel::launch\|ResidentCtrMean"
crates/ --include=*.rs` returns only definitions, their in-file callers, and one comment mention.
Six things later tasks must act on:

**1. PLAN §6 C-2 is now MEASURED, in both directions — and the SPEC sentence it corrects is dead.**
Under a deliberately f64-widened accumulator, `btmv_f32_accumulation_width_is_load_bearing`
(`divisor = 3`) fails with **38 of 96** documents mismatching (checker predicted 22–41), first at
doc 0 `device 0x40E00000 (7)` vs `f32 reference 0x40E00001 (7.0000005)`. **In the same mutated
build, `btmv_prefix_matches_cpu_reference_at_binclf` (`divisor = 1`) and
`btmv_sum_output_buffer_is_f32_wide` both PASSED.** ⇒ `SPEC.md` DCTR-12's "an f64 device sum must
FAIL this test" is false at binclf, exactly as C-2 says; the multiclass detector is the only proof,
and the buffer-width test is an output-shape pin whose own comment states it would pass under an
`Array<f64>` accumulator. **Any later task tempted to "simplify" the synthetic `divisor` parameter
away destroys the only evidence DCTR-12 has.**

**2. T16 must call `launch_btmv_ctr_resident`, NOT `launch_ordered_ctr_resident`.** T08's guard
still rejects `ctr_type == 2` with `CbError::OutOfRange` and **was not widened**. The new entry
point takes `(client, perm, bins, class, prior, divisor, bucket_count, n)` — **no `ctr_type`, no
`target_border_idx`** — plus a `divisor` (= `targetClassesCount - 1`), which on the simple CTR path
is `SIMPLE_CLASSES_COUNT - 1 == 1` and nothing else. A BTMV arm that threads a `target_border_idx`
in will not type-check, which is the point. It returns `ResidentCtrMean`, **not** `ResidentCtr`
(the numerator channel is an f32 `sum`, not an integer `good`), so the `build_ctr_cindex_columns`
match arm cannot simply be copied from the Counter arm — it must bind `res.value` off the new type.
`binarize_ctr_column_resident` and the per-column border table then apply **unchanged** (C-7).

**3. There is no `CTR_TYPE_BTMV` const yet.** T16 should add `CTR_TYPE_BTMV: i8 = 2` beside
`CTR_TYPE_BORDERS`/`_BUCKETS`/`_COUNTER` (`ctr_device.rs`, the `:466`-ish block) with the same C-3
discriminant citation, rather than spelling a bare `2` in `session.rs`. T09 finding 1 + T12 finding
2 already specify the rest of T16's hunk (both `ctr_covered` closures, the match arm, moving `2`
from the decline loop to the admit loop, moving `session_ctr_type_test`'s mixed structure/averaging
pair off BTMV onto an out-of-enum stray such as `6`, and adding BTMV to
`buckets_keeps_the_borders_shape_checks`' loop).

**4. T04's ≥3-documents finding is now enforced mechanically, and T15 inherits it.** A
`max_bucket_occupancy(bins) >= 3` assertion guards both BTMV parity oracles. It is not decoration:
MUT-B (leakage — fold the target in before reading) leaves every FIRST-in-bucket document matching,
so a fixture whose buckets hold ≤2 documents detects roughly nothing. **T15's differential must
drive ≥3 documents per bucket too**, and must use `divisor = 1`: the BTMV ≡ Borders@0 identity is
binclf-only (it holds because the addend is `{0,1}`, so `Sum` counts class-1 documents exactly),
and at `divisor = 3` it is simply false. `max_bucket_occupancy` is already in `ctr_device_test.rs`
(`:219`) — reuse it.

**5. A green §2.5 mutation is not always evidence the mutation MISSED — it can mean the test is
vacuous. This one was.** MUT-C (delete the `divisor == 0` guard) **PASSED on its first run**: the
sibling `class > divisor` guard fired instead on the fixture's class-1 documents, so the assertion
had been green without the guard it claimed to pin ever existing. The fix (applied while the
mutation was live) was an **all-zero class column**, after which MUT-C fails with the real payload —
an `Ok(([NaN, 0.0, NaN, …]))` CTR column, which `binarize_ctr_kernel` maps to bin `0` everywhere
because `NaN > border` is false. **Generalisation for T15/T16/T19/T21/T22: when a negative test has
two or more guards that could each reject the same input, mutating one of them proves nothing
unless the input is constructed so the others cannot fire.** Order-dependent guard chains make this
easy to get wrong and impossible to see without the mutation.

**6. `ctr_device.rs` now holds THREE launchers with three different contracts** —
`launch_ordered_ctr_resident` (`perm, bins, class, prior, bucket_count, n, ctr_type,
target_border_idx` ⇒ `ResidentCtr`), `launch_counter_ctr_resident` (`bins, prior, bucket_count, n`
⇒ `ResidentCtr`), `launch_btmv_ctr_resident` (`perm, bins, class, prior, divisor, bucket_count, n`
⇒ `ResidentCtrMean`). T24's DoD should note that the "one entry point per statistic, differing in
exactly the arguments the statistic depends on" property is what makes a copy-paste routing error a
compile error rather than a silent wrong numerator.

**D-04 held at T14:** `device_ctr_fit_test` `max |Δpred| = 4.483e-11`, unchanged;
`device_ctr_buckets_fit_test` `2.776e-17`; `device_ctr_counter_fit_test` `1.388e-17`;
`device grows = 5` on all three; `bash ./run_device_tests.sh` **24 PASS / 0 FAIL** (+ perf lane,
Poisson 10.5×, no R-13 flake); `cb-backend --lib` under rocm `271 passed / 0 failed; 2 ignored`
(267 at T11/T12 + T14's 4 new tests); the same 14 `ctr_device_test` tests also pass on the **default
cpu** backend; `cb-train --lib` `396 passed / 0 failed`; `cargo test -p cb-train` **109 targets, all
ok**. Zero clippy diagnostics originate from `ctr_device.rs` / `ctr_device_test.rs`; the pre-existing
`erasing_op` at `score_split.rs:374` is still the only `error` on that lane. **No new
pre-existing-failure item was discovered.**

## From T15 → **T16** (primary), **T19/T21/T22** (the §2.5 pattern), **T24**

T15 added `btmv_and_borders_emit_identical_bins_at_binclf` (`ctr_device_test.rs:1026`) plus three
test-side helpers (`CTR_BORDER_COUNT` `:217`, `calc_normalization` `:223`, `device_ctr_border_table`
`:239`). **Test-only — zero production change**: `git diff --numstat` on
`crates/cb-backend/src/kernels/ctr_device.rs` reads `375  0` both before and after T15, i.e. still
byte-identical to the state T14 left it in.

**1. PLAN §2.5's offered substitute for T15 was DECLINED, and the reasoning generalises.** §2.5 says
T15's `>= 2 distinct bins` guard is an "accepted substitute" for a full mutation. T15 kept the guard
**and** ran two real mutations, because the guard cannot see the vacuity mode specific to a
**differential**: an equality between two columns is also satisfied when the two columns are the
*same* column. A test that called one arm twice would be green, non-degenerate, and would assert
nothing — T14's MUT-C failure mode transposed. A mutation of **one** arm settles it in both
directions at once (a self-comparison moves both sides equally, or neither, and stays green).
⇒ **Rule for T19/T21/T22: for any test whose assertion is `A == B` across two paths, the ≥N-distinct
-values guard proves non-degeneracy, never non-tautology. Only a one-sided mutation proves the two
arms are distinct.** Verbatim failures in `notes/T15.md` §3.

**2. Buckets@0 and Borders@0 are the COMPLEMENTARY numerators at binclf, and this was measured
twice, independently.** MUT-1 (BTMV's accumulator folds `1 − class`, so `Sum = N[0]`) and MUT-2
(the reference arm's selector moved from `CTR_TYPE_BORDERS` to `CTR_TYPE_BUCKETS` inside
`binarize_ctr_column_host`) each turn the test red on **all 128 of 128 documents**, and MUT-2's
`left`/`right` vectors are **element-for-element MUT-1's `right`/`left`**. Two one-line mutations in
two different kernels reached through two different launchers produce the same column pair with the
arms exchanged. Useful to T22: on a `Prior = 0.5` binclf fixture the `Buckets@0` and `Borders@0`
**bin columns are a mirrored pair**, which is the per-column form of T10's finding #2 (mirrored bins
⇒ the same oblivious partition with one level bit flipped ⇒ identical predictions). A
prediction-level differential cannot separate them; a **cindex-column-level** one can, and does.

**3. The production border table is now reproducible from `cb-backend` test code.**
`device_ctr_border_table(prior, bc)` implements `build_device_ctr_config`'s own
`borders[k] = ((k+1)·norm/bc − shift).next_down()` at the upstream default `bc = 15`, over an inline
transcription of `calc_normalization` (C-3 — `cb-backend` must not import `cb-train`). Every earlier
`ctr_device_test` oracle used an ad-hoc table (`[0.2, 0.4, 0.5, 0.6, 0.8]`, `COUNTER_BORDERS`).
**T16 and any later kernel-level differential should prefer `device_ctr_border_table`**: it is the
table a real fit emits, and T15 measured that it resolves an `N[1] → N[0]` numerator swap on *every*
object, whereas an ad-hoc coarse table need not.

**4. T04's ≥3-documents constraint is satisfied with a wide margin on `synth_fixture(128, 5, 3)`** —
recomputed independently from the LCG, the five buckets hold `{32, 19, 24, 28, 25}` documents
(class counts `{0: 63, 1: 65}`). The runtime `max_bucket_occupancy(&bins) >= 3` assertion is in the
test, so this is enforced rather than assumed. `BTMV_DIVISOR_BINCLF = 1` throughout, per T14 §8 —
the equivalence is binclf-only and false at `divisor = 3`.

**5. Nothing new blocks T16.** T14 §8's wiring instructions stand unchanged: call
`launch_btmv_ctr_resident` (never `launch_ordered_ctr_resident`, whose guard still rejects
`ctr_type == 2`), add `CTR_TYPE_BTMV: i8 = 2` beside the other three consts with the C-3 citation,
then T09 finding 1's three-part hunk and T12 finding 2's obligations (move
`session_ctr_type_test`'s mixed structure/averaging pair off `BINARIZED_TARGET_MEAN_VALUE` onto an
out-of-enum stray such as `6`; add BTMV to `buckets_keeps_the_borders_shape_checks`' loop).
`ResidentCtrMean` — not `ResidentCtr` — is what the new match arm must bind `res.value` off.

**6. `CountingGpu` duplication count is UNCHANGED at FIVE** (T15 added no e2e). T16 and T19 are
still expected to add one each.

**D-04 held at T15:** `device_ctr_fit_test` `max |Δpred| = 4.483e-11`, unchanged;
`device_ctr_buckets_fit_test` `2.776e-17`; `device_ctr_counter_fit_test` `1.388e-17`; `device grows
= 5` on all three; `bash ./run_device_tests.sh` **24 PASS / 0 FAIL** (+ perf lane, Poisson 10.6×, no
R-13 flake); `cb-backend --lib` under rocm `272 passed / 0 failed; 2 ignored` (271 at T14 + T15's 1
new test); the same 15 `ctr_device_test` tests also pass on the **default cpu** backend; the wgpu
`cargo check` is clean with zero `ctr_device` diagnostics. Zero clippy diagnostics originate from
`ctr_device.rs` / `ctr_device_test.rs`; the pre-existing `erasing_op` at `score_split.rs:374` is
still the only `error` on that lane. **No new pre-existing-failure item was discovered.**

## From T16 → **T19** (primary), **T22/T23** (the closed type list), **T24**

T16 wired T14's BTMV kernel into production: `CTR_TYPE_BTMV` + `BTMV_DIVISOR_BINCLF`
(`ctr_device.rs:488`/`:497`, beside the other three consts), `2` added to **both** `ctr_covered`
closures, a `CTR_TYPE_BTMV => launch_btmv_ctr_resident(client, permutation, &bins, target_class,
col.prior, BTMV_DIVISOR_BINCLF, buckets, n)?.value` arm in `build_ctr_cindex_columns`, and
`ECtrType::BinarizedTargetMeanValue` added to the `cb-train` gate's `from_i8` admission list.
`device_ctr_btmv_fit_test`: **`grown = 5`, `max |Δpred| = 2.776e-17`, 1.69 s** (CPU fallback
0.01 s). Gate-state **row 5 flipped to `true`** in the same edit; rows 1-4 and 6-10 untouched.

**1. THE DEVICE CTR TYPE LIST IS NOW CLOSED at the four CPU-legal types**
(`{Borders, Buckets, BinarizedTargetMeanValue, Counter}`). The only conjunct left in
`ctr_types_are_device_covered` is `col.projection.is_simple()` — **T19's**. A new pin,
`boosting_ctr_gate_tests::the_admitted_set_is_exactly_the_cpu_supported_types`, makes that
executable: the gate must name each `is_cpu_supported()` type exactly once and **never** name
`FloatTargetMeanValue` / `FeatureFreq`. ⇒ **T23's rewrite to delegate to
`from_i8`/`is_cpu_supported` is now a near-identity transformation of the admission list**, but
it must retire or update **all NINE** source-scan pins in `boosting_ctr_gate_test.rs` (T01's 2 +
T10's 4 + T16's 3), not the six T10 counted.

**2. MATERIAL for T19/T22 — the BTMV e2e CANNOT police its own routing, and no fixture choice
could fix that.** T15's binclf identity (`Sum == N[1]`, `Count == Total` ⇒ BTMV and `Borders@0`
emit IDENTICAL cindex bins) means a device path that kept routing these columns through
`launch_ordered_ctr_resident` at `(CTR_TYPE_BORDERS, 0)` would produce identical predictions and
pass the ≤1e-5 bar. This is **structural and holds for every binclf BTMV fixture** — unlike T06's
Counter-prior blind spot, which was fixture-specific. The three layers that close it: the
per-split `ctr_type` assertion in the e2e; the launchers' incompatible signatures (a swap is a
compile error); and DCTR-12's kernel self-oracle. **T22's combination × BTMV arm must therefore
compare something other than predictions** — a cindex-column-level differential works (T15 §2),
a prediction-level one does not.

**3. MUT-6 is a new, cheap, GENERALLY USEFUL mutation shape: delete the dispatch ARM, not the
gate.** Replacing `CTR_TYPE_BTMV => {` with an unreachable discriminant made the e2e fail with
`device BTMV CTR train failed: Unsupported("device CTR type 2 is not implemented")` in 0.14 s.
Unlike T20's `&& false` (which proves the fit *can* commit) this proves the fit reaches **the
specific new production call site** — the thing a device-commitment assertion alone cannot say.
Every later task adding a `build_ctr_cindex_columns` arm should run it. It also records C-14 in
its strongest form: because `ctr_covered` admits the type, the `Ok(None)` decline path is not
taken and the mismatch surfaces as a **typed error out of `train_cat`**, not even as
`grown == 0`.

**4. Fourth data point on the delta.** On `ctr_device_btmv` the device and the CPU fallback print
the **same** `2.776e-17` (as on `ctr_device_buckets`); on `ctr_device_counter` and
`ctr_device_mixed` they differ. `grown == iterations` and the runtime (1.65-1.71 s device vs
0.01 s CPU) are still the only two signals that held every time.

**5. The prior trap is per-type and INVERTS between Counter and BTMV.** Counter's default is the
single `0/1`; **BTMV's is the `{0, 0.5, 1}` TRIPLE** — so an omitted `simple_ctr_priors` on a
BTMV fit materializes THREE CTR columns against a one-descriptor fixture. Both directions are now
pinned in `boosting_ctr_gate_test.rs` and both e2es assert their own `simple_ctr_priors` before
the fit. **Any later task copying one e2e's `ctr_params()` into another must re-check
`ECtrType::default_priors`' arm for the new type.**

**6. `session_ctr_type_test`'s mixed structure/averaging pair now uses an OUT-OF-ENUM STRAY**
(`UNKNOWN_DISCRIMINANT: i8 = 6`), and this is the **durable** end state: with the CPU-legal set
fully admitted, no future task can implement a stray, so the pair can never need swapping again.
It was mutation-proved to still discriminate (deleting the type conjunct from the averaging
closure alone reddens it). **Never delete those two assertions** — they are the only pins that
BOTH `.all(..)` closures carry the type conjunct.

**7. `build_ctr_cindex_columns`' `match` now binds `value: Handle`, not a `res` struct** —
forced, because `launch_btmv_ctr_resident` returns `ResidentCtrMean` and the other two return
`ResidentCtr`. Binding only the shared f64 `value` channel makes C-7 visually obvious: the
`binarize_ctr_column_resident` below cannot see the CTR type at all. T17/T19 editing this region
should keep that shape. (T09 finding 4's orphaned doc block at `session.rs:213-224` is still
attached to `CtrSearchState` — **still not re-attached**, still out of scope.)

**8. `CountingGpu` is now duplicated SIX times** (`device_ctr_gate_test.rs:82-138` canonical,
`device_ctr_fit_test.rs`, `device_ctr_buckets_fit_test.rs`, `device_ctr_counter_fit_test.rs`,
`device_ctr_type_gate_test.rs`, `device_ctr_btmv_fit_test.rs`). **T24 must register the new
binary `device_ctr_btmv_fit_test`** in `run_device_tests.sh`'s `TESTS=(…)` — the fourth of the
four (with T10's, T12's and T13's).

**D-04 held at T16:** `device_ctr_fit_test` `max |Δpred| = 4.483e-11`, unchanged;
`device_ctr_buckets_fit_test` `2.776e-17`; `device_ctr_counter_fit_test` `1.388e-17`; `device
grows = 5` on all four e2es; `bash ./run_device_tests.sh` **24 PASS / 0 FAIL** (+ perf lane,
Poisson 10.6×, no R-13 flake); `cb-backend --lib` under rocm `272 passed / 0 failed; 2 ignored`
(identical to T15 — T16 added no `cb-backend` test); `cb-train --lib` `399 passed / 0 failed`
(396 at T12/T14 + T16's 3 new gate pins); `cargo test -p cb-train` **110 targets, all ok**.
**DCTR-05 held**: `ctr_btmv_simple_oracle_test`'s `--test-threads=1` output is byte-identical to
T03's captured baseline, and `crates/cb-oracle/fixtures/ctr_device_btmv/` is byte-untouched.
**No new pre-existing-failure item was discovered** — `cb-backend --lib` under default cpu is
still exactly 60 failures (T02's item), now `222 passed` as the suite grew.

## From the coordinator, consolidating T10 §2 + T16 → **T22** (BOTH non-Borders arms are prediction-blind)

Two independent measurements now say the same thing, and together they constrain T22's
design before it writes a line:

* **Buckets (T10 §2)** — at `Prior = 0.5` the `b=0` and `b=1` columns are exact mirrors, so
  device `[0]×8` and CPU `[0,1,0,0,1,1,0,1]` are the *same partition with one level bit
  flipped*, both at `2.776e-17`. Chosen splits legitimately diverge while predictions agree.
* **BTMV (T16)** — T15's binclf identity makes BTMV and `Borders@0` **prediction-identical on
  any binclf fixture**, so the BTMV e2e cannot detect that BTMV was routed to the Borders
  numerator at all.

⇒ For T22, **predictions are not a routing detector for either arm**, and a raw
`(feature, border)` split-sequence comparison is not one for Buckets. T16's recommendation,
which the coordinator endorses: **T22's differential must compare at the cindex-column
level** (the materialized bin columns), not predictions — or, for the Buckets arm, a
partition-invariant projection of the split set with a prior ≠ 0.5.

State this choice up front in T22's note with the reason. Do **not** discover it as a red
test and then loosen the assertion — that is precisely the class of change §2.5 exists to
prevent. And note it cuts both ways: a raw comparison that *passes* is not evidence the two
paths agree, because these degeneracies make disagreement invisible.

## From T17 → **T18** (primary), **T19** (the Track D tail), **T22/T24**, and a CPU-side gap

T17 landed D-1: `resident_combination_eligible` (`gpu_runtime/mod.rs`, beside
`resident_cat_feature_weight`), the seam → `CtrSearchState` → `ResidentCtrSearch` threading of
`projection_members`, the tree-lifetime `chosen_ctr_projections` local inside
`grow_oblivious_tree_resident`, the pre-scoring `continue` in pass C, and the winner push.
**Test-visible behaviour is byte-unchanged and that was measured, not assumed**: D-04
`4.483e-11` / Buckets `2.776e-17` / Counter `1.388e-17` / BTMV `2.776e-17`, `grows = 5` on all
four; `run_device_tests.sh` **24 PASS / 0 FAIL**; `cb-backend --lib` under rocm
`274 passed / 0 failed; 2 ignored` (272 at T15/T16 + T17's 2).

**1. MATERIAL — the CPU's `combination_ctr_eligible` covering tests are BLIND to the subset
conjunct, and so was T17's faithful transcription of them.** Mutating
`q.iter().all(|m| members.contains(m))` → `any` left **all seven** transcribed cases GREEN.
Cause: in every one of `cb-train/src/tree_test.rs:296-361`'s cases the chosen `q` is either a
full subset or wholly disjoint, and `|q| == 1` (where `all ≡ any`) or the arity conjunct rejects
first. The **same one-word edit to `cb_train::tree::combination_ctr_eligible` would also leave
the CPU suite green** — this is a real CPU-side coverage gap, not a device artifact. T17 added
an eighth case, a PARTIAL OVERLAP of the right arity (`members = [1,2,3]`, `chosen = [[1,9]]`
⇒ `false`), which is the only shape that separates "SUBSET of `p`" from "intersects `p`".
⇒ **T18 shares this predicate; do not re-derive the rule, call it.** And any later task adding
a CPU-side eligibility test should add the partial-overlap case there too. Generalisation, for
the third time in this phase (T14 §5, T15 §1, now here): **a case list transcribed from an
existing suite inherits that suite's blind spots — the mutation, not the transcription, is what
establishes discrimination.**

**2. T18 has everything it needs in scope, and C-16 is now written in-source.** At the
`eligible_max` line, `cs.projection_members` (index-aligned with `cs.bucket_counts`) and
`chosen_ctr_projections` are both live; the filter is a `zip`/`enumerate` over the two.
The comment there was rewritten by T17 and states explicitly that the expression is
**deliberately still unfiltered, T18 owns it**, names the CPU mirror
(`eligible_max_bucket_count`, `tree.rs:2920-2933`) and restates **C-16** — filter the INNER max
only, leave `.max(phantom_max).max(1)` outside it. **T18 has no e2e detector** while the arity
conjunct stands (every column simple ⇒ the filter is a no-op), exactly as DCTR-16 / R-20 say;
its evidence must be the unit test on the filtered max. Mirror T17's convention that an
ABSENT member list counts as SIMPLE (`is_none_or`), so the two gates cannot disagree on a
degenerate column.

**3. T19: `chosen_ctr_projections` is pushed for SIMPLE winners too — deliberately.** A simple
projection is the only `q` a 2-member combination can extend, so restricting the push to
combinations makes every 2-member column permanently ineligible and `ctr_device_combo` would
silently reproduce today's CPU-fallback numbers. **T19 owns the R-2 tree-lifetime mutation**
(hoist the list onto `CtrSearchState`, show `ctr_device_combo` reddens); the declaration sits
beside `float_split_count` inside `grow_oblivious_tree_resident`, so the hoist is a two-line
edit. Note also that `grow_oblivious_tree_ordered_resident` has **no** `ctr_search` parameter
and no CTR pass C at all — `grow_oblivious_tree_resident` is the complete gate surface.

**4. T10 §1's single-producer pairing is PRESERVED and the mechanism is worth restating.** The
eligibility gate skips a column from **scoring**; it never filters, reorders or de-duplicates
either the structure or the averaging column list, and `projection_members` is an
order-preserving `map` over the same `ctr.columns` that `bucket_counts`/`weight_groups` walk.
Column `c` still means `ctr_base + c`. Detector green throughout (`device_ctr_buckets_fit_test`
`2.776e-17`). **T18/T19 must keep that property — a filtered CINDEX list, as opposed to a
filtered SCORING loop, would break it at `|Δ| = 2.506e-1`.**

**5. The `bucket_counts` fallback took FOLD-ALL-MEMBERS, not the typed error — PLAN's premise
for the cheap option was wrong.** The plan calls the `bucket_count == 0` branch "unreachable in
practice"; it is unreachable from **production** but **not** from the committed test corpus.
`DeviceCtrColumn::default()` yields `bucket_count == 0`, and six committed literals use
`..DeviceCtrColumn::default()` — including `session_depth_gt1_test.rs:630`, the 2-member
combination. A `CbError::Degenerate` there would have reddened
`session_ctr_augments_resident_cindex`. The landed shape: production arm unchanged,
single-member arm **byte-unchanged**, combination arm folds all members through the SAME
`combine_projection_bins` identity `build_ctr_cindex_columns` uses. The false comment
*"(single-member columns; the gate admits only simple projections)"* is **deleted**.
⇒ **general caution for the remaining tasks: "unreachable in production" ≠ "unreachable", and
`..DeviceCtrColumn::default()` is how the test corpus reaches these branches.**

**6. The eight-site `projection_members` re-review is CLOSED.** All 8 test literals now state
the field explicitly; `grep -rn "DeviceCtrColumn {"` returns 10 lines (struct def + production +
8). **Site `session_depth_gt1_test.rs:630` was the only genuine misrepresentation** (a 2-member
combination that defaulted to an empty list) and is now `vec![0, 1]`; the other seven were
truthful but latently trap-shaped and are now `vec![0]`. One fact later tasks should know:
**`ctr_leaf_values_use_averaging_permutation_bins` (`:779`) is the ONLY fixture in that file
that actually GROWS a tree through pass C** — every other CTR fixture there stops at `begin`.
So it is the only one whose member list the gate really reads, and its `len() == 1` is what
makes T17 inert for it.

**7. Pre-existing-failure list CORRECTION for T24 — the default-cpu count is a RANGE, not 60.**
`cargo test -p cb-backend --lib` under default cpu now reads `225 passed / 59 failed` where T16
recorded `222 / 60`. All 59 are in `kernels::*` (zero in `gpu_runtime::*`, the only namespace
T17's production diff touches). The delta is a **flake**:
`kernels::exact_quantile_test::exact_quantile_weighted_matches_cpu` FAILED in three consecutive
isolated runs and passed in a fourth, on untouched code. ⇒ record the item as **59–60 failed**.
Also: **T10's `duplicated_attributes` moved to `gpu_runtime/mod.rs:4462`/`:4494`** (T17 inserted
the predicate above it) — same single diagnostic, new line numbers. T08's `erasing_op` at
`score_split.rs:374` is still the only `error` on the rocm clippy lane, and **no new
pre-existing-failure item was discovered**.

**8. `CountingGpu` duplication count is UNCHANGED at SIX** and `run_device_tests.sh` was **not**
touched (C-8, T24 owns it). T17 adds **no** new test binary — its test is a `cb-backend` lib
module (`gpu_runtime/ctr_eligibility_test.rs`, mounted with the plain
`#[cfg(test)] mod …;` sibling form that module already uses, not `#[path]`; C-11's `#[path]`
requirement applies to modules mounted inside `session.rs`, which this is not). **T24's
registration list is unchanged.**

## From T18 → **T22** (primary — R-20 is OPEN), **T19** (the Track D tail), **T24**

T18 landed D-2: `resident_eligible_max_bucket_count` (`gpu_runtime/mod.rs`, immediately after
T17's `resident_combination_eligible`, which it CALLS rather than re-deriving), pass C's
`eligible_max` now going through it, and the pass-C comment rewritten with the
`CalcMaxFeatureValueCount` (`greedy_tensor_search.cpp:1070-1088`) / `eligible_max_bucket_count`
(`tree.rs:2920-2933`) citations. **C-16 honoured**: `let max_bucket_count =
eligible_max.max(phantom_max).max(1);` is byte-unchanged and the phantom is not passed through
the filter. **D-04 held exactly** (`4.483e-11` / `2.776e-17` / `1.388e-17` / `2.776e-17`,
`grows = 5` on all four, against a baseline captured before the first edit);
`run_device_tests.sh` **24 PASS / 0 FAIL**; `cb-backend --lib` under rocm
`277 passed / 0 failed; 2 ignored` (274 at T17 + T18's 3).

**1. R-20 IS OPEN. T18 did not close it and could not have — do not let it be quietly closed.**
While the cb-train arity conjunct stands (T19's), every column reaching pass C has exactly one
member, so the filter is the IDENTITY on every reachable production input and **no fit on any
committed fixture can differ between filtered and unfiltered `eligible_max`**. T18 proves
(a) the helper filters (red-first, `left: 40, right: 6`, every conjunct mutation-proved), (b) the
composition is C-16-shaped, (c) at the SOURCE-TEXT level that pass C calls it, (d) via MUT-W
(`.max(1)` → `.max(1000)`) that the helper's VALUE is consumed by the production cat-feature
weight (`device_ctr_fit_test` → `|Δ| = 1.188e-1`). None of that says the FILTER changes an
observable outcome. **T22's mutation 1 — put the unfiltered `.max()` back at the call site and
check whether the device split SEQUENCE moves — is the outstanding measurement, and it is
UNMEASURED.** If it does not move, record R-20 as still open; do not cite T18's unit tests as
closure. (Reverting the call site also reddens T18's source-scan test; that is expected and is
not the measurement.)

**2. MATERIAL — D-2's UNIT tests cannot see the call site at all, and this was measured twice.**
Under MUT-4 (call site un-wired back to the inline unfiltered expression) and under MUT-5 (a real
C-16 violation: the phantom routed THROUGH the filter), **all four unit tests stayed green**;
only the source scan `pass_c_calls_the_filtered_max_and_folds_the_phantom_outside_it`
(`include_str!("mod.rs")`, the `boosting_ctr_gate_test.rs` pattern) reddened. Generalisation for
T19/T21/T22, now the fourth time in this phase (T14 §5, T15 §1, T17 §1): **a unit test on an
extracted helper proves the helper, never the call site.** If a task's whole value is at the
call site, it needs either a behavioural mutation of the call site or an explicit source pin —
the helper's own tests will happily stay green through a complete un-wiring.

**3. A green mutation that is NOT the T17 MUT-1c blindness mode — how to tell them apart.**
`.unwrap_or(1)` → `.unwrap_or(0)` and `.max(1)` → `.max(0)` BOTH pass. That is not test
blindness: the two clauses are mutually redundant, so `x.unwrap_or(0).max(1) ==
x.unwrap_or(1).max(1)` for every input and **no test can separate them**. Both directions were
driven before concluding this. ⇒ when a §2.5 mutation passes, drive its complement before
deciding which of the two diagnoses applies; and **do not "simplify" that guard away** — nothing
will catch it.

**4. T19 is what makes D-2 observable, and the intended behaviour will look like a change.**
Once a ≥2-member column can reach pass C, `eligible_max` can legitimately DIFFER between levels
of the same tree (a combination ineligible at level 0 and eligible at level 2 changes `maxCount`,
hence every unused column's cat-feature weight, hence potentially the winner). That is CPU
parity, not a regression. T10 §1's single-producer pairing is untouched by T18 — it filters a
FOLD over `bucket_counts`, never a column list; pass C still loops `for c in 0..cs.n_ctr` and
column `c` still means `ctr_base + c` (`device_ctr_buckets_fit_test` green at `2.776e-17`).

**5. Nothing new for T24's pre-existing list; two line-number moves.** `duplicated_attributes`
(T10's item) moved `gpu_runtime/mod.rs:4462`/`:4494` → **`:4535`/`:4567`** (T18 inserted the
helper above it) — same single diagnostic. `erasing_op` at `score_split.rs:374` is still the only
`error` on the rocm clippy lane, and the warning count is back to T17's **18** after T18 fixed
the one `needless_borrow` its own call site introduced. T18 adds **no** new test binary (its 3
tests are `cb-backend` lib modules in T17's `gpu_runtime/ctr_eligibility_test.rs`), so T24's
registration list is unchanged and `CountingGpu`'s duplication count stays at **SIX**.
`run_device_tests.sh` was **not** touched (C-8). Default-cpu `cb-backend --lib` re-measured at
T18: **`227 passed / 60 failed; 2 ignored`** (287 total = T17's 284 + T18's 3), all 60 in
`kernels::*`, **zero in `gpu_runtime::*`**, with the named flake
`kernels::exact_quantile_test::exact_quantile_weighted_matches_cpu` in the failure list —
i.e. T17's **59–60 RANGE** confirmed by a second independent observation, not a new item.

## From T19 → **T22** (R-20 is sharper, not closed), **T21/T23**, **T24**

T19 deleted the cb-train gate's LAST non-type conjunct (`col.projection.is_simple()`), flipped
T00's gate-state **row 2** in the same change, un-ignored `device_ctr_combo_fit_test` and drove
it through `CountingGpu`. Result: **`grown = 5`, `max |Δpred| = 2.082e-17`, 1.6-1.7 s**, 8 CTR
splits of which **3 are ≥2-member combinations**. PLAN §6 assumption 5's `≈2.082e-17` — an
expectation from a reverted spike — is now a measurement on the landed code. D-04 held exactly
(`4.483e-11` / `2.776e-17` / `1.388e-17` / `2.776e-17`, `grows = 5` on all four, against a
baseline captured before the first edit); `run_device_tests.sh` **24 PASS / 0 FAIL**;
`cb-backend --lib` under rocm `277 passed / 0 failed; 2 ignored` — **identical to T18**, because
T19's `cb-backend` diff is comments only. `cb-train --lib` **401 passed**; `cargo test -p
cb-train` **110 targets, all ok**.

**1. R-20 IS STILL OPEN, and the REASON it is open has changed — T22 must not misread this.**
T18 could not close R-20 because the filter was the IDENTITY on every reachable input while the
arity conjunct stood. That excuse is gone: ≥2-member columns now reach pass C. T19 therefore ran
T22's mutation 1 early as a cheap **observation** — D-2's call site reverted to the unfiltered
`cs.bucket_counts.iter().copied().max().unwrap_or(1).max(1)`, combo e2e re-run — and the result
is **byte-identical in every printed quantity**: `2.082e-17`, `grown = 5`, 8 splits / 3
combinations. ⇒ **the filter is live but no committed fixture moves under it.** R-20 stays OPEN;
T19 explicitly does **not** claim closure. T22's designated measurement (the device-vs-CPU split
**SEQUENCE**, per the coordinator's consolidated T10 §2 + T16 finding: NOT predictions) is still
outstanding, and if it too does not move, R-20 must be recorded as still open and `SPEC.md` must
say so. A fixture whose combination has a much larger bucket count than its members' simple
projections is the plausible discriminator, but that is an untested hypothesis, not a finding.

**2. The R-2 tree-lifetime failure mode is REAL, the combo e2e detects it, and its `|Δ|` equals
the control arm's.** MUT-1 (the genuine hoist: `chosen_ctr_projections` moved onto
`CtrSearchState` + `&'a mut` on `ResidentCtrSearch`, tree-local deleted) fails at
**`|Δ| = 2.746e-2`** — bit-for-bit the number PLAN/T00 record for "the arity gate simply opened
with NO per-level eligibility gate". That coincidence is informative: a fit-lifetime list makes
the gate vacuous from tree 1 onward, which is observationally identical to having no gate at
all. ⇒ `device_ctr_combo_fit_test` is now a live scope detector for any later change to D-1's
list lifetime.

**3. T10 §1's single-producer pairing is now EXECUTABLE, and it lives in
`boosting_ctr_gate_test.rs`.** T19's structural pin
(`combination_arity_is_structurally_bounded_and_carried_whole`) asserts by source scan that
`boosting.rs` holds exactly ONE `DeviceCtrColumn {` construction, exactly TWO `build_columns(`
calls (structure + averaging), and exactly ONE `tensor_ctr_candidates(` call, plus the
`max_ctr_complexity` arity bound and `TProjection`'s sorted/deduped members. Any later task that
filters, reorders or de-duplicates one column list without the other now trips a cb-train unit
test in addition to `device_ctr_buckets_fit_test`'s `|Δ| = 2.506e-1`. **Gotcha for whoever edits
that region: the closure binding `let build_columns = |…|` carries no `(`, so the expected count
is 2, not 3** (T19 hit this as its own first red).

**4. T23's source-scan pin count is ELEVEN, not nine.** T01's 2 + T10's 4 + T16's 3 + **T19's
2** (`the_device_gate_no_longer_reads_the_projection_arity`, a `gate_body()` scan for
`is_simple`; and the structural pin above). Only the first is about the gate expression — the
structural pin scans `build_device_ctr_config`'s shape and survives a gate rewrite untouched.
**T10 §4's trap now has a third member**: the gate body must not spell `prior_denom`,
`target_border_idx`, `ECtrType::Borders.as_i8()` **or `is_simple`**, not even in an inline
comment. T19's replacement comment says "projection-arity conjunct" in prose for that reason.
Also: **the gate body is now a single `matches!` over `from_i8`** — T23's delegation rewrite is
a near-identity transformation with no other conjunct left to preserve.

**5. `≥1 CTR split` is NOT a sufficient vacuity guard for a COMBINATION e2e** (T13 §2
generalised). `device_ctr_combo_fit_test`'s module doc had always claimed it "re-asserts" that
the trained model contains a ≥2-member projection; it did not. T19 made it executable (≥1 chosen
CTR split with `projection.cat_features().len() >= 2`) and prints the count. **T22's combination
arms must carry the same guard** — a fit that silently degrades to simple projections reproduces
`ctr_device_combo`'s predictions perfectly well.

**6. The retired `#[ignore]` rationale must never be restored verbatim.** Its claim — "this fit
runs on the CPU grower and the arm-routing assertion below would fail" — was false: run with
`--ignored` the test PASSED at `1.388e-17` in **0.01 s**, the CPU-fallback fingerprint, because
its only routing assertions were `oblivious_trees.len() == iterations` and the empty
non-symmetric/region lists. T19's rollback clause records a corrected rationale. This is the
cleanest worked example of R-8 in the phase and is written into the file's module doc.

**7. `CountingGpu` duplication count is now SEVEN** (`device_ctr_gate_test.rs:82-138` canonical,
`device_ctr_fit_test.rs`, `device_ctr_buckets_fit_test.rs`, `device_ctr_counter_fit_test.rs`,
`device_ctr_type_gate_test.rs`, `device_ctr_btmv_fit_test.rs`, `device_ctr_combo_fit_test.rs`).
**T19 adds NO new test binary** — `device_ctr_combo_fit_test` is already line 13 of
`run_device_tests.sh`, so the un-ignore converted a vacuous "1 ignored" into a real run without
touching the array (`git diff --stat run_device_tests.sh` empty; C-8, T24 owns it). T24's
registration list is unchanged: T10's, T12's, T13's and T16's four binaries.

**8. Nothing new for T24's pre-existing list; nothing moved either.** `erasing_op`
(`score_split.rs:374`) is still the only rocm clippy `error`; the rocm warning count is still
**18** and none names a T19-touched file (the 4 `doc list item without indentation` are
pre-existing in `kernels/grow_loop.rs`); `duplicated_attributes` is still at
`gpu_runtime/mod.rs:4535`/`:4567` (T19 inserts no code above it). Default-cpu `cb-backend --lib`
re-measured at **`227 passed; 60 failed; 2 ignored`** — identical to T18, a third observation
confirming T17's **59-60 range**.

## From T22 → **T23** (unblocked), **T24**, and TWO UNOWNED DEFECT CANDIDATES needing triage

T22 created `crates/cb-train/tests/device_ctr_combo_types_diff_test.rs` (DCTR-20): three
device-vs-CPU **split-sequence** differentials over the frozen `ctr_device_combo` corpus, all
green, all four assertions live. Production change: **comments only** (the R-20 paragraph in
`gpu_runtime/mod.rs`, `27 8`, executable surface unchanged). D-04 held exactly
(`4.483e-11` / `2.776e-17` / `1.388e-17` / `2.776e-17` / combo `2.082e-17`, `grows = 5` on all
five, against a baseline captured before the first edit); `run_device_tests.sh` **24 PASS /
0 FAIL**; `cb-backend --lib` rocm `277 passed / 0 failed / 2 ignored` (identical to T18/T19);
`cb-train --lib` `401 passed`; `cargo test -p cb-train` **111 targets, all ok**. The upstream
half of the chain is closed in the same place: `ctr_mixed_simple_vs_combo_oracle_test`
(`2 passed`) and `tensor_ctr_e2e_oracle_test` (`3 passed`).

**1. ⚠ R-20 IS STILL OPEN — now measured by the detector R-20 itself designated, at TWO
horizons. Do not let it be quietly closed.** MUT-1 (D-2's call site reverted to the unfiltered
`.max()`) leaves the split-sequence differential **byte-identical** at its shipped
configuration, AND byte-identical on a deliberately longer-horizon probe (all three arms at
`iterations = 20`: 40 level decisions per arm, 13 combination splits on the Buckets arm). The
one arm that fails at that horizon fails **identically with and without the mutation**, so it
is not attributable to D-2 (it is finding 4 below). ⇒ D-2 still has NO behavioural detector on
any committed fixture; the evidence remains `gpu_runtime::ctr_eligibility_test`'s unit +
source pins. **One hypothesis is now REFUTED**: T19 §5 proposed "a fixture whose combination's
bucket count much exceeds its members'" as the plausible discriminator — on `ctr_device_combo`
that ratio is already ~3× (simple 3/4 vs combined ≤12) and, because `phantom_max == 0` at
level 0 where a combination is always ineligible, the filtered and unfiltered `eligible_max`
genuinely DIFFER at every tree's level 0 (weights `(0.756, 0.707)` vs `(0.894, 0.866)`) — and
the greedy winner still never flips. The ratio alone is not the missing ingredient. All of
this is now written into the `mod.rs` R-20 comment in measured form.

**2. ⚠ `T22-OBS-1` — an UNOWNED, PRE-EXISTING device-vs-CPU divergence: ~1e-3 on a CTR fit's
CTR-FREE trees.** On any tree of a CTR fit whose greedy search chooses **zero** CTR splits,
the device and CPU leaf VALUES diverge by ~1e-3 while their split sequences stay identical:
`tree 23 → 1.069e-3`, `tree 27 → 1.280e-3` (Counter combos, 30 iters/depth 2), against ~1e-17
on every CTR-carrying tree. **It reproduces unchanged with `combinations_ctr = Borders`, i.e.
on the ALREADY-SHIPPED `ctr_device_combo` configuration** merely run to 30 iterations instead
of the fixture's 5 (`7.824e-4` / `1.223e-3` / `1.943e-3` / `1.296e-3` at trees 23/25/28/29).
Every committed device CTR fixture stops at 5 iterations, where every tree still carries a CTR
split — which is why nothing has caught it. **Correlate (stated as a correlate, not a
diagnosis)**: `device_has_ctr_split == false` ⇒ `fused_unit_fold == true`
(`crates/cb-train/src/boosting.rs:5665`), the branch that consumes the device's resident
`dev_tree.leaf_of` instead of the host CTR-aware `assign_leaf_over_ctr_columns` walk (T10 §1's
two-path split). The same trees are also the ones where the device leaves `level_kinds` empty
and the CPU does not. **This is neither T17's fallback nor T18's filter, so it has no owning
task in P1.** T22 did not patch it (its task text forbids it). It needs triage — plausibly a
P2/P3 item, but a decision, not silence. Containment: every T22 arm sits strictly below the
first CTR-free tree and each run PRINTS its CTR-free tree count.

**3. NEW, and it costs nothing to know: `ObliviousTree::level_kinds` is exercised DIFFERENTLY
by the two growers.** On an all-float tree the device leaves it EMPTY (the documented
single-kind fallback, SPEC-OH-31) while the CPU emits `[Float(0), Float(1)]`. Both decode to
the same tree, so T22 canonicalises to the DECODED sequence and additionally asserts the
precondition the fallback rests on (an empty vector must belong to a genuinely single-kind
tree — an empty one on an interleaved tree would be a real defect). **No prior test compares
`level_kinds` across the two growers**; any later task that does must canonicalise the same
way or it will get a false red.

**4. ⚠ `T22-OBS-2` — MATERIAL CORRECTION to the coordinator's own T22 guidance: a prior ≠ 0.5
is NECESSARY BUT NOT SUFFICIENT for a Buckets differential.** The consolidated T10 §2 + T16
entry offers two remedies (prior ≠ 0.5, or a partition-invariant projection). T22 took the
first, up front and in writing. It is **not enough at a long horizon**: at
`Buckets, Prior = 0.25`, 20 iters/depth 2, tree 12 level 1 the device picks
`([0,1], Buckets, b=0, border 11.999999)` and the CPU `([0,1], Buckets, b=1, border 0.999999)`
— T10 §2's signature, at a prior where the exact algebraic mirror
(`ctr(b0)+ctr(b1) = 1`) is gone. Reason: a prior ≠ 0.5 removes the mirror IDENTITY but not the
ordinal anti-monotonicity `bin(b0) + bin(b1) ≈ const`, and with ~12 buckets over 15 CTR bins
many threshold pairs still induce the SAME partition and therefore an exact score tie.
⇒ **for any FUTURE long-horizon Buckets differential, only the coordinator's other option — a
genuinely partition-invariant projection of the split set — is robust.** T22's shipped Buckets
arm is at 5 iterations, below the tie, per the ladder's "lowest rung that satisfies guard 4"
discipline; the 20-iteration run existed only as a MUT-1 probe. Verified independent of D-2
(identical failure with MUT-1 live and reverted).

**5. Guard 4 needed the escalation ladder for Counter, and rung 1 sufficed — at 20, not 10.**
Combination **Counter** chooses ZERO ≥2-member splits on this corpus at rungs 0-3 (5 and 10
iters, depth 2 and 3, and priors `0.0/0.25/1.0/2.0`), on BOTH arms — a search outcome, not a
device defect. Root cause, measured: the combination Counter column carries only the JOINT
FREQUENCY of the two cat columns, `corr(joint freq, y) = 0.163` on this fixture, against
`simple_ctr = Borders` columns that encode the target statistic directly and a
`(1 + count/maxCount)^-0.5` weight that penalises the combination's larger bucket count. A
`simple_ctr = Counter` diagnostic (30 iters/depth 3) still yields 0 combinations, so "Borders
dominates" is not the whole story. **`iterations = 20, depth = 2` → 24 CTR splits, 4 of them
≥2-member**, and that shipped. Guard 4 was never weakened. **General carry**: a CTR type that
carries no target signal (Counter, FeatureFreq) will not produce combination splits on a
small, near-uniform categorical corpus at short horizons — do not design a
combination-Counter test around a 5-iteration fixture.

**6. T17's `bucket_counts` combination fallback is PROVED CORRECT, by forcing it onto the
production path.** MUT-2 (restore the pre-T17 `member_bins.first()`-only fallback) is GREEN,
because the branch is production-unreachable (`col.bucket_count > 0` always). Rather than stop
there (T14 §5 / T18 §3: a green mutation can mean a vacuous test), T22 drove both complements
with `if false && col.bucket_count > 0`: **MUT-2b** (pre-T17 fallback forced onto the path)
turns **all three arms RED** with the exact failure shape T17's comment predicts — the
under-counted combination inflates every column's `cat_feature_weight` so CTR candidates start
beating floats (`device: 10 CTR splits (5 ≥2-member)` vs `cpu: 8 (3)`, first divergence a
FLOAT split at tree 0). **MUT-2c** (same forcing, T17's fold-all-members arm restored) is
**GREEN and byte-identical to the unmutated run on all three arms**. ⇒ T17's
`combine_projection_bins` fallback reproduces the production `bucket_count` exactly. No
previously-shipped test makes that statement.

**7. T23 is UNBLOCKED** (SPEC scenario 10 satisfied). T22 touches no gate expression and moves
no gate-state row: `device_ctr_combo_config_tests` `8 passed` with every row unmoved and
`boosting_ctr_gate_tests` `13 passed`, both unchanged from T19. T23 inherits exactly what T19
left — a single `matches!` over `from_i8`, and **eleven** source-scan pins to retire or update.

**8. T24: ONE new binary to register, and `CountingGpu` is now duplicated EIGHT times.**
`run_device_tests.sh` was **NOT** touched (C-8). T24's registration list is now FIVE binaries:
`device_ctr_buckets_fit_test` (T10), `device_ctr_counter_fit_test` (T12),
`device_ctr_type_gate_test` (T13), `device_ctr_btmv_fit_test` (T16) and
**`device_ctr_combo_types_diff_test`** (T22). The eighth `CountingGpu` copy lives in T22's file;
T22 also adds a `CountingCpu` sibling (same body, `inner: CpuRefRuntime`) so the CPU reference
arm's `grown == 0` / `accepted_begins == 0` is an observation rather than a type-level
assumption — a shape later device-vs-CPU differentials should copy.

**9. Pre-existing list: nothing new; ONE line-number correction.**
`clippy::duplicated_attributes` is at **`gpu_runtime/mod.rs:4542`/`:4574`** at committed
`HEAD 657b7dd` (T18/T19 recorded `:4535`/`:4567` against their uncommitted trees) — same
single diagnostic. `erasing_op` at `kernels/score_split.rs:374` is still the only `error` on
the rocm clippy lane, warning count still **18**; the 12 `cb-train` clippy test targets (T04)
and the default-cpu `cb-backend --lib` **59-60 range** (T17 §7) are untouched — T22's
`cb-backend` diff is comments only.

## From T21 → **T23** (no impact, but read §4), **T24** (DoD scenario 9 + a phase-level fact)

T21 discharged DCTR-03 without authoring a third one-hot × CTR test (C-18 honoured) and added
the two remaining acceptance-9 boundary pins plus a control to
`crates/cb-train/tests/device_ctr_type_gate_test.rs` (`7 passed`: T13's 4 + T21's 3).
Production change: **comment only** (`boosting.rs`, `48 7`, every changed line `//`-prefixed —
verified by diff filter); **zero `cb-backend` diff**. D-04 held exactly (`4.483e-11` /
`2.776e-17` / `1.388e-17` / `2.776e-17` / combo `2.082e-17`, `grows = 5` on all five, against a
baseline captured before the first edit); `run_device_tests.sh` **24 PASS / 0 FAIL** (Poisson
10.9×, no R-13 flake); `cb-backend --lib` rocm `277 passed / 0 failed / 2 ignored` (identical to
T18/T19/T22); `cb-train --lib` `401 passed`; `cargo test -p cb-train` **111 targets, all ok**
(identical to T22 — T21 adds no binary).

**1. WHICH CLAUSES ACTUALLY SURVIVE (T21 verified against the current source, not the plan's
snapshot).** `ctr_types_are_device_covered` is now a single `matches!` over `from_i8` — no
type-as-Borders, arity, target-border or prior conjunct — so **no P1 boundary lives inside it**.
The surviving CTR-side exclusions and their pins: `learning_folds_for_cycle == 1`
(`device_ctr_gate_test::multi_permutation_ctr_declines_to_device`, **P3**);
`one_hot_bins.is_empty()` (`device_fpp_composition_test::one_hot_x_ctr_still_declines`,
**retained by design, never inverts**); the type list (`boosting_ctr_gate_test.rs`,
`session_ctr_type_test`); `eval_sets.is_empty()` (T13's 2×2, **P3**);
`has_any_scorable_feature` (**NEW** `cat_only_ctr_pool_declines_to_device`, **P2/C-5**); and
`ctr_covered`'s `col.borders.len() + 1 == n_bins` (**NEW**
`non_15_border_count_ctr_pool_declines_to_device`, **P2/C-1**). ⇒ **acceptance scenario 9 is
complete: all five negatives have a passing, annotated test.**

**2. ⚠ MATERIAL, PHASE-LEVEL — "run the mutation for the clause you CLAIM" (T13 §1) is
sometimes IMPOSSIBLE, and T21 measured three cases of it.** Two of the five negatives are
**overdetermined by mutually-redundant guards**, so a single-clause mutation leaves the test
green and that is a property of the CODE, not blindness (T18 §3's mode, not T17 MUT-1c's):
  * **multi-permutation** — `learning_folds_for_cycle` is also passed to
    `begin_device_training` as `fold_count`, and every backend coverage mapper declines
    `fold_count != 1` (`session.rs:524/627/679/731/786`). Removing the host conjunct alone:
    GREEN. Removing the backend's view alone: GREEN. Removing **both**: RED at
    `left: 5, right: 0`. Both complements were driven before concluding.
  * **cat-only** — FOUR guards, all keyed on the single fact that a CTR pool with no float and
    no one-hot column has an EMPTY device feature axis: `has_any_scorable_feature`,
    `device_active`'s `device_n_bins > 0`, the session `begin` preamble
    (`n == 0 || n_features == 0 || n_bins == 0`) and `ctr_covered`'s shape check (`16 != 0`).
    Cumulative mutation of the first three: all GREEN. Driving all four was DECLINED
    deliberately — it yields `n_features == 0, n_bins == 0`, and a red from an empty problem
    says nothing about the boundary.
  ⇒ **the honest substitute, which T21 shipped: a CONTROL ARM through the same helper**
  (`unmodified_float_half_commits_to_device`, `grown = 5/5`), so each pin is a one-factor
  experiment whose assertion is provably sensitive to the attribute it varies. **Any later task
  writing a decline test should check for overdetermination FIRST and budget for a control
  arm** — and should not read a green mutation as "the code is right".

**3. DCTR-03's retention is now MEASURED, not argued.** Splitting the mandated joint mutation
into two rungs: with the **SPEC-OH-26 rejection alone** disabled the mixed pool TRAINS and
`grown = 0` in **0.05 s** (the retained `one_hot_bins.is_empty()` conjunct is what holds it
off the device); with **both** disabled it **COMMITS at `grown = 5`** in 52.81 s. The conjunct
the research called provably dead is one deletion away from being the only thing standing
between a mixed pool and an untested device path. Both numbers are written into the
`boosting.rs` comment above the conjunct. **Do not delete it.**

**4. `non_15_border_count` degrades SILENTLY, which is worth knowing.** Under MUT-3 (the
`borders.len() + 1 == n_bins` conjunct neutralised in both `ctr_covered` closures) the fit
commits (`grown = 5/5`) and produces **zero CTR splits** — the CTR columns' 16 bins forced into
an 8-bin histogram line simply never win a candidate. No error, no warning; only the
`≥1 CTR split` vacuity guard sees it. That guard is therefore load-bearing in this file, not
decoration.

**5. Process — report BEFORE asserting, generalised from T20 §2.** MUT-3's first run failed on
the vacuity guard and hid `grown = 5`, the number the completion criterion asks for. T21's
helper now PRINTS `grown` / `ctr_splits` / tree count before every guard, with an in-source
comment saying not to "fix" the ordering. Any decline test with non-vacuity guards ahead of its
route assertion has the same problem.

**6. T23 is unaffected; T24's registration list is unchanged.** T21 changes no gate expression
and moves no gate-state row (`device_ctr_combo_config_tests` `8 passed`, rows unmoved;
`boosting_ctr_gate_tests` `13 passed`), so T23 still inherits **eleven** source-scan pins. T21
adds **no** new binary (its three tests join `device_ctr_type_gate_test`, already on T24's
list), and `CountingGpu` duplication stays at **EIGHT**. One preservation note for T23: T21's
new `boosting.rs` comments are `//`-prefixed and sit OUTSIDE `gate_body()`, which is what keeps
`code_lines_mentioning` from counting them.

**7. Pre-existing list: nothing new, one addition of an already-true fact.**
`cargo clippy -p cb-train --no-default-features --features rocm --test device_ctr_gate_test`
emits `clippy::type_complexity` on `device_ctr_gate_test.rs:140` (`load_inputs`) — a WARNING,
verified pre-existing at `HEAD` (same line, same signature), distinct from T04's 12 error-level
targets. `erasing_op` (`score_split.rs:374`), `duplicated_attributes`
(`gpu_runtime/mod.rs:4542`/`:4574`) and the default-cpu `cb-backend --lib` 59–60 range are all
unmoved — T21's `cb-backend` diff is empty. **`T22-OBS-1` / `T22-OBS-2` were read and
deliberately NOT chased** (unowned, out of T21's scope); they still need triage.

## Coordinator disposition of T22-OBS-1 → **T24** (USER DECISION, 2026-08-10)

T22 localised **T22-OBS-1**: a **pre-existing, unowned ~1e-3 device-vs-CPU divergence in
leaf values on a CTR fit's CTR-FREE trees**, reproducing on the shipped
`combinations_ctr = Borders` config at 30 iterations and correlating with `fused_unit_fold`
(`boosting.rs:5665`). It is **not caused by this phase**, and no P1 acceptance test is
affected — P1's fixtures run at 5–20 iterations, where it stays under the ≤1e-5 bar.

**The user was asked and ruled: RECORD ONLY, decide later.** No bug chase, no spec, no
plan, no fix in this phase.

⇒ **T24 must carry OBS-1 into the phase completion summary** as an explicit open item, with
its reproduction (config + 30 iterations), the `fused_unit_fold` correlation, and the fact
that it is pre-existing and was deliberately not patched. Do **not** present the phase as
having no open findings, and do **not** attempt a fix. Same treatment for **T22-OBS-2**
(a prior ≠ 0.5 is necessary but NOT sufficient for a Buckets differential — the b=0/b=1 tie
survives via ordinal anti-monotonicity, so only a partition-invariant projection is robust
at long horizons).
