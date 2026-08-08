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
