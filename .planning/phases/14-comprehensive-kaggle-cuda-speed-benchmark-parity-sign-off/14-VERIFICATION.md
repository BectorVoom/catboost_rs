---
phase: 14-comprehensive-kaggle-cuda-speed-benchmark-parity-sign-off
verified: 2026-07-05T01:36:48Z
status: passed
score: 9/9 must-haves verified (plan-level); ROADMAP SC1/SC3 scope caveat formally overridden by human sign-off (2026-07-05)
behavior_unverified: 0
overrides_applied: 1
overrides:
  - sc: "SC1 / SC3 — 'aggregating the per-phase speed checks (BENCH-02) from Phases 10–13'"
    decision: "ACCEPTED AS DELIVERED. Human (echinops27@gmail.com) formally ratified the A4/D-03/D-04 scope-narrowing on 2026-07-05: 'aggregate whichever per-phase BENCH-02 JSON is actually committed (Phase 12 + Phase 13 only)'. Phase 14 / BENCH-03 stands complete as delivered — the >=20x-vs-host-CPU hard gate (D-01) is fully proven for all 12 committed rows (23.9x–42.1x), and CUDA correctness (Part A) passed ALL-PASS before any speed number."
    standing_debt: "Phase 10 (depth-1) and Phase 11 (depth-6) BENCH-02 Kaggle runs were never executed; GPUT-14 remains Pending. These stay as explicitly-flagged milestone-close debt (see bench/BENCH-03-SIGNOFF.md 'Standing debt — NOT closed here' + the bench/RESULTS.md TBD tables). Per D-04 they are OUT OF SCOPE for this speed-only sign-off and are deferred to the v1.1 milestone-close audit or a dedicated follow-up."
    rationale: "The scope decision was made once during Phase-14 discuss-phase (14-DISCUSSION-LOG.md 'Correctness gate scope' → 'Assume green, speed only'; 14-RESEARCH.md A4) and is transparently disclosed in the sign-off document. This override formally records that decision against the ROADMAP contract so the literal 'Phases 10–13' wording no longer blocks phase completion."
human_verification: []
---

# Phase 14: Comprehensive Kaggle CUDA Speed Benchmark + Parity Sign-Off Verification Report

**Phase Goal:** The device-resident path demonstrably closes the >20× gap via a comprehensive final sign-off that AGGREGATES the per-phase speed checks (BENCH-02) from Phases 10–13, with CUDA correctness gated before any speed number.
**Verified:** 2026-07-05T01:36:48Z
**Status:** passed (ROADMAP SC1/SC3 scope caveat formally overridden by human sign-off — see frontmatter `overrides`)
**Re-verification:** No — initial verification

## Summary

Every artifact this phase's three plans actually committed to build was independently re-executed (not trusted from SUMMARY.md) and passes: the offline aggregator prints the claimed 12-row matrix and `BENCH-03: PASS`, its 9-test suite is green, the Kaggle driver compiles/imports/exposes the frozen CatBoost-GPU config, its 6-test `gen()` suite is green, the committed `bench03-result.json` shows `correctness_verdict: ALL-PASS` / `catboost_gpu_verdict: OK` with Region cells `N/A`, `BENCH-03-SIGNOFF.md` contains the verdict banner + 12-row matrix + provenance + standing-debt section, and `RESULTS.md` gained exactly one cross-link line with zero TBD-table edits. Zero files under `crates/` changed across all Phase-14 commits, matching the "bench-only phase" framing.

However, goal-backward tracing of the ROADMAP's own success-criteria wording surfaced a real, material gap that was NOT invented by this phase but was inherited and knowingly left unresolved: **Phase 10's depth-1 BENCH-02 speed number and Phase 11's depth-6 BENCH-02/GPUT-14 correctness+speed numbers were never actually recorded** — `bench/RESULTS.md` still shows both sections as "_No authoritative Kaggle CUDA run recorded yet_" / `TBD`, and `.planning/REQUIREMENTS.md` line 93 still lists `GPUT-14 | Phase 11 ... | Pending`. Phase 14's aggregator (`aggregate.py`) therefore stitches only **Phase 12 + Phase 13** (the only two committed JSONs — confirmed by `find bench -maxdepth 1 -type d`), not "Phases 10–13" as ROADMAP SC1/SC3 literally state. This was an explicit, contemporaneously-documented user decision (14-DISCUSSION-LOG.md "Correctness gate scope" section, `A4` in 14-RESEARCH.md, prohibitions baked into all 3 PLAN.md frontmatters) — not a hidden defect — and the delivered sign-off transparently flags it in its own "Standing debt — NOT closed here" section rather than concealing it. Because this narrows a literal ROADMAP success criterion without a formally recorded override, it is surfaced here for an explicit human decision rather than silently passed or silently failed.

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria — the non-negotiable contract)

| # | Truth (ROADMAP SC) | Status | Evidence |
|---|---|---|---|
| 1 | The Phase-10 Kaggle CUDA harness times official CatBoost GPU vs catboost-rs across the full workload matrix, **aggregating the per-phase speed checks (BENCH-02) from Phases 10–13** into one comprehensive comparison. | ⚠️ PARTIAL — see Human Verification #1 | `aggregate.py` correctly aggregates the ONLY two committed BENCH-02 JSONs that exist (`bench/phase12_cuda_oracle/bench02-result.json`, `bench/phase13_cuda_oracle/result.json`); Phase-10 (depth-1) and Phase-11 (depth-6) sections of `bench/RESULTS.md` are still literally `TBD` / "No authoritative ... run recorded yet" (lines 71-84, 140-153) — those two phases' Kaggle CUDA runs were never executed. `find bench -maxdepth 1 -type d` shows no `phase10_*`/`phase11_*` directory exists. This is a real gap vs the literal SC1 wording, but is a known, documented, user-approved scope decision (14-RESEARCH.md line 310 `A4`; 14-DISCUSSION-LOG.md "Correctness gate scope" → "Assume green, speed only"), not something Phase 14 fabricated or hid. |
| 2 | The correctness oracle is re-confirmed on the CUDA backend (≤1e-4 vs Rust CPU path) as a **blocking gate before any speed number is reported**, reusing the authoritative Phase-10 CUDA oracle. | ✓ VERIFIED | `bench/phase14_cuda_signoff/bench03-result.json`: `"correctness_verdict": "ALL-PASS"`, 5 CUDA families / 44 device self-oracle tests, 0 failed (re-read directly from the committed JSON, not from SUMMARY narration). `oracle.py` structurally blocks: `grep -n "sys.exit(2)"` at lines 140 and 205, both BEFORE the Part C CatBoost-GPU timing code (line ~225+); the correctness roll-up gates `cat_ok`/Part-C entry. |
| 3 | The device-resident path demonstrably closes the >20× gap (BENCH-03): a documented, signed-off **comprehensive final** speed-parity result vs official CatBoost GPU, measured against the pre-Phase-10 host-light baseline, **aggregating every per-phase speed measurement** into the milestone-closing sign-off. | ⚠️ PARTIAL — see Human Verification #1 | The >=20x-vs-host-CPU gate itself is fully and correctly proven: independently re-ran `python3 bench/phase14_cuda_signoff/aggregate.py` — 12 rows printed, every row's speedup 23.888×–42.080×, `BENCH-03: PASS` (exit 0). `BENCH-03-SIGNOFF.md` references the pre-Phase-10 host-light baseline (`.planning/notes/gpu-training-host-light-root-cause.md`) and states the reversal correctly. BUT "aggregating every per-phase speed measurement" is not literally true — only Phase 12 + Phase 13's measurements exist to aggregate; Phase 10/11 never produced one. Same caveat as truth #1. |

**Score:** 1/3 roadmap SCs cleanly VERIFIED; 2/3 carry a documented, non-hidden scope caveat requiring human sign-off (not a code defect).

### Plan-Level Must-Haves (14-01 / 14-02 / 14-03 PLAN.md frontmatter) — re-executed independently

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | `aggregate.py` prints all 12 rows (6 P12 + 6 P13) | ✓ VERIFIED | Ran `python3 bench/phase14_cuda_signoff/aggregate.py` directly — 12 data rows printed (6 tagged P12, 6 tagged P13). |
| 2 | Every row's speedup float-cast + flagged >=20x; verdict `BENCH-03: PASS` | ✓ VERIFIED | Same run: min speedup 23.888×, max 42.080×, all `yes`; final line `BENCH-03: PASS`; exit code 0. |
| 3 | `aggregate_test.py` passes offline, asserts 12 rows + all >=20.0 | ✓ VERIFIED | Ran `python3 -m pytest bench/phase14_cuda_signoff/ -q` — 9 passed (3 in `aggregate_test.py`, 6 in `oracle_gen_test.py`). Read `aggregate_test.py` source: asserts `len(rows) == 12`, `isinstance(speedup, float)`, `speedup >= 20.0`, and the Phase-13-alone-yields-6-rows nested-schema proof — all against the REAL committed files (`PHASE12_JSON`/`PHASE13_JSON` imported from `aggregate.py`, not mocked). |
| 4 | `oracle.py` compiles/imports with no GPU; run body guarded by `__main__` | ✓ VERIFIED | Ran `python3 -m py_compile` (success) and a spec-based import that asserted `hasattr(m,'gen')`/`hasattr(m,'main')` with zero side effects — confirms no run triggered on import. |
| 5 | `gen(n)` reproduces Rust `bench_grow_speed_test.rs::gen()` bit-for-bit | ✓ VERIFIED | Read `crates/cb-train/tests/bench_grow_speed_test.rs` lines 43-65 directly: `h = i.wrapping_mul(2_654_435_761).wrapping_add(f.wrapping_mul(40_503))`, `(h % nbins) as f32`, target `a + 0.5*b > nbins*0.75`. Compared line-by-line against `oracle.py`'s `gen()` (uint64 wraparound arithmetic, identical hash/threshold) — exact match. `oracle_gen_test.py::test_spot_check_hash_formula` independently re-derives `(i*2654435761 + f*40503) % 32` for 20 (i,f) pairs and asserts equality — ran and passed. |
| 6 | Part A is a BLOCKING pre-flight that aborts before Part C on any failure | ✓ VERIFIED | `grep -n "sys.exit(2)"` at lines 140, 205 in `oracle.py`, both preceding the Part-C CatBoost-GPU timing block (line ~225 onward). |
| 7 | `kernel-metadata.json` declares Phase-14 id with GPU+internet enabled | ✓ VERIFIED | Parsed directly: `enable_gpu=True`, `enable_internet=True`, `id="yensen2/catboost-rs-phase14-cuda-signoff"`. |
| 8 | Human-gated Kaggle run: Part A ALL-PASS before Part C; real depthwise `catboost_gpu_s`; Region `N/A` | ✓ VERIFIED | Read committed `bench03-result.json` directly: `correctness_verdict: "ALL-PASS"`; `bench03.runs`: 3 depthwise rows each with a real numeric `catboost_gpu_s` (0.6733/0.7052/0.8181) and `grow_policy_used: "Depthwise"`; 3 region rows each `catboost_gpu_s: null`, `grow_policy_used: "N/A"`. |
| 9 | `BENCH-03-SIGNOFF.md` verdict banner + provenance + standing-debt section; `RESULTS.md` gets exactly one cross-link line, no TBD backfill | ✓ VERIFIED | Read `BENCH-03-SIGNOFF.md` in full: `# BENCH-03: PASS` banner present, 12-row matrix with per-row provenance labels (`Phase-12 P100 (2026-07-04)` / `Phase-13 P100 (2026-07-04)` / `Phase-14 P100 (2026-07-05)`), "Standing debt — NOT closed here" section names both `GPUT-14` and the `bench/RESULTS.md` TBD table explicitly. `git show 8e5acb2 -- bench/RESULTS.md` confirms exactly `1 insertion(+)`, zero deletions/modifications to any existing TBD table. |

**Score:** 9/9 plan-level must-haves VERIFIED.

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `bench/phase14_cuda_signoff/aggregate.py` | Offline schema-branching aggregator, `def load_rows` | ✓ VERIFIED | Exists, 184 lines, substantive (docstrings + schema branch + float cast + ge20x flag), wired (imported by `aggregate_test.py`, invoked directly by `BENCH-03-SIGNOFF.md`'s methodology), data-flow confirmed by direct execution against real committed JSON — real numbers, not stubs. |
| `bench/phase14_cuda_signoff/aggregate_test.py` | Offline unit test, `def test_` | ✓ VERIFIED | 81 lines, 3 real tests against real committed files, all pass. |
| `bench/phase14_cuda_signoff/oracle.py` | Kaggle CUDA driver, `def gen` | ✓ VERIFIED | Compiles, imports safely, `gen()` verified bit-exact vs Rust source, Part A/Part C structure present and correctly gated. |
| `bench/phase14_cuda_signoff/kernel-metadata.json` | Fresh kernel descriptor, `catboost-rs-phase14` | ✓ VERIFIED | Parses; GPU+internet enabled; Phase-14 id. |
| `bench/phase14_cuda_signoff/oracle_gen_test.py` | Offline `gen()` reproduction test | ✓ VERIFIED | 78 lines, 6 tests, all pass, includes an independent hash-formula spot-check (not just shape checks). |
| `bench/phase14_cuda_signoff/bench03-result.json` | Kaggle run output: correctness + CatBoost-GPU timings | ✓ VERIFIED | Committed, `correctness_verdict=ALL-PASS`, `catboost_gpu_verdict=OK`, real depthwise timings, Region `N/A` — matches the human-gated-run claim in 14-03-SUMMARY.md. |
| `bench/BENCH-03-SIGNOFF.md` | Milestone-closing sign-off, `BENCH-03: PASS` | ✓ VERIFIED | 148 lines, contains verdict banner, matrix, provenance, divergences, standing-debt section. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `aggregate.py` | `bench/phase12_cuda_oracle/bench02-result.json` | root `.runs[]` | ✓ WIRED | `PHASE12_JSON` constant resolves via `__file__`; `load_rows` reads and yields 6 rows (verified by direct execution). |
| `aggregate.py` | `bench/phase13_cuda_oracle/result.json` | nested `.bench02.runs[]` | ✓ WIRED | `PHASE13_JSON` constant; schema branch fires the nested path (proven both by execution output tagging `P13` rows and by the dedicated `test_phase13_nested_schema_resolves` unit test). |
| `oracle.py` | `crates/cb-train/tests/bench_grow_speed_test.rs` | `gen()`/`params()` reproduction | ✓ WIRED | Verified bit-for-bit by direct source comparison (see truth #5 above), not merely grep'd for keyword presence. |
| `oracle.py` | `bench/phase13_cuda_oracle/oracle.py` | Part A structure reused | ✓ WIRED | Part A code block (family filters, `cargo test --release --no-default-features --features cuda`) present, structurally identical pattern to Phase-13's proven driver. |
| `BENCH-03-SIGNOFF.md` | `bench/phase14_cuda_signoff/aggregate.py` | device/host-CPU/speedup/>=20x columns | ✓ WIRED | Matrix numbers in the sign-off doc match byte-for-byte the numbers this verification's own `aggregate.py` re-run produced (cross-checked row by row). |
| `BENCH-03-SIGNOFF.md` | `bench/phase14_cuda_signoff/bench03-result.json` | informational CatBoost-GPU column | ✓ WIRED | `catboost_gpu_s` values in the sign-off (0.6733/0.7052/0.8181, Region `N/A`) match the committed JSON exactly. |
| `BENCH-03-SIGNOFF.md` | `.planning/notes/gpu-training-host-light-root-cause.md` | host-light baseline reference | ✓ WIRED | File exists, is linked, and the sign-off's narrative of the reversal (host-light >20x slower → 24-42x faster) is consistent with that note's own root-cause description. |

### Behavioral Spot-Checks (independently re-run by this verifier, not trusted from SUMMARY.md)

| Behavior | Command | Result | Status |
|---|---|---|---|
| Aggregator prints 12 rows + PASS | `python3 bench/phase14_cuda_signoff/aggregate.py` | 12 rows, `BENCH-03: PASS`, exit 0 | ✓ PASS |
| Aggregator unit tests | `python3 -m pytest bench/phase14_cuda_signoff/ -q` | 9 passed | ✓ PASS |
| oracle.py compiles + imports safely | `python3 -m py_compile ...` + spec-import | success, `gen`/`main` present, no side effects | ✓ PASS |
| gen() offline reproduction tests | `python3 -m pytest bench/phase14_cuda_signoff/oracle_gen_test.py -x -q` | 6 passed | ✓ PASS |
| kernel-metadata.json valid | `python3 -c "import json; ..."` | `enable_gpu=True enable_internet=True id=...phase14...` | ✓ PASS |
| No crate files touched by Phase 14 | `git diff --stat <pre-14-01>..<post-14-03> -- crates/` | empty output | ✓ PASS |
| RESULTS.md gained exactly 1 line | `git show 8e5acb2 -- bench/RESULTS.md` | `+1` line, no deletions | ✓ PASS |

### Probe Execution

No `scripts/*/tests/probe-*.sh` convention applies to this phase; the PLAN-declared automated `<verify>` commands (aggregate.py run, pytest suites, py_compile, JSON assertions) were used instead and are reported under Behavioral Spot-Checks above. Step 7c: SKIPPED (no probe-* scripts declared or discovered for this phase).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| BENCH-03 | 14-01, 14-02, 14-03 | Comprehensive final speed-parity sign-off vs official CatBoost GPU, aggregating per-phase BENCH-02 | ✓ SATISFIED (with caveat) | `.planning/REQUIREMENTS.md` line 105 marks `BENCH-03 | Phase 14 | Complete`. The >=20x-vs-host-CPU hard gate (D-01) is genuinely and correctly proven for all 12 available rows. The "aggregating... from Phases 10–13" framing is only partially true (see Human Verification #1) — this is flagged, not silently accepted. |

No orphaned requirements: `grep -n "Phase 14" .planning/REQUIREMENTS.md` returns only the BENCH-03 row.

**Adjacent requirement sanity-check (not owned by this phase, but load-bearing to its goal):** `GPUT-14` (`.planning/REQUIREMENTS.md` line 93) is still `Pending`, mapped to "Phase 11 (standing — enforced onward through 13)". Phase 14's own must-haves explicitly prohibit flipping this (D-04) and the sign-off explicitly flags it as standing debt — so this is NOT a Phase-14 regression, but it does mean the milestone-level claim "the device-resident path demonstrably closes the >20x gap" rests on a correctness gate (GPUT-14) that has never been formally closed for Phase 11 depth-6 on Kaggle CUDA. Surfaced for milestone-close audit, per the sign-off doc's own recommendation.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| `bench/phase14_cuda_signoff/oracle.py` | 312 | `"TBD"` string literal | ℹ️ INFO | Legitimate do-not-fabricate fallback for a hypothetical failed Part-C fit (never triggered this run — Region already uses a distinct `"N/A"` path); not a stub, not unreferenced debt. |
| `bench/BENCH-03-SIGNOFF.md` | 18, 132-134 | `TBD` references | ℹ️ INFO | Documentation correctly describing the still-open `bench/RESULTS.md` TBD tables (Phase 10/11) — intentional, referenced, not concealed. |

No `FIXME`/`XXX`/unresolved `TODO`/`HACK`/`PLACEHOLDER` markers found in any Phase-14-created file. No blocker-level anti-patterns.

### Human Verification Required

#### 1. Scope-narrowing of ROADMAP SC1/SC3 ("aggregating... from Phases 10–13")

**Test:** Decide whether the already-documented A4/D-03/D-04 scope decision (aggregate only the two committed BENCH-02 JSONs — Phase 12 + Phase 13 — and leave Phase 10 (depth-1) / Phase 11 (depth-6) BENCH-02 + GPUT-14 as explicit standing debt) is accepted as the final, correct interpretation of "comprehensive final aggregate" for this milestone.

**Expected:** Either (a) add a formal `overrides:` entry to this VERIFICATION.md accepting the narrower scope (the decision is already fully documented in `14-DISCUSSION-LOG.md` and `14-RESEARCH.md` A4 — this just needs an `accepted_by`/`accepted_at` to close the audit trail), or (b) route this to `/gsd-audit-milestone` / a dedicated follow-up phase to actually run the Phase-10/Phase-11 Kaggle CUDA notebooks and backfill `bench/RESULTS.md` + flip `GPUT-14`, or (c) update ROADMAP.md's Phase-14 SC1/SC3 wording to match the delivered scope so future audits don't re-flag this.

**Why human:** This is a policy/scope call about what "comprehensive final" is allowed to mean when 2 of the 4 referenced phases never produced a number — not something a script or grep can adjudicate. The underlying facts (Phase 10/11 RESULTS.md sections still TBD, GPUT-14 still Pending) are independently confirmed by this verification and are not in dispute; only the disposition (accept / reopen / reword) requires a human decision.

## Gaps Summary

No code-level gaps were found in what Phase 14 actually built — every artifact, script, test, and document this phase's three plans committed to deliver was independently re-executed and verified correct, substantive, and wired. The one open item is a scope/documentation-contract question inherited from an earlier, already-discussed decision (not a defect introduced by this phase's execution): ROADMAP.md's Phase-14 success criteria literally say "from Phases 10–13," but only Phase 12 and Phase 13 ever produced a committed BENCH-02 number to aggregate. This was surfaced transparently by the phase's own deliverable (`BENCH-03-SIGNOFF.md`'s "Standing debt — NOT closed here" section) rather than hidden, and is routed here as `human_needed` rather than `gaps_found` because there is no actionable code-level fix within Phase 14's scope — closing it requires either a policy decision (override/reword) or reopening Phase 10/11's human-gated Kaggle runs, both of which are decisions for the developer, not further Phase-14 execution.

---

*Verified: 2026-07-05T01:36:48Z*
*Verifier: Claude (gsd-verifier)*
