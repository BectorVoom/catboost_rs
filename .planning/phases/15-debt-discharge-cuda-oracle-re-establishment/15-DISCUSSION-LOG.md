# Phase 15: Debt Discharge & CUDA Oracle Re-establishment - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-05
**Phase:** 15-debt-discharge-cuda-oracle-re-establishment
**Areas discussed:** Hazard fix-vs-retire policy, Aggregate run assembly, BENCH-03 recompute protocol, Evidence artifacts & bookkeeping

---

## RV-13-01..04 Hazard Disposition (HARD-03)

| Option | Description | Selected |
|--------|-------------|----------|
| Fix 03, oracle-retire 01/02/04 | Fix the crash guard; guard/assert 01/02/04 + record unreachable evidence | |
| Fix all four with demonstrating oracles | Real reproducing oracle for each, fix, prove | ✓ |
| Retire all four with recorded evidence | Document as unreachable, defer real fixes to device-grow wiring | |

**User's choice:** Fix all four with demonstrating oracles.
**Notes:** The four hazards are training-derivative / grow-path divergences, latent because families decline `Ok(None)`→CPU; no v1.2 phase (incl. Phase 19 inference) reaches them. User chose the thorough option to avoid carrying latent numeric debt into a future device-grow milestone. Baked-in scope guard (not re-asked): oracles are **unit/kernel-level direct invocations**, NOT an e2e device-grow wire-up, keeping the phase inside its no-grow-seam boundary; the hazard oracles ride in the same single Kaggle CUDA session as HARD-01.

---

## Aggregate GPUT-14 Run Assembly (HARD-01)

| Option | Description | Selected |
|--------|-------------|----------|
| Single combined kernel session | One Kaggle notebook, one P100/driver/seed, one verdict + one JSON | ✓ |
| Separate jobs aggregated by script | Multiple Kaggle jobs combined via aggregate.py (mixed-session provenance) | |

**User's choice:** Single combined kernel session (recommended).
**Notes:** Taken literally as what makes it "one authoritative row, no stitched gaps" — the opposite of the current BENCH-03's explicit mixed-session aggregation. Correctness stays a blocking pre-gate to any timing.

---

## BENCH-02 Rows + BENCH-03 Recompute Protocol (HARD-02)

| Option | Description | Selected |
|--------|-------------|----------|
| Same host-light baseline, same session, warm/median | Keep pre-Phase-10 host-light CPU baseline; add depth-1/6 rows in the correctness session | ✓ |
| Same baseline, separate speed session | Host-light baseline but speed rows in own Kaggle session | |
| Re-baseline against official CatBoost CPU | Drop host-light, re-baseline vs official CatBoost CPU | |

**User's choice:** Same host-light baseline, same session, warm/median (recommended).
**Notes:** Re-baseline vs official CatBoost CPU explicitly deferred to Phase 21 (adoption benchmark) — doing it here would conflate internal device≫CPU evidence with the adoption story. `catboost_gpu_s` stays informational; Region stays N/A.

---

## Evidence Artifacts & Requirement Bookkeeping

| Option | Description | Selected |
|--------|-------------|----------|
| Update in place, flip GPUT-14, clear Known Gaps | Rewrite bench sign-off/RESULTS in place; 15-EVIDENCE.md; flip GPUT-14 + clear two Known Gaps | ✓ |
| New phase-15 artifacts, leave v1.1 docs historical | Fresh sign-off/evidence files; keep old as historical with cross-link | |

**User's choice:** Update in place, flip GPUT-14, clear Known Gaps (recommended).
**Notes:** Rewrite `bench/BENCH-03-SIGNOFF.md` + `bench/RESULTS.md` with real un-stitched numbers (no fabrication — fill TBD cells from real session output); per-hazard evidence in `15-EVIDENCE.md`; on completion flip GPUT-14 + HARD-01/02/03 to satisfied and clear the two MILESTONES Known Gaps + STATE Deferred Items.

## Claude's Discretion

- Exact notebook structure, JSON schema, aggregate.py reuse-vs-replace (must yield one single-session authoritative record).
- Precise deterministic tie-break rule for RV-13-01/04 (oracle must prove CPU-equivalence).
- median-of-N choice (N=3 vs N=5) within the Kaggle time budget.

## Deferred Ideas

- Wiring the e2e device-grow seam (makes RV-13-01/02/04 live through the real grow loop) — future milestone.
- Re-baseline speed vs official CatBoost CPU — Phase 21.
- RV-13-05..09 (efficiency/cleanup debt from the same Phase-13 review, not named in HARD-03) — hardening backlog; fix only if subsumed by an 01/02/04 fix (e.g., RV-13-07 shares the RV-13-01 sort site).
