# Phase 14: Comprehensive Kaggle CUDA Speed Benchmark + Parity Sign-Off - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-04
**Phase:** 14-comprehensive-kaggle-cuda-speed-benchmark-parity-sign-off
**Areas discussed:** BENCH-03 sign-off criterion, Workload matrix, Aggregate vs re-run, Correctness gate scope

---

## BENCH-03 sign-off criterion (what "closes the >20× gap" must show)

| Option | Description | Selected |
|--------|-------------|----------|
| vs host-CPU ≥20× only | Passes if device beats pre-Phase-10 host-light CPU baseline ≥20×; CatBoost GPU informational only | |
| Also within X× of CatBoost GPU | Adds a hard second gate: device within e.g. ≤2–3× of official CatBoost GPU | |
| Both, but CatBoost-GPU informational | ≥20× vs host-CPU is the hard gate; head-to-head vs CatBoost GPU recorded/discussed but shortfall documented not blocking | ✓ |

**User's choice:** Both, but CatBoost-GPU informational
**Notes:** Milestone goal is closing our own >20× host-light gap (fully in our control; Phase 12 already 30–42×). Matching a mature C++/CUDA library's absolute throughput is a stretch, not the definition of done → CatBoost-GPU head-to-head is informational context. (D-01)

---

## Workload matrix (datasets & configs)

| Option | Description | Selected |
|--------|-------------|----------|
| Real named + synthetic large-n | Add Higgs/Epsilon alongside synthetic across families at depth-6 | |
| Synthetic large-n only | Reuse bench/generator.py large-n synthetic across already-timed families; no external data staging | ✓ |
| Real named datasets only | Higgs/Epsilon at CatBoost-standard configs, drop synthetic | |

**User's choice:** Synthetic large-n only
**Notes:** No external dataset staging; fully reproducible. Timing stays valid on synthetic random data (throughput is data-shape-driven). Higgs/Epsilon stays deferred as a post-milestone stretch. (D-02)

---

## How the numbers are produced (aggregate vs re-run)

| Option | Description | Selected |
|--------|-------------|----------|
| One fresh comprehensive re-run | Single new Kaggle notebook re-times whole matrix (device/host-CPU/CatBoost-GPU) in one session | |
| Aggregate + add CatBoost-GPU only | Roll up committed per-phase result.json; new run adds only the missing official-CatBoost-GPU timing | ✓ |

**User's choice:** Aggregate + add CatBoost-GPU only
**Notes:** Saves GPU time. Accepted cost = mixed-session provenance; mitigated by labeling each number's source run (hardware, date). (D-03)

---

## Correctness gate scope (GPUT-14 / RESULTS.md backfill)

| Option | Description | Selected |
|--------|-------------|----------|
| Re-confirm correctness + close GPUT-14 | Run oracle as blocking gate, fill TBD RESULTS.md table, formally flip GPUT-14 satisfied | |
| Assume green, speed only | Correctness already established by Phase 12/13 P100 runs; oracle runs as pre-flight only; do not own GPUT-14 backfill | ✓ |

**User's choice:** Assume green, speed only
**Notes:** Oracle still runs as a blocking pre-flight (SC-2) but Phase 14 is scoped to the BENCH-03 speed sign-off. GPUT-14 formal close + the still-TBD depth-1/depth-6 RESULTS.md table are surfaced as out-of-scope standing debt for milestone-close audit. (D-04)

---

## Claude's Discretion

- Official CatBoost GPU parameter-matching (depth/iters/lr/bootstrap/grow-policy/border_count) against CatBoost docs + benchmark.py; document any config that can't be matched.
- BENCH-03 deliverable format & location (extend bench/RESULTS.md vs new bench/BENCH-03-SIGNOFF.md; coverage/speed matrix layout; JSON stitching) — must carry per-number source provenance.
- Which per-phase families become matrix rows beyond depth-6 RMSE/Logloss (bounded by which BENCH-02 numbers exist).

## Deferred Ideas

- Real named-dataset (Higgs/Epsilon) head-to-head — declined for Phase 14 (D-02); post-milestone stretch.
- Hard "within X× of CatBoost GPU" parity gate — declined (D-01); revisit only if absolute-throughput parity is later wanted.
- Formally closing GPUT-14 + backfilling the TBD RESULTS.md oracle table — out of scope (D-04); milestone-close audit / follow-up.
- One fresh comprehensive single-session re-run — declined (D-03) to save GPU time.
