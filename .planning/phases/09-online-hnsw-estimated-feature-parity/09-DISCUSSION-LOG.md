# Phase 9: Online HNSW Estimated-Feature Parity - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-23
**Phase:** 9-online-hnsw-estimated-feature-parity
**Areas discussed:** Construction-order fidelity model, Port scope / fidelity boundary, Existing exact calcer disposition, Evidence & verification strategy

---

## Construction-order fidelity model

Discussion-time finding surfaced to the user: a grep of `library/cpp/online_hnsw/base/`
and `library/cpp/hnsw/` found NO RNG (no `TFastRng`/`Shuffle`/`RandomNumber`; level
growth is deterministic via `AddNewLevelIfLastIsFull`). This contradicts the ROADMAP's
"RNG-driven build order" framing.

| Option | Description | Selected |
|--------|-------------|----------|
| Source-faithful, both paths | Lock target = deterministic AddItem-order replication + byte-identical tie-break/comparator; no RNG work | |
| Neighbors-only, defer tie-break | Replicate insertion order + search; treat tie-breaking as a finding to resolve only if a tie occurs | |
| Keep RNG framing | Distrust the grep; researcher proves/disproves RNG (incl. NHnsw routines + GetLevelSizes) before committing | ✓ |

**User's choice:** Keep RNG framing
**Notes:** A research gate now blocks the fidelity commitment — the rebuilt instrumented
trainer (Area 4) conclusively settles RNG vs deterministic before the planner locks the
bit-exact strategy. If refuted, fidelity target falls back to deterministic insertion-order
+ tie-break replication.

---

## Port scope / fidelity boundary

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal build+search now | Port only the in-memory path SC-1..SC-4 exercise; defer index_reader/writer + snapshot serialization | |
| Full transcription now | Port all files incl. index_reader/writer + snapshot serialization in one pass (~832 LOC) | ✓ |
| Minimal + serialization stubs | Build+search fully; reader/writer signatures round-trip in-memory but defer byte-format parity | |

**User's choice:** Full transcription now
**Notes:** Complete the `.cbm` save/load path for a trained KNN cloud once so `online_hnsw`
is never revisited, even though no SC exercises serialization.

---

## Existing exact calcer disposition

| Option | Description | Selected |
|--------|-------------|----------|
| Replace entirely | HNSW becomes the only KNN path; delete brute-force code | |
| Keep behind a flag | HNSW default (parity); brute-force-exact selectable via builder/param flag | ✓ |
| Keep for non-parity tests | Replace in production; retain exact impl in test code as cross-check | |

**User's choice:** Keep behind a flag
**Notes:** Default MUST be HNSW so the oracle gate exercises the parity path. Introduces a
config surface upstream lacks — planner names the flag and documents exact-mode as
non-parity. Exact path doubles as SC-1 cross-check evidence.

---

## Evidence & verification strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Frozen evidence + fixture | Rely on captured knn_neighbors evidence + frozen text_embedding_xor/ fixture; no rebuild | |
| Rebuild trainer up front | Reprovision clang-18 + rebuild instrumented trainer at phase start for full evidence regeneration | ✓ |
| Frozen first, rebuild on-demand | Start frozen; reprovision only if the port fails and captured evidence is insufficient | |

**User's choice:** Rebuild trainer up front
**Notes:** Highest-fidelity debugging; also the tool that settles Area 1's research gate.
Recipe in `catboost-instrumented-trainer-build` memory + `build_instrumented_trainer.sh`;
trainer no longer in /tmp, conan/ninja persist in ~/.local/bin. Frozen fixture stays the
oracle gate — trainer is for evidence, not re-baking.

## Claude's Discretion

- Crate/module placement of the port (cb-compute module vs new cb-hnsw crate) — pure-CPU,
  keep free of backend/cubecl symbols.
- f32 accumulation order + comparator details — fall out of Area 1's research; transcribe to match.

## Deferred Ideas

None — discussion stayed within phase scope.
