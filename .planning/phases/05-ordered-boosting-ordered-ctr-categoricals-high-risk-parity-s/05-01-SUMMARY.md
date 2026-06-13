---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
plan: 01
subsystem: cb-oracle (parity harness)
tags: [oracle, ctr, ordered-boosting, permutation, fixtures, transcribe-then-self-oracle]
requires:
  - cb-oracle::compare_stage (1e-5 gate, Phase 1)
  - cb-oracle::model_json (borders-only parser, Phase 3)
  - cb-oracle::fixture (load_f64_vec / load_config, Phase 1)
  - cb-core::TFastRng64 (bit-exact RNG, Phase 1 — permutation seed cross-check)
  - cb-model::generated::ctr_data_generated (ECtrType, Phase 4)
provides:
  - cb-oracle::Stage::{Permutation, OnlineCtr, OrderedApprox}
  - cb-oracle::compare_permutation (integer-exact, D-03 linchpin)
  - cb-oracle::model_json::CtrTableJson + ModelJson::ctr_data()
  - crates/cb-oracle/generator/ordered_oracle.cpp (per-object oracle harness)
  - 5 frozen categorical fixtures (one_hot_cat/plain_ctr/ordered_ctr/ordered_boost/tensor_ctr)
affects:
  - 05-02..05-06 (downstream slices oracle against these Stage variants + fixtures)
tech-stack:
  added: []
  patterns:
    - transcribe-then-self-oracle (zero-catboost-include C++ harness, cityhash precedent)
    - integer-exact permutation compare (== not 1e-5) gating value stages (D-03)
    - "#[serde(default)] backward-compatible model.json extension"
key-files:
  created:
    - crates/cb-oracle/generator/ordered_oracle.cpp
    - crates/cb-oracle/fixtures/one_hot_cat/{config.json,*.npy}
    - crates/cb-oracle/fixtures/plain_ctr/{config.json,*.npy}
    - crates/cb-oracle/fixtures/ordered_ctr/{config.json,*.npy}
    - crates/cb-oracle/fixtures/ordered_boost/{config.json,*.npy}
    - crates/cb-oracle/fixtures/tensor_ctr/{config.json,*.npy}
  modified:
    - crates/cb-oracle/src/compare.rs
    - crates/cb-oracle/src/compare_test.rs
    - crates/cb-oracle/src/error.rs
    - crates/cb-oracle/src/model_json.rs
    - crates/cb-oracle/src/model_json_test.rs
    - crates/cb-oracle/src/lib.rs
    - crates/cb-oracle/generator/README.md
decisions:
  - "Stage::Permutation compares i64 indices with == (integer-exact), short-circuiting at the first mismatch BEFORE any value stage runs (D-03 linchpin)."
  - "CtrTableJson stores hash_map as Vec<serde_json::Value> (heterogeneous hash-string + int-count array); bucket_counts() strips the hash and returns typed-error on ragged blobs (no panic, T-05-01-01)."
  - "ordered_oracle.cpp is a zero-catboost-include transcription (cityhash precedent); the four research-flagged TUs cannot be linked in isolation (RESEARCH ESCALATION)."
  - "Permutation self-oracle anchor verified: harness [4 2 0 3 1] == cb-core::TFastRng64 Fisher-Yates (seed=42,N=5)."
metrics:
  duration_min: 8
  completed: "2026-06-13"
  tasks: 3
  files: 13
---

# Phase 5 Plan 01: Wave-0 Oracle Infrastructure Summary

Stood up every oracle-harness gap for the high-risk parity slice: three new
`compare_stage` Stage variants (integer-exact `Permutation` + `≤1e-5` `OnlineCtr`
/`OrderedApprox`), a backward-compatible `model.json` `ctr_data` parser, the
zero-catboost-include `ordered_oracle.cpp` per-object transcription, and five
frozen purpose-built categorical fixtures — so each downstream slice
(05-02…05-06) lands against a ready comparator.

## What Shipped

### Task 1 — Stage variants + integer-exact permutation compare (`03e0dd2`)
- `Stage::{Permutation, OnlineCtr, OrderedApprox}` added to `compare.rs`.
- `compare_permutation(&[i64], &[i64])`: `==` comparison, rejects the first
  index mismatch — the D-03 linchpin lock (permutation reproduced exactly BEFORE
  any value stage runs). `OnlineCtr`/`OrderedApprox` route through the existing
  `≤1e-5` `compare_stage` path; the `!(diff <= tol)` NaN-safe guard preserved.
- New typed `OracleError::{PermutationLengthMismatch, PermutationDiverged}`
  (panic/unwrap-free).
- `compare_test.rs`: match-pass, single-swap-fail-at-first-index, integer
  near-miss reject, length mismatch, plus pass+fail for both `≤1e-5` paths.

### Task 2 — model.json ctr_data parser (`0c1a68b`)
- `CtrTableJson` serde struct over the upstream `hash_map`/`hash_stride`/
  `counter_denominator` shape (`json_model_helpers.cpp:475-482`).
- `bucket_counts()` strips the per-bucket hash string and returns the integer
  counts; ragged blob → `MalformedModel` (typed, no panic — T-05-01-01).
- `ModelJson.ctr_data` is `#[serde(default)] BTreeMap` + a `ctr_data()` accessor:
  borders-only Phase-3/4 fixtures keep parsing (RESEARCH A5).
- `model_json_test.rs`: borders-only empty map, Borders round-trip, Counter
  single-count + denominator, ragged-blob typed error.

### Task 3 — ordered_oracle.cpp + 5 frozen fixtures (`4f84d08`)
- `ordered_oracle.cpp` (416 lines): standalone ZERO-catboost-include
  transcription of (a) `TFastRng64`, (b) Fisher-Yates `Shuffle`, (c) online CTR
  read-before-increment + `CalcCTR`, (d) body/tail prefix, (e) ordered approx
  prefix update — each cited file:line. Dumps the D-02 `.npy` schema. Compiles
  with `g++ -O2 -std=c++17`; runs OFFLINE only (D-09).
- **Self-oracle anchor verified:** harness permutation `[4 2 0 3 1]` ==
  `cb-core::TFastRng64` Fisher-Yates (seed=42, N=5), the bitstream-locked RNG.
- Five fixture dirs, each `config.json` pinning every knob (boosting_type
  Plain/Ordered — never auto, simple/combinations_ctr + explicit prior,
  one_hot_max_size, max_ctr_complexity, permutation_count, fold_len_multiplier,
  counter_calc_method, thread_count=1, catboost_version=1.2.10, seed) + a frozen
  `.npy` stack. `one_hot_cat` carries a column of cardinality EXACTLY
  `one_hot_max_size` (one-hot) AND `+1` (CTR) per ORD-04 (RESEARCH Pitfall 3).
- README.md documents the transcription sources + self-oracle anchors + layout.

## Verification

- `cargo test -p cb-oracle` — **32 unit + 4 integration tests green** (compare +
  model_json suites included).
- `ordered_oracle.cpp` compiles standalone (`g++ -std=c++17`); `grep` confirms
  ZERO `catboost/` includes.
- All five fixture dirs present with explicit per-knob `config.json`; `.npy`
  stacks read by both numpy (1-D `<f8`/`<i4`) and Rust `load_f64_vec`/`load_config`
  (verified via throwaway examples, then removed).
- `Stage::Permutation` integer-exact; `OnlineCtr`/`OrderedApprox` ≤1e-5.

## Deviations from Plan

None — plan executed exactly as written. No Rule 1/2/3 auto-fixes were needed;
no architectural (Rule 4) decisions arose. The two pre-existing scratch-example
cross-checks (Rust permutation anchor + fixture-read smoke) were created in
throwaway `examples/` dirs and removed immediately; nothing extra was committed.

## Notes / Environment

- Disk pressure (~96% full, ~9.6G free) held throughout; all work was scoped to
  `cargo test -p cb-oracle` (lightweight crate, no MLIR/cubecl test-profile link)
  so the cb-compute link-failure risk noted in the plan was never hit.
- `clippy -p cb-oracle --lib` is clean except the one PRE-EXISTING
  `neg_cmp_op_on_partial_ord` warning on the documented `!(diff <= tol)` NaN-safe
  guard (compare.rs:58, not introduced by this plan). A new
  `is_multiple_of` clippy suggestion on the ctr_data length check was applied.
- The five fixtures' `.npy` stacks were generated by the transcribed harness fed
  deterministic seed-0 inputs from the pinned generator `.venv`
  (numpy 1.26.4 / catboost 1.2.10). Downstream slices may regenerate `ctr_value`
  / final-CTR anchors against the trained-model `ctr_data` as those slices wire
  in the actual cat-hash projection; the frozen integer permutation/count layout
  is the stable Wave-0 contract.

## Self-Check: PASSED

All 13 created/modified files exist on disk; all three task commits
(`03e0dd2`, `0c1a68b`, `4f84d08`) are present in git history.
