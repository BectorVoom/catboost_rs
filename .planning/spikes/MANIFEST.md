# Spike Manifest

## Idea

Determine whether catboost 1.2.10's training-time `ComputeOnlineCTRs(AveragingFold)`
ui8 bins can be reproduced bit-exact OFFLINE in pure Rust/Python from committed
fixtures — the feasibility gate for closing Phase-5 bar (c) (pc=4 / SC-1 / ORD-01
production-default AveragingFold parity). The answer drives the choice between
re-planning 05-18 for a real offline CTR port vs. a live-instrumentation path vs.
deferring bar (c).

## Requirements

- An offline parity oracle for the AveragingFold online CTR must be derived from a
  **self-consistent** `(permutation, bins)` pair. The currently-committed pair
  (`upstream_avg_perm` + `upstream_avg_ctr_bins_avg_order`) is internally
  inconsistent under the upstream algorithm and is NOT a valid oracle (Spike 001).
- cb-train's `materialize_ctr_feature` / online-prefix / ui8 quantization must NOT
  be "fixed" to chase the committed bins — it is already bit-exact to the upstream
  C++ algorithm (Spike 001). Any re-plan must target the oracle/ground-truth, not
  this code.

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 001 | online-ctr-averaging-fold-offline | standard | Offline reproduction of `ComputeOnlineCTRs(AveragingFold)` ui8 bins bit-exact from committed inputs | ✗ NOT-ACHIEVABLE (committed oracle pair proven internally inconsistent; cb-train CTR code already correct) | ctr, parity, online-ctr, averaging-fold, phase-05, ord-01, pc4, bar-c |
