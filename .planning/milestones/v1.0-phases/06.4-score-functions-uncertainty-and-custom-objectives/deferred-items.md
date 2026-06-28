# Deferred Items — Phase 06.4

Out-of-scope discoveries logged during execution (not fixed here per the executor
SCOPE BOUNDARY rule — pre-existing failures unrelated to the current task's changes).

## 06.4-04 (LOSS-07 custom objective/metric)

- **`catboost-rs::builder_oracle_test` — `builder_regression_full_cycle` /
  `builder_binclf_full_cycle` PRE-EXISTING oracle divergence (Predictions stage).**
  - Confirmed failing on the pre-plan committed `builder.rs` (HEAD before 06.4-04),
    so NOT caused by this plan's `custom_objective`/`custom_metric` builder additions
    (which are behavior-preserving for the default `eval_metric: None` path).
  - regression: `expected 1.3333638442345577, actual 1.20251427407194, diff 0.131`
  - binclf: `expected 0.19854958472762677, actual 0.20185865161058725, diff 0.0033`
  - Root cause is in the facade `fit` default training path (likely a borders /
    boost-from-average / leaf-method facade default drift), independent of LOSS-07.
  - Action: triage in a Phase 6.4 hardening / facade-parity pass.
