
## 06.6-06 out-of-scope discoveries (pre-existing clippy lints, NOT introduced by this plan)

- `cb-model/src/json.rs:275` — `indexing_slicing` (`node_id_to_leaf_id[id] = ...`). Pre-existing (06.6-03/05 era), file untouched by 06.6-06. `cargo build` / `cargo test` pass; only `cargo clippy` strict flags it.
- `cb-model/src/cbm.rs:259` — clippy lint, pre-existing.
- `cb-model/src/predict.rs:200` — clippy lint, pre-existing.
- `cb-backend/src/cpu_runtime.rs:1025` — `slicing may panic`, pre-existing, unrelated crate.

These are NOT regressions from 06.6-06 (only `fstr.rs` + `lib.rs` were modified; `fstr.rs` is clippy-clean). Deferred per executor SCOPE BOUNDARY.
