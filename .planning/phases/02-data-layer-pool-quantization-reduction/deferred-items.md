# Deferred Items — Phase 02 data-layer

Out-of-scope discoveries logged during execution (not fixed; not caused by the
current plan's changes).

## 02-03

- **cb-oracle clippy `neg_cmp_op_on_partial_ord` (pre-existing).**
  `cargo clippy -p cb-data --all-targets -- -D warnings` fails, but the error is
  in `crates/cb-oracle/src/compare.rs:44` (`!(diff <= tol)`) — cb-oracle is a
  dev-dependency dragged in by cb-data's integration tests. The lint was
  introduced by Phase-01 commit `902368d` (the deliberate NaN/Inf-as-divergence
  fix, where `!(diff <= tol)` is intentional to catch non-finite diffs) and
  surfaces only under the newer toolchain (rust-1.96.0). The plan's required gate
  `cargo clippy -p cb-data --lib -- -D warnings` is clean. Fixing would mean
  adding `#[allow(clippy::neg_cmp_op_on_partial_ord)]` (or restructuring the
  finite check) in cb-oracle — out of scope for the float-quantization slice.
  Recommend addressing in a cb-oracle housekeeping plan.
## [02-04] Pre-existing clippy lint in cb-oracle/src/compare.rs:44
- `clippy::neg_cmp_op_on_partial_ord` fires on `if !(diff <= tol)` (NaN-aware divergence check from Phase 1 commit 902368d).
- Out of scope for 02-04 (cat-hash plan; compare.rs untouched). Surfaces only under `--all-targets` clippy of cb-data's dependency graph; the plan's gate `clippy -p cb-data --lib` is clean.
- Suggested fix (later): rewrite as `if matches!(diff.partial_cmp(&tol), Some(Ordering::Greater) | None)` or add a scoped `#[allow]` with a NaN-handling comment.
