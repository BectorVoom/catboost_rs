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

## [02-05] Pre-existing clippy lints surfacing under newer toolchain (rust-1.96.0)
- **cb-oracle `neg_cmp_op_on_partial_ord` (recurring, see 02-03/02-04):** still
  fires under `cargo clippy --workspace --lib -- -D warnings`. compare.rs
  untouched by 02-05. The plan's per-crate gates (`clippy -p cb-data --lib`,
  `clippy -p cb-core --lib`) are clean.
- **cb-core `unnecessary_literal_unwrap` in error_test.rs:** the pre-existing
  `cb_result_ok_path_round_trips` test (`let ok: CbResult<u32> = Ok(42);
  assert_eq!(ok.unwrap(), 42);`) trips `clippy::unnecessary_literal_unwrap` under
  rust-1.96.0 `--all-targets`. The pattern predates 02-05 (Phase-1 error_test);
  my edit only shifted its line number. Out of scope for the ingestion+weights
  slice. Suggested fix (later, in a cb-core housekeeping pass): replace with a
  non-literal `Ok` value or a scoped `#[allow]`.
