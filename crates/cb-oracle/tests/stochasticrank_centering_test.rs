//! Cargo discovery shim for the WR-03 stochasticrank centering accumulation-order
//! parity test (Plan 06.3-08). The test body lives next to the generator it
//! validates (`crates/cb-oracle/generator/stochasticrank_centering_test.rs`,
//! source/test separation / INFRA-06); this thin shim makes
//! `cargo test -p cb-oracle stochasticrank_centering --tests` discover it.

#[path = "../generator/stochasticrank_centering_test.rs"]
mod stochasticrank_centering;
