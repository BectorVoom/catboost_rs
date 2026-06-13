---
phase: 02
slug: data-layer-pool-quantization-reduction
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-13
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (`cargo test`) + cb-oracle ≤1e-5 compare harness (`compare_stage`, `assert_abs_close`, `Stage`) |
| **Config file** | none — workspace `Cargo.toml` |
| **Quick run command** | `cargo test -p cb-data -p cb-core` |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | ~30s quick / ~60s full (fast `cargo test` runs over the small frozen corpus; no long-running training) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p cb-data -p cb-core`
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~60 seconds (full workspace suite)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | DATA-07 | T-02-01 | Reduction order locked (naive-sequential, no Kahan/pairwise) | unit | `cargo test -p cb-core reduction` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | DATA-07 | T-02-02 / T-02-SC | D-08 grep bans raw float `.sum()`/`.fold(0.0`; pinned `Cargo.lock` for audited arrow/polars | unit | `bash scripts/check-no-raw-float-sum.sh; cargo build -p cb-data` | ❌ W0 | ⬜ pending |
| 02-01-03 | 01 | 1 | DATA-03 / DATA-04 / DATA-05 / DATA-08 | T-02-03 / T-02-SC | Fixtures frozen at pinned seed + `thread_count=1`; A1–A5 recorded in `config.json` | oracle (fixture gen, manual gate) | `test -f crates/cb-oracle/fixtures/inputs/numeric_nan/X.npy && test -f crates/cb-oracle/fixtures/borders_quant/config.json && test -f crates/cb-oracle/fixtures/cat_hash/cat_hashes.npy && test -f crates/cb-oracle/fixtures/class_weights/balanced.npy && echo FIXTURES_OK` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 2 | DATA-01 | T-02-04 / T-02-05 | Column-length mismatch → typed `CbResult` error, no panic/index | unit | `cargo test -p cb-data pool 2>&1 \| tail -5; cargo test -p cb-data ingest 2>&1 \| tail -5` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02 | 2 | DATA-03 | T-02-06 | Borders in f32; sums via `cb-core::reduction`; `-0.0`/duplicate-collapse locked | unit + oracle | `cargo test -p cb-data borders 2>&1 \| tail -6; cargo test -p cb-data --test borders_oracle_test 2>&1 \| tail -8` | ❌ W0 | ⬜ pending |
| 02-03-01 | 03 | 3 | DATA-04 | T-02-07 | Strict `value > border` (equal → lower bin); NaN sentinel placement | unit | `cargo test -p cb-data nan_mode 2>&1 \| tail -6` | ❌ W0 | ⬜ pending |
| 02-03-02 | 03 | 3 | DATA-02 / DATA-04 | T-02-08 / T-02-09 | Width arm bounded `<65536` for float; NaN budget off-by-one guarded; u32 cat-only | unit + oracle | `cargo test -p cb-data quantized_pool 2>&1 \| tail -5; cargo test -p cb-data --test quantize_oracle_test 2>&1 \| tail -8` | ❌ W0 | ⬜ pending |
| 02-04-01 | 04 | 4 | DATA-05 | T-02-10 | Ported CityHash64 (vendored variant), bit-exact `assert_eq!` incl. >16-byte tail | unit | `cargo test -p cb-data cat_hash 2>&1 \| tail -6` | ❌ W0 | ⬜ pending |
| 02-04-02 | 04 | 4 | DATA-05 | T-02-11 | Uniq-count bound to `u32::MAX` (MAX_UNIQ_CAT_VALUES) → `CbResult` error, no panic | oracle | `cargo test -p cb-data --test cat_hash_oracle_test 2>&1 \| tail -8` | ❌ W0 | ⬜ pending |
| 02-05-01 | 05 | 5 | DATA-06 | T-02-13 / T-02-14 | Dtype/length/NaN-in-categorical → typed `CbError` (Clone+Eq preserved), no panic | unit | `cargo test -p cb-core error 2>&1 \| tail -4; cargo test -p cb-data ingest 2>&1 \| tail -8` | ❌ W0 | ⬜ pending |
| 02-05-02 | 05 | 5 | DATA-08 | T-02-15 / T-02-16 | `1e-8` floor guards degenerate class (no div-by-zero); sums via `cb-core::reduction` | unit + oracle | `cargo test -p cb-data weights 2>&1 \| tail -5; cargo test -p cb-data --test weights_oracle_test 2>&1 \| tail -8` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*
*File Exists `❌ W0` = the source/test files do not yet exist; they are created during Wave 0 (Plans 02-01…02-05 fill the `cb-data` stub additively).*

---

## Wave 0 Requirements

- [ ] `cb-core/src/reduction.rs` + `reduction_test.rs` — the audited sequential f64 reduction primitive (Plan 02-01 Task 1)
- [ ] `scripts/check-no-raw-float-sum.sh` (D-08 CI-grep backstop) wired into `.github/workflows/ci.yml`; `arrow = "59.0.0"` + `polars = "0.54.4"` added to `[workspace.dependencies]`; `Cargo.lock` committed (Plan 02-01 Task 2)
- [ ] cb-oracle fixtures generated from pinned `catboost==1.2.10`: `borders_quant/` (border + quant), `cat_hash/` (`cat_hashes.npy` + `perfect_hash_bins.npy`), `class_weights/` (`balanced.npy` + `sqrt_balanced.npy`), and the new `inputs/numeric_nan/` dataset (Plan 02-01 Task 3)
- [ ] Assumptions A1–A5 resolved empirically and recorded in fixture `config.json` + `02-01-SUMMARY.md` (NaN sentinel in `get_borders()` A3, default `border_count` A2, integer-cat stringification A4, CityHash64 `(string→ui32)` vectors / port-vs-crate A1/A5) — resolved via the Plan 02-01 Task 3 human-verify checkpoint

*All Wave 0 work is delivered by Plan 02-01 (Wave 1), whose outputs every downstream plan (02-02…02-05) consumes.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Oracle fixture generation + A1–A5 resolution (Plan 02-01 Task 3) | DATA-03 / DATA-04 / DATA-05 / DATA-08 | The `catboost==1.2.10` fixture generator runs build-time-only on the dev machine (pinned venv), NOT in CI; a human must confirm the generated `.npy`/`config.json` and the recorded A1–A5 resolutions before approving | 1. `cd crates/cb-oracle/generator && .venv/bin/python gen_inputs.py && .venv/bin/python gen_fixtures.py`. 2. Confirm `inputs/numeric_nan/X.npy` contains NaN entries. 3. Confirm `{borders_quant,cat_hash,class_weights}/` each contain their `.npy` + `config.json`. 4. Open each `config.json` and confirm A1–A5 are explicitly recorded. Type "approved" to resume. |

*The downstream comparison BEHAVIORS these fixtures gate (borders, quant, cat-hash, weights) all have automated `cargo test … --test *_oracle_test` verification in the map above — only the build-time fixture generation is manual.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 60s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
