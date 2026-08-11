#!/usr/bin/env bash
# Run every cb-train ROCm device test named in the PLAN's Definition of Done, one
# `--test` at a time. Prints one PASS/FAIL line per test binary.
#
# Roster: 29 binaries (the 23 green at `a0a67ec` + the 5 added by the "Device CTR
# Coverage P1" phase: `device_ctr_buckets_fit_test` (T10), `device_ctr_counter_fit_test`
# (T12), `device_ctr_type_gate_test` (T13), `device_ctr_btmv_fit_test` (T16) and
# `device_ctr_combo_types_diff_test` (T22); + `device_ctr_eligible_max_diff_test`, the
# R-20 closure detector for D-2). `device_ctr_combo_fit_test` was already
# listed and was un-ignored by T19, so it converts from a vacuous pass to a real one
# without changing this array. The count is DERIVED from the array below, never aimed
# at — a new device binary that exists but is unregistered here is a phase-DoD failure
# even if it passes standalone.
#
# ONE `--test <name>` PER BINARY IS MANDATORY, and it is not a stylistic choice (R-10):
# a package-wide `cargo test -p cb-train --no-default-features --features rocm` fails
# `E0432` because ~52 test files `use cb_backend::CpuBackend;` at module scope, which is
# not compiled under rocm — the whole run dies before any device test executes. The CTR
# binaries below are therefore separate `--test` invocations rather than one filtered
# run. Equally mandatory (R-9): `--features rocm` NEVER without `--no-default-features`,
# or `cb-backend`'s `default = ["cpu"]` wins the `cfg` chain, `SelectedRuntime` resolves
# to `cubecl::cpu::CpuRuntime`, and every "device" test silently compares cpu with cpu.
set -uo pipefail

TESTS=(
  device_bias_gate_test
  device_exact_leaf_gate_test
  device_nonsym_bootstrap_gate_test
  device_bias_fit_test
  device_exact_leaf_fit_test
  device_ctr_combo_fit_test
  device_nonsym_bootstrap_test
  device_ordered_fit_test
  device_fpp_composition_test
  device_nonsym_fit_test
  device_region_fit_test
  device_weighted_fit_test
  device_weighted_nonsym_fit_test
  device_ctr_fit_test
  device_ctr_gate_test
  device_weight_gate_test
  device_fold_count_gate_test
  device_gate_composition_test
  bootstrap_dev_oracle_test
  device_bootstrap_parity_test
  device_poisson_bootstrap_test
  device_one_hot_parity_test
  device_seam_test
  # ── Device CTR Coverage P1 (appended; the 5 binaries the phase created) ────────────
  device_ctr_buckets_fit_test        # T10 — DCTR-08 Buckets e2e
  device_ctr_counter_fit_test        # T12 — DCTR-10 Counter e2e
  device_ctr_type_gate_test          # T13 + T21 — counter_calc_method / surviving-clause pins
  device_ctr_btmv_fit_test           # T16 — DCTR-14 BinarizedTargetMeanValue e2e
  device_ctr_combo_types_diff_test   # T22 — DCTR-20 combination x non-Borders differential
  # ── R-20 closure (post-phase) ─────────────────────────────────────────────────────
  device_ctr_eligible_max_diff_test  # DCTR-16 / D-2 — the behavioural detector for the
                                     # per-level `eligible_max` eligibility filter (R-20)
)

fail=0
for t in "${TESTS[@]}"; do
  out=$(cargo test -p cb-train --no-default-features --features rocm --test "$t" 2>&1)
  line=$(printf '%s\n' "$out" | grep -E '^test result:' | head -1)
  if printf '%s\n' "$out" | grep -q 'test result: FAILED\|^error'; then
    echo "FAIL  $t  ${line:-<no result line>}"
    printf '%s\n' "$out" | grep -E '^test .* FAILED|^error' | head -5 | sed 's/^/        /'
    fail=1
  else
    echo "PASS  $t  ${line:-<no result line>}"
  fi
done

# ── Isolated perf lane ────────────────────────────────────────────────────────────────
# `#[ignore]`d timing tests, each in its OWN process with a single test thread, because
# they compare wall-clock against a co-scheduled baseline and need an uncontended device.
# Inside the package-wide cb-backend suite the Poisson ratio reads 3.3–3.8x (bar: 5x)
# purely from GPU contention; alone it reads 10.5–10.7x. See the module docs on
# `poisson_bootstrap_speed_test` for the measurements and for why a fixed time budget
# would NOT have fixed it.
PERF_TESTS=(
  kernels::poisson_bootstrap_speed_test
)

for t in "${PERF_TESTS[@]}"; do
  out=$(cargo test -p cb-backend --no-default-features --features rocm \
          --lib "$t" -- --ignored --nocapture --test-threads=1 2>&1)
  line=$(printf '%s\n' "$out" | grep -E '^test result:' | head -1)
  if printf '%s\n' "$out" | grep -q 'test result: FAILED\|^error'; then
    echo "FAIL  [perf] $t  ${line:-<no result line>}"
    printf '%s\n' "$out" | grep -E '^test .* FAILED|^error|speed\]' | head -5 | sed 's/^/        /'
    fail=1
  else
    echo "PASS  [perf] $t  ${line:-<no result line>}"
    printf '%s\n' "$out" | grep -E 'speed\]' | head -2 | sed 's/^/        /'
  fi
done

exit "$fail"
