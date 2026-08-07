#!/usr/bin/env bash
# Run every cb-train ROCm device test named in the PLAN's Definition of Done, one
# `--test` at a time (a package-wide run fails E0432 — ~52 test files import CpuBackend
# at module scope). Prints one PASS/FAIL line per test binary.
set -uo pipefail

TESTS=(
  device_bias_gate_test
  device_exact_leaf_gate_test
  device_nonsym_bootstrap_gate_test
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
exit "$fail"
