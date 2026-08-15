#!/usr/bin/env bash
# Run every oracle test that covers a STRING-VALUED CatBoost parameter.
#
# The string-param surface is spread over the facade crate (`catboost-rs`, one
# oracle test file per param family), `cb-data` (the quantization-time params:
# feature_border_type / nan_mode) and `cb-train` (the routing pins). This script
# is the single entry point so the suite can be re-run as a unit.
#
# Usage: scripts/run_string_param_oracles.sh
set -uo pipefail
cd "$(dirname "$0")/.."

FACADE_TESTS=(
  string_param_matrix_test      # one cell per value, every pre-existing string param
  nan_mode_oracle_test          # nan_mode: Min / Max / Forbidden
  sampling_oracle_test          # sampling_unit / sampling_frequency
  model_shrink_oracle_test      # model_shrink_mode: Constant / Decreasing
  random_score_type_oracle_test # random_score_type: NormalWithModelSizeDecrease / Gumbel
  ctr_modes_oracle_test         # final_ctr_computation_mode / ctr_history_unit
  leaf_backtracking_oracle_test # leaf_estimation_backtracking: No / AnyImprovement
  leaf_iterations_oracle_test   # leaf_estimation_iterations (the backtracking driver)
  const_label_oracle_test       # allow_const_label
  wireup_params_test            # model_size_reg + the snapshot trio
  builder_oracle_test           # loss_function / leaf_estimation_method / bootstrap_type / score_function
  class_weights_facade_test     # auto_class_weights: Balanced / SqrtBalanced
)

status=0
run() {
  echo "=== $* ==="
  "$@" || status=1
}

for t in "${FACADE_TESTS[@]}"; do
  run cargo test -p catboost-rs --test "$t" -- --test-threads=4
done

# feature_border_type lives in cb-data (the binarizer itself).
run cargo test -p cb-data border_type
run cargo test -p cb-data nan_mode

# The device-routing pins for the string-param wave.
run cargo test -p cb-train --test string_param_device_routing_test

if [ "$status" -eq 0 ]; then
  echo "ALL STRING-PARAM ORACLES PASSED"
else
  echo "STRING-PARAM ORACLE FAILURES PRESENT"
fi
exit "$status"
