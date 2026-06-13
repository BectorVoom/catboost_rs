#!/usr/bin/env bash
# Primary defense is the SINGLE-PRIMITIVE rule (D-07): every float sum routes
# through `cb-core::sum_f64`/`sum_f32_in_f64` (the one sanctioned hand-written
# loop). This grep (D-08) is a belt-and-suspenders backstop that bans every
# OTHER raw float summation form so a reordered/compensated sum cannot silently
# break the <= 1e-5 oracle gate downstream (threat T-02-02).
#
# Scans the six core-lib crates for raw float summation (`.sum()` and the
# `.fold(0.0, +)` family) in non-test Rust source. Two paths are excluded:
#   * `*_test.rs` — test code may sum freely (it builds expected values).
#   * `crates/cb-core/src/reduction.rs` — THE sanctioned primitive (D-07).
# Exits 1 on any violation.
set -euo pipefail

# Resolve repo root so the script works from any CWD.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CORE_DIRS=(
  "crates/cb-core"
  "crates/cb-data"
  "crates/cb-compute"
  "crates/cb-backend"
  "crates/cb-train"
  "crates/cb-model"
)

# The sanctioned reduction primitive — the only file allowed a raw float loop.
REDUCTION_PRIMITIVE="crates/cb-core/src/reduction.rs"

# Raw float-summation patterns (extended regex). Matches `.sum()` and the
# `.fold(0.0 ...)` / `.fold(0.0_f... )` / `.fold(0f... )` zero-seed-float forms.
SUM_PATTERN='\.sum\(\)|\.fold\(0\.0|\.fold\(0_f|\.fold\(0f'

violations=0
for dir in "${CORE_DIRS[@]}"; do
  # Tolerate not-yet-existing dirs: later plans create the other core-lib crates,
  # so in early waves only some of these exist. Skip missing dirs silently.
  [ -d "$dir" ] || continue

  # List candidate files first, then filter excluded paths, then report hits.
  while IFS= read -r file; do
    case "$file" in
      *_test.rs) continue ;;
      "$REDUCTION_PRIMITIVE") continue ;;
    esac
    if grep -InE "$SUM_PATTERN" "$file" >/dev/null 2>&1; then
      echo "D-08 violation: raw float summation in core library code: $file"
      grep -InE "$SUM_PATTERN" "$file" || true
      violations=1
    fi
  done < <(grep -RIlE --include='*.rs' -e "$SUM_PATTERN" "$dir" 2>/dev/null || true)
done

if [ "$violations" -ne 0 ]; then
  echo "ERROR: raw float summation in core library non-test code (D-08 violation)"
  echo "Route all float sums through cb-core::sum_f64 / sum_f32_in_f64 (D-07)."
  exit 1
fi

echo "OK: no raw float summation in core library code (cb-core::reduction excluded)"
exit 0
