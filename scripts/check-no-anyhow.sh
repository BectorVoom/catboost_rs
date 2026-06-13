#!/usr/bin/env bash
# Primary defense is the STRUCTURAL ban (D-14): `anyhow` is absent from `[dependencies]` of every core lib, so it cannot be used in non-test code. This grep is a belt-and-suspenders backstop only.
#
# Scans the six core-lib crates for any reference to `anyhow` in non-test Rust
# source. Test files (`*_test.rs`) are excluded — `anyhow` is permitted as a
# dev-dependency / at test edges (D-14/D-15). Exits 1 on any violation.
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

violations=0
for dir in "${CORE_DIRS[@]}"; do
  # Tolerate not-yet-existing dirs: Plans 02/03 create the other core-lib crates,
  # so in Wave 1 only some of these exist. Skip missing dirs silently.
  [ -d "$dir" ] || continue

  # Fixed-string scan for portability across GNU/BSD grep. List matching files,
  # filter out `*_test.rs`, then report any line hits in the survivors.
  while IFS= read -r file; do
    case "$file" in
      *_test.rs) continue ;;
    esac
    if grep -In 'anyhow' "$file" >/dev/null 2>&1; then
      echo "D-14 violation: 'anyhow' referenced in core library code: $file"
      grep -In 'anyhow' "$file" || true
      violations=1
    fi
  done < <(grep -RIl --include='*.rs' -e 'anyhow' "$dir" 2>/dev/null || true)
done

if [ "$violations" -ne 0 ]; then
  echo "ERROR: anyhow referenced in core library non-test code (D-14 violation)"
  exit 1
fi

echo "OK: no anyhow in core library code"
exit 0
