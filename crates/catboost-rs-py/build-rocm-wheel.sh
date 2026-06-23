#!/usr/bin/env bash
# Build the `catboost-rs-rocm` wheel WITHOUT leaving a swapped pyproject.toml in
# the tracked checkout (IN-04).
#
# Background: maturin reads `pyproject.toml` by name, and the rocm distribution
# needs the SEPARATE `pyproject-rocm.toml` (different [project].name). A manual
# `cp pyproject-rocm.toml pyproject.toml` swap is error-prone — a half-applied or
# forgotten swap leaves a stale pyproject.toml that would publish the wrong
# distribution name. This wrapper performs the swap transactionally: it backs up
# the cpu pyproject.toml, swaps the rocm one in, runs maturin, and ALWAYS restores
# the original on exit (success, failure, or interrupt), so the tracked file is
# never left mutated.
#
# Build IN-ENV ONLY (gfx1100 / ROCm 7.1) — NEVER in GitHub Actions (D-06). See
# pyproject-rocm.toml for the patchelf prerequisite and the LD_PRELOAD runtime
# requirement.
#
# Usage:
#   crates/catboost-rs-py/build-rocm-wheel.sh [extra maturin args...]
# e.g.
#   crates/catboost-rs-py/build-rocm-wheel.sh        # build --release wheel
#   crates/catboost-rs-py/build-rocm-wheel.sh --out dist-rocm

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cpu_pyproject="${script_dir}/pyproject.toml"
rocm_pyproject="${script_dir}/pyproject-rocm.toml"

test -f "$rocm_pyproject" || {
  echo "error: ${rocm_pyproject} not found" >&2
  exit 1
}

# Back up the existing cpu pyproject.toml (if any) so we can restore it verbatim.
backup=""
if [ -f "$cpu_pyproject" ]; then
  backup="$(mktemp "${cpu_pyproject}.cpu-backup.XXXXXX")"
  cp "$cpu_pyproject" "$backup"
fi

restore() {
  if [ -n "$backup" ]; then
    mv -f "$backup" "$cpu_pyproject"
  else
    # There was no cpu pyproject.toml before; remove the swapped-in rocm copy so
    # the checkout is left exactly as we found it.
    rm -f "$cpu_pyproject"
  fi
}
# ALWAYS restore — success, error (set -e), or interrupt — so a half-applied swap
# can never persist in the tracked checkout.
trap restore EXIT INT TERM

cp "$rocm_pyproject" "$cpu_pyproject"

# cpu-free rocm build (08-01 feature wiring): --no-default-features --features rocm.
maturin build --no-default-features --features rocm --release "$@"
