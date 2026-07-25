#!/usr/bin/env bash
# Build the `catboost-rs-wgpu` wheel WITHOUT leaving a swapped pyproject.toml in
# the tracked checkout.
#
# Usage:
#   crates/catboost-rs-py/build-wgpu-wheel.sh [extra maturin args...]

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cpu_pyproject="${script_dir}/pyproject.toml"
wgpu_pyproject="${script_dir}/pyproject-wgpu.toml"

test -f "$wgpu_pyproject" || {
  echo "error: ${wgpu_pyproject} not found" >&2
  exit 1
}

backup=""
if [ -f "$cpu_pyproject" ]; then
  backup="$(mktemp "${cpu_pyproject}.cpu-backup.XXXXXX")"
  cp "$cpu_pyproject" "$backup"
fi

restore() {
  if [ -n "$backup" ]; then
    mv -f "$backup" "$cpu_pyproject"
  else
    rm -f "$cpu_pyproject"
  fi
}
trap restore EXIT INT TERM

cp "$wgpu_pyproject" "$cpu_pyproject"

maturin build --no-default-features --features wgpu --release "$@"
