#!/usr/bin/env bash
# Source/test separation gate (INFRA-06 / D-17).
#
# Embedding `#[cfg(test)] mod tests { ... }` (an inline test MODULE BODY) at the
# bottom of a production source file is strictly prohibited. All tests must live
# in dedicated `*_test.rs` files. This script FAILS if any production
# `crates/*/src/**/*.rs` file (excluding `*_test.rs`) contains an inline
# `#[cfg(test)]` module body.
#
# The DECLARATION form `#[cfg(test)] mod <name>_test;` (a `mod` line ending in a
# semicolon, pointing at a separate file) is ALLOWED — that is exactly the
# convention this rule enforces. Only the brace form `mod ... {` is flagged.
#
# Primary defense is code review; this grep is the CI backstop.
set -euo pipefail

# Resolve repo root so the script works from any CWD.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

violations=0

# Enumerate production source files: crates/*/src/**/*.rs, excluding *_test.rs.
while IFS= read -r file; do
  case "$file" in
    *_test.rs) continue ;;
  esac

  # Find every `#[cfg(test)]` line, then inspect the NEXT non-blank line. If it
  # is a `mod ... {` (brace) form, it introduces an inline test module body ->
  # violation. A `mod ...;` (semicolon) declaration is allowed.
  #
  # awk walks the file: when we see #[cfg(test)] we remember it; on the next
  # non-blank line we decide. We also catch a same-line `#[cfg(test)] mod x {`.
  hits="$(
    awk '
      function check_modline(line, lineno) {
        # A module body opener: `mod <name> {` (brace), with or without pub.
        if (line ~ /(^|[[:space:]])mod[[:space:]]+[A-Za-z0-9_]+[[:space:]]*\{/) {
          printf("%d:%s\n", lineno, line)
          return 1
        }
        return 0
      }
      {
        # Same-line form: #[cfg(test)] mod tests {
        if ($0 ~ /#\[cfg\(test\)\]/ && $0 ~ /mod[[:space:]]+[A-Za-z0-9_]+[[:space:]]*\{/) {
          printf("%d:%s\n", NR, $0)
          pending = 0
          next
        }
        if ($0 ~ /#\[cfg\(test\)\]/) { pending = 1; next }
        if (pending) {
          # skip blank lines / attribute lines between cfg(test) and mod
          if ($0 ~ /^[[:space:]]*$/) next
          check_modline($0, NR)
          pending = 0
        }
      }
    ' "$file"
  )"

  if [ -n "$hits" ]; then
    echo "D-17 violation: inline #[cfg(test)] module body in production source: $file"
    echo "$hits"
    violations=1
  fi
done < <(find crates -type f -path '*/src/*' -name '*.rs' 2>/dev/null | sort)

if [ "$violations" -ne 0 ]; then
  echo "ERROR: inline #[cfg(test)] test module(s) found in production code (INFRA-06 / D-17 violation)."
  echo "       Move test bodies into dedicated *_test.rs files."
  exit 1
fi

echo "OK: no inline #[cfg(test)] module bodies in production source"
exit 0
