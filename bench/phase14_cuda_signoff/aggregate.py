"""Offline aggregator for the Phase-14 CUDA speed sign-off (BENCH-03).

Aggregate committed results, do NOT re-run (D-03)
-------------------------------------------------
Every device / host-CPU / speedup number in the BENCH-03 sign-off comes from the
per-phase BENCH-02 JSON that Phase-12 and Phase-13 already committed on the real
Tesla P100 Kaggle CUDA image. This script re-reads those two committed files
*offline* (stdlib only, no GPU, no network) and stitches them into one 12-row
device / host-CPU / speedup / >=20x matrix. No value is computed or invented --
each row is copied verbatim and tagged with its source phase / gpu / date so the
sign-off document preserves provenance.

Two committed sources, two schemas (D-03 mixed-session aggregation)
------------------------------------------------------------------
The two files store the same per-row shape under DIFFERENT keys:

  * Phase-12 ``bench/phase12_cuda_oracle/bench02-result.json`` -- rows at the ROOT
    ``.runs[]``; ``gpu`` / ``nvcc`` at the top level.
  * Phase-13 ``bench/phase13_cuda_oracle/result.json`` -- rows NESTED under
    ``.bench02.runs[]``; ``gpu`` / ``nvcc`` at the top level.

:func:`load_rows` resolves both with a single schema branch
``d.get("runs") or d.get("bench02", {}).get("runs", [])`` so neither shape silently
drops its 6 rows (Pitfall 1). Each row's ``speedup`` is stored in the JSON as a
STRING (e.g. ``"39.987"``); we ``float()``-cast it before any numeric comparison
(Pitfall 2).

The D-01 hard gate
------------------
Every aggregated row must be ``speedup >= 20.0`` (device >= 20x the host CPU). The
verdict line reads ``BENCH-03: PASS`` only when every row clears the gate, else
``BENCH-03: FAIL`` naming the offending rows.
"""

import argparse
import json
import os
import sys

# --------------------------------------------------------------------------------------
# Committed source locations (resolved relative to the repo root, never hardcoded absolute)
# --------------------------------------------------------------------------------------

#: The D-01 hard gate: device must be at least this many times faster than the host CPU.
GE20X_GATE = 20.0

#: Repo root, derived from this file's location (``bench/phase14_cuda_signoff/aggregate.py``)
#: so the script runs correctly from any cwd.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

#: The ONLY two committed BENCH-02 sources (A4: nothing under Phase-10/Phase-11).
PHASE12_JSON = os.path.join(_REPO_ROOT, "bench", "phase12_cuda_oracle", "bench02-result.json")
PHASE13_JSON = os.path.join(_REPO_ROOT, "bench", "phase13_cuda_oracle", "result.json")


def load_rows(path, phase, gpu, date):
    """Read a committed BENCH-02 JSON and yield normalized speed rows.

    Resolves the Phase-12 root ``.runs[]`` shape and the Phase-13 nested
    ``.bench02.runs[]`` shape via one schema branch, so neither silently drops
    its 6 rows (Pitfall 1). Each yielded dict merges the original row fields with
    the string ``speedup`` cast to ``float`` (Pitfall 2), the source provenance
    (``phase`` / ``gpu`` / ``date``), and the D-01 gate flag ``ge20x``.

    Args:
        path: Path to the committed BENCH-02 JSON file.
        phase: Provenance tag for the source phase (e.g. ``"P12"``).
        gpu: GPU string read from the file's top level (provenance).
        date: Run date string (provenance).

    Returns:
        list[dict]: one normalized row per benchmarked ``(family, n)``.

    Raises:
        ValueError: if the file contains neither schema's ``runs`` list, or a
            row is missing ``speedup`` / has a non-numeric ``speedup`` -- a loud
            failure on schema drift rather than a bare ``KeyError`` (T-14-02).
    """
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)

    # Schema branch: Phase-12 root .runs[] OR Phase-13 nested .bench02.runs[].
    rows = d.get("runs") or d.get("bench02", {}).get("runs", [])
    if not rows:
        raise ValueError(
            f"{path}: no BENCH-02 rows found under either '.runs' or "
            f"'.bench02.runs' -- schema drift or wrong file (Pitfall 1)."
        )

    out = []
    for r in rows:
        if "speedup" not in r:
            raise ValueError(f"{path}: row {r!r} is missing a 'speedup' field.")
        try:
            speedup = float(r["speedup"])  # JSON stores it as a string (Pitfall 2).
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: row {r!r} has a non-numeric 'speedup' {r['speedup']!r}: {exc}"
            ) from exc
        out.append(
            {
                "phase": phase,
                "gpu": gpu,
                "date": date,
                "family": r.get("family"),
                "n": r.get("n"),
                "device_s": r.get("device_s"),
                "cpu_s": r.get("cpu_s"),
                "speedup": speedup,
                "dev_trees": r.get("dev_trees"),
                "cpu_trees": r.get("cpu_trees"),
                "ge20x": speedup >= GE20X_GATE,  # D-01 hard gate flag.
            }
        )
    return out


def _top_level(path, key):
    """Read a top-level scalar (e.g. ``gpu`` / ``nvcc``) from a committed JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh).get(key)


def aggregate():
    """Load and concatenate all 12 committed rows (6 Phase-12 + 6 Phase-13)."""
    p12_gpu = _top_level(PHASE12_JSON, "gpu")
    p13_gpu = _top_level(PHASE13_JSON, "gpu")
    rows = []
    rows += load_rows(PHASE12_JSON, phase="P12", gpu=p12_gpu, date="2026-07-04")
    rows += load_rows(PHASE13_JSON, phase="P13", gpu=p13_gpu, date="2026-07-04")
    return rows


def _format_table(rows):
    """Render the aggregated rows as a markdown table."""
    header = "| phase | family | n | device_s | cpu_s | speedup | >=20x? |"
    sep = "| --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['phase']} | {r['family']} | {r['n']} | "
            f"{r['device_s']} | {r['cpu_s']} | {r['speedup']:.3f} | "
            f"{'yes' if r['ge20x'] else 'NO'} |"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Offline aggregator over committed Phase-12/Phase-13 "
        "BENCH-02 JSON (D-03). Prints the 12-row speed matrix + BENCH-03 verdict."
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="Also dump the combined rows + boolean verdict to this JSON file.",
    )
    args = parser.parse_args(argv)

    rows = aggregate()
    below = [r for r in rows if not r["ge20x"]]
    verdict_pass = not below

    print(_format_table(rows))
    print()
    if verdict_pass:
        print("BENCH-03: PASS")
    else:
        offenders = ", ".join(
            f"{r['phase']}/{r['family']}/n={r['n']} ({r['speedup']:.3f}x)" for r in below
        )
        print(f"BENCH-03: FAIL (rows below 20x: {offenders})")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows, "verdict_pass": verdict_pass}, fh, indent=2)

    return 0 if verdict_pass else 1


if __name__ == "__main__":
    sys.exit(main())
