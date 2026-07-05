"""Offline unit test for the Phase-14 BENCH-02 aggregator.

Per CLAUDE.md source/test separation, the test lives in its own file and never
inside ``aggregate.py``. It asserts the aggregator against the REAL committed
Phase-12 / Phase-13 JSON (a fixture-drift guard): offline, seconds, no GPU, no
network. Import ``load_rows`` from the sibling ``aggregate`` module by adding this
file's directory to ``sys.path`` so pytest can import it from the repo root.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from aggregate import (  # noqa: E402  (sys.path shim must precede this import)
    PHASE12_JSON,
    PHASE13_JSON,
    load_rows,
)


def _p12_rows():
    return load_rows(PHASE12_JSON, phase="P12", gpu="test-gpu", date="2026-07-04")


def _p13_rows():
    return load_rows(PHASE13_JSON, phase="P13", gpu="test-gpu", date="2026-07-04")


def test_twelve_rows_from_both_schemas():
    """Both committed files together yield exactly 12 rows.

    Guards Pitfall 1: if the nested ``.bench02.runs[]`` branch were broken, the
    Phase-13 file would contribute 0 rows and the total would be 6, not 12.
    """
    rows = _p12_rows() + _p13_rows()
    assert len(rows) == 12, (
        f"expected 12 aggregated rows, got {len(rows)} -- a total of 6 means the "
        f"Phase-13 nested '.bench02.runs[]' branch dropped its rows (Pitfall 1)."
    )


def test_every_speedup_ge_20x():
    """Every aggregated speedup is a float >= 20.0 (D-01 hard gate).

    Proves the string-to-float cast happened (a raw JSON string would not be a
    ``float`` instance) and that every row clears the >=20x device-vs-host gate.
    """
    rows = _p12_rows() + _p13_rows()
    for r in rows:
        assert isinstance(r["speedup"], float), (
            f"speedup for {r['phase']}/{r['family']}/n={r['n']} is "
            f"{type(r['speedup']).__name__}, not float -- the float() cast is missing."
        )
        assert r["speedup"] >= 20.0, (
            f"{r['phase']}/{r['family']}/n={r['n']} speedup {r['speedup']} < 20x (D-01)."
        )
        assert r["ge20x"] is True


def test_phase13_nested_schema_resolves():
    """The Phase-13 file alone yields its 6 rows via the nested branch.

    If ``load_rows`` only read the root ``.runs[]`` shape, the Phase-13 file
    (which has no root ``runs``) would resolve 0 rows. Six rows proves the
    ``.bench02.runs[]`` branch fired.
    """
    rows = _p13_rows()
    assert len(rows) == 6, (
        f"Phase-13 file resolved {len(rows)} rows via load_rows; expected 6 "
        f"through the nested '.bench02.runs[]' branch."
    )


if __name__ == "__main__":
    import sys as _sys

    import pytest

    _sys.exit(pytest.main([__file__, "-x", "-q"]))
