"""F22 (SPEC-CATF-Δ8, acceptance A13) — the bench's facade-routing preflight
passes WITHOUT editing the bench.

`bench/one_hot_gpu_speed/one_hot_bench_colab.py` bails `BLOCKED-FACADE-ROUTING`
unless `train_cat` appears in `crates/catboost-rs/src/builder.rs`. Its own check
is a bare `grep -c`, which a COMMENT would satisfy — so this test asserts the
stronger property the bench actually means.
"""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_bench_preflight_facade_routing_marker_is_present():
    builder = (REPO / "crates/catboost-rs/src/builder.rs").read_text()

    # 1. The bench's preflight (one_hot_bench_colab.py:117, checked at :210-227)
    #    only does `grep -c "train_cat"` — a comment would satisfy it. Assert
    #    the marker is in a CALL position on a non-comment line.
    call_sites = [
        ln
        for ln in builder.splitlines()
        if re.search(r"\btrain_cat\s*\(", ln) and not ln.lstrip().startswith("//")
    ]
    assert call_sites, (
        "`train_cat` must appear as an actual CALL in builder.rs, not merely as a "
        "comment: bench/one_hot_gpu_speed/one_hot_bench_colab.py:117 greps for the "
        "bare string, so a comment would make the bench pass while fit() still "
        "routes float-only"
    )

    # 2. The bench's OWN check must also pass (the grep it really runs).
    out = subprocess.run(
        ["grep", "-c", "train_cat", str(REPO / "crates/catboost-rs/src/builder.rs")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert int(out.stdout.strip()) > 0


def test_bench_file_is_byte_unchanged():
    """The bench must pass UNEDITED — that is the whole point of A13."""
    out = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "bench/one_hot_gpu_speed/one_hot_bench_colab.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.stdout.strip() == "", (
        f"the bench must not be edited to make its preflight pass:\n{out.stdout}"
    )
