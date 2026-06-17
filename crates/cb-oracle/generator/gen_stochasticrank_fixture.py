#!/usr/bin/env python3
"""OFFLINE / RUN-ONCE driver for the StochasticRank end-to-end fixture + per-tree
per-group Gaussian noise ground truth (Plan 06.3-18, gap #3 of LOSS-04).

Runs the EXISTING `gen_ranking_fixtures.gen_loss("StochasticRank")` target (now
pinning `StochasticRank:metric=DCG`) under the INSTRUMENTED catboost 1.2.10
`_catboost.so` with `CB_INSTRUMENT_LOG` set, so a SINGLE single-thread training
run produces both:

  * ranking_corpus/stochasticrank/{model.json,staged.npy,predictions.npy,config.json}
    — the frozen upstream StochasticRank model (Splits/LeafValues/StagedApprox/
      Predictions oracle target);
  * the raw instrumentation JSONL, from which we extract the per-group
    `srank_noise` events (at %.17g, carrying the per-group `seed`) and freeze them
    as `stochasticrank_pertree_noise_groundtruth.jsonl` — the D-07 trainer-level
    ground truth distinct from the standalone single-group self-oracle.

Invocation (RUN-ONCE; the instrumented .so + toolchain live under /tmp):

    CB_INSTRUMENT_LOG=/tmp/srank_instr.jsonl \
      PYTHONPATH=/tmp/cb_build313/instr_pkg \
      .venv/bin/python crates/cb-oracle/generator/gen_stochasticrank_fixture.py

The model.json + the four committed fixture files are the only artifacts checked
in; the raw /tmp log is intermediate.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

GENERATOR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GENERATOR_DIR))

import gen_ranking_fixtures as g  # noqa: E402


def main() -> int:
    raw_log = os.environ.get("CB_INSTRUMENT_LOG")
    if not raw_log:
        print(
            "REFUSING: CB_INSTRUMENT_LOG must be set so the per-group srank_noise "
            "draws are captured by the instrumented trainer.",
            file=sys.stderr,
        )
        return 2

    # Truncate any stale log so the per-tree noise we freeze is exactly this run.
    Path(raw_log).write_text("", encoding="utf-8")

    # Ensure the corpus inputs exist, then train the StochasticRank scenario. The
    # generator now pins `StochasticRank:metric=DCG` and writes to the lowercase
    # `stochasticrank/` scenario dir the Rust oracle reads.
    g.write_inputs()
    g.gen_loss("StochasticRank")

    scenario = g.RANKING_CORPUS / "stochasticrank"
    # Extract ONLY the per-group srank_noise events (the D-07 ground truth). The
    # raw log also carries leaf_der / tree fences from the instrumented .so; we
    # freeze just the noise stream at full precision.
    gt_path = scenario / "stochasticrank_pertree_noise_groundtruth.jsonl"
    lines = Path(raw_log).read_text(encoding="utf-8").splitlines()
    noise_events = [ln for ln in lines if '"event":"srank_noise"' in ln]
    if not noise_events:
        print(
            "REFUSING: the instrumented run emitted NO srank_noise events — the "
            "StochasticRank der path was not exercised (check metric/loss spec).",
            file=sys.stderr,
        )
        return 3
    gt_path.write_text("\n".join(noise_events) + "\n", encoding="utf-8")

    print(f"wrote {len(noise_events)} srank_noise events to {gt_path}")
    print(f"StochasticRank fixture under {scenario}")
    # Report the distinct per-group seeds seen (D-07 spot-check anchor).
    seeds = sorted({ln.split('"seed":')[1].split(",")[0] for ln in noise_events})
    print(f"distinct per-group seeds in log: {seeds[:20]}{' ...' if len(seeds) > 20 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
