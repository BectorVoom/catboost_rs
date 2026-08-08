"""Generate the COUNTER-CTR device-parity fixture (Device CTR Coverage P1, DCTR-10/T06).

FROZEN GENERATOR — the committed artifacts under this directory are the GROUND TRUTH the
device Counter CTR path (Track C: whole-learn-set tally, constant max denominator,
permutation-independent value) is compared against at <=1e-5. CI does NOT run this script
and never regenerates these fixtures.

# Why this fixture exists (SPEC DCTR-10)

Every existing Counter fixture in the corpus (`ctr_counter_simple/`,
`ctr_counter_full_eval/`) is a categorical-ONLY pool. A cat-only pool gives
`device_n_float == 0`, so `has_any_scorable_feature` (`boosting.rs`) declines and those
fixtures can NEVER reach the device. `ctr_device_mixed/` IS device-reachable but is a
Borders fixture. This is the first Counter fixture that is device-reachable.

# THE COUNTER TRAP (R-11 sibling) — the reason this fixture pins the prior explicitly

Upstream's DEFAULT Counter prior is `0/1`, NOT `0.5` (`cb-train/src/ctr/mod.rs`
`default_priors()`). The prior is therefore pinned EXPLICITLY on both sides:
`simple_ctr = ["Counter:Prior=0.5"]` here, and `simple_ctr_priors = vec![0.5]` in the
Rust `BoostParams` of the T12 e2e test. A mismatch produces a silent, plausible-looking
divergence with no compile error and no shape error — which is exactly why the smoke test
`crates/cb-oracle/tests/ctr_device_counter_fixture_smoke_test.rs` asserts the params
string verbatim.

# The two HARD gate constraints (inherited from `ctr_device_mixed/` / `ctr_device_combo/`)

  - `ctr_covered` (`session.rs`) requires `col.borders.len() + 1 == n_bins` for every CTR
    column, where `n_bins` is the device histogram width derived from the FLOAT
    quantization. `ctr_border_count_default() == 15` ⇒ 16 CTR buckets ⇒ the float
    `border_count` MUST be 15 (16 bins) or the CTR arm declines.
  - The float borders fed to the Rust trainer must be the FULL quantization set, not
    model.json's pruned USED subset — frozen here as `borders.npy`.

# Deltas from `ctr_device_combo/gen_fixtures.py` (T06's pinned recipe)

  - `SCENARIO = "ctr_device_counter"`, `CARDS = (6,)` — ONE cat column, so `X_cat.npy` is
    1-D `[N]` exactly as in `ctr_device_mixed/` (its single-cat precedent).
  - `simple_ctr = ["Counter:Prior=0.5"]` (EXPLICIT prior, see the trap above).
  - `combinations_ctr = []`, `max_ctr_complexity = 1` — simple projections only; Track C's
    scope is the Counter STATISTIC, not projection arity (that is Track D / T17-T19).
  - The target's cat term is a single-column ramp instead of the combo XOR pair.

GATE-load-bearing and unchanged: `border_count=15`, `one_hot_max_size=1` (routes the cat
column to CTR rather than one-hot), `permutation_count=1`, `bootstrap_type="No"`,
`random_strength=0`, `boost_from_average=False`, `N_FLOAT >= 1`, `score_function="L2"`,
`leaf_estimation_method="Gradient"`, `leaf_estimation_iterations=1`, `thread_count=1`,
and NO `task_type` (this is the CPU oracle).

# Seed search (mandatory, recorded)

Whether upstream actually SELECTS a Counter split is data-dependent: the Counter statistic
is a function of the category's whole-set frequency only, so a seed whose frequencies carry
no usable signal yields a model with zero CTR descriptors and a vacuous fixture.
`SEARCH_SEEDS` is scanned in order and the FIRST seed whose trained model contains a
`Counter` CTR descriptor AND at least one float split is frozen; the winning seed is
recorded in `config.json` as `data_seed` and the escalation rung as `escalation_rung`.
If the search fails, widen `SEARCH_SEEDS` or raise the cardinality — never lower a guard.

# Reproducibility caveat

CatBoost quantization is run-to-run nondeterministic on categorical routing; this is why
every CTR fixture in this repo is FROZEN and never regenerated in CI. Regenerating
invalidates the <=1e-5 gate for every downstream test.

# Fixed-point overflow margin (SPEC §9, mandatory)

Logloss der1 ∈ (-1, 1) and weights are uniform 1.0, so
`n · max(w) · max(|der1|) <= 64 · 1.0 · 1.0 = 64` << 2^33 ≈ 8.6e9 — margin > 1.3e8x.
"""

import json
import os
import subprocess
import sys

import numpy as np
from catboost import CatBoost, Pool

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.dirname(HERE)
SCENARIO = "ctr_device_counter"

N_ROWS = 64
N_FLOAT = 2
CARDS = (6,)
SEARCH_SEEDS = tuple(range(24))
# Escalation ladder rung actually used (1 = seeds 0..23 at CARDS=(6,), the starting rung).
ESCALATION_RUNG = 1

PARAMS = {
    "loss_function": "Logloss",
    "iterations": 5,
    "depth": 2,
    "learning_rate": 0.1,
    "l2_leaf_reg": 3.0,
    "border_count": 15,
    "boosting_type": "Plain",
    "one_hot_max_size": 1,
    "max_ctr_complexity": 1,
    # EXPLICIT prior — upstream's Counter default is 0/1, not 0.5. See the module docstring.
    "simple_ctr": ["Counter:Prior=0.5"],
    "combinations_ctr": [],
    "permutation_count": 1,
    "fold_len_multiplier": 2.0,
    "counter_calc_method": "SkipTest",
    "leaf_estimation_method": "Gradient",
    "leaf_estimation_iterations": 1,
    "bootstrap_type": "No",
    "random_strength": 0,
    "score_function": "L2",
    "boost_from_average": False,
    "random_seed": 0,
    "thread_count": 1,
    "verbose": False,
}


def counter_descriptors(model_json):
    """CTR descriptors of type Counter in a serialised model. Upstream serialises only the
    CTRs the model actually USES, so a non-empty result means >=1 Counter split was chosen
    (verified against `ctr_device_mixed/model.json`, whose guard is the Borders sibling)."""
    ctrs = model_json["features_info"].get("ctrs", [])
    return [c for c in ctrs if c.get("ctr_type") == "Counter"]


def make_data(seed):
    rng = np.random.RandomState(seed)
    x = rng.rand(N_ROWS, N_FLOAT).astype(np.float32)
    cat = rng.randint(0, CARDS[0], size=N_ROWS).astype(np.int32)
    # Per-object FLOAT ramp in the target (R-14): a purely categorical +-1 target would let
    # a structure-vs-averaging leaf swap hide. Same shape as the combo generator's
    # `3.0 * (cat_term - 0.5) + 2.0 * (x0 - x1)`, with the single-column cat term
    # `cat % 2 == 0` standing in for the combo XOR pair.
    cat_term = (cat % 2 == 0).astype(np.float64)
    logit = 3.0 * (cat_term - 0.5) + 2.0 * (x[:, 0] - x[:, 1])
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.rand(N_ROWS) < prob).astype(np.int32)
    return x, cat, y


def make_pool(x, cat, y):
    df = [[int(cat[i]), float(x[i, 0]), float(x[i, 1])] for i in range(N_ROWS)]
    return Pool(df, label=y, cat_features=[0])


def fit(params, x, cat, y, pool=None):
    model = CatBoost(params)
    model.fit(pool if pool is not None else make_pool(x, cat, y))
    preds = model.predict(make_pool(x, cat, y), prediction_type="RawFormulaVal")
    return model, preds


def dump_model_json(model, seed):
    if hasattr(model, "dumps_model"):
        return json.loads(model.dumps_model())
    tmp = os.path.join(HERE, f".probe_seed{seed}.json")
    model.save_model(tmp, format="json")
    with open(tmp) as fh:
        model_json = json.load(fh)
    os.remove(tmp)
    return model_json


def evaluate_seed(seed):
    """Return (x, cat, y, n_counter) or None if the seed's fit is unusable (degenerate
    target, constant predictions, no Counter CTR descriptor, or no float split)."""
    x, cat, y = make_data(seed)
    if len(np.unique(y)) < 2:
        return None
    model, preds = fit(PARAMS, x, cat, y)
    if preds.std() <= 1e-6:
        return None
    model_json = dump_model_json(model, seed)
    n_counter = len(counter_descriptors(model_json))
    has_float_split = any(
        len(f.get("borders", [])) > 0
        for f in model_json["features_info"]["float_features"]
    )
    if n_counter < 1 or not has_float_split:
        return None
    return x, cat, y, n_counter


def main():
    chosen = None
    tried = []
    for seed in SEARCH_SEEDS:
        result = evaluate_seed(seed)
        tried.append(seed)
        if result is not None:
            chosen = (seed, result)
            break
    if chosen is None:
        sys.exit(
            f"no seed in {SEARCH_SEEDS} produced BOTH a Counter CTR descriptor and a float "
            "split — widen SEARCH_SEEDS or the cardinality (escalation ladder); do NOT "
            "lower the bar"
        )
    seed, (x, cat, y, probe_counter) = chosen
    print(f"seed search: scanned {tried}, chose {seed} (Counter descriptors={probe_counter})")

    # Freeze the FULL float border set. Pool column order is [cat, f0, f1];
    # `save_quantization_borders` indexes by the pool's FLOAT feature order.
    qpool = make_pool(x, cat, y)
    qpool.quantize(border_count=PARAMS["border_count"])
    borders_tsv = os.path.join(HERE, "borders.tsv")
    qpool.save_quantization_borders(borders_tsv)
    per_feature = {}
    with open(borders_tsv) as fh:
        for line in fh:
            fi, bv = line.split()
            per_feature.setdefault(int(fi), []).append(float(bv))
    os.remove(borders_tsv)
    float_ids = sorted(per_feature)
    assert len(float_ids) == N_FLOAT, f"expected {N_FLOAT} float border rows, got {float_ids}"
    for fi in float_ids:
        assert len(per_feature[fi]) == PARAMS["border_count"], (
            f"feature {fi}: expected {PARAMS['border_count']} borders, got {len(per_feature[fi])}"
        )
    borders = np.array([sorted(per_feature[fi]) for fi in float_ids], dtype=np.float64)

    model, predictions = fit(PARAMS, x, cat, y, pool=qpool)

    # The quantized-pool fit must be bit-identical to the raw-pool fit, or the frozen
    # borders would not describe the committed predictions.
    _raw_model, raw_preds = fit(PARAMS, x, cat, y)
    assert np.abs(predictions - raw_preds).max() == 0.0, (
        "quantized-pool fit diverged from raw-pool fit — border freezing is unsound here"
    )

    model_json_path = os.path.join(HERE, "model.json")
    model.save_model(model_json_path, format="json")
    with open(model_json_path) as fh:
        model_json = json.load(fh)

    # --- MANDATORY anti-false-pass guards (T06) --------------------------------
    ctrs = model_json["features_info"].get("ctrs", [])
    counters = counter_descriptors(model_json)
    # 1. THE discriminating assertion: >=1 Counter CTR descriptor, i.e. upstream really
    #    chose a Counter split. Without it the device Counter path is unexercised.
    assert len(counters) >= 1, (
        f"model.json has {len(ctrs)} CTR descriptor(s) but none of type Counter — the "
        "device Counter path would be unexercised; re-seed, do not lower the bar"
    )
    # 1b. Only Counter descriptors may appear: `simple_ctr` carries a single Counter entry
    #     and `combinations_ctr` is empty, so any other type would mean the recipe drifted.
    assert all(c.get("ctr_type") == "Counter" for c in ctrs), (
        f"non-Counter CTR descriptor present: {sorted({c.get('ctr_type') for c in ctrs})}"
    )
    # 1c. The prior must be the pinned 0.5 (= 1/2), never upstream's 0/1 default.
    for c in counters:
        num, den = c.get("prior_numerator"), c.get("prior_denomerator")
        assert (num, den) == (0.5, 1) or (den not in (None, 0) and num / den == 0.5), (
            f"Counter descriptor prior is {num}/{den}, expected 0.5/1 — the explicit "
            "`Counter:Prior=0.5` pin did not take effect"
        )
    # 2. >=1 float split, or the device n_features arithmetic is untested.
    assert any(
        len(f.get("borders", [])) > 0
        for f in model_json["features_info"]["float_features"]
    ), "no float split in the model — the float axis is decorative"
    # 3. Non-degenerate outputs.
    assert predictions.std() > 1e-6, "degenerate constant predictions"

    np.save(os.path.join(HERE, "borders.npy"), borders)
    np.save(os.path.join(HERE, "X.npy"), x)
    np.save(os.path.join(HERE, "X_cat.npy"), cat)
    np.save(os.path.join(HERE, "y.npy"), y.astype(np.float64))
    np.save(os.path.join(HERE, "predictions.npy"), predictions.astype(np.float64))

    config = {
        "scenario": SCENARIO,
        "requirement": "DCTR-10",
        "data_seed": seed,
        "seed_search": list(SEARCH_SEEDS),
        "seeds_tried": tried,
        "escalation_rung": ESCALATION_RUNG,
        "catboost_version": "1.2.10",
        "thread_count": 1,
        "n_rows": N_ROWS,
        "cardinalities": list(CARDS),
        "counter_descriptor_count": len(counters),
        "counter_prior": 0.5,
        "counter_prior_pin": (
            "Upstream's DEFAULT Counter prior is 0/1, NOT 0.5 (cb-train/src/ctr/mod.rs "
            "default_priors()). The prior is pinned EXPLICITLY on BOTH sides: "
            "simple_ctr=['Counter:Prior=0.5'] here, and simple_ctr_priors=vec![0.5] in the "
            "Rust BoostParams of the DCTR-10 e2e test. A mismatch is a silent divergence "
            "with no compile or shape error."
        ),
        "overflow_margin": (
            "n*max(w)*max(|der1|) <= 64*1.0*1.0 = 64 << 2^33 ~= 8.6e9 (margin > 1.3e8x) "
            "— Logloss der1 is bounded by 1 in magnitude"
        ),
        "description": (
            "Counter-CTR device-parity fixture. 64 rows, two float columns "
            "(border_count=15, FULL border set frozen as borders.npy) plus ONE "
            "cardinality-6 cat column routed to a simple Counter CTR "
            "(one_hot_max_size=1, Prior=0.5 pinned explicitly, CtrBorderCount default "
            "15 = 16 buckets matching the 16 float bins — the ctr_covered n_bins "
            "requirement). Logloss depth-2 x5, Plain, permutation_count=1, no sampling, "
            "bias 0, Gradient leaf, L2 score. The target's logit carries a categorical "
            "term plus a per-object float ramp, so a structure-vs-averaging leaf swap "
            "cannot hide. Pins the Counter statistic (whole-learn-set tally, constant max "
            "denominator, permutation-independent) the device Track C path must reproduce "
            "at <=1e-5. NEVER regenerated in CI."
        ),
        "params": PARAMS,
        "npy_schema": {
            "borders.npy": "[2,15] f64 — FULL frozen float border set (feed to train_cat)",
            "X.npy": "[N,2] f32 — float feature matrix",
            "X_cat.npy": "[N] int32 — categorical codes (stringified A4 form on the Rust side)",
            "y.npy": "[N] f64 — binclf label",
            "predictions.npy": "[N] f64 — RawFormulaVal (<=1e-5 gate)",
        },
        "note": (
            "FROZEN. Generated once by this directory's own gen_fixtures.py and NEVER "
            "regenerated in CI. CatBoost quantization is run-to-run nondeterministic on "
            "categorical routing; regenerating invalidates the <=1e-5 gate."
        ),
    }
    with open(os.path.join(HERE, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")

    out = subprocess.run(
        ["git", "status", "--porcelain", FIXTURES],
        capture_output=True,
        text=True,
        cwd=FIXTURES,
    ).stdout
    offenders = [line for line in out.splitlines() if line.strip() and SCENARIO not in line]
    if offenders:
        print("corpus contamination — this generator touched paths outside its scenario:")
        for line in offenders:
            print("   ", line)
        sys.exit(1)

    print(
        f"wrote {SCENARIO}: seed={seed}, ctr descriptors={len(ctrs)} "
        f"(Counter={len(counters)}), predictions.std()={predictions.std():.6f}"
    )


if __name__ == "__main__":
    main()
