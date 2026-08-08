"""Generate the BUCKETS-CTR device-parity fixture (DCTR-08 / T05).

FROZEN GENERATOR — the committed artifacts under this directory are the GROUND TRUTH the
device Buckets-CTR path (SPEC §4.2's numerator contract, `ctr_device.rs`) is compared
against at <=1e-5. CI does NOT run this script and never regenerates these fixtures.

# Why this fixture exists

`ctr_buckets_simple/` already pins the Buckets numerator on the CPU, but it is a
categorical-ONLY pool: `device_n_float == 0`, so `has_any_scorable_feature`
(`boosting.rs:3284-3286`) and the session's own `n_features == 0` decline both fire and it
can NEVER reach the device. `ctr_device_mixed/` is device-reachable but is a *Borders*
fixture. This is the first fixture that is BOTH device-reachable (two float columns) and
carries `ECtrType::Buckets` columns.

For a Buckets CTR at binclf upstream emits one candidate column PER `target_border_idx` in
`0..targetClassesCount` (`GetTargetBorderCount`, `ctr_helper.h:34-42`; the
`(ctrIdx, targetBorderIdx, priorIdx)` nesting of `greedy_tensor_search.cpp:400-428`) — i.e.
BOTH the class-0 and the class-1 numerator. SPEC §4.2:

    Buckets @ b  =>  good = counts[b]
    otherwise    =>  good = total - sum_{c <= b} counts[c]

so a model that only ever selected `b = 0` would leave the device kernel's `Buckets@1`
numerator entirely unexercised while every `<=1e-5` comparison still passed. THE
discriminating guard below asserts both indices are present in the committed model.

# GATE-load-bearing parameters — never change these

  - `border_count=15` — `ctr_covered` (`session.rs:134-163`) requires
    `col.borders.len() + 1 == n_bins` for every CTR column, and
    `ctr_border_count_default() == 15` ⇒ 16 CTR buckets ⇒ the float `border_count` MUST be
    15 or the CTR arm declines (R-11).
  - `one_hot_max_size=1` — routes the cat column(s) to the CTR path, not one-hot.
  - `permutation_count=1`, `bootstrap_type="No"`, `random_strength=0`,
    `boost_from_average=False`, `score_function="L2"`,
    `leaf_estimation_method="Gradient"`, `leaf_estimation_iterations=1`,
    `thread_count=1`, and NO `task_type` (this is the CPU oracle).
  - `N_FLOAT >= 1` — a cat-only pool can never reach the device.
  - `simple_ctr=["Buckets:Prior=0.5"]`, `combinations_ctr=[]`, `max_ctr_complexity=1` —
    simple Buckets only; no combination projection can form, so Track A's scope is
    unchanged.
  - The float borders fed to the Rust trainer must be the FULL quantization set, not
    model.json's pruned USED subset — frozen here as `borders.npy` (R-15).

# TUNABLE — the escalation ladder (plan T05, checker MAJOR-4)

`CARDS`, `N_ROWS`, `iterations` and `SEARCH_SEEDS` are SEARCH parameters, not invariants.
The only in-repo configuration known to achieve the both-borders guard is
`ctr_buckets_simple`'s (60 rows, TWO cat columns of cardinality 6 and 5, 10 iterations) —
and that is a cat-only pool, so here two float columns additionally compete for the same
split slots, making the guard STRICTLY harder. The rungs below are therefore applied in
order, stopping at the first success, and the winning rung is recorded in `config.json` as
`escalation_rung`. If every rung fails, STOP and escalate — a Buckets fixture without both
target borders cannot discharge DCTR-08. NEVER weaken the guard to make a lower rung pass.

# Reproducibility caveat

CatBoost quantization is run-to-run nondeterministic on categorical routing; this is why
every CTR fixture in this repo is FROZEN and never regenerated in CI.

# Fixed-point overflow margin (SPEC §9, mandatory)

Logloss der1 in (-1, 1) and weights are uniform 1.0, so
`n * max(w) * max(|der1|) <= 64 * 1.0 * 1.0 = 64` << 2^33 ~= 8.6e9 — margin > 1.3e8x.
"""

import json
import os
import subprocess
import sys

import numpy as np
from catboost import CatBoost, Pool

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.dirname(HERE)
SCENARIO = "ctr_device_buckets"

N_ROWS = 64
N_FLOAT = 2

# The escalation ladder, in the order the plan mandates. Rung 0 is the pinned starting
# point; each later rung is tried only after every earlier rung has exhausted its seed
# search without satisfying the guards.
RUNGS = (
    {
        "rung": 0,
        "cards": (6,),
        "n_rows": N_ROWS,
        "iterations": 5,
        "seeds": tuple(range(24)),
        "why": "pinned starting point",
    },
    {
        "rung": 1,
        "cards": (6,),
        "n_rows": N_ROWS,
        "iterations": 5,
        "seeds": tuple(range(64)),
        "why": "widen SEARCH_SEEDS to range(64)",
    },
    {
        "rung": 2,
        "cards": (8,),
        "n_rows": N_ROWS,
        "iterations": 5,
        "seeds": tuple(range(64)),
        "why": "raise the cardinality to 8",
    },
    {
        "rung": 3,
        "cards": (6, 5),
        "n_rows": N_ROWS,
        "iterations": 5,
        "seeds": tuple(range(64)),
        "why": "add a second cat column — ctr_buckets_simple's cardinalities",
    },
    {
        "rung": 4,
        "cards": (6, 5),
        "n_rows": N_ROWS,
        "iterations": 10,
        "seeds": tuple(range(64)),
        "why": "raise iterations to 10 — ctr_buckets_simple's value",
    },
)

BASE_PARAMS = {
    "loss_function": "Logloss",
    "iterations": 5,
    "depth": 2,
    "learning_rate": 0.1,
    "l2_leaf_reg": 3.0,
    "border_count": 15,
    "boosting_type": "Plain",
    "one_hot_max_size": 1,
    "max_ctr_complexity": 1,
    "simple_ctr": ["Buckets:Prior=0.5"],
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


def params_for(rung):
    return dict(BASE_PARAMS, iterations=rung["iterations"])


def cat_term(cat):
    """A GRADED categorical signal in [0, 1].

    A binary cat signal collapses the two Buckets numerators onto the same ordering, so a
    graded per-category positive rate is what makes both `target_border_idx` values
    genuinely selectable. The two-column form mirrors `ctr_buckets_simple`'s
    `1.2*(c0%2) - 0.9*(c1%3) + 0.3*(c0//3)` (the only recipe known to satisfy the guard),
    rescaled from its [-1.8, 1.5] range onto [0, 1].
    """
    c0 = cat[:, 0].astype(np.float64)
    if cat.shape[1] == 1:
        return 0.8 * (c0 % 2) + 0.2 * ((c0 // 3) % 2)
    c1 = cat[:, 1].astype(np.float64)
    raw = 1.2 * (c0 % 2) - 0.9 * (c1 % 3) + 0.3 * ((c0 // 3) % 2)
    return (raw + 1.8) / 3.3


def make_data(seed, rung):
    rng = np.random.RandomState(seed)
    n_rows = rung["n_rows"]
    x = rng.rand(n_rows, N_FLOAT).astype(np.float32)
    cat = np.stack(
        [rng.randint(0, c, size=n_rows).astype(np.int32) for c in rung["cards"]], axis=1
    )
    # Per-object FLOAT ramp in the target (R-14): a purely categorical +-1 target would let
    # a structure-vs-averaging leaf swap hide, and would make the float axis decorative.
    logit = 3.0 * (cat_term(cat) - 0.5) + 2.0 * (x[:, 0] - x[:, 1])
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.rand(n_rows) < prob).astype(np.int32)
    return x, cat, y


def make_pool(x, cat, y):
    n_cat = cat.shape[1]
    df = [
        [int(v) for v in cat[i]] + [float(v) for v in x[i]] for i in range(x.shape[0])
    ]
    return Pool(df, label=y, cat_features=list(range(n_cat)))


def fit(params, x, cat, y, pool=None):
    model = CatBoost(params)
    model.fit(pool if pool is not None else make_pool(x, cat, y))
    preds = model.predict(make_pool(x, cat, y), prediction_type="RawFormulaVal")
    return model, preds


def dump_model_json(model, tag):
    if hasattr(model, "dumps_model"):
        return json.loads(model.dumps_model())
    tmp = os.path.join(HERE, f".probe_{tag}.json")
    model.save_model(tmp, format="json")
    with open(tmp) as fh:
        model_json = json.load(fh)
    os.remove(tmp)
    return model_json


def target_border_idxs(model_json):
    """The set of `target_border_idx` values in the model's CTR descriptors.

    The key is TOP-LEVEL on each `features_info.ctrs[i]` — verified against
    `ctr_buckets_simple/model.json`, whose descriptor key set is exactly
    `borders, ctr_type, elements, identifier, prior_denomerator, prior_numerator, scale,
    shift, target_border_idx`. Indexed with `[...]`, never a defaulting `.get(..., 0)`:
    a default would make the discriminating guard silently vacuous.
    """
    ctrs = model_json["features_info"].get("ctrs", [])
    return {c["target_border_idx"] for c in ctrs}


def has_float_split(model_json):
    return any(
        len(f.get("borders", [])) > 0
        for f in model_json["features_info"]["float_features"]
    )


def evaluate_seed(seed, rung):
    """Return (x, cat, y) or None if the seed's fit fails any guard."""
    x, cat, y = make_data(seed, rung)
    if len(np.unique(y)) < 2:
        return None
    model, preds = fit(params_for(rung), x, cat, y)
    if preds.std() <= 1e-6:
        return None
    model_json = dump_model_json(model, f"seed{seed}")
    ctrs = model_json["features_info"].get("ctrs", [])
    if not ctrs or any(c["ctr_type"] != "Buckets" for c in ctrs):
        return None
    if sorted(target_border_idxs(model_json)) != [0, 1]:
        return None
    if not has_float_split(model_json):
        return None
    return x, cat, y


def main():
    chosen = None
    attempts = []
    for rung in RUNGS:
        tried = []
        for seed in rung["seeds"]:
            result = evaluate_seed(seed, rung)
            tried.append(seed)
            if result is not None:
                chosen = (rung, seed, result)
                break
        attempts.append({"rung": rung["rung"], "why": rung["why"], "seeds_tried": tried})
        print(
            f"rung {rung['rung']} ({rung['why']}): scanned {len(tried)} seed(s)"
            + (f" — CHOSE seed {tried[-1]}" if chosen else " — no seed passed")
        )
        if chosen:
            break
    if chosen is None:
        sys.exit(
            "no (rung, seed) produced a model carrying BOTH target_border_idx 0 and 1 with "
            "a float split — escalate; do NOT weaken the guard. A Buckets fixture without "
            "both target borders cannot discharge DCTR-08."
        )
    rung, seed, (x, cat, y) = chosen
    params = params_for(rung)

    # Freeze the FULL float border set. Pool column order is [cat..., f0, f1];
    # `save_quantization_borders` indexes by the pool's FLOAT feature order.
    qpool = make_pool(x, cat, y)
    qpool.quantize(border_count=params["border_count"])
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
        assert len(per_feature[fi]) == params["border_count"], (
            f"feature {fi}: expected {params['border_count']} borders, got {len(per_feature[fi])}"
        )
    borders = np.array([sorted(per_feature[fi]) for fi in float_ids], dtype=np.float64)

    model, predictions = fit(params, x, cat, y, pool=qpool)

    # The quantized-pool fit must be bit-identical to the raw-pool fit, or the frozen
    # borders would not describe the committed predictions.
    _raw_model, raw_preds = fit(params, x, cat, y)
    assert np.abs(predictions - raw_preds).max() == 0.0, (
        "quantized-pool fit diverged from raw-pool fit — border freezing is unsound here"
    )

    model_json_path = os.path.join(HERE, "model.json")
    model.save_model(model_json_path, format="json")
    with open(model_json_path) as fh:
        model_json = json.load(fh)

    # --- MANDATORY anti-false-pass guards -------------------------------------
    ctrs = model_json["features_info"].get("ctrs", [])
    assert ctrs, "no CTR descriptor in model.json — fixture is vacuous"
    # 1. THE discriminating assertion: BOTH Buckets numerators must be present.
    bad_types = sorted({c["ctr_type"] for c in ctrs if c["ctr_type"] != "Buckets"})
    assert not bad_types, f"non-Buckets CTR descriptors present: {bad_types}"
    idxs = target_border_idxs(model_json)
    assert sorted(idxs) == [0, 1], (
        f"ctr_device_buckets requires BOTH target_border_idx 0 and 1; model.json has "
        f"{sorted(idxs)}. Without both, the device Buckets@1 numerator is unexercised and "
        "DCTR-08 is untestable. Escalate the ladder — do NOT weaken this assertion."
    )
    # 2. >=1 float split, or the device n_features arithmetic is untested.
    assert has_float_split(model_json), (
        "no float split in the model — the float axis is decorative"
    )
    # 3. Non-degenerate predictions.
    assert predictions.std() > 1e-6, "degenerate constant predictions"

    np.save(os.path.join(HERE, "borders.npy"), borders)
    np.save(os.path.join(HERE, "X.npy"), x)
    # A single cat column is stored 1-D, matching `ctr_device_mixed/X_cat.npy`'s `(64,)`;
    # two columns are stored `(N, 2)`, matching `ctr_device_combo`.
    np.save(os.path.join(HERE, "X_cat.npy"), cat[:, 0] if cat.shape[1] == 1 else cat)
    np.save(os.path.join(HERE, "y.npy"), y.astype(np.float64))
    np.save(os.path.join(HERE, "predictions.npy"), predictions.astype(np.float64))

    n_cat = cat.shape[1]
    cat_schema = (
        "[N] int32 — categorical codes (stringified A4 form on the Rust side)"
        if n_cat == 1
        else f"[N,{n_cat}] int32 — categorical codes (stringified A4 form on the Rust side)"
    )
    config = {
        "scenario": SCENARIO,
        "requirement": "DCTR-08",
        "data_seed": seed,
        "escalation_rung": rung["rung"],
        "escalation_rung_reason": rung["why"],
        "seed_search": list(rung["seeds"]),
        "escalation_attempts": attempts,
        "catboost_version": "1.2.10",
        "thread_count": 1,
        "n_rows": rung["n_rows"],
        "n_float": N_FLOAT,
        "cardinalities": list(rung["cards"]),
        "observed_target_border_idxs": sorted(idxs),
        "overflow_margin": (
            "n*max(w)*max(|der1|) <= 64*1.0*1.0 = 64 << 2^33 ~= 8.6e9 (margin > 1.3e8x) "
            "— Logloss der1 is bounded by 1 in magnitude"
        ),
        "description": (
            "Buckets-CTR device-parity fixture (DCTR-08). 64 rows, two float columns "
            "(border_count=15, FULL border set frozen as borders.npy) plus small-"
            "cardinality cat column(s) routed to CTR (one_hot_max_size=1), with "
            "simple_ctr=Buckets:Prior=0.5, combinations_ctr=[] and max_ctr_complexity=1 so "
            "only SIMPLE Buckets columns can form. The float columns are mandatory: a "
            "cat-only pool can never reach the device. The committed model carries Buckets "
            "splits at BOTH target_border_idx 0 and 1 (asserted), which is the only thing "
            "in the phase that exercises the device Buckets@1 numerator "
            "(good = counts[1], SPEC 4.2). NEVER regenerated in CI."
        ),
        "params": params,
        "npy_schema": {
            "borders.npy": f"[{N_FLOAT},15] f64 — FULL frozen float border set (feed to train_cat)",
            "X.npy": f"[N,{N_FLOAT}] f32 — float feature matrix",
            "X_cat.npy": cat_schema,
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
        f"wrote {SCENARIO}: rung={rung['rung']}, seed={seed}, cards={list(rung['cards'])}, "
        f"iterations={params['iterations']}, ctr descriptors={len(ctrs)}, "
        f"target_border_idxs={sorted(idxs)}, predictions.std()={predictions.std():.6f}"
    )


if __name__ == "__main__":
    main()
