"""Freeze the Langevin / SGLB parity facts against catboost 1.2.10.

Stochastic Gradient Langevin Boosting adds seeded Gaussian noise to the
gradients. Upstream injects it at TWO places and couples it to model shrinkage,
so "langevin works" is really four separable claims:

# 1. The option RESOLUTION

`langevin=True` alone selects `diffusion_temperature = 10000` AND
`model_shrink_rate = 0.001`. That second one is not obvious and is the whole
reason a `diffusion_temperature = 0` fit still differs from the default: the
shrink is doing it, not the noise. Supplying a temperature turns Langevin ON by
itself; an explicit `langevin=False` keeps it off and makes the fit
bit-identical to the default. An explicit `model_shrink_rate` OVERRIDES the
0.001 default -- including an explicit `0.0`.

# 2. The NOISE RATE

`sqrt(2 / (learning_rate * diffusion_temperature))`, so a HIGHER temperature
means LESS noise. The `dt` sweep cells pin the whole curve rather than one point.

# 3. BOTH injection sites

The derivative noise changes the tree STRUCTURE; the leaf-sum noise changes the
LEAF VALUES. A `depth=0`-ish cell cannot separate them, so instead the Newton
cell exercises the `sqrt(|SumDer2| + l2)` leaf scale and the
`leaf_estimation_iterations>1` cell exercises the one-seed-per-step rule.

# 4. `posterior_sampling`

A preset deriving `diffusion_temperature = n` and `model_shrink_rate = 1/(2n)`
from the LEARN SET SIZE, overriding an explicitly supplied shrink rate.

# Vacuity guards

The generator REFUSES to write a fixture whose claims are not exhibited: the
inert cells must be exactly equal, every temperature must give a distinct model,
and the `dt=0` cell must equal the shrink-only replay (which is what proves
claim 1).

Run:  python3 crates/cb-oracle/generator/gen_langevin_fixtures.py
"""

import json
import os

import numpy as np
from catboost import CatBoostRegressor

CATBOOST_VERSION = "1.2.10"

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "fixtures")
OUT_DIR = os.path.join(FIXTURES, "langevin")

SEED = 20260815
N_TRAIN = 300
N_FEATURES = 5
N_EVAL = 12

#: Pinned EXPLICITLY on both sides. `model_shrink_rate` is deliberately ABSENT --
#: leaving it unset is exactly what lets Langevin's 0.001 default fire, which is
#: fact 1. Pinning it here would silence the very behaviour under test.
BASE = dict(
    iterations=5,
    depth=3,
    learning_rate=0.3,
    l2_leaf_reg=3.0,
    random_strength=0,
    leaf_estimation_iterations=1,
    score_function="L2",
    leaf_estimation_method="Gradient",
    random_seed=0,
    thread_count=1,
    border_count=32,
    boost_from_average=False,
    bootstrap_type="No",
    grow_policy="SymmetricTree",
    boosting_type="Plain",
    verbose=False,
    allow_writing_files=False,
)

#: The temperature sweep. Each must yield a distinct model (guard below).
TEMPERATURES = (1.0, 100.0, 10000.0)

#: `f32(0.001)` -- upstream stores model_shrink_rate as a float.
LANGEVIN_DEFAULT_SHRINK = float(np.float32(0.001))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N_TRAIN, N_FEATURES)).astype(np.float64)
    y = (
        2.0 * x[:, 0] - 1.3 * x[:, 1] + 0.9 * x[:, 2] - 0.6 * x[:, 3] + 0.4 * x[:, 4]
    ).astype(np.float64)
    x_eval = np.ascontiguousarray(rng.normal(size=(N_EVAL, N_FEATURES)), dtype=np.float64)

    np.save(os.path.join(OUT_DIR, "X.npy"), x)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUT_DIR, "X_eval.npy"), x_eval)

    def fit(**over):
        m = CatBoostRegressor(**dict(BASE, **over))
        m.fit(x, y)
        return m

    def preds(**over):
        return np.asarray(
            fit(**over).predict(x_eval, prediction_type="RawFormulaVal"), dtype=np.float64
        )

    def resolved(**over):
        ap = fit(**over).get_all_params()
        return {
            k: ap.get(k)
            for k in ("langevin", "diffusion_temperature", "model_shrink_rate",
                      "model_shrink_mode", "posterior_sampling")
            if k in ap
        }

    meta = {
        "scenario": "langevin",
        "catboost_version": CATBOOST_VERSION,
        "seed": SEED,
        "n_train": N_TRAIN,
        "params": BASE,
        "temperatures": list(TEMPERATURES),
        "langevin_default_shrink_rate": LANGEVIN_DEFAULT_SHRINK,
    }

    base = preds()
    np.save(os.path.join(OUT_DIR, "preds_default.npy"), base)

    # ---- Fact 1: option resolution -------------------------------------------
    meta["resolved"] = {
        "langevin_only": resolved(langevin=True),
        "posterior_sampling": resolved(posterior_sampling=True),
        "langevin_explicit_shrink": resolved(langevin=True, model_shrink_rate=0.5),
    }
    r = meta["resolved"]["langevin_only"]
    if abs(r["model_shrink_rate"] - LANGEVIN_DEFAULT_SHRINK) > 1e-12:
        raise AssertionError(
            "langevin=True must select model_shrink_rate=%r; got %r"
            % (LANGEVIN_DEFAULT_SHRINK, r["model_shrink_rate"])
        )
    if r["diffusion_temperature"] != 10000:
        raise AssertionError("langevin=True must select diffusion_temperature=10000")

    # `langevin=False` + a temperature is INERT.
    inert = float(np.max(np.abs(preds(langevin=False, diffusion_temperature=1.0) - base)))
    if inert != 0.0:
        raise AssertionError(
            "langevin=False with a temperature must be bit-identical to the default; "
            "got max|diff| = %g" % inert
        )
    meta["langevin_false_is_inert_max_abs_diff"] = inert

    # `dt = 0` is EXACTLY the shrink-only fit -- the proof that Langevin's
    # observable effect at zero temperature is model shrinkage, not noise.
    zero_temp = preds(langevin=True, diffusion_temperature=0.0)
    shrink_only = preds(model_shrink_rate=LANGEVIN_DEFAULT_SHRINK)
    d = float(np.max(np.abs(zero_temp - shrink_only)))
    if d != 0.0:
        raise AssertionError(
            "langevin@dt=0 must equal the shrink-only replay; got max|diff| = %g" % d
        )
    np.save(os.path.join(OUT_DIR, "preds_zero_temperature.npy"), zero_temp)
    meta["zero_temperature_equals_shrink_only_max_abs_diff"] = d
    print("fact 1 OK: langevin selects dt=10000 + shrink=%.10g; dt=0 == shrink-only"
          % LANGEVIN_DEFAULT_SHRINK)

    # ---- Fact 2: the temperature sweep ---------------------------------------
    seen = {}
    for t in TEMPERATURES:
        p = preds(langevin=True, diffusion_temperature=t)
        tag = ("%g" % t).replace(".", "p")
        np.save(os.path.join(OUT_DIR, "preds_dt_%s.npy" % tag), p)
        for prev_t, prev_p in seen.items():
            if float(np.max(np.abs(p - prev_p))) == 0.0:
                raise AssertionError(
                    "dt=%g and dt=%g give IDENTICAL models -- VACUOUS cells" % (t, prev_t)
                )
        seen[t] = p
    meta["temperature_tags"] = {"%g" % t: ("%g" % t).replace(".", "p") for t in TEMPERATURES}
    # The noise must SHRINK as the temperature rises (the sqrt(2/(lr*dt)) law).
    spread = {("%g" % t): float(np.max(np.abs(p - base))) for t, p in seen.items()}
    ordered = [spread["%g" % t] for t in TEMPERATURES]
    if not (ordered[0] > ordered[-1]):
        raise AssertionError(
            "noise must DECREASE with temperature; got spreads %r" % (ordered,)
        )
    meta["deviation_from_default_by_temperature"] = spread
    print("fact 2 OK: %d distinct temperatures, deviation falls %.4g -> %.4g"
          % (len(TEMPERATURES), ordered[0], ordered[-1]))

    # ---- Fact 3: both injection sites -----------------------------------------
    # Newton exercises the `sqrt(|SumDer2| + l2)` leaf-sum scale.
    np.save(
        os.path.join(OUT_DIR, "preds_newton.npy"),
        preds(langevin=True, diffusion_temperature=100.0, leaf_estimation_method="Newton"),
    )
    # >1 leaf-estimation step exercises the one-seed-PER-STEP rule.
    np.save(
        os.path.join(OUT_DIR, "preds_leaf_iters3.npy"),
        preds(langevin=True, diffusion_temperature=100.0, leaf_estimation_iterations=3),
    )
    # An explicit model_shrink_rate OVERRIDES the 0.001 default.
    np.save(
        os.path.join(OUT_DIR, "preds_explicit_shrink.npy"),
        preds(langevin=True, diffusion_temperature=100.0, model_shrink_rate=0.0),
    )
    print("fact 3 OK: wrote Newton / multi-step / explicit-shrink cells")

    # ---- Fact 3b: the SINGLE-TREE cells ----------------------------------------
    # A one-iteration fit isolates the FIRST tree, i.e. both noise sites acting on
    # a known starting approx with no accumulated cross-tree RNG phase. These are
    # the cells the engine currently reproduces exactly; the multi-tree cells above
    # are the ones whose per-tree phase is still open. Keeping them separate is the
    # point -- it says precisely how far parity reaches.
    for t in TEMPERATURES:
        tag = ("%g" % t).replace(".", "p")
        np.save(
            os.path.join(OUT_DIR, "preds_iters1_dt_%s.npy" % tag),
            preds(iterations=1, langevin=True, diffusion_temperature=t),
        )
    one_base = preds(iterations=1)
    np.save(os.path.join(OUT_DIR, "preds_iters1_default.npy"), one_base)
    for t in TEMPERATURES:
        p = preds(iterations=1, langevin=True, diffusion_temperature=t)
        if float(np.max(np.abs(p - one_base))) == 0.0:
            raise AssertionError(
                "a 1-iteration langevin fit at dt=%g is identical to the default, so the "
                "single-tree cell is VACUOUS" % t
            )
    print("fact 3b OK: wrote single-tree (iterations=1) cells for %s" % (TEMPERATURES,))

    # ---- Fact 4: posterior_sampling -------------------------------------------
    ps = resolved(posterior_sampling=True)
    if ps["diffusion_temperature"] != N_TRAIN:
        raise AssertionError(
            "posterior_sampling must set diffusion_temperature = n (%d); got %r"
            % (N_TRAIN, ps["diffusion_temperature"])
        )
    expected_shrink = float(np.float32(1.0 / (2.0 * N_TRAIN)))
    if abs(ps["model_shrink_rate"] - expected_shrink) > 1e-12:
        raise AssertionError(
            "posterior_sampling must set model_shrink_rate = 1/(2n) = %r; got %r"
            % (expected_shrink, ps["model_shrink_rate"])
        )
    np.save(os.path.join(OUT_DIR, "preds_posterior_sampling.npy"),
            preds(posterior_sampling=True))
    # It overrides an EXPLICIT shrink rate (unlike plain langevin).
    a = preds(posterior_sampling=True)
    b = preds(posterior_sampling=True, model_shrink_rate=0.5)
    d = float(np.max(np.abs(a - b)))
    if d != 0.0:
        raise AssertionError(
            "posterior_sampling must override an explicit model_shrink_rate; got %g" % d
        )
    meta["posterior_overrides_explicit_shrink_max_abs_diff"] = d
    meta["posterior_expected_shrink_rate"] = expected_shrink
    print("fact 4 OK: posterior_sampling -> dt=%d, shrink=%.10g, overrides explicit shrink"
          % (N_TRAIN, expected_shrink))

    # ---- Refusals --------------------------------------------------------------
    rejections = {}
    for label, kw in (
        ("posterior_without_langevin", dict(posterior_sampling=True, langevin=False)),
        ("posterior_with_temperature", dict(posterior_sampling=True, diffusion_temperature=7)),
        ("posterior_with_decreasing_shrink",
         dict(posterior_sampling=True, model_shrink_mode="Decreasing")),
    ):
        try:
            fit(**kw)
            raise AssertionError("%s must be refused by upstream" % label)
        except AssertionError:
            raise
        except Exception as exc:
            rejections[label] = " ".join(str(exc).split())
    meta["rejections"] = rejections
    print("refusals OK: %s" % ", ".join(rejections))

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print("\nwrote %s" % OUT_DIR)


if __name__ == "__main__":
    main()
