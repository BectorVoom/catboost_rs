"""Seeded synthetic data + serial-reference generator for the Kaggle CUDA oracle.

Single source (D-06)
--------------------
The SAME generator produces BOTH:

  * the small-n **depth-1 correctness fixture** (the <=1e-5 device-vs-CPU oracle,
    plus bit-exact primitive/cindex references), and
  * the large-n (~1e6x50) **wall-clock speed workload**.

pinned to one seeded function so correctness and speed can never drift apart.

Design constraints
------------------
* **Dependency-light:** numpy only (already used by ``benchmark.py``). No external
  download, no secrets, no environment variables.
* **Deterministic:** a fixed seed yields byte-identical ``X`` / ``y`` on every run and
  on every machine. We use the legacy ``numpy.random.RandomState`` (Mersenne-Twister)
  rather than the new ``default_rng`` Generator, because the legacy stream is
  guaranteed stable across numpy versions -- so a fixture committed here reproduces
  bit-for-bit on the Kaggle CUDA image regardless of its numpy version.

Why two configs from one function
---------------------------------
A depth-1 tree is the single most launch-overhead-bound workload in the milestone
(D-10-09). At small n the CPU grows a stump in microseconds and the device *cannot*
win -- that is physics, not a tuning gap. So:

  * ``CORRECTNESS_CONFIG`` (small n) is what the <=1e-5 oracle runs on -- cheap, every
    serial reference trivially checkable.
  * ``SPEED_CONFIG`` (~1e6x50, tunable ABOVE the launch-overhead break-even) is where
    the O(n*features) histogram parallelizes enough to amortize fixed GPU launch
    latency -- the ONLY regime where depth-1 device >= CPU is achievable.

Both come out of :func:`generate` with a different ``(n_rows, n_features)`` -- the
D-06 single-source rule.
"""

import argparse
import hashlib
import json
import os

import numpy as np

# --------------------------------------------------------------------------------------
# Canonical configs (single source, D-06)
# --------------------------------------------------------------------------------------

#: Small-n depth-1 correctness fixture: <=1e-5 device-vs-CPU oracle + primitive/cindex
#: references. Small enough to commit and to eyeball; the epsilon bar is what matters.
CORRECTNESS_CONFIG = dict(n_rows=2000, n_features=10, seed=42)

#: Large-n speed workload (~1e6x50). Tunable ABOVE the launch-overhead break-even
#: (D-10-09): the Kaggle run is the arbiter of the exact crossover. NOT committed as a
#: fixture -- regenerated from ``seed`` on the fly (see fixtures/README.md).
SPEED_CONFIG = dict(n_rows=1_000_000, n_features=50, seed=42)

#: Number of quantization borders per feature for the bit-packed cindex reference.
CINDEX_N_BINS = 32

#: L2 leaf regularization used by the depth-1 serial reference (CatBoost default 3.0).
DEPTH1_L2 = 3.0

#: Learning rate used by the depth-1 serial reference (matches benchmark.py).
DEPTH1_LEARNING_RATE = 0.1


# --------------------------------------------------------------------------------------
# Core seeded generator (D-06 single source)
# --------------------------------------------------------------------------------------


def generate(n_rows, n_features, seed=42, dtype=np.float32):
    """Deterministically generate a regression design matrix and linear-plus-noise target.

    Mirrors ``benchmark.py``'s seeded ``randn`` + linear target with 0.1 gaussian noise,
    parameterized on ``n_rows`` / ``n_features`` so ONE function feeds both the
    correctness fixture and the speed workload (D-06).

    Returns
    -------
    X : np.ndarray, shape (n_rows, n_features), dtype ``dtype``
    y : np.ndarray, shape (n_rows,), dtype ``dtype``   (continuous regression target)
    """
    rng = np.random.RandomState(seed)  # legacy Mersenne -> version-stable bytes
    X = rng.randn(n_rows, n_features).astype(dtype)
    weights = rng.randn(n_features).astype(dtype)
    noise = (rng.randn(n_rows) * 0.1).astype(dtype)
    y = (X.dot(weights) + noise).astype(dtype)
    return X, y


def binary_target(X, seed=42, dtype=np.float32):
    """Deterministic {0,1} target for the Logloss depth-1 fixture.

    A separate, seed-derived linear score is thresholded through a logistic link so the
    two classes are separable-but-noisy (a well-conditioned Logloss oracle input).
    """
    n_rows, n_features = X.shape
    rng = np.random.RandomState(seed + 1)  # decorrelate from the regression weights
    w = rng.randn(n_features).astype(np.float64)
    score = X.astype(np.float64).dot(w)
    prob = 1.0 / (1.0 + np.exp(-score))
    draws = rng.rand(n_rows)
    return (draws < prob).astype(dtype)


# --------------------------------------------------------------------------------------
# Serial CPU references (the "expected values" the device oracle is checked against)
#
# These are intentionally the SIMPLEST correct serial implementations -- their job is to
# be obviously-correct, not fast. Integer/index primitives are bit-exact; float reduces
# accumulate in float64 (mirroring the device f64 re-accumulation, SPIKE-REDUCTION.md).
# --------------------------------------------------------------------------------------


def serial_inclusive_scan(values):
    """Inclusive prefix sum (f64 accumulate). Device scan oracle target."""
    return np.cumsum(values.astype(np.float64))


def serial_exclusive_scan(values):
    """Exclusive prefix sum (f64 accumulate)."""
    inc = np.cumsum(values.astype(np.float64))
    return inc - values.astype(np.float64)


def serial_segmented_scan(values, seg_heads):
    """Inclusive scan that RESETS at each segment head (``seg_heads[i] == 1``).

    Device segmented-scan oracle target. ``seg_heads`` is a 0/1 mask, 1 at the first
    element of each segment.
    """
    values = values.astype(np.float64)
    out = np.empty_like(values)
    acc = 0.0
    for i in range(values.shape[0]):
        acc = values[i] if seg_heads[i] else acc + values[i]
        out[i] = acc
    return out


def serial_sort_reorder(keys):
    """Stable ascending argsort of integer keys. Device radix-sort/reorder oracle target.

    Returns the permutation (bit-exact against a stable device sort).
    """
    return np.argsort(keys, kind="stable").astype(np.int64)


def serial_reduce_by_key(keys, values):
    """Sum ``values`` grouped by CONSECUTIVE equal ``keys`` (f64 accumulate).

    Returns ``(unique_keys, summed_values)`` -- the device reduce-by-key oracle target.
    """
    values = values.astype(np.float64)
    out_keys = []
    out_vals = []
    for k, v in zip(keys, values):
        if out_keys and out_keys[-1] == k:
            out_vals[-1] += v
        else:
            out_keys.append(int(k))
            out_vals.append(float(v))
    return np.array(out_keys, dtype=np.int64), np.array(out_vals, dtype=np.float64)


def serial_segmented_reduce(values, offsets):
    """Sum ``values`` within segments defined by ``offsets`` (CSR-style, f64 accumulate).

    ``offsets`` has ``n_segments + 1`` entries; segment ``s`` is
    ``values[offsets[s]:offsets[s+1]]``. Device segmented-reduce oracle target.
    """
    values = values.astype(np.float64)
    return np.array(
        [values[offsets[s] : offsets[s + 1]].sum() for s in range(len(offsets) - 1)],
        dtype=np.float64,
    )


def compute_borders(X, n_bins=CINDEX_N_BINS):
    """Per-feature ascending quantile borders (uniform-quantile bordering).

    NOTE: this is a *self-consistent* reference bordering for the primitive/cindex
    fixtures -- NOT a claim of bit-parity with CatBoost's GreedyLogSum. The AUTHORITATIVE
    depth-1 oracle in the notebook is device-vs-Rust-CPU (both use the Rust bordering);
    this numpy bordering only backs the standalone cindex bit-packing reference.
    """
    n_features = X.shape[1]
    qs = np.linspace(0.0, 1.0, n_bins + 2)[1:-1]  # interior quantiles -> n_bins borders
    return [np.quantile(X[:, f].astype(np.float64), qs) for f in range(n_features)]


def quantize(X, borders):
    """Map each value to its bin index = count of borders it strictly exceeds.

    ``bin > bin_id  <=>  value > borders[bin_id]`` -- the exact round-trip the device
    ``bin_id -> border`` join relies on (10-08 key-decision). Bit-exact reference.
    Returns an ``int32`` cindex of shape ``X.shape``.
    """
    n_features = X.shape[1]
    cindex = np.empty(X.shape, dtype=np.int32)
    for f in range(n_features):
        cindex[:, f] = np.searchsorted(borders[f], X[:, f].astype(np.float64), side="right")
    return cindex


def bitpack_cindex(cindex, bits):
    """Pack a per-feature cindex column into ``bits``-wide fields of a uint32 stream.

    Bit-exact reference for the GPUT-15 bit-packed cindex layout: feature-major, each
    object's bin packed into a ``bits``-wide little-endian field. ``bits`` must divide 32.
    """
    assert 32 % bits == 0, "bits must divide 32 for this reference packer"
    per_word = 32 // bits
    n_rows, n_features = cindex.shape
    packed = []
    for f in range(n_features):
        col = cindex[:, f].astype(np.uint32)
        n_words = (n_rows + per_word - 1) // per_word
        words = np.zeros(n_words, dtype=np.uint32)
        for i in range(n_rows):
            w, slot = divmod(i, per_word)
            words[w] |= (col[i] & ((1 << bits) - 1)) << (slot * bits)
        packed.append(words)
    return packed


def serial_depth1_tree(X, y, weights, loss, borders,
                       l2=DEPTH1_L2, learning_rate=DEPTH1_LEARNING_RATE):
    """Serial depth-1 oblivious-tree reference: Cosine-score best split + calc_average leaves.

    Matches the device depth-1 path (10-08 contract):

    * **der1** (first-order): RMSE -> ``der1 = y - approx`` (approx == 0 here);
      Logloss -> ``der1 = y - sigmoid(approx) = y - 0.5``. Newton der2 is Phase 11 --
      the Logloss reference is pinned to FIRST-ORDER ``calc_average`` leaves
      (RESEARCH line 318 / CONTEXT scope anchor), NOT Newton.
    * **Cosine score** of a split = ``sum_leaf (sum_der^2 / (count + l2))``; the best
      (feature, bin_id) maximizes it.
    * **leaf value** (calc_average, first-order) = ``learning_rate * sum_der / (count + l2)``,
      UN-centered (RMSE/Logloss). ``bin > bin_id`` goes right (forward bit).

    Returns a dict: ``best_feature``, ``best_bin``, ``best_border``, ``leaf_left``,
    ``leaf_right``, ``score``.
    """
    y = y.astype(np.float64)
    weights = weights.astype(np.float64)
    if loss == "rmse":
        der1 = y  # approx == 0, weight folded as count below
    elif loss == "logloss":
        der1 = y - 0.5  # sigmoid(0) == 0.5
    else:
        raise ValueError("loss must be 'rmse' or 'logloss'")
    der1 = der1 * weights

    n_features = X.shape[1]
    best = None
    for f in range(n_features):
        b = borders[f]
        col = X[:, f].astype(np.float64)
        for bin_id in range(len(b)):
            right = col > b[bin_id]
            left = ~right
            wl = weights[left].sum()
            wr = weights[right].sum()
            if wl <= 0.0 or wr <= 0.0:
                continue  # degenerate split, skip
            sl = der1[left].sum()
            sr = der1[right].sum()
            score = sl * sl / (wl + l2) + sr * sr / (wr + l2)
            if best is None or score > best["score"]:
                best = dict(
                    best_feature=int(f),
                    best_bin=int(bin_id),
                    best_border=float(b[bin_id]),
                    score=float(score),
                    leaf_left=float(learning_rate * sl / (wl + l2)),
                    leaf_right=float(learning_rate * sr / (wr + l2)),
                )
    if best is None:
        raise ValueError("no non-degenerate depth-1 split found")
    return best


# --------------------------------------------------------------------------------------
# Fixture emission + commit-discipline manifest
# --------------------------------------------------------------------------------------


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_fixtures(out_dir):
    """Emit the committed small-n correctness fixtures + a sha256 manifest.

    ONLY the small-n correctness fixtures are committed; the large-n speed workload is
    regenerated from :data:`SPEED_CONFIG` on the fly (never committed). The manifest is
    the commit-discipline contract: a changed sha means the generator changed and the
    fixtures must be regenerated + re-reviewed.
    """
    os.makedirs(out_dir, exist_ok=True)
    cfg = CORRECTNESS_CONFIG
    X, y_reg = generate(cfg["n_rows"], cfg["n_features"], cfg["seed"])
    y_bin = binary_target(X, cfg["seed"])
    _, gen_weights = (None, np.random.RandomState(cfg["seed"]).randn(cfg["n_features"]))
    weights = np.ones(cfg["n_rows"], dtype=np.float64)  # unit object weights

    borders = compute_borders(X, CINDEX_N_BINS)
    cindex = quantize(X, borders)

    # Primitive reference inputs (deterministic small integer/float streams).
    rng = np.random.RandomState(cfg["seed"] + 7)
    prim_vals = rng.randn(256).astype(np.float64)
    seg_heads = np.zeros(256, dtype=np.int32)
    seg_heads[::32] = 1  # a segment every 32 elements
    keys = np.repeat(np.arange(8, dtype=np.int64), 32)  # 8 consecutive runs of 32
    offsets = np.arange(0, 257, 32, dtype=np.int64)

    files = {}

    def _save(name, arr):
        path = os.path.join(out_dir, name)
        np.save(path, arr)
        files[name if name.endswith(".npy") else name + ".npy"] = None

    # Correctness inputs.
    _save("X_small.npy", X)
    _save("y_small_reg.npy", y_reg)
    _save("y_small_bin.npy", y_bin)
    _save("cindex_small.npy", cindex)

    # Primitive reference inputs + serial expected values.
    _save("prim_values.npy", prim_vals)
    _save("prim_seg_heads.npy", seg_heads)
    _save("prim_keys.npy", keys)
    _save("prim_offsets.npy", offsets)
    _save("expected_inclusive_scan.npy", serial_inclusive_scan(prim_vals))
    _save("expected_exclusive_scan.npy", serial_exclusive_scan(prim_vals))
    _save("expected_segmented_scan.npy", serial_segmented_scan(prim_vals, seg_heads))
    _save("expected_sort_perm.npy", serial_sort_reorder(keys[::-1]))
    rbk_k, rbk_v = serial_reduce_by_key(keys, prim_vals)
    _save("expected_reduce_by_key_keys.npy", rbk_k)
    _save("expected_reduce_by_key_vals.npy", rbk_v)
    _save("expected_segmented_reduce.npy", serial_segmented_reduce(prim_vals, offsets))

    # cindex bit-packing reference (bits=8 -> 4 bins-per-word, CINDEX_N_BINS=32 fits 8 bits).
    packed = bitpack_cindex(cindex[:, :1], bits=8)  # single feature column, illustrative
    _save("expected_cindex_packed_f0_bits8.npy", packed[0])

    # Depth-1 tree references (RMSE + Logloss, first-order calc_average leaves).
    depth1 = dict(
        config=cfg,
        l2=DEPTH1_L2,
        learning_rate=DEPTH1_LEARNING_RATE,
        n_bins=CINDEX_N_BINS,
        rmse=serial_depth1_tree(X, y_reg, weights, "rmse", borders),
        logloss=serial_depth1_tree(X, y_bin, weights, "logloss", borders),
    )
    depth1_path = os.path.join(out_dir, "expected_depth1_tree.json")
    with open(depth1_path, "w") as fh:
        json.dump(depth1, fh, indent=2, sort_keys=True)
    files["expected_depth1_tree.json"] = None

    # Manifest: shapes/seeds + sha256 of every committed fixture (commit-discipline gate).
    manifest = dict(
        generator="bench/generator.py",
        correctness_config=cfg,
        speed_config=SPEED_CONFIG,
        cindex_n_bins=CINDEX_N_BINS,
        depth1_l2=DEPTH1_L2,
        depth1_learning_rate=DEPTH1_LEARNING_RATE,
        note=(
            "Only small-n correctness fixtures are committed; the large-n speed "
            "workload is regenerated from speed_config on the fly. A changed sha256 "
            "means the generator changed -- regenerate and re-review."
        ),
        sha256={name: _sha256(os.path.join(out_dir, name)) for name in sorted(files)},
    )
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest


def _main():
    ap = argparse.ArgumentParser(description="Seeded synthetic + serial-reference generator.")
    ap.add_argument("--write", metavar="DIR", help="write committed fixtures into DIR")
    ap.add_argument("--check", metavar="DIR", help="verify DIR fixtures match a fresh regen")
    args = ap.parse_args()
    if args.write:
        manifest = write_fixtures(args.write)
        print(f"wrote {len(manifest['sha256'])} fixtures to {args.write}")
    elif args.check:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            fresh = write_fixtures(td)
        committed = json.load(open(os.path.join(args.check, "manifest.json")))
        mismatch = {
            k: (committed["sha256"].get(k), v)
            for k, v in fresh["sha256"].items()
            if committed["sha256"].get(k) != v
        }
        if mismatch:
            print("MISMATCH -- fixtures drifted from generator:")
            for k, (a, b) in mismatch.items():
                print(f"  {k}: committed={a} regen={b}")
            raise SystemExit(1)
        print(f"OK -- {len(fresh['sha256'])} fixtures reproduce bit-for-bit")
    else:
        # Smoke: prove determinism (same seed -> same bytes).
        X1, y1 = generate(**CORRECTNESS_CONFIG)
        X2, y2 = generate(**CORRECTNESS_CONFIG)
        assert (X1.tobytes() == X2.tobytes()) and (y1.tobytes() == y2.tobytes())
        print("generator.py deterministic OK; run with --write DIR to emit fixtures")


if __name__ == "__main__":
    _main()
