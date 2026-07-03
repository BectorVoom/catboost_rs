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

#: Tree depth for the Phase-11 depth-6 RMSE + Logloss correctness fixtures (D-03).
DEPTH6 = 6

#: Newton leaf-estimation iteration count, PINNED to 1 in the fixture (A1 / D-02).
#: The Rust CPU oracle (`newton_leaf_delta`) is a SINGLE closed-form Newton step
#: with NO refinement/backtracking loop; every existing fixture pins this to 1, so
#: the depth-6 Logloss device path has a single-step ε=1e-4 target. If a future
#: fixture ever wants iterations>1, the iterative walker must be built in cb-compute
#: (the oracle) FIRST — a scope expansion, NOT a device-only change (RESEARCH A1).
LEAF_ESTIMATION_ITERATIONS = 1

#: Split-score function PINNED for the depth-6 fixture (A2). BOTH arms score splits
#: with the SAME Cosine function `serial_depth1_tree` uses — the per-leaf Cosine fold
#: whose denominator weight is Σweight (the object count in the unweighted path), NOT
#: Σder2. This is the channel-0 semantics assumption A2: for SCORING the histogram
#: channel-0 carries Σweight (matching the depth-1 reference and the cb-compute CPU
#: oracle); the Logloss Newton HESSIAN (Σder2·weight) enters only the leaf VALUE via
#: `newton_leaf_delta`, not the split score. Downstream device plans cross-check their
#: first-few-tree split scores against this pinned reference.
DEPTH6_SCORE_FUNCTION = "Cosine"

#: The channel-0 semantics the depth-6 score pins (A2) — recorded in the fixture so a
#: later device change to der2-in-score surfaces as an oracle mismatch, not silent drift.
DEPTH6_SCORE_CHANNEL0 = "sum_weight"

#: Large-n depth-6 SPEED workload committed as a reproducible seed artifact
#: (`X_depth6_speed.npy`, sha-manifested) for later BENCH-02 use. This is a
#: committable representative "large-n" array (clearly larger than the 2000x10
#: correctness fixture); the FULL ~1e6x50 speed run is regenerated on the fly from
#: :data:`SPEED_CONFIG` on the Kaggle image (never committed — D-06 / fixtures/README).
DEPTH6_SPEED_CONFIG = dict(n_rows=10_000, n_features=50, seed=42)


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
    # WR-02: the shipped path sums the UNWEIGHTED der into histogram channel 0
    # (`reduce_leaf_stats` -> sum_f64(deltas); device `bin_sums[cell] += der1[obj]`),
    # folding weight only into the leaf/count DENOMINATOR. Do NOT pre-multiply der1 by
    # weights here, or a future weighted fixture would make this reference diverge from
    # the device/CPU result it certifies.

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
                # IN-01 (Phase 11 review): this depth-1 reference uses a STRICTER
                # admissibility rule than the depth-6 reference (`_cosine_split_score`) and
                # the device partition scorer, which BOTH permit an empty side (the `l2>0`
                # denominator keeps `avg = 0/(0+l2) = 0` finite, contributing 0 to the fold).
                # The two rules agree whenever no degenerate-side candidate ever WINS. On the
                # committed 2000x10 gaussian fixture every interior border keeps objects on
                # both sides, so `expected_depth1_tree.json` is unaffected by the difference.
                # This stricter skip is retained deliberately: it keeps the depth-1 reference
                # from ever selecting an all-one-side stump (a no-op split). If a future
                # fixture can produce a degenerate winner, drop this `continue` and score the
                # empty side with the zero-average fold to match the device (then regenerate
                # the pinned fixture under the Kaggle sign-off).
                continue  # degenerate split, skip (see IN-01 note above)
            sl = der1[left].sum()
            sr = der1[right].sum()
            # WR-01: TRUE Cosine score (matching the device Cosine arm
            # `kernels.rs` / find_optimal_split_kernel and score.rs): the L2 fold
            # `folded = Σ avg·sum` divided by `sqrt(1e-100 + Σ avg²·w)`, where
            # `avg = sum / (w + l2)` per leaf. argmax(Cosine) != argmax(L2) in
            # general, so the reference must use the SAME score fn as the device.
            avg_l = sl / (wl + l2)
            avg_r = sr / (wr + l2)
            folded = avg_l * sl + avg_r * sr
            denominator = 1e-100 + avg_l * avg_l * wl + avg_r * avg_r * wr
            score = folded / np.sqrt(denominator)
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


def _cosine_split_score(der1, weights, mask, right, l2):
    """Total Cosine split score of one (feature, bin) candidate over CURRENT partitions.

    ``mask`` selects, per partition, that partition's objects; ``right`` is the global
    ``col > border`` boolean. For every partition the candidate is scored with the SAME
    per-leaf Cosine fold `serial_depth1_tree` uses (``avg = Σder1 / (Σweight + l2)``;
    ``folded = Σ avg·Σder1``; ``denom = 1e-100 + Σ avg²·Σweight``), summed across all
    partitions, and returned as ``folded_total / sqrt(denom_total)``.

    Empty split sides are permitted (an oblivious tree may leave a partition all on one
    side): the ``l2>0`` denominator keeps ``avg = 0/(0+l2) = 0`` finite, contributing 0
    to both sums — so NO candidate is rejected for a degenerate per-partition side (A2 /
    upstream oblivious scoring). The score channel-0 weight is Σweight, NOT Σder2.
    """
    folded_total = 0.0
    denom_total = 1e-100
    rp = mask & right
    lp = mask & ~right
    wl = weights[lp].sum()
    wr = weights[rp].sum()
    sl = der1[lp].sum()
    sr = der1[rp].sum()
    avg_l = sl / (wl + l2)
    avg_r = sr / (wr + l2)
    folded_total += avg_l * sl + avg_r * sr
    denom_total += avg_l * avg_l * wl + avg_r * avg_r * wr
    return folded_total, denom_total


def serial_depth6_tree(X, y, weights, loss, borders,
                       l2=DEPTH1_L2, learning_rate=DEPTH1_LEARNING_RATE,
                       depth=DEPTH6):
    """Serial depth-6 OBLIVIOUS-tree reference (D-03): per-level Cosine best split.

    Mirrors :func:`serial_depth1_tree` but recurses ``depth`` levels. At each level the
    tree is oblivious — ONE ``(feature, bin_id)`` split is chosen for ALL current
    ``2**level`` partitions, maximizing the SUMMED Cosine score across those partitions
    (channel-0 = Σweight, A2). Objects route forward-bit: ``col > border`` adds
    ``2**level`` to the leaf index, growing ``2**(level+1)`` partitions. At the final
    level the ``2**depth`` leaves get their values:

    * **RMSE** leaf value = ``calc_average(Σder1, Σweight, scaled_l2)`` — der2 == −1
      collapses Newton to the average (RMSE der1 = ``y − approx``, approx == 0 here).
    * **Logloss** leaf value = ``newton_leaf_delta(Σder1, Σ(der2·weight), scaled_l2)``
      = ``Σder1 / (−Σ(der2·weight) + scaled_l2)`` — a SINGLE closed-form Newton step
      (``leaf_estimation_iterations == 1``, A1), der1 = ``y − sigmoid(approx) = y − 0.5``,
      der2 = ``−p(1−p) = −0.25``.

    ``scaled_l2`` = :func:`scale_l2_reg`-equivalent = ``l2`` for the unit-weight path
    (``Σweight == n``). Leaf values are stored as RAW deltas (pre-learning-rate) so the
    cb-compute cross-check (Plan 11-01 Task 2) compares them DIRECTLY to
    ``calc_average`` / ``newton_leaf_delta`` output; the boosting loop applies
    ``learning_rate`` downstream (kept in the fixture ``config`` block).

    Returns a dict carrying the 6-level ``splits`` sequence, the ``2**depth``
    ``leaf_values`` (raw deltas), per-leaf reduced ``sum_der1`` / ``sum_weight`` /
    ``sum_der2`` sums, and the per-object ``leaf_of`` / ``der1`` / ``weight`` /
    ``weighted_der2`` arrays the Rust ``reduce_leaf_stats`` / ``reduce_leaf_der2``
    cross-check consumes (so the Rust test needs NO ``.npy`` parser or X routing).
    """
    y = y.astype(np.float64)
    weights = weights.astype(np.float64)
    n = X.shape[0]
    if loss == "rmse":
        der1 = y.copy()  # approx == 0; weight folded into leaf/count denominator only
        der2 = -np.ones(n)  # RMSE hessian == −1 (Newton collapses to calc_average)
    elif loss == "logloss":
        p = np.full(n, 0.5)  # sigmoid(approx == 0) == 0.5
        der1 = y - p
        der2 = -p * (1.0 - p)  # −0.25 per object
    else:
        raise ValueError("loss must be 'rmse' or 'logloss'")
    weighted_der2 = der2 * weights  # the Σ(der2·weight) channel reduce_leaf_der2 consumes

    scaled_l2 = l2 * (weights.sum() / n) if n > 0 else l2  # scale_l2_reg (== l2 unit-weight)

    n_features = X.shape[1]
    cols = [X[:, f].astype(np.float64) for f in range(n_features)]
    leaf_of = np.zeros(n, dtype=np.int64)
    splits = []

    for level in range(depth):
        n_parts = 1 << level
        best = None
        best_score = None
        for f in range(n_features):
            col = cols[f]
            b = borders[f]
            for bin_id in range(len(b)):
                right = col > b[bin_id]
                folded_total = 0.0
                denom_total = 1e-100
                for part in range(n_parts):
                    mask = leaf_of == part
                    folded, denom = _cosine_split_score(der1, weights, mask, right, l2)
                    folded_total += folded
                    denom_total += denom
                score = folded_total / np.sqrt(denom_total)
                if best is None or score > best_score:
                    best = (int(f), int(bin_id), float(b[bin_id]))
                    best_score = float(score)
        if best is None:
            raise ValueError(f"no depth-{depth} split found at level {level}")
        f, bin_id, border = best
        right = cols[f] > border
        leaf_of = leaf_of + (right.astype(np.int64) << level)  # forward-bit routing
        splits.append(dict(level=int(level), feature=int(f), bin=int(bin_id),
                           border=float(border), score=float(best_score)))

    n_leaves = 1 << depth
    leaf_values = []
    per_leaf = []
    for leaf in range(n_leaves):
        mask = leaf_of == leaf
        sum_w = float(weights[mask].sum())
        sum_d1 = float(der1[mask].sum())
        sum_d2 = float(weighted_der2[mask].sum())
        if loss == "rmse":
            # calc_average(Σder1, Σweight, scaled_l2): 0.0 for an empty leaf.
            val = sum_d1 / (sum_w + scaled_l2) if sum_w > 0.0 else 0.0
        else:
            # newton_leaf_delta(Σder1, Σ(der2·weight), scaled_l2) — single closed-form
            # step; denom == 0 (empty leaf, no L2) guards to 0.0 like cb-compute.
            denom = -sum_d2 + scaled_l2
            val = sum_d1 / denom if denom != 0.0 else 0.0
        leaf_values.append(float(val))
        per_leaf.append(dict(sum_der1=sum_d1, sum_weight=sum_w, sum_der2=sum_d2))

    return dict(
        loss=loss,
        depth=int(depth),
        n_leaves=int(n_leaves),
        scaled_l2=float(scaled_l2),
        splits=splits,
        leaf_values=leaf_values,
        per_leaf=per_leaf,
        leaf_of=leaf_of.astype(np.int64).tolist(),
        der1=der1.tolist(),
        weight=weights.tolist(),
        weighted_der2=weighted_der2.tolist(),
    )


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

    # Depth-6 tree references (RMSE calc_average leaves + Logloss single-step Newton
    # leaves), D-03. `config` pins A1 (leaf_estimation_iterations == 1) and A2
    # (score_function / channel-0 = Σweight) so a device drift surfaces as a mismatch.
    depth6 = dict(
        config=dict(
            correctness_config=cfg,
            leaf_estimation_iterations=LEAF_ESTIMATION_ITERATIONS,
            score_function=DEPTH6_SCORE_FUNCTION,
            score_channel0=DEPTH6_SCORE_CHANNEL0,
            l2_leaf_reg=DEPTH1_L2,
            learning_rate=DEPTH1_LEARNING_RATE,
            depth=DEPTH6,
            n_bins=CINDEX_N_BINS,
            seed=cfg["seed"],
            leaf_values_are_raw_deltas=True,
            note=(
                "leaf_values are RAW pre-learning-rate deltas (direct calc_average / "
                "newton_leaf_delta output); the boosting loop multiplies by "
                "learning_rate. A1: single closed-form Newton step (iterations==1). "
                "A2: split score uses the Cosine fn with channel-0 == Σweight; the "
                "Logloss Newton hessian (Σder2·weight) enters ONLY the leaf value."
            ),
        ),
        rmse=serial_depth6_tree(X, y_reg, weights, "rmse", borders),
        logloss=serial_depth6_tree(X, y_bin, weights, "logloss", borders),
    )
    depth6_path = os.path.join(out_dir, "expected_depth6_tree.json")
    with open(depth6_path, "w") as fh:
        json.dump(depth6, fh, indent=2, sort_keys=True)
    files["expected_depth6_tree.json"] = None

    # Large-n depth-6 SPEED workload from the SAME seeded generator (D-03): correctness
    # fixture and speed workload from one source. Committed as a reproducible seed
    # artifact; the FULL ~1e6x50 run is regenerated on the fly from SPEED_CONFIG (never
    # committed). Only the design matrix X is needed for wall-clock timing.
    sp = DEPTH6_SPEED_CONFIG
    X_speed, _ = generate(sp["n_rows"], sp["n_features"], sp["seed"])
    _save("X_depth6_speed.npy", X_speed)

    # Manifest: shapes/seeds + sha256 of every committed fixture (commit-discipline gate).
    manifest = dict(
        generator="bench/generator.py",
        correctness_config=cfg,
        speed_config=SPEED_CONFIG,
        depth6_speed_config=DEPTH6_SPEED_CONFIG,
        cindex_n_bins=CINDEX_N_BINS,
        depth1_l2=DEPTH1_L2,
        depth1_learning_rate=DEPTH1_LEARNING_RATE,
        depth6=DEPTH6,
        leaf_estimation_iterations=LEAF_ESTIMATION_ITERATIONS,
        depth6_score_function=DEPTH6_SCORE_FUNCTION,
        depth6_score_channel0=DEPTH6_SCORE_CHANNEL0,
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
        # No-arg default: prove determinism (same seed -> same bytes), then emit the
        # committed fixtures into the sibling `fixtures/` dir so a bare
        # `python generator.py` (run from `bench/`) reproduces every fixture — the
        # invocation Plan 11-01's verify uses. Explicit `--write DIR` / `--check DIR`
        # remain for out-of-tree emission and the commit-discipline sha diff.
        X1, y1 = generate(**CORRECTNESS_CONFIG)
        X2, y2 = generate(**CORRECTNESS_CONFIG)
        assert (X1.tobytes() == X2.tobytes()) and (y1.tobytes() == y2.tobytes())
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
        manifest = write_fixtures(default_dir)
        print(
            f"generator.py deterministic OK; wrote {len(manifest['sha256'])} fixtures "
            f"to {default_dir}"
        )


if __name__ == "__main__":
    _main()
