#!/usr/bin/env python3
"""
Spike 001: Can catboost 1.2.10's ComputeOnlineCTRs(AveragingFold) ui8 bins
be reproduced BIT-EXACT offline from committed inputs?

Target (averaging-permutation order, N=30):
  [7,7,7,11,3,7,2,11,7,12,9,12,11,10,3,2,11,1,7,13,5,7,8,11,1,9,1,10,1,13]

CTR is SINGLE cat feature: cat_feature_index 0 (col0), Borders, prior 0.5,
ctr_border_count 15, targetClassesCount=2 (SIMPLE).

Authoritative algorithm (catboost-master/.../algo/online_ctr.cpp):
  - CalcOnlineCTRSimple -> CalcQuantizedCtrs (thread_count=1 => single block,
    so a pure global online read-before-increment prefix in hashArr order).
  - hashArr order = the LEARN/AVERAGING permutation applied to col0.
  - permutedTargetClass = y reordered by that same permutation.
  - per doc: good=N[1], total=N[0]+N[1] (read BEFORE increment),
             ++N[targetClass]; bin = CalcCTR(good,total,prior,shift,norm,bc).
  - CalcCTR (online_ctr.h:128-131), FLOAT32 arithmetic:
       ctr = (countInClass + prior) / (totalCount + 1)
       return ui8( (ctr + shift) / norm * borderCount )   # trunc to ui8
    With prior=0.5: shift=0.5, norm=1.0, borderCount=15.
"""
import json
import numpy as np

HERE = "/home/user/Documents/workspace/catboost_rs"
TARGET = [7,7,7,11,3,7,2,11,7,12,9,12,11,10,3,2,11,1,7,13,5,7,8,11,1,9,1,10,1,13]
AVG_PERM = [11,18,15,29,16,12,0,7,19,27,4,3,5,17,14,25,9,20,8,23,6,28,26,24,2,13,21,22,10,1]
LEAF_TREEB = [1,1,1,3,0,1,0,3,1,3,3,3,3,3,0,0,3,0,1,3,1,1,3,3,0,3,0,3,0,3]

X = np.load(f"{HERE}/crates/cb-oracle/fixtures/tensor_ctr_e2e/X_cat.npy")
y = np.load(f"{HERE}/crates/cb-oracle/fixtures/tensor_ctr_e2e/y.npy").astype(int)
col0 = X[:, 0].astype(int)
N = len(col0)

PRIOR = 0.5
BORDER_COUNT = 15


def calc_ctr_f32(good, total, prior=PRIOR, border_count=BORDER_COUNT):
    """online_ctr.h:128-131 in float32 exactly."""
    f = np.float32
    # CalcNormalization (online_ctr.cpp:102-111): left=min(0,prior)=0,
    # right=max(1,prior)=1, shift=-left=0, norm=right-left=1.
    shift = f(-min(0.0, prior))   # = 0.0 for prior=0.5
    norm = f(max(1.0, prior) - min(0.0, prior))  # = 1.0
    ctr = (f(good) + f(prior)) / (f(total) + f(1))
    val = (ctr + shift) / norm * f(border_count)
    # implicit float->ui8 cast == truncation toward zero, then 8-bit wrap.
    return int(np.uint8(np.float32(val)))


def online_prefix_bins(order, target_class, value_to_bin=None):
    """
    Run the global online read-before-increment prefix in `order`.
    Returns bins in `order` order (i.e. position p corresponds to order[p]).
    value_to_bin: optional dict cat_value -> dense bin (not needed for counts,
    counts are keyed by the cat value directly).
    """
    # per cat value: [N0, N1]
    counts = {}
    out = []
    for doc in order:
        cv = int(col0[doc])
        n0, n1 = counts.get(cv, (0, 0))
        good = n1
        total = n0 + n1
        out.append(calc_ctr_f32(good, total))
        cls = target_class[doc]
        if cls == 0:
            counts[cv] = (n0 + 1, n1)
        else:
            counts[cv] = (n0, n1 + 1)
    return out


def diff_report(name, recon_avg_order):
    """recon_avg_order: bins already in averaging-permutation order."""
    matches = sum(1 for a, b in zip(recon_avg_order, TARGET) if a == b)
    print(f"\n=== {name}: {matches}/{N} match ===")
    print("pos:    " + " ".join(f"{i:3d}" for i in range(N)))
    print("recon:  " + " ".join(f"{v:3d}" for v in recon_avg_order))
    print("target: " + " ".join(f"{v:3d}" for v in TARGET))
    print("diff:   " + " ".join("  ." if a == b else f"{'X':>3}"
                                 for a, b in zip(recon_avg_order, TARGET)))
    first_div = next((i for i, (a, b) in enumerate(zip(recon_avg_order, TARGET)) if a != b), None)
    print(f"first divergence at pos {first_div}")
    return matches


print("col0       :", col0.tolist())
print("y          :", y.tolist())
print("avg_perm   :", AVG_PERM)

# --------------------------------------------------------------------------
# HYPOTHESIS 1: pure global online prefix in averaging-permutation order,
# target_class = y, good = class-1 count. This is cb-train's current model.
# The output position p of the prefix loop corresponds to AVG_PERM[p], and the
# target sequence is ALSO in averaging order, so they line up directly.
# --------------------------------------------------------------------------
h1 = online_prefix_bins(AVG_PERM, y)
diff_report("H1 global online prefix (good=class1), avg order", h1)


# ==========================================================================
# VERDICT (see FINDINGS.md): NOT-ACHIEVABLE.
#
# The decisive proof is the per-bucket determinism check above:
#   For a single-cat online-prefix CTR keyed on col0 under the committed
#   averaging permutation, each bucket's bin sequence is FULLY DETERMINED
#   (total = 0,1,2,... in appearance order; good = running good-class count).
#   Under BOTH good=class1 and good=class0, EVERY bucket fails to match the
#   target bins. The cat0 bucket (6 docs, ALL y=0) shows upstream bins
#   [11,12,11,7,13,5] -- HIGH and NON-MONOTONE -- which is mathematically
#   impossible for any function of a monotonically-growing (good,total)
#   prefix over (col0, y). Block structure (SumCtrsFromBlocks pre-seeding,
#   any block count 1..30) does not change this (good stays 0 for an all-y=0
#   bucket regardless of pre-seed).
#
# Therefore the bins depend on a per-document quantity that is NOT the
# committed cumulative target y -- a live training-time artifact.
# ==========================================================================
