# Phase 14: Comprehensive Kaggle CUDA Speed Benchmark + Parity Sign-Off - Research

**Researched:** 2026-07-05
**Domain:** Benchmark scripting + sign-off documentation (NOT source code). Kaggle CUDA harness extension; official-CatBoost-GPU config matching; aggregation of committed per-phase BENCH-02 JSON.
**Confidence:** HIGH (all findings grounded in committed repo artifacts + CatBoost official docs via Context7)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01 (hard gate = ≥20× vs host-CPU baseline; CatBoost-GPU informational):** The only BLOCKING BENCH-03 pass criterion is that the device-resident path beats the pre-Phase-10 host-light CPU baseline by ≥20× across the workload matrix — the original >20× *slowdown* is demonstrably reversed. Phase 12 already recorded 30–42× device≫host-CPU. The head-to-head vs official CatBoost GPU is recorded and discussed as **INFORMATIONAL context only**; any shortfall is documented, not a milestone blocker.
- **D-02 (synthetic large-n only — no external dataset staging):** The matrix reuses the existing large-n synthetic workloads spanning the loss families already timed per-phase (at minimum depth-6 RMSE + Logloss; extend to ranking/multiclass where a per-phase BENCH-02 number exists). **No Higgs/Epsilon or other external named datasets.**
- **D-03 (aggregate committed per-phase results + one new run that adds ONLY CatBoost-GPU):** Roll up already-committed per-phase BENCH-02 `result.json` files into one sign-off document. The **one new human-gated Kaggle CUDA run** adds **only** the missing **official CatBoost GPU** timing on the *same synthetic large-n configs* the per-phase runs used. Each number MUST be labeled with its source run/session (hardware, date) so mixed-session provenance is explicit.
- **D-04 (assume green; speed-only; oracle as pre-flight only — do NOT own GPUT-14):** The correctness oracle runs as a BLOCKING pre-flight before any timing, but Phase 14 does **NOT** flip GPUT-14 to satisfied, and does **NOT** backfill the still-`TBD` depth-1/depth-6 oracle table in `bench/RESULTS.md`. That standing-gate/backfill debt is explicitly **out of scope for Phase 14** — flag for milestone-close audit, not this speed phase.

### Claude's Discretion
- **Official CatBoost GPU config-matching** — the exact parameter mapping between catboost-rs and the official `catboost` Python package (depth / iterations / learning rate / bootstrap / grow-policy / border_count). Note where defaults differ; document any config that could NOT be matched. **← this research resolves it (see §Config-Matching).**
- **BENCH-03 deliverable format & location** — extension of `bench/RESULTS.md`, a new `bench/BENCH-03-SIGNOFF.md`, and/or a matrix table; how aggregated per-phase JSONs are stitched. Planner chooses; must include per-number source provenance (D-03) + the pre-Phase-10 host-light baseline reference.
- **Which per-phase families to include as matrix rows** beyond depth-6 RMSE/Logloss — bounded by which per-phase BENCH-02 numbers actually exist to aggregate.

### Deferred Ideas (OUT OF SCOPE)
- Real named-dataset benchmark (Higgs / Epsilon) head-to-head vs CatBoost GPU — declined in favor of synthetic-only (D-02).
- A hard "within X× of official CatBoost GPU" parity gate — declined (D-01: CatBoost-GPU is informational).
- Formally closing GPUT-14 + backfilling `bench/RESULTS.md`'s TBD depth-1/depth-6 oracle table — explicitly OUT of scope (D-04); standing bookkeeping debt for milestone-close.
- One fresh comprehensive re-run of the entire matrix in a single session — declined (D-03: aggregate + add-CatBoost-GPU-only).
- Estimated-feature stored-border-VALUE grid parity — unrelated (FEAT-07, Phase 9). Out of scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BENCH-03 | The device-resident training path demonstrably closes the >20× gap on Kaggle CUDA: a comprehensive final speed-parity sign-off vs official CatBoost GPU across the workload matrix, documented and signed off against the pre-Phase-10 host-light baseline, **aggregating the per-phase speed checks (BENCH-02)**. | §Aggregation Mechanics (exact JSON schemas + which files exist), §Config-Matching (CatBoost-GPU arm), §Notebook Harness Delta (one new timing arm), §BENCH-03 Deliverable Format (provenance conventions). Hard gate = D-01 ≥20× vs host-CPU; CatBoost-GPU informational. |
</phase_requirements>

## Summary

Phase 14 is a **benchmark-scripting + sign-off-document phase**, not a source-code phase. The `Runtime` seam, device kernels, and CPU path are all frozen (D-04 no-regression). The only new artifacts are under `bench/` plus a BENCH-03 sign-off document. Work decomposes into three mechanical parts: (1) **aggregate** the already-committed per-phase BENCH-02 numbers, (2) **add one** official-CatBoost-GPU timing arm via a variant of the existing Kaggle notebook harness, and (3) **write** the milestone-closing sign-off with per-number provenance.

Two findings materially shape the plan. **First, only Phase 12 and Phase 13 have committed BENCH-02 `result.json` files.** Phase 10 and Phase 11 runs are still recorded as `TBD`/pending in `bench/RESULTS.md` — there are no committed numbers to aggregate from them. The aggregate matrix therefore has exactly the Phase-12 and Phase-13 rows: grow-policy families `depthwise` and `region`, at `n ∈ {10 000, 100 000, 300 000}`, depth-6 / 20-iter / RMSE. **Second, those committed numbers do NOT come from `bench/generator.py`** — they come from the internal `gen()` in `crates/cb-train/tests/bench_grow_speed_test.rs` (integer-binned columns, 20 features, 32 bins, a ±1 sign target). To make the new CatBoost-GPU arm *comparable to the numbers being aggregated*, it must reproduce **that** workload, not `generator.py`'s regression design. CONTEXT D-02 names `generator.py`; the planner must reconcile this (recommendation below: match the workload that actually produced the aggregated numbers).

The config-matching unknown resolves cleanly against CatBoost's official docs with two documented divergences: **CatBoost has no "Region" grow_policy** (it is a catboost-rs-specific policy; official GPU supports only `SymmetricTree`/`Depthwise`/`Lossguide`), and **CatBoost's GPU `border_count` default is 128** (vs 254 on CPU) whereas the bench uses 32 — so `border_count=32` must be set explicitly. Both are "cannot force / must document" items, which is exactly what D-01's informational framing anticipates.

**Primary recommendation:** Write a new `bench/phase14_cuda_signoff/` driver (variant of `phase13_cuda_oracle/oracle.py`) that (a) runs the depth-6 correctness oracle as a BLOCKING pre-flight, then (b) times official `catboost` `task_type='GPU'` on the **same `bench_grow_speed_test.rs` `gen()` workload** (integer-binned, 20 feat, 32 bins, ±1 target, n=10k/100k/300k), matching depth=6 / iters=20 / lr=0.3 / l2=0.0 / RMSE / `bootstrap_type='No'` / `border_count=32` / `grow_policy='Depthwise'`. Emit `bench03-result.json`. Then author `bench/BENCH-03-SIGNOFF.md` aggregating Phase-12 + Phase-13 device/CPU columns with the new CatBoost-GPU column, each cell labeled with its source run (P100, date). Region's CatBoost-GPU cell is `N/A (no Region policy in official CatBoost)`.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Aggregate committed per-phase JSON | Bench tooling (Python/host, offline) | — | Reads `bench/phase12_*/`, `phase13_*/`; no GPU needed; pure stitching |
| New CatBoost-GPU timing arm | Kaggle CUDA notebook (human-gated) | `catboost` Python pkg on Kaggle image | Requires real NVIDIA GPU; official lib is Python-only; matches `benchmark.py` shape |
| Correctness pre-flight gate | Kaggle CUDA notebook (Rust `--features cuda`) | cb-backend device self-oracles | Reuses Phase-13 Part A structure; blocks before any speed number (T-10-25) |
| BENCH-03 sign-off document | Docs (`bench/`) | — | Human-authored markdown; provenance labels; no code |
| Device / host-CPU speed columns | (frozen) already-committed Phase 12/13 JSON | — | D-03: NOT re-run; aggregated from prior P100 sessions |

**No production `cb-backend`/`cb-train`/`cb-compute` source is touched.** Optional dev-only additions (per roadmap): `criterion 0.7.x` dev-dep and a `profiling` Cargo feature — both opportunistic, not required by BENCH-03.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `catboost` (Python) | latest on Kaggle image | The official GPU baseline to time head-to-head | The library whose >20× gap this milestone closes; already the shape of repo-root `benchmark.py` [VERIFIED: benchmark.py imports `from catboost import CatBoostRegressor`] |
| `numpy` | latest | Build the synthetic `X`/`y` for the CatBoost-GPU arm | Already the sole dep of `generator.py` [VERIFIED: bench/generator.py] |
| Rust stable + `--features cuda` | current | Build the correctness-gate + BENCH-02 anchor (reused) | Existing harness; `--no-default-features --features cuda` [VERIFIED: phase13 oracle.py] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `kaggle` CLI | current | Push notebook + dataset, poll status, pull output | Human-gated run driver; proven in Phase 12/13 [VERIFIED: MEMORY.md phase12/13 outcomes — `kaggle` CLI auth via `~/.kaggle/access_token`] |
| `criterion` | 0.7.x (dev-dep) | OPTIONAL in-env ROCm relative-timing regression during dev | Only if the planner wants a local smoke check; not a gate [CITED: 14-CONTEXT.md scope anchor] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Aggregate committed JSON (D-03) | Fresh single-session full re-run | Declined (D-03): saves GPU time; accepts mixed-session provenance mitigated by per-number labels |
| `bench_grow_speed_test.rs` `gen()` workload | `bench/generator.py` SPEED_CONFIG (1e6×50 regression) | The committed numbers use `gen()`, not `generator.py`; using `generator.py` for the CatBoost arm would make the head-to-head incomparable to the aggregated rows (see Open Questions) |
| `grow_policy='Depthwise'` proxy for Region | Omit CatBoost cell for Region | Region has no official equivalent; a Depthwise/SymmetricTree proxy must be clearly labeled if used at all |

**Installation:** No repo dependency added. `catboost` + `numpy` are consumed **on the Kaggle CUDA image only** (pre-installed on Kaggle GPU kernels; verify at run-time with `python -c "import catboost, catboost.__version__"` and `pip install -q catboost` as a fallback line — enable_internet is already `true` in the kernel metadata) [VERIFIED: phase13 kernel-metadata.json `enable_internet: true`].

## Package Legitimacy Audit

> This phase adds **no** package to `Cargo.toml` or any `pyproject.toml`. `catboost` is consumed at runtime on the Kaggle image (the official GPU baseline being measured), not vendored into the repo. Audit recorded for completeness.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| `catboost` (Python) | PyPI | 8+ yrs | millions/mo | github.com/catboost/catboost | OK | Approved — official Yandex library; the benchmark subject itself [VERIFIED: Context7 `/catboost/catboost`, Source Reputation High] |
| `criterion` (Rust, optional dev-dep) | crates.io | 8+ yrs | very high | github.com/bheisler/criterion.rs | OK | Approved — only if planner adds the optional in-env timing regression |

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** none.

## Config-Matching: catboost-rs bench → official `catboost` GPU

The anchor config that produced the committed BENCH-02 numbers is defined in `crates/cb-train/tests/bench_grow_speed_test.rs` (`params()` + `run()`) [VERIFIED: bench_grow_speed_test.rs lines 67–137].

| catboost-rs bench field | Value | Official `catboost` param | Matchable? | Notes |
|--------------------------|-------|---------------------------|------------|-------|
| `loss: Loss::Rmse` | RMSE | `loss_function='RMSE'` | ✅ | Regression on a ±1 target |
| `depth` | 6 | `depth=6` (a.k.a. `max_depth`) | ✅ | |
| `iterations` | 20 | `iterations=20` | ✅ | |
| `learning_rate` | 0.3 | `learning_rate=0.3` | ✅ | |
| `l2_leaf_reg` | 0.0 | `l2_leaf_reg=0.0` | ⚠️ | CatBoost default is 3.0; MUST set 0.0 explicitly [CITED: docs default] |
| `random_strength` | 0.0 | `random_strength=0.0` | ✅ | CatBoost default 1.0; set 0.0 |
| `boost_from_average` | false | `boost_from_average=False` | ✅ | CatBoost RMSE default True; set False (negligible timing effect) |
| `bootstrap_type: No` | No | `bootstrap_type='No'` | ✅ | Disables subsampling; CatBoost GPU default is `Bayesian` — MUST set `'No'` |
| `subsample` | 1.0 | (implied by `bootstrap_type='No'`) | ✅ | |
| `score_function: L2` | L2 | `score_function='L2'` | ✅ | CatBoost GPU default is `Cosine`; set `'L2'` to match |
| `nbins` | 32 | `border_count=32` | ⚠️ | **CatBoost GPU default is 128 (CPU 254)** — MUST set 32 [VERIFIED: Context7 /catboost/catboost — "border_count 254 for CPU, 128 for GPU"] |
| `random_seed` | 42 | `random_seed=42` | ✅ | |
| `grow_policy: Depthwise` | Depthwise | `grow_policy='Depthwise'` | ✅ | Official GPU supports it [VERIFIED: Context7 — SymmetricTree/Depthwise/Lossguide] |
| `grow_policy: Region` | Region | — | ❌ **CANNOT MATCH** | **Official CatBoost has NO "Region" grow_policy.** It is catboost-rs-specific (see MEMORY: "Region OUT v1.0 gap"). Record Region's CatBoost-GPU cell as `N/A`. |
| `min_data_in_leaf` | `min_data_in_leaf_default()` | `min_data_in_leaf=<same>` | ✅ | Read the default from `cb-train`; Depthwise/Lossguide accept it |
| (GPU selection) | — | `task_type='GPU', devices='0'` | ✅ | [VERIFIED: Context7 GPU training example] |

**Divergences to document in the sign-off (per D-01, informational):**
1. **Region grow_policy has no official analogue** — the Region row's CatBoost-GPU column is `N/A`. Do not fabricate a proxy number silently. If the planner wants a Region-adjacent informational point, run CatBoost with `Depthwise` (or `SymmetricTree`) and clearly label it as "not the same policy."
2. **`border_count` default mismatch (128 GPU vs bench's 32)** — set `border_count=32` explicitly, else CatBoost quantizes finer and the histogram cost differs.
3. **Quantization-cost asymmetry** — catboost-rs's committed device numbers time only the *grow loop* (quantization is host-side, uploaded once, EXCLUDED — GPUT-02). CatBoost's `fit()` wall-clock **includes** its own on-device border computation + quantization. This makes the absolute head-to-head inherently informational (aligns with D-01). Document it; do not attempt to subtract it.
4. **Feature representation** — the bench feeds pre-binned integer columns `0..31` as `f32`. Feed the SAME array to CatBoost as float features with `border_count=32`; with 32 distinct integer values this recovers near-identical bins. Exact histogram parity is NOT required (D-02: timing is data-shape-driven).

**Reference GPU-arm config (planner code target):**
```python
# Source: CatBoost official docs (Context7 /catboost/catboost) + bench_grow_speed_test.rs params()
from catboost import CatBoostRegressor
m = CatBoostRegressor(
    task_type='GPU', devices='0',
    loss_function='RMSE', score_function='L2',
    depth=6, iterations=20, learning_rate=0.3,
    l2_leaf_reg=0.0, random_strength=0.0, boost_from_average=False,
    bootstrap_type='No', border_count=32, random_seed=42,
    grow_policy='Depthwise',           # Region has NO official equivalent -> N/A row
    verbose=False,
)
# warm one untimed fit (JIT/context init), then time train-only:
m.fit(X_warm, y_warm)                  # excluded from the clock
t0 = time.time(); m.fit(X, y); cat_gpu_s = time.time() - t0
```

## Aggregation Mechanics

### Which committed BENCH-02 files exist (the ONLY aggregation sources)
| File | Phase | Schema | Rows to aggregate |
|------|-------|--------|-------------------|
| `bench/phase12_cuda_oracle/bench02-result.json` | 12 | top-level `runs[]` | depthwise/region × n∈{10k,100k,300k} = 6 rows [VERIFIED: read] |
| `bench/phase13_cuda_oracle/result.json` | 13 | `bench02.runs[]` (nested) | depthwise/region × n∈{10k,100k,300k} = 6 rows [VERIFIED: read] |
| `bench/phase12_cuda_oracle/correctness-result.json` | 12 | `families{}` | correctness provenance (not speed) [VERIFIED: read] |
| Phase 10 / Phase 11 | 10/11 | — | **NONE committed** — `bench/RESULTS.md` shows them `TBD`/pending [VERIFIED: RESULTS.md lines 77–153] |

⚠️ **Schema differs between Phase 12 and Phase 13.** Phase 12's speed rows are at the JSON **root** (`.runs[]`); Phase 13's are **nested** under `.bench02.runs[]`. The aggregator must branch on this.

### Exact per-row schema (both files, once you reach `runs[]`)
```json
{ "family": "depthwise", "n": 100000, "device_s": 0.7463, "cpu_s": 29.8424,
  "speedup": "39.987", "dev_trees": 20, "cpu_trees": 20 }
```
- `speedup` is a **string** in the JSON (e.g. `"39.987"`), not a float — cast before comparison.
- `device_s` / `cpu_s` are floats (seconds); `speedup == cpu_s / device_s`.
- Provenance lives at file top level: Phase 12 `.gpu = "Tesla P100-PCIE-16GB"`, `.nvcc = "release 12.8"`; Phase 13 `.gpu = "Tesla P100-PCIE-16GB, 580.159.04, 16384 MiB"`, `.nvcc`, `.correctness_verdict = "ALL-PASS"`, `.bench_verdict = "OK"` [VERIFIED: read both].

### Committed numbers available to stitch (the actual matrix)
| Source | family | n | device_s | cpu_s | speedup | ≥20×? |
|--------|--------|---|----------|-------|---------|-------|
| P12 | depthwise | 10k / 100k / 300k | 0.083 / 0.746 / 2.568 | 2.511 / 29.842 / 101.930 | 30.3× / 40.0× / 39.7× | ✅ |
| P12 | region | 10k / 100k / 300k | 0.101 / 0.875 / 2.872 | 3.178 / 36.817 / 113.303 | 31.3× / 42.1× / 39.5× | ✅ |
| P13 | depthwise | 10k / 100k / 300k | 0.108 / 0.917 / 2.972 | 2.665 / 30.389 / 101.561 | 24.7× / 33.2× / 34.2× | ✅ |
| P13 | region | 10k / 100k / 300k | 0.131 / 0.987 / 3.289 | 3.130 / 36.149 / 111.631 | 23.9× / 36.6× / 33.9× | ✅ |

All 12 committed rows already satisfy the D-01 ≥20× hard gate. The pre-Phase-10 host-light baseline (the ~20× *slowdown* being reversed) is defined in `.planning/notes/gpu-training-host-light-root-cause.md` [CITED: 14-CONTEXT canonical refs].

### The workload the numbers come from (NOT generator.py)
`bench_grow_speed_test.rs::gen(n, nf=20, nbins=32)` builds each feature column as `(hash(i,f) % 32)` cast to `f32`, borders at `k+0.5`, and target `sign(col0 + 0.5·col1 > 0.75·32) → ±1` [VERIFIED: bench_grow_speed_test.rs lines 43–65]. This is the array the CatBoost-GPU arm must replicate in Python for a comparable head-to-head. `bench/generator.py`'s `SPEED_CONFIG` (1e6×50 regression) feeds the notebook's depth-1/depth-6 *correctness* fixtures — a different workload (see Open Questions Q1).

## Notebook Harness Delta (add ONLY a CatBoost-GPU arm)

The existing driver `bench/phase13_cuda_oracle/oracle.py` already: stages repo→`/tmp/repo`, installs Rust, runs Part A per-family correctness (`--no-default-features --features cuda`, blocking gate), runs Part B `bench_grow_speed_test` (CB_BENCH=1, `BENCH_NS=10000,100000,300000`), and writes `result.json`/`result.md` to `/kaggle/working`, capturing `nvidia-smi`/`nvcc` env [VERIFIED: read oracle.py]. **Reuse this structure wholesale.** The minimal delta for Phase 14:

1. **Keep Part A (correctness) as a BLOCKING pre-flight** — reuse the depth-6 grow oracle (or the Phase-13 family self-oracles). If it fails, abort before any speed number (D-04: oracle is pre-flight only; do NOT flip GPUT-14 or backfill RESULTS.md).
2. **Add Part C — CatBoost-GPU timing arm** (new, pure Python, no Rust):
   - `pip install -q catboost` if not importable (internet is enabled).
   - Reproduce the `gen()` workload in numpy for each `n ∈ {10k,100k,300k}` (integer-binned 0..31 f32 columns, 20 features, ±1 target).
   - For `grow_policy='Depthwise'`: warm one untimed `fit`, then time a train-only `fit` with the Reference GPU-arm config above. `region` → record `N/A`.
   - Write `bench03-result.json` with rows `{family, n, catboost_gpu_s, grow_policy_used, note}` + top-level `gpu`/`nvcc`/`catboost_version`/`date`.
3. **New Kaggle kernel-metadata.json** (clone `phase13_cuda_oracle/kernel-metadata.json`): new `id`/`title`, `enable_gpu: true`, `enable_internet: true`, new `dataset_sources` (repo tarball) [VERIFIED: phase13 kernel-metadata.json].
4. **Measurement protocol (unchanged, ROADMAP checklist):** verify CUDA active via `nvidia-smi`; warm one untimed fit; **drain the queue** (CatBoost `fit` is synchronous, so its wall-clock already includes device completion — no extra drain needed, unlike CubeCL); train-only wall-clock; correctness gated first.

**Do-not-fabricate discipline (T-10-25 / RESULTS.md header):** every CatBoost-GPU number comes from THIS run; if the arm fails, its cells stay `TBD`, never invented.

## BENCH-03 Deliverable Format

**Recommendation:** a **new** `bench/BENCH-03-SIGNOFF.md` (keeps the milestone-closing sign-off self-contained; leaves the per-phase run-log `RESULTS.md` untouched, honoring D-04's "no RESULTS.md backfill"). Cross-link it from `RESULTS.md` with one line. Rationale: extending `RESULTS.md` risks touching its TBD tables (D-04 forbids that); a fresh doc is cleaner for provenance.

**Required content:**
- **Verdict banner:** `BENCH-03: PASS` iff every aggregated row is ≥20× vs host-CPU (D-01). State the pre-Phase-10 host-light baseline reference.
- **Aggregate matrix** (columns): `family | n | device_cuda_s | host_cpu_s | speedup_vs_host_cpu | ≥20×? | catboost_gpu_s (informational) | source run (GPU, date, session)`.
- **Per-number provenance labels (D-03):** each device/CPU cell tagged with its origin — Phase-12 P100 run (2026-07-04) or Phase-13 P100 run (2026-07-04); each CatBoost-GPU cell tagged with the Phase-14 session (P100, date). Mixed-session is explicit, never hidden.
- **Divergence notes:** Region CatBoost-GPU = `N/A` (no official Region policy); border_count 128→32; quantization-cost asymmetry (informational).
- **Out-of-scope flag (D-04):** a short "Standing debt — NOT closed here" section pointing at GPUT-14 (still `Pending` in REQUIREMENTS.md) and the `bench/RESULTS.md` depth-1/depth-6 `TBD` oracle table, to be resolved at milestone-close audit or a dedicated follow-up.

**Provenance conventions to reuse (from RESULTS.md header):** "Do NOT fabricate numbers"; "Correctness is a BLOCKING gate"; every `TBD` cell means "run not yet done." Preserve these verbatim in the new doc.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Time the device/CPU grow loop | A new speed harness | Committed Phase 12/13 `result.json` (D-03) | Numbers already exist on real P100; re-running wastes GPU budget and adds session variance |
| Drive the Kaggle run | A bespoke uploader | `kaggle` CLI + cloned `kernel-metadata.json` | Proven pipeline in Phase 12/13 (MEMORY) |
| Stage repo / build cuda wheel | New notebook from scratch | Clone `phase13_cuda_oracle/oracle.py` | Repo→/tmp staging, Rust install, `--features cuda`, env capture all solved |
| Synthetic workload for CatBoost arm | A fresh random generator | Reproduce `bench_grow_speed_test.rs::gen()` in numpy | Must match the workload that produced the aggregated numbers (else head-to-head is incomparable) |
| GPU quantization borders | Compute borders to match CatBoost | Feed the same integer-binned X with `border_count=32` | Exact histogram parity not required (D-02: data-shape-driven timing) |

**Key insight:** Phase 14's value is *aggregation + one comparison arm + a signed document*, not new measurement infrastructure. Every timing/staging primitive already exists and is committed.

## Common Pitfalls

### Pitfall 1: Mismatched aggregation schema (root vs nested `runs[]`)
**What goes wrong:** An aggregator that assumes `.runs[]` silently drops all Phase-13 rows (they're at `.bench02.runs[]`).
**Why it happens:** Phase 12 and Phase 13 emit different top-level shapes [VERIFIED].
**How to avoid:** Branch: `data.get("runs") or data.get("bench02", {}).get("runs", [])`.
**Warning signs:** Aggregate matrix has only 6 rows (one phase) instead of 12.

### Pitfall 2: `speedup` treated as a float
**What goes wrong:** `row["speedup"] >= 20` raises `TypeError` (it's a string `"39.987"`).
**How to avoid:** `float(row["speedup"])`, or recompute `cpu_s/device_s`.

### Pitfall 3: Comparing CatBoost-GPU to a DIFFERENT workload
**What goes wrong:** Using `generator.py` (1e6×50 regression) for the CatBoost arm while the aggregated device/CPU numbers came from `gen()` (n≤300k, 20 feat, ±1) — the head-to-head compares apples to oranges.
**How to avoid:** Reproduce `bench_grow_speed_test.rs::gen()` in the CatBoost arm; keep `BENCH_NS` identical.

### Pitfall 4: Forcing a Region CatBoost number
**What goes wrong:** Silently substituting `Depthwise`/`SymmetricTree` for Region produces a misleading "parity" cell.
**How to avoid:** Record Region's CatBoost-GPU cell as `N/A (no Region policy in official CatBoost)`; only add a proxy if explicitly labeled.

### Pitfall 5: Quoting a speed number when correctness didn't pass
**What goes wrong:** Violates the BLOCKING-gate discipline; a fast-but-wrong number is worthless.
**How to avoid:** Part A (oracle) aborts the notebook on failure before Part C runs.

### Pitfall 6: Accidentally backfilling GPUT-14 / RESULTS.md (D-04 breach)
**What goes wrong:** Plan drifts into flipping GPUT-14 to satisfied or filling the TBD oracle table — widening scope past BENCH-03.
**How to avoid:** Treat correctness as pre-flight ONLY; land the sign-off in a *new* doc; flag GPUT-14/RESULTS.md as standing debt.

## Code Examples

### Aggregate both committed BENCH-02 files (schema-branching)
```python
# Source: schemas verified from bench/phase12_cuda_oracle/bench02-result.json + phase13_cuda_oracle/result.json
import json
def load_rows(path, phase, gpu, date):
    d = json.load(open(path))
    rows = d.get("runs") or d.get("bench02", {}).get("runs", [])   # P12 root vs P13 nested
    for r in rows:
        yield {**r, "speedup": float(r["speedup"]),
               "phase": phase, "gpu": gpu, "date": date, "ge20x": float(r["speedup"]) >= 20.0}
```

### CatBoost-GPU arm reproducing gen() (Part C)
```python
# Source: bench_grow_speed_test.rs gen() (lines 43-65) reproduced in numpy + CatBoost GPU config
import numpy as np, time
from catboost import CatBoostRegressor
def gen(n, nf=20, nbins=32):
    X = np.empty((n, nf), dtype=np.float32)
    for f in range(nf):
        h = (np.arange(n, dtype=np.uint64) * np.uint64(2654435761) + np.uint64(f * 40503))
        X[:, f] = (h % np.uint64(nbins)).astype(np.float32)
    thresh = nbins * 0.75
    y = np.where(X[:, 0] + 0.5 * X[:, 1 % nf] > thresh, 1.0, -1.0).astype(np.float64)
    return X, y
def time_catboost_gpu(n):
    X, y = gen(n)
    m = CatBoostRegressor(task_type='GPU', devices='0', loss_function='RMSE',
        score_function='L2', depth=6, iterations=20, learning_rate=0.3,
        l2_leaf_reg=0.0, random_strength=0.0, boost_from_average=False,
        bootstrap_type='No', border_count=32, random_seed=42,
        grow_policy='Depthwise', verbose=False)
    m.fit(X[:2000], y[:2000])                 # warm (untimed)
    t0 = time.time(); m.fit(X, y); return round(time.time() - t0, 4)
```

## Runtime State Inventory

> This phase is additive benchmark scripting + a new doc; no rename/refactor/migration. No stored data, live-service config, OS-registered state, secrets, or build artifacts carry a string that changes.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — verified: aggregation READS committed JSON, writes new files | none |
| Live service config | Kaggle kernel (new `id`) — created fresh, no existing config mutated | new kernel-metadata.json |
| OS-registered state | None — verified: `kaggle` CLI is a one-shot push/poll | none |
| Secrets/env vars | `~/.kaggle/access_token` (existing, unchanged — MEMORY note) | none |
| Build artifacts | None new — Rust build happens in `/tmp` on Kaggle, discarded | none |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single end-of-milestone benchmark | Standing per-phase BENCH-02 + final BENCH-03 aggregate | v1.1 re-scope 2026-07-02 | Phase 14 aggregates, doesn't first-measure |
| GPU `border_count` 128 default | (unchanged) still 128 GPU / 254 CPU | current | Must set 32 explicitly to match the bench [VERIFIED] |

**Deprecated/outdated:** none relevant.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `catboost` is pre-installed on the Kaggle CUDA image (with `pip install` fallback, internet enabled) | Standard Stack | LOW — fallback line + enable_internet already true; worst case one install step |
| A2 | The CatBoost-GPU arm should match the `bench_grow_speed_test.rs::gen()` workload (not `generator.py`), because that is what produced the aggregated numbers | Aggregation / Config-Matching | MEDIUM — if planner instead follows D-02's literal `generator.py` wording, the head-to-head is incomparable; surfaced as Open Q1 for user/planner resolution |
| A3 | `min_data_in_leaf_default()` / `max_leaves_default()` in `cb-train` are readable and non-timing-critical | Config-Matching | LOW — read the const; Depthwise timing is dominated by n·features·bins |
| A4 | Only Phase 12 + Phase 13 have committed BENCH-02 JSON (Phase 10/11 are TBD) | Aggregation | LOW — verified by `find bench` + RESULTS.md; if a Phase-10/11 run lands before planning, add its rows |
| A5 | Quantization-cost asymmetry (CatBoost fit includes quantization; catboost-rs times grow-loop only) is acceptable because the head-to-head is informational (D-01) | Config-Matching | LOW — D-01 explicitly makes CatBoost-GPU informational |

## Open Questions

1. **Which synthetic workload backs the CatBoost-GPU arm — `gen()` or `generator.py`?**
   - What we know: committed BENCH-02 numbers come from `bench_grow_speed_test.rs::gen()` (n≤300k, 20 feat, 32 bins, ±1). CONTEXT D-02 literally names `bench/generator.py` (1e6×50 regression).
   - What's unclear: whether the user intends the CatBoost head-to-head against the *actual aggregated workload* (`gen()`) or the *named* `generator.py` workload (which has no committed device/CPU number to compare against).
   - Recommendation: match `gen()` (the workload that produced the numbers being aggregated); note the D-02 wording discrepancy in the plan and let discuss-phase/user confirm. A `generator.py` arm would need its OWN device/CPU run to be comparable — which D-03 explicitly declines.

2. **Region row: `N/A` or a labeled Depthwise/SymmetricTree proxy?**
   - What we know: no official Region policy exists.
   - Recommendation: `N/A` in the CatBoost-GPU column; optionally a clearly-labeled Depthwise proxy as an aside, never as a parity claim.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| NVIDIA CUDA GPU (Kaggle) | New CatBoost-GPU timing arm + correctness pre-flight | ✓ (human-gated Kaggle P100) | P100, CUDA 12.8 | none — human runs the notebook (not in-env) |
| `catboost` Python pkg | CatBoost-GPU arm | ✓ on Kaggle image | latest | `pip install -q catboost` (internet enabled) |
| `kaggle` CLI | Push/poll the run | ✓ in-env | current | auth via `~/.kaggle/access_token` (MEMORY) |
| In-env NVIDIA GPU | — | ✗ (only AMD gfx1100/ROCm) | — | ROCm smoke only; NOT the gate (D-04) |

**Missing dependencies with no fallback:** NVIDIA hardware in-env — by design; the Kaggle run is a human-gated external step (BENCH-01 framing).
**Missing dependencies with fallback:** `catboost` (pip install fallback).

## Validation Architecture

> `workflow.nyquist_validation: true` in config.json — section included.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust `#[test]` (`--features cuda`) for the correctness pre-flight; Python driver (`oracle.py`-style) for orchestration; no unit-test framework for the sign-off doc itself |
| Config file | none — benchmark scripts are standalone (mirrors `bench/generator.py`, `phase13 oracle.py`) |
| Quick run command | `python bench/generator.py --check bench/fixtures` (fixture-drift guard, in-env, no GPU) |
| Full suite command | Human-gated Kaggle CUDA notebook (Part A oracle + Part C CatBoost arm) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BENCH-03 | Aggregator stitches P12+P13 rows, casts `speedup`, flags ≥20× | unit (Python) | `python -m pytest bench/phase14_cuda_signoff/aggregate_test.py -x` | ❌ Wave 0 |
| BENCH-03 | CatBoost-GPU arm produces a number per (family,n) with correct config | manual (Kaggle GPU) | human runs the notebook; verifies `bench03-result.json` | ❌ Wave 0 |
| BENCH-03 | Correctness pre-flight BLOCKS before any speed number | integration (Kaggle GPU, reused) | Part A of the driver (reused from Phase 13) | ✅ (phase13 oracle.py) |
| BENCH-03 | Sign-off doc has provenance labels + verdict + out-of-scope flag | manual (doc review) | reviewer checks `bench/BENCH-03-SIGNOFF.md` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `python bench/generator.py --check bench/fixtures` (fixture integrity, seconds, no GPU).
- **Per wave merge:** the aggregation unit test (`aggregate_test.py`) against the committed P12/P13 JSON.
- **Phase gate:** human-gated Kaggle CUDA run green (Part A oracle PASS) before the CatBoost-GPU numbers are trusted; sign-off doc reviewed.

### Wave 0 Gaps
- [ ] `bench/phase14_cuda_signoff/aggregate.py` + `aggregate_test.py` — schema-branching aggregator over the two committed JSON files (offline, no GPU).
- [ ] `bench/phase14_cuda_signoff/oracle.py` (or `signoff.py`) — clone of `phase13 oracle.py` + Part C CatBoost-GPU arm.
- [ ] `bench/phase14_cuda_signoff/kernel-metadata.json` — new Kaggle kernel id.
- [ ] `bench/BENCH-03-SIGNOFF.md` — the deliverable (authored after the run fills the CatBoost-GPU cells).

## Security Domain

> `security_enforcement: true`, `security_asvs_level: 1` in config. This is an offline benchmark-scripting + documentation phase with no auth, no persistent service, no untrusted input, and no production-code change. Most ASVS categories are N/A.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Kaggle CLI uses existing `~/.kaggle/access_token`; no new auth surface |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | minimal | Aggregator parses trusted, repo-committed JSON only; guard `KeyError`/`float()` casts |
| V6 Cryptography | no | — |
| V14 Config / Secrets | yes | Do NOT print or commit the Kaggle `access_token`; the notebook already writes only compact result JSON to `/kaggle/working` (no secret leakage) [VERIFIED: oracle.py writes only result.json/md] |

### Known Threat Patterns for {benchmark scripting}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Secret leak in Kaggle logs/output | Information Disclosure | Keep `access_token` out of the notebook body; write only result files (existing practice) |
| Malformed committed JSON crashes aggregator | Denial of Service (local) | Defensive `.get()` + numeric casts; unit test over the real committed files |
| Fabricated / mislabeled speed number | Tampering (integrity of the sign-off) | Do-not-fabricate discipline; per-number provenance labels (D-03); correctness-gated-before-speed |

## Sources

### Primary (HIGH confidence)
- `/catboost/catboost` (Context7, Source Reputation High) — GPU training (`task_type='GPU'`, `devices`), `grow_policy` = SymmetricTree/Depthwise/Lossguide (no Region), `border_count` 254 CPU / 128 GPU, RMSE/l2_leaf_reg examples.
- Repo artifacts (read this session): `bench/phase12_cuda_oracle/bench02-result.json`, `bench/phase13_cuda_oracle/result.json`, `bench/phase12_cuda_oracle/correctness-result.json`, `bench/phase13_cuda_oracle/oracle.py`, `bench/phase13_cuda_oracle/kernel-metadata.json`, `bench/generator.py`, `bench/RESULTS.md`, `crates/cb-train/tests/bench_grow_speed_test.rs`, `benchmark.py`, `.planning/REQUIREMENTS.md`, `.planning/phases/14-.../14-CONTEXT.md`, `.planning/config.json`.

### Secondary (MEDIUM confidence)
- MEMORY.md (phase12/phase13 execution outcomes) — `kaggle` CLI auth via `~/.kaggle/access_token`; 30–42× / 23.9–36.6× device≫CPU; Region-OUT v1.0 gap.

### Tertiary (LOW confidence)
- Training-data recollection of CatBoost GPU defaults (bootstrap_type='Bayesian', score_function='Cosine', l2_leaf_reg=3.0) — cross-checked against Context7 where possible; treated as `[ASSUMED]` in the config table where docs didn't explicitly confirm the default.

## Metadata

**Confidence breakdown:**
- Config-matching: HIGH — official docs confirm grow_policy set (no Region) and border_count 128/254; bench config read directly from source.
- Aggregation mechanics: HIGH — both JSON schemas read this session; Phase 10/11 absence verified.
- Notebook delta: HIGH — existing driver read in full; delta is additive.
- Deliverable format / provenance: MEDIUM — planner's discretion (D-03); conventions grounded in RESULTS.md header.

**Research date:** 2026-07-05
**Valid until:** 2026-08-05 (stable; CatBoost GPU param surface changes slowly, committed JSON is frozen)
