# Phase 14: Comprehensive Kaggle CUDA Speed Benchmark + Parity Sign-Off - Pattern Map

**Mapped:** 2026-07-05
**Files analyzed:** 5 new artifacts (all under `bench/`)
**Analogs found:** 5 / 5

> **Scope note (carried from orchestrator):** This is a benchmark-scripting + sign-off-document
> phase. No production `cb-backend`/`cb-train`/`cb-compute` source is touched (D-04 no-regression).
> All new files live under `bench/phase14_cuda_signoff/` plus one root `bench/BENCH-03-SIGNOFF.md`.
> Nothing under `catboost-master/` is a valid analog (vendored upstream). Every analog below is a
> first-party `bench/` asset, already committed and proven on real P100.

## File Classification

| New File | Role | Data Flow | Closest Analog | Match Quality |
|----------|------|-----------|----------------|---------------|
| `bench/phase14_cuda_signoff/aggregate.py` | utility (offline stitcher) | transform / batch | `bench/generator.py` (Python `argparse` + `json` CLI shape) + research code example | role-match (new capability; no exact analog) |
| `bench/phase14_cuda_signoff/aggregate_test.py` | test (Python unit) | transform | `bench/generator.py` `--check` self-verify pattern | role-match |
| `bench/phase14_cuda_signoff/oracle.py` (a.k.a. `signoff.py`) | driver/script (Kaggle orchestrator) | event-driven / batch (subprocess + timing) | `bench/phase13_cuda_oracle/oracle.py` | **exact** |
| `bench/phase14_cuda_signoff/kernel-metadata.json` | config | request-response (Kaggle push/poll) | `bench/phase13_cuda_oracle/kernel-metadata.json` | **exact** |
| `bench/BENCH-03-SIGNOFF.md` | doc (sign-off) | — | `bench/RESULTS.md` (header conventions) + `bench/phase13_cuda_oracle/result.md` (table shape) | role-match |

**Data sources consumed (read-only, NOT created):**
- `bench/phase12_cuda_oracle/bench02-result.json` — schema: `runs[]` at **root**
- `bench/phase13_cuda_oracle/result.json` — schema: `runs[]` **nested** under `.bench02.runs[]`
- `crates/cb-train/tests/bench_grow_speed_test.rs` — `gen()` + `params()`: the workload/config the CatBoost-GPU arm must replicate

---

## Pattern Assignments

### `bench/phase14_cuda_signoff/oracle.py` (driver/script, event-driven + batch)

**Analog:** `bench/phase13_cuda_oracle/oracle.py` — **reuse wholesale**; delta = add Part C (CatBoost-GPU arm), keep Part A (correctness) as blocking pre-flight. Part B (Rust device-vs-CPU BENCH-02) is optional to re-run since D-03 aggregates the committed numbers.

**Module header + helpers pattern** (lines 17–40): copy verbatim — `log()` flushes, `sh()` runs subprocess with capture + timeout + tail-trimmed logging, `env_line()` for one-shot env capture.
```python
import os, subprocess, sys, shutil, time, json, re

WORK = "/kaggle/working"
os.makedirs(WORK, exist_ok=True)

def log(*a):
    print(*a, flush=True)

def sh(cmd, cwd=None, env=None, timeout=None):
    log("\n$", cmd if isinstance(cmd, str) else " ".join(cmd))
    try:
        r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, env=env,
                           shell=isinstance(cmd, str), timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"
    out = (r.stdout or "") + (("\nSTDERR:\n" + r.stderr) if r.stderr else "")
    log(out[-6000:])
    return r.returncode, out
```

**Env-capture pattern** (lines 42–49): copy — provenance (`gpu`/`nvcc`/`cuda_dirs`) goes into the result dict top-level, matching the schema the aggregator reads from committed JSON.
```python
result = {"phase": 14, "kind": "cuda-signoff-bench03", "families": {}, "bench03": {}}
sh("nvidia-smi", timeout=120)
result["gpu"]  = env_line("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
result["nvcc"] = env_line("nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1 || echo NO_NVCC")
result["cuda_dirs"] = env_line("ls -d /usr/local/cuda* 2>/dev/null | tr '\\n' ' '")
```

**Repo staging + Rust toolchain pattern** (lines 51–84): copy verbatim — stage repo tarball into `/tmp/repo` (NOT `/kaggle/working`), `CARGO_TARGET_DIR=/tmp/target`, assert the harness file exists. Same guardrails apply for Phase 14.

**Part A — correctness BLOCKING pre-flight pattern** (lines 86–122): keep this structure. `FAMILIES` list of `(label, req, crate, test-filters)` → `cargo test --release --no-default-features --features cuda -p cb-backend -- <filters> --nocapture --test-threads=1`, collect `test result:` summaries + divergence lines. Per D-04 this stays a **pre-flight gate only** — do NOT flip GPUT-14 or backfill RESULTS.md.
```python
cmd = ["cargo", "test", "--release", "--no-default-features", "--features", "cuda",
       "-p", crate, "--"] + filters + ["--nocapture", "--test-threads=1"]
rc, out = sh(cmd, cwd=REPO, env=env, timeout=5400)
summ = SUMMARY_RE.findall(out)                       # r"test result:.*"
result["families"][label] = {"req": req, "exit": rc, "ran_any_tests": ran_any, "summary": summ[-3:], ...}
```

**NEW — Part C: CatBoost-GPU timing arm** (no analog line; from RESEARCH §Notebook Harness Delta + Code Examples). Pure Python, no Rust. Reproduce `bench_grow_speed_test.rs::gen()` in numpy, warm one untimed `fit`, then time train-only `fit`. See the frozen config in the "Shared Patterns → CatBoost-GPU config" section below.
```python
# pip install -q catboost fallback (internet enabled in kernel-metadata)
for family, gp in [("depthwise", "Depthwise"), ("region", None)]:
    for n in [10000, 100000, 300000]:
        if gp is None:                                # Region has NO official policy
            rows.append({"family": family, "n": n, "catboost_gpu_s": None,
                         "grow_policy_used": "N/A", "note": "no Region policy in official CatBoost"})
            continue
        X, y = gen(n)
        m.fit(X[:2000], y[:2000])                     # warm (untimed)
        t0 = time.time(); m.fit(X, y); s = round(time.time() - t0, 4)
        rows.append({"family": family, "n": n, "catboost_gpu_s": s, "grow_policy_used": gp, "note": ""})
result["bench03"] = {"runs": rows, "catboost_version": env_line("python -c 'import catboost;print(catboost.__version__)'")}
```

**Result-emit + verdict pattern** (lines 150–172): copy — `json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)` plus a small human-readable `result.md` table. Name Phase 14's output `bench03-result.json` (per RESEARCH). Do-not-fabricate: cells stay `TBD`/`None` if the arm fails.

---

### `bench/phase14_cuda_signoff/kernel-metadata.json` (config)

**Analog:** `bench/phase13_cuda_oracle/kernel-metadata.json` — clone, bump `id`/`title`/`code_file`/`dataset_sources` for Phase 14.

**Full pattern** (lines 1–16): `enable_gpu: true`, `enable_internet: true` (required for `pip install catboost` fallback), `kernel_type: "script"`, `is_private: true`.
```json
{
  "id": "yensen2/catboost-rs-phase14-cuda-signoff",
  "title": "catboost-rs phase14 cuda signoff",
  "code_file": "catboost-rs-phase14-cuda-signoff.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "keywords": ["gpu"],
  "dataset_sources": ["yensen2/catboost-rs-phase14-src"],
  "kernel_sources": [], "competition_sources": [], "model_sources": []
}
```
Note: `enable_internet: true` is load-bearing for Phase 14 (the P13 run also had it) — the CatBoost-GPU arm's `pip install -q catboost` fallback depends on it.

---

### `bench/phase14_cuda_signoff/aggregate.py` (utility, transform/batch)

**Analog:** no exact code analog (new capability). Structural template = `bench/generator.py`'s `argparse` + `json` + `numpy`-only CLI shape (module docstring, `main()`, `--check` subcommand). Logic template = RESEARCH §Code Examples "Aggregate both committed BENCH-02 files (schema-branching)".

**Schema-branching load pattern** (CRITICAL — Pitfall 1 & 2 from RESEARCH):
```python
import json
def load_rows(path, phase, gpu, date):
    d = json.load(open(path))
    rows = d.get("runs") or d.get("bench02", {}).get("runs", [])   # P12 root vs P13 nested
    for r in rows:
        yield {**r, "speedup": float(r["speedup"]),                # speedup is a STRING in JSON
               "phase": phase, "gpu": gpu, "date": date, "ge20x": float(r["speedup"]) >= 20.0}
```
- Phase 12 rows: `bench02-result.json` → `.runs[]` at root, top-level `.gpu`, `.nvcc`.
- Phase 13 rows: `result.json` → `.bench02.runs[]` nested, top-level `.gpu`, `.nvcc`, `.correctness_verdict`, `.bench_verdict`.
- Per-row schema (both, once at `runs[]`): `{"family", "n", "device_s"(float), "cpu_s"(float), "speedup"(str), "dev_trees", "cpu_trees"}`.

**Python CLI/docstring style** (from `bench/generator.py` lines 1–44): triple-quoted module docstring explaining single-source rationale; stdlib `argparse`, `json`, `os`; deterministic, dependency-light. Mirror this house style.

---

### `bench/phase14_cuda_signoff/aggregate_test.py` (test, transform)

**Analog:** `bench/generator.py`'s `--check` self-verification idea (fixture-drift guard, runs offline, seconds, no GPU). RESEARCH §Validation names the run command:
`python -m pytest bench/phase14_cuda_signoff/aggregate_test.py -x`.

**Test target:** load the two REAL committed JSON files, assert 12 rows total (6 P12 + 6 P13 — Pitfall 1: only 6 means the nested-schema branch is broken), assert every `speedup` casts to float and is `>= 20.0` (D-01 hard gate). Per project CLAUDE.md rule, tests live in a **separate file** — never embed in `aggregate.py`.

---

### `bench/BENCH-03-SIGNOFF.md` (doc, sign-off)

**Analog (conventions):** `bench/RESULTS.md` header (lines 1–15) — copy the three standing disciplines **verbatim**:
```
- **Correctness is a BLOCKING gate.** No speed number is valid unless the oracle passes first (T-10-25).
- **ROCm in-env is smoke-only — NOT a gate.** Authoritative oracle of record = human-gated Kaggle CUDA run.
- **Do NOT fabricate numbers.** All `TBD` cells are filled from actual notebook output.
```

**Analog (table shape):** `bench/phase13_cuda_oracle/result.md` (generated by oracle.py lines 158–169) — the `| family | n | device_s | cpu_s | speedup | ... |` markdown table. Extend with the BENCH-03 columns from RESEARCH §Deliverable Format:
`family | n | device_cuda_s | host_cpu_s | speedup_vs_host_cpu | ≥20×? | catboost_gpu_s (informational) | source run (GPU, date, session)`.

**Required content (RESEARCH §Deliverable Format):**
- **Verdict banner:** `BENCH-03: PASS` iff every aggregated row ≥20× vs host-CPU (D-01). Reference the pre-Phase-10 host-light baseline (`.planning/notes/gpu-training-host-light-root-cause.md`).
- **Per-number provenance labels (D-03):** each device/CPU cell tagged Phase-12 P100 (2026-07-04) or Phase-13 P100 (2026-07-04); each CatBoost-GPU cell tagged Phase-14 session. Mixed-session explicit.
- **Divergence notes:** Region CatBoost-GPU = `N/A`; `border_count` 128→32; quantization-cost asymmetry (informational).
- **"Standing debt — NOT closed here" section (D-04):** point at GPUT-14 (`Pending` in REQUIREMENTS.md) + the `bench/RESULTS.md` depth-1/depth-6 `TBD` oracle table.

**The 12 committed rows to stitch** (from RESEARCH, already ≥20×):
| Source | family | n(10k/100k/300k) | speedup |
|--------|--------|------------------|---------|
| P12 | depthwise | — | 30.3× / 40.0× / 39.7× |
| P12 | region | — | 31.3× / 42.1× / 39.5× |
| P13 | depthwise | — | 24.7× / 33.2× / 34.2× |
| P13 | region | — | 23.9× / 36.6× / 33.9× |

---

## Shared Patterns

### CatBoost-GPU config (frozen — apply in `oracle.py` Part C)
**Source:** `crates/cb-train/tests/bench_grow_speed_test.rs` `params()` (lines 67–108) + `gen()` (lines 43–65), mapped to the official `catboost` Python API (RESEARCH §Config-Matching).
```python
def gen(n, nf=20, nbins=32):                          # reproduces bench_grow_speed_test.rs::gen()
    X = np.empty((n, nf), dtype=np.float32)
    for f in range(nf):
        h = (np.arange(n, dtype=np.uint64) * np.uint64(2654435761) + np.uint64(f * 40503))
        X[:, f] = (h % np.uint64(nbins)).astype(np.float32)
    y = np.where(X[:, 0] + 0.5 * X[:, 1 % nf] > nbins * 0.75, 1.0, -1.0)
    return X, y
m = CatBoostRegressor(
    task_type='GPU', devices='0', loss_function='RMSE', score_function='L2',
    depth=6, iterations=20, learning_rate=0.3,
    l2_leaf_reg=0.0, random_strength=0.0, boost_from_average=False,   # non-default: must set
    bootstrap_type='No', border_count=32, random_seed=42,            # border_count default 128 GPU → set 32
    grow_policy='Depthwise', verbose=False)                          # Region → N/A (no official policy)
```
**Apply to:** the CatBoost-GPU arm only. Two documented "cannot match" items (D-01 informational): (1) Region has no official `grow_policy` → cell `N/A`; (2) CatBoost `fit()` wall-clock includes quantization while catboost-rs times grow-loop only — do not subtract.

### Measurement protocol (ROADMAP checklist)
**Apply to:** every timed cell. Verify CUDA active via `nvidia-smi`; warm one untimed `fit`; train-only wall-clock; correctness gated first. Note: CatBoost `fit()` is synchronous so its wall-clock already includes device completion — no CubeCL-style queue-drain needed (unlike the Rust device path).

### Provenance + do-not-fabricate discipline
**Source:** `bench/RESULTS.md` header + oracle.py's "every value comes from THIS run" comment (lines 14–16).
**Apply to:** `oracle.py` (write only real numbers), `aggregate.py` (label each row's source phase/GPU/date), `BENCH-03-SIGNOFF.md` (per-number session tags). A failed arm leaves `TBD`/`None`, never invented.

### Result-JSON schema contract (aggregator ↔ oracle)
**Apply to:** `aggregate.py` reads `runs[]` with `{family, n, device_s, cpu_s, speedup(str), dev_trees, cpu_trees}` + top-level `gpu`/`nvcc`. Phase 14's new `bench03-result.json` adds `bench03.runs[]` with `{family, n, catboost_gpu_s, grow_policy_used, note}` + top-level `gpu`/`nvcc`/`catboost_version`/`date`.

---

## No Analog Found

None. All five new files map to a committed `bench/` analog. The only genuinely-new logic (the CatBoost-GPU arm in `oracle.py` Part C, and the schema-branching aggregator) has concrete templates in RESEARCH.md §Code Examples grounded in the read JSON schemas and `bench_grow_speed_test.rs` source.

## Metadata

**Analog search scope:** `bench/`, `bench/phase12_cuda_oracle/`, `bench/phase13_cuda_oracle/`, `crates/cb-train/tests/`, repo-root `benchmark.py`.
**Files scanned:** 8 (oracle.py, kernel-metadata.json, bench02-result.json, result.json, bench_grow_speed_test.rs, RESULTS.md, generator.py, benchmark.py).
**Excluded by scope:** all `catboost-master/` (vendored upstream — not a first-party analog).
**Pattern extraction date:** 2026-07-05
