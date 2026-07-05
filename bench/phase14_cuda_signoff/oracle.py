#!/usr/bin/env python3
# catboost-rs Phase-14 CUDA sign-off (Kaggle GPU) — BENCH-03.
#
# Structure (a clone of phase13_cuda_oracle/oracle.py, minimal + additive delta):
#
#   Part A — per-family CORRECTNESS gate (BLOCKING pre-flight, D-04). Runs each of
#     the Phase-13 families' device self-oracle (*_test.rs in cb-backend) under
#     --no-default-features --features cuda. Those tests compare the CUDA device
#     path to a Rust CPU reference at eps=1e-4. Per D-04 this is PRE-FLIGHT ONLY:
#     we do NOT flip GPUT-14 and do NOT write bench/RESULTS.md from here. If Part A
#     fails, we emit the result with a SOME-FAIL verdict and sys.exit(2) BEFORE
#     Part C runs — a fast-but-wrong number is never quoted (Pitfall 5).
#
#   Part C — NEW official-CatBoost-GPU timing arm (pure Python, no Rust). Times the
#     official `catboost` package with task_type='GPU' on the SAME workload the
#     aggregated device/CPU numbers came from: an exact numpy reproduction of
#     crates/cb-train/tests/bench_grow_speed_test.rs::gen() (integer-binned 0..31
#     f32 columns, 20 features, 32 bins, +/-1 target, n in {10k,100k,300k}) with the
#     frozen config (RMSE / L2 / depth 6 / iters 20 / lr 0.3 / l2 0.0 /
#     bootstrap No / border_count 32 / seed 42 / grow_policy Depthwise). Emits
#     bench03-result.json + bench03-result.md to /kaggle/working.
#
# Four INFORMATIONAL divergences (per D-01 the CatBoost-GPU column is informational,
# NOT a gate):
#   1. Region has NO official CatBoost grow_policy (official GPU = SymmetricTree/
#      Depthwise/Lossguide) — its cell is N/A, never a fabricated proxy (Pitfall 4).
#   2. CatBoost GPU border_count default is 128 (CPU 254); the bench uses 32, so
#      border_count=32 is set explicitly.
#   3. Quantization-cost asymmetry: CatBoost fit() wall-clock INCLUDES on-device
#      border computation + quantization, while catboost-rs times ONLY the grow
#      loop (quantization host-side, uploaded once, GPUT-02). Do NOT subtract it.
#   4. Feature representation: pre-binned integer columns 0..31 are fed as float32
#      with border_count=32; with 32 distinct values this recovers near-identical
#      bins (exact histogram parity NOT required — D-02: data-shape-driven timing).
#
# Do-not-fabricate discipline: every value comes from THIS run; correctness is
# gated BEFORE any speed number; a failed Part C leaves catboost_gpu_s null, never
# invented. Build lives in /tmp so the Kaggle output stays tiny; compact
# bench03-result.json + bench03-result.md written to /kaggle/working.
#
# All run-time work lives in main() under `if __name__ == "__main__"`, so importing
# this module in-env (no GPU, no numpy) exposes `gen`/`main` and triggers no run.

WORK = "/kaggle/working"

# BENCH-03 workload matrix + frozen grow config (module constants, no side effects).
BENCH_NS_DEFAULT = "10000,100000,300000"
NF = 20
NBINS = 32
DEPTH = 6
ITERS = 20
LEARNING_RATE = 0.3
L2_LEAF_REG = 0.0
RANDOM_STRENGTH = 0.0
RANDOM_SEED = 42


def gen(n, nf=NF, nbins=NBINS):
    """Exact numpy reproduction of bench_grow_speed_test.rs::gen().

    Per feature f, column = ((i * 2654435761 + f * 40503) mod 2^64) mod nbins,
    cast to float32; target = +1 if X[:,0] + 0.5*X[:,1%nf] > nbins*0.75 else -1.

    Rust uses 64-bit wrapping arithmetic (usize); numpy uint64 wraps mod 2^64
    with the same C semantics. Returns (X: (n, nf) float32, y: (n,) float64 +/-1).
    Lazy numpy import so importing oracle.py in-env needs no numpy.
    """
    import numpy as np

    idx = np.arange(n, dtype=np.uint64)
    mul = np.uint64(2654435761)
    nb = np.uint64(nbins)
    cols = []
    for f in range(nf):
        add = np.uint64((f * 40503) & 0xFFFFFFFFFFFFFFFF)
        h = idx * mul + add          # uint64 wraps mod 2^64 (matches wrapping_mul/add)
        col = (h % nb).astype(np.float32)
        cols.append(col)
    X = np.stack(cols, axis=1)       # (n, nf) float32
    a = X[:, 0].astype(np.float64)
    b = X[:, 1 % nf].astype(np.float64)
    thresh = float(nbins) * 0.75
    y = np.where(a + 0.5 * b > thresh, 1.0, -1.0).astype(np.float64)
    return X, y


def main():
    import os, subprocess, sys, shutil, time, json, re

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

    def env_line(cmd):
        try:
            return subprocess.run(cmd, text=True, capture_output=True, shell=True).stdout.strip()
        except Exception as e:
            return f"(err {e})"

    result = {"phase": 14, "kind": "cuda-signoff-bench03", "families": {}, "bench03": {}}

    log("=================== ENVIRONMENT ===================")
    sh("nvidia-smi", timeout=120)
    result["gpu"] = env_line("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
    result["nvcc"] = env_line("nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1 || echo NO_NVCC")
    result["cuda_dirs"] = env_line("ls -d /usr/local/cuda* 2>/dev/null | tr '\\n' ' '")
    log("gpu:", result["gpu"], "| nvcc:", result["nvcc"], "| cuda:", result["cuda_dirs"])

    # ---- stage repo into /tmp (NOT /kaggle/working, to keep output small) ----
    inp = "/kaggle/input"
    tarball = srcdir = None
    for dp, _, fs in os.walk(inp):
        if "repo.tar.gz" in fs:
            tarball = os.path.join(dp, "repo.tar.gz")
        if "Cargo.toml" in fs and os.path.exists(os.path.join(dp, "crates")):
            srcdir = dp
    REPO = "/tmp/repo"
    if os.path.exists(REPO):
        shutil.rmtree(REPO)
    os.makedirs(REPO)
    if tarball:
        sh(["tar", "-xzf", tarball, "-C", REPO])
    elif srcdir:
        sh(["bash", "-lc", f"cp -a '{srcdir}/.' '{REPO}/'"])
    else:
        result["fatal"] = "payload not found"
        json.dump(result, open(os.path.join(WORK, "bench03-result.json"), "w"), indent=2)
        sys.exit(2)
    assert os.path.exists(os.path.join(REPO, "Cargo.toml")), "no Cargo.toml after stage"
    assert os.path.exists(os.path.join(REPO, "crates/cb-train/tests/bench_grow_speed_test.rs")), \
        "bench harness missing — dataset not refreshed?"
    log("REPO staged ->", sorted(os.listdir(REPO))[:20])

    # ---- rust toolchain ----
    sh("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal")
    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env["PATH"]
    env["CARGO_TARGET_DIR"] = "/tmp/target"      # keep target OUT of /kaggle/working
    env["CARGO_BUILD_JOBS"] = "2"
    env["CARGO_NET_RETRY"] = "5"
    env["RUST_BACKTRACE"] = "1"
    sh("rustc --version && cargo --version", env=env)

    # =====================================================================
    # PART A — per-family CUDA CORRECTNESS gate (BLOCKING pre-flight, D-04)
    # (label, req, crate, test-name filters -> cb-backend device self-oracles)
    # =====================================================================
    FAMILIES = [
        ("pairwise (deriv + batched Cholesky) GPUT-11/21", "GPUT-11,GPUT-21", "cb-backend",
            ["pairwise_deriv", "cholesky_solve"]),
        ("ranking (query grouping + det + stochastic) GPUT-22", "GPUT-22", "cb-backend",
            ["query_helper", "ranking_det", "ranking_stoch"]),
        ("multiclass (softmax der + multi-Newton) GPUT-12", "GPUT-12", "cb-backend",
            ["multiclass", "multi_newton"]),
        ("ordered (resident approx trajectory) GPUT-13", "GPUT-13", "cb-backend",
            ["ordered"]),
        ("langevin (seeded Gaussian / SGLB) GPUT-20", "GPUT-20", "cb-backend",
            ["langevin"]),
    ]

    DIV_RE = re.compile(r"(abs|rel|div|diverg|max|eps|epsilon|1e-|e-1[0-9]|tol)", re.I)
    SUMMARY_RE = re.compile(r"test result:.*")

    for label, req, crate, filters in FAMILIES:
        cmd = ["cargo", "test", "--release", "--no-default-features", "--features", "cuda",
               "-p", crate, "--"] + filters + ["--nocapture", "--test-threads=1"]
        t0 = time.time()
        rc, out = sh(cmd, cwd=REPO, env=env, timeout=5400)
        dt = round(time.time() - t0, 1)
        summ = SUMMARY_RE.findall(out)
        divs = [l.strip() for l in out.splitlines() if DIV_RE.search(l) and ("test " not in l[:5])][:40]
        ran_any = any("running" in l and "test" in l for l in out.splitlines()) or bool(summ)
        result["families"][label] = {
            "req": req, "crate": crate, "filters": filters, "exit": rc, "secs": dt,
            "ran_any_tests": ran_any,
            "summary": summ[-3:] if summ else [],
            "divergences": divs,
            "tail": out[-3000:],
        }
        log(f"[{label}] exit={rc} {summ[-1:] if summ else ''}")

    # ---- BLOCKING roll-up: correctness gates the speed arm (Pitfall 5, D-04) ----
    corr_pass = all(f["exit"] == 0 and f["ran_any_tests"] for f in result["families"].values())
    result["correctness_verdict"] = "ALL-PASS" if corr_pass else "SOME-FAIL"
    if not corr_pass:
        # Do NOT run Part C on a failed pre-flight — never quote a fast-but-wrong number.
        result["catboost_gpu_verdict"] = "SKIPPED (correctness pre-flight failed)"
        json.dump(result, open(os.path.join(WORK, "bench03-result.json"), "w"), indent=2)
        with open(os.path.join(WORK, "bench03-result.md"), "w") as fh:
            fh.write(f"# Phase-14 CUDA sign-off (BENCH-03)\n\nGPU: {result['gpu']}\n")
            fh.write(f"correctness_verdict: {result['correctness_verdict']} — Part C SKIPPED\n")
        log("CORRECTNESS_VERDICT:", result["correctness_verdict"], "-> aborting before Part C")
        sys.exit(2)

    # =====================================================================
    # PART C — NEW official-CatBoost-GPU timing arm (pure Python)
    # =====================================================================
    try:
        import catboost  # noqa: F401
    except Exception:
        sh("pip install -q catboost", env=env, timeout=1200)
    try:
        from catboost import CatBoostRegressor
        cb_version = getattr(__import__("catboost"), "__version__", "unknown")
    except Exception as e:
        result["bench03"] = {"error": f"catboost import failed: {e}", "runs": []}
        result["catboost_gpu_verdict"] = "FAIL (catboost unavailable)"
        json.dump(result, open(os.path.join(WORK, "bench03-result.json"), "w"), indent=2)
        log("Part C: catboost import FAILED — no number fabricated")
        sys.exit(3)

    bench_ns = [int(x) for x in os.environ.get("BENCH_NS", BENCH_NS_DEFAULT).split(",") if x.strip()]
    # (family label, official grow_policy or None for Region — no official policy)
    c_families = [("depthwise", "Depthwise"), ("region", None)]
    rows = []
    for label, policy in c_families:
        for n in bench_ns:
            if policy is None:
                # Region has NO official CatBoost policy (Pitfall 4) — N/A, never fabricated.
                rows.append({
                    "family": label, "n": n,
                    "catboost_gpu_s": None,
                    "grow_policy_used": "N/A",
                    "note": "no Region policy in official CatBoost",
                })
                log(f"[catboost-gpu {label} n={n}] N/A (no official Region policy)")
                continue
            X, y = gen(n)
            m = CatBoostRegressor(
                task_type="GPU", devices="0",
                loss_function="RMSE", score_function="L2",
                depth=DEPTH, iterations=ITERS, learning_rate=LEARNING_RATE,
                l2_leaf_reg=L2_LEAF_REG, random_strength=RANDOM_STRENGTH,
                boost_from_average=False,
                bootstrap_type="No", border_count=NBINS, random_seed=RANDOM_SEED,
                grow_policy="Depthwise",
                verbose=False,
            )
            # warm one untimed fit (JIT / device-context init) — excluded from the clock.
            Xw, yw = X[:2000], y[:2000]
            try:
                m.fit(Xw, yw)
            except Exception as e:
                rows.append({"family": label, "n": n, "catboost_gpu_s": None,
                             "grow_policy_used": "Depthwise",
                             "note": f"warm fit failed: {e}"})
                log(f"[catboost-gpu {label} n={n}] warm fit FAILED: {e}")
                continue
            t0 = time.time()
            try:
                m.fit(X, y)                       # timed train-only fit
                cat_gpu_s = round(time.time() - t0, 4)
                rows.append({"family": label, "n": n, "catboost_gpu_s": cat_gpu_s,
                             "grow_policy_used": "Depthwise", "note": ""})
                log(f"[catboost-gpu {label} n={n}] catboost_gpu_s={cat_gpu_s}")
            except Exception as e:
                rows.append({"family": label, "n": n, "catboost_gpu_s": None,
                             "grow_policy_used": "Depthwise",
                             "note": f"timed fit failed: {e}"})
                log(f"[catboost-gpu {label} n={n}] timed fit FAILED: {e}")

    result["bench03"] = {
        "runs": rows,
        "catboost_version": cb_version,
        "config": {
            "task_type": "GPU", "loss_function": "RMSE", "score_function": "L2",
            "depth": DEPTH, "iterations": ITERS, "learning_rate": LEARNING_RATE,
            "l2_leaf_reg": L2_LEAF_REG, "random_strength": RANDOM_STRENGTH,
            "boost_from_average": False, "bootstrap_type": "No",
            "border_count": NBINS, "random_seed": RANDOM_SEED,
            "grow_policy": "Depthwise",
        },
        "divergences": [
            "Region has no official CatBoost grow_policy -> N/A (not a proxy)",
            "GPU border_count default 128; set to 32 to match the bench",
            "CatBoost fit() wall-clock includes on-device quantization; catboost-rs "
            "times only the grow loop — NOT subtracted (informational, D-01)",
            "integer-binned 0..31 columns fed as float32 with border_count=32",
        ],
    }
    # verdict: OK iff every non-N/A depthwise cell produced a real number.
    timed = [r for r in rows if r["grow_policy_used"] != "N/A"]
    cat_ok = bool(timed) and all(r["catboost_gpu_s"] is not None for r in timed)
    result["catboost_gpu_verdict"] = "OK" if cat_ok else "PARTIAL"

    # ---- emit compact result.json + human-readable md ----
    result["date"] = time.strftime("%Y-%m-%d", time.gmtime())
    json.dump(result, open(os.path.join(WORK, "bench03-result.json"), "w"), indent=2)
    with open(os.path.join(WORK, "bench03-result.md"), "w") as fh:
        fh.write(f"# Phase-14 CUDA sign-off (BENCH-03)\n\n")
        fh.write(f"GPU: {result['gpu']}\nnvcc: {result['nvcc']}\n")
        fh.write(f"catboost: {cb_version}\n")
        fh.write(f"correctness_verdict: {result['correctness_verdict']}\n")
        fh.write(f"catboost_gpu_verdict: {result['catboost_gpu_verdict']}\n\n")
        fh.write("## Official CatBoost-GPU timing (informational — D-01)\n\n")
        fh.write("| family | n | catboost_gpu_s | grow_policy_used | note |\n")
        fh.write("|---|---|---|---|---|\n")
        for r in rows:
            s = "N/A" if r["catboost_gpu_s"] is None and r["grow_policy_used"] == "N/A" else \
                ("TBD" if r["catboost_gpu_s"] is None else f"{r['catboost_gpu_s']:.4f}")
            fh.write(f"| {r['family']} | {r['n']} | {s} | {r['grow_policy_used']} | {r['note']} |\n")
    log("CORRECTNESS_VERDICT:", result["correctness_verdict"])
    log("CATBOOST_GPU_VERDICT:", result["catboost_gpu_verdict"], "rows:", len(rows))
    log("=================== PHASE-14 BENCH-03 SIGN-OFF COMPLETE ===================")


if __name__ == "__main__":
    main()
