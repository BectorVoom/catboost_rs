#!/usr/bin/env python3
# catboost-rs Phase-15 single-session CUDA oracle (Kaggle GPU) — HARD-01 / HARD-02.
#
# This is the ONE authoritative Kaggle CUDA kernel session (D-04): every v1.1 device
# self-oracle + the four RV-13 latent-parity oracles + the BENCH-02 depth-1/depth-6
# grow rows run in a SINGLE kernel invocation (one P100, one driver, one seed) and
# emit ONE verdict + ONE result.json. It supersedes the phase-14 aggregate.py
# multi-session stitching — we keep ONLY its >=20x verdict shape (GE20X_GATE).
#
# Structure (a clone of phase14_cuda_signoff/oracle.py; additive deltas marked NEW):
#
#   Part A — per-family CORRECTNESS gate (BLOCKING pre-flight, D-05). Runs every
#     v1.1 device family self-oracle (Phase-12 + Phase-13 *_test.rs in cb-backend /
#     cb-train) AND the four RV-13 hazard oracles under
#     --no-default-features --features cuda. Those tests compare the CUDA device
#     path to a Rust CPU reference at eps=1e-4 (`device_backend_active()` fires the
#     numeric asserts on the real cuda device). If ANY family fails or ran no tests,
#     we emit a SOME-FAIL verdict and sys.exit(2) BEFORE any timing — a fast-but-
#     wrong number is never quoted (D-05, do-not-fabricate).
#     NEW: the four RV-13 oracle test names ride the SAME cuda `cargo test`
#     invocations (tie_order_matches_cpu_stable_descending + softmax_weight_max_seed
#     on the ranking family; empty_group_means_no_fault on the ranking family's
#     query_helper filter; pairwise_near_equal_border_tiebreak on the pairwise
#     family's cholesky_solve filter).
#
#   Part B — NEW BENCH-02 depth-1 + depth-6 grow-speed rows (D-07). Only runs if
#     Part A is ALL-PASS. Times the device-accelerated grow loop vs the host-CPU
#     boosting loop via crates/cb-train/tests/bench_grow_speed_test.rs (CB_BENCH=1),
#     warm-run / JIT-excluded (the harness does an untimed 1-iter device warm-run) /
#     lazy-CubeCL-queue-drained (train() blocks on read-back) / median-of-N, at
#     20 iters / 20 feat / 32 bins, ONE depth per harness invocation via BENCH_DEPTH.
#     Depth-1 runs on the large-n SPEED_CONFIG (~1e6, Pitfall 5 / D-10-09) and the
#     crossover (n where device first beats CPU, or "not reached") is RECORDED, not
#     gated — depth-1 device>=CPU is NOT a pass condition (A4). Depth-6 keeps the
#     established 10k/100k/300k sweep for continuity with the existing 12 rows (D-06).
#
#   Part C — official-CatBoost-GPU informational timing arm (pure Python, D-08).
#     Times the official `catboost` package with task_type='GPU' on the SAME gen()
#     workload at each depth for the depthwise family; merged into Part B rows as the
#     informational `catboost_gpu_s` column. Region has NO official grow_policy → its
#     catboost_gpu_s stays N/A, never a fabricated proxy (Pitfall 4).
#
# Four INFORMATIONAL CatBoost-GPU divergences (per D-08 the CatBoost-GPU column is
# informational, NOT a gate): Region N/A; GPU border_count default 128 vs the bench 32
# (set explicitly); fit() wall-clock includes on-device quantization while catboost-rs
# times only the grow loop (NOT subtracted); pre-binned 0..31 columns fed as float32
# with border_count=32 (exact histogram parity not required).
#
# Do-not-fabricate discipline: every value comes from THIS run; correctness is gated
# BEFORE any speed number; a failed Part C leaves catboost_gpu_s null, never invented.
# Build lives in /tmp so the Kaggle output stays tiny (~1.8MB git-archive tarball, not
# the 2.9G crates/ tree); one compact result.json + result.md to /kaggle/working.
#
# All run-time work lives in main() under `if __name__ == "__main__"`, so importing
# this module in-env (no GPU, no numpy) exposes `gen`/`main` and triggers no run.

WORK = "/kaggle/working"

# BENCH-02 frozen grow config (module constants, no side effects). Matches
# crates/cb-train/tests/bench_grow_speed_test.rs::params (20 iters / 20 feat / 32 bins).
NF = 20
NBINS = 32
ITERS = 20
LEARNING_RATE = 0.3
L2_LEAF_REG = 0.0
RANDOM_STRENGTH = 0.0
RANDOM_SEED = 42

# The D-04/D-08 hard verdict shape kept verbatim from aggregate.py (GE20X_GATE=20.0):
# a depth-6 device row must be >= 20x the host CPU. Depth-1 is NOT gated on this
# (launch-overhead physics, A4 / Pitfall 5) — recorded with a crossover note instead.
GE20X_GATE = 20.0

# Part B depth rows (D-07). depth-6 keeps the existing 10k/100k/300k sweep (D-06);
# depth-1 runs the large-n SPEED_CONFIG so the crossover is observable (Pitfall 5).
DEPTH6_NS_DEFAULT = "10000,100000,300000"
DEPTH1_NS_DEFAULT = "100000,300000,1000000"
MEDIAN_N_DEFAULT = 3


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
    import os, subprocess, sys, shutil, time, json, re, statistics

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

    result = {"phase": 15, "kind": "cuda-oracle-single-session", "families": {}, "bench02": {}}

    log("=================== ENVIRONMENT ===================")
    sh("nvidia-smi", timeout=120)
    result["gpu"] = env_line("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
    result["driver"] = env_line("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    result["nvcc"] = env_line("nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1 || echo NO_NVCC")
    result["cuda_dirs"] = env_line("ls -d /usr/local/cuda* 2>/dev/null | tr '\\n' ' '")
    result["seed"] = RANDOM_SEED
    # Single-session provenance block (D-04): ONE gpu / driver / cuda / seed for the
    # whole verdict — this is what distinguishes it from aggregate.py stitching.
    result["provenance"] = {
        "gpu": result["gpu"], "driver": result["driver"], "cuda_ver": result["nvcc"],
        "seed": RANDOM_SEED, "single_session": True,
    }
    log("gpu:", result["gpu"], "| driver:", result["driver"], "| nvcc:", result["nvcc"],
        "| cuda:", result["cuda_dirs"])

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
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
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
    # PART A — v1.1 device correctness gate + 4 RV-13 oracles (BLOCKING, D-05)
    # (label, req, crate, extra-cargo-args, test-name filters) -> device self-oracles.
    # The union of the Phase-12 and Phase-13 families = all 44 v1.1 device self-oracles;
    # the 4 RV-13 oracle test names ride the ranking / pairwise families' SAME
    # `--features cuda` cargo invocations (D-03).
    # =====================================================================
    FAMILIES = [
        # ---- Phase-12 families (GPUT-09/10/17/18/19) ----
        ("nonsym_grow (Depthwise/Lossguide) GPUT-18", "GPUT-18", "cb-backend", [],
            ["nonsym_grow"]),
        ("region_device GPUT-18", "GPUT-18", "cb-backend", [],
            ["region_device"]),
        ("exact_quantile + segmented_sort GPUT-19", "GPUT-19", "cb-backend", [],
            ["exact_quantile", "segmented_sort"]),
        ("bootstrap_device GPUT-09", "GPUT-09", "cb-backend", [],
            ["bootstrap_device"]),
        ("mvs_device GPUT-17", "GPUT-17", "cb-backend", [],
            ["mvs_device"]),
        ("ctr_device GPUT-10", "GPUT-10", "cb-backend", [],
            ["ctr_device"]),
        ("device_nonsym_fit (e2e) GPUT-18", "GPUT-18", "cb-train",
            ["--test", "device_nonsym_fit_test"], []),
        ("device_region_fit (e2e) GPUT-18", "GPUT-18", "cb-train",
            ["--test", "device_region_fit_test"], []),
        # ---- Phase-13 families (GPUT-11/12/13/20/21/22) + RV-13 oracles ----
        ("pairwise (deriv + batched Cholesky) GPUT-11/21 [+RV-13-04]", "GPUT-11,GPUT-21",
            "cb-backend", [],
            ["pairwise_deriv", "cholesky_solve", "pairwise_near_equal_border_tiebreak"]),
        ("ranking (query grouping + det + stochastic) GPUT-22 [+RV-13-01/02/03]", "GPUT-22",
            "cb-backend", [],
            ["query_helper", "ranking_det", "ranking_stoch",
             "tie_order_matches_cpu_stable_descending", "softmax_weight_max_seed",
             "empty_group_means_no_fault"]),
        ("multiclass (softmax der + multi-Newton) GPUT-12", "GPUT-12", "cb-backend", [],
            ["multiclass", "multi_newton"]),
        ("ordered (resident approx trajectory) GPUT-13", "GPUT-13", "cb-backend", [],
            ["ordered"]),
        ("langevin (seeded Gaussian / SGLB) GPUT-20", "GPUT-20", "cb-backend", [],
            ["langevin"]),
    ]

    # The four RV-13 hazard oracles that MUST be reached in Part A (HARD-03 → HARD-01).
    RV13_ORACLES = [
        "tie_order_matches_cpu_stable_descending",   # RV-13-01 (ranking_stoch_test.rs)
        "softmax_weight_max_seed",                   # RV-13-02 (ranking_stoch_test.rs)
        "empty_group_means_no_fault",                # RV-13-03 (query_helper_test.rs)
        "pairwise_near_equal_border_tiebreak",       # RV-13-04 (cholesky_solve_test.rs)
    ]

    DIV_RE = re.compile(r"(abs|rel|div|diverg|max|eps|epsilon|1e-|e-1[0-9]|tol)", re.I)
    SUMMARY_RE = re.compile(r"test result:.*")

    for label, req, crate, extra, filters in FAMILIES:
        cmd = ["cargo", "test", "--release", "--no-default-features", "--features", "cuda",
               "-p", crate] + extra + ["--"] + filters + ["--nocapture", "--test-threads=1"]
        t0 = time.time()
        rc, out = sh(cmd, cwd=REPO, env=env, timeout=5400)
        dt = round(time.time() - t0, 1)
        summ = SUMMARY_RE.findall(out)
        divs = [l.strip() for l in out.splitlines() if DIV_RE.search(l) and ("test " not in l[:5])][:40]
        ran_any = any("running" in l and "test" in l for l in out.splitlines()) or bool(summ)
        # Confirm each RV-13 oracle riding this family actually appeared in the run
        # (a filter that reaches no test would silently under-count — do-not-fabricate).
        rv13_seen = [nm for nm in RV13_ORACLES if nm in out]
        result["families"][label] = {
            "req": req, "crate": crate, "filters": filters, "exit": rc, "secs": dt,
            "ran_any_tests": ran_any,
            "rv13_oracles_seen": rv13_seen,
            "summary": summ[-3:] if summ else [],
            "divergences": divs,
            "tail": out[-3000:],
        }
        log(f"[{label}] exit={rc} {summ[-1:] if summ else ''} rv13_seen={rv13_seen}")

    # ---- BLOCKING roll-up: correctness gates the speed arm (D-05, Pitfall 5) ----
    corr_pass = all(f["exit"] == 0 and f["ran_any_tests"] for f in result["families"].values())
    # Every RV-13 oracle must have been reached by exactly the family it rides.
    rv13_all_seen = sorted({nm for f in result["families"].values() for nm in f["rv13_oracles_seen"]})
    result["rv13_oracles_expected"] = RV13_ORACLES
    result["rv13_oracles_seen"] = rv13_all_seen
    rv13_ok = all(nm in rv13_all_seen for nm in RV13_ORACLES)
    corr_pass = corr_pass and rv13_ok
    result["correctness_verdict"] = "ALL-PASS" if corr_pass else "SOME-FAIL"
    if not corr_pass:
        # Do NOT run Part B/C on a failed pre-flight — never quote a fast-but-wrong number.
        result["bench_verdict"] = "SKIPPED (correctness pre-flight failed)"
        result["catboost_gpu_verdict"] = "SKIPPED (correctness pre-flight failed)"
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        with open(os.path.join(WORK, "result.md"), "w") as fh:
            fh.write(f"# Phase-15 single-session CUDA oracle\n\nGPU: {result['gpu']}\n")
            fh.write(f"correctness_verdict: {result['correctness_verdict']} — Part B/C SKIPPED\n")
            fh.write(f"rv13_oracles_seen: {rv13_all_seen}\n")
        log("CORRECTNESS_VERDICT:", result["correctness_verdict"], "-> aborting before timing")
        sys.exit(2)

    # =====================================================================
    # PART B — BENCH-02 depth-1 + depth-6 grow speed (device vs host CPU, D-07)
    # One harness invocation per (depth, repeat); median-of-N per (depth, family, n).
    # =====================================================================
    median_n = int(os.environ.get("BENCH_MEDIAN_N", str(MEDIAN_N_DEFAULT)))
    depth_cfgs = [
        (6, os.environ.get("BENCH_NS_DEPTH6", DEPTH6_NS_DEFAULT)),
        (1, os.environ.get("BENCH_NS_DEPTH1", DEPTH1_NS_DEFAULT)),
    ]
    BENCH_RE = re.compile(
        r"BENCH family=(\S+) n=(\d+) device_s=([\d.]+) cpu_s=([\d.]+) "
        r"speedup=([\d.naN]+)x dev_trees=(\d+) cpu_trees=(\d+) depth=(\d+)")

    # samples[(depth, family, n)] = {"device": [..], "cpu": [..], "dev_trees":.., "cpu_trees":..}
    samples = {}
    bench_meta = []
    bench_exit_ok = True
    for depth, ns_csv in depth_cfgs:
        for rep in range(median_n):
            benv = env.copy()
            benv["CB_BENCH"] = "1"
            benv["BENCH_DEPTH"] = str(depth)
            benv["BENCH_NS"] = ns_csv
            cmd = ["cargo", "test", "--release", "--no-default-features", "--features", "cuda",
                   "-p", "cb-train", "--test", "bench_grow_speed_test",
                   "--", "bench_grow_speed", "--nocapture", "--test-threads=1"]
            log(f"--- Part B depth={depth} rep={rep + 1}/{median_n} ns={ns_csv} ---")
            rc, out = sh(cmd, cwd=REPO, env=benv, timeout=9000)
            bench_exit_ok = bench_exit_ok and (rc == 0)
            for line in out.splitlines():
                if line.startswith("BENCH_META"):
                    bench_meta.append(line.strip())
                m = BENCH_RE.search(line)
                if not m:
                    continue
                fam, n, dev_s, cpu_s = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))
                row_depth = int(m.group(8))
                # Guard against a stale harness that ignores BENCH_DEPTH (would silently
                # report depth-6 for the depth-1 request — a fabrication we must catch).
                if row_depth != depth:
                    result.setdefault("bench_depth_mismatch", []).append(
                        {"requested": depth, "reported": row_depth, "family": fam, "n": n})
                    continue
                key = (depth, fam, n)
                s = samples.setdefault(key, {"device": [], "cpu": [],
                                             "dev_trees": int(m.group(6)),
                                             "cpu_trees": int(m.group(7))})
                s["device"].append(dev_s)
                s["cpu"].append(cpu_s)

    # median-of-N reduce → one depth_row per (depth, family, n)
    depth_rows = []
    for (depth, fam, n) in sorted(samples.keys(), key=lambda k: (k[0], k[1], k[2])):
        s = samples[(depth, fam, n)]
        dev_med = round(statistics.median(s["device"]), 4)
        cpu_med = round(statistics.median(s["cpu"]), 4)
        speedup = round(cpu_med / dev_med, 3) if dev_med > 0 else None
        depth_rows.append({
            "depth": depth, "family": fam, "n": n,
            "device_s": dev_med, "host_cpu_s": cpu_med,
            "catboost_gpu_s": None,          # filled by Part C (depthwise only)
            "speedup": speedup,
            "device_ge_cpu": (dev_med <= cpu_med) if dev_med > 0 else None,
            "n_reps": len(s["device"]),
            "dev_trees": s["dev_trees"], "cpu_trees": s["cpu_trees"],
        })

    # Depth-1 crossover (Pitfall 5 / D-10-09): smallest n where device first beats CPU
    # for the depthwise family, else "not reached". NOT a gate — recorded honestly (A4).
    d1_depthwise = sorted([r for r in depth_rows if r["depth"] == 1 and r["family"] == "depthwise"],
                          key=lambda r: r["n"])
    crossover_n = next((r["n"] for r in d1_depthwise if r["device_ge_cpu"]), None)
    crossover_note = (f"device first beats CPU at n={crossover_n}" if crossover_n is not None
                      else "not reached (device did not beat CPU at any tested depth-1 n)")

    result["bench02"] = {
        "depth_rows": depth_rows,
        "crossover": {"depth1_depthwise_n": crossover_n, "note": crossover_note},
        "ge20x_gate": GE20X_GATE,
        "median_n": median_n,
        "markers": bench_meta,
        "config": {
            "nf": NF, "nbins": NBINS, "iters": ITERS, "learning_rate": LEARNING_RATE,
            "l2_leaf_reg": L2_LEAF_REG, "random_strength": RANDOM_STRENGTH,
            "random_seed": RANDOM_SEED,
            "depth6_ns": depth_cfgs[0][1], "depth1_ns": depth_cfgs[1][1],
            "warm_run": "harness untimed 1-iter device warm-run (JIT excluded)",
            "queue": "lazy-CubeCL queue drained by train() read-back",
        },
    }
    bench_ok = bench_exit_ok and bool(depth_rows)
    result["bench_verdict"] = "OK" if bench_ok else "FAIL"
    # Depth-6 GE20X roll-up (kept from aggregate.py verdict shape; depth-1 excluded, A4).
    d6 = [r for r in depth_rows if r["depth"] == 6 and r["speedup"] is not None]
    result["bench02"]["depth6_ge20x"] = bool(d6) and all(r["speedup"] >= GE20X_GATE for r in d6)

    # =====================================================================
    # PART C — official-CatBoost-GPU informational timing arm (D-08)
    # Times catboost task_type='GPU' at each depth for the depthwise family; merged
    # into the depth_rows as `catboost_gpu_s`. Region stays N/A (Pitfall 4).
    # =====================================================================
    try:
        import catboost  # noqa: F401
    except Exception:
        sh("pip install -q catboost", env=env, timeout=1200)
    cb_version = "unavailable"
    try:
        from catboost import CatBoostRegressor
        cb_version = getattr(__import__("catboost"), "__version__", "unknown")
    except Exception as e:
        result["catboost_gpu_verdict"] = f"FAIL (catboost unavailable: {e})"
        CatBoostRegressor = None

    cat_divergences = [
        "Region has no official CatBoost grow_policy -> catboost_gpu_s N/A (not a proxy)",
        "GPU border_count default 128; set to 32 to match the bench",
        "CatBoost fit() wall-clock includes on-device quantization; catboost-rs times "
        "only the grow loop — NOT subtracted (informational, D-08)",
        "integer-binned 0..31 columns fed as float32 with border_count=32",
    ]
    if CatBoostRegressor is not None:
        cat_ok = True
        for depth, ns_csv in depth_cfgs:
            ns = [int(x) for x in ns_csv.split(",") if x.strip()]
            for n in ns:
                X, y = gen(n)
                m = CatBoostRegressor(
                    task_type="GPU", devices="0",
                    loss_function="RMSE", score_function="L2",
                    depth=depth, iterations=ITERS, learning_rate=LEARNING_RATE,
                    l2_leaf_reg=L2_LEAF_REG, random_strength=RANDOM_STRENGTH,
                    boost_from_average=False,
                    bootstrap_type="No", border_count=NBINS, random_seed=RANDOM_SEED,
                    grow_policy="Depthwise", verbose=False,
                )
                # warm one untimed fit (JIT / device-context init) — excluded from the clock.
                try:
                    m.fit(X[:2000], y[:2000])
                except Exception as e:
                    log(f"[catboost-gpu depth={depth} n={n}] warm fit FAILED: {e}")
                    cat_ok = False
                    continue
                t0 = time.time()
                try:
                    m.fit(X, y)                      # timed train-only fit
                    cat_gpu_s = round(time.time() - t0, 4)
                except Exception as e:
                    log(f"[catboost-gpu depth={depth} n={n}] timed fit FAILED: {e}")
                    cat_ok = False
                    continue
                # merge into the matching depthwise depth_row (Region left N/A).
                for r in depth_rows:
                    if r["depth"] == depth and r["family"] == "depthwise" and r["n"] == n:
                        r["catboost_gpu_s"] = cat_gpu_s
                log(f"[catboost-gpu depth={depth} n={n}] catboost_gpu_s={cat_gpu_s}")
        result["catboost_gpu_verdict"] = "OK" if cat_ok else "PARTIAL"
    result["catboost_gpu"] = {"version": cb_version, "divergences": cat_divergences}

    # ---- emit ONE compact result.json + one human-readable md ----
    result["date"] = time.strftime("%Y-%m-%d", time.gmtime())
    json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
    with open(os.path.join(WORK, "result.md"), "w") as fh:
        fh.write("# Phase-15 single-session CUDA oracle\n\n")
        fh.write(f"GPU: {result['gpu']}\nnvcc: {result['nvcc']}\ncatboost: {cb_version}\n")
        fh.write(f"correctness_verdict: {result['correctness_verdict']}\n")
        fh.write(f"bench_verdict: {result['bench_verdict']}\n")
        fh.write(f"catboost_gpu_verdict: {result.get('catboost_gpu_verdict')}\n")
        fh.write(f"rv13_oracles_seen: {rv13_all_seen}\n\n")
        fh.write("## BENCH-02 depth-1 / depth-6 grow speed (device vs host CPU)\n\n")
        fh.write("| depth | family | n | device_s | host_cpu_s | catboost_gpu_s | speedup | device>=CPU? |\n")
        fh.write("|---|---|---|---|---|---|---|---|\n")
        for r in depth_rows:
            cg = "N/A" if r["catboost_gpu_s"] is None and r["family"] == "region" else \
                ("TBD" if r["catboost_gpu_s"] is None else f"{r['catboost_gpu_s']:.4f}")
            sp = "n/a" if r["speedup"] is None else f"{r['speedup']:.3f}x"
            fh.write(f"| {r['depth']} | {r['family']} | {r['n']} | {r['device_s']:.4f} | "
                     f"{r['host_cpu_s']:.4f} | {cg} | {sp} | {r['device_ge_cpu']} |\n")
        fh.write(f"\nDepth-1 crossover: {crossover_note}\n")
    log("CORRECTNESS_VERDICT:", result["correctness_verdict"])
    log("BENCH_VERDICT:", result["bench_verdict"], "rows:", len(depth_rows))
    log("CROSSOVER:", crossover_note)
    log("=================== PHASE-15 SINGLE-SESSION ORACLE COMPLETE ===================")


if __name__ == "__main__":
    main()
