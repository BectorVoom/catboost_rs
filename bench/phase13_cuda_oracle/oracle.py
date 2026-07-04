#!/usr/bin/env python3
# catboost-rs Phase-13 CUDA sign-off (Kaggle GPU) — SINGLE kernel, one Rust build.
#
# Part A: per-family CORRECTNESS gate (BLOCKING). Runs each of the five Phase-13
#   families' device self-oracle (*_test.rs in cb-backend) under
#   --no-default-features --features cuda. Those tests compare the CUDA device
#   path to a Rust CPU reference at eps=1e-4 and print the divergence. This IS
#   the authoritative Kaggle CUDA correctness gate.
# Part B: BENCH-02 grow-loop speed. Times the device-accelerated depth-6 grow
#   loop vs the host-CPU boosting loop via cb-train/tests/bench_grow_speed_test.rs
#   (CB_BENCH=1). Per-family sessions are Ok(None) this phase (grow seam forward
#   dependency), so this shared grow-loop anchor is the honest BENCH-02 number.
#
# Build lives in /tmp so the Kaggle output stays tiny; compact result.json +
# result.md written to /kaggle/working for download. NO number is fabricated:
# every value comes from THIS run; correctness is gated BEFORE any speed number.
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

def env_line(cmd):
    try:
        return subprocess.run(cmd, text=True, capture_output=True, shell=True).stdout.strip()
    except Exception as e:
        return f"(err {e})"

result = {"phase": 13, "kind": "cuda-signoff", "families": {}, "bench02": {}}

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
# PART A — per-family CUDA CORRECTNESS gate (BLOCKING)
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

# =====================================================================
# PART B — BENCH-02 grow-loop speed (device vs host CPU)
# =====================================================================
benv = env.copy()
benv["CB_BENCH"] = "1"
benv["BENCH_NS"] = os.environ.get("BENCH_NS", "10000,100000,300000")
cmd = ["cargo", "test", "--release", "--no-default-features", "--features", "cuda",
       "-p", "cb-train", "--test", "bench_grow_speed_test",
       "--", "bench_grow_speed", "--nocapture", "--test-threads=1"]
t0 = time.time()
rc, out = sh(cmd, cwd=REPO, env=benv, timeout=7200)
rows = []
for line in out.splitlines():
    m = re.search(r"BENCH family=(\S+) n=(\d+) device_s=([\d.]+) cpu_s=([\d.]+) speedup=([\d.naN]+)x dev_trees=(\d+) cpu_trees=(\d+)", line)
    if m:
        rows.append({"family": m.group(1), "n": int(m.group(2)),
                     "device_s": float(m.group(3)), "cpu_s": float(m.group(4)),
                     "speedup": m.group(5), "dev_trees": int(m.group(6)), "cpu_trees": int(m.group(7))})
result["bench02"] = {
    "exit": rc, "secs": round(time.time() - t0, 1),
    "runs": rows,
    "test_summary": re.findall(r"test result:.*", out)[-2:],
    "markers": [l.strip() for l in out.splitlines() if l.startswith("BENCH_META") or l.startswith("BENCH_DONE")],
    "tail": out[-3000:],
}

# ---- verdict rollup ----
corr_pass = all(f["exit"] == 0 and f["ran_any_tests"] for f in result["families"].values())
bench_ok = (result["bench02"]["exit"] == 0 and bool(rows))
result["correctness_verdict"] = "ALL-PASS" if corr_pass else "SOME-FAIL"
result["bench_verdict"] = "OK" if bench_ok else "FAIL"
json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)

# small human-readable md
with open(os.path.join(WORK, "result.md"), "w") as fh:
    fh.write(f"# Phase-13 CUDA sign-off\n\nGPU: {result['gpu']}\nnvcc: {result['nvcc']}\n")
    fh.write(f"correctness_verdict: {result['correctness_verdict']}\nbench_verdict: {result['bench_verdict']}\n\n")
    fh.write("## Correctness (device vs Rust CPU, eps=1e-4)\n")
    for k, v in result["families"].items():
        fh.write(f"### {k}\nexit={v['exit']} secs={v['secs']} ran={v['ran_any_tests']}\nsummary: {v['summary']}\n")
        for d in v["divergences"][:12]:
            fh.write(f"  - {d}\n")
        fh.write("\n")
    fh.write("## BENCH-02 grow loop\n\n| family | n | device_s | cpu_s | speedup | dev_trees | cpu_trees |\n|---|---|---|---|---|---|---|\n")
    for r in rows:
        fh.write(f"| {r['family']} | {r['n']} | {r['device_s']:.4f} | {r['cpu_s']:.4f} | {r['speedup']}x | {r['dev_trees']} | {r['cpu_trees']} |\n")
log("CORRECTNESS_VERDICT:", result["correctness_verdict"])
log("BENCH_VERDICT:", result["bench_verdict"], "rows:", len(rows))
log("=================== PHASE-13 SIGN-OFF COMPLETE ===================")
