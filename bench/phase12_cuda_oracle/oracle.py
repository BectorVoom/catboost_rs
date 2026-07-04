#!/usr/bin/env python3
# catboost-rs Phase-12 CUDA CORRECTNESS oracle (Kaggle GPU).
# Runs each family's existing device self-oracle (*_test.rs) under
# --no-default-features --features cuda. Those tests compare the device path to a
# CPU reference at eps=1e-4 and print the divergence. This IS the authoritative
# Kaggle CUDA correctness gate (Task 1 of plan 12-09). Build lives in /tmp so the
# Kaggle output stays tiny; a compact result.json + result.md is written to
# /kaggle/working for download.
import os, subprocess, sys, glob, shutil, time, json, re

WORK = "/kaggle/working"
os.makedirs(WORK, exist_ok=True)

def log(*a):
    print(*a, flush=True)

def sh(cmd, cwd=None, env=None, timeout=None):
    """Run, stream-capture, return (rc, combined_output)."""
    log("\n$", cmd if isinstance(cmd, str) else " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, env=env,
                       shell=isinstance(cmd, str), timeout=timeout)
    out = (r.stdout or "") + (("\nSTDERR:\n" + r.stderr) if r.stderr else "")
    log(out[-6000:])
    return r.returncode, out

def env_line(cmd):
    try:
        return subprocess.run(cmd, text=True, capture_output=True, shell=True).stdout.strip()
    except Exception as e:
        return f"(err {e})"

result = {"phase": 12, "kind": "cuda-correctness-oracle", "families": {}}

log("=================== ENVIRONMENT ===================")
sh("nvidia-smi", timeout=120)
result["gpu"] = env_line("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
result["nvcc"] = env_line("nvcc --version 2>/dev/null | tail -2 | head -1 || echo NO_NVCC")
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

# ---- per-family CUDA correctness runs ----
# (label, crate, extra cargo args, test-name filters)
FAMILIES = [
    ("nonsym_grow (Depthwise/Lossguide) GPUT-18", "cb-backend", [], ["nonsym_grow"]),
    ("region_device GPUT-18",                     "cb-backend", [], ["region_device"]),
    ("exact_quantile + segmented_sort GPUT-19",   "cb-backend", [], ["exact_quantile", "segmented_sort"]),
    ("bootstrap_device GPUT-09",                  "cb-backend", [], ["bootstrap_device"]),
    ("mvs_device GPUT-17",                        "cb-backend", [], ["mvs_device"]),
    ("ctr_device GPUT-10",                        "cb-backend", [], ["ctr_device"]),
    ("device_nonsym_fit (e2e) GPUT-18",           "cb-train", ["--test", "device_nonsym_fit_test"], []),
    ("device_region_fit (e2e) GPUT-18",           "cb-train", ["--test", "device_region_fit_test"], []),
]

DIV_RE = re.compile(r"(abs|rel|div|diverg|max|eps|epsilon|1e-|e-1[0-9])", re.I)
SUMMARY_RE = re.compile(r"test result:.*")

for label, crate, extra, filters in FAMILIES:
    cmd = ["cargo", "test", "--release", "--no-default-features", "--features", "cuda",
           "-p", crate] + extra + ["--"] + filters + ["--nocapture", "--test-threads=1"]
    t0 = time.time()
    try:
        rc, out = sh(cmd, cwd=REPO, env=env, timeout=5400)
    except subprocess.TimeoutExpired:
        rc, out = 124, "TIMEOUT"
    dt = round(time.time() - t0, 1)
    summ = SUMMARY_RE.findall(out)
    divs = [l.strip() for l in out.splitlines() if DIV_RE.search(l) and ("test " not in l[:5])][:40]
    result["families"][label] = {
        "crate": crate, "filters": filters, "exit": rc, "secs": dt,
        "summary": summ[-3:] if summ else [],
        "divergences": divs,
        "tail": out[-2500:],
    }
    log(f"[{label}] exit={rc} {summ[-1:] if summ else ''}")

# ---- verdict rollup ----
allpass = all(f["exit"] == 0 for f in result["families"].values())
result["verdict"] = "ALL-PASS" if allpass else "SOME-FAIL"
json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)

# small human-readable md
with open(os.path.join(WORK, "result.md"), "w") as fh:
    fh.write(f"# Phase-12 CUDA correctness oracle\n\nGPU: {result['gpu']}\nnvcc: {result['nvcc']}\nverdict: {result['verdict']}\n\n")
    for k, v in result["families"].items():
        fh.write(f"## {k}\nexit={v['exit']} secs={v['secs']}\nsummary: {v['summary']}\n")
        for d in v["divergences"][:12]:
            fh.write(f"  - {d}\n")
        fh.write("\n")
log("ORACLE_VERDICT:", result["verdict"])
log("=================== ORACLE COMPLETE ===================")
