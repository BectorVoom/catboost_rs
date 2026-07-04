#!/usr/bin/env python3
# catboost-rs Phase-12 BENCH-02 speed kernel (Kaggle GPU).
# Times the device-accelerated grow loop vs the host-CPU boosting loop (Depthwise +
# Region) at large-n, train-only, warm-run (JIT excluded), in ONE --features cuda
# binary via crates/cb-train/tests/bench_grow_speed_test.rs (CB_BENCH=1). Writes a
# compact bench-result.json to /kaggle/working; build lives in /tmp.
import os, subprocess, sys, shutil, time, json, re

WORK = "/kaggle/working"
os.makedirs(WORK, exist_ok=True)

def log(*a):
    print(*a, flush=True)

def sh(cmd, cwd=None, env=None, timeout=None):
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

result = {"phase": 12, "kind": "cuda-bench02-speed", "runs": []}
log("=== ENV ==="); sh("nvidia-smi", timeout=120)
result["gpu"] = env_line("nvidia-smi --query-gpu=name --format=csv,noheader")
result["nvcc"] = env_line("nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1")

# stage repo into /tmp
inp = "/kaggle/input"; tarball = srcdir = None
for dp, _, fs in os.walk(inp):
    if "repo.tar.gz" in fs: tarball = os.path.join(dp, "repo.tar.gz")
    if "Cargo.toml" in fs and os.path.exists(os.path.join(dp, "crates")): srcdir = dp
REPO = "/tmp/repo"
if os.path.exists(REPO): shutil.rmtree(REPO)
os.makedirs(REPO)
if tarball: sh(["tar", "-xzf", tarball, "-C", REPO])
elif srcdir: sh(["bash", "-lc", f"cp -a '{srcdir}/.' '{REPO}/'"])
else:
    result["fatal"] = "payload not found"
    json.dump(result, open(os.path.join(WORK, "bench-result.json"), "w"), indent=2); sys.exit(2)
assert os.path.exists(os.path.join(REPO, "crates/cb-train/tests/bench_grow_speed_test.rs")), \
    "bench harness missing from payload — dataset not refreshed?"

# rust
sh("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal")
env = os.environ.copy()
env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env["PATH"]
env["CARGO_TARGET_DIR"] = "/tmp/target"
env["CARGO_BUILD_JOBS"] = "2"; env["CARGO_NET_RETRY"] = "5"; env["RUST_BACKTRACE"] = "1"
env["CB_BENCH"] = "1"
env["BENCH_NS"] = os.environ.get("BENCH_NS", "10000,100000,300000")
sh("rustc --version && cargo --version", env=env)

cmd = ["cargo", "test", "--release", "--no-default-features", "--features", "cuda",
       "-p", "cb-train", "--test", "bench_grow_speed_test",
       "--", "bench_grow_speed", "--nocapture", "--test-threads=1"]
t0 = time.time()
try:
    rc, out = sh(cmd, cwd=REPO, env=env, timeout=7200)
except subprocess.TimeoutExpired:
    rc, out = 124, "TIMEOUT"
result["exit"] = rc
result["secs"] = round(time.time() - t0, 1)

# parse BENCH lines: "BENCH family=depthwise n=100000 device_s=.. cpu_s=.. speedup=..x dev_trees=.. cpu_trees=.."
rows = []
for line in out.splitlines():
    m = re.search(r"BENCH family=(\S+) n=(\d+) device_s=([\d.]+) cpu_s=([\d.]+) speedup=([\d.naN]+)x dev_trees=(\d+) cpu_trees=(\d+)", line)
    if m:
        rows.append({"family": m.group(1), "n": int(m.group(2)),
                     "device_s": float(m.group(3)), "cpu_s": float(m.group(4)),
                     "speedup": m.group(5), "dev_trees": int(m.group(6)), "cpu_trees": int(m.group(7))})
    if line.startswith("BENCH_META") or line.startswith("BENCH_DONE"):
        result.setdefault("markers", []).append(line.strip())
result["runs"] = rows
result["test_summary"] = re.findall(r"test result:.*", out)[-2:]
result["tail"] = out[-3000:]
result["verdict"] = "OK" if (rc == 0 and rows) else "FAIL"
json.dump(result, open(os.path.join(WORK, "bench-result.json"), "w"), indent=2)

with open(os.path.join(WORK, "bench-result.md"), "w") as fh:
    fh.write(f"# Phase-12 BENCH-02 speed ({result['gpu']}, nvcc {result['nvcc']})\n\n")
    fh.write("| family | n | device_s | cpu_s | speedup | dev_trees | cpu_trees |\n|---|---|---|---|---|---|---|\n")
    for r in rows:
        fh.write(f"| {r['family']} | {r['n']} | {r['device_s']:.4f} | {r['cpu_s']:.4f} | {r['speedup']}x | {r['dev_trees']} | {r['cpu_trees']} |\n")
log("BENCH_VERDICT:", result["verdict"], "rows:", len(rows))
log("=== BENCH COMPLETE ===")
