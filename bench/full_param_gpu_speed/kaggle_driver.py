#!/usr/bin/env python3
# Kaggle P100 driver for the catboost-rs full-parameter GPU speed grid (SPD-03).
#
# Kaggle kernels take ONE code file, so this driver clones the repo at the pinned branch
# and hands off to `bench/full_param_gpu_speed/bench.py`, which holds the entire grid, the
# eligibility audit, the CB_GPU_PROF residency probe and the report writer. Nothing about
# the benchmark lives here — that keeps the committed harness and what actually ran the
# same file, which is the whole point of cloning rather than pasting.
#
# The dataset-source route the older kernels used is gone (those datasets were deleted), so
# provenance comes from the clone itself: the driver records the resolved commit SHA and
# the branch it was asked for, and the report carries them.

import json
import os
import subprocess
import sys
import time

REPO_URL = os.environ.get("CB_REPO_URL", "https://github.com/BectorVoom/catboost_rs.git")
BRANCH = os.environ.get("CB_BRANCH", "worktree-gpu-full-parameter-parity")
REPO = "/tmp/repo"
WORK = "/kaggle/working"


def sh(cmd, cwd=None, timeout=3600, env=None):
    print(f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd, shell=isinstance(cmd, str), cwd=cwd, env=env,
        capture_output=True, text=True, timeout=timeout,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out[-4000:], flush=True)
    return proc.returncode, out


def main():
    os.makedirs(WORK, exist_ok=True)
    started = time.time()
    prov = {"branch": BRANCH, "repo_url": REPO_URL}

    rc, out = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    prov["gpu"] = out.strip() if rc == 0 else f"nvidia-smi failed: {out[:200]}"
    if rc != 0:
        # Do-not-fabricate: without a real GPU there is no speed number to report.
        json.dump({"provenance": prov, "errors": {"gpu": "no GPU visible"}},
                  open(os.path.join(WORK, "result.json"), "w"), indent=2)
        print("NO GPU — aborting before any timing (do-not-fabricate).")
        return 1

    rc, _ = sh(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO],
               timeout=1800)
    if rc != 0:
        json.dump({"provenance": prov, "errors": {"clone": "git clone failed"}},
                  open(os.path.join(WORK, "result.json"), "w"), indent=2)
        return 1
    _rc, sha = sh(["git", "rev-parse", "HEAD"], cwd=REPO, timeout=120)
    prov["commit"] = sha.strip().splitlines()[-1] if sha.strip() else "unknown"
    print(f"cloned {BRANCH} at {prov['commit']}", flush=True)

    # Install a Rust toolchain if the image lacks one (the CUDA images usually do).
    rc, _ = sh("cargo --version")
    if rc != 0:
        sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
           timeout=1800)
        os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"]
        sh("cargo --version")

    env = dict(os.environ, CB_BENCH_WORK=WORK, CB_BENCH_REPO=REPO)
    bench = os.path.join(REPO, "bench", "full_param_gpu_speed", "bench.py")

    # Review the grid in the kernel log before spending the session on it.
    sh([sys.executable, bench, "--dry-run"], env=env, timeout=600)

    rc, _ = sh([sys.executable, bench], env=env, timeout=int(11 * 3600))

    # Stamp provenance into whatever result.json the harness wrote.
    path = os.path.join(WORK, "result.json")
    if os.path.exists(path):
        try:
            data = json.load(open(path))
            data.setdefault("provenance", {}).update(prov)
            data["provenance"]["driver_elapsed_s"] = round(time.time() - started, 1)
            json.dump(data, open(path, "w"), indent=2)
        except Exception as e:  # never let a provenance stamp lose the results
            print(f"provenance stamp failed (results preserved): {e!r}")
    print(f"done rc={rc} elapsed={time.time() - started:.0f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
