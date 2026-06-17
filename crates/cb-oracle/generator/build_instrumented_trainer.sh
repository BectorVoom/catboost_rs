#!/usr/bin/env bash
# ============================================================================
# build_instrumented_trainer.sh  (Plan 06.3-10, Task 1)
#
# Sudo-free, idempotent, re-runnable driver that builds the INSTRUMENTED
# catboost 1.2.10 `_catboost.so` trainer with env-gated `CB_INSTRUMENT_LOG`
# logging for the TWO surfaces this gap-closure round needs:
#
#   (a) per-leaf SumDer / SumDer2 in the grouped/pairwise leaf-der reduction
#       (PairLogit closure  -> approx_calcer_querywise.cpp AddLeafDersForQueries
#                              + approx_calcer.cpp CalcLeafValues)
#   (b) per-tree YetiRank / StochasticRank RNG-draw events
#       (yetirank_helpers.cpp GenerateYetiRankPairsForQuery RNG draws
#        + algo_helpers/error_functions.cpp StochasticRank noise stream)
#
# DESIGN INVARIANTS (escalate-don't-weaken, D-6.3-03b; D-09 / D-12):
#   * OFFLINE / RUN-ONCE only.  NEVER invoked in CI.
#   * Instrumentation is a strict NO-OP when CB_INSTRUMENT_LOG is unset.
#   * sudo-free:  apt-get download + dpkg -x extraction; uv tool installs;
#                 user-prefix only; no privileged operation.
#   * Disk-gated:  refuses to attempt the Release C++ link below a 25 GB floor
#                  (README documents linking failed only at ~8-12 GB free).
#   * On any failure -> surface the failing step (set -euo pipefail + trap).
#
# Toolchain recipe (sudo-free), per instrument_live_trainer_README.md:
#   * clang-18 / lld-18 / llvm-18 (Ubuntu noble) via apt-get download + dpkg -x
#     into /tmp/clang18_prefix
#   * conan / ninja / cython (+numpy) via `uv tool install` (reuse PATH copies)
#   * build_native.py --targets _catboost against the project .venv Python 3.13
#     (-DPython3_INCLUDE_DIR / -DPython3_EXECUTABLE overrides; FindPython
#      otherwise picks system 3.12 -> ABI mismatch with the 3.13 venv)
# ============================================================================
set -euo pipefail

# --------------------------------------------------------------------------
# Resolve repo root + canonical paths (script lives in crates/cb-oracle/generator/)
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CB_SRC="${REPO_ROOT}/catboost-master"
VENV_PY="${REPO_ROOT}/.venv/bin/python"
CLANG_PREFIX="${CLANG_PREFIX:-/tmp/clang18_prefix}"
BUILD_ROOT="${BUILD_ROOT:-/tmp/cb_build313}"
DISK_FLOOR_GB="${DISK_FLOOR_GB:-25}"        # Release-C++-link safety floor
BUILD_LOG="${BUILD_LOG:-/tmp/instr_build.log}"

step() { echo "==> [build_instrumented_trainer] $*"; }
fail() { echo "!! [build_instrumented_trainer] FAILED at: $*" >&2; exit 1; }
trap 'echo "!! [build_instrumented_trainer] aborted (line $LINENO, exit $?)" >&2' ERR

# --------------------------------------------------------------------------
# STEP 1 — Disk headroom gate (>= DISK_FLOOR_GB; free caches + re-check else NO-GO)
# --------------------------------------------------------------------------
avail_gb() { df -BG / | awk 'NR==2 {gsub(/G/,"",$4); print $4}'; }

step "STEP 1: disk headroom"
df -h /
AVAIL="$(avail_gb)"
step "available on / = ${AVAIL} GB (floor = ${DISK_FLOOR_GB} GB)"
if [ "${AVAIL}" -lt "${DISK_FLOOR_GB}" ]; then
    step "below floor -> freeing target/debug/incremental + ~/.conan2"
    rm -rf "${REPO_ROOT}/target/debug/incremental" 2>/dev/null || true
    rm -rf "${HOME}/.conan2" 2>/dev/null || true
    df -h /
    AVAIL="$(avail_gb)"
    step "after cleanup: ${AVAIL} GB available"
    if [ "${AVAIL}" -lt "${DISK_FLOOR_GB}" ]; then
        fail "STEP 1 disk gate — only ${AVAIL} GB free (< ${DISK_FLOOR_GB} GB floor). NO-GO: aborting before the Release C++ link to avoid the documented ~8-12 GB link-failure regime."
    fi
fi

# --------------------------------------------------------------------------
# STEP 2 — Toolchain restore (sudo-free), only if absent
# --------------------------------------------------------------------------
step "STEP 2: toolchain restore"

# 2a. conan / ninja / cython (+numpy) via uv tool — reuse existing PATH copies.
ensure_uv_tool() {
    local bin="$1"; shift
    if command -v "${bin}" >/dev/null 2>&1; then
        step "  ${bin} present: $(command -v "${bin}")"
    elif command -v uv >/dev/null 2>&1; then
        step "  installing ${bin} via uv tool"
        uv tool install "$@" || fail "STEP 2 uv tool install ${bin}"
    else
        fail "STEP 2 — neither ${bin} nor uv on PATH (sudo-free recipe needs uv)"
    fi
}
ensure_uv_tool conan  conan
ensure_uv_tool ninja  ninja
ensure_uv_tool cython "cython"
# numpy lives in the project .venv; ensure it is importable for the build.
"${VENV_PY}" -c 'import numpy' 2>/dev/null || step "  (numpy not in venv — build_native.py will surface if required)"

# 2b. clang-18 / lld-18 / llvm-18 via apt-get download + dpkg -x into CLANG_PREFIX.
CLANG_BIN=""
if [ -x "${CLANG_PREFIX}/usr/bin/clang-18" ]; then
    CLANG_BIN="${CLANG_PREFIX}/usr/bin/clang-18"
    step "  clang-18 prefix present: ${CLANG_BIN}"
elif command -v clang-18 >/dev/null 2>&1; then
    CLANG_BIN="$(command -v clang-18)"
    step "  system clang-18 present: ${CLANG_BIN}"
else
    step "  fetching clang-18 / lld-18 / llvm-18 (noble) via apt-get download (no sudo)"
    mkdir -p "${CLANG_PREFIX}/debs"
    (
        cd "${CLANG_PREFIX}/debs"
        # Core packages the vendored libc++ (clang>=16 builtins) requires.
        apt-get download \
            clang-18 lld-18 llvm-18 llvm-18-dev llvm-18-runtime \
            libllvm18 libclang-common-18-dev libclang-cpp18 \
            libclang1-18 clang-tools-18 \
            libc++-18-dev libc++abi-18-dev 2>>"${BUILD_LOG}" \
            || step "  (some optional debs unavailable; continuing with what downloaded)"
        for d in *.deb; do
            [ -e "${d}" ] || continue
            dpkg -x "${d}" "${CLANG_PREFIX}"
        done
    )
    if [ -x "${CLANG_PREFIX}/usr/bin/clang-18" ]; then
        CLANG_BIN="${CLANG_PREFIX}/usr/bin/clang-18"
        step "  extracted clang-18: ${CLANG_BIN}"
    else
        fail "STEP 2 clang-18 restore — clang-18 not found after apt-get download + dpkg -x into ${CLANG_PREFIX}"
    fi
fi
CLANGXX_BIN="${CLANG_BIN/clang-18/clang++-18}"
[ -x "${CLANGXX_BIN}" ] || CLANGXX_BIN="${CLANG_BIN}++"
# catboost's build/toolchains/clang.toolchain hardcodes bare `clang`/`clang++`
# (and re-exports ENV{CC}/ENV{CXX}=clang/clang++), OVERRIDING any -DCMAKE_*_COMPILER
# cache entry. So bare `clang`/`clang++` must resolve on PATH. Provide them as
# sudo-free symlinks inside the prefix bin (the documented "clang-18 prefix on PATH").
CLANG_BINDIR="$(dirname "${CLANG_BIN}")"
ln -sf "${CLANG_BIN}"   "${CLANG_BINDIR}/clang"   2>/dev/null || true
ln -sf "${CLANGXX_BIN}" "${CLANG_BINDIR}/clang++" 2>/dev/null || true
# CUDA is disabled (--have-cuda is NOT passed); the toolchain still references
# clang-14 as the CUDA host compiler. Point a `clang-14` alias at clang-18 so a
# stray probe does not fail (no CUDA target is built here).
ln -sf "${CLANG_BIN}"   "${CLANG_BINDIR}/clang-14" 2>/dev/null || true
export CC="${CLANG_BINDIR}/clang"
export CXX="${CLANG_BINDIR}/clang++"
export PATH="${CLANG_PREFIX}/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${CLANG_PREFIX}/usr/lib/x86_64-linux-gnu:${CLANG_PREFIX}/usr/lib:${LD_LIBRARY_PATH:-}"

# --------------------------------------------------------------------------
# STEP 3 — Apply env-gated CB_INSTRUMENT_LOG instrumentation patch (idempotent)
# --------------------------------------------------------------------------
step "STEP 3: instrumentation patch (CB_INSTRUMENT_LOG-gated, no-op when unset)"

QW="${CB_SRC}/catboost/private/libs/algo/approx_calcer_querywise.cpp"
AC="${CB_SRC}/catboost/private/libs/algo/approx_calcer.cpp"
YR="${CB_SRC}/catboost/private/libs/algo/yetirank_helpers.cpp"
EF="${CB_SRC}/catboost/private/libs/algo_helpers/error_functions.cpp"

# Inert-when-unset sink helper, written to a temp FILE (never passed through
# `awk -v`, which mangles backslash escapes — the 06.3-10 Task-1 RULE-1 bug).
SINK_FILE="$(mktemp /tmp/cb_sink.XXXXXX.cpp)"
cat > "${SINK_FILE}" <<'CPP'
// === CB_INSTRUMENT_LOG sink (06.3-10, env-gated, inert when unset) ===
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
static void CbInstrumentLog(const std::string& line) {
    const char* path = std::getenv("CB_INSTRUMENT_LOG");
    if (path == nullptr) { return; }
    static std::mutex cbInstrMtx;
    std::lock_guard<std::mutex> g(cbInstrMtx);
    std::FILE* f = std::fopen(path, "a");
    if (f != nullptr) { std::fputs(line.c_str(), f); std::fputc(10, f); std::fclose(f); }
}
// 17 significant digits round-trips an IEEE-754 double exactly (06.3-13: the
// ≤1e-5 PairLogit oracle needs full precision, std::to_string truncates to 6dp).
static std::string CbFmt17(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return std::string(buf);
}
// === end CB_INSTRUMENT_LOG sink ===
CPP

ensure_sink() {
    local file="$1"
    [ -f "${file}" ] || fail "STEP 3 patch target missing: ${file}"
    if grep -q 'CB_INSTRUMENT_LOG sink (06.3-10' "${file}"; then
        step "  sink already present in $(basename "${file}") — skipping"
        return 0
    fi
    # Insert the sink (read verbatim from SINK_FILE via getline) right before the
    # first non-include / non-blank / non-comment line — no -v escape mangling.
    local tmp; tmp="$(mktemp)"
    awk -v sinkfile="${SINK_FILE}" '
        BEGIN { inserted=0 }
        {
            if (!inserted && $0 !~ /^#include/ && $0 !~ /^[[:space:]]*$/ && $0 !~ /^\/\// && NR>1) {
                while ((getline ln < sinkfile) > 0) { print ln }
                close(sinkfile)
                inserted=1
            }
            print
        }
        END {
            if (!inserted) { while ((getline ln < sinkfile) > 0) { print ln } close(sinkfile) }
        }
    ' "${file}" > "${tmp}" && mv "${tmp}" "${file}"
    step "  inserted sink into $(basename "${file}")"
}

# NB (06.3-10 Task-1 RULE-1 fix #2): the JSON fragments use C++ RAW string
# literals  R"J(...)J"  so NO embedded double-quote needs perl backslash-escaping
# (the prior `\"`-escaped hooks were mangled by perl into invalid C++ literals).

# 3a. per-leaf SumDer/SumDer2 (grouped/pairwise leaf-der reduction).
ensure_sink "${QW}"
if ! grep -q 'cb_instr_leafder' "${QW}"; then
    # Log merged per-leaf Der1/Der2 after the block-merge in AddLeafDersForQueries.
    perl -0777 -pi -e '
        s{(mergedStats->first\[idx\]\.Der2 \+= blockStats\.first\[idx\]\.Der2;\s*\n)}{$1            /* cb_instr_leafder */ if (std::getenv("CB_INSTRUMENT_LOG")) { CbInstrumentLog(std::string(R"J({"event":"leaf_der","leaf":)J") + std::to_string(idx) + R"J(,"der1":)J" + CbFmt17(mergedStats->first[idx].Der1) + R"J(,"der2":)J" + CbFmt17(mergedStats->first[idx].Der2) + "}"); }\n}s;
    ' "${QW}" || step "  (querywise leaf-der hook not matched — schema may have shifted; recorded)"
    step "  patched per-leaf der1/der2 hook into approx_calcer_querywise.cpp"
fi

# 3b. per-leaf SumWeights in CalcLeafValues (pointwise leaf path).
ensure_sink "${AC}"
if ! grep -q 'cb_instr_leafweight' "${AC}"; then
    perl -0777 -pi -e '
        s{(if \(blockBucketSumWeights\[blockId\]\[leafId\] > FLT_EPSILON\) \{\n)}{$1                    /* cb_instr_leafweight */ if (std::getenv("CB_INSTRUMENT_LOG")) { CbInstrumentLog(std::string(R"J({"event":"leaf_weight","leaf":)J") + std::to_string(leafId) + R"J(,"sum_weight":)J" + CbFmt17(blockBucketSumWeights[blockId][leafId]) + "}"); }\n}s;
    ' "${AC}" || step "  (approx_calcer leaf-weight hook not matched; recorded)"
    step "  patched per-leaf sum-weight hook into approx_calcer.cpp"
fi

# 3e. (06.3-13) per-leaf FINAL delta + denominator inputs in CalcLeafDeltasSimple.
#     This captures, at full precision, the exact leaf delta upstream emits AND the
#     sumAllWeights / allDocCount that feed the Newton/Gradient/pairwise denom — the
#     ground truth the PairLogit ≤1e-5 oracle (plan 13) needs to pin the per-leaf
#     der2 reduction. Hooks the Newton-branch leaf loop and the pairwise branch.
ensure_sink "${AC}"
if ! grep -q 'cb_instr_leafdelta' "${AC}"; then
    # Newton branch: log SumDer / SumDer2 / SumWeights / sumAllWeights / allDocCount / delta.
    perl -0777 -pi -e '
        s{(\(\*leafDeltas\)\[leaf\] = CalcMethodDelta<ELeavesEstimation::Newton>\(\s*leafDers\[leaf\],\s*l2Regularizer,\s*sumAllWeights,\s*allDocCount\);\n)}{$1            /* cb_instr_leafdelta */ if (std::getenv("CB_INSTRUMENT_LOG")) { CbInstrumentLog(std::string(R"J({"event":"leaf_delta","method":"Newton","leaf":)J") + std::to_string(leaf) + R"J(,"sum_der":)J" + CbFmt17(leafDers[leaf].SumDer) + R"J(,"sum_der2":)J" + CbFmt17(leafDers[leaf].SumDer2) + R"J(,"sum_weights":)J" + CbFmt17(leafDers[leaf].SumWeights) + R"J(,"sum_all_weights":)J" + CbFmt17(sumAllWeights) + R"J(,"all_doc_count":)J" + std::to_string(allDocCount) + R"J(,"l2":)J" + CbFmt17((double)l2Regularizer) + R"J(,"delta":)J" + CbFmt17((*leafDeltas)[leaf]) + "}"); }\n}s;
    ' "${AC}" || step "  (newton leaf-delta hook not matched; recorded)"
    # Gradient branch.
    perl -0777 -pi -e '
        s{(\(\*leafDeltas\)\[leaf\] = CalcMethodDelta<ELeavesEstimation::Gradient>\(\s*leafDers\[leaf\],\s*l2Regularizer,\s*sumAllWeights,\s*allDocCount\);\n)}{$1            /* cb_instr_leafdelta */ if (std::getenv("CB_INSTRUMENT_LOG")) { CbInstrumentLog(std::string(R"J({"event":"leaf_delta","method":"Gradient","leaf":)J") + std::to_string(leaf) + R"J(,"sum_der":)J" + CbFmt17(leafDers[leaf].SumDer) + R"J(,"sum_der2":)J" + CbFmt17(leafDers[leaf].SumDer2) + R"J(,"sum_weights":)J" + CbFmt17(leafDers[leaf].SumWeights) + R"J(,"sum_all_weights":)J" + CbFmt17(sumAllWeights) + R"J(,"all_doc_count":)J" + std::to_string(allDocCount) + R"J(,"l2":)J" + CbFmt17((double)l2Regularizer) + R"J(,"delta":)J" + CbFmt17((*leafDeltas)[leaf]) + "}"); }\n}s;
    ' "${AC}" || step "  (gradient leaf-delta hook not matched; recorded)"
    # Pairwise branch: log the resulting per-leaf delta after CalculatePairwiseLeafValues.
    perl -0777 -pi -e '
        s{(\*leafDeltas = CalculatePairwiseLeafValues\(\s*pairwiseWeightSums,\s*derSums,\s*l2Regularizer,\s*pairwiseNonDiagReg\);\n)}{$1        /* cb_instr_leafdelta */ if (std::getenv("CB_INSTRUMENT_LOG")) { for (int cbL = 0; cbL < leafCount; ++cbL) { CbInstrumentLog(std::string(R"J({"event":"leaf_delta","method":"Pairwise","leaf":)J") + std::to_string(cbL) + R"J(,"sum_der":)J" + CbFmt17(leafDers[cbL].SumDer) + R"J(,"sum_der2":)J" + CbFmt17(leafDers[cbL].SumDer2) + R"J(,"l2":)J" + CbFmt17((double)l2Regularizer) + R"J(,"pairwise_non_diag_reg":)J" + CbFmt17((double)pairwiseNonDiagReg) + R"J(,"delta":)J" + CbFmt17((*leafDeltas)[cbL]) + "}"); } }\n}s;
    ' "${AC}" || step "  (pairwise leaf-delta hook not matched; recorded)"
    step "  patched per-leaf delta + denom hook into approx_calcer.cpp (CalcLeafDeltasSimple)"
fi

# 3c. YetiRank RNG-draw events (GenerateYetiRankPairsForQuery).
ensure_sink "${YR}"
if ! grep -q 'cb_instr_yetirng' "${YR}"; then
    perl -0777 -pi -e '
        s{(const float uniformValue = rand\.GenRandReal1\(\);\n)}{$1                        /* cb_instr_yetirng */ if (std::getenv("CB_INSTRUMENT_LOG")) { CbInstrumentLog(std::string(R"J({"event":"yeti_gumbel","u":)J") + std::to_string(uniformValue) + "}"); }\n}s;
    ' "${YR}" || step "  (yetirank RNG hook not matched; recorded)"
    step "  patched YetiRank RNG-draw hook into yetirank_helpers.cpp"
fi

# 3d. StochasticRank noise stream (algo_helpers/error_functions.cpp), if present.
if [ -f "${EF}" ]; then
    ensure_sink "${EF}"
    if ! grep -q 'cb_instr_srank' "${EF}"; then
        # Hook the StochasticRank Gaussian noise draw site if the symbol exists.
        if grep -q 'StdNormalDistribution\|StochasticRank' "${EF}"; then
            perl -0777 -pi -e '
                s{(StdNormalDistribution<[^>]*>\([^;]*\);\n)}{$1            /* cb_instr_srank */ if (std::getenv("CB_INSTRUMENT_LOG")) { CbInstrumentLog(std::string(R"J({"event":"srank_noise","noise":)J") + std::to_string(noise[docId]) + "}"); }\n}s;
            ' "${EF}" || step "  (stochasticrank noise hook not matched; recorded)"
            step "  patched StochasticRank noise hook into error_functions.cpp"
        else
            step "  (error_functions.cpp present but no StochasticRank/StdNormalDistribution site — skipped)"
        fi
    fi
else
    step "  (algo_helpers/error_functions.cpp absent — StochasticRank noise hook skipped)"
fi

# --------------------------------------------------------------------------
# STEP 4 — build_native.py --targets _catboost (clang-18 prefix + venv 3.13)
# --------------------------------------------------------------------------
step "STEP 4: build_native.py --targets _catboost"
[ -x "${VENV_PY}" ] || fail "STEP 4 — project venv Python 3.13 missing at ${VENV_PY}"

PY_INCLUDE="$("${VENV_PY}" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
[ -n "${PY_INCLUDE}" ] || fail "STEP 4 — could not resolve venv Python include dir"
step "  Python3 include = ${PY_INCLUDE}"
step "  clang   = ${CLANG_BIN}"
step "  clang++ = ${CLANGXX_BIN}"

mkdir -p "${BUILD_ROOT}"
set +e
# NB: the catboost clang.toolchain forces bare clang/clang++; the prefix symlinks
# created in STEP 2 make those resolve to clang-18 on PATH. Pass the symlink paths
# as the cache entries so CMake's compiler check + the toolchain agree.
CC="${CLANG_BINDIR}/clang" CXX="${CLANG_BINDIR}/clang++" \
"${VENV_PY}" "${CB_SRC}/build/build_native.py" \
    --build-root-dir "${BUILD_ROOT}" \
    --build-type Release \
    --targets _catboost \
    --verbose \
    -DCMAKE_C_COMPILER="${CLANG_BINDIR}/clang" \
    -DCMAKE_CXX_COMPILER="${CLANG_BINDIR}/clang++" \
    -DPython3_INCLUDE_DIR="${PY_INCLUDE}" \
    -DPython3_EXECUTABLE="${VENV_PY}" \
    2>&1 | tee -a "${BUILD_LOG}"
BUILD_RC=${PIPESTATUS[0]}
set -e

if [ "${BUILD_RC}" -ne 0 ]; then
    fail "STEP 4 build_native.py --targets _catboost (rc=${BUILD_RC}) — see ${BUILD_LOG}"
fi

# --------------------------------------------------------------------------
# STEP 5 — Locate built _catboost.so + drop into a venv-package copy
# --------------------------------------------------------------------------
step "STEP 5: locate + stage built _catboost.so"
STAGE_PKG="${BUILD_ROOT}/instr_pkg"
# Find the FRESHLY-built shared object. The canonical ninja output is
# catboost/python-package/catboost/lib_catboost.so; PREFER it. Critically, EXCLUDE
# the staged package dir (${STAGE_PKG}) from the search — a prior run leaves a
# `_catboost.so` there, and a bare `find ... | head -1` would pick that STALE copy
# over the just-built lib_catboost.so (06.3-13 staging bug: the stale 276 MB
# self-copy silently shipped an un-instrumented trainer).
BUILT_SO="$(find "${BUILD_ROOT}" -path "${STAGE_PKG}" -prune -o -name 'lib_catboost.so' -print 2>/dev/null | head -1)"
if [ -z "${BUILT_SO}" ]; then
    BUILT_SO="$(find "${BUILD_ROOT}" -path "${STAGE_PKG}" -prune -o -name '_catboost.so' -print 2>/dev/null | head -1)"
fi
if [ -z "${BUILT_SO}" ]; then
    fail "STEP 5 — no lib_catboost.so / _catboost.so found under ${BUILD_ROOT} (excluding ${STAGE_PKG}) despite rc=0"
fi
step "  built artifact: ${BUILT_SO}"

SRC_PKG="${REPO_ROOT}/.venv/lib/python3.13/site-packages/catboost"
if [ -d "${SRC_PKG}" ]; then
    rm -rf "${STAGE_PKG}"
    mkdir -p "${STAGE_PKG}"
    cp -r "${SRC_PKG}" "${STAGE_PKG}/catboost"
    cp "${BUILT_SO}" "${STAGE_PKG}/catboost/_catboost.so"
    step "  staged instrumented package: ${STAGE_PKG}"
    step "  RUN-ONCE: CB_INSTRUMENT_LOG=/tmp/instr_smoke.jsonl PYTHONPATH=${STAGE_PKG} ${VENV_PY} <fit_script.py>"
else
    step "  (venv catboost package not found at ${SRC_PKG}; built .so left at ${BUILT_SO})"
fi

step "DONE — instrumented _catboost trainer built. Artifact: ${BUILT_SO}"
echo "INSTR_BUILT_SO=${BUILT_SO}"
echo "INSTR_STAGE_PKG=${STAGE_PKG:-<none>}"
echo "INSTR_CLANG_PREFIX=${CLANG_PREFIX}"
