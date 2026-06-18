# Text & embedding pipeline instrumentation (Plan 06.5-01, D-07)

**Authorization.** Phase 6.5 CONTEXT.md **D-07** explicitly sanctions reusing the
persistent instrumented catboost 1.2.10 trainer for the *thin internals* that are
not cleanly Python-reachable: the tokenizer token stream, the dictionary token→id
assignment, the per-document `TText` `(tokenId,count)` list, the per-calcer
intermediate encodings, the online-estimation visiting order, the LDA projection
matrix, and the KNN per-query neighbor ids. **OFFLINE / RUN-ONCE; no fabricated
fixtures, no weakened tolerance, no `#[ignore]` — escalate-don't-weaken** (the
06.3 discipline). The vendored `catboost-master/` instrumentation patches stay
**UNCOMMITTED** (D-09 / D-12); only the driver script, this README, the reference
`instrument_text_pipeline.cpp`, the fixture generator, and the frozen fixtures are
committed.

## The 7 env-gated hooks (CB_INSTRUMENT_LOG-gated, no-op when unset)

| # | Event | File (vendored, UNCOMMITTED) | Symbol / insertion point | Captures |
|---|-------|------------------------------|--------------------------|----------|
| a | `token_stream` | `catboost/private/libs/text_processing/tokenizer.cpp` | `TTokenizer::Tokenize` (function end) | post-split / post-lowercase token list per text |
| b | `dict_ids` | `library/cpp/text_processing/dictionary/dictionary_builder.cpp` | `TUnigramDictionaryBuilderImpl::FinishBuilding` (after the sorted id-assign loop) | `(token-string, id, count)` table per dictionary after filter/sort/truncate |
| c | `ttext` | `catboost/private/libs/data_types/text.h` | `TText(TVector<ui32>&&)` ctor end | per-document `(tokenId, count)` list, sorted asc by tokenId |
| d | `calcer_encoding` | `catboost/private/libs/feature_estimator/base_text_feature_estimator.h` | `Compute(...)` (after `featureCalcer.Compute`) | per-document per-calcer feature row (column-major read-back) |
| e | `online_order` | `catboost/private/libs/feature_estimator/base_text_feature_estimator.h` | `ComputeOnlineFeatures(...)` (before the learn-perm loop) | learn-permutation visiting order (D-03 leakage control) |
| f | `lda_projection` | `catboost/private/libs/embedding_features/lda.cpp` | `CalculateProjection(...)` (after trailing-rows copy) | LDA projection matrix + eigenvalues (Pitfall 1, f32 LAPACK) |
| f2 | `lda_scatter` | `catboost/private/libs/embedding_features/lda.cpp` | `CalculateProjection(...)` (BEFORE `ssygst_`) | regularized betweenMatrix B (`scatter_inner`) + totalScatter A (`scatter_total`) — the EXACT generalized-eigenproblem inputs, captured before `ssygst_` mutates them (06.5-05 re-measure) |
| f3 | `lda_project` | `catboost/private/libs/embedding_features/lda.cpp` | `TLinearDACalcer::Compute(...)` (before `ForEachActiveFeature`) | per-document projected feature row (`cblas_sgemv` projection + per-class likelihoods) (06.5-05 re-measure) |
| g | `knn_neighbors` | `catboost/private/libs/embedding_features/knn.cpp` | `TKNNCalcer::Compute(...)` (after `GetNearestNeighbors`) | per-query HNSW neighbor id list (Pitfall 2) |

The verbatim hook bodies and the JSONL schema are documented in
[`instrument_text_pipeline.cpp`](instrument_text_pipeline.cpp). The exact in-place
patch logic lives in `build_instrumented_trainer.sh` **STEP 3bis**.

## JSONL schema (one object per line, keyed by `event`)

```json
{"event":"token_stream","tokens":["good","great","movie"]}
{"event":"dict_ids","gram_order":1,"entries":[{"token":"bad","id":0,"count":10}]}
{"event":"ttext","pairs":[[1,1],[18,1]]}
{"event":"online_order","perm":[0,2,4]}
{"event":"calcer_encoding","doc":3,"values":[0.0,1.0]}
{"event":"lda_projection","dim":4,"matrix":[],"eigenvalues":[]}
{"event":"lda_scatter","dim":4,"scatter_inner":[],"scatter_total":[]}
{"event":"lda_project","proj_dim":1,"total_dim":4,"values":[]}
{"event":"knn_neighbors","k":5,"neighbors":[1,3,0,2,4]}
```

All floats are emitted at 17 significant digits (`%.17g`) for full IEEE-754
round-trip (the `≤1e-5` oracle needs more than `std::to_string`'s 6 dp).

## Sinks

- `.cpp` targets (tokenizer / dictionary_builder / lda / knn) reuse the 06.3-10
  `CbInstrumentLog` + `CbFmt17` sink (build script STEP 3).
- **Header** targets (`text.h`, `base_text_feature_estimator.h`) carry a
  self-contained inline sink (`CbInstr065Log` / `CbInstr065Fmt17`) wrapped in an
  `#ifndef CB_INSTR065_SINK_DEFINED` include guard. `base_text_feature_estimator.h`
  transitively includes `text.h` in the **same TU**, so without the guard the two
  anonymous-namespace definitions collide (`redefinition of CbInstr065Log` — a real
  build failure hit and fixed during 06.5-01).

## RUN-ONCE recipe

```sh
# 1. (re)build the instrumented trainer — sudo-free, idempotent. Re-provisions
#    clang-18/lld-18 into /tmp/clang18_prefix and the build tree into
#    /tmp/cb_build313 if /tmp was cleared; otherwise incremental.
bash crates/cb-oracle/generator/build_instrumented_trainer.sh
#    -> stages /tmp/cb_build313/instr_pkg/catboost/_catboost.so (≈39.7 MB)

# 2. capture the D-01 tokenizer/dictionary corpus + freeze all 5 calcer fixtures:
.venv/bin/python crates/cb-oracle/generator/gen_text_embedding_fixtures.py \
    --all --instr-pkg /tmp/cb_build313/instr_pkg
```

The generator fits TWO instrumented models (BoW + NaiveBayes) over the frozen
corpus so the captured corpus covers **all 7** hook categories (BoW is
target-independent and does not exercise the online path; NaiveBayes does). It
asserts every one of the 7 categories is non-empty and **aborts (never fabricates)**
if any is missing.

## Build environment notes (2026-06-18, Plan 06.5-01)

- `/tmp` was **cleared** (`/tmp/cb_build313` and `/tmp/clang18_prefix` both absent),
  so the driver did a full sudo-free re-provision + Release build (disk: 68 GB free
  on `/`, well above the 25 GB link-safety floor).
- clang-18 / lld-18 / llvm-18 (noble) via `apt-get download` + `dpkg -x`;
  conan / ninja / cython via `uv tool`; `build_native.py --targets _catboost`
  against the `.venv` Python 3.13.
- The instrumented `_catboost.so` lives ONLY under `/tmp` (uncommitted); re-create
  it by re-running the driver.

## Build bugs hit & fixed during 06.5-01 (recorded for re-runs)

1. **token_stream hook** — the original `perl -0777` insert (i) placed the hook
   AFTER the function's closing brace (file scope → "expected unqualified-id") and
   (ii) mangled the `\"`-escaped JSON quote into a bare `"""`. Fixed: insert inside
   the function before the closing brace, and use raw-string-literal quotes
   `R"J(")J"` instead of `\"`. The driver's pattern was corrected to a robust
   single-line anchor.
2. **header sink redefinition** — `base_text_feature_estimator.h` includes `text.h`,
   so both sinks landed in one TU. Fixed with the `#ifndef CB_INSTR065_SINK_DEFINED`
   include guard.
3. **online_order / calcer_encoding** — the multi-line `perl -0777` slurp patterns
   were whitespace-fragile and silently did not substitute. Fixed: deterministic
   single-line-anchor inserts (and an `awk` fallback for calcer_encoding).
4. **lda final-link skip (06.5-05)** — after editing `lda.cpp`, the incremental
   `ninja _catboost` recompiled `lda.cpp.o` and re-archived the `.global.a` but did
   **not** relink `lib_catboost.so`: in this build the `*.global.a` whole-archive
   libs are tracked as *order-only* deps (`|`) of the `.so` link edge, so a changed
   `.o` does not trigger the relink. Fix: `rm -f catboost/python-package/catboost/lib_catboost.so`
   then `ninja catboost/python-package/catboost/lib_catboost.so` to force the link,
   and re-stage into `instr_pkg/catboost/_catboost.so`. (Without this the smoke fit
   silently runs the STALE trainer — the new `lda_scatter`/`lda_project` events never
   fire.)

## LDA re-instrument-and-re-measure (06.5-05)

The Plan-01 `lda_projection` hook alone could not localize the hand-roll-f32
divergence, so 06.5-05 added the `lda_scatter` (eigenproblem inputs) and
`lda_project` (per-document projection) hooks and re-measured. **Freeze the GT** with:

```sh
.venv/bin/python crates/cb-oracle/generator/gen_text_embedding_fixtures.py \
    --lda-scatter-gt --instr-pkg /tmp/cb_build313/instr_pkg
#   -> fixtures/embedding_calcers/LDA/scatter_projection_gt.json
```

**Re-measurement result (definitive):**

| What | Hand-roll f32 | f64 scipy | f32 reference LAPACK (`ssygv`) | Upstream dump |
|------|---------------|-----------|-------------------------------|---------------|
| `scatter_inner` (B) vs upstream | **≤4.66e-10** | — | — | (reference) |
| `scatter_total` (A) vs upstream | **0** | — | — | (reference) |
| dominant eigenvalue | 38.45 | 38.45 | 38.45 | **376.67** |
| dominant eigenvector | `[0.597,0.563,-0.439,-0.365]` | same | same | `[0.637,0.514,-0.455,-0.351]` |
| projection max-abs err vs upstream | 4.9e-2 | 4.9e-2 | 4.9e-2 | (reference) |
| per-document projected-feature max-abs err vs upstream | **3.9e-2** | — | — | (reference) |

**Localization:** the scatter construction is bit-faithful (B≤4.66e-10, A=0). The
divergence is entirely in the eigensolve. The hand-roll matches **both** f64 scipy
**and** f32 reference LAPACK `ssygv`. The upstream vendored-CLAPACK `ssyev_` dump
reports a dominant eigenvalue (376.67) that is **inconsistent with its own
eigenvector** (the eigenvector's Rayleigh quotient on the reduced matrix is 38.28,
and `||A_reduced·v − 376.67·v|| = 338`), and the eigenvector is 4.9e-2 away from the
reference dominant eigenvector. The per-flush eigenvalue ratio is non-constant
(20.0, 10.1, 9.6, 9.8), ruling out a fixed rescale. **Conclusion:** exact f32
reproduction of upstream's specific CLAPACK `ssyev_` iterate is not achievable by a
reference-faithful eigensolver — this is a documented-tolerance escalation candidate
(see `06.5-05-SUMMARY.md`).
