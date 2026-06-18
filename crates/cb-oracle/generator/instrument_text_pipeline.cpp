// ============================================================================
// instrument_text_pipeline.cpp  (Plan 06.5-01, Task 1)
//
// REFERENCE / SOURCE-OF-TRUTH for the 7 env-gated CB_INSTRUMENT_LOG hooks that
// `build_instrumented_trainer.sh` (STEP 3bis) injects into the vendored
// catboost-master/ text & embedding pipeline. This file is NOT compiled or linked
// on its own — the hooks are applied as in-place patches into the upstream C++
// translation units (so they can read each pipeline stage's locals). This TU
// exists to (a) document each hook's exact body and insertion point and (b) give
// a single auditable place where the JSONL schema is specified.
//
// DESIGN INVARIANTS (D-07 / D-09 / D-12, escalate-don't-weaken):
//   * OFFLINE / RUN-ONCE only.  NEVER invoked in CI.
//   * Every hook is a strict NO-OP when CB_INSTRUMENT_LOG is unset
//     (each is wrapped in `if (std::getenv("CB_INSTRUMENT_LOG")) { ... }`).
//   * The vendored catboost-master/ source patches stay UNCOMMITTED (the whole
//     catboost-master/ tree is untracked); ONLY this reference file, the driver
//     script, the README, and the frozen fixtures are committed.
//   * Floats are emitted at 17 significant digits (%.17g) so the <= 1e-5 oracle
//     has full IEEE-754 round-trip precision (std::to_string truncates to 6dp).
//
// SINKS
//   .cpp targets (tokenizer / dictionary_builder / lda / knn) reuse the existing
//   06.3-10 `CbInstrumentLog` + `CbFmt17` sink helpers (build script STEP 3).
//   HEADER targets (text.h, base_text_feature_estimator.h) cannot rely on that
//   TU helper being in scope, so they carry a self-contained inline sink
//   (`CbInstr065Log` / `CbInstr065Fmt17`) wrapped in an
//   `#ifndef CB_INSTR065_SINK_DEFINED` include guard — base_text_feature_estimator.h
//   transitively includes text.h in the SAME TU, so without the guard the two
//   anonymous-namespace definitions collide.
//
// JSONL SCHEMA (one object per line, keyed by "event"):
//   {"event":"token_stream","tokens":["good","great","movie"]}
//   {"event":"dict_ids","gram_order":1,"entries":[{"token":"bad","id":0,"count":10}, ...]}
//   {"event":"ttext","pairs":[[1,1],[18,1]]}                      // [tokenId,count], sorted asc
//   {"event":"online_order","perm":[0,2,4,...]}                   // learn permutation visiting order
//   {"event":"calcer_encoding","doc":3,"values":[0.0, 1.0, ...]}  // per-document calcer feature row
//   {"event":"lda_projection","dim":4,"matrix":[...],"eigenvalues":[...]}
//   {"event":"knn_neighbors","k":5,"neighbors":[1,3,0,2,4]}       // per-query neighbor ids
//
// HOOK INSERTION POINTS (file : symbol : what is captured)
// ----------------------------------------------------------------------------
// (a) token_stream
//     catboost/private/libs/text_processing/tokenizer.cpp
//     NCB::TTokenizer::Tokenize(...)  — at function end, tokens->View holds the
//     final post-split / post-lowercase token sequence (both NeedToModifyTokens
//     branches). The D-11 load-bearing surface (SC-1).
//
// (b) dict_ids
//     library/cpp/text_processing/dictionary/dictionary_builder.cpp
//     TUnigramDictionaryBuilderImpl::FinishBuilding() — right after the sorted
//     (count DESC, token ASC) token->id assignment loop; captures the full
//     (token-string, id, count) table per dictionary after filter/sort/truncate.
//
// (c) ttext
//     catboost/private/libs/data_types/text.h
//     TText(TVector<ui32>&&) ctor — after the sorted run-length collapse; captures
//     the per-document (tokenId, count) list, sorted ascending by tokenId.
//
// (d) calcer_encoding
//     catboost/private/libs/feature_estimator/base_text_feature_estimator.h
//     Compute(...) — after featureCalcer.Compute writes the doc's feature row;
//     reads back the column-major row (features[f*docCount + docId]) per document.
//
// (e) online_order
//     catboost/private/libs/feature_estimator/base_text_feature_estimator.h
//     ComputeOnlineFeatures(...) — before the `for (ui64 line : learnPermutation)`
//     loop; captures the exact learn-permutation visiting order (D-03 leakage).
//
// (f) lda_projection
//     catboost/private/libs/embedding_features/lda.cpp
//     CalculateProjection(...) — after the trailing-rows copy into
//     projectionMatrix; captures the LDA projection matrix + eigenvalues
//     (Pitfall 1 — the f32 LAPACK ssygst_/ssyev_ result).
//
// (g) knn_neighbors
//     catboost/private/libs/embedding_features/knn.cpp
//     TKNNCalcer::Compute(...) — right after Cloud->GetNearestNeighbors; captures
//     the per-query HNSW neighbor id list (Pitfall 2).
//
// The verbatim hook bodies below mirror exactly what STEP 3bis injects. They are
// shown as free functions purely for documentation; in the real patch they are
// inlined `if (std::getenv(...)) { ... }` blocks at the insertion points above.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace cb_instr_text_pipeline_reference {

    // Mirrors the 06.3-10 sink (CbInstrumentLog) used by the .cpp targets.
    inline void CbInstrumentLog(const std::string& line) {
        const char* path = std::getenv("CB_INSTRUMENT_LOG");
        if (path == nullptr) { return; }
        std::FILE* f = std::fopen(path, "a");
        if (f != nullptr) { std::fputs(line.c_str(), f); std::fputc(10, f); std::fclose(f); }
    }
    inline std::string CbFmt17(double v) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.17g", v);
        return std::string(buf);
    }

    // JSON-escape a token string for the "tokens"/"token" fields. Mirrors the
    // inline escape in the token_stream / dict_ids hooks (escape '\\' (92) and
    // '"' (34) only — token text is otherwise printable).
    inline std::string EscapeToken(const std::string& cbT) {
        std::string cbEsc;
        for (char cbC : cbT) {
            if (cbC == 92 || cbC == 34) { cbEsc += static_cast<char>(92); }
            cbEsc += cbC;
        }
        return cbEsc;
    }

    // (a) token_stream — emitted from tokenizer.cpp::Tokenize.
    inline void EmitTokenStream(const std::vector<std::string>& view) {
        if (!std::getenv("CB_INSTRUMENT_LOG")) { return; }
        std::string cbLine = R"J({"event":"token_stream","tokens":[)J";
        for (size_t cbI = 0; cbI < view.size(); ++cbI) {
            if (cbI) { cbLine += ","; }
            cbLine += std::string(R"J(")J") + EscapeToken(view[cbI]) + R"J(")J";
        }
        cbLine += "]}";
        CbInstrumentLog(cbLine);
    }

    // (g) knn_neighbors — emitted from knn.cpp::Compute.
    inline void EmitKnnNeighbors(long long k, const std::vector<long long>& neighbors) {
        if (!std::getenv("CB_INSTRUMENT_LOG")) { return; }
        std::string cbLine = std::string(R"J({"event":"knn_neighbors","k":)J") + std::to_string(k) + R"J(,"neighbors":[)J";
        for (size_t cbI = 0; cbI < neighbors.size(); ++cbI) {
            if (cbI) { cbLine += ","; }
            cbLine += std::to_string(neighbors[cbI]);
        }
        cbLine += "]}";
        CbInstrumentLog(cbLine);
    }

}  // namespace cb_instr_text_pipeline_reference
