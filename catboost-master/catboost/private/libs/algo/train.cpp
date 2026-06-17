#include "train.h"

#include "approx_calcer.h"
#include "approx_calcer_helpers.h"
#include "approx_updater_helpers.h"
#include "build_subset_in_leaf.h"
#include "fold.h"
#include "greedy_tensor_search.h"
#include "index_calcer.h"
#include "learn_context.h"
#include "monotonic_constraint_utils.h"
#include "online_ctr.h"
#include "tensor_search_helpers.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/private/libs/algo/approx_calcer/leafwise_approx_calcer.h>
#include <catboost/private/libs/algo_helpers/approx_calcer_helpers.h>
#include <catboost/private/libs/algo_helpers/error_functions.h>
#include <catboost/private/libs/distributed/master.h>
#include <catboost/private/libs/distributed/worker.h>

// === 05-17 LIVE-TRAINER INSTRUMENTATION (user-approved, 2026-06-15 CONTEXT
// decision revision; OFFLINE RUN-ONCE/COMMIT, gated by env CB_INSTRUMENT_LOG so
// it is fully inert in normal builds). Logs, per boosting iteration: the
// persistent RNG GetCallCount(), the selected structure fold index
// (train.cpp:208 Folds[Rand.GenRand() % foldCount]), and the resulting
// AveragingFold leaf partition + leaf values feeding CalcLeafValues
// (approx_calcer.cpp:1082). Ground truth for porting the structure-fold cycling
// rule into crates/cb-train/src/boosting.rs. NEVER wired into CI. ===
#include <util/string/builder.h>

#include <cstdlib>
#include <cstdio>
#include <mutex>


TErrorTracker BuildErrorTracker(
    EMetricBestValue bestValueType,
    double bestPossibleValue,
    bool hasTest,
    const TLearnContext& ctx
) {
    const auto& odOptions = ctx.Params.BoostingOptions->OverfittingDetector;
    return CreateErrorTracker(odOptions, bestPossibleValue, bestValueType, hasTest);
}

static void UpdateLearningFold(
    const NCB::TTrainingDataProviders& data,
    const IDerCalcer& error,
    const std::variant<TSplitTree, TNonSymmetricTreeStructure>& bestTree,
    ui64 randomSeed,
    TFold* fold,
    TLearnContext* ctx
) {
    TVector<TVector<TVector<double>>> approxDelta;

    CalcApproxForLeafStruct(
        data,
        error,
        *fold,
        bestTree,
        randomSeed,
        ctx,
        &approxDelta
    );

    if (error.GetIsExpApprox()) {
        UpdateBodyTailApprox</*StoreExpApprox*/true>(
            approxDelta,
            ctx->Params.BoostingOptions->LearningRate,
            ctx->LocalExecutor,
            fold
        );
    } else {
        UpdateBodyTailApprox</*StoreExpApprox*/false>(
            approxDelta,
            ctx->Params.BoostingOptions->LearningRate,
            ctx->LocalExecutor,
            fold
        );
    }
}

static void ScaleAllApproxes(
    const double approxMultiplier,
    const bool storeExpApprox,
    TLearnProgress* learnProgress,
    NPar::ILocalExecutor* localExecutor
) {
    TVector<TVector<TVector<double>>*> allApproxes;
    for (auto& fold : learnProgress->Folds) {
        for (auto &bodyTail : fold.BodyTailArr) {
            allApproxes.push_back(&bodyTail.Approx);
        }
    }
    allApproxes.push_back(&learnProgress->AveragingFold.BodyTailArr[0].Approx);
    const int learnApproxesCount = SafeIntegerCast<int>(allApproxes.size());
    allApproxes.push_back(&learnProgress->AvrgApprox);
    for (auto& testApprox : learnProgress->TestApprox) {
        allApproxes.push_back(&testApprox);
    }

    NPar::ParallelFor(
        *localExecutor,
        0,
        allApproxes.size(),
        [approxMultiplier, storeExpApprox, learnApproxesCount, localExecutor, &allApproxes](int index) {
            const bool isLearnApprox = (index < learnApproxesCount);
            if (isLearnApprox && storeExpApprox) {
                UpdateApprox(
                    [approxMultiplier](TConstArrayRef<double> /* delta */, TArrayRef<double> approx, size_t idx) {
                        approx[idx] = ApplyLearningRate<true>(approx[idx], approxMultiplier);
                    },
                    *allApproxes[index], // stub deltas
                    allApproxes[index],
                    localExecutor
                );
            } else {
                UpdateApprox(
                    [approxMultiplier](TConstArrayRef<double> /* delta */, TArrayRef<double> approx, size_t idx) {
                        approx[idx] = ApplyLearningRate<false>(approx[idx], approxMultiplier);
                    },
                    *allApproxes[index], // stub deltas
                    allApproxes[index],
                    localExecutor
                );
            }
        }
    );
}

static void CalcApproxesLeafwise(
    const NCB::TTrainingDataProviders& data,
    const IDerCalcer& error,
    const std::variant<TSplitTree, TNonSymmetricTreeStructure>& tree,
    TLearnContext* ctx,
    TVector<TVector<double>>* treeValues,
    TVector<TIndexType>* indices
) {
    *indices = BuildIndices(
        ctx->LearnProgress->AveragingFold,
        tree,
        data,
        EBuildIndicesDataParts::All,
        ctx->LocalExecutor
    );
    auto statistics = BuildSubset(
        *indices,
        GetLeafCount(tree),
        ctx
    );

    TVector<TDers> weightedDers;
    const int approxDimension = ctx->LearnProgress->AveragingFold.GetApproxDimension();
    if (approxDimension == 1) {
        const int scratchSize = APPROX_BLOCK_SIZE * CB_THREAD_LIMIT;
        weightedDers.yresize(scratchSize);
    }
    for (int leafIdx = 0; leafIdx < GetLeafCount(tree); ++leafIdx) {
        CalcLeafValues(
            error,
            &(statistics[leafIdx]),
            ctx,
            weightedDers
        );
    }
    AssignLeafValues(
        statistics,
        treeValues
    );

    // cycle for accordance with non leawfise approxes
    if (ctx->LearnProgress->AveragingFold.BodyTailArr[0].Approx.size() < 2
        && ctx->Params.ObliviousTreeOptions->LeavesEstimationMethod != ELeavesEstimation::Exact
    ) {
        ctx->LearnProgress->Rand.Advance(ctx->Params.ObliviousTreeOptions->LeavesEstimationIterations);
    }

}

// 05-17 instrumentation sink: appends a line to $CB_INSTRUMENT_LOG (offline only).
static void CbInstrumentLog(const TString& line) {
    static std::mutex mtx;
    const char* path = std::getenv("CB_INSTRUMENT_LOG");
    if (path == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> g(mtx);
    FILE* f = std::fopen(path, "a");
    if (f != nullptr) {
        std::fputs(line.c_str(), f);
        std::fputc('\n', f);
        std::fclose(f);
    }
}

// 05-18 (Spike-001 resolution) instrumentation: dump an ATOMIC self-consistent
// (projection, effective LearnPermutationFeaturesSubset, LearnTargetClass, ui8 CTR bins)
// tuple for ONE fold's OnlineCtr split, captured from the SAME fold state whose
// ComputeOnlineCTRs produced the bins. The prior 05-17 capture logged the AveragingFold's
// LearnPermutation (line fold.h:202) while ComputeOnlineCTRs actually consumes
// LearnPermutationFeaturesSubset (the Compose() at fold.cpp:62), and never logged
// LearnTargetClass — so (perm, bins) were mutually inconsistent (Spike 001 VERDICT).
// This logs all three together so the offline replay is correct-by-construction.
static void CbLogSelfConsistentCtr(
    size_t iter,
    const char* foldTag,
    int foldIdx,
    int splitIdx,
    const TFold& fold,
    const TSplit& sp
) {
    if (std::getenv("CB_INSTRUMENT_LOG") == nullptr) {
        return;
    }
    if (sp.Type != ESplitType::OnlineCtr) {
        return;
    }
    TStringBuilder sb;
    sb << "{\"event\":\"self_consistent_ctr\",\"iter\":" << iter
       << ",\"fold\":\"" << foldTag << "\",\"fold_idx\":" << foldIdx
       << ",\"split\":" << splitIdx
       << ",\"bin_border\":" << sp.BinBorder
       << ",\"ctr_idx\":" << static_cast<int>(sp.Ctr.CtrIdx)
       << ",\"target_border_idx\":" << static_cast<int>(sp.Ctr.TargetBorderIdx)
       << ",\"prior_idx\":" << static_cast<int>(sp.Ctr.PriorIdx)
       << ",\"proj_cat_features\":[";
    for (size_t i = 0; i < sp.Ctr.Projection.CatFeatures.size(); ++i) {
        if (i > 0) { sb << ","; }
        sb << sp.Ctr.Projection.CatFeatures[i];
    }
    sb << "],\"proj_bin_features\":" << sp.Ctr.Projection.BinFeatures.size()
       << ",\"proj_onehot_features\":" << sp.Ctr.Projection.OneHotFeatures.size();
    // position -> original object id, as ComputeOnlineCTRs consumes it (fold.cpp:62 Compose).
    TVector<ui32> permSubset(fold.GetLearnSampleCount());
    fold.LearnPermutationFeaturesSubset.ForEach(
        [&] (ui32 idx, ui32 srcIdx) { permSubset[idx] = srcIdx; }
    );
    sb << ",\"perm_features_subset\":[";
    for (size_t i = 0; i < permSubset.size(); ++i) {
        if (i > 0) { sb << ","; }
        sb << permSubset[i];
    }
    sb << "],\"learn_permutation\":[";
    {
        TConstArrayRef<ui32> lp = fold.GetLearnPermutationArray();
        for (size_t i = 0; i < lp.size(); ++i) {
            if (i > 0) { sb << ","; }
            sb << lp[i];
        }
    }
    sb << "],\"learn_target_class\":[";
    if (!fold.LearnTargetClass.empty()) {
        const TVector<int>& ltc = fold.LearnTargetClass[0];
        for (size_t i = 0; i < ltc.size(); ++i) {
            if (i > 0) { sb << ","; }
            sb << ltc[i];
        }
    }
    sb << "],\"bins\":[";
    {
        const TOnlineCtrBase& oc = fold.GetCtrs(sp.Ctr.Projection);
        TConstArrayRef<ui8> d = oc.GetData(sp.Ctr, 0);
        for (size_t i = 0; i < d.size(); ++i) {
            if (i > 0) { sb << ","; }
            sb << static_cast<int>(d[i]);
        }
    }
    sb << "]}";
    CbInstrumentLog(sb);
}

void TrainOneIteration(const NCB::TTrainingDataProviders& data, TLearnContext* ctx) {
    const auto error = BuildError(ctx->Params, ctx->ObjectiveDescriptor);
    ctx->LearnProgress->HessianType = error->GetHessianType();
    TProfileInfo& profile = ctx->Profile;

    const size_t iterationIndex = ctx->LearnProgress->TreeStruct.size();
    const int foldCount = ctx->LearnProgress->Folds.ysize();
    const double modelLength
        = double(iterationIndex) * ctx->Params.BoostingOptions->LearningRate;

    // 06.3-17 instrumentation: per-tree persistent-RNG call-count fence at the
    // START of TrainOneIteration. Combined with the per-phase fences below and the
    // tree_rng_end fence, this localizes the per-tree pairwise draw count that the
    // Rust YetiRankTreeSeeder must reproduce for trees 2+. Env-gated (no-op when
    // CB_INSTRUMENT_LOG is unset). RUN-ONCE/COMMIT (D-08/D-11).
    CbInstrumentLog(TStringBuilder()
        << "{\"event\":\"tree_rng_start\",\"iter\":" << iterationIndex
        << ",\"fold_count\":" << foldCount
        << ",\"cc\":" << ctx->LearnProgress->Rand.GetCallCount() << "}");

    CheckInterrupted(); // check after long-lasting operation

    const double modelShrinkRate = ctx->Params.BoostingOptions->ModelShrinkRate.Get();
    if (modelShrinkRate > 0) {
        if (iterationIndex > 0) {
            const double modelShrinkage =
                ctx->Params.BoostingOptions->ModelShrinkMode == EModelShrinkMode::Constant
                ? (1 - modelShrinkRate * ctx->Params.BoostingOptions->LearningRate)
                : (1 - modelShrinkRate / static_cast<double>(iterationIndex));
            ScaleAllApproxes(
                modelShrinkage,
                error->GetIsExpApprox(),
                ctx->LearnProgress.Get(),
                ctx->LocalExecutor
            );
            if (ctx->LearnProgress->StartingApprox.Defined()) {
                for (auto& approx : *ctx->LearnProgress->StartingApprox) {
                    approx = approx * modelShrinkage;
                }
            }
            ctx->LearnProgress->ModelShrinkHistory.push_back(modelShrinkage);
        } else {
            ctx->LearnProgress->ModelShrinkHistory.push_back(1.0);
        }
    }

    std::variant<TSplitTree, TNonSymmetricTreeStructure> bestTree;
    int gInstrTakenFoldIdx = -1; // 05-18: hoisted so the leaf-value block can re-access the structure fold
    {
        const ui64 callCountBeforeStructureDraw = ctx->LearnProgress->Rand.GetCallCount();
        const ui64 structureDrawRaw = ctx->LearnProgress->Rand.GenRand();
        const int takenFoldIdx = static_cast<int>(structureDrawRaw % static_cast<ui64>(foldCount));
        gInstrTakenFoldIdx = takenFoldIdx;
        TFold* takenFold = &ctx->LearnProgress->Folds[takenFoldIdx];
        CbInstrumentLog(TStringBuilder()
            << "{\"event\":\"structure_fold\",\"iter\":" << iterationIndex
            << ",\"fold_count\":" << foldCount
            << ",\"callcount_before\":" << callCountBeforeStructureDraw
            << ",\"draw_raw\":" << structureDrawRaw
            << ",\"taken_fold\":" << takenFoldIdx << "}");
        const TVector<ui64> randomSeeds = GenRandUI64Vector(
            takenFold->BodyTailArr.ysize(),
            ctx->LearnProgress->Rand.GenRand()
        );
        if (ctx->Params.SystemOptions->IsSingleHost()) {
            ctx->LocalExecutor->ExecRangeWithThrow(
                [&](int bodyTailId) {
                    CalcWeightedDerivatives(
                        *error,
                        bodyTailId,
                        ctx->Params,
                        randomSeeds[bodyTailId],
                        takenFold,
                        ctx->LocalExecutor
                    );
                },
                0,
                takenFold->BodyTailArr.ysize(),
                NPar::TLocalExecutor::WAIT_COMPLETE
            );
        } else {
            Y_ASSERT(takenFold->BodyTailArr.ysize() == 1);
            MapSetDerivatives(ctx);
        }
        profile.AddOperation("Calc derivatives");

        // 06.3-17: fence BEFORE GreedyTensorSearch (after structure draw + deriv
        // recalc randomSeeds). The delta tree_rng_pre_gts - tree_rng_start ==
        // structure-fold draw (1) + deriv-recalc GenRand (1) == 2 for single-host.
        CbInstrumentLog(TStringBuilder()
            << "{\"event\":\"tree_rng_pre_gts\",\"iter\":" << iterationIndex
            << ",\"cc\":" << ctx->LearnProgress->Rand.GetCallCount() << "}");
        GreedyTensorSearch(
            data,
            modelLength,
            profile,
            takenFold,
            ctx,
            &bestTree
        );
        // 06.3-17: fence AFTER GreedyTensorSearch. The delta
        // tree_rng_post_gts - tree_rng_pre_gts == the per-tree-level split-search
        // draw count on the PERSISTENT RNG — the exact pointwise-vs-pairwise
        // difference the Rust seeder calibrates.
        CbInstrumentLog(TStringBuilder()
            << "{\"event\":\"tree_rng_post_gts\",\"iter\":" << iterationIndex
            << ",\"cc\":" << ctx->LearnProgress->Rand.GetCallCount() << "}");
    }
    CheckInterrupted(); // check after long-lasting operation
    {
        TVector<TFold*> trainFolds;
        for (int foldId = 0; foldId < foldCount; ++foldId) {
            trainFolds.push_back(&ctx->LearnProgress->Folds[foldId]);
        }

        TrimOnlineCTRcache(trainFolds);
        TrimOnlineCTRcache({ &ctx->LearnProgress->AveragingFold });
        {
            TVector<TFold*> allFolds = trainFolds;
            allFolds.push_back(&ctx->LearnProgress->AveragingFold);

            struct TLocalJobData {
                const NCB::TTrainingDataProviders* data;
                TProjection Projection;
                TFold* Fold;
                TOwnedOnlineCtr* Ctr;

            public:
                void DoTask(TLearnContext* ctx) {
                    ComputeOnlineCTRs(*data, *Fold, Projection, ctx, Ctr);
                }
            };

            TVector<TLocalJobData> parallelJobsData;
            THashSet<TProjection> seenProjections;
            for (const auto& ctr : GetUsedCtrs(bestTree)) {
                const auto& proj = ctr.Projection;
                if (seenProjections.contains(proj)) {
                    continue;
                }
                for (auto* foldPtr : allFolds) {
                    auto* ownedCtrs = foldPtr->GetOwnedCtrs(proj);
                    if (ownedCtrs && ownedCtrs->Data[proj].Feature.empty()) {
                        parallelJobsData.emplace_back(
                            TLocalJobData{ &data, proj, foldPtr, ownedCtrs}
                        );
                    }
                }
                seenProjections.insert(proj);
            }

            ctx->LocalExecutor->ExecRange(
                [&](int taskId){
                    parallelJobsData[taskId].DoTask(ctx);
                },
                0,
                SafeIntegerCast<int>(parallelJobsData.size()),
                NPar::TLocalExecutor::WAIT_COMPLETE
            );
        }
        profile.AddOperation("ComputeOnlineCTRs for tree struct (train folds and test fold)");
        CheckInterrupted(); // check after long-lasting operation

        TVector<TVector<double>> treeValues; // [dim][leafId]
        TVector<double> sumLeafWeights; // [leafId]

        if (ctx->Params.SystemOptions->IsSingleHost()) {
            // 06.3-17: fence BEFORE the leaf-value-phase randomSeeds GenRand. The
            // delta tree_rng_pre_leaf - tree_rng_post_gts accounts for any draws
            // GreedyTensorSearch leaves consumed before the leaf phase begins.
            CbInstrumentLog(TStringBuilder()
                << "{\"event\":\"tree_rng_pre_leaf\",\"iter\":" << iterationIndex
                << ",\"cc\":" << ctx->LearnProgress->Rand.GetCallCount() << "}");
            const TVector<ui64> randomSeeds = GenRandUI64Vector(foldCount, ctx->LearnProgress->Rand.GenRand());

            TVector<TIndexType> indices;

            const bool treeHasMonotonicConstraints = !ctx->Params.ObliviousTreeOptions->MonotoneConstraints.GetUnchecked().empty();
            if (
                ctx->Params.ObliviousTreeOptions->DevLeafwiseApproxes.Get() &&
                ctx->Params.BoostingOptions->BoostingType.Get() == EBoostingType::Plain
                && !treeHasMonotonicConstraints
                && error->GetErrorType() == EErrorType::PerObjectError
            ) {
                CalcApproxesLeafwise(
                    data,
                    *error,
                    bestTree,
                    ctx,
                    &treeValues,
                    &indices
                );
            } else {
                CalcLeafValues(
                    data,
                    *error,
                    bestTree,
                    ctx,
                    &treeValues,
                    &indices
                );
            }

            ctx->Profile.AddOperation("CalcApprox result leaves");
            CheckInterrupted(); // check after long-lasting operation

            TConstArrayRef<ui32> learnPermutationRef = ctx->LearnProgress->AveragingFold.GetLearnPermutationArray();

            const size_t leafCount = treeValues[0].size();
            sumLeafWeights = SumLeafWeights(
                leafCount,
                indices,
                learnPermutationRef,
                GetWeights(*data.Learn->TargetData),
                ctx->LocalExecutor
            );
            if (std::getenv("CB_INSTRUMENT_LOG") != nullptr) {
                // Log tree split borders (BinBorder per split) for the chosen structure.
                TStringBuilder splitsSb;
                splitsSb << "{\"event\":\"tree_struct\",\"iter\":" << iterationIndex
                         << ",\"splits\":[";
                if (const TSplitTree* st = std::get_if<TSplitTree>(&bestTree)) {
                    for (int si = 0; si < st->Splits.ysize(); ++si) {
                        if (si > 0) { splitsSb << ","; }
                        splitsSb << "{\"bin_border\":" << st->Splits[si].BinBorder << "}";
                    }
                }
                splitsSb << "]}";
                CbInstrumentLog(splitsSb);
                // Log the AveragingFold leaf partition (sumLeafWeights = per-leaf weight)
                // and the pre-normalization leaf values.
                TStringBuilder partSb;
                partSb << "{\"event\":\"leaf_partition\",\"iter\":" << iterationIndex
                       << ",\"sum_leaf_weights\":[";
                for (size_t li = 0; li < sumLeafWeights.size(); ++li) {
                    if (li > 0) { partSb << ","; }
                    partSb << sumLeafWeights[li];
                }
                partSb << "],\"leaf_values_raw\":[";
                for (size_t di = 0; di < treeValues.size(); ++di) {
                    if (di > 0) { partSb << ","; }
                    partSb << "[";
                    for (size_t li = 0; li < treeValues[di].size(); ++li) {
                        if (li > 0) { partSb << ","; }
                        partSb << treeValues[di][li];
                    }
                    partSb << "]";
                }
                partSb << "]}";
                CbInstrumentLog(partSb);
                // Per-object AveragingFold leaf assignment (BuildIndices result),
                // in AveragingFold learn-permutation order, AND the averaging-fold
                // online CTR ui8 bin per object for the single Borders CTR.
                TStringBuilder idxSb;
                idxSb << "{\"event\":\"leaf_indices\",\"iter\":" << iterationIndex
                      << ",\"indices\":[";
                for (size_t ii = 0; ii < indices.size(); ++ii) {
                    if (ii > 0) { idxSb << ","; }
                    idxSb << static_cast<int>(indices[ii]);
                }
                idxSb << "],\"avg_perm\":[";
                {
                    TConstArrayRef<ui32> avgPerm =
                        ctx->LearnProgress->AveragingFold.GetLearnPermutationArray();
                    for (size_t ii = 0; ii < avgPerm.size(); ++ii) {
                        if (ii > 0) { idxSb << ","; }
                        idxSb << avgPerm[ii];
                    }
                }
                idxSb << "]}";
                CbInstrumentLog(idxSb);
                // Per-split AveragingFold online-CTR ui8 bin per object + the split
                // BinBorder, so the leaf assignment is fully reproducible offline.
                if (const TSplitTree* st = std::get_if<TSplitTree>(&bestTree)) {
                    const TFold& avgFold = ctx->LearnProgress->AveragingFold;
                    for (int si = 0; si < st->Splits.ysize(); ++si) {
                        const TSplit& sp = st->Splits[si];
                        TStringBuilder ctrSb;
                        ctrSb << "{\"event\":\"avg_ctr_bins\",\"iter\":" << iterationIndex
                              << ",\"split\":" << si
                              << ",\"split_type\":" << static_cast<int>(sp.Type)
                              << ",\"bin_border\":" << sp.BinBorder << ",\"bins\":[";
                        if (sp.Type == ESplitType::OnlineCtr) {
                            const TOnlineCtrBase& oc = avgFold.GetCtrs(sp.Ctr.Projection);
                            TConstArrayRef<ui8> data = oc.GetData(sp.Ctr, 0);
                            for (size_t di = 0; di < data.size(); ++di) {
                                if (di > 0) { ctrSb << ","; }
                                ctrSb << static_cast<int>(data[di]);
                            }
                        }
                        ctrSb << "]}";
                        CbInstrumentLog(ctrSb);
                    }
                    // 05-18 (Spike-001 resolution): ATOMIC self-consistent tuples — capture, for
                    // the SAME fold state whose ComputeOnlineCTRs produced the bins, the projection +
                    // effective LearnPermutationFeaturesSubset + LearnTargetClass + ui8 bins, for BOTH
                    // the fixed AveragingFold (leaf-value materialization) and the per-iteration
                    // structure (taken) fold (split-scoring materialization).
                    for (int si = 0; si < st->Splits.ysize(); ++si) {
                        CbLogSelfConsistentCtr(
                            iterationIndex, "averaging", -1, si, avgFold, st->Splits[si]);
                        if (gInstrTakenFoldIdx >= 0 && gInstrTakenFoldIdx < foldCount) {
                            CbLogSelfConsistentCtr(
                                iterationIndex,
                                "structure",
                                gInstrTakenFoldIdx,
                                si,
                                ctx->LearnProgress->Folds[gInstrTakenFoldIdx],
                                st->Splits[si]);
                        }
                    }
                }
            }
            const auto lossFunction = ctx->Params.LossFunctionDescription->GetLossFunction();
            const bool usePairs = UsesPairsForCalculation(lossFunction);
            NormalizeLeafValues(
                usePairs,
                ctx->Params.BoostingOptions->LearningRate,
                sumLeafWeights,
                &treeValues
            );

            TVector<TVector<double>>* foldZeroApprox = nullptr;
            if (UseAveragingFoldAsFoldZero(*ctx)) {
                foldZeroApprox = &trainFolds[0]->BodyTailArr[0].Approx;
            }
            UpdateAvrgApprox(
                error->GetIsExpApprox(),
                data.Learn->GetObjectCount(),
                indices,
                treeValues,
                data.Test,
                ctx->LearnProgress.Get(),
                ctx->LocalExecutor,
                foldZeroApprox);
            ctx->LocalExecutor->ExecRangeWithThrow(
                [&](int foldId)
                {
                    UpdateLearningFold(
                        data,
                        *error,
                        bestTree,
                        randomSeeds[foldId],
                        trainFolds[foldId],
                        ctx);
                },
                /*firstId=*/static_cast<int>(foldZeroApprox != nullptr),
                foldCount,
                NPar::TLocalExecutor::WAIT_COMPLETE);
            profile.AddOperation("CalcApprox tree struct and update tree structure approx");
            CheckInterrupted(); // check after long-lasting operation
        } else {
            const bool isMultiTarget = dynamic_cast<const TMultiDerCalcer*>(error.Get()) != nullptr;

            if (isMultiTarget || (ctx->LearnProgress->ApproxDimension != 1)) {
                MapSetApproxesMulti(*error, bestTree, &treeValues, &sumLeafWeights, ctx);
            } else {
                MapSetApproxesSimple(*error, bestTree, &treeValues, &sumLeafWeights, ctx);
            }
        }

        ctx->LearnProgress->TreeStats.emplace_back();
        ctx->LearnProgress->TreeStats.back().LeafWeightsSum = std::move(sumLeafWeights);
        ctx->LearnProgress->LeafValues.push_back(std::move(treeValues));
        ctx->LearnProgress->TreeStruct.push_back(std::move(bestTree));

        // 06.3-17: fence at the END of TrainOneIteration. The delta
        // tree_rng_end - tree_rng_start is the TOTAL per-tree persistent-RNG draw
        // count — the single number the Rust YetiRankTreeSeeder.next_tree() must
        // advance the context RNG by, per tree, for trees 0..iterations-1.
        CbInstrumentLog(TStringBuilder()
            << "{\"event\":\"tree_rng_end\",\"iter\":" << iterationIndex
            << ",\"cc\":" << ctx->LearnProgress->Rand.GetCallCount() << "}");

        profile.AddOperation("Update final approxes");
        CheckInterrupted(); // check after long-lasting operation
    }
}
