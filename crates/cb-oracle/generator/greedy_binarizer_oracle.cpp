// Ground-truth oracle for the GREEDY binarizer's heap tie-break behaviour.
//
// Upstream `library/cpp/grid_creator/binarization.cpp` drives the greedy split
// with a REAL `std::priority_queue<TBinType>` whose `operator<` compares only
// `Score()`. When several bins tie on score — which happens constantly on
// integer object counts (any two equal-sized bins splitting evenly score
// identically) — the bin that `top()` returns is decided by the binary-heap
// ARRAY STRUCTURE, not by insertion order. Reproducing that in another language
// is error-prone, so this program links the real STL container and prints the
// borders it produces.
//
// Build & run (see gen_border_type_fixtures.py for the fixture side):
//   g++ -O2 -std=c++20 -o /tmp/greedy_oracle greedy_binarizer_oracle.cpp
//   /tmp/greedy_oracle <max_borders> < values.txt
//
// stdin : one float32-representable value per line (UNSORTED is fine; the
//         program sorts, matching upstream's pre-sorted feature vector).
// stdout: the emitted borders, one per line, sorted ascending, printed with
//         enough digits to round-trip float.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <queue>
#include <set>
#include <string>
#include <vector>

namespace {

// Penalty<EPenaltyType::MaxSumLog>(w) = -log(w + 1e-8)
double PenaltyMaxSumLog(double weight) { return -std::log(weight + 1e-8); }

// Penalty<EPenaltyType::MinEntropy>(w) = w * log(w + 1e-8)
double PenaltyMinEntropy(double weight) { return weight * std::log(weight + 1e-8); }

enum class EPenalty { MaxSumLog, MinEntropy };

EPenalty gPenalty = EPenalty::MaxSumLog;

double Penalty(double weight) {
    return gPenalty == EPenalty::MaxSumLog ? PenaltyMaxSumLog(weight)
                                           : PenaltyMinEntropy(weight);
}

// A faithful transcription of IFeatureBin / TFeatureBin.
class TFeatureBin {
private:
    unsigned BinStart;
    unsigned BinEnd;
    std::vector<float>::const_iterator FeaturesStart;
    unsigned BestSplit;
    double BestScore;

    double CalcSplitScore(unsigned splitPos) const {
        if (splitPos == BinStart || splitPos == BinEnd) {
            return -std::numeric_limits<double>::infinity();
        }
        const double leftPartScore = -Penalty(double(splitPos - BinStart));
        const double rightPartScore = -Penalty(double(BinEnd - splitPos));
        const double currBinScore = -Penalty(double(BinEnd - BinStart));
        return leftPartScore + rightPartScore - currBinScore;
    }

    void UpdateBestSplitProperties() {
        const int mid = BinStart + (BinEnd - BinStart) / 2;
        float midValue = *(FeaturesStart + mid);

        const unsigned lb =
            unsigned(std::lower_bound(FeaturesStart + BinStart, FeaturesStart + mid, midValue) -
                     FeaturesStart);
        const unsigned ub =
            unsigned(std::upper_bound(FeaturesStart + mid, FeaturesStart + BinEnd, midValue) -
                     FeaturesStart);

        const double scoreLeft = CalcSplitScore(lb);
        const double scoreRight = CalcSplitScore(ub);
        BestSplit = scoreLeft >= scoreRight ? lb : ub;
        BestScore = BestSplit == lb ? scoreLeft : scoreRight;
    }

public:
    TFeatureBin(unsigned binStart, unsigned binEnd, std::vector<float>::const_iterator featuresStart)
        : BinStart(binStart), BinEnd(binEnd), FeaturesStart(featuresStart), BestSplit(binStart),
          BestScore(0.0) {
        UpdateBestSplitProperties();
    }

    bool operator<(const TFeatureBin& bf) const { return Score() < bf.Score(); }

    double Score() const { return BestScore; }

    bool CanSplit() const { return BinStart != BestSplit && BinEnd != BestSplit; }

    bool IsFirst() const { return BinStart == 0; }

    float LeftBorder() const {
        // 0.5f * values[start-1] + 0.5f * values[start]
        float borderValue = 0.5f * (*(FeaturesStart + BinStart - 1));
        borderValue += 0.5f * (*(FeaturesStart + BinStart));
        return borderValue;
    }

    TFeatureBin Split() {
        TFeatureBin left(BinStart, BestSplit, FeaturesStart);
        BinStart = BestSplit;
        UpdateBestSplitProperties();
        return left;
    }
};

// A faithful transcription of TWeightedFeatureBin (binarization.cpp:1427-1497).
// This is the bin type the WEIGHTED greedy entry (BestWeightedSplitImpl,
// EOptimizationType::Greedy) uses: it runs over UNIQUE values with CUMULATIVE
// weights, and finds its split at the median cumulative WEIGHT rather than at
// the middle INDEX -- a genuinely different search rule from TFeatureBin.
class TWeightedFeatureBin {
private:
    unsigned BinStart;
    unsigned BinEnd;
    std::vector<float>::const_iterator FeaturesStart;
    std::vector<float>::const_iterator CumulativeWeightsStart;
    unsigned BestSplit;
    double BestScore;

    double CalcSplitScore(unsigned splitPos) const {
        if (splitPos == BinStart || splitPos == BinEnd) {
            return -std::numeric_limits<double>::infinity();
        }
        const float leftBinsWeight = (BinStart == 0 ? 0.0f : *(CumulativeWeightsStart + BinStart - 1));
        const float leftPartWeight = *(CumulativeWeightsStart + splitPos - 1) - leftBinsWeight;
        const float rightPartWeight =
            *(CumulativeWeightsStart + BinEnd - 1) - *(CumulativeWeightsStart + splitPos - 1);
        const double currBinScore = -Penalty(leftPartWeight + rightPartWeight);
        const double newBinsScore = -(Penalty(leftPartWeight) + Penalty(rightPartWeight));
        return newBinsScore - currBinScore;
    }

    void UpdateBestSplitProperties() {
        const float leftBinsWeight = (BinStart == 0 ? 0.0f : *(CumulativeWeightsStart + BinStart - 1));
        const double midCumulativeWeight =
            0.5 * (double(leftBinsWeight) + double(*(CumulativeWeightsStart + BinEnd - 1)));

        const unsigned lb = unsigned(std::lower_bound(CumulativeWeightsStart + BinStart,
                                                     CumulativeWeightsStart + BinEnd,
                                                     float(midCumulativeWeight)) -
                                     CumulativeWeightsStart);
        const unsigned ub = lb + 1;

        const double scoreLeft = CalcSplitScore(lb);
        const double scoreRight = CalcSplitScore(ub);
        BestSplit = scoreLeft >= scoreRight ? lb : ub;
        BestScore = BestSplit == lb ? scoreLeft : scoreRight;
    }

public:
    TWeightedFeatureBin(unsigned binStart, unsigned binEnd,
                        std::vector<float>::const_iterator featuresStart,
                        std::vector<float>::const_iterator cumulativeWeightsStart)
        : BinStart(binStart), BinEnd(binEnd), FeaturesStart(featuresStart),
          CumulativeWeightsStart(cumulativeWeightsStart), BestSplit(binStart), BestScore(0.0) {
        UpdateBestSplitProperties();
    }

    bool operator<(const TWeightedFeatureBin& bf) const { return Score() < bf.Score(); }
    double Score() const { return BestScore; }
    bool CanSplit() const { return BinStart != BestSplit && BinEnd != BestSplit; }
    bool IsFirst() const { return BinStart == 0; }

    float LeftBorder() const {
        float borderValue = 0.5f * (*(FeaturesStart + BinStart - 1));
        borderValue += 0.5f * (*(FeaturesStart + BinStart));
        return borderValue;
    }

    TWeightedFeatureBin Split() {
        TWeightedFeatureBin left(BinStart, BestSplit, FeaturesStart, CumulativeWeightsStart);
        BinStart = BestSplit;
        UpdateBestSplitProperties();
        return left;
    }
};

// GreedySplit (binarization.cpp:1499-1520), generic over the bin type.
template <class TBinType>
std::set<float> GreedySplit(const TBinType& initialBin, int maxBordersCount) {
    std::priority_queue<TBinType> splits;
    splits.push(initialBin);

    while (splits.size() <= (unsigned)maxBordersCount && splits.top().CanSplit()) {
        auto top = splits.top();
        splits.pop();
        auto left = top.Split();
        splits.push(left);
        splits.push(top);
    }

    std::set<float> borders;
    while (!splits.empty()) {
        auto top = splits.top();
        splits.pop();
        if (!top.IsFirst()) {
            float b = top.LeftBorder();
            if (b == 0.0f) b = 0.0f;  // normalize -0.0f
            borders.insert(b);
        }
    }
    return borders;
}

// Same loop, but with a STABLE priority queue: among bins tied on score, the
// EARLIEST-INSERTED one is popped. std::priority_queue makes no such guarantee
// (the winner is decided by the heap array layout), so this variant exists to
// test whether catboost's observed borders correspond to insertion-order
// tie-breaking.
template <class TBinType>
std::set<float> GreedySplitStable(const TBinType& initialBin, int maxBordersCount) {
    std::vector<std::pair<TBinType, unsigned long long>> splits;  // (bin, insertion seq)
    unsigned long long seq = 0;
    splits.emplace_back(initialBin, seq++);

    auto topIndex = [&]() {
        size_t best = 0;
        for (size_t i = 1; i < splits.size(); ++i) {
            if (splits[i].first.Score() > splits[best].first.Score() ||
                (splits[i].first.Score() == splits[best].first.Score() &&
                 splits[i].second < splits[best].second)) {
                best = i;
            }
        }
        return best;
    };

    while (splits.size() <= (unsigned)maxBordersCount) {
        size_t ti = topIndex();
        if (!splits[ti].first.CanSplit()) break;
        auto top = splits[ti].first;
        splits.erase(splits.begin() + ti);
        auto left = top.Split();
        splits.emplace_back(left, seq++);
        splits.emplace_back(top, seq++);
    }

    std::set<float> borders;
    for (auto& p : splits) {
        if (!p.first.IsFirst()) {
            float b = p.first.LeftBorder();
            if (b == 0.0f) b = 0.0f;
            borders.insert(b);
        }
    }
    return borders;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <max_borders> [MaxSumLog|MinEntropy] "
                     "[unweighted|weighted|weighted_norm] < values.txt\n",
                     argv[0]);
        return 2;
    }
    const int maxBordersCount = std::atoi(argv[1]);
    if (argc >= 3 && std::string(argv[2]) == "MinEntropy") {
        gPenalty = EPenalty::MinEntropy;
    }
    const std::string mode = argc >= 4 ? argv[3] : "unweighted";

    std::vector<float> values;
    {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            values.push_back(std::strtof(line.c_str(), nullptr));
        }
    }
    std::sort(values.begin(), values.end());
    if (values.size() < 2) return 0;

    std::set<float> borders;
    if (mode == "unweighted") {
        TFeatureBin initialBin(0, unsigned(values.size()), values.begin());
        borders = GreedySplit(initialBin, maxBordersCount);
    } else if (mode == "unweighted_stable") {
        TFeatureBin initialBin(0, unsigned(values.size()), values.begin());
        borders = GreedySplitStable(initialBin, maxBordersCount);
    } else {
        // GroupAndSortWeighedValues: unique values + their object counts, then
        // CUMULATIVE weights (cumulativeWeights=true for the greedy path).
        std::vector<float> unique;
        std::vector<float> weights;
        for (float v : values) {
            if (!unique.empty() && unique.back() == v) {
                weights.back() += 1.0f;
            } else {
                unique.push_back(v);
                weights.push_back(1.0f);
            }
        }
        if (mode == "weighted_norm") {
            double total = 0.0;
            for (float w : weights) total += w;
            for (float& w : weights) w = float(w / total);
        }
        std::vector<float> cum(weights.size());
        float running = 0.0f;
        for (size_t i = 0; i < weights.size(); ++i) {
            running += weights[i];
            cum[i] = running;
        }
        if (unique.size() < 2) return 0;
        TWeightedFeatureBin initialBin(0, unsigned(unique.size()), unique.begin(), cum.begin());
        borders = (mode == "weighted_stable")
                      ? GreedySplitStable(initialBin, maxBordersCount)
                      : GreedySplit(initialBin, maxBordersCount);
    }

    std::cout << std::setprecision(9);
    for (float b : borders) {
        std::cout << b << "\n";
    }
    return 0;
}
