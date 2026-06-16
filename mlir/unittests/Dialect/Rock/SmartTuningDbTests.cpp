//===- SmartTuningDbTests.cpp - Tests for SmartTuningDb ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/SmartTuningDb.h"
#include "mlir/Dialect/Rock/Tuning/SmartTuningFeatures.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cmath>
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::rock;

//===- Database invariants -----------------------------------------------===//

TEST(SmartTuningDb, IsSortedByKey) {
  EXPECT_TRUE(SmartTuningDb::isSortedByKey());
}

//===- Model resolution --------------------------------------------------===//
// gfx942 gemm/conv/attention models are committed; resolution normalizes
// decorated arch strings and rejects ops/arches with no embedded model.

TEST(SmartTuningDb, ResolveModel) {
  const auto *gemm = SmartTuningDb::resolveModel("gfx942", KernelType::Gemm);
  ASSERT_NE(gemm, nullptr);
  EXPECT_EQ(StringRef(gemm->key), "gfx942_gemm");
  EXPECT_EQ(gemm->numFeatures, SmartTuningFeatures::gemmFeatureNames().size());

  const auto *conv = SmartTuningDb::resolveModel("gfx942", KernelType::Conv);
  ASSERT_NE(conv, nullptr);
  EXPECT_EQ(StringRef(conv->key), "gfx942_conv");
  EXPECT_EQ(conv->numFeatures, SmartTuningFeatures::convFeatureNames().size());
  // All conv directions share the one conv model.
  EXPECT_EQ(SmartTuningDb::resolveModel("gfx942", KernelType::ConvBwdData),
            conv);
  EXPECT_EQ(SmartTuningDb::resolveModel("gfx942", KernelType::ConvBwdWeight),
            conv);

  const auto *attn =
      SmartTuningDb::resolveModel("gfx942", KernelType::Attention);
  ASSERT_NE(attn, nullptr);
  EXPECT_EQ(StringRef(attn->key), "gfx942_attention");
  EXPECT_EQ(attn->numFeatures,
            SmartTuningFeatures::attentionFeatureNames().size());

  // Decorated target string resolves to the same model.
  EXPECT_EQ(
      SmartTuningDb::resolveModel("gfx942:sramecc+:xnack-", KernelType::Gemm),
      gemm);

  // No model for an unknown arch or an op without a learned model.
  EXPECT_EQ(SmartTuningDb::resolveModel("gfx900", KernelType::Gemm), nullptr);
  EXPECT_EQ(
      SmartTuningDb::resolveModel("gfx942", KernelType::GemmElementwiseGemm),
      nullptr);
}

//===- Feature parity ----------------------------------------------------===//
// The C++ extractor must reproduce the Python pipeline
// (tuning_eval/features.py) bit-for-bit. Goldens are captured from
// feature_record(); regenerate them when the feature set changes (mirrors
// QuickTuningProblemKey's golden discipline).

namespace {
struct GoldenCase {
  SmartTuningFeatures::GemmSig sig;
  StringRef perfConfig;
  std::vector<double> expected;
};

void expectFeatures(const GoldenCase &c) {
  SmallVector<double> got;
  SmartTuningFeatures::gemmFeatures(c.sig, c.perfConfig, got);
  ASSERT_EQ(got.size(), c.expected.size());
  ASSERT_EQ(got.size(), SmartTuningFeatures::gemmFeatureNames().size());
  for (size_t i = 0; i < got.size(); ++i) {
    double tol = std::max(1e-6, 1e-9 * std::abs(c.expected[i]));
    EXPECT_NEAR(got[i], c.expected[i], tol)
        << "feature[" << i
        << "] = " << SmartTuningFeatures::gemmFeatureNames()[i];
  }
}
} // namespace

TEST(SmartTuningFeatures, GemmFeatureCount) {
  EXPECT_EQ(SmartTuningFeatures::gemmFeatureNames().size(), 61u);
}

TEST(SmartTuningFeatures, GemmParityF16) {
  expectFeatures({{"gfx942", "f16", 304, 8, /*transA=*/true, /*transB=*/false,
                   1, 2048, 256, 2048},
                  "v2:64,128,8,16,32,4,1,1,2",
                  {1.0,
                   0.0,
                   1.0,
                   2048.0,
                   2048.0,
                   256.0,
                   11.0,
                   11.0,
                   8.0,
                   0.0,
                   1.0,
                   8.0,
                   2147483648.0,
                   31.0,
                   204.8,
                   16.0,
                   1.0,
                   0.0,
                   304.0,
                   8.0,
                   64.0,
                   1.0,
                   0.0,
                   65536.0,
                   65536.0,
                   512.0,
                   8.0,
                   4.0,
                   64.0,
                   128.0,
                   8.0,
                   16.0,
                   4.0,
                   1.0,
                   64.0,
                   128.0,
                   8.0,
                   16.0,
                   32.0,
                   4.0,
                   1.0,
                   1.0,
                   2.0,
                   -1.0,
                   -1.0,
                   -1.0,
                   -1.0,
                   -1.0,
                   32.0,
                   16.0,
                   8.0,
                   512.0,
                   1.6842105263157894,
                   1.1875,
                   1.0,
                   1.0,
                   0.0,
                   0.0,
                   12288.0,
                   0.1875,
                   5.0}});
}

TEST(SmartTuningFeatures, GemmParityF32) {
  expectFeatures({{"gfx942", "f32", 304, 8, /*transA=*/false, /*transB=*/false,
                   1, 1024, 1024, 512},
                  "v2:128,128,8,32,32,4,1,1,0",
                  {0.0,
                   0.0,
                   1.0,
                   1024.0,
                   512.0,
                   1024.0,
                   10.0,
                   9.0,
                   10.0,
                   0.0,
                   2.0,
                   1.0,
                   1073741824.0,
                   30.0,
                   128.0,
                   32.0,
                   1.0,
                   0.0,
                   304.0,
                   8.0,
                   64.0,
                   1.0,
                   0.0,
                   65536.0,
                   65536.0,
                   512.0,
                   8.0,
                   4.0,
                   128.0,
                   128.0,
                   8.0,
                   32.0,
                   4.0,
                   1.0,
                   128.0,
                   128.0,
                   8.0,
                   32.0,
                   32.0,
                   4.0,
                   1.0,
                   1.0,
                   0.0,
                   -1.0,
                   -1.0,
                   -1.0,
                   -1.0,
                   -1.0,
                   8.0,
                   4.0,
                   32.0,
                   32.0,
                   0.10526315789473684,
                   9.5,
                   1.0,
                   1.0,
                   0.0,
                   0.0,
                   32768.0,
                   0.5,
                   2.0}});
}

TEST(SmartTuningFeatures, GemmParityI8) {
  expectFeatures({{"gfx942", "i8", 304, 8, /*transA=*/false, /*transB=*/true, 1,
                   512, 1024, 1024},
                  "v2:256,128,4,64,32,8,1,1,0",
                  {0.0,
                   1.0,
                   1.0,
                   512.0,
                   1024.0,
                   1024.0,
                   9.0,
                   10.0,
                   10.0,
                   0.0,
                   0.5,
                   0.5,
                   1073741824.0,
                   30.0,
                   512.0,
                   8.0,
                   0.0,
                   0.0,
                   304.0,
                   8.0,
                   64.0,
                   1.0,
                   0.0,
                   65536.0,
                   65536.0,
                   512.0,
                   8.0,
                   4.0,
                   256.0,
                   128.0,
                   4.0,
                   64.0,
                   8.0,
                   1.0,
                   256.0,
                   128.0,
                   4.0,
                   64.0,
                   32.0,
                   8.0,
                   1.0,
                   1.0,
                   0.0,
                   -1.0,
                   -1.0,
                   -1.0,
                   -1.0,
                   -1.0,
                   2.0,
                   8.0,
                   32.0,
                   16.0,
                   0.05263157894736842,
                   19.0,
                   1.0,
                   1.0,
                   0.0,
                   0.0,
                   12288.0,
                   0.1875,
                   5.0}});
}

namespace {
struct ConvGoldenCase {
  SmartTuningFeatures::ConvSig sig;
  StringRef perfConfig;
  std::vector<double> expected;
};

void expectConvFeatures(const ConvGoldenCase &c) {
  SmallVector<double> got;
  SmartTuningFeatures::convFeatures(c.sig, c.perfConfig, got);
  ASSERT_EQ(got.size(), c.expected.size());
  ASSERT_EQ(got.size(), SmartTuningFeatures::convFeatureNames().size());
  for (size_t i = 0; i < got.size(); ++i) {
    double tol = std::max(1e-6, 1e-9 * std::abs(c.expected[i]));
    EXPECT_NEAR(got[i], c.expected[i], tol)
        << "feature[" << i
        << "] = " << SmartTuningFeatures::convFeatureNames()[i];
  }
}

struct AttentionGoldenCase {
  SmartTuningFeatures::AttentionSig sig;
  StringRef perfConfig;
  std::vector<double> expected;
};

void expectAttentionFeatures(const AttentionGoldenCase &c) {
  SmallVector<double> got;
  SmartTuningFeatures::attentionFeatures(c.sig, c.perfConfig, got);
  ASSERT_EQ(got.size(), c.expected.size());
  ASSERT_EQ(got.size(), SmartTuningFeatures::attentionFeatureNames().size());
  for (size_t i = 0; i < got.size(); ++i) {
    double tol = std::max(1e-6, 1e-9 * std::abs(c.expected[i]));
    EXPECT_NEAR(got[i], c.expected[i], tol)
        << "feature[" << i
        << "] = " << SmartTuningFeatures::attentionFeatureNames()[i];
  }
}
} // namespace

TEST(SmartTuningFeatures, ConvFeatureCount) {
  EXPECT_EQ(SmartTuningFeatures::convFeatureNames().size(), 99u);
}

TEST(SmartTuningFeatures, AttentionFeatureCount) {
  EXPECT_EQ(SmartTuningFeatures::attentionFeatureNames().size(), 74u);
}

// Goldens captured from features.py feature_record (num_cu=304,
// num_chiplets=1).

TEST(SmartTuningFeatures, ConvParityFwdF32) {
  expectConvFeatures(
      {{"gfx942", "f32", 304, 1, "fwd", "gkc01", "ngc01", "ngk01", 1, 512, 28,
        28,       128,   1,   1, 1,     1,       1,       1,       0, 0},
       "v4:16,16,4,16,16,16,1,1,4,2,0,0,1,1",
       {1.0,
        0.0,
        0.0,
        1.0,
        512.0,
        28.0,
        28.0,
        128.0,
        1.0,
        1.0,
        0.0,
        9.0,
        4.807354922057604,
        4.807354922057604,
        7.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        28.0,
        28.0,
        4.807354922057604,
        4.807354922057604,
        1.0,
        128.0,
        784.0,
        512.0,
        7.0,
        9.614709844115207,
        9.0,
        102760448.0,
        26.614709844115207,
        45.285198555956676,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        0.0,
        1.0,
        -1.0,
        2.0,
        3.0,
        4.0,
        0.0,
        1.0,
        2.0,
        -1.0,
        3.0,
        4.0,
        32.0,
        1.0,
        0.0,
        304.0,
        1.0,
        64.0,
        1.0,
        0.0,
        65536.0,
        65536.0,
        512.0,
        8.0,
        4.0,
        16.0,
        16.0,
        4.0,
        16.0,
        1.0,
        1.0,
        16.0,
        16.0,
        4.0,
        16.0,
        16.0,
        16.0,
        1.0,
        1.0,
        4.0,
        2.0,
        0.0,
        0.0,
        1.0,
        1.0,
        8.0,
        49.0,
        128.0,
        392.0,
        1.2894736842105263,
        1.5510204081632653,
        1.0,
        1.0,
        0.0,
        0.0,
        512.0,
        0.0078125,
        128.0}});
}

TEST(SmartTuningFeatures, ConvParityBwdF16) {
  expectConvFeatures(
      {{"gfx942", "f16", 304, 1, "bwd", "gkc01", "ngc01", "ngk01", 8, 64, 14,
        14,       64,    3,   3, 2,     2,       1,       1,       1, 1},
       "v4:32,64,8,16,32,16,16,1,1,2,0,0,1,1",
       {0.0,
        1.0,
        0.0,
        8.0,
        64.0,
        14.0,
        14.0,
        64.0,
        3.0,
        3.0,
        3.0,
        6.0,
        3.807354922057604,
        3.807354922057604,
        6.0,
        2.0,
        2.0,
        1.0,
        1.0,
        1.0,
        1.0,
        7.0,
        7.0,
        2.807354922057604,
        2.807354922057604,
        9.0,
        64.0,
        512.0,
        256.0,
        6.0,
        9.0,
        8.0,
        16777216.0,
        24.0,
        51.68454258675079,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        0.0,
        1.0,
        -1.0,
        2.0,
        3.0,
        4.0,
        0.0,
        1.0,
        2.0,
        -1.0,
        3.0,
        4.0,
        16.0,
        1.0,
        0.0,
        304.0,
        1.0,
        64.0,
        1.0,
        0.0,
        65536.0,
        65536.0,
        512.0,
        8.0,
        4.0,
        32.0,
        64.0,
        8.0,
        16.0,
        16.0,
        1.0,
        32.0,
        64.0,
        8.0,
        16.0,
        32.0,
        16.0,
        16.0,
        1.0,
        1.0,
        2.0,
        0.0,
        0.0,
        1.0,
        1.0,
        2.0,
        8.0,
        2.0,
        64.0,
        0.21052631578947367,
        4.75,
        1.0,
        1.0,
        0.0,
        0.0,
        24576.0,
        0.375,
        2.0}});
}

TEST(SmartTuningFeatures, ConvParityWrwF32) {
  expectConvFeatures(
      {{"gfx942", "f32", 304, 1, "wrw", "gkc01", "ngc01", "ngk01", 4, 128, 32,
        32,       256,   3,   3, 1,     1,       1,       1,       1, 1},
       "v4:64,128,8,32,32,4,1,1,0",
       {0.0,
        0.0,
        1.0,
        4.0,
        128.0,
        32.0,
        32.0,
        256.0,
        3.0,
        3.0,
        2.0,
        7.0,
        5.0,
        5.0,
        8.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        32.0,
        32.0,
        5.0,
        5.0,
        9.0,
        256.0,
        1152.0,
        4096.0,
        8.0,
        10.169925001442312,
        12.0,
        2415919104.0,
        31.169925001442312,
        323.36842105263156,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        0.0,
        1.0,
        -1.0,
        2.0,
        3.0,
        4.0,
        0.0,
        1.0,
        2.0,
        -1.0,
        3.0,
        4.0,
        32.0,
        1.0,
        0.0,
        304.0,
        1.0,
        64.0,
        1.0,
        0.0,
        65536.0,
        65536.0,
        512.0,
        8.0,
        4.0,
        64.0,
        128.0,
        8.0,
        32.0,
        1.0,
        1.0,
        64.0,
        128.0,
        8.0,
        32.0,
        32.0,
        4.0,
        1.0,
        1.0,
        0.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        4.0,
        9.0,
        512.0,
        36.0,
        0.11842105263157894,
        8.444444444444445,
        1.0,
        1.0,
        0.0,
        0.0,
        6144.0,
        0.09375,
        10.0}});
}

TEST(SmartTuningFeatures, AttentionParityF32) {
  expectAttentionFeatures(
      {{"gfx942", "f32", 304,   1,  false, true, false, false, false, false,
        1,        false, false, 12, 1,     1,    384,   384,   64,    64},
       "attn:v3:16,16,16,64,16,16,16,4,1,3,2,0,1",
       {0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        12.0,
        1.0,
        1.0,
        1.0,
        12.0,
        384.0,
        384.0,
        64.0,
        64.0,
        8.584962500721156,
        8.584962500721156,
        6.0,
        6.0,
        3.584962500721156,
        1.0,
        1.0,
        452984832.0,
        28.75488750216347,
        96.0,
        32.0,
        1.0,
        0.0,
        304.0,
        1.0,
        64.0,
        1.0,
        0.0,
        65536.0,
        65536.0,
        512.0,
        8.0,
        4.0,
        16.0,
        16.0,
        64.0,
        16.0,
        4.0,
        1.0,
        16.0,
        16.0,
        16.0,
        64.0,
        16.0,
        16.0,
        16.0,
        4.0,
        1.0,
        3.0,
        2.0,
        0.0,
        1.0,
        -1.0,
        24.0,
        24.0,
        1.0,
        6912.0,
        22.736842105263158,
        1.0115740740740742,
        1.0,
        1.0,
        0.0,
        0.0,
        32768.0,
        0.5,
        2.0}});
}

TEST(SmartTuningFeatures, AttentionParityF16) {
  expectAttentionFeatures(
      {{"gfx942", "f16", 304,  1, false, false, false, false, true, false,
        2,        false, true, 4, 8,     2,     2048,  2048,  128,  128},
       "attn:v3:64,16,16,128,32,16,16,4,1,3,2,0,1",
       {0.0,         0.0,
        0.0,         0.0,
        1.0,         0.0,
        2.0,         0.0,
        1.0,         4.0,
        8.0,         2.0,
        4.0,         32.0,
        2048.0,      2048.0,
        128.0,       128.0,
        11.0,        11.0,
        7.0,         7.0,
        5.0,         1.0,
        1.0,         34359738368.0,
        35.0,        819.2,
        16.0,        1.0,
        0.0,         304.0,
        1.0,         64.0,
        1.0,         0.0,
        65536.0,     65536.0,
        512.0,       8.0,
        4.0,         64.0,
        16.0,        128.0,
        32.0,        4.0,
        1.0,         64.0,
        16.0,        16.0,
        128.0,       32.0,
        16.0,        16.0,
        4.0,         1.0,
        3.0,         2.0,
        0.0,         1.0,
        -1.0,        32.0,
        128.0,       1.0,
        131072.0,    431.1578947368421,
        1.001953125, 1.0,
        1.0,         0.0,
        0.0,         81920.0,
        1.25,        0.0}});
}

//===- Ranking logic -----------------------------------------------------===//
// rankConfigs tiers predicted-applicable configs first (by optimality), then
// the rest (by applicability). Driven by a hand-built two-stage model so the
// tiering is checked independently of any trained weights.

namespace {
// Applicability tree: feature[0] <= 0.5 -> +10 (applicable), else -10.
const SmartTuningDb::TreeNode kApplicNodes[] = {
    {0, 0.5, 1, 2, 0.0}, {-1, 0.0, -1, -1, 10.0}, {-1, 0.0, -1, -1, -10.0}};
const unsigned kApplicRoots[] = {0};
// Optimality tree: feature[1] <= 0.5 -> -1, else +1 (bigger = better).
const SmartTuningDb::TreeNode kOptimalNodes[] = {
    {1, 0.5, 1, 2, 0.0}, {-1, 0.0, -1, -1, -1.0}, {-1, 0.0, -1, -1, 1.0}};
const unsigned kOptimalRoots[] = {0};

SmartTuningDb::Model makeSyntheticModel() {
  return {"synthetic_gemm",
          2,
          {kApplicNodes, 3, kApplicRoots, 1, 0.0},
          {kOptimalNodes, 3, kOptimalRoots, 1, 0.0}};
}
} // namespace

TEST(SmartTuningDb, StageMargin) {
  SmartTuningDb::Model m = makeSyntheticModel();
  std::vector<double> applicable = {0.0, 1.0};
  std::vector<double> inapplicable = {1.0, 0.0};
  EXPECT_DOUBLE_EQ(SmartTuningDb::stageMargin(m.applicability, applicable),
                   10.0);
  EXPECT_DOUBLE_EQ(SmartTuningDb::stageMargin(m.applicability, inapplicable),
                   -10.0);
  EXPECT_DOUBLE_EQ(SmartTuningDb::stageMargin(m.optimality, applicable), 1.0);
}

TEST(SmartTuningDb, RankConfigsTiersByApplicabilityThenOptimality) {
  SmartTuningDb::Model m = makeSyntheticModel();
  std::vector<std::vector<double>> rows = {
      {0.0, 1.0}, // 0: applicable, high optimality
      {0.0, 0.0}, // 1: applicable, low optimality
      {1.0, 1.0}, // 2: inapplicable
  };
  SmallVector<ArrayRef<double>> featureRows;
  for (auto &r : rows)
    featureRows.push_back(r);

  auto ranked = SmartTuningDb::rankConfigs(m, featureRows, /*maxK=*/10);
  ASSERT_EQ(ranked.size(), 3u);
  EXPECT_EQ(ranked[0], 0u); // applicable + best optimality first
  EXPECT_EQ(ranked[1], 1u); // applicable but worse optimality
  EXPECT_EQ(ranked[2], 2u); // inapplicable last

  // maxK truncates while preserving order.
  auto top2 = SmartTuningDb::rankConfigs(m, featureRows, /*maxK=*/2);
  ASSERT_EQ(top2.size(), 2u);
  EXPECT_EQ(top2[0], 0u);
  EXPECT_EQ(top2[1], 1u);
}
