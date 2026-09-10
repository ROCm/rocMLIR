//===- AmdArchDbTests.cpp - Tests for the AMD arch database ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"

#include "gtest/gtest.h"

#include "hip/hip_runtime_api.h"

#include <numeric>

using namespace mlir::rock;

class NativeArchTest : public ::testing::TestWithParam<int> {
public:
  static auto getDeviceIds() {
    int count;
    if (auto err = hipGetDeviceCount(&count); err != hipSuccess) {
      return ::testing::ValuesIn({0});
    }
    std::vector<int> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    return ::testing::ValuesIn(ids);
  }

protected:
  void SetUp() override {
    if (auto err = hipGetDeviceProperties(&prop, GetParam());
        err != hipSuccess) {
      FAIL() << "hipGetDeviceProperties failed with error: "
             << hipGetErrorString(err);
    }
  }

  hipDeviceProp_t prop;
};

TEST_P(NativeArchTest, NativeArchInfoMatchesPresetInfo) {
  auto presetInfo = lookupArchInfo(prop.gcnArchName);
  auto nativeInfo = lookupArchInfo("native:" + std::to_string(GetParam()));

  EXPECT_EQ(presetInfo.defaultFeatures, nativeInfo.defaultFeatures);
  EXPECT_EQ(presetInfo.waveSize, nativeInfo.waveSize);
  EXPECT_EQ(presetInfo.maxWavesPerEU, nativeInfo.maxWavesPerEU);
  EXPECT_EQ(presetInfo.totalSGPRPerEU, nativeInfo.totalSGPRPerEU);
  EXPECT_EQ(presetInfo.totalVGPRPerEU, nativeInfo.totalVGPRPerEU);
  EXPECT_EQ(presetInfo.totalSharedMemPerCU, nativeInfo.totalSharedMemPerCU);
  EXPECT_EQ(presetInfo.maxSharedMemPerWG, nativeInfo.maxSharedMemPerWG);
  EXPECT_EQ(presetInfo.numEUPerCU, nativeInfo.numEUPerCU);
  EXPECT_LE(presetInfo.minNumCU, nativeInfo.minNumCU);
  EXPECT_EQ(presetInfo.hasFp8ConversionInstrs,
            nativeInfo.hasFp8ConversionInstrs);
  EXPECT_EQ(presetInfo.hasOcpFp8ConversionInstrs,
            nativeInfo.hasOcpFp8ConversionInstrs);
  EXPECT_EQ(presetInfo.hasFp4, nativeInfo.hasFp4);
  EXPECT_EQ(presetInfo.hasScaledGemm, nativeInfo.hasScaledGemm);
  EXPECT_GE(presetInfo.maxNumXCC, nativeInfo.maxNumXCC);
  EXPECT_EQ(presetInfo.hasLdsTransposeLoad, nativeInfo.hasLdsTransposeLoad);
}

INSTANTIATE_TEST_SUITE_P(NativeArchTests, NativeArchTest,
                         NativeArchTest::getDeviceIds());

// --- getLastLevelCacheSize ---

TEST(AmdArchDbTest, LastLevelCacheSize) {
  constexpr int64_t kMiB = 1024 * 1024;
  EXPECT_EQ(getLastLevelCacheSize("gfx906"), 4 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx908"), 8 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx90a"), 8 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx942"), 256 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx950"), 256 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1010"), 4 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1030"), 128 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1100"), 96 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1200"), 64 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1250"), 256 * kMiB);
}

TEST(AmdArchDbTest, LastLevelCacheSizeWithTriple) {
  constexpr int64_t kMiB = 1024 * 1024;
  EXPECT_EQ(getLastLevelCacheSize("amdgcn-amd-amdhsa:gfx942"), 256 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("amdgcn-amd-amdhsa:gfx906:xnack-"), 4 * kMiB);
}
