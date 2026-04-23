//===- AmdArchDbTests.cpp - Tests for the AMD arch database ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"

#include "gtest/gtest.h"

#include <numeric>
#include <string>
#include <vector>

using namespace mlir::rock;

// NOTE: this file deliberately does NOT include hip/hip_runtime_api.h or link
// libamdhip64. Doing so would pull in amd_comgr -> ROCm's libLLVM.so, which
// collides with the LLVM that the test binary embeds. Device enumeration and
// arch-name lookup go through the AmdArchDb public API, which delay-loads HIP
// at run time via a private `dlopen` / `LoadLibraryW` call.

class NativeArchTest : public ::testing::TestWithParam<unsigned> {
public:
  static auto getDeviceIds() {
    unsigned count = nativeDeviceCount();
    if (count == 0) {
      // Keep gtest happy when no GPU/HIP is available; the SetUp() below will
      // skip the test for the synthetic device id.
      return ::testing::ValuesIn(std::vector<unsigned>{0});
    }
    std::vector<unsigned> ids(count);
    std::iota(ids.begin(), ids.end(), 0u);
    return ::testing::ValuesIn(ids);
  }

protected:
  void SetUp() override {
    archName = nativeArchName(GetParam());
    if (archName.empty())
      GTEST_SKIP() << "No AMD GPU visible to HIP (or `libamdhip64` not on the "
                      "loader path); skipping native arch comparison for "
                      "device "
                   << GetParam();
  }

  std::string archName;
};

TEST_P(NativeArchTest, NativeArchInfoMatchesPresetInfo) {
  auto presetInfo = lookupArchInfo(archName);
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
  EXPECT_GE(presetInfo.maxNumXCC, nativeInfo.maxNumXCC);
}

INSTANTIATE_TEST_SUITE_P(NativeArchTests, NativeArchTest,
                         NativeArchTest::getDeviceIds());
