//===- AmdArchDbTests.cpp - Tests for the AMD arch database
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/AmdArchDb.h"

#include "gtest/gtest.h"
#include "gtest/internal/gtest-internal.h"

#include "hip/hip_runtime_api.h"

#include <numeric>

using namespace mlir::rock;

#define SOFT_CHECK_EQ(val1, val2)                                              \
  do {                                                                         \
    auto v1 = (val1);                                                          \
    auto v2 = (val2);                                                          \
    if (!(v1 == v2)) {                                                         \
      GTEST_LOG_(INFO)                                                         \
          << "(SOFT_CHECK_EQ) Expected equality of these values:\n"            \
          << "  " #val1 "\n    Which is: " << v1 << "\n"                       \
          << "  " #val2 "\n    Which is: " << v2 << "\n";                      \
    }                                                                          \
  } while (0)

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
  SOFT_CHECK_EQ(presetInfo.maxWavesPerEU, nativeInfo.maxWavesPerEU);
  SOFT_CHECK_EQ(presetInfo.totalSGPRPerEU, nativeInfo.totalSGPRPerEU);
  SOFT_CHECK_EQ(presetInfo.totalVGPRPerEU, nativeInfo.totalVGPRPerEU);
  EXPECT_EQ(presetInfo.totalSharedMemPerCU, nativeInfo.totalSharedMemPerCU);
  EXPECT_EQ(presetInfo.maxSharedMemPerWG, nativeInfo.maxSharedMemPerWG);
  EXPECT_EQ(presetInfo.numEUPerCU, nativeInfo.numEUPerCU);
  EXPECT_LE(presetInfo.minNumCU, nativeInfo.minNumCU);
  EXPECT_EQ(presetInfo.hasFp8ConversionInstrs,
            nativeInfo.hasFp8ConversionInstrs);
  EXPECT_EQ(presetInfo.hasOcpFp8ConversionInstrs,
            nativeInfo.hasOcpFp8ConversionInstrs);
  EXPECT_EQ(presetInfo.maxNumXCC, nativeInfo.maxNumXCC);
}

INSTANTIATE_TEST_SUITE_P(NativeArchTests, NativeArchTest,
                         NativeArchTest::getDeviceIds());
