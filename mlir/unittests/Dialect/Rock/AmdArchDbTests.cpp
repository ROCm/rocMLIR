//===- AmdArchDbTests.cpp - Tests for the AMD arch database
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/AmdArchDb.h"

#include "gtest/gtest.h"

#include "hip/hip_runtime_api.h"

using namespace mlir::rock;

#define SOFT_CHECK_EQ(val1, val2)                                              \
  do {                                                                         \
    auto v1 = (val1);                                                          \
    auto v2 = (val2);                                                          \
    if (!(v1 == v2)) {                                                         \
      llvm::errs() << "(SOFT_CHECK_EQ) Expected equality of these values:\n"   \
                   << "  " #val1 "\n    Which is: " << v1 << "\n"              \
                   << "  " #val2 "\n    Which is: " << v2 << "\n\n";           \
    }                                                                          \
  } while (0)

TEST(AmdArchDbTest, Native) {
  hipDeviceProp_t prop;
  if (auto err = hipGetDeviceProperties(&prop, 0); err != hipSuccess) {
    FAIL() << "hipGetDeviceProperties failed with error: "
           << hipGetErrorString(err);
  }

  auto presetInfo = lookupArchInfo(prop.gcnArchName);
  auto nativeInfo = lookupArchInfo("native");

  SOFT_CHECK_EQ(presetInfo.defaultFeatures, nativeInfo.defaultFeatures);
  SOFT_CHECK_EQ(presetInfo.waveSize, nativeInfo.waveSize);
  SOFT_CHECK_EQ(presetInfo.maxWavesPerEU, nativeInfo.maxWavesPerEU);
  SOFT_CHECK_EQ(presetInfo.totalSGPRPerEU, nativeInfo.totalSGPRPerEU);
  SOFT_CHECK_EQ(presetInfo.totalVGPRPerEU, nativeInfo.totalVGPRPerEU);
  SOFT_CHECK_EQ(presetInfo.totalSharedMemPerCU, nativeInfo.totalSharedMemPerCU);
  SOFT_CHECK_EQ(presetInfo.maxSharedMemPerWG, nativeInfo.maxSharedMemPerWG);
  SOFT_CHECK_EQ(presetInfo.numEUPerCU, nativeInfo.numEUPerCU);
  SOFT_CHECK_EQ(presetInfo.minNumCU, nativeInfo.minNumCU);
  SOFT_CHECK_EQ(presetInfo.hasFp8ConversionInstrs,
                nativeInfo.hasFp8ConversionInstrs);
  SOFT_CHECK_EQ(presetInfo.hasOcpFp8ConversionInstrs,
                nativeInfo.hasOcpFp8ConversionInstrs);
  SOFT_CHECK_EQ(presetInfo.maxNumXCC, nativeInfo.maxNumXCC);
}
