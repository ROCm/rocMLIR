//===- InitParamsNonAccelTests.cpp - Tests for InitParamsNonAccel --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::rock;

namespace {

//===----------------------------------------------------------------------===//
// v3 perfconfig
//===----------------------------------------------------------------------===//

TEST(V3Config, First) {
  InitParamsAccel validParams;
  bool isValidPerfConfig =
      validParams.deserialize("v3:64,32,32,4,2,4,1,1,2");

  EXPECT_EQ(isValidPerfConfig, true);
  EXPECT_EQ(validParams.gemmBThreadCopyMoreGemmKPack, true);
  EXPECT_EQ(validParams.gemmAThreadCopyMoreGemmK, true);
  EXPECT_EQ(validParams.getKPack(), 4);
  EXPECT_EQ(validParams.gemmKPack, 4);
  EXPECT_EQ(validParams.splitKFactor, 2);
  EXPECT_EQ(validParams.gemmMPerWave, 32);
  EXPECT_EQ(validParams.gemmNPerWave, 32);
  EXPECT_EQ(validParams.gemmNPerWaveOrMnPerXdl, 0);
  EXPECT_EQ(validParams.gemmScheduleVersion, 3);
  EXPECT_EQ(validParams.gemmMnPerXdl, 16);
  EXPECT_EQ(validParams.outputSwizzle, 2);
  EXPECT_EQ(validParams.getVersion(), InitParamsAccel::Version::V4);
  EXPECT_EQ(validParams.gemmMPerBlock, 64);
  EXPECT_EQ(validParams.gemmNPerBlock, 64);
  EXPECT_EQ(validParams.gemmKPerBlock, 8);
}

TEST(V4Config, Second) {
  InitParamsAccel validParams;
  bool isValidPerfConfig =
      validParams.deserialize("v3:64,64,32,4,2,4,1,1,2");

  EXPECT_EQ(isValidPerfConfig, true);
  EXPECT_EQ(validParams.gemmBThreadCopyMoreGemmKPack, true);
  EXPECT_EQ(validParams.gemmAThreadCopyMoreGemmK, false);
  EXPECT_EQ(validParams.getKPack(), 4);
  EXPECT_EQ(validParams.gemmKPack, 4);
  EXPECT_EQ(validParams.splitKFactor, 9);
  EXPECT_EQ(validParams.gemmMPerWave, 64);
  EXPECT_EQ(validParams.gemmNPerWave, 32);
  EXPECT_EQ(validParams.gemmNPerWaveOrMnPerXdl, 0);
  EXPECT_EQ(validParams.gemmScheduleVersion, 2);
  EXPECT_EQ(validParams.gemmMnPerXdl, 32);
  EXPECT_EQ(validParams.outputSwizzle, 2);
  EXPECT_EQ(validParams.getVersion(), InitParamsAccel::Version::V4);
  EXPECT_EQ(validParams.gemmMPerBlock, 128);
  EXPECT_EQ(validParams.gemmNPerBlock, 64);
  EXPECT_EQ(validParams.gemmKPerBlock, 8);
}

//===----------------------------------------------------------------------===//
// Negative Tests
//===----------------------------------------------------------------------===//

TEST(NegativeTests, NoVersion) {
  InitParamsAccel validParams;
  bool isValidPerfConfig =
      validParams.deserialize("128,64,8,64,32,4,9,2,2,0,1");

  EXPECT_EQ(isValidPerfConfig, false);
}

TEST(NegativeTests, WrongNumberV3) {
  InitParamsAccel validParams;
  bool isValidPerfConfig =
      validParams.deserialize("v3:64,32,32,4,2,4,1,1");

  EXPECT_EQ(isValidPerfConfig, false);
}

TEST(NegativeTests, Empty) {
  InitParamsAccel validParams;
  bool isValidPerfConfig = validParams.deserialize("");

  EXPECT_EQ(isValidPerfConfig, false);
}

} // end anonymous namespace
