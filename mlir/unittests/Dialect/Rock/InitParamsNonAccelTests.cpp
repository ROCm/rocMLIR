//===- InitParamsNonAccelTests.cpp - Tests for InitParamsNonAccel
//--------------===//
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
  InitParamsNonAccel validParams;
  bool isValidPerfConfig = validParams.deserialize("v3:64,32,32,4,2,4,1,1,2");

  EXPECT_EQ(isValidPerfConfig, true);
  EXPECT_EQ(validParams.blockSize, static_cast<uint32_t>(64));
  EXPECT_EQ(validParams.gemmMPerBlock, 32);
  EXPECT_EQ(validParams.gemmNPerBlock, 32);
  EXPECT_EQ(validParams.gemmKPerBlock, 4);
  EXPECT_EQ(validParams.gemmMPerThread, 2);
  EXPECT_EQ(validParams.gemmNPerThread, 4);
  EXPECT_EQ(validParams.splitKFactor, 1);
  EXPECT_EQ(validParams.gemmScheduleVersion, 1);
  EXPECT_EQ(validParams.outputSwizzle, 2);
  EXPECT_EQ(validParams.getKPack(), 1);
  EXPECT_EQ(validParams.getVersion(), InitParamsNonAccel::Version::V3);
}

TEST(V3Config, Second) {
  InitParamsNonAccel validParams;
  bool isValidPerfConfig = validParams.deserialize("v3:128,64,32,8,4,2,3,1,2");

  EXPECT_EQ(isValidPerfConfig, true);
  EXPECT_EQ(validParams.blockSize, static_cast<uint32_t>(128));
  EXPECT_EQ(validParams.gemmMPerBlock, 64);
  EXPECT_EQ(validParams.gemmNPerBlock, 32);
  EXPECT_EQ(validParams.gemmKPerBlock, 8);
  EXPECT_EQ(validParams.gemmMPerThread, 4);
  EXPECT_EQ(validParams.gemmNPerThread, 2);
  EXPECT_EQ(validParams.splitKFactor, 3);
  EXPECT_EQ(validParams.gemmScheduleVersion, 1);
  EXPECT_EQ(validParams.outputSwizzle, 2);
  EXPECT_EQ(validParams.getKPack(), 1);
  EXPECT_EQ(validParams.getVersion(), InitParamsNonAccel::Version::V3);
}

//===----------------------------------------------------------------------===//
// Negative Tests
//===----------------------------------------------------------------------===//

TEST(NegativeTests, NoVersion) {
  InitParamsNonAccel validParams;
  bool isValidPerfConfig =
      validParams.deserialize("128,64,8,64,32,4,9,2,2,0,1");

  EXPECT_EQ(isValidPerfConfig, false);
}

TEST(NegativeTests, WrongNumberV3) {
  InitParamsNonAccel validParams;
  bool isValidPerfConfig = validParams.deserialize("v3:64,32,32,4,2,4,1,1");

  EXPECT_EQ(isValidPerfConfig, false);
}

TEST(NegativeTests, Empty) {
  InitParamsNonAccel validParams;
  bool isValidPerfConfig = validParams.deserialize("");

  EXPECT_EQ(isValidPerfConfig, false);
}

} // end anonymous namespace
