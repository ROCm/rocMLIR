//===- AccelGemmParamsAttrTests.cpp - Tests for AccelGemmParamsAttr -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::rock;

namespace {

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class AccelGemmParamsAttrTest : public ::testing::Test {
protected:
  void SetUp() override {
    ctx.loadDialect<RockDialect>();
    builder = std::make_unique<OpBuilder>(&ctx);
  }

  AccelGemmParamsAttr parse(StringRef perfConfig, bool isWmma = false) {
    return AccelGemmParamsAttr::get(builder->getStringAttr(perfConfig), isWmma);
  }

  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
};

//===----------------------------------------------------------------------===//
// v4 perfconfig tests
//===----------------------------------------------------------------------===//

TEST_F(AccelGemmParamsAttrTest, V4ConfigFirst) {
  auto params = parse("v4:64,64,8,32,32,16,4,2,3,2,0,0,1,1");

  ASSERT_TRUE(params);
  EXPECT_EQ(params.getKpackPerBlock(), 8);
  EXPECT_EQ(params.getMPerBlock(), 64);
  EXPECT_EQ(params.getNPerBlock(), 64);
  EXPECT_EQ(params.getKpack(), 4);
  EXPECT_EQ(params.getMPerWave(), 32);
  EXPECT_EQ(params.getNPerWave(), 32);
  EXPECT_EQ(params.getMnPerXdl(), 16);
  EXPECT_EQ(params.getSplitKFactor(), 2);
  EXPECT_EQ(params.getScheduleVersion(), 3);
  EXPECT_EQ(params.getOutputSwizzle(), 2);
  EXPECT_EQ(params.getWavesPerEU(), 0);
  EXPECT_EQ(params.getGridGroupSize(), 0);
  EXPECT_EQ(params.getForceUnroll(), true);
}

TEST_F(AccelGemmParamsAttrTest, V4ConfigSecond) {
  auto params = parse("v4:128,64,8,64,32,32,4,9,2,0,8,64,0,1");

  ASSERT_TRUE(params);
  EXPECT_EQ(params.getKpackPerBlock(), 8);
  EXPECT_EQ(params.getMPerBlock(), 128);
  EXPECT_EQ(params.getNPerBlock(), 64);
  EXPECT_EQ(params.getKpack(), 4);
  EXPECT_EQ(params.getMPerWave(), 64);
  EXPECT_EQ(params.getNPerWave(), 32);
  EXPECT_EQ(params.getMnPerXdl(), 32);
  EXPECT_EQ(params.getSplitKFactor(), 9);
  EXPECT_EQ(params.getScheduleVersion(), 2);
  EXPECT_EQ(params.getOutputSwizzle(), 0);
  EXPECT_EQ(params.getWavesPerEU(), 8);
  EXPECT_EQ(params.getGridGroupSize(), 64);
  EXPECT_EQ(params.getForceUnroll(), false);
}

//===----------------------------------------------------------------------===//
// v3 perfconfig tests - MFMA
//===----------------------------------------------------------------------===//

TEST_F(AccelGemmParamsAttrTest, V3ConfigMfmaFirst) {
  auto params = parse("v3:64,64,8,32,16,4,2,3,2,1,1", /*isWmma=*/false);

  ASSERT_TRUE(params);
  EXPECT_EQ(params.getKpackPerBlock(), 8);
  EXPECT_EQ(params.getMPerBlock(), 64);
  EXPECT_EQ(params.getNPerBlock(), 64);
  EXPECT_EQ(params.getKpack(), 4);
  EXPECT_EQ(params.getMPerWave(), 32);
  EXPECT_EQ(params.getNPerWave(), 32);
  EXPECT_EQ(params.getMnPerXdl(), 16);
  EXPECT_EQ(params.getSplitKFactor(), 2);
  EXPECT_EQ(params.getScheduleVersion(), 3);
  EXPECT_EQ(params.getOutputSwizzle(), 2);
  EXPECT_EQ(params.getWavesPerEU(), 0);
  EXPECT_EQ(params.getGridGroupSize(), 0);
  EXPECT_EQ(params.getForceUnroll(), true);
}

TEST_F(AccelGemmParamsAttrTest, V3ConfigMfmaSecond) {
  auto params = parse("v3:128,64,8,64,32,4,9,2,2,0,1", /*isWmma=*/false);

  ASSERT_TRUE(params);
  EXPECT_EQ(params.getKpackPerBlock(), 8);
  EXPECT_EQ(params.getMPerBlock(), 128);
  EXPECT_EQ(params.getNPerBlock(), 64);
  EXPECT_EQ(params.getKpack(), 4);
  EXPECT_EQ(params.getMPerWave(), 64);
  EXPECT_EQ(params.getNPerWave(), 32);
  EXPECT_EQ(params.getMnPerXdl(), 32);
  EXPECT_EQ(params.getSplitKFactor(), 9);
  EXPECT_EQ(params.getScheduleVersion(), 2);
  EXPECT_EQ(params.getOutputSwizzle(), 2);
  EXPECT_EQ(params.getWavesPerEU(), 0);
  EXPECT_EQ(params.getGridGroupSize(), 0);
  EXPECT_EQ(params.getForceUnroll(), false);
}

//===----------------------------------------------------------------------===//
// v3 perfconfig tests - WMMA
//===----------------------------------------------------------------------===//

TEST_F(AccelGemmParamsAttrTest, V3ConfigWmmaFirst) {
  auto params = parse("v3:64,64,8,32,32,4,2,3,2,1,1", /*isWmma=*/true);

  ASSERT_TRUE(params);
  EXPECT_EQ(params.getKpackPerBlock(), 8);
  EXPECT_EQ(params.getMPerBlock(), 64);
  EXPECT_EQ(params.getNPerBlock(), 64);
  EXPECT_EQ(params.getKpack(), 4);
  EXPECT_EQ(params.getMPerWave(), 32);
  EXPECT_EQ(params.getNPerWave(), 32);
  EXPECT_EQ(params.getMnPerXdl(), 16);
  EXPECT_EQ(params.getSplitKFactor(), 2);
  EXPECT_EQ(params.getScheduleVersion(), 3);
  EXPECT_EQ(params.getOutputSwizzle(), 2);
  EXPECT_EQ(params.getWavesPerEU(), 0);
  EXPECT_EQ(params.getGridGroupSize(), 0);
  EXPECT_EQ(params.getForceUnroll(), true);
}

TEST_F(AccelGemmParamsAttrTest, V3ConfigWmmaSecond) {
  auto params = parse("v3:128,64,8,64,32,4,9,2,2,0,1", /*isWmma=*/true);

  ASSERT_TRUE(params);
  EXPECT_EQ(params.getKpackPerBlock(), 8);
  EXPECT_EQ(params.getMPerBlock(), 128);
  EXPECT_EQ(params.getNPerBlock(), 64);
  EXPECT_EQ(params.getKpack(), 4);
  EXPECT_EQ(params.getMPerWave(), 64);
  EXPECT_EQ(params.getNPerWave(), 32);
  EXPECT_EQ(params.getMnPerXdl(), 16);
  EXPECT_EQ(params.getSplitKFactor(), 9);
  EXPECT_EQ(params.getScheduleVersion(), 2);
  EXPECT_EQ(params.getOutputSwizzle(), 2);
  EXPECT_EQ(params.getWavesPerEU(), 0);
  EXPECT_EQ(params.getGridGroupSize(), 0);
  EXPECT_EQ(params.getForceUnroll(), false);
}

//===----------------------------------------------------------------------===//
// Negative tests
//===----------------------------------------------------------------------===//

TEST_F(AccelGemmParamsAttrTest, NoVersionPrefix) {
  auto params = parse("128,64,8,64,32,4,9,2,2,0,1");
  EXPECT_FALSE(params);
}

TEST_F(AccelGemmParamsAttrTest, WrongNumberOfParamsV3) {
  auto params = parse("v3:128,64,8,64,32,4,9,2,2,1");
  EXPECT_FALSE(params);
}

TEST_F(AccelGemmParamsAttrTest, WrongNumberOfParamsV4) {
  auto params = parse("v4:64,64,8,32,32,16,4,4,3,2,1");
  EXPECT_FALSE(params);
}

TEST_F(AccelGemmParamsAttrTest, EmptyString) {
  auto params = parse("");
  EXPECT_FALSE(params);
}

TEST_F(AccelGemmParamsAttrTest, InvalidVersion) {
  auto params = parse("v5:64,64,8,32,32,16,4,2,3,2,0,0,1,1");
  EXPECT_FALSE(params);
}

TEST_F(AccelGemmParamsAttrTest, MalformedInput) {
  auto params = parse("v4:not,valid,numbers");
  EXPECT_FALSE(params);
}

} // end anonymous namespace
