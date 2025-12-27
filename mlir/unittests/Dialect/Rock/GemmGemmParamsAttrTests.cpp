//===- GemmGemmParamsAttrTests.cpp - Tests for GemmGemmParamsAttr
//-----------------===//
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

class GemmGemmParamsAttrTest : public ::testing::Test {
protected:
  void SetUp() override {
    ctx.loadDialect<RockDialect>();
    builder = std::make_unique<OpBuilder>(&ctx);
  }

  GemmGemmParamsAttr parse(StringRef perfConfig, bool isWmma = false) {
    return GemmGemmParamsAttr::get(builder->getStringAttr(perfConfig), isWmma);
  }

  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
};

//===----------------------------------------------------------------------===//
// v3 perfconfig tests
//===----------------------------------------------------------------------===//

TEST_F(GemmGemmParamsAttrTest, V3ConfigFirst) {
  auto params = parse("attn:v3:64,64,64,8,32,32,16,4,1,3,2,0,1");

  ASSERT_TRUE(params);
  EXPECT_EQ(params.getMPerBlockG0(), 64);
  EXPECT_EQ(params.getMPerBlockG1(), 64);
  EXPECT_EQ(params.getNPerBlockG0(), 64);
  EXPECT_EQ(params.getKpackPerBlock(), 8);
  EXPECT_EQ(params.getMPerWave(), 32);
  EXPECT_EQ(params.getNPerWave(), 32);
  EXPECT_EQ(params.getMnPerXdl(), 16);
  EXPECT_EQ(params.getKpack(), 4);
  EXPECT_EQ(params.getSplitKFactor(), 1);
  EXPECT_EQ(params.getScheduleVersion(), 3);
  EXPECT_EQ(params.getOutputSwizzle(), 2);
  EXPECT_EQ(params.getWavesPerEU(), 0);
  EXPECT_EQ(params.getForceUnroll(), true);
}

TEST_F(GemmGemmParamsAttrTest, V3ConfigSecond) {
  auto params = parse("attn:v3:128,64,64,8,64,32,32,4,2,2,0,8,0");

  ASSERT_TRUE(params);
  EXPECT_EQ(params.getMPerBlockG0(), 128);
  EXPECT_EQ(params.getMPerBlockG1(), 64);
  EXPECT_EQ(params.getNPerBlockG0(), 64);
  EXPECT_EQ(params.getKpackPerBlock(), 8);
  EXPECT_EQ(params.getMPerWave(), 64);
  EXPECT_EQ(params.getNPerWave(), 32);
  EXPECT_EQ(params.getMnPerXdl(), 32);
  EXPECT_EQ(params.getKpack(), 4);
  EXPECT_EQ(params.getSplitKFactor(), 2);
  EXPECT_EQ(params.getScheduleVersion(), 2);
  EXPECT_EQ(params.getOutputSwizzle(), 0);
  EXPECT_EQ(params.getWavesPerEU(), 8);
  EXPECT_EQ(params.getForceUnroll(), false);
}

//===----------------------------------------------------------------------===//
// v2 perfconfig tests - MFMA
//===----------------------------------------------------------------------===//

TEST_F(GemmGemmParamsAttrTest, V2ConfigMfmaFirst) {
  auto params = parse("attn:v2:64,64,64,8,32,16,4,1,3,2,1", /*isWmma=*/false);

  ASSERT_TRUE(params);
  EXPECT_EQ(params.getMPerBlockG0(), 64);
  EXPECT_EQ(params.getMPerBlockG1(), 64);
  EXPECT_EQ(params.getNPerBlockG0(), 64);
  EXPECT_EQ(params.getKpackPerBlock(), 8);
  EXPECT_EQ(params.getMPerWave(), 32);
  EXPECT_EQ(params.getNPerWave(), 32);
  EXPECT_EQ(params.getMnPerXdl(), 16);
  EXPECT_EQ(params.getKpack(), 4);
  EXPECT_EQ(params.getSplitKFactor(), 1);
  EXPECT_EQ(params.getScheduleVersion(), 3);
  EXPECT_EQ(params.getOutputSwizzle(), 2);
  EXPECT_EQ(params.getWavesPerEU(), 0);
  EXPECT_EQ(params.getForceUnroll(), true);
}

TEST_F(GemmGemmParamsAttrTest, V2ConfigMfmaSecond) {
  auto params = parse("attn:v2:128,64,64,8,64,32,4,2,2,0,0", /*isWmma=*/false);

  ASSERT_TRUE(params);
  EXPECT_EQ(params.getMPerBlockG0(), 128);
  EXPECT_EQ(params.getMPerBlockG1(), 64);
  EXPECT_EQ(params.getNPerBlockG0(), 64);
  EXPECT_EQ(params.getKpackPerBlock(), 8);
  EXPECT_EQ(params.getMPerWave(), 64);
  EXPECT_EQ(params.getNPerWave(), 32);
  EXPECT_EQ(params.getMnPerXdl(), 32);
  EXPECT_EQ(params.getKpack(), 4);
  EXPECT_EQ(params.getSplitKFactor(), 2);
  EXPECT_EQ(params.getScheduleVersion(), 2);
  EXPECT_EQ(params.getOutputSwizzle(), 0);
  EXPECT_EQ(params.getWavesPerEU(), 0);
  EXPECT_EQ(params.getForceUnroll(), false);
}

//===----------------------------------------------------------------------===//
// v2 perfconfig tests - WMMA
//===----------------------------------------------------------------------===//

TEST_F(GemmGemmParamsAttrTest, V2ConfigWmmaFirst) {
  auto params = parse("attn:v2:64,64,64,8,32,32,4,1,3,2,1", /*isWmma=*/true);

  ASSERT_TRUE(params);
  EXPECT_EQ(params.getMPerBlockG0(), 64);
  EXPECT_EQ(params.getMPerBlockG1(), 64);
  EXPECT_EQ(params.getNPerBlockG0(), 64);
  EXPECT_EQ(params.getKpackPerBlock(), 8);
  EXPECT_EQ(params.getMPerWave(), 32);
  EXPECT_EQ(params.getNPerWave(), 32);
  EXPECT_EQ(params.getMnPerXdl(), 16);
  EXPECT_EQ(params.getKpack(), 4);
  EXPECT_EQ(params.getSplitKFactor(), 1);
  EXPECT_EQ(params.getScheduleVersion(), 3);
  EXPECT_EQ(params.getOutputSwizzle(), 2);
  EXPECT_EQ(params.getWavesPerEU(), 0);
  EXPECT_EQ(params.getForceUnroll(), true);
}

TEST_F(GemmGemmParamsAttrTest, V2ConfigWmmaSecond) {
  auto params = parse("attn:v2:128,64,64,8,64,32,4,2,2,0,0", /*isWmma=*/true);

  ASSERT_TRUE(params);
  EXPECT_EQ(params.getMPerBlockG0(), 128);
  EXPECT_EQ(params.getMPerBlockG1(), 64);
  EXPECT_EQ(params.getNPerBlockG0(), 64);
  EXPECT_EQ(params.getKpackPerBlock(), 8);
  EXPECT_EQ(params.getMPerWave(), 64);
  EXPECT_EQ(params.getNPerWave(), 32);
  EXPECT_EQ(params.getMnPerXdl(), 16);
  EXPECT_EQ(params.getKpack(), 4);
  EXPECT_EQ(params.getSplitKFactor(), 2);
  EXPECT_EQ(params.getScheduleVersion(), 2);
  EXPECT_EQ(params.getOutputSwizzle(), 0);
  EXPECT_EQ(params.getWavesPerEU(), 0);
  EXPECT_EQ(params.getForceUnroll(), false);
}

//===----------------------------------------------------------------------===//
// Negative tests
//===----------------------------------------------------------------------===//

TEST_F(GemmGemmParamsAttrTest, NoPrefix) {
  auto params = parse("v3:64,64,64,8,32,32,16,4,1,3,2,0,1");
  EXPECT_FALSE(params);
}

TEST_F(GemmGemmParamsAttrTest, WrongPrefix) {
  auto params = parse("gemm:v3:64,64,64,8,32,32,16,4,1,3,2,0,1");
  EXPECT_FALSE(params);
}

TEST_F(GemmGemmParamsAttrTest, WrongNumberOfParamsV2) {
  auto params = parse("attn:v2:64,64,64,8,32,16,4,1,3,2");
  EXPECT_FALSE(params);
}

TEST_F(GemmGemmParamsAttrTest, WrongNumberOfParamsV3) {
  auto params = parse("attn:v3:64,64,64,8,32,32,16,4,1,3,2,0");
  EXPECT_FALSE(params);
}

TEST_F(GemmGemmParamsAttrTest, EmptyString) {
  auto params = parse("");
  EXPECT_FALSE(params);
}

TEST_F(GemmGemmParamsAttrTest, InvalidVersion) {
  auto params = parse("attn:v5:64,64,64,8,32,32,16,4,1,3,2,0,1");
  EXPECT_FALSE(params);
}

TEST_F(GemmGemmParamsAttrTest, MalformedInput) {
  auto params = parse("attn:v3:not,valid,numbers");
  EXPECT_FALSE(params);
}

} // end anonymous namespace
