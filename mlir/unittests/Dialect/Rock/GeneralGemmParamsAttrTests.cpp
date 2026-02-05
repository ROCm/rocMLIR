//===- GeneralGemmParamsAttrTests.cpp - Tests for GeneralGemmParamsAttr ---===//
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

class GeneralGemmParamsAttrTest : public ::testing::Test {
protected:
  void SetUp() override {
    ctx.loadDialect<RockDialect>();
    builder = std::make_unique<OpBuilder>(&ctx);
  }

  GeneralGemmParamsAttr parse(StringRef perfConfig) {
    return GeneralGemmParamsAttr::get(builder->getStringAttr(perfConfig));
  }

  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
};

//===----------------------------------------------------------------------===//
// v3 perfconfig tests
//===----------------------------------------------------------------------===//

TEST_F(GeneralGemmParamsAttrTest, V3ConfigFirst) {
  auto params = parse("v3:64,32,32,4,2,4,1,1,2");

  ASSERT_TRUE(params);
  EXPECT_EQ(params.getBlockSize(), 64);
  EXPECT_EQ(params.getMPerBlock(), 32);
  EXPECT_EQ(params.getNPerBlock(), 32);
  EXPECT_EQ(params.getKPerBlock(), 4);
  EXPECT_EQ(params.getMPerThread(), 2);
  EXPECT_EQ(params.getNPerThread(), 4);
  EXPECT_EQ(params.getKPerThread(), 1);
  EXPECT_EQ(params.getSplitKFactor(), 1);
  EXPECT_EQ(params.getScheduleVersion(), 1);
  EXPECT_EQ(params.getOutputSwizzle(), 2);
  EXPECT_EQ(params.getKpack(), 1);
}

TEST_F(GeneralGemmParamsAttrTest, V3ConfigSecond) {
  auto params = parse("v3:128,64,32,8,4,2,3,1,2");

  ASSERT_TRUE(params);
  EXPECT_EQ(params.getBlockSize(), 128);
  EXPECT_EQ(params.getMPerBlock(), 64);
  EXPECT_EQ(params.getNPerBlock(), 32);
  EXPECT_EQ(params.getKPerBlock(), 8);
  EXPECT_EQ(params.getMPerThread(), 4);
  EXPECT_EQ(params.getNPerThread(), 2);
  EXPECT_EQ(params.getKPerThread(), 1);
  EXPECT_EQ(params.getSplitKFactor(), 3);
  EXPECT_EQ(params.getScheduleVersion(), 1);
  EXPECT_EQ(params.getOutputSwizzle(), 2);
  EXPECT_EQ(params.getKpack(), 1);
}

//===----------------------------------------------------------------------===//
// Negative tests
//===----------------------------------------------------------------------===//

TEST_F(GeneralGemmParamsAttrTest, NoVersionPrefix) {
  auto params = parse("128,64,8,64,32,4,9,2,2,0,1");
  EXPECT_FALSE(params);
}

TEST_F(GeneralGemmParamsAttrTest, WrongNumberOfParamsV3) {
  auto params = parse("v3:64,32,32,4,2,4,1,1");
  EXPECT_FALSE(params);
}

TEST_F(GeneralGemmParamsAttrTest, EmptyString) {
  auto params = parse("");
  EXPECT_FALSE(params);
}

TEST_F(GeneralGemmParamsAttrTest, InvalidVersion) {
  auto params = parse("v5:64,32,32,4,2,4,1,1,2");
  EXPECT_FALSE(params);
}

TEST_F(GeneralGemmParamsAttrTest, MalformedInput) {
  auto params = parse("v3:not,valid,numbers");
  EXPECT_FALSE(params);
}

} // end anonymous namespace
