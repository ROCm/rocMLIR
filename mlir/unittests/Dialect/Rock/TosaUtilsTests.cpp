//===- TosaUtilsTests.cpp - Tests for Tosa Utils --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/tosaUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct TestEnv {
  MLIRContext ctx;
  OpBuilder builder;
  ModuleOp module;

  TestEnv(bool withFunc = true) : builder(&ctx) {
    DialectRegistry reg;
    reg.insert<mlir::tosa::TosaDialect>();
    if (withFunc)
      reg.insert<func::FuncDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module.getBody());
  }
};

static mlir::tosa::ConstOp buildConst(OpBuilder &b, Location loc,
                                      RankedTensorType type,
                                      double startVal = 0.0) {
  SmallVector<Attribute> elems;
  elems.reserve(type.getNumElements());
  double v = startVal;
  for (int64_t i = 0, e = type.getNumElements(); i < e; ++i, v += 1.0)
    elems.push_back(b.getF32FloatAttr((float)v));
  auto attr = DenseElementsAttr::get(type, elems);
  return mlir::tosa::ConstOp::create(b, loc, type, attr);
}
} // namespace

TEST(TosaUtilsTest, SpecificValueAttribute) {
  TestEnv env;
  auto &ctx = env.ctx;
  OpBuilder b(&ctx);

  auto i32Ty = b.getI32Type();
  auto si8Ty = IntegerType::get(&ctx, 8, IntegerType::Signed);
  auto ui8Ty = IntegerType::get(&ctx, 8, IntegerType::Unsigned);

  // Target == 0.0 fast path (true)
  {
    Attribute zero = IntegerAttr::get(i32Ty, 0);
    EXPECT_TRUE(isSpecificValueAttribute(zero, 0.0));
  }
  // Target == 0.0 fast path (false)
  {
    Attribute five = IntegerAttr::get(i32Ty, 5);
    EXPECT_FALSE(isSpecificValueAttribute(five, 0.0));
  }
  // Exact positive integer match
  {
    Attribute seven = IntegerAttr::get(i32Ty, 7);
    EXPECT_TRUE(isSpecificValueAttribute(seven, 7.0));
  }
  // Positive integer mismatch
  {
    Attribute seven = IntegerAttr::get(i32Ty, 7);
    EXPECT_FALSE(isSpecificValueAttribute(seven, 6.0));
  }
  // Non-integer target (should be false even if floor(value)!=value)
  {
    Attribute five = IntegerAttr::get(i32Ty, 5);
    EXPECT_FALSE(isSpecificValueAttribute(five, 5.5));
  }
  // Negative signed integer match (tests isSigned = true path)
  {
    Attribute neg = IntegerAttr::get(si8Ty, -4);
    EXPECT_TRUE(isSpecificValueAttribute(neg, -4.0));
  }
  // Negative signed integer mismatch
  {
    Attribute neg = IntegerAttr::get(si8Ty, -4);
    EXPECT_FALSE(isSpecificValueAttribute(neg, -5.0));
  }
  // Unsigned integer compare with same positive target
  {
    Attribute u = IntegerAttr::get(ui8Ty, 200);
    EXPECT_TRUE(isSpecificValueAttribute(u, 200.0));
  }
  // Unsigned integer with negative target (should be false)
  {
    Attribute u = IntegerAttr::get(ui8Ty, 3);
    EXPECT_FALSE(isSpecificValueAttribute(u, -3.0));
  }
  // Already existing float cases to ensure no regressions.
  {
    auto f32 = b.getF32Type();
    Attribute fa = b.getFloatAttr(f32, 1.0);
    EXPECT_TRUE(isSpecificValueAttribute(fa, 1.0));
    EXPECT_FALSE(isSpecificValueAttribute(fa, 2.0));
  }
}

// Verifies behavior of positive vs negative zero for both float and integer
// attrs.
TEST(TosaUtilsTest, SpecificValueAttributeSignedZero) {
  TestEnv env;
  OpBuilder b(&env.ctx);

  auto f32Ty = b.getF32Type();

  // Construct +0.0f and -0.0f FloatAttr explicitly.
  Attribute posZeroAttr = b.getFloatAttr(f32Ty, 0.0);
  llvm::APFloat negZeroAp =
      llvm::APFloat::getZero(f32Ty.getFloatSemantics(), /*Negative=*/true);
  Attribute negZeroAttr = b.getFloatAttr(f32Ty, negZeroAp);

  // For FloatAttr, the sign of zero must match the target's sign for a match.
  // We assert that only +0.0 matches target +0.0, and only -0.0 matches target
  // -0.0. We also probe -0.0 target.
  EXPECT_TRUE(isSpecificValueAttribute(posZeroAttr, 0.0));
  EXPECT_FALSE(isSpecificValueAttribute(negZeroAttr, 0.0));

  // Target expressed as -0.0 (bitwise sign in the literal) should only match
  // -0.0, not +0.0.
  double negZeroLiteral = -0.0;
  EXPECT_FALSE(isSpecificValueAttribute(posZeroAttr, negZeroLiteral));
  EXPECT_TRUE(isSpecificValueAttribute(negZeroAttr, negZeroLiteral));

  // Sanity: a non‑zero small value should not match either zero.
  EXPECT_FALSE(isSpecificValueAttribute(posZeroAttr, 1e-6));
  EXPECT_FALSE(isSpecificValueAttribute(negZeroAttr, -1e-6));

  // Integer zero should also match both +0.0 and -0.0 targets.
  auto i32Ty = b.getI32Type();
  Attribute intZero = IntegerAttr::get(i32Ty, 0);
  EXPECT_TRUE(isSpecificValueAttribute(intZero, 0.0));
  EXPECT_TRUE(isSpecificValueAttribute(intZero, negZeroLiteral));
}

// Integer precision loss death test. The chosen double (2^53 + 1) cannot be
// represented exactly; casting to int64_t loses 1 and makes targetInt64 !=
// target
TEST(TosaUtilsTest, SpecificValueAttributeLargeIntegerMismatch) {
  TestEnv env(false);
  OpBuilder &b = env.builder;
  auto i64Ty = b.getI64Type();
  Attribute anyInt = IntegerAttr::get(i64Ty, 0);

  // 2^53 + 1 cannot be exactly represented as int64_t round‑trip from double.
  double badTarget = 9007199254740993.0;
  EXPECT_FALSE(isSpecificValueAttribute(anyInt, badTarget));
}

TEST(TosaUtilsTest, ConstantValuePredicatesScalars) {
  TestEnv env; // func + tosa
  OpBuilder &builder = env.builder;
  Location loc = builder.getUnknownLoc();

  auto funcType = builder.getFunctionType({}, {});
  auto func = func::FuncOp::create(builder, loc, "test", funcType);
  auto &entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  {
    auto tType = RankedTensorType::get({}, builder.getF32Type());
    auto attr = DenseElementsAttr::get(tType, builder.getF32FloatAttr(0.0f));
    auto cst = mlir::tosa::ConstOp::create(builder, loc, tType, attr);
    EXPECT_TRUE(isConstantZero(cst));
    EXPECT_FALSE(isConstantOne(cst));
    EXPECT_TRUE(isConstantValue(cst, 0.0));
  }
  {
    auto tType = RankedTensorType::get({}, builder.getF8E8M0Type());
    auto attr = DenseElementsAttr::get(
        tType, builder.getFloatAttr(builder.getF8E8M0Type(), 1.0f));
    auto cst = mlir::tosa::ConstOp::create(builder, loc, tType, attr);
    EXPECT_FALSE(isConstantZero(cst));
    EXPECT_TRUE(isConstantOne(cst));
    EXPECT_FALSE(isConstantValue(cst, 0.0));
    EXPECT_TRUE(isConstantValue(cst, 1.0));
  }
  {
    auto tType = RankedTensorType::get({}, builder.getF32Type());
    auto attr = DenseElementsAttr::get(tType, builder.getF32FloatAttr(1.0f));
    auto cst = mlir::tosa::ConstOp::create(builder, loc, tType, attr);
    EXPECT_TRUE(isConstantOne(cst));
    EXPECT_FALSE(isConstantZero(cst));
    EXPECT_TRUE(isConstantValue(cst, 1.0));
  }
  {
    auto tType = RankedTensorType::get({}, builder.getI32Type());
    auto attr = DenseElementsAttr::get(tType, builder.getI32IntegerAttr(1));
    auto cst = mlir::tosa::ConstOp::create(builder, loc, tType, attr);
    EXPECT_TRUE(isConstantOne(cst));
    EXPECT_FALSE(isConstantZero(cst));
  }
}

TEST(TosaUtilsTest, ConstantValuePredicatesTensors) {
  TestEnv env;
  OpBuilder &builder = env.builder;
  Location loc = builder.getUnknownLoc();

  auto funcType = builder.getFunctionType({}, {});
  auto func = func::FuncOp::create(builder, loc, "test2", funcType);
  auto &entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  {
    auto tType = RankedTensorType::get({2, 2}, builder.getF16Type());
    SmallVector<Attribute> elems(4, builder.getF16FloatAttr(0.0));
    auto attr = DenseElementsAttr::get(tType, elems);
    auto cst = mlir::tosa::ConstOp::create(builder, loc, tType, attr);
    EXPECT_TRUE(isConstantZero(cst));
    EXPECT_FALSE(isConstantOne(cst));
  }
  {
    auto tType = RankedTensorType::get({3}, builder.getI32Type());
    SmallVector<Attribute> elems(3, builder.getI32IntegerAttr(1));
    auto attr = DenseElementsAttr::get(tType, elems);
    auto cst = mlir::tosa::ConstOp::create(builder, loc, tType, attr);
    EXPECT_TRUE(isConstantOne(cst));
    EXPECT_FALSE(isConstantZero(cst));
  }
  {
    auto tType = RankedTensorType::get({2}, builder.getF32Type());
    SmallVector<Attribute> elems;
    elems.push_back(
        builder.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    elems.push_back(
        builder.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    auto attr = DenseElementsAttr::get(tType, elems);
    auto cst = mlir::tosa::ConstOp::create(builder, loc, tType, attr);
    EXPECT_TRUE(isConstNegInf(cst));
  }
  {
    auto tType = RankedTensorType::get({8}, builder.getI32Type());
    SmallVector<Attribute> elems;
    for (int i = 0; i < 8; ++i)
      elems.push_back(builder.getI32IntegerAttr(i));
    auto attr = DenseElementsAttr::get(tType, elems);
    auto cst = mlir::tosa::ConstOp::create(builder, loc, tType, attr);
    EXPECT_TRUE(isConstRange(cst));
  }
  {
    auto tType = RankedTensorType::get({4}, builder.getI32Type());
    SmallVector<Attribute> elems;
    elems.push_back(builder.getI32IntegerAttr(0));
    elems.push_back(builder.getI32IntegerAttr(2));
    elems.push_back(builder.getI32IntegerAttr(3));
    elems.push_back(builder.getI32IntegerAttr(1));
    auto attr = DenseElementsAttr::get(tType, elems);
    auto cst = mlir::tosa::ConstOp::create(builder, loc, tType, attr);
    EXPECT_FALSE(isConstRange(cst));
  }
}

TEST(TosaUtilsTest, AccTypeSelection) {
  TestEnv env(false); // only tosa
  OpBuilder &b = env.builder;
  EXPECT_TRUE(rock::tosa::getAccType(b, b.getF16Type()).isF32());
  EXPECT_TRUE(rock::tosa::getAccType(b, b.getBF16Type()).isF32());
  EXPECT_TRUE(
      rock::tosa::getAccType(b, IntegerType::get(&env.ctx, 8)).isInteger(32));
}

TEST(TosaUtilsTest, CreateOpAndInferMulHelper) {
  TestEnv env;
  OpBuilder &builder = env.builder;
  Location loc = builder.getUnknownLoc();

  auto funcType = builder.getFunctionType({}, {});
  auto func = func::FuncOp::create(builder, loc, "test3", funcType);
  auto &entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  auto tType = RankedTensorType::get({2, 2}, builder.getF32Type());
  SmallVector<Attribute> elemsA(4, builder.getF32FloatAttr(2.0f));
  SmallVector<Attribute> elemsB(4, builder.getF32FloatAttr(3.0f));
  auto aAttr = DenseElementsAttr::get(tType, elemsA);
  auto bAttr = DenseElementsAttr::get(tType, elemsB);
  auto aConst = mlir::tosa::ConstOp::create(builder, loc, tType, aAttr);
  auto bConst = mlir::tosa::ConstOp::create(builder, loc, tType, bAttr);

  auto mulOp = rock::tosa::getMulOp(builder, loc, aConst.getResult(),
                                    bConst.getResult(), builder.getF32Type());
  auto resType = cast<RankedTensorType>(mulOp.getResult().getType());
  EXPECT_EQ(resType.getShape().size(), 2u);
  EXPECT_EQ(resType.getDimSize(0), 2);
  EXPECT_EQ(resType.getDimSize(1), 2);
  EXPECT_TRUE(resType.getElementType().isF32());
}

TEST(TosaTransposeUtilsTest, Basic3DTranspose) {
  TestEnv env;
  OpBuilder &b = env.builder;
  auto srcType = RankedTensorType::get({2, 3, 4}, b.getF32Type());
  auto c0 = buildConst(b, b.getUnknownLoc(), srcType);
  SmallVector<int32_t> perm = {2, 1, 0};
  auto tx =
      rock::tosa::getTransposeOp(b, b.getUnknownLoc(), c0.getResult(), perm);
  auto resType = cast<RankedTensorType>(tx.getResult().getType());
  ASSERT_EQ(resType.getRank(), 3);
  EXPECT_EQ(resType.getDimSize(0), 4);
  EXPECT_EQ(resType.getDimSize(1), 3);
  EXPECT_EQ(resType.getDimSize(2), 2);
  auto permsAttr = tx->getAttrOfType<DenseI32ArrayAttr>("perms");
  ASSERT_TRUE(permsAttr);
  ASSERT_EQ(permsAttr.size(), 3u);
  EXPECT_EQ(permsAttr[0], 2);
  EXPECT_EQ(permsAttr[1], 1);
  EXPECT_EQ(permsAttr[2], 0);
}

TEST(TosaTransposeUtilsTest, IdentityPermutation) {
  TestEnv env(false); // only tosa
  OpBuilder &b = env.builder;
  auto srcType = RankedTensorType::get({5, 6}, b.getF32Type());
  auto c0 = buildConst(b, b.getUnknownLoc(), srcType);
  SmallVector<int32_t> perm = {0, 1};
  auto tx =
      rock::tosa::getTransposeOp(b, b.getUnknownLoc(), c0.getResult(), perm);
  auto resType = cast<RankedTensorType>(tx.getResult().getType());
  EXPECT_EQ(resType.getShape(), ArrayRef<int64_t>({5, 6}));
}

TEST(TosaTransposeUtilsTest, FourDPermutation) {
  TestEnv env(false);
  OpBuilder &b = env.builder;
  auto srcType = RankedTensorType::get({1, 8, 16, 32}, b.getF32Type());
  auto c0 = buildConst(b, b.getUnknownLoc(), srcType);
  SmallVector<int32_t> perm = {0, 2, 3, 1};
  auto tx =
      rock::tosa::getTransposeOp(b, b.getUnknownLoc(), c0.getResult(), perm);
  auto resType = cast<RankedTensorType>(tx.getResult().getType());
  ASSERT_EQ(resType.getRank(), 4);
  EXPECT_EQ(resType.getDimSize(0), 1);
  EXPECT_EQ(resType.getDimSize(1), 16);
  EXPECT_EQ(resType.getDimSize(2), 32);
  EXPECT_EQ(resType.getDimSize(3), 8);
}
