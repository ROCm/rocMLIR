#include "mlir/Dialect/Rock/utility/tosaUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
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
    reg.insert<tosa::TosaDialect>();
    if (withFunc)
      reg.insert<func::FuncDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module.getBody());
  }
};

static tosa::ConstOp buildConst(OpBuilder &b, Location loc,
                                RankedTensorType type, double startVal = 0.0) {
  SmallVector<Attribute> elems;
  elems.reserve(type.getNumElements());
  double v = startVal;
  for (int64_t i = 0, e = type.getNumElements(); i < e; ++i, v += 1.0)
    elems.push_back(b.getF32FloatAttr((float)v));
  auto attr = DenseElementsAttr::get(type, elems);
  return b.create<tosa::ConstOp>(loc, type, attr);
}
} // namespace

TEST(TosaUtilsTest, SpecificValueAttribute) {
  TestEnv env; // with func + tosa
  auto &ctx = env.ctx;
  auto f32Ty = Float32Type::get(&ctx);
  Attribute a = FloatAttr::get(f32Ty, 1.0);
  EXPECT_TRUE(isSpecificValueAttribute(a, 1.0));
  EXPECT_FALSE(isSpecificValueAttribute(a, 0.0));
  Attribute posZero = FloatAttr::get(f32Ty, 0.0);
  EXPECT_TRUE(isSpecificValueAttribute(posZero, 0.0));
  EXPECT_FALSE(isSpecificValueAttribute(posZero, -0.0));

  Attribute b = IntegerAttr::get(IntegerType::get(&ctx, 32), 7);
  EXPECT_TRUE(isSpecificValueAttribute(b, 7.0));
  EXPECT_FALSE(isSpecificValueAttribute(b, 6.0));
  Attribute zero = IntegerAttr::get(IntegerType::get(&ctx, 32), 0);
  EXPECT_TRUE(isSpecificValueAttribute(zero, 0.0));
  EXPECT_TRUE(isSpecificValueAttribute(zero, -0.0));
}

TEST(TosaUtilsTest, ConstantValuePredicatesScalars) {
  TestEnv env; // func + tosa
  OpBuilder &builder = env.builder;
  Location loc = builder.getUnknownLoc();

  auto funcType = builder.getFunctionType({}, {});
  auto func =
      builder.create<func::FuncOp>(loc, "test", funcType);
  auto &entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  {
    auto tType = RankedTensorType::get({}, builder.getF32Type());
    auto attr = DenseElementsAttr::get(tType, builder.getF32FloatAttr(0.0f));
    auto cst = builder.create<tosa::ConstOp>(loc, tType, attr);
    EXPECT_TRUE(isConstantZero(cst));
    EXPECT_FALSE(isConstantOne(cst));
    EXPECT_TRUE(isConstantValue(cst, 0.0));
  }
  {
    auto tType = RankedTensorType::get({}, builder.getF8E8M0Type());
    auto attr = DenseElementsAttr::get(tType, builder.getFloatAttr(
                                             builder.getF8E8M0Type(), 1.0f));
    auto cst = builder.create<tosa::ConstOp>(loc, tType, attr);
    EXPECT_FALSE(isConstantZero(cst));
    EXPECT_TRUE(isConstantOne(cst));
    EXPECT_FALSE(isConstantValue(cst, 0.0));
    EXPECT_TRUE(isConstantValue(cst, 1.0));
  }
  {
    auto tType = RankedTensorType::get({}, builder.getF32Type());
    auto attr = DenseElementsAttr::get(tType, builder.getF32FloatAttr(1.0f));
    auto cst = builder.create<tosa::ConstOp>(loc, tType, attr);
    EXPECT_TRUE(isConstantOne(cst));
    EXPECT_FALSE(isConstantZero(cst));
    EXPECT_TRUE(isConstantValue(cst, 1.0));
  }
  {
    auto tType = RankedTensorType::get({}, builder.getI32Type());
    auto attr = DenseElementsAttr::get(tType, builder.getI32IntegerAttr(1));
    auto cst = builder.create<tosa::ConstOp>(loc, tType, attr);
    EXPECT_TRUE(isConstantOne(cst));
    EXPECT_FALSE(isConstantZero(cst));
  }
}

TEST(TosaUtilsTest, ConstantValuePredicatesTensors) {
  TestEnv env;
  OpBuilder &builder = env.builder;
  Location loc = builder.getUnknownLoc();

  auto funcType = builder.getFunctionType({}, {});
  auto func =
      builder.create<func::FuncOp>(loc, "test2", funcType);
  auto &entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  {
    auto tType = RankedTensorType::get({2, 2}, builder.getF16Type());
    SmallVector<Attribute> elems(4, builder.getF16FloatAttr(0.0));
    auto attr = DenseElementsAttr::get(tType, elems);
    auto cst = builder.create<tosa::ConstOp>(loc, tType, attr);
    EXPECT_TRUE(isConstantZero(cst));
    EXPECT_FALSE(isConstantOne(cst));
  }
  {
    auto tType = RankedTensorType::get({3}, builder.getI32Type());
    SmallVector<Attribute> elems(3, builder.getI32IntegerAttr(1));
    auto attr = DenseElementsAttr::get(tType, elems);
    auto cst = builder.create<tosa::ConstOp>(loc, tType, attr);
    EXPECT_TRUE(isConstantOne(cst));
    EXPECT_FALSE(isConstantZero(cst));
  }
  {
    auto tType = RankedTensorType::get({2}, builder.getF32Type());
    SmallVector<Attribute> elems;
    elems.push_back(builder.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    elems.push_back(builder.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    auto attr = DenseElementsAttr::get(tType, elems);
    auto cst = builder.create<tosa::ConstOp>(loc, tType, attr);
    EXPECT_TRUE(isConstNegInf(cst));
  }
  {
    auto tType = RankedTensorType::get({8}, builder.getI32Type());
    SmallVector<Attribute> elems;
    for (int i = 0; i < 8; ++i)
      elems.push_back(builder.getI32IntegerAttr(i));
    auto attr = DenseElementsAttr::get(tType, elems);
    auto cst = builder.create<tosa::ConstOp>(loc, tType, attr);
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
    auto cst = builder.create<tosa::ConstOp>(loc, tType, attr);
    EXPECT_FALSE(isConstRange(cst));
  }
}

TEST(TosaUtilsTest, AccTypeSelection) {
  TestEnv env(false); // only tosa
  OpBuilder &b = env.builder;
  EXPECT_TRUE(getTosaAccType(b, b.getF16Type()).isF32());
  EXPECT_TRUE(getTosaAccType(b, b.getBF16Type()).isF32());
  EXPECT_TRUE(getTosaAccType(b, IntegerType::get(&env.ctx, 8)).isInteger(32));
}

TEST(TosaUtilsTest, CreateOpAndInferMulHelper) {
  TestEnv env;
  OpBuilder &builder = env.builder;
  Location loc = builder.getUnknownLoc();

  auto funcType = builder.getFunctionType({}, {});
  auto func =
      builder.create<func::FuncOp>(loc, "test3", funcType);
  auto &entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  auto tType = RankedTensorType::get({2, 2}, builder.getF32Type());
  SmallVector<Attribute> elemsA(4, builder.getF32FloatAttr(2.0f));
  SmallVector<Attribute> elemsB(4, builder.getF32FloatAttr(3.0f));
  auto aAttr = DenseElementsAttr::get(tType, elemsA);
  auto bAttr = DenseElementsAttr::get(tType, elemsB);
  auto aConst = builder.create<tosa::ConstOp>(loc, tType, aAttr);
  auto bConst = builder.create<tosa::ConstOp>(loc, tType, bAttr);

  auto mulOp =
      getTosaMulOp(builder, loc, aConst.getResult(), bConst.getResult(),
                   builder.getF32Type());
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
  auto tx = getTosaTransposeOp(b, b.getUnknownLoc(), c0.getResult(), perm);
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
  auto tx = getTosaTransposeOp(b, b.getUnknownLoc(), c0.getResult(), perm);
  auto resType = cast<RankedTensorType>(tx.getResult().getType());
  EXPECT_EQ(resType.getShape(), ArrayRef<int64_t>({5, 6}));
}

TEST(TosaTransposeUtilsTest, FourDPermutation) {
  TestEnv env(false);
  OpBuilder &b = env.builder;
  auto srcType = RankedTensorType::get({1, 8, 16, 32}, b.getF32Type());
  auto c0 = buildConst(b, b.getUnknownLoc(), srcType);
  SmallVector<int32_t> perm = {0, 2, 3, 1};
  auto tx = getTosaTransposeOp(b, b.getUnknownLoc(), c0.getResult(), perm);
  auto resType = cast<RankedTensorType>(tx.getResult().getType());
  ASSERT_EQ(resType.getRank(), 4);
  EXPECT_EQ(resType.getDimSize(0), 1);
  EXPECT_EQ(resType.getDimSize(1), 16);
  EXPECT_EQ(resType.getDimSize(2), 32);
  EXPECT_EQ(resType.getDimSize(3), 8);
}
