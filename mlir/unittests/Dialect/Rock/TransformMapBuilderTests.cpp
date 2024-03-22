//===- TransformMapBuilderTests.cpp - Tests for the Rock Transform
// Map Builder -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"
#include "gtest/internal/gtest-internal.h"

using namespace mlir;
using namespace mlir::rock;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class TMBuilderTest : public ::testing::Test {
protected:
  TMBuilderTest() : b(&context) { context.getOrLoadDialect<RockDialect>(); }

  TopDownTMBuilder makeTopDown(ArrayRef<StringRef> upperNames,
                               ArrayRef<int64_t> upperShape) {
    return TopDownTMBuilder(b, upperNames, upperShape, b.getUnknownLoc());
  }

  BottomUpTMBuilder makeBottomUp(ArrayRef<StringRef> lowerNames,
                                 ArrayRef<int64_t> lowerShape) {
    return BottomUpTMBuilder(b, lowerNames, lowerShape, b.getUnknownLoc());
  }

  AffineExpr affD(uint32_t d) { return b.getAffineDimExpr(d); }

  AffineExpr affC(int64_t c) { return b.getAffineConstantExpr(c); }

protected:
  MLIRContext context;
  Builder b;
};

#define EXPECT_ARRAY_EQ(T, t1, t2)                                             \
  do {                                                                         \
    const ArrayRef<T> a1 = t1;                                                 \
    const ArrayRef<T> a2 = t2;                                                 \
    EXPECT_EQ(a1.size(), a2.size())                                            \
        << "Array sizes unequal while comparing " #t1 " and " #t2 "\n";        \
    size_t e = std::min(a1.size(), a2.size());                                 \
    for (size_t i = 0; i < e; ++i) {                                           \
      EXPECT_EQ(a1[i], a2[i])                                                  \
          << " while comparing element " << i << " of " #t1 " and " #t2 "\n";  \
    }                                                                          \
  } while (0)

TEST_F(TMBuilderTest, PassThroughDoesNothingTD) {
  auto buildDown = makeTopDown({"x"}, {4});
  buildDown.passThrough("x");
  TransformMapAttr res = buildDown.get();

  SmallVector<int64_t> bounds = {4};
  EXPECT_ARRAY_EQ(int64_t, res.getUpperBounds(), bounds);
  EXPECT_ARRAY_EQ(int64_t, res.getLowerBounds(), bounds);
  EXPECT_EQ(res.getMap().getAffineMap(), b.getDimIdentityMap());
  ArrayRef<TransformAttr> ops = res.getOps();
  ASSERT_EQ(ops.size(), 1UL);
  EXPECT_EQ(ops[0], TransformAttr::get(&context, TransformType::PassThrough, {},
                                       {"x"}, {0}, {"x"}, {0}));
}

TEST_F(TMBuilderTest, PassThroughWorksBothWays) {
  auto buildUp = makeBottomUp({"b"}, {4});
  auto buildDown = makeTopDown({"a"}, {4});

  buildUp.passThrough("a", "b");
  buildDown.passThrough("b", "a");

  TransformMapAttr resUp = buildUp.get();
  TransformMapAttr resDown = buildDown.get();
  ASSERT_EQ(resUp, resDown);
}

TEST_F(TMBuilderTest, PassThroughExchange) {
  auto buildUp = makeBottomUp({"x", "y"}, {1, 2});

  buildUp.passThrough({"y"}, {0}, {"y"});
  buildUp.passThrough({"x"}, {1}, {"x"});
  TransformMapAttr resUp = buildUp.get();

  EXPECT_EQ(resUp.getMap().getAffineMap(),
            AffineMap::get(2, 0, {affD(1), affD(0)}, &context));
  llvm::SmallVector<int64_t> upperShape = {2, 1};
  EXPECT_ARRAY_EQ(int64_t, resUp.getUpperBounds(), upperShape);

  auto buildDown = makeTopDown({"y", "x"}, upperShape);
  buildDown.passThrough({"x", "y"}, {1, 0}, {"y", "x"});

  auto resDown = buildDown.get();
  EXPECT_EQ(resDown.getMap().getAffineMap(), resUp.getMap().getAffineMap());
  EXPECT_ARRAY_EQ(int64_t, resDown.getLowerBounds(), resUp.getLowerBounds());
  EXPECT_ARRAY_EQ(int64_t, resDown.getUpperBounds(), resUp.getUpperBounds());
}

TEST_F(TMBuilderTest, Padding) {
  llvm::SmallVector<int64_t> upperBounds = {7, 5};
  llvm::SmallVector<int64_t> lowerBounds = {4, 4};

  auto buildUp = makeBottomUp({"a", "b"}, lowerBounds);
  buildUp.pad({"a", "b"}, {2, 1, 1, 0});
  TransformMapAttr resUp = buildUp.get();
  EXPECT_ARRAY_EQ(int64_t, resUp.getUpperBounds(), upperBounds);
  EXPECT_EQ(
      resUp.getMap().getAffineMap(),
      AffineMap::get(2, 0, {affD(0) - affC(2), affD(1) - affC(1)}, &context));

  auto buildDown = makeTopDown({"a", "b"}, upperBounds);
  buildDown.pad({"a", "b"}, {2, 1, 1, 0});
  TransformMapAttr resDown = buildDown.get();
  EXPECT_ARRAY_EQ(int64_t, resDown.getLowerBounds(), lowerBounds);
  EXPECT_EQ(resUp, resDown);
}

TEST_F(TMBuilderTest, Embed) {
  auto buildDown = makeTopDown({"a", "b", "c"}, {2, 3, 4});
  auto buildUp = makeBottomUp({"a"}, {24});

  buildDown.embed("a", 0, 24, {"a", "b", "c"}, {6, 2, 1});
  buildUp.embed({"a", "b", "c"}, {0, 1, 2}, {2, 3, 4}, "a", {6, 2, 1});

  TransformMapAttr resDown = buildDown.get();
  TransformMapAttr resUp = buildUp.get();

  EXPECT_EQ(
      resDown.getMap().getAffineMap(),
      AffineMap::get(3, 0, {affD(0) * affC(6) + affD(1) * affC(2) + affD(2)}));
  EXPECT_EQ(resDown, resUp);
}

TEST_F(TMBuilderTest, Unmerge) {
  auto buildDown = makeTopDown({"a", "b", "c"}, {2, 3, 4});
  auto buildUp = makeBottomUp({"a"}, {24});

  buildDown.unmerge("a", 0, {"a", "b", "c"}, {2, 3, 4});
  buildUp.unmerge({"a", "b", "c"}, {0, 1, 2}, "a", {2, 3, 4});

  TransformMapAttr resDown = buildDown.get();
  TransformMapAttr resUp = buildUp.get();

  EXPECT_EQ(resDown.getMap().getAffineMap(),
            AffineMap::get(3, 0,
                           {(affD(0) * affC(3) + affD(1)) * affC(4) + affD(2)},
                           &context));
  EXPECT_EQ(resDown, resUp);
}

TEST_F(TMBuilderTest, Merge) {
  auto buildDown = makeTopDown({"top"}, {30});
  auto buildUp = makeBottomUp({"x", "y", "z"}, {2, 3, 5});

  buildDown.merge({"x", "y", "z"}, {0, 1, 2}, "top", {2, 3, 5});
  buildUp.merge("top", 0, {"x", "y", "z"});

  TransformMapAttr resDown = buildDown.get();
  TransformMapAttr resUp = buildUp.get();

  EXPECT_EQ(
      resDown.getMap().getAffineMap(),
      AffineMap::get(1, 0,
                     {affD(0).floorDiv(affC(15)),
                      (affD(0) % affC(15)).floorDiv(5), affD(0) % affC(5)},
                     &context));
  EXPECT_EQ(resDown, resUp);
}

TEST_F(TMBuilderTest, AddDim) {
  auto buildUp = makeBottomUp({"a", "c"}, {1, 3});
  auto buildDown = makeTopDown({"a", "b", "c"}, {1, 2, 3});

  buildDown.passThrough({"a", "c"}, {0, 1}, {"a", "c"});
  buildUp.passThrough({"a", "c"}, {0, 2}, {"a", "c"});

  buildDown.ignore("b");
  buildUp.addDim("b", 1, 2);

  TransformMapAttr resDown = buildDown.get();
  TransformMapAttr resUp = buildUp.get();

  EXPECT_EQ(resUp.getMap().getAffineMap(),
            AffineMap::get(3, 0, {affD(0), affD(2)}, &context));
  EXPECT_EQ(resUp, resDown);
}

TEST_F(TMBuilderTest, ConstDim) {
  auto buildDown = makeTopDown({}, {});

  buildDown.constDim("a", 0, 0, 1);
  buildDown.constDim({"b", "c"}, {1, 2}, {1, 2}, {2, 3});

  TransformMapAttr resDown = buildDown.get();

  EXPECT_EQ(resDown.getMap().getAffineMap(),
            AffineMap::get(0, 0, {affC(0), affC(1), affC(2)}, &context));
  SmallVector<int64_t> expectedLowerBounds = {1, 2, 3};
  EXPECT_ARRAY_EQ(int64_t, resDown.getLowerBounds(), expectedLowerBounds);
}

TEST_F(TMBuilderTest, Broadcast) {
  auto buildUp = makeBottomUp({"a", "b"}, {1, 3});
  auto buildDown = makeTopDown({"a", "b"}, {3, 3});

  buildUp.passThrough({1}, {1});
  buildUp.broadcast({0}, {3});

  buildDown.passThrough({1}, {1});
  buildDown.takeRemainder("a", 1);

  TransformMapAttr resUp = buildUp.get();
  TransformMapAttr resDown = buildDown.get();

  EXPECT_EQ(resUp.getMap().getAffineMap(),
            AffineMap::get(2, 0, {affC(0), affD(1)}, &context));
  EXPECT_EQ(resUp, resDown);
}

TEST_F(TMBuilderTest, LongBroadcast) {
  auto buildUp = makeBottomUp({"a", "b"}, {2, 3});
  auto buildDown = makeTopDown({"a", "b"}, {3, 3});

  buildUp.passThrough({1}, {1});
  buildUp.broadcast({0}, {3});

  buildDown.passThrough({1}, {1});
  buildDown.takeRemainder("a", 2);

  TransformMapAttr resUp = buildUp.get();
  TransformMapAttr resDown = buildDown.get();

  EXPECT_EQ(resUp.getMap().getAffineMap(),
            AffineMap::get(2, 0, {affD(0) % affC(2), affD(1)}, &context));
  EXPECT_EQ(resUp, resDown);

}

TEST_F(TMBuilderTest, GemmOut) {
  auto buildDown =
      makeTopDown({"gemmG", "gemmM", "gemmN"}, {1, 64, 32 * 14 * 14});
  auto buildUp = makeBottomUp({"n", "g", "k", "h", "w"}, {32, 1, 64, 14, 14});

  buildDown.passThrough({"g"}, {1}, {"gemmG"});
  buildDown.passThrough({"k"}, {2}, {"gemmM"});
  buildDown.merge({"n", "h", "w"}, {0, 3, 4}, "gemmN", {32, 14, 14});

  buildUp.passThrough({"gemmG"}, {0}, {"g"});
  buildUp.passThrough({"gemmM"}, {1}, {"k"});
  buildUp.merge("gemmN", 2, {"n", "h", "w"});

  TransformMapAttr resDown = buildDown.get();
  TransformMapAttr resUp = buildUp.get();

  EXPECT_EQ(
      resDown.getMap().getAffineMap(),
      AffineMap::get(3, 0,
                     {affD(2).floorDiv(affC(196)), affD(0), affD(1),
                      (affD(2) % affC(196)).floorDiv(affC(14)), affD(2) % 14},
                     &context));
  EXPECT_EQ(resDown, resUp);
}
