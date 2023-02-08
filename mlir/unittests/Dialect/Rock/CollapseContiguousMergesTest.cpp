//===- CollapseContiguousMergesTest.cpp - ------------------ -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"
#include "gtest/internal/gtest-internal.h"

using namespace mlir;
using namespace mlir::rock;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class CollapseMergeTest : public ::testing::Test {
protected:
  CollapseMergeTest() : b(&context) { context.getOrLoadDialect<RockDialect>(); }

  TopDownTMBuilder tmBuilder(ArrayRef<StringRef> upperNames,
                             ArrayRef<int64_t> upperShape) {
    return TopDownTMBuilder(b, upperNames, upperShape, b.getUnknownLoc());
  }

  AffineExpr affD(uint32_t d) { return b.getAffineDimExpr(d); }

  AffineExpr affC(int64_t c) { return b.getAffineConstantExpr(c); }

  ArrayAttr toArr(TransformMapAttr tmap) { return b.getArrayAttr({tmap}); }

protected:
  MLIRContext context;
  Builder b;
};

TEST_F(CollapseMergeTest, UnfoldLike) {
  auto tmb = tmBuilder({"a", "x"}, {5, 24});
  tmb.passThrough({"a"}, {3}, {"a"});
  tmb.merge({"b", "c", "d"}, {0, 1, 2}, "x", {4, 3, 2});
  TransformMapAttr map = tmb.get();
  ArrayAttr arr = toArr(map);
  // Pass in wrong shape, does nothing.
  EXPECT_EQ(collapseContiguousMerges(arr, {5, 4, 3, 5}), arr);

  ArrayAttr collapsedArr = collapseContiguousMerges(arr, {4, 3, 2, 5});
  auto colMap = collapsedArr[0].cast<TransformMapAttr>();
  // Order of transform map attributes should be preserved, we're relying on
  // that instead of looking for the merge. If that stops being true in the
  // future, change this over not doing the change.
  EXPECT_EQ(colMap.getOps()[1].getParams()[0], 1);
  EXPECT_EQ(colMap.getOps()[1].getParams()[1], 1);
  EXPECT_EQ(colMap.getOps()[1].getParams()[2], 24);
  EXPECT_EQ(
      colMap.getMap().getAffineMap(),
      AffineMap::get(2, 0, {affC(0), affC(0), affD(1), affD(0)}, &context));
}

TEST_F(CollapseMergeTest, PartialLike) {
  auto tmb = tmBuilder({"gemmM", "gemmN"}, {5, 24});
  tmb.passThrough({"k"}, {1}, {"gemmM"});
  tmb.merge({"n", "h", "w"}, {0, 2, 3}, "gemmN", {4, 3, 2});
  TransformMapAttr map = tmb.get();
  ArrayAttr arr = toArr(map);
  ArrayAttr collapsedArr = collapseContiguousMerges(arr, {4, 5, 3, 2});
  auto colMap = collapsedArr[0].cast<TransformMapAttr>();
  EXPECT_EQ(colMap.getOps()[1].getParams()[0], 4);
  EXPECT_EQ(colMap.getOps()[1].getParams()[1], 1);
  EXPECT_EQ(colMap.getOps()[1].getParams()[2], 6);
  EXPECT_EQ(colMap.getMap().getAffineMap(),
            AffineMap::get(2, 0,
                           {affD(1).floorDiv(affC(6)), affD(0), affC(0),
                            affD(1) % affC(6)},
                           &context));
}
