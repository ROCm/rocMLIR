//===- CoordTransformBuilderTests.cpp - Tests for the MIOpen Coordinate Transform Builder -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"
#include "gtest/internal/gtest-internal.h"

using namespace mlir;
using namespace mlir::miopen;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class CTBuilderTest : public ::testing::Test {
protected:
  CTBuilderTest() : b(&context) { context.getOrLoadDialect<MIOpenDialect>(); }

  TopDownCTBuilder makeTopDown(ArrayRef<StringRef> upperNames,
                               ArrayRef<int64_t> upperShape) {
    return TopDownCTBuilder(b, upperNames, upperShape, b.getUnknownLoc());
  }

  BottomUpCTBuilder getBottomUp(ArrayRef<StringRef> lowerNames,
                                ArrayRef<int64_t> lowerShape) {
    return BottomUpCTBuilder(b, lowerNames, lowerShape, b.getUnknownLoc());
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

TEST_F(CTBuilderTest, PassThroughDoesNothingTD) {
  auto ctb = makeTopDown({"x"}, {4});
  ctb.passThrough("x");
  TransformsAttr res = ctb.get();

  SmallVector<int64_t> bounds = {4};
  EXPECT_ARRAY_EQ(int64_t, res.getUpperBounds(), bounds);
  EXPECT_ARRAY_EQ(int64_t, res.getLowerBounds(), bounds);
  EXPECT_EQ(res.getMap().getAffineMap(), b.getDimIdentityMap());
  ArrayRef<TransformAttr> ops = res.getOps();
  ASSERT_EQ(ops.size(), 1UL);
  EXPECT_EQ(ops[0], TransformAttr::get(&context, TransformType::PassThrough, {},
                                       {"x"}, {0}, {"x"}, {0}));
}
