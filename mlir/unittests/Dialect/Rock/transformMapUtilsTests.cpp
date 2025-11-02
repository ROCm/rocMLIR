//===- transformMapUtilsTests.cpp - Tests for Rock transformMapUtils ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::rock;

namespace {

struct TestEnv {
  MLIRContext ctx;
  OpBuilder builder;
  ModuleOp module;

  TestEnv() : builder(&ctx) {
    DialectRegistry reg;
    reg.insert<memref::MemRefDialect, gpu::GPUDialect, RockDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module.getBody());
  }
};

// Helper to create a transformed memref with a simple identity transformation
Value createTransformedMemRef(OpBuilder &b, Location loc,
                              ArrayRef<int64_t> shape, Type elemType,
                              Attribute memorySpace = nullptr) {
  // Create base memref
  auto baseType = MemRefType::get(shape, elemType, nullptr, memorySpace);
  Value base = b.create<memref::AllocOp>(loc, baseType);

  // Create an identity transform to get a transformed value
  BottomUpTMBuilder builder(b, shape, loc);
  SmallVector<StringRef> names;
  builder.getStartNames(names);
  if (!names.empty())
    builder.passThrough(names);
  TransformMapAttr transform = builder.get();

  // Apply transform
  return b.create<TransformOp>(loc, base, transform);
}

//===----------------------------------------------------------------------===//
// addPassThroughIndices Tests
//===----------------------------------------------------------------------===//

// Test: Add extra indices at position 0 (beginning)
TEST(AddPassThroughIndicesTest, AddAtPositionZero) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Create a simple 1D memref<16xf16>
  auto privateSpace =
      gpu::AddressSpaceAttr::get(&env.ctx, gpu::AddressSpace::Private);
  Value transformed =
      createTransformedMemRef(b, loc, {16}, b.getF16Type(), privateSpace);

  // Add 2 dimensions at position 0 with sizes [2, 8]
  Value result = addPassThroughIndices(b, transformed, {2, 8}, 0);

  ASSERT_TRUE(result);
  auto resultType = cast<MemRefType>(result.getType());

  // Result should have shape [2, 8, 16] (new dims prepended)
  EXPECT_EQ(resultType.getRank(), 3);
  EXPECT_EQ(resultType.getShape()[0], 2);
  EXPECT_EQ(resultType.getShape()[1], 8);
  EXPECT_EQ(resultType.getShape()[2], 16);
  EXPECT_TRUE(resultType.getElementType().isF16());
}

// Test: Add extra indices at position 1 (middle)
TEST(AddPassThroughIndicesTest, AddAtPositionMiddle) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Create a 2D memref<8x2xf16>
  auto privateSpace =
      gpu::AddressSpaceAttr::get(&env.ctx, gpu::AddressSpace::Private);
  Value transformed =
      createTransformedMemRef(b, loc, {8, 2}, b.getF16Type(), privateSpace);

  // Add 1 dimension at position 1 with size [4]
  Value result = addPassThroughIndices(b, transformed, {4}, 1);

  ASSERT_TRUE(result);
  auto resultType = cast<MemRefType>(result.getType());

  // Result should have shape [8, 4, 2] (new dim inserted in middle)
  EXPECT_EQ(resultType.getRank(), 3);
  EXPECT_EQ(resultType.getShape()[0], 8);
  EXPECT_EQ(resultType.getShape()[1], 4);
  EXPECT_EQ(resultType.getShape()[2], 2);
}

// Test: Add extra indices at end position
TEST(AddPassThroughIndicesTest, AddAtPositionEnd) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Create a 2D memref<8x2xf16>
  auto privateSpace =
      gpu::AddressSpaceAttr::get(&env.ctx, gpu::AddressSpace::Private);
  Value transformed =
      createTransformedMemRef(b, loc, {8, 2}, b.getF16Type(), privateSpace);

  // Add 2 dimensions at position 2 (end) with sizes [3, 4]
  Value result = addPassThroughIndices(b, transformed, {3, 4}, 2);

  ASSERT_TRUE(result);
  auto resultType = cast<MemRefType>(result.getType());

  // Result should have shape [8, 2, 3, 4] (new dims appended)
  EXPECT_EQ(resultType.getRank(), 4);
  EXPECT_EQ(resultType.getShape()[0], 8);
  EXPECT_EQ(resultType.getShape()[1], 2);
  EXPECT_EQ(resultType.getShape()[2], 3);
  EXPECT_EQ(resultType.getShape()[3], 4);
}

// Test: Add no extra indices (empty array)
TEST(AddPassThroughIndicesTest, AddEmptyIndices) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Create a simple 1D memref<16xf16>
  auto privateSpace =
      gpu::AddressSpaceAttr::get(&env.ctx, gpu::AddressSpace::Private);
  Value transformed =
      createTransformedMemRef(b, loc, {16}, b.getF16Type(), privateSpace);

  // Add 0 dimensions - should return the original value
  Value result = addPassThroughIndices(b, transformed, {}, 0);

  ASSERT_TRUE(result);
  // When no indices are added, it should return the original transformed value
  EXPECT_EQ(result, transformed);
}

// Test: Multi-buffer case - add indices at position 0 for a 2D memref
TEST(AddPassThroughIndicesTest, MultiBufferAtPositionZero) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Create a 2D memref<2x16xf16> (multi-buffer case)
  auto privateSpace =
      gpu::AddressSpaceAttr::get(&env.ctx, gpu::AddressSpace::Private);
  Value transformed =
      createTransformedMemRef(b, loc, {2, 16}, b.getF16Type(), privateSpace);

  // Add extra indices at position 0 with sizes [2]
  Value result = addPassThroughIndices(b, transformed, {2}, 0);

  ASSERT_TRUE(result);
  auto resultType = cast<MemRefType>(result.getType());

  // Result should have shape [2, 2, 16]
  EXPECT_EQ(resultType.getRank(), 3);
  EXPECT_EQ(resultType.getShape()[0], 2);
  EXPECT_EQ(resultType.getShape()[1], 2);
  EXPECT_EQ(resultType.getShape()[2], 16);
}

// Test: Complex case with multiple dimensions at position 0
TEST(AddPassThroughIndicesTest, MultipleDimensionsAtPositionZero) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Create a 3D memref<8x2x4xf32>
  Value transformed =
      createTransformedMemRef(b, loc, {8, 2, 4}, b.getF32Type());

  // Add 3 dimensions at position 0 with sizes [1, 2, 3]
  Value result = addPassThroughIndices(b, transformed, {1, 2, 3}, 0);

  ASSERT_TRUE(result);
  auto resultType = cast<MemRefType>(result.getType());

  // Result should have shape [1, 2, 3, 8, 2, 4]
  EXPECT_EQ(resultType.getRank(), 6);
  EXPECT_EQ(resultType.getShape()[0], 1);
  EXPECT_EQ(resultType.getShape()[1], 2);
  EXPECT_EQ(resultType.getShape()[2], 3);
  EXPECT_EQ(resultType.getShape()[3], 8);
  EXPECT_EQ(resultType.getShape()[4], 2);
  EXPECT_EQ(resultType.getShape()[5], 4);
}

// Test: Verify memory space is preserved
TEST(AddPassThroughIndicesTest, PreservesMemorySpace) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Create with workgroup memory space
  auto workgroupSpace =
      gpu::AddressSpaceAttr::get(&env.ctx, gpu::AddressSpace::Workgroup);
  Value transformed =
      createTransformedMemRef(b, loc, {16}, b.getF16Type(), workgroupSpace);

  Value result = addPassThroughIndices(b, transformed, {2, 8}, 0);

  ASSERT_TRUE(result);
  auto resultType = cast<MemRefType>(result.getType());
  EXPECT_EQ(resultType.getMemorySpace(), workgroupSpace);
}

// Test: Single dimension addition at position 0
TEST(AddPassThroughIndicesTest, SingleDimensionAtPositionZero) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  Value transformed = createTransformedMemRef(b, loc, {16}, b.getF16Type());

  // Add just 1 dimension at position 0
  Value result = addPassThroughIndices(b, transformed, {5}, 0);

  ASSERT_TRUE(result);
  auto resultType = cast<MemRefType>(result.getType());

  EXPECT_EQ(resultType.getRank(), 2);
  EXPECT_EQ(resultType.getShape()[0], 5);
  EXPECT_EQ(resultType.getShape()[1], 16);
}

// Test: Verify transform stack is properly constructed
// This test checks that the resulting value has a valid transform stack
TEST(AddPassThroughIndicesTest, ValidTransformStack) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  Value transformed = createTransformedMemRef(b, loc, {16}, b.getF16Type());

  Value result = addPassThroughIndices(b, transformed, {2, 8}, 0);

  ASSERT_TRUE(result);

  // The result should be a TransformOp
  auto transformOp = result.getDefiningOp<TransformOp>();
  ASSERT_TRUE(transformOp);

  // The transform should have proper bounds
  auto transform = transformOp.getTransform();
  auto upperBounds = transform.getUpperBounds();
  auto lowerBounds = transform.getLowerBounds();

  // Upper bounds should be [2, 8, 16]
  EXPECT_EQ(upperBounds.size(), 3U);
  EXPECT_EQ(upperBounds[0], 2);
  EXPECT_EQ(upperBounds[1], 8);
  EXPECT_EQ(upperBounds[2], 16);

  // The addPassThroughIndices function creates a transform that widens
  // the existing transform stack, so lower bounds include the added dimensions
  EXPECT_EQ(lowerBounds.size(), 3U);
  EXPECT_EQ(lowerBounds[0], 2);
  EXPECT_EQ(lowerBounds[1], 8);
  EXPECT_EQ(lowerBounds[2], 16);
}

} // end anonymous namespace
