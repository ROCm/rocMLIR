//===- loweringUtilsTests.cpp - Tests for Rock loweringUtils --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
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
    reg.insert<arith::ArithDialect, memref::MemRefDialect, gpu::GPUDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module.getBody());
  }
};

// Helper to create a test i8 buffer
Value createI8Buffer(OpBuilder &b, Location loc, int64_t numBytes,
                     Attribute memorySpace = nullptr) {
  auto bufferType =
      MemRefType::get({numBytes}, b.getI8Type(), nullptr, memorySpace);
  return b.create<memref::AllocOp>(loc, bufferType);
}

//===----------------------------------------------------------------------===//
// Positive Tests - Scalar Types
//===----------------------------------------------------------------------===//

TEST(ViewBufferAsTest, Float16Scalar) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // float16 is 16 bits = 2 bytes, so we need 2 bytes for 1 element
  Value buffer = createI8Buffer(b, loc, 2);
  auto result = viewBufferAs(b, buffer, b.getF16Type());

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 1);
  EXPECT_EQ(resultType.getShape()[0], 1);
  EXPECT_TRUE(resultType.getElementType().isF16());
}

TEST(ViewBufferAsTest, Int8Scalar) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // int8 is 8 bits = 1 byte
  Value buffer = createI8Buffer(b, loc, 1);
  auto result = viewBufferAs(b, buffer, b.getI8Type());

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 1);
  EXPECT_EQ(resultType.getShape()[0], 1);
  EXPECT_TRUE(resultType.getElementType().isInteger(8));
}

TEST(ViewBufferAsTest, Float32Scalar) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // float32 is 32 bits = 4 bytes
  Value buffer = createI8Buffer(b, loc, 4);
  auto result = viewBufferAs(b, buffer, b.getF32Type());

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 1);
  EXPECT_EQ(resultType.getShape()[0], 1);
  EXPECT_TRUE(resultType.getElementType().isF32());
}

TEST(ViewBufferAsTest, Float4E2M1FNScalar) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Float4E2M1FN is 4 bits, so 2 elements fit in 1 byte
  Value buffer = createI8Buffer(b, loc, 1);
  auto f4Type = Float4E2M1FNType::get(&env.ctx);
  auto result = viewBufferAs(b, buffer, f4Type);

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 1);
  EXPECT_EQ(resultType.getShape()[0], 2);
  EXPECT_TRUE(isa<Float4E2M1FNType>(resultType.getElementType()));
}

TEST(ViewBufferAsTest, Float6E2M3FNScalar) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Float6E2M3FN is 6 bits
  // 24 bits (3 bytes) = 4 elements of 6 bits each
  Value buffer = createI8Buffer(b, loc, 3);
  auto f6Type = Float6E2M3FNType::get(&env.ctx);
  auto result = viewBufferAs(b, buffer, f6Type);

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 1);
  EXPECT_EQ(resultType.getShape()[0], 4);
  EXPECT_TRUE(isa<Float6E2M3FNType>(resultType.getElementType()));
}

TEST(ViewBufferAsTest, Float8E8M0FNUScalar) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Float8E8M0FNU is 8 bits = 1 byte
  Value buffer = createI8Buffer(b, loc, 1);
  auto f8Type = Float8E8M0FNUType::get(&env.ctx);
  auto result = viewBufferAs(b, buffer, f8Type);

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 1);
  EXPECT_EQ(resultType.getShape()[0], 1);
  EXPECT_TRUE(isa<Float8E8M0FNUType>(resultType.getElementType()));
}

TEST(ViewBufferAsTest, Float8E4M3FNScalar) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Float8E4M3FN is 8 bits = 1 byte
  Value buffer = createI8Buffer(b, loc, 1);
  auto f8Type = Float8E4M3FNType::get(&env.ctx);
  auto result = viewBufferAs(b, buffer, f8Type);

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 1);
  EXPECT_EQ(resultType.getShape()[0], 1);
  EXPECT_TRUE(isa<Float8E4M3FNType>(resultType.getElementType()));
}

//===----------------------------------------------------------------------===//
// Positive Tests - Scalar Types with Custom Dimensions
//===----------------------------------------------------------------------===//

TEST(ViewBufferAsTest, Float16ScalarWithDimensions) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // 4 elements of float16 = 4 * 2 = 8 bytes
  Value buffer = createI8Buffer(b, loc, 8);
  auto result = viewBufferAs(b, buffer, b.getF16Type(), {4});

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 1);
  EXPECT_EQ(resultType.getShape()[0], 4);
  EXPECT_TRUE(resultType.getElementType().isF16());
}

TEST(ViewBufferAsTest, Float32MultiDimensional) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // 2x3 tensor of float32 = 6 elements * 4 bytes = 24 bytes
  Value buffer = createI8Buffer(b, loc, 24);
  auto result = viewBufferAs(b, buffer, b.getF32Type(), {2, 3});

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 2);
  EXPECT_EQ(resultType.getShape()[0], 2);
  EXPECT_EQ(resultType.getShape()[1], 3);
  EXPECT_TRUE(resultType.getElementType().isF32());
}

TEST(ViewBufferAsTest, Int8MultiDimensional) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // 4x4 tensor of int8 = 16 elements * 1 byte = 16 bytes
  Value buffer = createI8Buffer(b, loc, 16);
  auto result = viewBufferAs(b, buffer, b.getI8Type(), {4, 4});

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 2);
  EXPECT_EQ(resultType.getShape()[0], 4);
  EXPECT_EQ(resultType.getShape()[1], 4);
  EXPECT_TRUE(resultType.getElementType().isInteger(8));
}

TEST(ViewBufferAsTest, Float4E2M1FNMultiDimensional) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // 8x8 tensor of f4E2M1FN = 64 elements * 4 bits = 256 bits = 32 bytes
  Value buffer = createI8Buffer(b, loc, 32);
  auto f4Type = Float4E2M1FNType::get(&env.ctx);
  auto result = viewBufferAs(b, buffer, f4Type, {8, 8});

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 2);
  EXPECT_EQ(resultType.getShape()[0], 8);
  EXPECT_EQ(resultType.getShape()[1], 8);
  EXPECT_TRUE(isa<Float4E2M1FNType>(resultType.getElementType()));
}

//===----------------------------------------------------------------------===//
// Positive Tests - Vector Types
//===----------------------------------------------------------------------===//

TEST(ViewBufferAsTest, Float16Vector) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Vector<4xf16> = 4 elements * 16 bits = 64 bits = 8 bytes per vector
  // 2 vectors = 16 bytes
  Value buffer = createI8Buffer(b, loc, 16);
  auto vecType = VectorType::get({4}, b.getF16Type());
  auto result = viewBufferAs(b, buffer, vecType);

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 1);
  EXPECT_EQ(resultType.getShape()[0], 2);
  auto elemType = resultType.getElementType();
  EXPECT_TRUE(isa<VectorType>(elemType));
  auto vecElemType = cast<VectorType>(elemType);
  EXPECT_EQ(vecElemType.getShape()[0], 4);
  EXPECT_TRUE(vecElemType.getElementType().isF16());
}

TEST(ViewBufferAsTest, Float32Vector) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Vector<8xf32> = 8 elements * 32 bits = 256 bits = 32 bytes per vector
  Value buffer = createI8Buffer(b, loc, 32);
  auto vecType = VectorType::get({8}, b.getF32Type());
  auto result = viewBufferAs(b, buffer, vecType);

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 1);
  EXPECT_EQ(resultType.getShape()[0], 1);
  auto elemType = resultType.getElementType();
  EXPECT_TRUE(isa<VectorType>(elemType));
  auto vecElemType = cast<VectorType>(elemType);
  EXPECT_EQ(vecElemType.getShape()[0], 8);
  EXPECT_TRUE(vecElemType.getElementType().isF32());
}

TEST(ViewBufferAsTest, Int8Vector) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Vector<16xi8> = 16 elements * 8 bits = 128 bits = 16 bytes per vector
  Value buffer = createI8Buffer(b, loc, 16);
  auto vecType = VectorType::get({16}, b.getI8Type());
  auto result = viewBufferAs(b, buffer, vecType);

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 1);
  EXPECT_EQ(resultType.getShape()[0], 1);
  auto elemType = resultType.getElementType();
  EXPECT_TRUE(isa<VectorType>(elemType));
  auto vecElemType = cast<VectorType>(elemType);
  EXPECT_EQ(vecElemType.getShape()[0], 16);
  EXPECT_TRUE(vecElemType.getElementType().isInteger(8));
}

TEST(ViewBufferAsTest, Float4E2M1FNVector) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Vector<2xf4E2M1FN> = 2 elements * 4 bits = 8 bits = 1 byte per vector
  // 4 vectors = 4 bytes
  Value buffer = createI8Buffer(b, loc, 4);
  auto f4Type = Float4E2M1FNType::get(&env.ctx);
  auto vecType = VectorType::get({2}, f4Type);
  auto result = viewBufferAs(b, buffer, vecType);

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 1);
  EXPECT_EQ(resultType.getShape()[0], 4);
  auto elemType = resultType.getElementType();
  EXPECT_TRUE(isa<VectorType>(elemType));
  auto vecElemType = cast<VectorType>(elemType);
  EXPECT_EQ(vecElemType.getShape()[0], 2);
  EXPECT_TRUE(isa<Float4E2M1FNType>(vecElemType.getElementType()));
}

TEST(ViewBufferAsTest, Float8E4M3FNVector) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Vector<4xf8E4M3FN> = 4 elements * 8 bits = 32 bits = 4 bytes per vector
  Value buffer = createI8Buffer(b, loc, 8);
  auto f8Type = Float8E4M3FNType::get(&env.ctx);
  auto vecType = VectorType::get({4}, f8Type);
  auto result = viewBufferAs(b, buffer, vecType);

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 1);
  EXPECT_EQ(resultType.getShape()[0], 2);
  auto elemType = resultType.getElementType();
  EXPECT_TRUE(isa<VectorType>(elemType));
  auto vecElemType = cast<VectorType>(elemType);
  EXPECT_EQ(vecElemType.getShape()[0], 4);
  EXPECT_TRUE(isa<Float8E4M3FNType>(vecElemType.getElementType()));
}

TEST(ViewBufferAsTest, Float8E8M0FNUVector) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Vector<8xf8E8M0FNU> = 8 elements * 8 bits = 64 bits = 8 bytes per vector
  Value buffer = createI8Buffer(b, loc, 16);
  auto f8Type = Float8E8M0FNUType::get(&env.ctx);
  auto vecType = VectorType::get({8}, f8Type);
  auto result = viewBufferAs(b, buffer, vecType);

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 1);
  EXPECT_EQ(resultType.getShape()[0], 2);
  auto elemType = resultType.getElementType();
  EXPECT_TRUE(isa<VectorType>(elemType));
  auto vecElemType = cast<VectorType>(elemType);
  EXPECT_EQ(vecElemType.getShape()[0], 8);
  EXPECT_TRUE(isa<Float8E8M0FNUType>(vecElemType.getElementType()));
}

//===----------------------------------------------------------------------===//
// Positive Tests - Vector Types with Custom Dimensions
//===----------------------------------------------------------------------===//

TEST(ViewBufferAsTest, Float16VectorWithDimensions) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // 2x3 tensor of Vector<4xf16>
  // = 6 vectors * (4 elements * 16 bits) = 6 * 64 bits = 384 bits = 48 bytes
  Value buffer = createI8Buffer(b, loc, 48);
  auto vecType = VectorType::get({4}, b.getF16Type());
  auto result = viewBufferAs(b, buffer, vecType, {2, 3});

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 2);
  EXPECT_EQ(resultType.getShape()[0], 2);
  EXPECT_EQ(resultType.getShape()[1], 3);
  auto elemType = resultType.getElementType();
  EXPECT_TRUE(isa<VectorType>(elemType));
  auto vecElemType = cast<VectorType>(elemType);
  EXPECT_EQ(vecElemType.getShape()[0], 4);
  EXPECT_TRUE(vecElemType.getElementType().isF16());
}

//===----------------------------------------------------------------------===//
// Positive Tests - Memory Spaces
//===----------------------------------------------------------------------===//

TEST(ViewBufferAsTest, GPUWorkgroupMemorySpace) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  auto workgroupSpace =
      gpu::AddressSpaceAttr::get(&env.ctx, gpu::AddressSpace::Workgroup);
  Value buffer = createI8Buffer(b, loc, 8, workgroupSpace);
  auto result = viewBufferAs(b, buffer, b.getF16Type(), {4});

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 1);
  EXPECT_EQ(resultType.getShape()[0], 4);
  EXPECT_TRUE(resultType.getElementType().isF16());
  EXPECT_EQ(resultType.getMemorySpace(), workgroupSpace);
}

TEST(ViewBufferAsTest, GPUPrivateMemorySpace) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  auto privateSpace =
      gpu::AddressSpaceAttr::get(&env.ctx, gpu::AddressSpace::Private);
  Value buffer = createI8Buffer(b, loc, 4, privateSpace);
  auto result = viewBufferAs(b, buffer, b.getF32Type());

  auto resultType = result.getType();
  EXPECT_EQ(resultType.getRank(), 1);
  EXPECT_EQ(resultType.getShape()[0], 1);
  EXPECT_TRUE(resultType.getElementType().isF32());
  EXPECT_EQ(resultType.getMemorySpace(), privateSpace);
}

// Death tests rely on assert which is disabled in release mode.
#ifndef NDEBUG

//===----------------------------------------------------------------------===//
// Negative Tests - Assertion Failures
//===----------------------------------------------------------------------===//

TEST(ViewBufferAsDeathTest, BufferNotRank1) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Create a 2D i8 buffer (should fail - must be 1D)
  auto badBufferType = MemRefType::get({4, 4}, b.getI8Type());
  Value badBuffer = b.create<memref::AllocOp>(loc, badBufferType);

  EXPECT_DEATH(
      { viewBufferAs(b, badBuffer, b.getF32Type()); },
      "Buffer type must be a 1D memref for viewBufferAs");
}

TEST(ViewBufferAsDeathTest, BufferNotI8) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Create a 1D f32 buffer (should fail - must be i8)
  auto badBufferType = MemRefType::get({4}, b.getF32Type());
  Value badBuffer = b.create<memref::AllocOp>(loc, badBufferType);

  EXPECT_DEATH(
      { viewBufferAs(b, badBuffer, b.getF32Type()); },
      "Buffer type must be a i8 memref for viewBufferAs");
}

TEST(ViewBufferAsDeathTest, BufferSizeMismatch) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // 3 bytes cannot evenly fit float32 (which needs 4 bytes)
  Value buffer = createI8Buffer(b, loc, 3);

  EXPECT_DEATH(
      { viewBufferAs(b, buffer, b.getF32Type()); },
      "Can't evenly fit type into buffer");
}

TEST(ViewBufferAsDeathTest, DimensionsMismatch) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // 8 bytes = 2 float32 elements, but we're trying to create 3 elements
  // This will fail the bufferBitWidth check since 8*8=64 bits != 3*32=96 bits
  Value buffer = createI8Buffer(b, loc, 8);

  EXPECT_DEATH(
      { viewBufferAs(b, buffer, b.getF32Type(), {3}); },
      "Can't evenly fit type into buffer");
}

TEST(ViewBufferAsDeathTest, VectorBufferSizeMismatch) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Vector<4xf32> = 16 bytes, but buffer is only 15 bytes
  Value buffer = createI8Buffer(b, loc, 15);
  auto vecType = VectorType::get({4}, b.getF32Type());

  EXPECT_DEATH(
      { viewBufferAs(b, buffer, vecType); },
      "Can't evenly fit type into buffer");
}

TEST(ViewBufferAsDeathTest, VectorDimensionsMismatch) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // 16 bytes can fit 1 vector<4xf32> (16 bytes), not 2
  Value buffer = createI8Buffer(b, loc, 16);
  auto vecType = VectorType::get({4}, b.getF32Type());

  EXPECT_DEATH(
      { viewBufferAs(b, buffer, vecType, {2}); },
      "Can't evenly fit type into buffer");
}

TEST(ViewBufferAsDeathTest, Float6BufferMismatch) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // 2 bytes = 16 bits, but Float6E2M3FN needs multiples of 6 bits
  // 16 is not evenly divisible by 6
  Value buffer = createI8Buffer(b, loc, 2);
  auto f6Type = Float6E2M3FNType::get(&env.ctx);

  EXPECT_DEATH(
      { viewBufferAs(b, buffer, f6Type); },
      "Can't evenly fit type into buffer");
}

#endif // NDEBUG

} // end anonymous namespace
