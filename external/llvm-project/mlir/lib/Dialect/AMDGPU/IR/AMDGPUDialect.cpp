//===- AMDGPUDialect.cpp - MLIR AMDGPU dialect implementation --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AMDGPU dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/AMDGPUDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::amdgpu;

#include "mlir/Dialect/AMDGPU/AMDGPUDialect.cpp.inc"

void AMDGPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/AMDGPU/AMDGPU.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/AMDGPU/AMDGPUAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// RawBuffer*Op
//===----------------------------------------------------------------------===//
template <typename T>
static LogicalResult verifyRawBufferOp(T &op) {
  MemRefType bufferType = op.memref().getType().template cast<MemRefType>();
  if (bufferType.getMemorySpaceAsInt() != 0)
    return op.emitOpError(
        "Buffer ops must operate on a memref in global memory");
  if (!bufferType.hasRank())
    return op.emitOpError(
        "Cannot meaningfully buffer_store to an unranked memref");
  if (static_cast<int64_t>(op.indices().size()) != bufferType.getRank())
    return op.emitOpError("Expected " + Twine(bufferType.getRank()) +
                          " indices to memref");
  return success();
}

LogicalResult RawBufferLoadOp::verify() { return verifyRawBufferOp(*this); }

LogicalResult RawBufferStoreOp::verify() { return verifyRawBufferOp(*this); }

LogicalResult RawBufferAtomicFaddOp::verify() {
  return verifyRawBufferOp(*this);
}

//===----------------------------------------------------------------------===//
// MFMAOp
//===----------------------------------------------------------------------===//
LogicalResult MFMAOp::verify() {
  Builder b(getOperation());
  StringRef instrName = stringifyMFMAInstr(instr());

  Type inType = sourceA().getType();
  switch (instr()) {
  case MFMAInstr::f32_32x32x1f32:
  case MFMAInstr::f32_16x16x1f32:
  case MFMAInstr::f32_4x4x1f32:
  case MFMAInstr::f32_32x32x2f32:
  case MFMAInstr::f32_16x16x4f32:
    if (inType != b.getF32Type())
      return emitOpError(instrName + " requires f32 inputs");
    break;
  case MFMAInstr::f32_32x32x4f16:
  case MFMAInstr::f32_16x16x4f16:
  case MFMAInstr::f32_4x4x4f16:
  case MFMAInstr::f32_32x32x8f16:
  case MFMAInstr::f32_16x16x16f16:
    if (inType != VectorType::get(4, b.getF16Type()))
      return emitOpError(instrName + " requires vector<4xf16> inputs");
    break;
  case MFMAInstr::i32_32x32x4i8:
  case MFMAInstr::i32_16x16x4i8:
  case MFMAInstr::i32_4x4x4i8:
  case MFMAInstr::i32_32x32x8i8:
  case MFMAInstr::i32_16x16x16i8:
    if (inType != b.getI32Type() && inType != VectorType::get(4, b.getI8Type()))
      return emitOpError(instrName + " requires i32 or vector<4xi8> inputs");
    break;
  case MFMAInstr::f32_32x32x2bf16:
  case MFMAInstr::f32_16x16x2bf16:
  case MFMAInstr::f32_4x4x2bf16:
  case MFMAInstr::f32_32x32x4bf16:
  case MFMAInstr::f32_16x16x8bf16:
    if (inType != VectorType::get(2, b.getBF16Type()))
      return emitOpError(instrName + " requires vector<2xbf16> inputs");
    break;
  case MFMAInstr::f32_32x32x4bf16_1k:
  case MFMAInstr::f32_16x16x4bf16_1k:
  case MFMAInstr::f32_4x4x4bf16_1k:
  case MFMAInstr::f32_32x32x8bf16_1k:
  case MFMAInstr::f32_16x16x16bf16_1k:
    if (inType != VectorType::get(4, b.getBF16Type()))
      return emitOpError(instrName + " requires vector<4xbf16> inputs");
    break;
  case MFMAInstr::f64_16x16x4f64:
  case MFMAInstr::f64_4x4x4f64:
    if (inType != b.getF64Type())
      return emitOpError(instrName + " requires f64 inputs");
    break;
  case MFMAInstr::i32_16x16x32_i8:
  case MFMAInstr::i32_32x32x16_i8:
    if (inType != b.getI64Type() && inType != VectorType::get(8, b.getI8Type()))
      return emitOpError(instrName + " requires i64 or vector<8xi8> inputs");
    break;
  case MFMAInstr::f32_16x16x8_xf32:
  case MFMAInstr::f32_32x32x4_xf32:
    if (inType != VectorType::get(2, b.getF32Type()))
      return emitOpError(instrName + " requires vector<2xf32> inputs");
    break;
  }

  Type outType = destC().getType();
  switch (instr()) {
  case MFMAInstr::f32_32x32x1f32:
  case MFMAInstr::f32_32x32x4f16:
  case MFMAInstr::f32_32x32x2bf16:
  case MFMAInstr::f32_32x32x4bf16_1k:
    if (outType != VectorType::get(32, b.getF32Type()))
      return emitOpError(instrName + " must have vector<32xf32> outputs");
    break;
  case MFMAInstr::f32_16x16x1f32:
  case MFMAInstr::f32_32x32x2f32:
  case MFMAInstr::f32_16x16x4f16:
  case MFMAInstr::f32_32x32x8f16:
  case MFMAInstr::f32_16x16x2bf16:
  case MFMAInstr::f32_32x32x4bf16:
  case MFMAInstr::f32_16x16x4bf16_1k:
  case MFMAInstr::f32_32x32x8bf16_1k:
  case MFMAInstr::f32_32x32x4_xf32:
    if (outType != VectorType::get(16, b.getF32Type()))
      return emitOpError(instrName + " must have vector<16xf32> outputs");
    break;
  case MFMAInstr::f32_4x4x1f32:
  case MFMAInstr::f32_16x16x4f32:
  case MFMAInstr::f32_4x4x4f16:
  case MFMAInstr::f32_16x16x16f16:
  case MFMAInstr::f32_4x4x2bf16:
  case MFMAInstr::f32_16x16x8bf16:
  case MFMAInstr::f32_4x4x4bf16_1k:
  case MFMAInstr::f32_16x16x16bf16_1k:
  case MFMAInstr::f32_16x16x8_xf32:
    if (outType != VectorType::get(4, b.getF32Type()))
      return emitOpError(instrName + " must have vector<4xf32> outputs");
    break;
  case MFMAInstr::i32_32x32x4i8:

    if (outType != VectorType::get(32, b.getI32Type()))
      return emitOpError(instrName + " must have vector<32xi32> outputs");
    break;
  case MFMAInstr::i32_16x16x4i8:
  case MFMAInstr::i32_32x32x8i8:
  case MFMAInstr::i32_32x32x16_i8:
    if (outType != VectorType::get(16, b.getI32Type()))
      return emitOpError(instrName + " must have vector<16xi32> outputs");
    break;
  case MFMAInstr::i32_4x4x4i8:
  case MFMAInstr::i32_16x16x16i8:
  case MFMAInstr::i32_16x16x32_i8:
    if (outType != VectorType::get(4, b.getI32Type()))
      return emitOpError(instrName + " must have vector<4xi32> outputs");
    break;
  case MFMAInstr::f64_16x16x4f64:
    if (outType != VectorType::get(4, b.getF64Type()))
      return emitOpError(instrName + " must have vector<4xf64> outputs");
    break;
  case MFMAInstr::f64_4x4x4f64:
    if (outType != b.getF64Type())
      return emitOpError(instrName + " must have f64 outputs");
  }
  return success();
}

#include "mlir/Dialect/AMDGPU/AMDGPUEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/AMDGPU/AMDGPUAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/AMDGPU/AMDGPU.cpp.inc"
