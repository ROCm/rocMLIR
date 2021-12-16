//===- MIOpenOps.cpp - MIOpen MLIR Operations -----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::miopen;

#include "mlir/Dialect/MIOpen/MIOpenOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// MIOpenDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {

} // namespace

namespace mlir {
namespace miopen {
Optional<ConvOpType> getConvOpTypeForName(const StringRef name) {
  if (name == "conv2d") {
    return Conv2DOpType;
  }
  if (name == "conv2d_bwd_data") {
    return Conv2DBwdDataOpType;
  }
  if (name == "conv2d_bwd_weight") {
    return Conv2DBwdWeightOpType;
  }
  return llvm::None;
}

const char *getNameForConvOpType(const miopen::ConvOpType op) {
  switch (op) {
  case Conv2DOpType:
    return "conv2d";
  case Conv2DBwdDataOpType:
    return "conv2d_bwd_data";
  case Conv2DBwdWeightOpType:
    return "conv2d_bwd_weight";
  }
  llvm_unreachable("Invalid ConvOp type");
}
} // namespace miopen
} // namespace mlir
//===----------------------------------------------------------------------===//
// MIOpenDialect
//===----------------------------------------------------------------------===//

void MIOpenDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MIOpen/MIOpenOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Convolution operations
//===----------------------------------------------------------------------===//
template <typename T>
static LogicalResult verifyConvOp(T op) {
  auto isDisjointed = [&](llvm::StringRef tensor, llvm::StringRef dim1,
                          llvm::StringRef dim2) {
    auto layout = op->getAttr(tensor).template cast<ArrayAttr>().getValue();
    auto pos1 = -1, pos2 = -1;
    for (unsigned int i = 0; i < layout.size(); ++i) {
      if (layout[i].template cast<StringAttr>().getValue() == dim1)
        pos1 = i;
      if (layout[i].template cast<StringAttr>().getValue() == dim2)
        pos2 = i;
    }

    if ((pos2 != pos1 + 1) && (pos1 != pos2 + 1))
      return true;
    else
      return false;
  };

  if (isDisjointed("filter_layout", "y", "x") ||
      isDisjointed("input_layout", "hi", "wi"))
    return op.emitError("Disjointed yx or hw!");
  else
    return success();
}

// Utility static member function of TransformOp to populate an ArrayAttr to
// track the bounds of a MemRefType.
ArrayAttr TransformOp::buildMemRefShapeAttr(OpBuilder &b,
                                            MemRefType memRefType) {
  auto shape = memRefType.getShape();
  SmallVector<Attribute> shapeAttr;
  for (auto s : shape)
    shapeAttr.push_back(b.getI32IntegerAttr(s));
  return b.getArrayAttr(shapeAttr);
}


//===----------------------------------------------------------------------===//
// ThreadwiseCopyOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ThreadwiseCopyOp op) {
  auto sourceCoord = op.sourceCoord();
  auto destCoord = op.destCoord();
  auto sourceType = op.source().getType().cast<MemRefType>();
  auto destType = op.dest().getType().cast<MemRefType>();
  auto sourceRank = sourceType.getRank();
  auto destRank = destType.getRank();
  auto sourceAffineMap = sourceType.getLayout().getAffineMap();
  auto destAffineMap = destType.getLayout().getAffineMap();

  unsigned expectedSourceCoords = sourceRank;
  unsigned expectedDestCoords = destRank;

  // check if memrefs have embedded affine maps.
  expectedSourceCoords = sourceAffineMap.getNumInputs();
  expectedDestCoords = destAffineMap.getNumInputs();

  // check if memrefs have externally defined affine maps.
  auto coordTransformAttrs = op->getAttr("coord_transforms");
  if (coordTransformAttrs) {
    for (auto coordTransformAttr :
         coordTransformAttrs.cast<ArrayAttr>().getValue()) {
      auto coordTransformDictAttr = coordTransformAttr.cast<DictionaryAttr>();
      auto operandIndex =
          coordTransformDictAttr.get("operand").cast<IntegerAttr>().getInt();
      auto affineMapsArrayAttr =
          coordTransformDictAttr.get("transforms").cast<ArrayAttr>().getValue();
      auto firstTransform =
          affineMapsArrayAttr[0].cast<AffineMapAttr>().getValue();
      auto lastTransform = affineMapsArrayAttr[affineMapsArrayAttr.size() - 1]
                               .cast<AffineMapAttr>()
                               .getValue();

      if (operandIndex == 0) {
        if (lastTransform.getNumResults() != sourceRank)
          return op.emitError(
              "Number of coordindates in externally defined affine map doesn't "
              "match the rank of the source memref");

        expectedSourceCoords = firstTransform.getNumInputs();
      } else if (operandIndex == 1) {
        if (lastTransform.getNumResults() != destRank)
          return op.emitError(
              "Number of coordindates in externally defined affine map doesn't "
              "match the rank of the destination memref");

        expectedDestCoords = firstTransform.getNumInputs();
      }
    }
  }

  if (sourceCoord.size() != expectedSourceCoords)
    return op.emitError(
        "Number of coordinates supplied doesn't match the rank, or affine maps "
        "of source memref");
  if (destCoord.size() != expectedSourceCoords)
    return op.emitError(
        "Number of coordinates supplied doesn't match the rank, or affine maps "
        "of destination memref");

  return success();
}

//===----------------------------------------------------------------------===//
// InWarpTransposeOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(InWarpTransposeOp op) {
  constexpr size_t swizzleGroupSize = InWarpTransposeOp::swizzleGroupSize;
  if (!llvm::isPowerOf2_32(op.size())) {
    return op.emitOpError("transpose size " + Twine(op.size()) +
                          "must be a power of 2");
  }
  if (op.size() <= 0) {
    return op.emitOpError("transpose size must be strictly positive");
  }

  auto vectorLen = static_cast<size_t>(
      op.vector().getType().cast<VectorType>().getNumElements());
  if (vectorLen < swizzleGroupSize) {
    return op.emitOpError("Vector input must have at least" +
                          Twine(swizzleGroupSize) + "elements");
  }
  if (vectorLen < op.size()) {
    return op.emitError("Vector input can't be shorter than transpose size");
  }

  if (op.vector().getType().cast<VectorType>().getRank() != 1) {
    return op.emitError("Input vector must be 1-dimensional");
  }

  auto inGroupPerm = op.inGroupPerm();

  llvm::SmallSet<uint32_t, swizzleGroupSize> expected;
  llvm::SmallSet<uint32_t, swizzleGroupSize> found;

  for (uint32_t i = 0; i < swizzleGroupSize; i++) {
    expected.insert(i);
  }

  for (auto &i : inGroupPerm) {
    found.insert(i.cast<IntegerAttr>().getValue().getZExtValue());
  }

  if (found != expected) {
    return op.emitOpError("inGroupPerm is not a permutation on the output row");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MIOpen/MIOpenOps.cpp.inc"
