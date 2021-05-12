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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::miopen;

//===----------------------------------------------------------------------===//
// MIOpenDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {

} // namespace

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
// Conv2DOp
//===----------------------------------------------------------------------===//

static ParseResult parseConv2DOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<Type, 3> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, Conv2DOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(Conv2DOp op) {

  auto isDisjointed = [&](llvm::StringRef tensor, llvm::StringRef dim1,
                          llvm::StringRef dim2) {
    auto layout = op->getAttr(tensor).cast<ArrayAttr>().getValue();
    auto pos1 = -1, pos2 = -1;
    for (unsigned int i = 0; i < layout.size(); ++i) {
      if (layout[i].dyn_cast<StringAttr>().getValue() == dim1)
        pos1 = i;
      if (layout[i].dyn_cast<StringAttr>().getValue() == dim2)
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

//===----------------------------------------------------------------------===//
// Conv2DBwdDataOp
//===----------------------------------------------------------------------===//

static ParseResult parseConv2DBwdDataOp(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<Type, 3> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, Conv2DBwdDataOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(Conv2DBwdDataOp op) {

  auto isDisjointed = [&](llvm::StringRef tensor, llvm::StringRef dim1,
                          llvm::StringRef dim2) {
    auto layout = op->getAttr(tensor).cast<ArrayAttr>().getValue();
    auto pos1 = -1, pos2 = -1;
    for (unsigned int i = 0; i < layout.size(); ++i) {
      if (layout[i].dyn_cast<StringAttr>().getValue() == dim1)
        pos1 = i;
      if (layout[i].dyn_cast<StringAttr>().getValue() == dim2)
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

//===----------------------------------------------------------------------===//
// Conv2DBwdWeightOp
//===----------------------------------------------------------------------===//

static ParseResult parseConv2DBwdWeightOp(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<Type, 3> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, Conv2DBwdWeightOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(Conv2DBwdWeightOp op) {

  auto isDisjointed = [&](llvm::StringRef tensor, llvm::StringRef dim1,
                          llvm::StringRef dim2) {
    auto layout = op->getAttr(tensor).cast<ArrayAttr>().getValue();
    auto pos1 = -1, pos2 = -1;
    for (unsigned int i = 0; i < layout.size(); ++i) {
      if (layout[i].dyn_cast<StringAttr>().getValue() == dim1)
        pos1 = i;
      if (layout[i].dyn_cast<StringAttr>().getValue() == dim2)
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
} //===----------------------------------------------------------------------===//
// Conv2DDummyOp
//===----------------------------------------------------------------------===//

static ParseResult parseConv2DDummyOp(OpAsmParser &parser,
                                      OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<Type, 3> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, Conv2DDummyOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(Conv2DDummyOp op) { return success(); }

//===----------------------------------------------------------------------===//
// TransformOp
//===----------------------------------------------------------------------===//

static ParseResult parseTransformOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType src;
  Type srcType, dstType;
  return failure(
      parser.parseLParen() ||
      parser.parseOperand(src) ||
      parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(srcType) ||
      parser.resolveOperand(src, srcType, result.operands) ||
      parser.parseKeywordType("to", dstType) ||
      parser.addTypeToList(dstType, result.types));
  return success();
}

static void print(OpAsmPrinter &p, TransformOp op) {
  p << op.getOperationName() << "(" << op.getOperand() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperand().getType() << " to " << op.getType();
}

static LogicalResult verify(TransformOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// GridwiseGemmOp
//===----------------------------------------------------------------------===//

static ParseResult parseGridwiseGemmOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<Type, 3> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, GridwiseGemmOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(GridwiseGemmOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// GridwiseGemmV2Op
//===----------------------------------------------------------------------===//

static ParseResult parseGridwiseGemmV2Op(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<Type, 3> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, GridwiseGemmV2Op op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(GridwiseGemmV2Op op) {
  return success();
}

//===----------------------------------------------------------------------===//
// GpuAllocOp
//===----------------------------------------------------------------------===//

static ParseResult parseGpuAllocOp(OpAsmParser &parser, OperationState &result) {
  Type allocatedType;

  return failure(
      parser.parseLParen() ||
      parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(allocatedType) ||
      parser.addTypeToList(allocatedType, result.types));
}

static void print(OpAsmPrinter &p, GpuAllocOp op) {
  p << op.getOperationName() << "()";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getType();
}

static LogicalResult verify(GpuAllocOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// SubviewOp
//===----------------------------------------------------------------------===//

static ParseResult parseSubviewOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType src, offset;
  Type srcType, dstType;
  return failure(
      parser.parseLParen() ||
      parser.parseOperand(src) ||
      parser.parseComma() ||
      parser.parseOperand(offset) ||
      parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(srcType) ||
      parser.resolveOperand(src, srcType, result.operands) ||
      parser.resolveOperand(offset, parser.getBuilder().getIndexType(), result.operands) ||
      parser.parseKeywordType("to", dstType) ||
      parser.addTypeToList(dstType, result.types));
  return success();
}

static void print(OpAsmPrinter &p, miopen::SubviewOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperands()[0].getType() << " to " << op.getType();
}

static LogicalResult verify(miopen::SubviewOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// FillOp
//===----------------------------------------------------------------------===//

static ParseResult parseFillOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType src, constantValue;
  Type srcType;

  return failure(
      parser.parseLParen() ||
      parser.parseOperand(src) ||
      parser.parseComma() ||
      parser.parseOperand(constantValue) ||
      parser.parseRParen() ||
      parser.parseColonType(srcType) ||
      parser.resolveOperand(src, srcType, result.operands) ||
      parser.resolveOperand(constantValue, srcType.cast<MemRefType>().getElementType(), result.operands));
}

static void print(OpAsmPrinter &p, FillOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p << " : " << op.getOperands()[0].getType();
}

static LogicalResult verify(FillOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// MovePosOp
//===----------------------------------------------------------------------===//

static ParseResult parseMovePosOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  Type srcType;

  auto ret = parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
             parser.parseColonType(srcType) ||
             parser.resolveOperand(ops[0], srcType, result.operands);

  for (unsigned i = 1; i < ops.size(); ++i) {
    ret &= succeeded(parser.resolveOperand(
        ops[i], srcType.cast<MemRefType>().getElementType(), result.operands));
  }
  return failure(ret);
}

static void print(OpAsmPrinter &p, MovePosOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p << " : " << op.getOperands()[0].getType();
}

static LogicalResult verify(MovePosOp op) { return success(); }

//===----------------------------------------------------------------------===//
// MovePosV2Op
//===----------------------------------------------------------------------===//

static ParseResult parseMovePosV2Op(OpAsmParser &parser,
                                    OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  Type srcType;

  auto ret = parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
             parser.parseColonType(srcType) ||
             parser.resolveOperand(ops[0], srcType, result.operands) ||
             parser.addTypeToList(srcType, result.types);

  for (unsigned i = 1; i < ops.size(); ++i) {
    ret &= succeeded(parser.resolveOperand(
        ops[i], srcType.cast<VectorType>().getElementType(), result.operands));
  }
  return failure(ret);
}

static void print(OpAsmPrinter &p, MovePosV2Op op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p << " : " << op.getType();
}

static LogicalResult verify(MovePosV2Op op) { return success(); }

//===----------------------------------------------------------------------===//
// WorkgroupBarrierOp
//===----------------------------------------------------------------------===//

static ParseResult parseWorkgroupBarrierOp(OpAsmParser &parser,
                                           OperationState &result) {
  return success();
}

static void print(OpAsmPrinter &p, WorkgroupBarrierOp op) {
  p << op.getOperationName();
}

static LogicalResult verify(WorkgroupBarrierOp op) { return success(); }

//===----------------------------------------------------------------------===//
// LDSBarrierOp
//===----------------------------------------------------------------------===//

static ParseResult parseLDSBarrierOp(OpAsmParser &parser,
                                     OperationState &result) {
  return success();
}

static void print(OpAsmPrinter &p, LDSBarrierOp op) {
  p << op.getOperationName();
}

static LogicalResult verify(LDSBarrierOp op) { return success(); }

//===----------------------------------------------------------------------===//
// WorkgroupIdOp
//===----------------------------------------------------------------------===//

static ParseResult parseWorkgroupIdOp(OpAsmParser &parser,
                                      OperationState &result) {
  Type retType;
  return failure(
    parser.parseColonType(retType) ||
    parser.addTypeToList(retType, result.types));
}

static void print(OpAsmPrinter &p, WorkgroupIdOp op) {
  p << op.getOperationName() << " : " << op.getType();
}

static LogicalResult verify(WorkgroupIdOp op) { return success(); }

//===----------------------------------------------------------------------===//
// WorktemIdOp
//===----------------------------------------------------------------------===//

static ParseResult parseWorkitemIdOp(OpAsmParser &parser,
                                     OperationState &result) {
  Type retType;
  return failure(
    parser.parseColonType(retType) ||
    parser.addTypeToList(retType, result.types));
}

static void print(OpAsmPrinter &p, WorkitemIdOp op) {
  p << op.getOperationName() << " : " << op.getType();
}

static LogicalResult verify(WorkitemIdOp op) { return success(); }

//===----------------------------------------------------------------------===//
// BlockwiseGemmOp
//===----------------------------------------------------------------------===//

static ParseResult parseBlockwiseGemmOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 5> ops;
  SmallVector<Type, 5> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, BlockwiseGemmOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(BlockwiseGemmOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// ThreadwiseGemmOp
//===----------------------------------------------------------------------===//

static ParseResult parseThreadwiseGemmOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<Type, 3> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, ThreadwiseGemmOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(ThreadwiseGemmOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// BlockwiseCopyOp
//===----------------------------------------------------------------------===//

static ParseResult parseBlockwiseCopyOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> ops;
  SmallVector<Type, 4> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, BlockwiseCopyOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(BlockwiseCopyOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// ThreadwiseCopyOp
//===----------------------------------------------------------------------===//

static ParseResult parseThreadwiseCopyOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 6> ops;
  SmallVector<Type, 2> types;

  auto ret = parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
             parser.parseOptionalAttrDict(result.attributes) ||
             parser.parseColonTypeList(types) ||
             parser.resolveOperand(ops[0], types[0], result.operands) ||
             parser.resolveOperand(ops[1], types[1], result.operands);

  for (unsigned i = 2; i < ops.size(); ++i) {
    ret &= succeeded(parser.resolveOperand(
        ops[i], parser.getBuilder().getIntegerType(32), result.operands));
  }
  return failure(ret);
}

static void print(OpAsmPrinter &p, ThreadwiseCopyOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperands()[0].getType() << ", "
    << op.getOperands()[1].getType();
}

static LogicalResult verify(ThreadwiseCopyOp op) {
  auto coords = op.sourceAndDestCoord();
  auto sourceType = op.source().getType().cast<MemRefType>();
  auto destType = op.dest().getType().cast<MemRefType>();
  auto sourceRank = sourceType.getRank();
  auto destRank = destType.getRank();
  auto sourceAffineMaps = sourceType.getAffineMaps();
  auto destAffineMaps = destType.getAffineMaps();

  unsigned expectedSourceCoords = sourceRank;
  unsigned expectedDestCoords = destRank;

  // check if memrefs have embedded affine maps.
  if (sourceAffineMaps.size() != 0)
    expectedSourceCoords = sourceAffineMaps[0].getNumInputs();
  if (destAffineMaps.size() != 0)
    expectedDestCoords = destAffineMaps[0].getNumInputs();

  // check if memrefs have externally defined affine maps.
  auto coordTransformAttrs = op->getAttr("coord_transforms");
  if (coordTransformAttrs) {
    for (auto coordTransformAttr :
         coordTransformAttrs.cast<ArrayAttr>().getValue()) {
      auto coordTransformDictAttr = coordTransformAttr.cast<DictionaryAttr>();
      auto operandIndex =
          coordTransformDictAttr.get("operand").cast<IntegerAttr>().getInt();
      auto transform = coordTransformDictAttr.get("transforms")
                           .cast<ArrayAttr>()
                           .getValue()[0]
                           .cast<AffineMapAttr>()
                           .getValue();

      if (operandIndex == 0) {
        if (transform.getNumResults() != sourceRank)
          return op.emitError(
              "Number of coordindates in externally defined affine map doesn't "
              "match the rank of the source memref");

        expectedSourceCoords = transform.getNumInputs();
      } else if (operandIndex == 1) {
        if (transform.getNumResults() != destRank)
          return op.emitError(
              "Number of coordindates in externally defined affine map doesn't "
              "match the rank of the destination memref");

        expectedDestCoords = transform.getNumInputs();
      }
    }
  }

  if (coords.size() != expectedSourceCoords + expectedDestCoords)
    return op.emitError(
        "Number of coordinates supplied doesn't match the rank, or affine maps "
        "of source and destination memrefs");
  return success();
}

//===----------------------------------------------------------------------===//
// ThreadwiseCopyV2Op
//===----------------------------------------------------------------------===//

static ParseResult parseThreadwiseCopyV2Op(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 5> ops;
  SmallVector<Type, 2> types;

  auto ret = parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
             parser.parseOptionalAttrDict(result.attributes) ||
             parser.parseColonTypeList(types) ||
             parser.resolveOperand(ops[0], types[0], result.operands) ||
             parser.resolveOperand(ops[1], types[1], result.operands);

  // resolve source offset.
  // resolve destination coordinates.
  for (unsigned i = 2; i < ops.size(); ++i) {
    ret &= succeeded(parser.resolveOperand(
        ops[i], parser.getBuilder().getIntegerType(32), result.operands));
  }
  return failure(ret);
}

static void print(OpAsmPrinter &p, ThreadwiseCopyV2Op op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperands()[0].getType() << ", "
    << op.getOperands()[1].getType();
}

static LogicalResult verify(ThreadwiseCopyV2Op op) {
  return success();
}

//===----------------------------------------------------------------------===//
// MFMAV2Op
//===----------------------------------------------------------------------===//

static ParseResult parseMFMAV2Op(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> ops;
  SmallVector<Type, 2> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperand(ops[0], types[0], result.operands) ||
      parser.resolveOperand(ops[1], types[0], result.operands) ||
      parser.resolveOperand(ops[2], types[1], result.operands) ||
      parser.addTypeToList(types[1], result.types));
}

static void print(OpAsmPrinter &p, MFMAV2Op op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperand(0).getType() << ", " << op.getType();
}

static LogicalResult verify(miopen::MFMAV2Op op) {
  return success();
}

//===----------------------------------------------------------------------===//
// XdlopsGemmV2Op
//===----------------------------------------------------------------------===//

static ParseResult parseXdlopsGemmV2Op(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 6> ops;
  SmallVector<Type, 4> types;

  auto ret = parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
             parser.parseOptionalAttrDict(result.attributes) ||
             parser.parseColonTypeList(types) ||
             parser.resolveOperand(ops[0], types[0], result.operands) ||
             parser.resolveOperand(ops[1], types[1], result.operands) ||
             parser.resolveOperand(ops[2], types[2], result.operands) ||
             parser.resolveOperand(ops[3], types[3], result.operands) ||
             parser.resolveOperand(ops[4], types[4], result.operands) ||
             parser.resolveOperand(ops[5], types[5], result.operands);

  for (unsigned i = 6; i < ops.size(); ++i) {
    ret &= succeeded(parser.resolveOperand(ops[i], types[i], result.operands));
    parser.addTypeToList(types[i], result.types);
  }
  return failure(ret);
}

static void print(OpAsmPrinter &p, XdlopsGemmV2Op op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(XdlopsGemmV2Op op) {
  return success();
}

//===----------------------------------------------------------------------===//
// BlockwiseGemmV2Op
//===----------------------------------------------------------------------===//

static ParseResult parseBlockwiseGemmV2Op(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 6> ops;
  SmallVector<Type, 4> types;

  auto ret = parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
             parser.parseOptionalAttrDict(result.attributes) ||
             parser.parseColonTypeList(types) ||
             parser.resolveOperand(ops[0], types[0], result.operands) ||
             parser.resolveOperand(ops[1], types[1], result.operands) ||
             parser.resolveOperand(ops[2], types[2], result.operands) ||
             parser.resolveOperand(ops[3], types[3], result.operands) ||
             parser.resolveOperand(ops[4], types[4], result.operands) ||
             parser.resolveOperand(ops[5], types[5], result.operands);

  for (unsigned i = 6; i < ops.size(); ++i) {
    ret &= succeeded(parser.resolveOperand(ops[i], types[i], result.operands));
    parser.addTypeToList(types[i], result.types);
  }
  return failure(ret);
}

static void print(OpAsmPrinter &p, BlockwiseGemmV2Op op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(BlockwiseGemmV2Op op) {
  return success();
}

//===----------------------------------------------------------------------===//
// DataConverOp
//===----------------------------------------------------------------------===//

static ParseResult parseDataConvertOp(OpAsmParser &parser,
                                      OperationState &result) {
  OpAsmParser::OperandType ops;
  Type type;
  Type retType;
  auto ret = parser.parseOperand(ops) || parser.parseColonType(type) ||
             parser.resolveOperand(ops, type, result.operands) ||
             parser.parseKeywordType("to", retType) ||
             parser.addTypeToList(retType, result.types);

  return failure(ret);
}

static void print(OpAsmPrinter &p, DataConvertOp op) {
  p << op.getOperationName() << " " << op.getOperand() << " ";
  p << " : " << op.getOperand().getType();
  p << " to " << op.getType();
}

static LogicalResult verify(DataConvertOp op) { return success(); }
//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

namespace mlir {

#define GET_OP_CLASSES
#include "mlir/Dialect/MIOpen/MIOpenOps.cpp.inc"

} // namespace mlir
