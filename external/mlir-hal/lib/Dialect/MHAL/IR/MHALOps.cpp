//===- MHAL.cpp - MLIR MHAL Operations ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MHAL/IR/MHAL.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::mhal;

// constexpr StringRef MHALDialect::kAllowedToBlockAttrName;

#include "mlir/Dialect/MHAL/IR/MHALOps.cpp.inc"
#include "mlir/Dialect/MHAL/IR/MHALOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
/// LaunchOp
//===----------------------------------------------------------------------===//

void LaunchOp::build(OpBuilder &builder, OperationState &result,
                     func::FuncOp func, ValueRange dependencies,
                     ValueRange operands) {
  // set callee
  result.addAttribute(getCalleeAttrName(result.name), SymbolRefAttr::get(func));

  result.addOperands(dependencies);
  result.addOperands(operands);

  // Add derived `operand_segment_sizes` attribute based on parsed operands.
  int32_t numDependencies = dependencies.size();
  int32_t numOperands = operands.size();
  auto operandSegmentSizes =
      builder.getDenseI32ArrayAttr({numDependencies, numOperands});
  result.addAttribute(getOperandSegmentSizesAttrName(result.name),
                      operandSegmentSizes);

  // First result is always a token, and then `resultTypes` wrapped into
  // `mhal.value`.
  result.addTypes({TokenType::get(result.getContext())});
  for (Type type : func.getResultTypes())
    result.addTypes(type);
}

/// Return the callee of this operation.
CallInterfaceCallable LaunchOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>(getCalleeAttrName());
}

/// Set the callee for this operation.
void LaunchOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

/// Return the operands passed to the callee.
Operation::operand_range LaunchOp::getArgOperands() {
  return getLaunchOperands();
}

/// Return the callee results.
Operation::result_range LaunchOp::getCallResults() {
  return {++result_begin(), result_end()};
}

/// Return the callee result types.
Operation::result_type_range LaunchOp::getCallResultTypes() {
  return getResults();
}

/// Recompute the operand_segment_sizes attribute.
void LaunchOp::updateSegmentSizes(MLIRContext *ctx) {
  auto tokenTy = TokenType::get(ctx);
  int32_t numDependencies = 0;
  int32_t numOperands = 0;
  for (const auto &oper : getOperands()) {
    if (oper.getType() == tokenTy) {
      // All tokens should come first.
      assert(numOperands == 0);
      numDependencies++;
    } else
      numOperands++;
  }

  auto operandSegmentSizes =
      DenseI32ArrayAttr::get(ctx, {numDependencies, numOperands});
  (*this)->setAttr(getOperandSegmentSizesAttrName(), operandSegmentSizes);

  assert(!(*this)->hasAttr("result_segment_sizes"));
}

LogicalResult LaunchOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto callable = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!callable)
    return emitOpError("requires a 'callee' symbol reference attribute");

  func::FuncOp func =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, callable);
  if (!func)
    return emitOpError() << "'" << callable.getValue()
                         << "' does not reference a valid function";

  auto funcResultTypes = func.getResultTypes();
  // The result types should be a leading mhal.token and matching return types
  // of the func.
  auto resultTypes = getResultTypes();
  if (resultTypes.size() != (funcResultTypes.size() + 1))
    return emitOpError(
        "requires matching result types with a leading mhal.token");

  auto resultItr = ++resultTypes.begin();
  for (auto resType : funcResultTypes) {
    if (*resultItr++ != resType)
      return emitOpError("requires matching result types with func");
  }

  // Match operand types
  auto funcArgumentTypes = func.getArgumentTypes();
  if (funcArgumentTypes.size() != getLaunchOperands().size())
    return emitOpError("incorrect number of operands for callee");

  for (auto tuple : llvm::zip(getLaunchOperands(), funcArgumentTypes)) {
    if (std::get<0>(tuple).getType() != std::get<1>(tuple))
      return emitOpError("requires matching operand types");
  }

  return success();
}

LogicalResult LaunchOp::verify() {
  MLIRContext *ctx = getContext();
  auto tokenTy = TokenType::get(ctx);

  // The dependencies must be mhal.tokens
  for (auto dep : getDependencies()) {
    if (dep.getType() != tokenTy)
      return emitOpError("requires all dependencies to be mhal.token");
  }

  return success();
}

//===----------------------------------------------------------------------===//
/// AwaitOp
//===----------------------------------------------------------------------===//

void AwaitOp::build(OpBuilder &builder, OperationState &result, Value operand,
                    ArrayRef<NamedAttribute> attrs) {
  result.addOperands({operand});
  result.attributes.append(attrs.begin(), attrs.end());
}

static ParseResult parseAwaitResultType(OpAsmParser &parser, Type &operandType,
                                        Type &resultType) {
  if (parser.parseType(operandType))
    return failure();

  return success();
}

static void printAwaitResultType(OpAsmPrinter &p, Operation *op,
                                 Type operandType, Type resultType) {
  p << operandType;
}

LogicalResult AwaitOp::verify() {
  Type argType = getOperand().getType();

  // Awaiting on a token does not have any results.
  if (argType.isa<TokenType>() && !getResultTypes().empty())
    return emitOpError("awaiting on a token must have empty result");

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MHAL/IR/MHALOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd type method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MHAL/IR/MHALOpsTypes.cpp.inc"
