//===- GetRockInfo.h - functions used to calculate information about Rock ops
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_ROCK_IR_GETROCKINFO_H
#define MLIR_DIALECT_ROCK_IR_GETROCKINFO_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
class Operation;
class Type;

namespace rock {

// Get the arch from the op
FailureOr<StringAttr> getArch(Operation *op);

// Get the arch from the op and error out if it cannot be found
StringAttr getArchValue(Operation *op);

// Get the num_cu from the op
FailureOr<int64_t> getNumCU(Operation *op);

// Get the num_cu from the op, and error out if it cannot be found
int64_t getNumCUValue(Operation *op);

inline rock::GemmFeatures intersectGemmFeatures(rock::GemmFeatures a,
                                                rock::GemmFeatures b) {
  return a & b;
}

// Get the features enabled for the specified op. These will be dependent on
// the architecture being used, and the type of the op.
rock::GemmFeatures getFeatures(Operation *op);

// This function returns a boolean value if the underlying op has support for
// an optional 'GemmFeatures' attribute
bool opHasOptionalFeature(Operation *op);

// This function returns the func or gpu.func of a given op
inline Operation *getParentFuncOp(Operation *op) {
  Operation *func;
  if (isa<func::FuncOp, gpu::GPUFuncOp>(op)) {
    func = op;
  } else {
    func = op->getParentOfType<func::FuncOp>();
    if (!func) {
      func = op->getParentOfType<gpu::GPUFuncOp>();
    }
  }

  return func;
}

// Helper function to get attributes from parents
template <typename RetAttrType>
FailureOr<RetAttrType> getAttrFromOpOrParents(
    Operation *op, StringRef opAttr,
    std::optional<StringRef> maybeDialectAttr = std::nullopt) {
  StringRef dialectAttr = maybeDialectAttr.value_or(opAttr);
  Operation *func = getParentFuncOp(op);
  RetAttrType attr;
  auto getAnyAttr = [&](ArrayRef<StringRef> attrNames, Operation *op) {
    for (StringRef attrName : attrNames) {
      if (!attr) {
        attr = op->getAttrOfType<RetAttrType>(attrName);
      } else {
        return;
      }
    }
  };

  // First check for the attribute on the op
  getAnyAttr({opAttr}, op);
  if (!attr) {
    // If that fails then try checking for the attribute on the func
    getAnyAttr({opAttr, dialectAttr}, func);
  }

  // If there is no desired attribute on the func, then check the nearest parent
  // with a symbol table (covers both ModuleOp and gpu::GPUModuleOp)
  if (!attr) {
    if (auto symbolTableOp = func->getParentWithTrait<OpTrait::SymbolTable>()) {
      getAnyAttr({opAttr, dialectAttr}, symbolTableOp);
      if (attr)
        return attr;
    }
  }

  if (!attr) {
    return failure();
  }
  return attr;
}

} // End namespace rock
} // End namespace mlir
#endif // MLIR_DIALECT_ROCK_IR_GETROCKINFO_H
