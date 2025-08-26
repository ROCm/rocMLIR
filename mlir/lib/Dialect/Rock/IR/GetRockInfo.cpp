//===------- GetRockInfo.cpp - Utility functions to get Rock Op info ------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/GetRockInfo.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "rock-info-utils"

using namespace mlir;
using namespace mlir::rock;

Operation *mlir::rock::getParentFuncOp(Operation *op) {
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

bool mlir::rock::isAccel(rock::GemmFeatures features) {
  return bitEnumContainsAny(features, GemmFeatures::wmma | GemmFeatures::mfma);
}

FailureOr<StringAttr> mlir::rock::getArch(Operation *op) {
  return getAttrFromOpOrParents<StringAttr>(op, "arch", "mhal.arch");
}

StringAttr mlir::rock::getArchValue(Operation *op) {
  auto maybeArch = rock::getArch(op);
  if (failed(maybeArch))
    llvm_unreachable("No 'arch' attribute on kernel");

  return maybeArch.value();
}

FailureOr<int64_t> mlir::rock::getNumCU(Operation *op) {
  FailureOr<StringAttr> maybeArch = getArch(op);
  if (failed(maybeArch)) {
    return failure();
  }
  StringAttr arch = maybeArch.value();
  FailureOr<IntegerAttr> maybeNumCU =
      getAttrFromOpOrParents<IntegerAttr>(op, "num_cu", "numCU");
  if (failed(maybeNumCU)) {
    return failure();
  }
  IntegerAttr numCU = maybeNumCU.value();
  AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  if (numCU.getValue().getSExtValue() < archInfo.minNumCU) {
    return op->emitError() << "num_cu=" << numCU
                           << " cannot be lower than arch minNumCU="
                           << archInfo.minNumCU;
  }
  return numCU.getValue().getSExtValue();
}

int64_t mlir::rock::getNumCUValue(Operation *op) {
  auto maybeCU = rock::getNumCU(op);
  if (succeeded(maybeCU)) {
    return maybeCU.value();
  }

  // Otherwise, we will need to get the minimum CU value from the architecture
  auto archStr = rock::getArchValue(op);
  int64_t minCU = rock::lookupArchInfo(archStr).minNumCU;
  LLVM_DEBUG(llvm::dbgs() << "Could not find num_cu, defaulting to minimum "
                          << "CU value for " << archStr << ": " << minCU
                          << "\n");
  return minCU;
}

mlir::rock::GemmFeatures mlir::rock::getFeatures(Operation *op) {
  // First, check to see if the func has a 'features' attribute.
  auto func = getParentFuncOp(op);
  if (func) {
    if (auto features = func->getAttrOfType<rock::GemmFeaturesAttr>("features"))
      return features.getValue();

    // If the initial op is a func and there is no `features` attribute, then
    // we cannot proceed
    if (isa<func::FuncOp>(op) || isa<gpu::GPUFuncOp>(op))
      llvm_unreachable("Trying to get 'features' for an invalid func op");
  }

  // Next, check to see if the op has a 'features' attribute.
  if (auto features = op->getAttrOfType<rock::GemmFeaturesAttr>("features"))
    return features.getValue();

  // In this case, the op does not have a 'Features' attribute, so we can
  // calculate the default features based on the architecture.
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(rock::getArchValue(op));
  // Get the types needed for feature calculation using TypeSwitch
  SmallVector<Type> typesForFeature =
      llvm::TypeSwitch<Operation *, SmallVector<Type>>(op)
          .Case<RockGemmFeaturesInterface, rock::ReduceOp>(
              [](auto opWithFeatures) {
                return opWithFeatures.getTypesForFeature();
              })
          .Default([](Operation *op) -> SmallVector<Type> {
            llvm_unreachable("Trying to get feature type on unsupported op");
          });

  std::optional<rock::GemmFeatures> features = std::nullopt;
  for (auto &ty : typesForFeature) {
    // If features is not yet set, then we can update features without having to
    // do an set intersection first
    auto newFeatures = archInfo.getDefaultFeatures(ty);
    if (!features.has_value()) {
      features = newFeatures;
      continue;
    }

    // For all other types, we need to do a set intersection
    features = intersectGemmFeatures(features.value(), newFeatures);
  }

  // Handle the case where no types were found, and we could not calculate
  // features
  if (!features.has_value()) {
    llvm_unreachable("Unable to calculate features for the operation");
  }

  return features.value();
}
