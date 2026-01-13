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
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
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

static FailureOr<StringAttr> getArchInternal(Operation *op) {
  return getAttrFromOpOrParents<StringAttr>(op, "arch", "mhal.arch");
}

StringAttr mlir::rock::getArchValue(Operation *op) {
  // llvm::errs() << "getArchValue\n";
  auto maybeArch = getArchInternal(op);
  if (failed(maybeArch))
    llvm_unreachable("No 'arch' attribute on kernel");

  if (maybeArch.value().getValue().empty())
    llvm_unreachable("Empty 'arch' attribute on kernel");

  return maybeArch.value();
}

FailureOr<int64_t> mlir::rock::getNumCU(Operation *op) {
  FailureOr<StringAttr> maybeArch = getArchInternal(op);
  if (failed(maybeArch)) {
    LLVM_DEBUG(llvm::dbgs() << "arch not found\n");
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

FailureOr<int64_t> mlir::rock::getNumChiplets(Operation *op) {
  StringAttr arch = rock::getArchValue(op);
  FailureOr<IntegerAttr> maybeNumChiplets =
      getAttrFromOpOrParents<IntegerAttr>(op, "num_chiplets");
  if (failed(maybeNumChiplets)) {
    LLVM_DEBUG(llvm::dbgs() << "Could not find num_chiplets\n");
    return failure();
  }
  IntegerAttr numChiplets = maybeNumChiplets.value();
  AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  if (numChiplets.getValue().getSExtValue() <= 0) {
    return op->emitError() << "num_chiplets must be greater than zero";
  }
  if (numChiplets.getValue().getSExtValue() > archInfo.maxNumXCC) {
    return op->emitError() << "num_chiplets=" << numChiplets
                           << " cannot be greater than arch maxNumXCC="
                           << archInfo.maxNumXCC;
  }
  return numChiplets.getValue().getSExtValue();
}

int64_t mlir::rock::getNumChipletsValue(Operation *op) {
  auto maybeChiplets = rock::getNumChiplets(op);
  if (succeeded(maybeChiplets)) {
    return maybeChiplets.value();
  }

  // Otherwise, we will need to get the max chiplets value from the architecture
  auto archStr = rock::getArchValue(op);
  int64_t maxChiplets = rock::lookupArchInfo(archStr).maxNumXCC;
  LLVM_DEBUG(
      llvm::dbgs() << "Could not find num_chiplets, defaulting to maximum "
                   << "chiplets value for " << archStr << ": " << maxChiplets
                   << "\n");
  return maxChiplets;
}

bool mlir::rock::isAccel(GemmFeatures features) {
  return bitEnumContainsAny(features, GemmFeatures::wmma | GemmFeatures::mfma);
}

LogicalResult mlir::rock::isScheduleVersionSupported(int64_t scheduleVersion,
                                                     AmdArchInfo archInfo,
                                                     ArrayRef<Type> types,
                                                     StringRef arch) {
  std::optional<GemmLoadTileType> maybeLoadType =
      rock::symbolizeGemmLoadTileType(scheduleVersion);
  if (!maybeLoadType.has_value()) {
    LLVM_DEBUG(llvm::dbgs() << "Schedule version value is incorrect\n");
    return failure();
  }

  auto loadType = maybeLoadType.value();
  bool directToLDS = loadType == GemmLoadTileType::DirectToLDSDefault ||
                     loadType == GemmLoadTileType::DirectToLDSDoubleBuffer;
  if (directToLDS) {
    // Check if direct-to-LDS is supported (use first type if available)
    Type dataType = types.empty() ? Type() : types[0];
    int64_t numBytes = 0; // Check for any direct-to-LDS support
    bool supported = archInfo.isDirectToLDS(dataType, numBytes) ||
                     archInfo.isAsyncDirectToLDS(arch, dataType, numBytes);

    if (!supported) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Requested direct to LDS but not supported by the hardware\n");
      return failure();
    }
  }
  return success();
}
