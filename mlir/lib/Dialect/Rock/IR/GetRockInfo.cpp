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
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::rock;

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
  return minCU;
}

bool mlir::rock::opHasOptionalFeature(Operation *op) {
  bool hasOptionalFeature = llvm::TypeSwitch<Operation*,
                                             bool>(op)
  .Case<rock::GemmOp, rock::ConvOp, rock::ConvBwdDataOp,
        rock::ConvBwdWeightOp,  rock::GridwiseGemmOp,
        rock::GridwiseGemmAccelOp, rock::AttentionOp, rock::ReduceOp,
        rock::GlobalStoreOp, rock::GemmElementwiseGemmOp,
        rock::ConvElementwiseGemmOp, rock::ThreadwiseWriteAllOp,
        rock::BlockwiseGemmAccelOp, rock::ThreadwiseAccelGemmOp,
        rock::ConvertingCopyKernelOp,
        rock::GridwiseAttentionAccelOp>([](auto opWithFeatures) {

    return true;
  })
  .Default([](Operation *op) -> bool {
    return false;
  });
  
  return hasOptionalFeature;
}

mlir::rock::GemmFeatures mlir::rock::intersectGemmFeatures(GemmFeatures a,
                                                           GemmFeatures b) {
  return static_cast<GemmFeatures>(static_cast<uint32_t>(a) &
                                   static_cast<uint32_t>(b));
}

mlir::rock::GemmFeatures mlir::rock::getFeatures(Operation *op) {
  // First, check to see if the func has a 'features' attribute.
  auto func = getParentFuncOp(op);
  if (func) {
    if (auto features = func->getAttrOfType<rock::GemmFeaturesAttr>("features"))  
        return features.getValue();
  }

  // Next, check to see if the op has a 'features' attribute.
  if (auto features = op->getAttrOfType<rock::GemmFeaturesAttr>("features"))  
    return features.getValue(); 

  // In this case, the op does not have a 'Features' attribute, so we can
  // calculate the default features based on the architecture.
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(rock::getArchValue(op));
  // Get the types needed for feature calculation using TypeSwitch
  SmallVector<Type> typesForFeature = llvm::TypeSwitch<Operation*,
                                                       SmallVector<Type>>(op)
    .Case<rock::GemmOp, rock::ConvOp, rock::ConvBwdDataOp,
          rock::ConvBwdWeightOp,  rock::GridwiseGemmOp,
          rock::GridwiseGemmAccelOp, rock::AttentionOp, rock::ReduceOp,
          rock::GlobalStoreOp, rock::GemmElementwiseGemmOp,
          rock::ConvElementwiseGemmOp, rock::ThreadwiseWriteAllOp,
          rock::BlockwiseGemmAccelOp, rock::ThreadwiseAccelGemmOp,
          rock::ConvertingCopyKernelOp,
          rock::GridwiseAttentionAccelOp>([](auto opWithFeatures) {

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
    return rock::GemmFeatures::none;
  }

  return features.value();
}