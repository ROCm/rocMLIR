//===--------- RockTuning.h - MLIR tuning parameter generation ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR base types for tuning
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_ROCKTUNINGTYPE_H
#define MLIR_DIALECT_ROCK_ROCKTUNINGTYPE_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace rock {

// Parameter container holding a parameter and serialized string
struct ParamEntry {
  RockTuningParamAttrInterface param;
  KernelType primaryOpType;
  std::string configStr;
};

// Total tuning space
struct TunableParams {
  std::vector<RockTuningParamAttrInterface> tuningRange;
  KernelType primaryOpType;
  int numHeuristicQuick;
};

TunableParams *createTunableParamSpace(ModuleOp &mod);
bool tuningSetParam(ModuleOp &mod, ParamEntry *paramEntry);
bool tuningSetStr(ModuleOp &mod, std::string perfConfig);

struct ProblemCompare {
  bool operator()(RockGemmWrapperInterface lhs, RockGemmWrapperInterface rhs) {
    KernelType commonType = lhs.getKernelType();
    if (commonType < rhs.getKernelType())
      return false;
    // conv case
    if (commonType == KernelType::Conv2D) {
      Operation lhsOp = dyn_cast<Operation>(lhs);
      Operation rhsOp = dyn_cast<Operation>(rhs)
      RockConvInterface lhsConv = dyn_cast<RockConvInterface>(lhsOp);
      RockConvInterface rhsConv = dyn_cast<RockConvInterface>(rhsOp);
      if (lhsConv.getFilter().getType() != rhsConv.getFilter().getType())
        return false;
      if (lhsConv.getInput().getType() != rhsConv.getInput().getType())
        return false;
      if (lhsConv.getPadding() != rhsConv.getPadding())
        return false;
      if (lhsConv.getStrides() != rhsConv.getStrides())
        return false;
      if (lhsConv.getDilations() != rhsConv.getDilations())
        return false;
      if (lhsConv.getFeatures() != rhsConv.getFeatures())
        return false;
    }
    // gemm case
    else if (commonType == KernelType::Gemm) {
      if (lhs.getInputType() != rhs.getInputType())
        return false;
      if (lhs.getGemmSize().m != rhs.getGemmSize().m ||
          lhs.getGemmSize().k != rhs.getGemmSize().k ||
          lhs.getGemmSize().n != rhs.getGemmSize().n)
        return false;
      if (lhs.getGemmFeatures() != rhs.getGemmFeatures())
        return false;
    } else
      return false;

    return true;
  }
};

struct TuningTable {
  std::map<RockGemmWrapperInterface, std::pair<std::string, float>,
           ProblemCompare>
      tuningMap;
};

TuningTable *tuningTableCreate();
bool tuningTableUpdate(TuningTable *perfTable,
                       RockGemmWrapperInterface primaryOp,
                       std::string perfConfig, float time);
std::string tuningTableLookup(TuningTable *perfTable,
                              RockGemmWrapperInterface primaryOp);

} // namespace rock
} // namespace mlir
#endif // MLIR_DIALECT_ROCK_ROCKTUNINGTYPE_H
