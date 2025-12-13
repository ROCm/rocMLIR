//===- GridwiseGemmGemmParams.h - MLIR tuning parameter generation --------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR tuning parameter generation for gemm+gemm (attn) ops
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_GRIDWISE_GEMM_GEMM_PARAMS_H
#define MLIR_DIALECT_ROCK_GRIDWISE_GEMM_GEMM_PARAMS_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Tuning/ParamLookupTable.h"

namespace mlir {
namespace rock {

class PopulateParamsAttn {
public:
  static std::vector<AttnPerfConfigAttr>
  getQuickTuningRange(OpBuilder &b, RockGemmGemmWrapperInterface op);

  static LogicalResult paramsProbablyValid(OpBuilder &b,
                                           RockGemmGemmWrapperInterface op,
                                           AttnPerfConfigAttr params);

  static FailureOr<std::pair<RockAccelTuningParamAttrInterface,
                             RockAccelTuningParamAttrInterface>>
  getGemmGemmTuningParams(OpBuilder &b, RockGemmGemmWrapperInterface op,
                          AttnPerfConfigAttr params);

protected:
  static AttnPerfConfigAttr
  deserializePerfConfig(OpBuilder &b, RockGemmGemmWrapperInterface op,
                        StringRef config);

  static std::vector<AttnPerfConfigAttr>
  deserializePerfConfigs(OpBuilder &b, RockGemmGemmWrapperInterface op,
                         ArrayRef<StringRef> configs);

  template <typename GemmParamsAttrType>
  static RockAccelTuningParamAttrInterface
  getGemm0TuningParams(OpBuilder &b, AttnPerfConfigAttr params);

  template <typename GemmParamsAttrType>
  static RockAccelTuningParamAttrInterface
  getGemm1TuningParams(OpBuilder &b, AttnPerfConfigAttr params);

private:
#define Attn_DECLARATIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef Attn_DECLARATIONS_GEN

  friend class ParamLookupTable<StringRef>;
};

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_GRIDWISE_GEMM_GEMM_PARAMS_H
