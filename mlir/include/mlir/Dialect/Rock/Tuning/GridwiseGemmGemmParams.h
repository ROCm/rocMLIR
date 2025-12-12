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
  struct PerfConfig {
    static constexpr size_t n = 12;
    const int64_t data[n];
  };

  static std::vector<AttnPerfConfigAttr>
  getQuickTuningRange(OpBuilder &b, RockGemmGemmWrapperInterface op);

  static AttnPerfConfigAttr perfConfigToAttr(OpBuilder &b,
                                             const PerfConfig &config);

  static std::vector<AttnPerfConfigAttr>
  perfConfigsToAttrs(OpBuilder &b, const std::vector<PerfConfig> &configs);

  static LogicalResult paramsProbablyValid(OpBuilder &b,
                                           RockGemmGemmWrapperInterface op,
                                           AttnPerfConfigAttr params);

  static FailureOr<std::pair<RockAccelTuningParamAttrInterface,
                             RockAccelTuningParamAttrInterface>>
  getGemmGemmTuningParams(OpBuilder &b, RockGemmGemmWrapperInterface op,
                          AttnPerfConfigAttr params);

protected:
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

  friend class ParamLookupTable<PerfConfig>;
};

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_GRIDWISE_GEMM_GEMM_PARAMS_H
