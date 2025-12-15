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
  static std::vector<AttnParamsAttr>
  getQuickTuningRange(OpBuilder &b, RockGemmGemmWrapperInterface op);

  static LogicalResult paramsProbablyValid(OpBuilder &b,
                                           RockGemmGemmWrapperInterface op,
                                           AttnParamsAttr params);

  static FailureOr<std::pair<AccelGemmParamsAttr, AccelGemmParamsAttr>>
  getGemmGemmTuningParams(OpBuilder &b, RockGemmGemmWrapperInterface op,
                          AttnParamsAttr params);

protected:
  static AttnParamsAttr deserializePerfConfig(OpBuilder &b,
                                              RockGemmGemmWrapperInterface op,
                                              StringRef config);

  static std::vector<AttnParamsAttr>
  deserializePerfConfigs(OpBuilder &b, RockGemmGemmWrapperInterface op,
                         ArrayRef<StringRef> configs);

  static AccelGemmParamsAttr getGemm0TuningParams(OpBuilder &b,
                                                  AttnParamsAttr params);

  static AccelGemmParamsAttr getGemm1TuningParams(OpBuilder &b,
                                                  AttnParamsAttr params);

private:
#define Attn_DECLARATIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef Attn_DECLARATIONS_GEN

  friend class ParamLookupTable;
};

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_GRIDWISE_GEMM_GEMM_PARAMS_H
