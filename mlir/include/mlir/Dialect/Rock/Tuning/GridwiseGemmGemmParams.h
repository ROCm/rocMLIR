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

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Tuning/ParamLookupTable.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>

namespace mlir {
namespace rock {

// Default attention (gemm+gemm) perf config, used when no tuned or explicit
// config is available.
inline constexpr llvm::StringLiteral kDefaultAttnPerfConfig =
    "attn:v3:32,32,32,32,32,32,16,1,1,1,2,0,1";

class PopulateParamsGemmGemm {
public:
  static std::vector<GemmGemmParamsAttr>
  getTuningParameters(OpBuilder &b, RockGemmGemmWrapperInterface op);

  static LogicalResult paramsProbablyValid(OpBuilder &b,
                                           RockGemmGemmWrapperInterface op,
                                           GemmGemmParamsAttr params);

  static FailureOr<std::pair<AccelGemmParamsAttr, AccelGemmParamsAttr>>
  getAccelGemmParams(OpBuilder &b, RockGemmGemmWrapperInterface op,
                     GemmGemmParamsAttr params);

protected:
  static GemmGemmParamsAttr
  deserializePerfConfig(OpBuilder &b, RockGemmGemmWrapperInterface op,
                        StringRef config);

  static std::vector<GemmGemmParamsAttr>
  deserializePerfConfigs(OpBuilder &b, RockGemmGemmWrapperInterface op,
                         ArrayRef<StringRef> configs);

  static AccelGemmParamsAttr getGemm0Params(OpBuilder &b,
                                            GemmGemmParamsAttr params);

  static AccelGemmParamsAttr getGemm1Params(OpBuilder &b,
                                            GemmGemmParamsAttr params);

private:
#define GemmGemm_DECLARATIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef GemmGemm_DECLARATIONS_GEN

  friend class ParamLookupTable<GemmGemmParamsAttr>;
};

/// Conservative peak-LDS footprint for rocMLIR's transposed GEMM+GEMM
/// lowering. In the first phase, the A (Q) tile is N0 x K and the B (K) tile
/// is M0 x K. In the second phase, the first GEMM's M0 x N0 result and the
/// M0 x M1 C (V) tile are live. LDS is reused between phases, so the peak is
/// their maximum, scaled for double-buffered schedules.
static inline int64_t gemmGemmLdsBytes(GemmGemmParamsAttr params,
                                       int64_t gemm1MPerBlock, int64_t aBits,
                                       int64_t bBits, int64_t cBits) {
  int64_t mPerBlockG0 = params.getMPerBlockG0();
  int64_t nPerBlockG0 = params.getNPerBlockG0();
  int64_t kPerBlock = params.getKpackPerBlock() * params.getKpack();

  int64_t phaseABits =
      (nPerBlockG0 * kPerBlock * aBits) + (mPerBlockG0 * kPerBlock * bBits);
  int64_t phaseBBits = (mPerBlockG0 * nPerBlockG0 * cBits) +
                       (mPerBlockG0 * gemm1MPerBlock * cBits);

  std::optional<GemmLoadTileType> loadType =
      symbolizeGemmLoadTileType(params.getScheduleVersion());
  int64_t numStages =
      loadType == GemmLoadTileType::DoubleBuffer ||
              loadType == GemmLoadTileType::DirectToLDSDoubleBuffer
          ? 2
          : 1;
  return llvm::divideCeil(numStages * std::max(phaseABits, phaseBBits),
                          int64_t{8});
}

/// Conservative tuning-parameter LDS check using the actual A, B, and C
/// element widths. Operation verification remains responsible for deciding
/// whether those element types are supported by the requested accelerator.
inline bool isGemmGemmParamsConservativelyApplicable(GemmGemmParamsAttr params,
                                                     Type aElemType,
                                                     Type bElemType,
                                                     Type cElemType,
                                                     StringAttr arch) {
  int64_t bytes = gemmGemmLdsBytes(
      params, params.getMPerBlockG1(), aElemType.getIntOrFloatBitWidth(),
      bElemType.getIntOrFloatBitWidth(), cElemType.getIntOrFloatBitWidth());
  return bytes <= lookupArchInfo(arch).maxSharedMemPerWG;
}

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_GRIDWISE_GEMM_GEMM_PARAMS_H
