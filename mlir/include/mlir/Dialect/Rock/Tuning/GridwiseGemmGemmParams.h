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

#include "mlir/Dialect/Rock/IR/GemmGemmSize.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Tuning/ParamLookupTable.h"
#include "mlir/Dialect/Rock/Tuning/Serializable.h"

namespace mlir {
namespace rock {

struct InitParamsAttn : Serializable<InitParamsAttn> {
  int64_t mPerBlockG0;
  int64_t mPerBlockG1;
  int64_t nPerBlockG0;
  int64_t kpackPerBlock;
  int64_t mPerWave;
  int64_t nPerWave;
  int64_t mnPerXdl;
  int64_t kpack;
  int64_t splitKFactor;
  int64_t scheduleVersion;
  int64_t outputSwizzle;
  bool forceUnroll;

  constexpr InitParamsAttn(int64_t mPerBlockG0, int64_t mPerBlockG1,
                           int64_t nPerBlockG0, int64_t kpackPerBlock,
                           int64_t mPerWave, int64_t nPerWave, int64_t mnPerXdl,
                           int64_t kpack, int64_t splitKFactor,
                           int64_t scheduleVersion, int64_t outputSwizzle,
                           bool forceUnroll)
      : mPerBlockG0(mPerBlockG0), mPerBlockG1(mPerBlockG1),
        nPerBlockG0(nPerBlockG0), kpackPerBlock(kpackPerBlock),
        mPerWave(mPerWave), nPerWave(nPerWave), mnPerXdl(mnPerXdl),
        kpack(kpack), splitKFactor(splitKFactor),
        scheduleVersion(scheduleVersion), outputSwizzle(outputSwizzle),
        forceUnroll(forceUnroll) {}

  constexpr InitParamsAttn()
      : InitParamsAttn(0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 1LL, 1LL, 2LL,
                       true) {}

  InitParamsAttn(AttnPerfConfigAttr attr)
      : mPerBlockG0(attr.getMPerBlockG0()), mPerBlockG1(attr.getMPerBlockG1()),
        nPerBlockG0(attr.getNPerBlockG0()),
        kpackPerBlock(attr.getKpackPerBlock()), mPerWave(attr.getMPerWave()),
        nPerWave(attr.getNPerWave()), mnPerXdl(attr.getMnPerXdl()),
        kpack(attr.getKpack()), splitKFactor(attr.getSplitKFactor()),
        scheduleVersion(attr.getScheduleVersion()),
        outputSwizzle(attr.getOutputSwizzle()),
        forceUnroll(attr.getForceUnroll()) {}

  template <class Self, class F>
  static void visit(Self &&self, F f) {
    f(self.mPerBlockG0);
    f(self.mPerBlockG1);
    f(self.nPerBlockG0);
    f(self.kpackPerBlock);
    f(self.mPerWave);
    f(self.nPerWave);
    f(self.mnPerXdl);
    f(self.kpack);
    f(self.splitKFactor);
    f(self.scheduleVersion);
    f(self.outputSwizzle);
    f(self.forceUnroll);
  }
};

struct PopulateParamsAttnInfo {
  GemmGemmSize gemmGemmSize;
  SmallString<32> arch;
  GemmFeatures gemmFeatures;
  Type gemmAType;
  Type gemmBType;
  Type gemmCType;
  KernelType kernelType;

  PopulateParamsAttnInfo(GemmGemmSize gemmGemmSize, StringRef arch,
                         GemmFeatures gemmFeatures, Type gemmAType,
                         Type gemmBType, Type gemmCType, KernelType kernelType)
      : gemmGemmSize(gemmGemmSize), arch(arch), gemmFeatures(gemmFeatures),
        gemmAType(gemmAType), gemmBType(gemmBType), gemmCType(gemmCType),
        kernelType(kernelType) {}

  static PopulateParamsAttnInfo fromOp(RockGemmGemmWrapperInterface op);
};

class PopulateParamsAttn {
public:
  virtual ~PopulateParamsAttn() = default;

  static std::unique_ptr<PopulateParamsAttn> select(GemmFeatures features);

  std::vector<InitParamsAttn> getTuningParameters(KernelType kernelType,
                                                  Type dataType,
                                                  StringRef arch) const;

  Attribute getAttnParamsAttr(OpBuilder &b, const InitParamsAttn &params) const;

  LogicalResult paramsProbablyValid(OpBuilder &b,
                                    const PopulateParamsAttnInfo &info,
                                    const InitParamsAttn &params);
};

class PopulateParamsAttnXDL : public PopulateParamsAttn {
private:
#define Attn_XDL_DECLARATIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef Attn_XDL_DECLARATIONS_GEN

  friend class ParamLookupTable<InitParamsAttn>;
};

class PopulateParamsAttnWmma : public PopulateParamsAttn {
private:
#define Attn_Wmma_DECLARATIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef Attn_Wmma_DECLARATIONS_GEN

  friend class ParamLookupTable<InitParamsAttn>;
};

FailureOr<std::pair<RockAccelTuningParamAttrInterface,
                    RockAccelTuningParamAttrInterface>>
getAttentionTuningParams(OpBuilder &b, const PopulateParamsAttnInfo &info,
                         const InitParamsAttn &params);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_GRIDWISE_GEMM_GEMM_PARAMS_H
