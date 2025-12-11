#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rock-tuning-parameter"

using namespace mlir;
using namespace mlir::rock;

PopulateParamsAttnInfo
PopulateParamsAttnInfo::fromOp(RockGemmGemmWrapperInterface op) {
  return PopulateParamsAttnInfo(
      op.getGemmGemmSize(), rock::getArchValue(op), rock::getFeatures(op),
      cast<MemRefType>(op.getAType()).getElementType(),
      cast<MemRefType>(op.getBType()).getElementType(),
      cast<MemRefType>(op.getCType()).getElementType(), op.getKernelType());
}

std::unique_ptr<PopulateParamsAttn>
PopulateParamsAttn::select(GemmFeatures features) {
  if (bitEnumContainsAll(features, GemmFeatures::mfma)) {
    return std::make_unique<PopulateParamsAttnXDL>();
  } else if (bitEnumContainsAll(features, GemmFeatures::wmma)) {
    return std::make_unique<PopulateParamsAttnWmma>();
  } else {
    return nullptr;
  }
}

std::vector<InitParamsAttn>
PopulateParamsAttn::getTuningParameters(KernelType kernelType, Type dataType,
                                        StringRef arch) const {
  auto params =
      ParamLookupTable<InitParamsAttn>::lookup(arch, kernelType, dataType);
  return std::vector<InitParamsAttn>(params.begin(), params.end());
}

Attribute
PopulateParamsAttn::getAttnParamsAttr(OpBuilder &b,
                                      const InitParamsAttn &params) const {
  return b.getAttr<AttnPerfConfigAttr>(
      params.mPerBlockG0, params.mPerBlockG1, params.nPerBlockG0,
      params.kpackPerBlock, params.mPerWave, params.nPerWave, params.mnPerXdl,
      params.kpack, params.splitKFactor, params.scheduleVersion,
      params.outputSwizzle, params.forceUnroll);
}

LogicalResult
PopulateParamsAttn::paramsProbablyValid(OpBuilder &b,
                                        const PopulateParamsAttnInfo &info,
                                        const InitParamsAttn &params) {
  if (succeeded(getAttentionTuningParams(b, info, params))) {
    return success();
  } else {
    return failure();
  }
}

static RockAccelTuningParamAttrInterface
deriveGemm1TuningParams(OpBuilder &b,
                        RockAccelTuningParamAttrInterface gemm0TuningParams,
                        const InitParamsAttn &params) {
  int64_t gemm1KPack = gemm0TuningParams.getKpack();
  if (auto gemm0XdlDerivedParams =
          dyn_cast<MfmaGemmParamsAttr>(gemm0TuningParams)) {
    return MfmaGemmParamsAttr::get(
        b.getContext(), gemm0TuningParams.getMPerBlock() / gemm1KPack,
        params.mPerBlockG1, gemm0XdlDerivedParams.getNPerBlock(),
        gemm0TuningParams.getKpack(),
        gemm0TuningParams.getMPerWave() *
            (params.mPerBlockG1 / gemm0TuningParams.getMPerBlock()),
        gemm0XdlDerivedParams.getNPerWave(),
        gemm0XdlDerivedParams.getMnPerXdl(), params.splitKFactor,
        gemm0XdlDerivedParams.getScheduleVersion(),
        gemm0XdlDerivedParams.getOutputSwizzle(),
        gemm0XdlDerivedParams.getForceUnroll());
  } else {
    return WmmaGemmParamsAttr::get(
        b.getContext(), gemm0TuningParams.getMPerBlock() / gemm1KPack,
        params.mPerBlockG1, params.nPerBlockG0, gemm0TuningParams.getKpack(),
        gemm0TuningParams.getMPerWave() *
            (params.mPerBlockG1 / gemm0TuningParams.getMPerBlock()),
        gemm0TuningParams.getNPerWave(), gemm0TuningParams.getMnPerXdl(),
        params.splitKFactor, gemm0TuningParams.getScheduleVersion(),
        gemm0TuningParams.getOutputSwizzle(),
        gemm0TuningParams.getForceUnroll());
  }
}

FailureOr<std::pair<RockAccelTuningParamAttrInterface,
                    RockAccelTuningParamAttrInterface>>
mlir::rock::getAttentionTuningParams(OpBuilder &b,
                                     const PopulateParamsAttnInfo &info,
                                     const InitParamsAttn &params) {
  GemmFeatures features = info.gemmFeatures;
  RockAccelTuningParamAttrInterface accelParams0;
  int64_t splitKFactor = 1;
  if (bitEnumContainsAny(features, GemmFeatures::mfma)) {
    accelParams0 = MfmaGemmParamsAttr::get(
        b.getContext(), params.kpackPerBlock, params.mPerBlockG0,
        params.nPerBlockG0, params.kpack, params.mPerWave, params.nPerWave,
        params.mnPerXdl, splitKFactor, params.scheduleVersion,
        params.outputSwizzle, params.forceUnroll);
  } else {
    accelParams0 = WmmaGemmParamsAttr::get(
        b.getContext(), params.kpackPerBlock, params.mPerBlockG0,
        params.nPerBlockG0, params.kpack, params.mPerWave, params.nPerWave,
        params.mnPerXdl, splitKFactor, params.scheduleVersion,
        params.outputSwizzle, params.forceUnroll);
  }
  if (params.mPerBlockG1 % params.mPerBlockG0 != 0) {
    return failure();
  }
  if (params.mPerBlockG0 % params.kpack != 0) {
    return failure();
  }
  RockAccelTuningParamAttrInterface accelParams1 =
      deriveGemm1TuningParams(b, accelParams0, params);
  auto populateParamsAccelPtr = PopulateParamsAccel::select(features);
  LogicalResult isValidBlockwiseGemm0 =
      populateParamsAccelPtr->isValidBlockwiseGemm(accelParams0, info.gemmAType,
                                                   info.gemmBType, info.arch);
  LogicalResult isValidBlockwiseGemm1 =
      populateParamsAccelPtr->isValidBlockwiseGemm(accelParams1, info.gemmCType,
                                                   info.gemmCType, info.arch);
  if (isValidBlockwiseGemm0.failed() || isValidBlockwiseGemm1.failed()) {
    return failure();
  }
  return std::make_pair(accelParams0, accelParams1);
}

#define Attn_XDL_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef Attn_XDL_DEFINITIONS_GEN

#define Attn_Wmma_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef Attn_Wmma_DEFINITIONS_GEN
