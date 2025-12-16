#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rock-tuning-parameter"

using namespace mlir;
using namespace mlir::rock;

#define Attn_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef Attn_DEFINITIONS_GEN

std::vector<AttnParamsAttr>
PopulateParamsAttn::getQuickTuningRange(OpBuilder &b,
                                        RockGemmGemmWrapperInterface op) {
  if (!rock::isAccel(rock::getFeatures(op))) {
    return {};
  }
  auto perfConfigs = ParamLookupTable<AttnParamsAttr>::lookup(
      rock::getArchValue(op), op.getKernelType(),
      cast<MemRefType>(op.getAType()).getElementType());
  return deserializePerfConfigs(b, op, perfConfigs);
}

AttnParamsAttr PopulateParamsAttn::deserializePerfConfig(
    OpBuilder &b, RockGemmGemmWrapperInterface op, StringRef config) {
  auto stringAttr = b.getStringAttr(config);
  auto isWmma = bitEnumContainsAll(rock::getFeatures(op), GemmFeatures::wmma);
  return AttnParamsAttr::get(stringAttr, isWmma);
}

std::vector<AttnParamsAttr>
PopulateParamsAttn::deserializePerfConfigs(OpBuilder &b,
                                           RockGemmGemmWrapperInterface op,
                                           ArrayRef<StringRef> configs) {
  std::vector<AttnParamsAttr> ret;
  ret.reserve(configs.size());
  std::transform(
      configs.begin(), configs.end(), std::back_inserter(ret),
      [&](StringRef config) { return deserializePerfConfig(b, op, config); });
  return ret;
}

LogicalResult PopulateParamsAttn::paramsProbablyValid(
    OpBuilder &b, RockGemmGemmWrapperInterface op, AttnParamsAttr params) {
  if (succeeded(getGemmGemmTuningParams(b, op, params))) {
    return success();
  } else {
    return failure();
  }
}

FailureOr<std::pair<AccelGemmParamsAttr, AccelGemmParamsAttr>>
PopulateParamsAttn::getGemmGemmTuningParams(OpBuilder &b,
                                            RockGemmGemmWrapperInterface op,
                                            AttnParamsAttr params) {
  auto features = rock::getFeatures(op);
  if (!rock::isAccel(features)) {
    return failure();
  }

  if ((params.getMPerBlockG1() % params.getMPerBlockG0()) ||
      (params.getMPerBlockG0() % params.getKpack())) {
    return failure();
  }

  AccelGemmParamsAttr accelParams0 = getGemm0TuningParams(b, params);
  AccelGemmParamsAttr accelParams1 = getGemm1TuningParams(b, params);

  auto populateParamsAccelPtr = PopulateParamsAccel::select(features);
  LogicalResult isValidBlockwiseGemm0 =
      populateParamsAccelPtr->isValidBlockwiseGemm(
          accelParams0, cast<MemRefType>(op.getAType()).getElementType(),
          cast<MemRefType>(op.getBType()).getElementType(),
          rock::getArchValue(op));
  LogicalResult isValidBlockwiseGemm1 =
      populateParamsAccelPtr->isValidBlockwiseGemm(
          accelParams1, cast<MemRefType>(op.getCType()).getElementType(),
          cast<MemRefType>(op.getCType()).getElementType(),
          rock::getArchValue(op));
  if (isValidBlockwiseGemm0.failed() || isValidBlockwiseGemm1.failed()) {
    return failure();
  }

  return std::make_pair(accelParams0, accelParams1);
}

AccelGemmParamsAttr
PopulateParamsAttn::getGemm0TuningParams(OpBuilder &b, AttnParamsAttr params) {
  constexpr auto splitKFactor = 1, gridGroupSize = 0;
  return AccelGemmParamsAttr::get(
      b.getContext(), params.getKpackPerBlock(), params.getMPerBlockG0(),
      params.getNPerBlockG0(), params.getKpack(), params.getMPerWave(),
      params.getNPerWave(), params.getMnPerXdl(), splitKFactor,
      params.getScheduleVersion(), params.getOutputSwizzle(),
      params.getWavesPerEU(), gridGroupSize, params.getForceUnroll());
}

AccelGemmParamsAttr
PopulateParamsAttn::getGemm1TuningParams(OpBuilder &b, AttnParamsAttr params) {
  constexpr auto gridGroupSize = 0;
  return AccelGemmParamsAttr::get(
      b.getContext(), params.getMPerBlockG0() / params.getKpack(),
      params.getMPerBlockG1(), params.getNPerBlockG0(), params.getKpack(),
      params.getMPerWave() *
          (params.getMPerBlockG1() / params.getMPerBlockG0()),
      params.getNPerWave(), params.getMnPerXdl(), params.getSplitKFactor(),
      params.getScheduleVersion(), params.getOutputSwizzle(),
      params.getWavesPerEU(), gridGroupSize, params.getForceUnroll());
}
