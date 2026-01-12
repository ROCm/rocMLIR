#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rock-tuning-parameter"

using namespace mlir;
using namespace mlir::rock;

#define GemmGemm_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef GemmGemm_DEFINITIONS_GEN

std::vector<GemmGemmParamsAttr>
PopulateParamsGemmGemm::getTuningParameters(OpBuilder &b,
                                            RockGemmGemmWrapperInterface op) {
  StringAttr arch = rock::getArchValue(op);
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  if (!archInfo.isAccel(op)) {
    return {};
  }
  auto perfConfigs = ParamLookupTable<GemmGemmParamsAttr>::lookup(
      arch, op.getKernelType(),
      cast<MemRefType>(op.getAType()).getElementType());
  return deserializePerfConfigs(b, op, perfConfigs);
}

GemmGemmParamsAttr PopulateParamsGemmGemm::deserializePerfConfig(
    OpBuilder &b, RockGemmGemmWrapperInterface op, StringRef config) {
  auto stringAttr = b.getStringAttr(config);
  StringAttr arch = rock::getArchValue(op);
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  bool isWmma = archInfo.isWmma(op.getAType(), op.getBType());
  return GemmGemmParamsAttr::get(stringAttr, isWmma);
}

std::vector<GemmGemmParamsAttr>
PopulateParamsGemmGemm::deserializePerfConfigs(OpBuilder &b,
                                               RockGemmGemmWrapperInterface op,
                                               ArrayRef<StringRef> configs) {
  std::vector<GemmGemmParamsAttr> ret;
  ret.reserve(configs.size());
  std::transform(
      configs.begin(), configs.end(), std::back_inserter(ret),
      [&](StringRef config) { return deserializePerfConfig(b, op, config); });
  return ret;
}

LogicalResult PopulateParamsGemmGemm::paramsProbablyValid(
    OpBuilder &b, RockGemmGemmWrapperInterface op, GemmGemmParamsAttr params) {
  if (succeeded(getAccelGemmParams(b, op, params))) {
    return success();
  } else {
    return failure();
  }
}

FailureOr<std::pair<AccelGemmParamsAttr, AccelGemmParamsAttr>>
PopulateParamsGemmGemm::getAccelGemmParams(OpBuilder &b,
                                           RockGemmGemmWrapperInterface op,
                                           GemmGemmParamsAttr params) {
  StringAttr arch = rock::getArchValue(op);
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  if (!archInfo.isAccel(op)) {
    return failure();
  }

  if ((params.getMPerBlockG1() % params.getMPerBlockG0()) ||
      (params.getMPerBlockG0() % params.getKpack())) {
    return failure();
  }

  AccelGemmParamsAttr accelParams0 = getGemm0Params(b, params);
  AccelGemmParamsAttr accelParams1 = getGemm1Params(b, params);

  auto populateParamsAccelPtr = PopulateParamsAccel::select(archInfo.defaultFeatures);
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
PopulateParamsGemmGemm::getGemm0Params(OpBuilder &b,
                                       GemmGemmParamsAttr params) {
  constexpr auto splitKFactor = 1, gridGroupSize = 0;
  return AccelGemmParamsAttr::get(
      b.getContext(), params.getKpackPerBlock(), params.getMPerBlockG0(),
      params.getNPerBlockG0(), params.getKpack(), params.getMPerWave(),
      params.getNPerWave(), params.getMnPerXdl(), splitKFactor,
      params.getScheduleVersion(), params.getOutputSwizzle(),
      params.getWavesPerEU(), gridGroupSize, params.getForceUnroll());
}

AccelGemmParamsAttr
PopulateParamsGemmGemm::getGemm1Params(OpBuilder &b,
                                       GemmGemmParamsAttr params) {
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
