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
  if (!rock::isAccel(rock::getFeatures(op))) {
    return {};
  }
  auto perfConfigs = ParamLookupTable<GemmGemmParamsAttr>::lookup(
      rock::getArchValue(op), op.getKernelType(),
      cast<MemRefType>(op.getAType()).getElementType());
  return deserializePerfConfigs(b, op, perfConfigs);
}

GemmGemmParamsAttr PopulateParamsGemmGemm::deserializePerfConfig(
    OpBuilder &b, RockGemmGemmWrapperInterface op, StringRef config) {
  auto stringAttr = b.getStringAttr(config);
  return GemmGemmParamsAttr::get(stringAttr);
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

FailureOr<std::pair<GemmParamsAttr, GemmParamsAttr>>
PopulateParamsGemmGemm::getGemmParams(OpBuilder &b,
                                           RockGemmGemmWrapperInterface op,
                                           GemmGemmParamsAttr params) {
  auto features = rock::getFeatures(op);
  if (!rock::isAccel(features)) {
    return failure();
  }

  GemmParamsAttr accelParams0 = getGemm0Params(b, params);
  GemmParamsAttr accelParams1 = getGemm1Params(b, params);

  auto populateParamsAccelPtr = PopulateParamsAccel::select(features);
  return std::make_pair(accelParams0, accelParams1);
}

GemmParamsAttr
PopulateParamsGemmGemm::getGemm0Params(OpBuilder &b,
                                       GemmGemmParamsAttr params) {
  constexpr auto splitKFactor = 1;
  
  return GemmParamsAttr::get(
      b.getContext(), params.getMPerBlockG0(),
      params.getNPerBlockG0(), params.getKpackPerBlock(), params.getKpack(), params.getNumCTAs(),
      params.getNumWaves(), params.getMatrixInstrNonkdim(), splitKFactor,
      params.getNumStages(),
      params.getWavesPerEU(), params.getGridGroupSize());
}

GemmParamsAttr
PopulateParamsGemmGemm::getGemm1Params(OpBuilder &b,
                                       GemmGemmParamsAttr params) {
  // let parameters = (ins "int64_t":$mPerBlock,
  //     "int64_t":$nPerBlock, "int64_t":$kpackPerBlock, "int64_t":$kpack, "int64_t":$numCTAs, "int64_t":$numWaves,
  //     "int64_t":$matrixInstrNonkdim, "int64_t":$splitKFactor,
  //     "int64_t":$numStages, 
  //     "int64_t":$wavesPerEU, "int64_t":$gridGroupSize);
  return GemmParamsAttr::get(
      b.getContext(), params.getMPerBlockG0() / params.getKpack(),
      params.getMPerBlockG1(), params.getNPerBlockG0(), params.getKpack(),
      params.getNumCTAs(),
      params.getNumWaves(), params.getMatrixInstrNonkdim(), params.getSplitKFactor(),
      params.getNumStages(), 
      params.getWavesPerEU(), params.getGridGroupSize());
}
