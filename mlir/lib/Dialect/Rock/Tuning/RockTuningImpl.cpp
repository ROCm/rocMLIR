//===- RockTuningImpl.cpp - tuning API implementation ----*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file implements the tuning interfaces
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "llvm/ADT/SmallString.h"

namespace mlir {
namespace rock {

// Brute-force search in incremental order
void createGemmTuningRangeBF(struct TunableParams *newSpace,
                             RockGemmWrapperInterface gemmOp) {

  // blockSize M/block N/block K/block M/thread N/thread
  const std::vector<std::vector<uint32_t>> ValidRangeGeneralGemmParams = {
      {64, 128, 256}, {32, 64, 128}, {32, 64, 128}, {4, 8, 16}, {2, 4}, {2, 4}};

  // M/block N/block K/block M/wave N/wave kPack aCopyMore/forceUnroll
  const std::vector<std::vector<uint32_t>> ValidRangeXdlopsGemmParams = {
      {4, 8, 16, 32, 64, 128, 256},
      {16, 32, 64, 128, 256},
      {1, 2, 4, 8},
      {4, 8, 16, 32, 64, 128},
      {4, 8, 16, 32, 64, 128},
      {1, 4, 8},
      {0, 1}};

  // M/block N/block K/block M/wave N/wave kPack aCopyMore/forceUnroll
  const std::vector<std::vector<uint32_t>>
      ValidRangeXdlopsGemmParams8BitReduction = {{4, 8, 16, 32, 64, 128, 256},
                                                 {16, 32, 64, 128, 256},
                                                 {4, 8, 16, 32},
                                                 {4, 8, 16, 32, 64, 128},
                                                 {4, 8, 16, 32, 64, 128},
                                                 {1, 4, 8, 16},
                                                 {0, 1}};

  // M/block N/block K/block M/wave N/wave kPack aCopyMore/forceUnroll
  const std::vector<std::vector<uint32_t>> ValidRangeWmmaGemmParams = {
      {4, 8, 16, 32, 64, 128, 256},
      {16, 32, 64, 128, 256},
      {16, 32},
      {4, 8, 16, 32, 64, 128},
      {4, 8, 16, 32, 64, 128},
      {16},
      {0, 1}};

  OpBuilder b(gemmOp.getContext());
  GemmFeatures currentFeatures = gemmOp.getGemmFeatures();
  if (bitEnumContainsAll(currentFeatures, GemmFeatures::mfma)) {
    // XDLOPS
    Type inTypeA = gemmOp.getAType();
    bool is8BitReduction = inTypeA.isInteger(8) || inTypeA.isFloat8E5M2FNUZ() ||
                           inTypeA.isFloat8E4M3FNUZ();
    const std::vector<std::vector<uint32_t>> &xdlopsParams =
        is8BitReduction ? ValidRangeXdlopsGemmParams8BitReduction
                        : ValidRangeXdlopsGemmParams;
    for (uint32_t gemmMPerBlock : xdlopsParams[0]) {
      for (uint32_t gemmNPerBlock : xdlopsParams[1]) {
        for (uint32_t gemmKPerBlock : xdlopsParams[2]) {
          for (uint32_t gemmMPerWave : xdlopsParams[3]) {
            for (uint32_t gemmNPerWave : xdlopsParams[4]) {
              for (uint32_t gemmKPack : xdlopsParams[5]) {
                for (uint32_t forceUnroll : xdlopsParams[6]) {
                  XdlopsGemmParamsAttr gemmParams =
                      b.getAttr<XdlopsGemmParamsAttr>(
                          gemmKPerBlock, gemmMPerBlock, gemmNPerBlock,
                          gemmKPack, gemmMPerWave, gemmNPerWave, forceUnroll);
                  newSpace->tuningRange.push_back(
                      gemmParams.cast<RockTuningParamAttrInterface>());
                }
              }
            }
          }
        }
      }
    }
  } else if (bitEnumContainsAll(currentFeatures, GemmFeatures::wmma)) {
    // Wmma
    const std::vector<std::vector<uint32_t>> &wmmaParams =
        ValidRangeWmmaGemmParams;
    for (uint32_t gemmMPerBlock : wmmaParams[0]) {
      for (uint32_t gemmNPerBlock : wmmaParams[1]) {
        for (uint32_t gemmKPerBlock : wmmaParams[2]) {
          for (uint32_t gemmMPerWave : wmmaParams[3]) {
            for (uint32_t gemmNPerWave : wmmaParams[4]) {
              for (uint32_t gemmKPack : wmmaParams[5]) {
                for (uint32_t forceUnroll : wmmaParams[6]) {
                  WmmaGemmParamsAttr gemmParams = b.getAttr<WmmaGemmParamsAttr>(
                      gemmKPerBlock, gemmMPerBlock, gemmNPerBlock, gemmKPack,
                      gemmMPerWave, gemmNPerWave, forceUnroll);
                  newSpace->tuningRange.push_back(
                      gemmParams.cast<RockTuningParamAttrInterface>());
                }
              }
            }
          }
        }
      }
    }
  } else {
    // Non-XDLOPS
    for (uint32_t blockSize : ValidRangeGeneralGemmParams[0]) {
      for (uint32_t gemmMPerBlock : ValidRangeGeneralGemmParams[1]) {
        for (uint32_t gemmNPerBlock : ValidRangeGeneralGemmParams[2]) {
          for (uint32_t gemmKPerBlock : ValidRangeGeneralGemmParams[3]) {
            for (uint32_t gemmMPerThread : ValidRangeGeneralGemmParams[4]) {
              for (uint32_t gemmNPerThread : ValidRangeGeneralGemmParams[5]) {
                GeneralGemmParamsAttr gemmParams =
                    b.getAttr<GeneralGemmParamsAttr>(
                        blockSize, gemmKPerBlock, gemmMPerBlock, gemmNPerBlock,
                        /*kPerThread=*/1, gemmMPerThread, gemmNPerThread,
                        /*kpack=*/1);
                newSpace->tuningRange.push_back(
                    gemmParams.cast<RockTuningParamAttrInterface>());
              }
            }
          }
        }
      }
    }
  }

  newSpace->numHeuristicQuick = 0;
}

TunableParams *createTunableParamSpace(ModuleOp &mod) {
  struct TunableParams *newSpace;
  newSpace = new TunableParams();

  // create range and heuristic
  WalkResult findPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        createGemmTuningRangeBF(newSpace, op);
        newSpace->primaryOpType = op.getKernelType();
        return WalkResult::interrupt();
      });
  if (!findPrimary.wasInterrupted()) {
    delete newSpace;
  }
  return newSpace;
}

bool tuningGetParam(TunableParams *tuningSpace, int pos,
                    ParamEntry *paramEntry) {
  // out of bound check.
  if (pos < 0 || (unsigned int)pos > tuningSpace->tuningRange.size() - 1)
    return false;
  paramEntry->param = tuningSpace->tuningRange[pos];
  return true;
}

bool tuningSetParam(ModuleOp &mod, ParamEntry *paramEntry) {
  WalkResult setPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        auto ctx = op.getContext();
        std::string perfConfig;
        paramEntry->param.getPerfConfigStr(perfConfig);
        StringAttr attr = StringAttr::get(ctx, perfConfig);
        op->setAttr("perf_config", attr);
        return WalkResult::interrupt();
      });
  return setPrimary.wasInterrupted();
}

bool tuningSetStr(ModuleOp &mod, std::string perfConfig) {
  WalkResult setPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        auto ctx = op.getContext();
        StringAttr attr = StringAttr::get(ctx, perfConfig);
        op->setAttr("perf_config", attr);
        return WalkResult::interrupt();
      });
  return setPrimary.wasInterrupted();
}

TuningTable *tuningTableCreate() {
  struct TuningTable *newTable = new TuningTable();
  return newTable;
}

// Suppose to return the structure of the given problem to tune, currently
// combines the string representation of the selected field of the primary
// operation. String format of the problem will not be required by the DB,
// since it can store each field separately.
// Currently serialize the problem in MIOpenDriver command friendly format
std::string getTuningProblemStr(ModuleOp &mod) {
  rock::RockGemmWrapperInterface gemmIF;
  WalkResult findPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        gemmIF = op;
        return WalkResult::interrupt();
      });
  if (!findPrimary.wasInterrupted())
    return std::string();
  std::string problemStr;
  char sep = ' ';
  char tab = '\t';
  llvm::raw_string_ostream problemOS(problemStr);
  KernelType opType = gemmIF.getKernelType();
  Operation *gemmOp = gemmIF.getOperation();

  // ARCH string
  problemOS << gemmIF.getArch() << tab;

  if (opType == KernelType::Conv2D || opType == KernelType::Conv2DBwdData ||
      opType == KernelType::Conv2DBwdWeight) { // conv cases
    RockConvInterface convIF = dyn_cast<RockConvInterface>(gemmOp);

    ShapedType inType = convIF.getInput().getType();
    ArrayRef<int64_t> inShape = inType.getShape();
    ShapedType filType = convIF.getFilter().getType();
    ArrayRef<int64_t> filShape = filType.getShape();

    // Extract layout information
    auto filterLayoutAttr =
        gemmOp->template getAttrOfType<ArrayAttr>("filter_layout");
    auto inputLayoutAttr =
        gemmOp->template getAttrOfType<ArrayAttr>("input_layout");
    auto outputLayoutAttr =
        gemmOp->template getAttrOfType<ArrayAttr>("output_layout");

    unsigned size = filterLayoutAttr.size();
    std::map<StringRef, unsigned> fLayoutMap;
    std::map<StringRef, unsigned> iLayoutMap;
    std::map<StringRef, unsigned> oLayoutMap;

    for (unsigned i = 0; i < size; ++i) {
      auto filterAttr =
          filterLayoutAttr.getValue()[i].template cast<StringAttr>();
      fLayoutMap[filterAttr.getValue()] = i;
    }
    for (unsigned i = 0; i < size; ++i) {
      auto inputAttr =
          inputLayoutAttr.getValue()[i].template cast<StringAttr>();
      iLayoutMap[inputAttr.getValue()] = i;
    }
    for (unsigned i = 0; i < size; ++i) {
      auto outputAttr =
          outputLayoutAttr.getValue()[i].template cast<StringAttr>();
      oLayoutMap[outputAttr.getValue()] = i;
    }

    SmallString<5> fLayout("#####");
    SmallString<5> iLayout("#####");
    SmallString<5> oLayout("#####");

    // dimensions need to be mapped 1 to 1.
    fLayout[fLayoutMap["k"]] = 'N';
    fLayout[fLayoutMap["c"]] = 'C';
    fLayout[fLayoutMap["y"]] = 'H';
    fLayout[fLayoutMap["x"]] = 'W';
    fLayout[fLayoutMap["g"]] = 'G';
    iLayout[iLayoutMap["ni"]] = 'N';
    iLayout[iLayoutMap["ci"]] = 'C';
    iLayout[iLayoutMap["hi"]] = 'H';
    iLayout[iLayoutMap["wi"]] = 'W';
    iLayout[iLayoutMap["gi"]] = 'G';
    oLayout[oLayoutMap["no"]] = 'N';
    oLayout[oLayoutMap["ko"]] = 'C';
    oLayout[oLayoutMap["ho"]] = 'H';
    oLayout[oLayoutMap["wo"]] = 'W';
    oLayout[oLayoutMap["go"]] = 'G';

    // Please keep these in sync with mlir/utils/performance/perfRunner.py

    // OP datatype
    if (inType.getElementType().isF32()) {
      problemOS << "conv ";
    } else if (inType.getElementType().isF16()) {
      problemOS << "convfp16 ";
    } else if (inType.getElementType().isBF16()) {
      problemOS << "convbfp16 ";
    } else if (inType.getElementType().isInteger(8)) {
      problemOS << "convint8 ";
    } else {
      llvm_unreachable("Unknown data type.\n");
    }

    // OP direction
    switch (opType) {
    case KernelType::Conv2D:
      problemOS << "-F 1" << sep;
      break;
    case KernelType::Conv2DBwdData:
      problemOS << "-F 2" << sep;
      break;
    case KernelType::Conv2DBwdWeight:
      problemOS << "-F 4" << sep;
      break;
    default:
      llvm_unreachable("Unknown conv kernel type.\n");
    }

    // filter layout
    problemOS << "-f " << fLayout << sep;
    // input layout
    problemOS << "-I " << iLayout << sep;
    // output layout
    problemOS << "-O " << oLayout << sep;
    // N
    problemOS << "-n " << inShape[iLayoutMap["ni"]] << sep;
    // C
    problemOS << "-c " << inShape[iLayoutMap["ci"]] << sep;
    // H
    problemOS << "-H " << inShape[iLayoutMap["hi"]] << sep;
    // W
    problemOS << "-W " << inShape[iLayoutMap["wi"]] << sep;
    // K
    problemOS << "-k " << filShape[fLayoutMap["k"]] << sep;
    // Y
    problemOS << "-y " << filShape[fLayoutMap["y"]] << sep;
    // X
    problemOS << "-x " << filShape[fLayoutMap["x"]] << sep;

    auto paddingVal = extractFromI64ArrayAttr(convIF.getPadding());
    auto strideVal = extractFromI64ArrayAttr(convIF.getStrides());
    auto dilationVal = extractFromI64ArrayAttr(convIF.getDilations());
    // padding
    problemOS << "-p " << paddingVal[0] << " -q " << paddingVal[2] << sep;
    // stride
    problemOS << "-u " << strideVal[0] << " -v " << strideVal[1] << sep;
    // dilation
    problemOS << "-l " << dilationVal[0] << " -j " << dilationVal[1] << sep;
    // group
    problemOS << "-g " << inShape[iLayoutMap["gi"]] << sep;

  } else if (opType == KernelType::Gemm) { // gemm case
    rock::GemmOp rGemmOp = dyn_cast<rock::GemmOp>(gemmOp);
    // Please keep these in sync with mlir/utils/performance/perfRunner.py
    // Data type
    problemOS << "-t ";
    Type elemTypeA = gemmIF.getAType(), elemTypeB = gemmIF.getBType();
    if (elemTypeA.isF32() && elemTypeB.isF32()) {
      problemOS << "f32";
    } else if (elemTypeA.isF16() && elemTypeB.isF16()) {
      problemOS << "f16";
    } else if (elemTypeA.isBF16() && elemTypeB.isBF16()) {
      problemOS << "bf16";
    } else if (elemTypeA.isInteger(8) && elemTypeB.isInteger(8)) {
      problemOS << "i8";
    } else if (elemTypeA.isFloat8E4M3FNUZ() && elemTypeB.isFloat8E4M3FNUZ()) {
      problemOS << "fp8_fp8";
    } else if (elemTypeA.isFloat8E4M3FNUZ() && elemTypeB.isFloat8E5M2FNUZ()) {
      problemOS << "fp8_bf8";
    } else if (elemTypeA.isFloat8E5M2FNUZ() && elemTypeB.isFloat8E4M3FNUZ()) {
      problemOS << "bf8_fp8";
    } else if (elemTypeA.isFloat8E5M2FNUZ() && elemTypeB.isFloat8E5M2FNUZ()) {
      problemOS << "bf8_bf8";
    } else {
      // Unknown data type
      return std::string();
    }

    // OUtput datatype
    auto outType = gemmIF.getOutArgument()->get().getType();
    auto elemTypeC = outType.dyn_cast<mlir::MemRefType>().getElementType();
    problemOS << " -out_datatype ";
    if (elemTypeC.isFloat8E4M3FNUZ()) {
      problemOS << "fp8" << sep;
    } else if (elemTypeC.isFloat8E5M2FNUZ()) {
      problemOS << "bf8" << sep;
    } else {
      problemOS << elemTypeC << sep;
    }

    // TransA
    problemOS << "-transA ";
    if (rGemmOp.getATransposed())
      problemOS << "true ";
    else
      problemOS << "false ";

    // TransB
    problemOS << "-transB ";
    if (rGemmOp.getBTransposed())
      problemOS << "true ";
    else
      problemOS << "false ";

    // Gemmsize G/M/N/K
    problemOS << "-g " << gemmIF.getGemmSize().g << sep;
    problemOS << "-m " << gemmIF.getGemmSize().m << sep;
    problemOS << "-n " << gemmIF.getGemmSize().n << sep;
    problemOS << "-k " << gemmIF.getGemmSize().k << sep;
  } else {
    // Unknown op type, unreachable.
    return std::string();
  }

  return problemStr;
}

bool tuningTableUpdate(TuningTable *perfTable, std::string problem,
                       std::string perfConfig, float time) {
  if (problem.empty())
    return false;
  auto search = perfTable->tuningMap.find(problem);
  if (search != perfTable->tuningMap.end()) {
    auto entry = perfTable->tuningMap[problem];
    if (entry.second <= time) {
      return false;
    }
  }
  perfTable->tuningMap[problem] = std::make_pair(perfConfig, time);
  return true;
}

std::string tuningTableLookup(TuningTable *perfTable, ModuleOp &mod) {
  std::string problem = getTuningProblemStr(mod);
  if (problem.empty())
    return std::string();
  auto search = perfTable->tuningMap.find(problem);
  if (search != perfTable->tuningMap.end()) {
    auto entry = perfTable->tuningMap[problem];
    return entry.first;
  }
  return std::string();
}

} // namespace rock
} // namespace mlir
