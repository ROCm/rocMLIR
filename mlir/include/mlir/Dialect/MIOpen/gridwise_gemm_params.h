//===- gridwise_gemm_params.h - MLIR tuning parameter generation --------*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR tuning parameter generation
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIOPEN_GRIDWISE_GEMM_PARAMS_H
#define MLIR_DIALECT_MIOPEN_GRIDWISE_GEMM_PARAMS_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"

#define DEBUG_TYPE "miopen-tuning-parameter"

using namespace mlir;

extern llvm::cl::opt<std::string> TunableParametersYAMLFile;
extern llvm::cl::opt<bool> IsPopulateTunableParameters;

LLVM_YAML_IS_STRING_MAP(int)

// greatest common divisor, aka highest common factor
template <typename T>
T gcd(T x, T y) {
  if (x == y || x == 0) {
    return y;
  } else if (y == 0) {
    return x;
  } else if (x > y) {
    return gcd(x - y, y);
  } else {
    return gcd(x, y - x);
  }
}

template <typename T, typename... Ys>
T gcd(T x, Ys... ys) {
  return gcd(x, gcd(ys...));
}

// least common multiple
template <typename T>
T lcm(T x, T y) {
  if (x == 0 || y == 0) {
    return 0;
  } else {
    return (x * y) / gcd(x, y);
  }
}

template <typename T, typename... Ys>
T lcm(T x, Ys... ys) {
  return lcm(x, lcm(ys...));
}

template <typename T>
T integer_divide_ceil(T x, T y) {
  return (x + y - 1) / y;
}

template <typename T>
T integer_least_multiple(T x, T y) {
  return y * integer_divide_ceil(x, y);
}

struct ConvolutionContext {
  mlir::miopen::ConvOpType opType;
  llvm::StringMap<std::pair<size_t, int64_t>> dimIndexVal;
  llvm::SmallVector<int64_t, 0> strideVal;
  llvm::SmallVector<int64_t, 0> dilationVal;
  llvm::SmallVector<int64_t, 0> paddingVal;
};

struct InitParams {
  int64_t gemmMPerBlock;
  int64_t gemmNPerBlock;
  int64_t gemmKPerBlock;
};

struct GemmSize {
  int64_t gemmM;
  int64_t gemmN;
  int64_t gemmK;
};

struct DerivedParams {
  int64_t srcDataPerRead;
  int64_t dstDataPerWrite;
  int64_t clusterLenGemmPos1;
  int64_t clusterLenGemmPos2;
  DerivedParams()
      : srcDataPerRead(1), dstDataPerWrite(1), clusterLenGemmPos1(0),
        clusterLenGemmPos2(0) {}
};

static constexpr int kConv2DTensorDimension = 4;
static constexpr StringLiteral kVarName[3] = {"weight", "input", "output"};

static void EmitLayoutString(llvm::raw_ostream &output,
                             llvm::ArrayRef<mlir::Attribute> &layoutArrayAttr,
                             llvm::StringRef prefix, llvm::StringRef suffix,
                             llvm::StringRef delimiter = "") {
  for (int i = 0; i < kConv2DTensorDimension; ++i) {
    auto attr = layoutArrayAttr[i];
    if (auto strAttr = attr.dyn_cast<StringAttr>()) {
      output << prefix << strAttr.getValue() << suffix;
    }
    if (i < kConv2DTensorDimension - 1) {
      output << delimiter;
    }
  }
}

static miopen::ConvOpType ObtainConvDirection(miopen::GridwiseGemmOp &op) {
  miopen::ConvOpType opType;
  auto kernel_algorithm = op.getAttrOfType<StringAttr>("kernel_algorithm");
  if (kernel_algorithm.getValue().find(StringRef("backward_data")) !=
      StringRef::npos) {
    opType = miopen::ConvOpType::Conv2DBwdDataOpType;
  } else if (kernel_algorithm.getValue().find(StringRef("backward_weight")) !=
             StringRef::npos) {
    opType = miopen::ConvOpType::Conv2DBwdWeightOpType;
  } else {
    opType = miopen::ConvOpType::Conv2DOpType;
  }
  return opType;
}

static void
populateDimVal(const ArrayAttr &layoutAttr, const ArrayAttr &dimAttr,
               llvm::StringMap<std::pair<size_t, int64_t>> &dimIndexVal) {
  assert(layoutAttr.size() == dimAttr.size());
  size_t dimValSize = layoutAttr.size();
  for (size_t i = 0; i < dimValSize; ++i) {
    auto key = layoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();
    auto value = dimAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
    dimIndexVal[key] = std::make_pair(i, value);
  }
}

static void populateSeqVal(const ArrayAttr &seqAttr,
                           llvm::SmallVector<int64_t, 0> &seqVal) {
  size_t seqValSize = seqAttr.size();
  for (size_t i = 0; i < seqValSize; ++i) {
    // Not nested array, push back the value and be done
    if (seqAttr.getValue()[i].dyn_cast<ArrayAttr>() == nullptr) {
      seqVal.push_back(seqAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt());
      continue;
    }
    // There is nested values, continue to populate those
    for (size_t j = 0; j < seqAttr.getValue()[i].dyn_cast<ArrayAttr>().size();
         ++j) {
      seqVal.push_back(seqAttr.getValue()[i]
                           .dyn_cast<ArrayAttr>()
                           .getValue()[j]
                           .dyn_cast<IntegerAttr>()
                           .getInt());
    }
  }
}

static ConvolutionContext populateConvContext(miopen::GridwiseGemmOp &op) {
  miopen::ConvOpType opType = ObtainConvDirection(op);

  llvm::StringMap<std::pair<size_t, int64_t>> dimIndexVal;

  auto filterLayoutAttr = op.getAttrOfType<ArrayAttr>("filter_layout");
  auto filterDimensionAttr = op.getAttrOfType<ArrayAttr>("filter_dimension");
  populateDimVal(filterLayoutAttr, filterDimensionAttr, dimIndexVal);
  auto inputLayoutAttr = op.getAttrOfType<ArrayAttr>("input_layout");
  auto inputDimensionAttr = op.getAttrOfType<ArrayAttr>("input_dimension");
  populateDimVal(inputLayoutAttr, inputDimensionAttr, dimIndexVal);
  auto outputLayoutAttr = op.getAttrOfType<ArrayAttr>("output_layout");
  auto outputDimensionAttr = op.getAttrOfType<ArrayAttr>("output_dimension");
  populateDimVal(outputLayoutAttr, outputDimensionAttr, dimIndexVal);

  auto strideAttr = op.getAttrOfType<ArrayAttr>("strides");
  llvm::SmallVector<int64_t, 0> strideVal;
  populateSeqVal(strideAttr, strideVal);

  auto dilationAttr = op.getAttrOfType<ArrayAttr>("dilations");
  llvm::SmallVector<int64_t, 0> dilationVal;
  populateSeqVal(dilationAttr, dilationVal);

  auto paddingAttr = op.getAttrOfType<ArrayAttr>("padding");
  llvm::SmallVector<int64_t, 0> paddingVal;
  populateSeqVal(paddingAttr, paddingVal);

  return {opType, dimIndexVal, strideVal, dilationVal, paddingVal};
}

class PopulateParamsBase {
public:
  static void obtainGemmADimKVectorizable(
      mlir::miopen::ConvOpType opType,
      llvm::StringMap<std::pair<size_t, int64_t>> &dimIndexVal,
      bool &input1GemmKVectorizable) {
    // Vectorizable flag is opposite between forwad and bwd_data
    if (opType == mlir::miopen::ConvOpType::Conv2DOpType) {
      // When K is not the fastest changing dimension,
      // gemmK dimension is vectorizable, gemmM is not, and vice versa.
      // Vectorization width depending on which among C, Y, X be the fastest
      // changing dimension.
      if (dimIndexVal["k"].first == 3) {
        input1GemmKVectorizable = false;
      } else {
        input1GemmKVectorizable = true;
      }
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdDataOpType) {
      // When K is the fastest changing dimension(3),
      // gemmK dimension is vectorizable, gemmM is not, and vice versa.
      // Vectorization width depending on length of K.
      if (dimIndexVal["k"].first == 3) {
        input1GemmKVectorizable = true;
      } else {
        input1GemmKVectorizable = false;
      }
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdWeightOpType) {
      // When K is the fastest changing dimension,
      // gemmM dimension is vectorizable, gemmK is not, and vice versa.
      // Vectorization width depending on which among N, and HoWo be the fastest
      // changing dimension.
      if (dimIndexVal["k"].first == 3) {
        input1GemmKVectorizable = false;
      } else {
        input1GemmKVectorizable = true;
      }
    }
  }

  static void obtainGemmBDimKVectorizable(
      mlir::miopen::ConvOpType opType,
      llvm::StringMap<std::pair<size_t, int64_t>> &dimIndexVal,
      bool &input2GemmKVectorizable) {
    // Vectorizable flag is opposite between forwad and bwd_data
    if (opType == mlir::miopen::ConvOpType::Conv2DOpType) {
      // For input tensor.
      // When C is the fastest changing dimension,
      // gemmK dimension is vectorizable, gemmN is not, and vice versa.
      // Vectorization width depending on length of C.
      if (dimIndexVal["ci"].first == 3) {
        input2GemmKVectorizable = true;
      } else {
        input2GemmKVectorizable = false;
      }
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdDataOpType) {
      // For output tensor.
      // When K is the fastest changing dimension(3),
      // gemmK dimension is vectorizable, gemmN is not, and vice versa.
      // Vectorization width depending on length of K.
      if (dimIndexVal["ko"].first == 3) {
        input2GemmKVectorizable = true;
      } else {
        input2GemmKVectorizable = false;
      }
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdWeightOpType) {
      // For input tensor
      // When C is the fastest changing dimension,
      // gemmN dimension is vectorizable, gemmK is not, and vice versa.
      // Vectorization width depending on length of C.
      if (dimIndexVal["ci"].first == 3) {
        input2GemmKVectorizable = false;
      } else {
        input2GemmKVectorizable = true;
      }
    }
  }

  static void obtainFilterVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
    auto dimIndexVal = ctx.dimIndexVal;
    // Vectorization length logic is the same for forward and bwd_data
    if (dimIndexVal["k"].first == 3) {
      vecLen = dimIndexVal["k"].second;
    } else if (dimIndexVal["k"].first == 0) {
      // dimKF is the lowest changing dimension, which means dimC/dimY/dimX
      vecLen = dimIndexVal["c"].second * dimIndexVal["y"].second *
               dimIndexVal["x"].second;
    } else if (dimIndexVal["k"].first == 1) {
      // K's position is at 1, vectorization legnth is last two dimension
      if (dimIndexVal["c"].first == 0) {
        vecLen = dimIndexVal["y"].second * dimIndexVal["x"].second;
      } else if (dimIndexVal["y"].first == 0) {
        vecLen = dimIndexVal["c"].second * dimIndexVal["x"].second;
      } else {
        vecLen = dimIndexVal["c"].second * dimIndexVal["y"].second;
      }
    } else {
      // K's position is 2, vectorization legnth is last dimension
      if (dimIndexVal["c"].first == 3) {
        vecLen = dimIndexVal["c"].second;
      } else if (dimIndexVal["y"].first == 3) {
        vecLen = dimIndexVal["y"].second;
      } else {
        vecLen = dimIndexVal["x"].second;
      }
    }
  }

  static void obtainInputVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
    auto dimIndexVal = ctx.dimIndexVal;
    if (dimIndexVal["ni"].first == 3) {
      vecLen = dimIndexVal["ni"].second;
    } else if (dimIndexVal["ci"].first == 3) {
      vecLen = dimIndexVal["ci"].second;
    } else {
      if (ctx.strideVal[0] == 1 && ctx.strideVal[1] == 1 &&
          ctx.paddingVal[0] == 0 && ctx.paddingVal[1] == 0 &&
          ctx.paddingVal[2] == 0 && ctx.paddingVal[3] == 0)
        vecLen = dimIndexVal["hi"].second * dimIndexVal["wi"].second;
      else
        vecLen = 1;
    }
  }
  static void obtainOutputVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
    auto dimIndexVal = ctx.dimIndexVal;
    if (dimIndexVal["ko"].first == 3) {
      vecLen = dimIndexVal["ko"].second;
    } else if (dimIndexVal["ko"].first == 0) {
      // dimKO is the lowest changing dimension, which means dimN/dimHo/dimWo
      vecLen = dimIndexVal["no"].second * dimIndexVal["ho"].second *
               dimIndexVal["wo"].second;
    } else if (dimIndexVal["ko"].first == 1) {
      // Ko's position is at 1, vectorization legnth is last two dimensions
      if (dimIndexVal["no"].first == 0) {
        vecLen = dimIndexVal["ho"].second * dimIndexVal["wo"].second;
      } else if (dimIndexVal["ho"].first == 0) {
        vecLen = dimIndexVal["no"].second * dimIndexVal["wo"].second;
      } else {
        vecLen = dimIndexVal["no"].second * dimIndexVal["ho"].second;
      }
    } else {
      // K's position is 2, vectorization legnth is last dimension
      if (dimIndexVal["no"].first == 3) {
        vecLen = dimIndexVal["no"].second;
      } else if (dimIndexVal["ho"].first == 3) {
        vecLen = dimIndexVal["ho"].second;
      } else {
        vecLen = dimIndexVal["wo"].second;
      }
    }
  }

  static void obtainGemmAVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
    auto opType = ctx.opType;
    if (opType == mlir::miopen::ConvOpType::Conv2DOpType) {
      obtainFilterVecLen(ctx, vecLen);
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdDataOpType) {
      obtainFilterVecLen(ctx, vecLen);
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdWeightOpType) {
      obtainOutputVecLen(ctx, vecLen);
    }
  }

  static void obtainGemmBVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
    auto opType = ctx.opType;
    if (opType == mlir::miopen::ConvOpType::Conv2DOpType) {
      obtainInputVecLen(ctx, vecLen);
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdDataOpType) {
      obtainOutputVecLen(ctx, vecLen);
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdWeightOpType) {
      obtainInputVecLen(ctx, vecLen);
    }
  }

  static void obtainGemmCVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
    auto opType = ctx.opType;
    if (opType == mlir::miopen::ConvOpType::Conv2DOpType) {
      obtainOutputVecLen(ctx, vecLen);
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdDataOpType) {
      obtainInputVecLen(ctx, vecLen);
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdWeightOpType) {
      obtainFilterVecLen(ctx, vecLen);
    }
  }

protected:
  mlir::LogicalResult calculateInputDerivedParams(InitParams *param,
                                                  int64_t blockSize,
                                                  ConvolutionContext &ctx,
                                                  bool isGemmA,
                                                  DerivedParams &derived) {

    bool gemmPos1Vectorizable = false;
    int64_t vectorizableLength = 0;
    if (isGemmA) {
      obtainGemmADimKVectorizable(ctx.opType, ctx.dimIndexVal,
                                  gemmPos1Vectorizable);
      obtainGemmAVecLen(ctx, vectorizableLength);
    } else {
      obtainGemmBDimKVectorizable(ctx.opType, ctx.dimIndexVal,
                                  gemmPos1Vectorizable);
      obtainGemmBVecLen(ctx, vectorizableLength);
    }

    // calculate threadwise copy size
    int64_t dataPerThreadCopy = 0;
    if (isGemmA) {
      dataPerThreadCopy =
          (param->gemmKPerBlock * param->gemmMPerBlock) / blockSize;
    } else {
      dataPerThreadCopy =
          (param->gemmKPerBlock * param->gemmNPerBlock) / blockSize;
    }

    if (!(dataPerThreadCopy > 0))
      return mlir::failure();

    // srcDataPerRead bounded by size of threadwise copy
    const int64_t vectorizationSize = 4;
    if ((vectorizableLength > 0) && (vectorizableLength % 4 == 0)) {
      derived.srcDataPerRead = gcd(vectorizationSize, dataPerThreadCopy);
    }

    // decide threadwise copy lengths
    const auto dataPerThreadCopyGemmVectorized = derived.srcDataPerRead;
    const auto dataPerThreadCopyGemmNonvectorized =
        dataPerThreadCopy / dataPerThreadCopyGemmVectorized;

    int64_t dataPerThreadCopyGemmPos1 = 0;
    int64_t dataPerThreadCopyGemmPos2 = 0;
    if (gemmPos1Vectorizable) {
      dataPerThreadCopyGemmPos1 = dataPerThreadCopyGemmVectorized;
      dataPerThreadCopyGemmPos2 = dataPerThreadCopyGemmNonvectorized;
    } else {
      dataPerThreadCopyGemmPos1 = dataPerThreadCopyGemmNonvectorized;
      dataPerThreadCopyGemmPos2 = dataPerThreadCopyGemmVectorized;
    }

    // dstDataPerWrite also bounded by size of threadwise copy
    derived.dstDataPerWrite = gcd(vectorizationSize, dataPerThreadCopyGemmPos2);

    // calculate blockwise copy thread cluster lengths
    if (isGemmA) {
      derived.clusterLenGemmPos1 =
          param->gemmKPerBlock / dataPerThreadCopyGemmPos1;
      derived.clusterLenGemmPos2 =
          param->gemmMPerBlock / dataPerThreadCopyGemmPos2;
    } else {
      derived.clusterLenGemmPos1 =
          param->gemmKPerBlock / dataPerThreadCopyGemmPos1;
      derived.clusterLenGemmPos2 =
          param->gemmNPerBlock / dataPerThreadCopyGemmPos2;
    }

    if (!(derived.clusterLenGemmPos1 > 0 && derived.clusterLenGemmPos2 > 0))
      return mlir::failure();

    return mlir::success();
  }

  static void obtainGemmSize(ConvolutionContext &ctx, GemmSize &gemmSize) {
    if (ctx.opType == mlir::miopen::ConvOpType::Conv2DOpType) {
      gemmSize.gemmM = ctx.dimIndexVal["k"].second;
      gemmSize.gemmN = ctx.dimIndexVal["no"].second *
                       ctx.dimIndexVal["ho"].second *
                       ctx.dimIndexVal["wo"].second;
      gemmSize.gemmK = ctx.dimIndexVal["c"].second *
                       ctx.dimIndexVal["y"].second *
                       ctx.dimIndexVal["x"].second;
    } else if (ctx.opType == mlir::miopen::ConvOpType::Conv2DBwdDataOpType) {
      gemmSize.gemmM = ctx.dimIndexVal["c"].second *
                       ctx.dimIndexVal["y"].second *
                       ctx.dimIndexVal["x"].second;
      gemmSize.gemmN = ctx.dimIndexVal["no"].second *
                       ctx.dimIndexVal["ho"].second *
                       ctx.dimIndexVal["wo"].second;
      gemmSize.gemmK = ctx.dimIndexVal["k"].second;
    } else if (ctx.opType == mlir::miopen::ConvOpType::Conv2DBwdWeightOpType) {
      gemmSize.gemmM = ctx.dimIndexVal["k"].second;
      gemmSize.gemmK = ctx.dimIndexVal["no"].second *
                       ctx.dimIndexVal["ho"].second *
                       ctx.dimIndexVal["wo"].second;
      gemmSize.gemmN = ctx.dimIndexVal["c"].second *
                       ctx.dimIndexVal["y"].second *
                       ctx.dimIndexVal["x"].second;
    }
  }

  int64_t obtainGridSize(GemmSize &gemmSize, InitParams *param) {
    return (gemmSize.gemmM / param->gemmMPerBlock) *
           (gemmSize.gemmN / param->gemmNPerBlock);
  }

  mlir::LogicalResult isValidGemm(InitParams *param, GemmSize &gemmSize) {
    if (!(gemmSize.gemmM % param->gemmMPerBlock == 0 &&
          gemmSize.gemmN % param->gemmNPerBlock == 0 &&
          gemmSize.gemmK % param->gemmKPerBlock == 0)) {
      return mlir::failure();
    }
    return mlir::success();
  }
};

class TunableParameters {
public:
  // Default constructor: empty map of params
  TunableParameters() {}

  // params constructor: populate with existing values
  TunableParameters(std::map<std::string, int> parameters)
      : params(parameters) {}

  // yaml constrcutor: Use YAML to capture all parameters
  TunableParameters(llvm::StringRef &&yamlFileName) {
    auto yaml = mlir::openInputFile(yamlFileName);
    assert(yaml != nullptr);
    loadYAML(yaml->getBuffer());
  }

  void print(llvm::raw_ostream &os) {
    for (auto kv : params) {
      os << " -D" << kv.first << "=" << kv.second;
    }
  }
  void dump(llvm::StringRef &&yamlFileName) {
    auto outputYAMLFile = mlir::openOutputFile(yamlFileName);
    if (outputYAMLFile) {
      printYAML(outputYAMLFile->os());
      outputYAMLFile->keep();
    } else {
      llvm::errs() << "\nOpen output file failed: " << yamlFileName << "\n";
    }
  }
  void printYAML(llvm::raw_ostream &os) {
    llvm::yaml::Output xout(os, nullptr, 0);
    xout << params;
    os.flush();
  }
  void loadYAML(llvm::StringRef yaml) {
    params.clear();
    llvm::yaml::Input yin(yaml);
    yin >> params;
  }
  int operator[](llvm::StringRef str) {
    if (params.find(str.str()) != params.end()) {
      return params[str.str()];
    }
    return 0;
  }
  void setValue(llvm::StringRef str, int value) { params[str.str()] = value; }

protected:
  std::map<std::string, int> params;
};

struct InitParamsNonXDL : InitParams {
  InitParamsNonXDL(int64_t mPerBlock, int64_t nPerBlock, int64_t kPerBlock,
                   int64_t mPerThread, int64_t nPerThread, int64_t bSize)
      : InitParams{mPerBlock, nPerBlock, kPerBlock}, gemmMPerThread(mPerThread),
        gemmNPerThread(nPerThread), blockSize(bSize) {}
  int64_t gemmMPerThread;
  int64_t gemmNPerThread;
  int64_t blockSize;
};

class PopulateParams : public PopulateParamsBase {
private:
  llvm::SmallVector<InitParamsNonXDL, 4> initParameters = {
      // M/block N/block K/block M/thread N/thread blockSize
      {128, 128, 8, 4, 4, 256},
      {128, 64, 8, 4, 4, 128},
      {64, 128, 4, 4, 4, 128},
      {32, 32, 4, 2, 2, 64},
  };

  LogicalResult
  calculateGemmABlockCopyPerformanceParameters(InitParamsNonXDL *param,
                                               ConvolutionContext &ctx,
                                               DerivedParams &derived) {
    return calculateInputDerivedParams(param, param->blockSize, ctx, true,
                                       derived);
  }

  LogicalResult
  calculateGemmBBlockCopyPerformanceParameters(InitParamsNonXDL *param,
                                               ConvolutionContext &ctx,
                                               DerivedParams &derived) {

    return calculateInputDerivedParams(param, param->blockSize, ctx, false,
                                       derived);
  }

  int64_t calculateGemmCDestDataPerWrite(ConvolutionContext &ctx) {
    int64_t outputVecLen = 0;
    if ((ctx.opType == miopen::ConvOpType::Conv2DOpType) &&
        (ctx.dimIndexVal["ko"].first == 3)) {
      // gemmM vectorizable. However, there is no parameters for vectorizing
      // gemmM dimension for matrix C. Do nothing here.
    } else if ((ctx.opType == miopen::ConvOpType::Conv2DBwdDataOpType) &&
               (ctx.dimIndexVal["ci"].first == 3)) {
      // gemmM vectorizable. However, there is no parameters for vectorizing
      // gemmM dimension for matrix C. Do nothing here.
    } else {
      obtainGemmCVecLen(ctx, outputVecLen);
    }

    if ((outputVecLen > 0) && (outputVecLen % 4 == 0)) {
      return 4;
    } else if ((outputVecLen > 0) && (outputVecLen % 2 == 0)) {
      return 2;
    }

    return 1;
  }

public:
  LogicalResult paramsFromCtx(ConvolutionContext &ctx,
                              InitParamsNonXDL &validParams, GemmSize &gemmSize,
                              DerivedParams &gemmADerivedParam,
                              DerivedParams &gemmBDerivedParam,
                              int64_t &gemmCDstPerWrite, int64_t &gridSize) {
    LogicalResult res(LogicalResult::Failure);

    obtainGemmSize(ctx, gemmSize);

    for (auto &params : initParameters) {

      res = isValidGemm(&params, gemmSize);
      if (failed(res)) {
        LLVM_DEBUG(llvm::dbgs() << "Gemm size and gemm/block "
                                << "size does not divide exactly.\n");
        continue;
      }

      res = calculateGemmABlockCopyPerformanceParameters(&params, ctx,
                                                         gemmADerivedParam);
      if (failed(res)) {
        LLVM_DEBUG(llvm::dbgs() << "Incoherent gemmA tuning parameter "
                                << " size.\n");
        continue;
      }

      res = calculateGemmBBlockCopyPerformanceParameters(&params, ctx,
                                                         gemmBDerivedParam);

      if (failed(res)) {
        LLVM_DEBUG(llvm::dbgs() << "Incoherent gemmB tuning parameter "
                                << " size.\n");
        continue;
      }

      validParams = params;
      break;
    }

    if (failed(res)) {
      // All initParameters have failed, shouldn't happen
      llvm::errs() << "FATAL ERROR! COULD NOT FIND VALID TUNING PARAMETERS!\n";
      return res;
    }

    gridSize = obtainGridSize(gemmSize, &validParams);
    gemmCDstPerWrite = calculateGemmCDestDataPerWrite(ctx);
    return res;
  }
};

#endif // MLIR_DIALECT_MIOPEN_GRIDWISE_GEMM_PARAMS_H
