#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/Target/MIOpenCPP.h"

#define DEBUG_TYPE "miopen-tuning-parameter"

using namespace mlir;

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
