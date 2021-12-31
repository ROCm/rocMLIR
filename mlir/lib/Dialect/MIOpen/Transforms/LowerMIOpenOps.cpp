//===- LowerMIOpenOps.cpp - MLIR MIOpen ops lowering passes ---------------===//
//
// Copyright 2020 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This pass converts miopen.conv2d into miopen.transform and
// miopen.gridwise_gemm.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/MIOpen/LowerMIOpenOps.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {
struct LowerMIOpenOpsStep1Pass : public MIOpenOpsStep1PassBase<LowerMIOpenOpsStep1Pass> {
  void runOnOperation() override;
};

struct LowerMIOpenOpsStep2Pass : public MIOpenOpsStep2PassBase<LowerMIOpenOpsStep2Pass> {
  void runOnOperation() override;
};

struct LowerMIOpenOpsStep3Pass : public MIOpenOpsStep3PassBase<LowerMIOpenOpsStep3Pass> {
  void runOnOperation() override;
};

struct LowerMIOpenOpsStep4Pass
    : public MIOpenOpsStep4PassBase<LowerMIOpenOpsStep4Pass> {
  void runOnOperation() override;
};

struct LowerMIOpenOpsStep5Pass
    : public MIOpenOpsStep5PassBase<LowerMIOpenOpsStep5Pass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// High level convolution operation always have
// [filter, input, output]
// as the convolution argument. The only difference between different
// hight level convolution operations is the argument sequence. For
// simplicity, we always arrange the first two arguments to be input
// and the last argument to be output

template <>
const ArgumentFields Conv2DRewritePattern<miopen::Conv2DOp>::fields = {
    {0, 1, 2},
    {"KM", "KN", "MN"},
};
template <>
const miopen::ConvOpType Conv2DRewritePattern<miopen::Conv2DOp>::convOpType =
    miopen::ConvOpType::Fwd;

template <>
const ArgumentFields Conv2DRewritePattern<miopen::Conv2DBwdDataOp>::fields = {
    {0, 2, 1},
    {"KM", "MN", "KN"},
};

template <>
const miopen::ConvOpType
    Conv2DRewritePattern<miopen::Conv2DBwdDataOp>::convOpType =
        miopen::ConvOpType::BwdData;

template <>
const ArgumentFields Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>::fields = {
    {2, 1, 0},
    {"MN", "KN", "KM"},
};

template <>
const miopen::ConvOpType
    Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>::convOpType =
        miopen::ConvOpType::BwdWeight;

// Explicitly instantiate the template to operation type
template struct Conv2DRewritePattern<miopen::Conv2DOp>;
template struct Conv2DRewritePattern<miopen::Conv2DBwdDataOp>;
template struct Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>;

/// Lowerings for particular convolution algorithms (TODO, new file?)
LogicalResult backwardWeightAtomicAdd(miopen::Conv2DBwdWeightOp op,
                                      PatternRewriter &b) {
  auto loc = op.getLoc();
  auto gemmIdAttr = op->template getAttrOfType<IntegerAttr>("gemm_id");
  auto archAttr = op->template getAttrOfType<StringAttr>("arch");
  auto numCuAttr = op->template getAttrOfType<IntegerAttr>("num_cu");

  auto filterLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->template getAttrOfType<ArrayAttr>("input_layout");
  auto outputLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("output_layout");

  auto dilationsAttr = op->template getAttrOfType<ArrayAttr>("dilations");
  auto stridesAttr = op->template getAttrOfType<ArrayAttr>("strides");
  auto paddingAttr = op->template getAttrOfType<ArrayAttr>("padding");

  // Get shape of filter tensor.
  auto filterType = op.filter().getType().template cast<MemRefType>();
  auto filterShape = filterType.getShape();

  // Get shape of input tensor.
  auto inputType = op.input().getType().template cast<MemRefType>();
  auto inputShape = inputType.getShape();

  // Get shape of output tensor.
  auto outputType = op.output().getType().template cast<MemRefType>();
  auto outputShape = outputType.getShape();

  // Obtain convolution parameters: padding / dialtion / stride.
  auto leftPadH =
      paddingAttr.getValue()[0].template cast<IntegerAttr>().getInt();
  auto leftPadW =
      paddingAttr.getValue()[2].template cast<IntegerAttr>().getInt();
  auto rightPadH =
      paddingAttr.getValue()[1].template cast<IntegerAttr>().getInt();
  auto rightPadW =
      paddingAttr.getValue()[3].template cast<IntegerAttr>().getInt();

  auto dilationH =
      dilationsAttr.getValue()[0].template cast<IntegerAttr>().getInt();
  auto dilationW =
      dilationsAttr.getValue()[1].template cast<IntegerAttr>().getInt();
  auto strideH =
      stridesAttr.getValue()[0].template cast<IntegerAttr>().getInt();
  auto strideW =
      stridesAttr.getValue()[1].template cast<IntegerAttr>().getInt();
  // get y, x, ho, wo, hi, wi
  int64_t g, n, k, c, y, x, ho, wo, hi, wi;
  g = n = k = c = y = x = ho = wo = hi = wi = 0;
  llvm::SmallVector<StringRef, 5> filterNames, inputNames, outputNames;
  for (unsigned i = 0; i < filterLayoutAttr.size(); ++i) {
    auto filterAttr =
        filterLayoutAttr.getValue()[i].template cast<StringAttr>();
    auto inputAttr = inputLayoutAttr.getValue()[i].template cast<StringAttr>();
    auto outputAttr =
        outputLayoutAttr.getValue()[i].template cast<StringAttr>();

    filterNames.push_back(filterAttr.getValue());
    inputNames.push_back(inputAttr.getValue());
    outputNames.push_back(outputAttr.getValue());

    if (filterAttr.getValue() == "g") {
      g = filterShape[i];
    } else if (filterAttr.getValue() == "k") {
      k = filterShape[i];
    } else if (filterAttr.getValue() == "c") {
      c = filterShape[i];
    } else if (filterAttr.getValue() == "y") {
      y = filterShape[i];
    } else if (filterAttr.getValue() == "x") {
      x = filterShape[i];
    }

    if (inputAttr.getValue() == "ni") {
      n = inputShape[i];
    } else if (inputAttr.getValue() == "hi") {
      hi = inputShape[i];
    } else if (inputAttr.getValue() == "wi") {
      wi = inputShape[i];
    }

    if (outputAttr.getValue() == "ho") {
      ho = outputShape[i];
    } else if (outputAttr.getValue() == "wo") {
      wo = outputShape[i];
    }
  }
  // calculate gemmKBlocks

  // static const int64_t MaxSubBlockNum = 2048 / standardBockNum;
  int64_t gemmKBlocks = calculateKBlockNum(n, ho, wo);

  Value gemmFilter, gemmInput, gemmOutput;
  OobCheckSet filterOobCheckDims, inputOobCheckDims, outputOobCheckDims;
  // Transform filter tensor.
  {
    // Add a dimension, that'll be ignored when writing the output, for KBlock
    // The existence of this dimension makes the mapping between the C matrix
    // and the filter tensor uninvertable, hence the need for atomic add

    // TODO(kdrewnia): After the refactor goes through, it'll be much less noisy
    // to keep the filter tensor (and other tensors) in their native layout
    // (KCYX/KYXC) and just add the KBlock dimension instead of forcing the
    // layout change. This sort of layout change, especially when done naievely
    // here is something I suspect of causing performance problems. The catch
    // then is that G and KBlock may not be consecutive ... but do we care?
    BottomUpCTBuilder addKBlockTransform(b, filterNames, filterShape, loc);
    addKBlockTransform.passThrough({"g"}, {0}, {"g"});
    addKBlockTransform.addDim("kBlock", 1, gemmKBlocks);
    addKBlockTransform.passThrough({"k", "c", "y", "x"}, {2, 3, 4, 5},
                                   {"k", "c", "y", "x"});

    TransformsAttr addKBlockTransformAttr = addKBlockTransform.get();
    Value withKBlock =
        b.create<miopen::TransformOp>(loc, op.filter(), addKBlockTransformAttr);

    // Create GEMM filter tensor
    // Here, we merge the KBlock dimension into the G dimension
    // and send K to the M dimension and CYX to the N dimension as usual
    auto gemmTransform =
        BottomUpCTBuilder::above(addKBlockTransform, addKBlockTransformAttr);
    gemmTransform.merge("gemmG", 0, {"g", "kBlock"});
    gemmTransform.passThrough({"gemmM"}, {1}, {"k"});
    gemmTransform.merge("gemmN", 2, {"c", "y", "x"});

    TransformsAttr gemmTransformAttr = gemmTransform.get();
    gemmFilter =
        b.create<miopen::TransformOp>(loc, withKBlock, gemmTransformAttr);
    // This kernel is only invoked when there's no need for gemm padding
  }

  // Transform input tensor
  {
    // Force the layout of the input tensor to GNCHW, pad H and W, and split n
    // into n0 and n1 where n0 has size kBlocks and n1 is what's left
    BottomUpCTBuilder firstTransform(b, inputNames, inputShape, loc);
    firstTransform.passThrough({"gi"}, {0}, {"gi"});
    firstTransform.unmerge({"n0", "n1"}, {1, 2}, "ni",
                           {gemmKBlocks, n / gemmKBlocks});
    firstTransform.passThrough({"ci"}, {3}, {"ci"});
    firstTransform.pad({"hipad", "wipad"}, {4, 5}, {"hi", "wi"},
                       {leftPadH, rightPadH, leftPadW, rightPadW});

    TransformsAttr firstTransformAttr = firstTransform.get();
    Value firstTransformed =
        b.create<miopen::TransformOp>(loc, op.input(), firstTransformAttr);

    bool hasHPad = leftPadH != 0 || rightPadH != 0;
    bool hasWPad = leftPadW != 0 || rightPadW != 0;
    if (hasHPad) {
      inputOobCheckDims.insert(firstTransform.startIndex("hi"));
    }
    if (hasWPad) {
      inputOobCheckDims.insert(firstTransform.startIndex("wi"));
    }

    // The usual mapping of input space to dimensions such that filter elements
    // get multiplied by the right thing
    auto embedTransform =
        BottomUpCTBuilder::above(firstTransform, firstTransformAttr);
    embedTransform.passThrough("gi");
    embedTransform.passThrough("n0");
    embedTransform.passThrough("n1");
    embedTransform.passThrough("ci");
    embedTransform.embed({"y", "ho"}, {4, 5}, {y, ho}, "hipad",
                         {dilationH, strideH});
    embedTransform.embed({"x", "wo"}, {6, 7}, {x, wo}, "wipad",
                         {dilationW, strideW});

    TransformsAttr embedTransformAttr = embedTransform.get();
    Value embedded = b.create<miopen::TransformOp>(loc, firstTransformed,
                                                   embedTransformAttr);

    // Merge N1HoWO to gemmK and CYX to gemmN
    auto gemmTransform =
        BottomUpCTBuilder::above(embedTransform, embedTransformAttr);
    gemmTransform.merge("gemmG", 0, {"gi", "n0"});
    gemmTransform.merge("gemmK", 1, {"n1", "ho", "wo"});
    gemmTransform.merge("gemmN", 2, {"ci", "y", "x"});

    TransformsAttr gemmTransformAttr = gemmTransform.get();
    gemmInput = b.create<miopen::TransformOp>(loc, embedded, gemmTransformAttr);
  }

  // Transform output tensor
  {
    // First, split the N dimension as in the input and force a relayout
    BottomUpCTBuilder firstTransform(b, outputNames, outputShape, loc);
    firstTransform.passThrough({"go"}, {0}, {"go"});
    firstTransform.unmerge({"n0", "n1"}, {1, 2}, "no",
                           {gemmKBlocks, n / gemmKBlocks});
    firstTransform.passThrough({"ko", "ho", "wo"}, {3, 4, 5},
                               {"ko", "ho", "wo"});

    TransformsAttr firstTransformAttr = firstTransform.get();
    Value transformed =
        b.create<miopen::TransformOp>(loc, op.output(), firstTransformAttr);

    // Map G and N0 to gemmG, N1HW to gemmK and K to gemmM
    auto gemmTransform =
        BottomUpCTBuilder::above(firstTransform, firstTransformAttr);
    gemmTransform.merge("gemmG", 0, {"go", "n0"});
    gemmTransform.merge("gemmK", 1, {"n1", "ho", "wo"});
    gemmTransform.passThrough({"gemmM"}, {2}, {"ko"});

    TransformsAttr gemmTransformAttr = gemmTransform.get();
    gemmOutput =
        b.create<miopen::TransformOp>(loc, transformed, gemmTransformAttr);
  }

  // Set attributes for gridwise_gemm op.
  llvm::SmallVector<NamedAttribute, 8> gridwiseGemmAttrs{
      b.getNamedAttr("gemm_id", gemmIdAttr), b.getNamedAttr("arch", archAttr),
      b.getNamedAttr("num_cu", numCuAttr)};

  // xdlopsV2.
  auto xdlopsV2Attr = op->template getAttrOfType<BoolAttr>("xdlopsV2");
  bool isXdlops = (xdlopsV2Attr && xdlopsV2Attr.getValue() == true);
  if (isXdlops)
    gridwiseGemmAttrs.push_back(
        b.getNamedAttr("xdlopsV2", b.getBoolAttr(true)));

  gridwiseGemmAttrs.push_back(b.getNamedAttr(
      "kernel_algorithm", b.getStringAttr("backward_weight_v4r4")));

  // This kernel is not run when there is padding on the GEMM
  auto paddingInfo =
      PaddingInfoAttr::get(b.getContext(), 0, 0, 0, BwdPaddingKernelInfo::NA);
  auto dataOperation = InMemoryDataOperation::AtomicAdd;

  Value gemmA = gemmOutput;
  ArrayAttr oobA =
      getBoundsCheckAttr(b, outputOobCheckDims, outputShape.size());
  Value gemmB = gemmInput;
  ArrayAttr oobB = getBoundsCheckAttr(b, inputOobCheckDims, inputShape.size());
  Value gemmC = gemmFilter;
  ArrayAttr oobC =
      getBoundsCheckAttr(b, filterOobCheckDims, filterShape.size());

  if (isXdlops) {
    auto gop = b.create<miopen::GridwiseGemmV2Op>(
        loc, gemmA, gemmB, gemmC, oobA, oobB, oobC, paddingInfo, dataOperation,
        gridwiseGemmAttrs);
    affixGridwiseGemmAttributes(op, gop, b);
  } else {
    op->emitOpError("Backward weight atomic add kernel requires xdlops and "
                    "shouldn't have been tried without them");
  }

  // Finally, erase the original Conv2D op.
  op.erase();

  return success();
}

LogicalResult backwardData(miopen::Conv2DBwdDataOp op, PatternRewriter &b) {
  auto loc = op.getLoc();
  auto gemmIdAttr = op->template getAttrOfType<IntegerAttr>("gemm_id");
  auto archAttr = op->template getAttrOfType<StringAttr>("arch");
  auto numCuAttr = op->template getAttrOfType<IntegerAttr>("num_cu");

  auto filterLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->template getAttrOfType<ArrayAttr>("input_layout");
  auto outputLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("output_layout");

  auto dilationsAttr = op->template getAttrOfType<ArrayAttr>("dilations");
  auto stridesAttr = op->template getAttrOfType<ArrayAttr>("strides");
  auto paddingAttr = op->template getAttrOfType<ArrayAttr>("padding");

  // Get shape of filter tensor.
  auto filterType = op.filter().getType().template cast<MemRefType>();
  auto filterShape = filterType.getShape();

  // Get shape of input tensor.
  auto inputType = op.input().getType().template cast<MemRefType>();
  auto inputShape = inputType.getShape();

  // Get shape of output tensor.
  auto outputType = op.output().getType().template cast<MemRefType>();
  auto outputShape = outputType.getShape();

  // Obtain convolution parameters: padding / dialtion / stride.
  int64_t leftPadH =
      paddingAttr.getValue()[0].template cast<IntegerAttr>().getInt();
  int64_t leftPadW =
      paddingAttr.getValue()[2].template cast<IntegerAttr>().getInt();
  int64_t rightPadH =
      paddingAttr.getValue()[1].template cast<IntegerAttr>().getInt();
  int64_t rightPadW =
      paddingAttr.getValue()[3].template cast<IntegerAttr>().getInt();

  int64_t dilationH =
      dilationsAttr.getValue()[0].template cast<IntegerAttr>().getInt();
  int64_t dilationW =
      dilationsAttr.getValue()[1].template cast<IntegerAttr>().getInt();
  int64_t strideH =
      stridesAttr.getValue()[0].template cast<IntegerAttr>().getInt();
  int64_t strideW =
      stridesAttr.getValue()[1].template cast<IntegerAttr>().getInt();

  // get y, x, ho, wo, hi, wi, k, c, n
  int64_t y, x, ho, wo, hi, wi, k, c, n;
  y = x = ho = wo = hi = wi = k = c = n = 0;
  llvm::SmallVector<StringRef, 5> filterNames, inputNames, outputNames;
  for (uint32_t i = 0; i < filterLayoutAttr.size(); ++i) {
    auto filterAttr =
        filterLayoutAttr.getValue()[i].template cast<StringAttr>();
    auto inputAttr = inputLayoutAttr.getValue()[i].template cast<StringAttr>();
    auto outputAttr =
        outputLayoutAttr.getValue()[i].template cast<StringAttr>();

    filterNames.push_back(filterAttr.getValue());
    inputNames.push_back(inputAttr.getValue());
    outputNames.push_back(outputAttr.getValue());

    if (filterAttr.getValue() == "y") {
      y = filterShape[i];
    } else if (filterAttr.getValue() == "x") {
      x = filterShape[i];
    } else if (filterAttr.getValue() == "k") {
      k = filterShape[i];
    } else if (filterAttr.getValue() == "c") {
      c = filterShape[i];
    }

    if (inputAttr.getValue() == "hi") {
      hi = inputShape[i];
    } else if (inputAttr.getValue() == "wi") {
      wi = inputShape[i];
    } else if (inputAttr.getValue() == "ni") {
      n = inputShape[i];
    }

    if (outputAttr.getValue() == "ho") {
      ho = outputShape[i];
    } else if (outputAttr.getValue() == "wo") {
      wo = outputShape[i];
    }
  }

  if (failed(
          checkNames(filterNames, {"k", "g", "c", "y", "x"}, "filter", op)) ||
      failed(checkNames(inputNames, {"ni", "gi", "ci", "hi", "wi"}, "input",
                        op)) ||
      failed(checkNames(outputNames, {"no", "go", "ko", "ho", "wo"}, "output",
                        op))) {
    return failure();
  }

  int64_t gcdStrideDilationH = math_util::gcd(strideH, dilationH);
  int64_t gcdStrideDilationW = math_util::gcd(strideW, dilationW);

  int64_t yTilda = strideH / gcdStrideDilationH;
  int64_t xTilda = strideW / gcdStrideDilationW;

  int64_t yDot = math_util::integer_divide_ceil(y, yTilda);
  int64_t xDot = math_util::integer_divide_ceil(x, xTilda);

  int64_t hTilda =
      ho + math_util::integer_divide_ceil(dilationH * (y - 1), strideH);
  int64_t wTilda =
      wo + math_util::integer_divide_ceil(dilationW * (x - 1), strideW);

  int64_t iHTildaLeft = math_util::integer_divide_floor(
      std::max(0l, leftPadH - dilationH * (yTilda - 1)), strideH);
  int64_t iWTildaLeft = math_util::integer_divide_floor(
      std::max(0l, leftPadW - dilationW * (xTilda - 1)), strideW);

  int64_t iHTildaRight = std::min(
      hTilda, math_util::integer_divide_ceil(leftPadH + hi - 1, strideH) + 1);
  int64_t iWTildaRight = std::min(
      wTilda, math_util::integer_divide_ceil(leftPadW + wi - 1, strideW) + 1);

  int64_t hTildaSlice = iHTildaRight - iHTildaLeft;
  int64_t wTildaSlice = iWTildaRight - iWTildaLeft;

  auto getGemmId = [&](int kernelId) {
    // kernelId 0 must be gemmId 0
    if (kernelId <= 0)
      return 0;

    llvm::SmallVector<int> gemmIds;
    for (int gemmId = 0; gemmId < yTilda * xTilda; gemmId++) {
      // gemm_k size is different for each GEMM
      const auto iYTilda = gemmId / xTilda;
      const auto iXTilda = gemmId % xTilda;

      auto yDotSlice = math_util::integer_divide_ceil(y - iYTilda, yTilda);
      auto xDotSlice = math_util::integer_divide_ceil(x - iXTilda, xTilda);
      // gemmK must > 0, otherwise not need to run
      if (yDotSlice * xDotSlice > 0) {
        gemmIds.push_back(gemmId);
      }
    }
    assert(gemmIds.size() > static_cast<size_t>(kernelId));
    return gemmIds[kernelId];
  };
  auto gemmId = getGemmId(gemmIdAttr.getInt());
  auto iYTilda = gemmId / xTilda;
  auto iXTilda = gemmId % xTilda;
  auto yDotSlice = math_util::integer_divide_ceil(y - iYTilda, yTilda);
  auto xDotSlice = math_util::integer_divide_ceil(x - iXTilda, xTilda);

  bool needExtraPad = false;
  bool isOriginalKernelSupport = true;
  int64_t gemmMSize, gemmNSize, gemmKSize;
  int64_t gemmMExtra, gemmNExtra, gemmKExtra;
  // backward data only, it's igemm v4r1 algo
  // c is input chaneels , k is output channels
  // n is batch , yDotSlice,xDotSlice computed in above
  gemmMExtra = gemmNExtra = gemmKExtra = 0;
  gemmMSize = c;
  gemmKSize = k * yDotSlice * xDotSlice;
  gemmNSize = n * hTildaSlice * wTildaSlice;

  bool isXdlops = false;
  auto xdlopsV2Attr = op->template getAttrOfType<BoolAttr>("xdlopsV2");
  if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true)
    isXdlops = true;

  bool hasLongSlicesH = (hTildaSlice > ho && strideH > 1);
  bool hasLongSlicesW = (wTildaSlice > wo && strideW > 1);
  bool hasLongSlices = hasLongSlicesH || hasLongSlicesW;
  bool hasHPadding = (leftPadH != 0 || rightPadH != 0);
  bool hasWPadding = (leftPadW != 0 || rightPadW != 0);
  bool hasPadding = hasHPadding || hasWPadding;
  bool isStride1 = (strideH == 1 && strideW == 1);
  bool isStride2Pad1 = ((strideH > 1 || strideW > 1) && hasPadding);

  // we use this lamda to compute extra padding size
  // for example, if gemmM size is 3 and gemmMPerBlock is 64
  // we gemmMExtra is 64 so (gemmM + gemmMExtra )%gemmMPerBlock =0
  auto calculatePaddingKernelSize =
      [&isOriginalKernelSupport, &needExtraPad, gemmMSize, gemmNSize, gemmKSize,
       &gemmMExtra, &gemmNExtra, &gemmKExtra](auto &populateParams) {
        auto configParams = populateParams.getTuningParameters();
        unsigned numOfFailedConfigs = 0;
        for (auto &params : configParams) {
          if (gemmMSize % params.gemmMPerBlock == 0 &&
              gemmKSize % params.gemmKPerBlock == 0 &&
              gemmNSize % params.gemmNPerBlock == 0) {
            isOriginalKernelSupport = true;
            break;
          }
          isOriginalKernelSupport = false;
          numOfFailedConfigs++;
        }

        auto extraParams = populateParams.getUniversalParameters();
        if (numOfFailedConfigs == configParams.size()) {

          needExtraPad = true;
          int gemmMRemain, gemmKRemain, gemmNRemain;

          gemmMRemain = gemmMSize % extraParams.gemmMPerBlock;
          if (gemmMRemain != 0)
            gemmMExtra = extraParams.gemmMPerBlock - gemmMRemain;

          gemmNRemain = gemmNSize % extraParams.gemmNPerBlock;
          if (gemmNRemain != 0)
            gemmNExtra = extraParams.gemmNPerBlock - gemmNRemain;

          gemmKRemain = gemmKSize % extraParams.gemmKPerBlock;
          if (gemmKRemain != 0)
            gemmKExtra = extraParams.gemmKPerBlock - gemmKRemain;

          // llvm::errs() << "gemmMExtra: " << gemmMExtra << "gemmNExtra: " <<
          // gemmNExtra << "gemmKExtra: " << gemmKExtra << "\n";
        }
      }; // calculatePaddingKernelSize end

  if (!isXdlops) {
    PopulateParams populateParams;
    calculatePaddingKernelSize(populateParams);
  } else { // xdlops
    PopulateParamsXDL populateParamsXDL;
    calculatePaddingKernelSize(populateParamsXDL);
  }

  LogicalResult supportedPaddingKernel = isSupportedBackwardDataPaddingKernel(
      isXdlops, isStride2Pad1, gemmMExtra, gemmKExtra, gemmNExtra, op);
  // don't do backward data padding kernel if isSupportPaddingKernel=false
  if (failed(supportedPaddingKernel) && !isOriginalKernelSupport)
    return failure();

  Value gemmFilter, gemmInput, gemmOutput;
  OobCheckSet filterOobCheckDims, inputOobCheckDims, outputOobCheckDims;
  // Transform filter tensor.
  {
    // Transform filter tensor, mapping to (g, k, c, ydot, ytilda, xdot,
    // xtilda) forcing the dimensions into that order and embedding (Why the
    // particular embed coefficients is in a presentation somewhere)
    BottomUpCTBuilder firstTransform(b, filterNames, filterShape, loc);

    firstTransform.passThrough({"g", "k", "c"}, {0, 1, 2}, {"g", "k", "c"});
    firstTransform.embed({"ydot", "ytilda"}, {3, 4}, {yDot, yTilda}, "y",
                         {strideH / gcdStrideDilationH, 1});
    firstTransform.embed({"xdot", "xtilda"}, {5, 6}, {xDot, xTilda}, "x",
                         {strideW / gcdStrideDilationW, 1});

    TransformsAttr firstTransformAttr = firstTransform.get();
    Value firstTransformedFilter =
        b.create<miopen::TransformOp>(loc, op.filter(), firstTransformAttr);

    // Take slices in the ydot, ytilda, xdot, and xtilda dimensions
    // to reflect which kernel we're performing
    auto sliceTransform =
        BottomUpCTBuilder::above(firstTransform, firstTransformAttr);
    sliceTransform.passThrough({"g", "k", "c"}, {0, 1, 2}, {"g", "k", "c"});
    sliceTransform.slice({"ydotslice", "xdotslice"}, {"ydot", "xdot"}, {0, 0},
                         {yDotSlice, xDotSlice});
    sliceTransform.slice({"ytildaslice", "xtildaslice"}, {"ytilda", "xtilda"},
                         {iYTilda, iXTilda}, {iYTilda + 1, iXTilda + 1});

    TransformsAttr sliceTransformAttr = sliceTransform.get();
    Value slicedFilter = b.create<miopen::TransformOp>(
        loc, firstTransformedFilter, sliceTransformAttr);

    // Set up gemm by passing g -> gemmG, merging
    // [k, ydotslice, xdotslice] to gemmK, and [c, ytildaslice, xtildaslice]
    // to gemmM
    auto gemmTransform =
        BottomUpCTBuilder::above(sliceTransform, sliceTransformAttr);
    gemmTransform.passThrough({"gemmG"}, {0}, {"g"});
    gemmTransform.merge("gemmK", 1, {"k", "ydotslice", "xdotslice"});
    gemmTransform.merge("gemmM", 2, {"c", "ytildaslice", "xtildaslice"});

    TransformsAttr gemmTransformAttr = gemmTransform.get();
    gemmFilter =
        b.create<miopen::TransformOp>(loc, slicedFilter, gemmTransformAttr);

    // Filter padding
    bool filterCheckPadGemmM = (gemmMExtra > 0);
    bool filterCheckPadGemmK = (gemmKExtra > 0);
    if (filterCheckPadGemmM || filterCheckPadGemmK) {
      auto padTransform =
          BottomUpCTBuilder::above(gemmTransform, gemmTransformAttr);
      padTransform.passThrough("gemmG");
      if (filterCheckPadGemmK) {
        if (isXdlops) {
          padTransform.pad("gemmKPad", "gemmK", gemmKExtra, 0);
        } else {
          padTransform.pad("gemmKPad", "gemmK", 0, gemmKExtra);
        }
        // First dimension only due to the optimizer bug
        filterOobCheckDims.insert(firstTransform.startIndex("k"));
      } else {
        padTransform.passThrough("gemmK");
      }

      if (filterCheckPadGemmM) {
        if (isXdlops) {
          padTransform.pad("gemmMPad", "gemmM", gemmMExtra, 0);
        } else {
          padTransform.pad("gemmMPad", "gemmM", 0, gemmMExtra);
        }
        filterOobCheckDims.insert(firstTransform.startIndex("c"));
      } else {
        padTransform.passThrough("gemmM");
      }

      TransformsAttr padTransformAttr = padTransform.get();
      // Replace filter gemm with padded version
      gemmFilter =
          b.create<miopen::TransformOp>(loc, gemmFilter, padTransformAttr);
    }
  }

  // outside its usual scope so we can look up input tensor dim order
  // for the backwards padding kernel info
  BottomUpCTBuilder toGNCHWPaddedTransform(b, inputNames, inputShape, loc);
  // Transform input tensor
  {
    toGNCHWPaddedTransform.passThrough({"gi"}, {0}, {"gi"});
    toGNCHWPaddedTransform.passThrough({"ni"}, {1}, {"ni"});
    toGNCHWPaddedTransform.passThrough({"ci"}, {2}, {"ci"});
    toGNCHWPaddedTransform.pad({"hipad", "wipad"}, {3, 4}, {"hi", "wi"},
                               {leftPadH, rightPadH, leftPadW, rightPadW});

    TransformsAttr toGNCHWPaddedTransformAttr = toGNCHWPaddedTransform.get();
    Value gnchwPaddedInput = b.create<miopen::TransformOp>(
        loc, op.input(), toGNCHWPaddedTransformAttr);

    // FIXME:  wTildaSlice > wo will let stride2 backwaed data kernel
    // fail, so when (wTildaSlice > wo),  h and w dim check is must but if
    // stride =1 and wTildaSlice > wo , don't do additional check or the
    // padding kernel will fail due compiler issue
    // if stride = 1, slice will make it not out range
    bool needOobChecks = (hasLongSlices || (!hasPadding && !isStride1));
    if (needOobChecks) {
      if (hasHPadding || hasLongSlicesH) {
        inputOobCheckDims.insert(toGNCHWPaddedTransform.startIndex("hi"));
      }
      if (hasWPadding || hasLongSlicesW) {
        inputOobCheckDims.insert(toGNCHWPaddedTransform.startIndex("wi"));
      }
    }

    // Split hipad, wipad into g, n, c, ytilda, htilda, xtilda, wtilda
    auto tildaEmbedTransform = BottomUpCTBuilder::above(
        toGNCHWPaddedTransform, toGNCHWPaddedTransformAttr);
    tildaEmbedTransform.passThrough({"gi", "ni", "ci"}, {0, 1, 2},
                                    {"gi", "ni", "ci"});
    tildaEmbedTransform.embed({"ytilda", "htilda"}, {3, 4}, {yTilda, hTilda},
                              "hipad", {dilationH, strideH});
    tildaEmbedTransform.embed({"xtilda", "wtilda"}, {5, 6}, {xTilda, wTilda},
                              "wipad", {dilationW, strideW});

    TransformsAttr tildaEmbedTransformAttr = tildaEmbedTransform.get();
    Value tildaEmbedded = b.create<miopen::TransformOp>(
        loc, gnchwPaddedInput, tildaEmbedTransformAttr);

    // Slice all the tilda dimensions: ytilda and xtilda get slices of length
    // 1 while htilda and wtilda have slice indices computed above
    auto sliceTransform =
        BottomUpCTBuilder::above(tildaEmbedTransform, tildaEmbedTransformAttr);
    sliceTransform.passThrough({"gi", "ni", "ci"}, {0, 1, 2},
                               {"gi", "ni", "ci"});
    sliceTransform.slice({"yslice", "xslice"}, {"ytilda", "xtilda"},
                         {iYTilda, iXTilda}, {iYTilda + 1, iXTilda + 1});
    sliceTransform.slice({"hslice", "wslice"}, {"htilda", "wtilda"},
                         {iHTildaLeft, iWTildaLeft},
                         {iHTildaRight, iWTildaRight});

    TransformsAttr sliceTransformAttr = sliceTransform.get();
    Value sliced =
        b.create<miopen::TransformOp>(loc, tildaEmbedded, sliceTransformAttr);

    // C plus the length 1 slices (yslice and xslice) become the gemmM
    // dimension G, N, and the h and w slices become gemmN
    auto gemmTransform =
        BottomUpCTBuilder::above(sliceTransform, sliceTransformAttr);
    gemmTransform.passThrough({"gemmG"}, {0}, {"gi"});
    gemmTransform.merge("gemmM", 1, {"ci", "yslice", "xslice"});
    gemmTransform.merge("gemmN", 2, {"ni", "hslice", "wslice"});

    TransformsAttr gemmTransformAttr = gemmTransform.get();
    gemmInput = b.create<miopen::TransformOp>(loc, sliced, gemmTransformAttr);

    bool inputCheckPadGemmM = (gemmMExtra > 0);
    bool inputCheckPadGemmN = (gemmNExtra > 0);
    if (inputCheckPadGemmM || inputCheckPadGemmN) {
      auto padTransform =
          BottomUpCTBuilder::above(gemmTransform, gemmTransformAttr);
      padTransform.passThrough("gemmG");
      if (inputCheckPadGemmM) {
        if (isXdlops) {
          padTransform.pad("gemmMPad", "gemmM", gemmMExtra, 0);
        } else {
          padTransform.pad("gemmMPad", "gemmM", 0, gemmMExtra);
        }
        // gemmM is c
        inputOobCheckDims.insert(toGNCHWPaddedTransform.startIndex("ci"));
      } else {
        padTransform.passThrough("gemmM");
      }

      if (inputCheckPadGemmN) {
        if (isXdlops) {
          padTransform.pad("gemmNPad", "gemmN", gemmNExtra, 0);
        } else {
          padTransform.pad("gemmNPad", "gemmN", 0, gemmNExtra);
        }
        inputOobCheckDims.insert(toGNCHWPaddedTransform.startIndex("ni"));
      } else {
        padTransform.passThrough("gemmN");
      }

      TransformsAttr padTransformAttr = padTransform.get();
      // Replace input gemm with padded version
      gemmInput =
          b.create<miopen::TransformOp>(loc, gemmInput, padTransformAttr);
    }
  }

  // Transform output tensor
  {
    // Embed ho to ydot and htilda and wo to xdot and ytilda
    BottomUpCTBuilder embedTransform(b, outputNames, outputShape, loc);
    embedTransform.passThrough({"go"}, {0}, {"go"});
    embedTransform.passThrough({"no"}, {1}, {"no"});
    embedTransform.passThrough({"ko"}, {2}, {"ko"});
    embedTransform.embed({"ydot", "htilda"}, {3, 4}, {yDot, hTilda}, "ho",
                         {(-dilationH) / gcdStrideDilationH, 1});
    embedTransform.embed({"xdot", "wtilda"}, {5, 6}, {xDot, wTilda}, "wo",
                         {(-dilationW) / gcdStrideDilationW, 1});

    TransformsAttr embedTransformAttr = embedTransform.get();
    Value embedded =
        b.create<miopen::TransformOp>(loc, op.output(), embedTransformAttr);

    if (y > 1) {
      if (!((leftPadH == rightPadH) && (y - leftPadH == 1))) {
        outputOobCheckDims.insert(embedTransform.startIndex("ho"));
      }
    }
    if (x > 1) {
      if (!((leftPadW == rightPadW) && (x - leftPadW == 1))) {
        outputOobCheckDims.insert(embedTransform.startIndex("wo"));
      }
    }

    // Take the same slices in ydot, xdot, htilda, and wtilda as were taken in
    // the filter and input
    auto sliceTransform =
        BottomUpCTBuilder::above(embedTransform, embedTransformAttr);
    sliceTransform.passThrough({"go", "no", "ko"}, {0, 1, 2},
                               {"go", "no", "ko"});
    sliceTransform.slice({"yslice", "xslice"}, {"ydot", "xdot"}, {0, 0},
                         {yDotSlice, xDotSlice});
    sliceTransform.slice({"hslice", "wslice"}, {"htilda", "wtilda"},
                         {iHTildaLeft, iWTildaLeft},
                         {iHTildaRight, iWTildaRight});

    TransformsAttr sliceTransformAttr = sliceTransform.get();
    Value sliced =
        b.create<miopen::TransformOp>(loc, embedded, sliceTransformAttr);

    // Merge k, yslice, and xslice to gemmK and n, hslice, and wslice to gemmN
    auto gemmTransform =
        BottomUpCTBuilder::above(sliceTransform, sliceTransformAttr);
    gemmTransform.passThrough({"gemmG"}, {0}, {"go"});
    gemmTransform.merge("gemmK", 1, {"ko", "yslice", "xslice"});
    gemmTransform.merge("gemmN", 2, {"no", "hslice", "wslice"});

    TransformsAttr gemmTransformAttr = gemmTransform.get();
    gemmOutput = b.create<miopen::TransformOp>(loc, sliced, gemmTransformAttr);

    bool outputCheckPadGemmK = (gemmKExtra > 0);
    bool outputCheckPadGemmN = (gemmNExtra > 0);
    if (outputCheckPadGemmK || outputCheckPadGemmN) {
      auto padTransform =
          BottomUpCTBuilder::above(gemmTransform, gemmTransformAttr);
      padTransform.passThrough("gemmG");
      if (outputCheckPadGemmK) {
        if (isXdlops) {
          padTransform.pad("gemmKPad", "gemmK", gemmKExtra, 0);
        } else {
          padTransform.pad("gemmKPad", "gemmK", 0, gemmKExtra);
        }
        // gemmM is k - the usual standing optimizer bug applies here
        outputOobCheckDims.insert(embedTransform.startIndex("ko"));
      } else {
        padTransform.passThrough("gemmK");
      }

      if (outputCheckPadGemmN) {
        if (isXdlops) {
          padTransform.pad("gemmNPad", "gemmN", gemmNExtra, 0);
        } else {
          padTransform.pad("gemmNPad", "gemmN", 0, gemmNExtra);
        }
        outputOobCheckDims.insert(embedTransform.startIndex("no"));
      } else {
        padTransform.passThrough("gemmN");
      }

      TransformsAttr padTransformAttr = padTransform.get();
      // Replace output gemm with padded version
      gemmOutput =
          b.create<miopen::TransformOp>(loc, gemmOutput, padTransformAttr);
    }
  }

  // Set attributes for gridwise_gemm op.
  llvm::SmallVector<NamedAttribute, 8> gridwiseGemmAttrs{
      b.getNamedAttr("gemm_id", gemmIdAttr), b.getNamedAttr("arch", archAttr),
      b.getNamedAttr("num_cu", numCuAttr)};
  // xdlopsV2.
  if (isXdlops)
    gridwiseGemmAttrs.push_back(
        b.getNamedAttr("xdlopsV2", b.getBoolAttr(true)));
  gridwiseGemmAttrs.push_back(b.getNamedAttr(
      "kernel_algorithm", b.getStringAttr("backward_data_v4r1")));

  // Set up which backward data padding kernel hacks we need
  BwdPaddingKernelInfo hacks = BwdPaddingKernelInfo::NA;
  if (strideH > 1 && strideW > 1 && hasPadding) {
    hacks = hacks | BwdPaddingKernelInfo::StrideTwo;

    uint32_t inputN = toGNCHWPaddedTransform.startIndex("ni");
    uint32_t inputC = toGNCHWPaddedTransform.startIndex("ci");
    uint32_t inputH = toGNCHWPaddedTransform.startIndex("hi");
    uint32_t inputW = toGNCHWPaddedTransform.startIndex("wi");

    if (inputN < inputC && inputC + 1 == inputH && inputH + 1 == inputW) {
      hacks = hacks | BwdPaddingKernelInfo::isNCHW;
    }
    // We don't collect padding info in the nonxdlops case because that
    // isn't needed yet
    if (isXdlops) {
      hacks = hacks | BwdPaddingKernelInfo::Xdlops;
      if (gemmMExtra > 0) {
        hacks = hacks | BwdPaddingKernelInfo::PadM;
      }
      if (gemmNExtra > 0) {
        hacks = hacks | BwdPaddingKernelInfo::PadN;
      }
    }
  }

  auto paddingInfo = PaddingInfoAttr::get(b.getContext(), gemmMExtra,
                                          gemmKExtra, gemmNExtra, hacks);

  Value gemmA = gemmFilter;
  ArrayAttr oobA =
      getBoundsCheckAttr(b, filterOobCheckDims, filterShape.size());
  Value gemmB = gemmOutput;
  ArrayAttr oobB =
      getBoundsCheckAttr(b, outputOobCheckDims, outputShape.size());
  Value gemmC = gemmInput;
  ArrayAttr oobC = getBoundsCheckAttr(b, inputOobCheckDims, inputShape.size());
  // Emit miopen.gridwise_gemm op.
  // Emit miopen.gridwise_gemm_v2 if using xdlops
  if (isXdlops) {
    auto gop = b.create<miopen::GridwiseGemmV2Op>(
        loc, gemmA, gemmB, gemmC, oobA, oobB, oobC, paddingInfo,
        InMemoryDataOperation::Set, gridwiseGemmAttrs);
    affixGridwiseGemmAttributes(op, gop, b);
  } else {
    auto gop =
        b.create<miopen::GridwiseGemmOp>(loc, gemmA, gemmB, gemmC, oobA, oobB,
                                         oobC, paddingInfo, gridwiseGemmAttrs);
    affixGridwiseGemmAttributes(op, gop, b);
  }
  // Finally, erase the original Conv2D op.
  op.erase();

  return success();
}

void LowerMIOpenOpsStep1Pass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  OwningRewritePatternList patterns(ctx);
  patterns.insert<Conv2DRewritePattern<miopen::Conv2DOp>>(ctx);
  patterns.insert<Conv2DRewritePattern<miopen::Conv2DBwdDataOp>>(ctx);
  patterns.insert<Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

void LowerMIOpenOpsStep2Pass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  OwningRewritePatternList patterns(ctx);
  patterns.insert<GridwiseGemmRewritePattern>(ctx);
  patterns.insert<GridwiseGemmV2RewritePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

void LowerMIOpenOpsStep3Pass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  OwningRewritePatternList patterns(ctx);
  patterns.insert<FillRewritePattern>(ctx);
  patterns.insert<SubviewRewritePattern>(ctx);
  patterns.insert<TransformRewritePattern>(ctx);
  patterns.insert<BlockwiseGemmRewritePattern>(ctx);
  patterns.insert<BlockwiseGemmV2RewritePattern>(ctx);
  patterns.insert<BlockwiseLoadRewritePattern>(ctx);
  patterns.insert<BlockwiseStoreRewritePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

void LowerMIOpenOpsStep4Pass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  OwningRewritePatternList patterns(ctx);
  patterns.insert<InWarpTransposeRewritePattern>(ctx);
  patterns.insert<ThreadwiseGemmRewritePattern>(ctx);
  patterns.insert<ThreadwiseCopyRewritePattern>(ctx);
  patterns.insert<ThreadwiseLoadRewritePattern>(ctx);
  patterns.insert<ThreadwiseStoreRewritePattern>(ctx);
  patterns.insert<ThreadwiseCopyV2RewritePattern>(ctx);
  patterns.insert<XdlopsGemmV2RewritePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

void LowerMIOpenOpsStep5Pass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  OwningRewritePatternList patterns(ctx);
  populateAffineToStdConversionPatterns(patterns);
  populateLoopToStdConversionPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep1Pass() {
  return std::make_unique<LowerMIOpenOpsStep1Pass>();
}

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep2Pass() {
  return std::make_unique<LowerMIOpenOpsStep2Pass>();
}

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep3Pass() {
  return std::make_unique<LowerMIOpenOpsStep3Pass>();
}

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep4Pass() {
  return std::make_unique<LowerMIOpenOpsStep4Pass>();
}

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep5Pass() {
  return std::make_unique<LowerMIOpenOpsStep5Pass>();
}
