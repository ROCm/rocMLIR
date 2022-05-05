//===- ConvToGemm.cpp - MLIR MIOpen ops lowering passes ------------===//
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
// ============================================================
//
// This pass converts miopen.conv2d into miopen.transform and
// miopen.gridwise_gemm.
//
//===-----------------------------------------------------===//
#include "PassDetail.h"

#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/TransformMapBuilder.h"
#include "mlir/Dialect/MIOpen/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/MIOpen/Tuning/UtilityParams.h"
#include "mlir/Dialect/MIOpen/utility/builderUtils.h"
#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::miopen;
//===----------------------------------------------------------------------===//
// Conv2D (forward, backward) lowering.
//===----------------------------------------------------------------------===//
// High level convolution operation always have
// [filter, input, output]
// as the convolution argument. The only difference between different
// hight level convolution operations is the argument sequence. For
// simplicity, we always arrange the first two arguments to be input
// and the last argument to be output

namespace {
// The ArgumentFields keep track of differences between conv operations
struct ArgumentFields {
  int gridwiseGemmArgumentPosition[3];
  StringRef gemmTargetCharName[3];
};

struct LowerMIOpenOpsStep1Pass
    : public MIOpenOpsStep1PassBase<LowerMIOpenOpsStep1Pass> {
  void runOnOperation() override;
};

LogicalResult
isSupportedBackwardDataPaddingKernel(bool isXdlops, bool isStride2Pad1,
                                     int64_t gemmMExtra, int64_t gemmKExtra,
                                     int64_t gemmNExtra, Conv2DBwdDataOp &op) {
  if (gemmNExtra && gemmKExtra) {
    return op.emitOpError(
        "can't support backward data padding kernel when both pad "
        "gemmN and gemmK due to load issue\n");
  }

  if (isXdlops && (gemmMExtra || gemmNExtra)) {
    if (isStride2Pad1) {
      return op->emitOpError(
          "can't support backward data padding kernel when xdlops stride 2 "
          "pad_h,pad_w>0 and pad gemmM or gemmN due to store issue\n");
    }
  }
  return success();
}

template <typename T>
LogicalResult checkNames(ArrayRef<StringRef> actual,
                         ArrayRef<StringRef> expected, StringRef argName,
                         T op) {
  if (actual.size() != expected.size()) {
    return op.emitOpError("Layout mismatch in ")
           << argName << " tensor: Expected " << expected.size()
           << " dimensions but have " << actual.size();
  }
  for (StringRef name : expected) {
    if (std::find(actual.begin(), actual.end(), name) == actual.end()) {
      return op.emitOpError("Layout mismatch in ")
             << argName << " tensor: Expected it to have a `" << name
             << "` dimension";
    }
  }
  return success();
}

void affixGridwiseGemmAttributes(Operation *convOp, Operation *gop,
                                 OpBuilder &b) {
  gop->setAttr("block_size", convOp->getAttr("block_size"));
  gop->setAttr("m_per_block", convOp->getAttr("m_per_block"));
  gop->setAttr("n_per_block", convOp->getAttr("n_per_block"));
  gop->setAttr("k_per_block", convOp->getAttr("k_per_block"));
  gop->setAttr("matrix_a_dest_data_per_write_dim_m",
               convOp->getAttr("matrix_a_dest_data_per_write_dim_m"));
  gop->setAttr("matrix_a_source_data_per_read",
               convOp->getAttr("matrix_a_source_data_per_read"));
  gop->setAttr("matrix_a_source_vector_read_dim",
               convOp->getAttr("matrix_a_source_vector_read_dim"));
  gop->setAttr("matrix_b_dest_data_per_write_dim_n",
               convOp->getAttr("matrix_b_dest_data_per_write_dim_n"));
  gop->setAttr("matrix_b_source_data_per_read",
               convOp->getAttr("matrix_b_source_data_per_read"));
  gop->setAttr("matrix_b_source_vector_read_dim",
               convOp->getAttr("matrix_b_source_vector_read_dim"));
  gop->setAttr("matrix_c_data_per_copy",
               convOp->getAttr("matrix_c_data_per_copy"));
  gop->setAttr("matrix_c_dest_vector_write_dim",
               convOp->getAttr("matrix_c_dest_vector_write_dim"));
  gop->setAttr("matrix_c_source_vector_read_dim",
               convOp->getAttr("matrix_c_source_vector_read_dim"));

  auto xdlopsV2Attr = convOp->getAttrOfType<BoolAttr>("xdlopsV2");
  if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true) {
    gop->setAttr("m_per_wave", convOp->getAttr("m_per_wave"));
    gop->setAttr("n_per_wave", convOp->getAttr("n_per_wave"));
  } else {
    gop->setAttr("m_per_thread", convOp->getAttr("m_per_thread"));
    gop->setAttr("n_per_thread", convOp->getAttr("n_per_thread"));
    gop->setAttr("k_per_thread", convOp->getAttr("k_per_thread"));
    gop->setAttr("m_level0_cluster", convOp->getAttr("m_level0_cluster"));
    gop->setAttr("m_level1_cluster", convOp->getAttr("m_level1_cluster"));
    gop->setAttr("n_level0_cluster", convOp->getAttr("n_level0_cluster"));
    gop->setAttr("n_level1_cluster", convOp->getAttr("n_level1_cluster"));
  }
}

void createElementwiseLoop(OpBuilder &b, Location loc, int64_t bound,
                           function_ref<void(Value)> emitBodyFunc) {
  // Pseudo code:
  // size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  // size_t stride = hipBlockDim_x * hipGridDim_x;
  // for (size_t i = offset; i < sizeof(collapsedOutput); i+= stride)
  //     collapsedOutput[i] = 0;

  auto workgroupId = b.create<WorkgroupIdOp>(loc, b.getIndexType());
  auto workgroupDim = b.create<ConstantIndexOp>(loc, kUtilityKernelBlockSize);
  auto workitemId = b.create<WorkitemIdOp>(loc, b.getIndexType());
  auto offset = b.create<AddIOp>(
      loc, b.create<MulIOp>(loc, workgroupId, workgroupDim), workitemId);
  auto gridDim = b.create<ConstantIndexOp>(loc, kUtilityKernelGridSize);
  auto stride = b.create<MulIOp>(loc, workgroupDim, gridDim);

  auto loop = b.create<scf::ForOp>(
      loc, offset, b.create<ConstantIndexOp>(loc, bound), stride);
  b.setInsertionPointToStart(loop.getBody());
  emitBodyFunc(loop.getInductionVar());
}

Value createKPackLogic(OpBuilder &b, Location loc, Value source,
                       BottomUpTMBuilder &sourceTransform,
                       TransformMapAttr &sourceTransformAttr, int64_t KPack) {
  Value result = source;
  if (KPack > 1) {
    BottomUpTMBuilder kpackGemmTransform =
        BottomUpTMBuilder::above(sourceTransform, sourceTransformAttr);

    // Passthrough gemmG (dim 0) and gemmN (dim 2).
    kpackGemmTransform.passThrough(sourceTransform.endName(0));
    kpackGemmTransform.passThrough(sourceTransform.endName(2));
    // Use Unmerge to split gemmK (dim 1) into gemmK and gemmKPack, place
    // gemmKPack at dim 3.
    int64_t gemmKLength = sourceTransform.endSize(1);
    auto gemmKName = sourceTransform.endName(1);
    kpackGemmTransform.unmerge({gemmKName, "gemmKPack"}, {1, 3}, gemmKName,
                               {gemmKLength / KPack, KPack});
    TransformMapAttr kpackGemmTransformAttr = kpackGemmTransform.get();
    result = b.create<TransformOp>(loc, source, kpackGemmTransformAttr);
  }
  return result;
}

/// 0-initialize the output for a backward data convolution.
/// The output is the input tensor.
LogicalResult zeroInit(Conv2DBwdDataOp op, PatternRewriter &b) {
  auto loc = op.getLoc();
  Value output = op.input();
  auto outputDataType = output.getType().cast<MemRefType>().getElementType();
  auto collapsedOutput = createCollapseShapeOp(b, loc, output);
  ArrayRef<int64_t> collapsedOutputShape =
      collapsedOutput.getType().cast<MemRefType>().getShape();

  createElementwiseLoop(b, loc, collapsedOutputShape[0], [&](Value iv) {
    auto zeroOp = createZeroConstantOp(b, loc, outputDataType);
    b.create<memref::StoreOp>(loc, zeroOp, collapsedOutput, iv);
  });

  b.eraseOp(op);
  return success();
}

/// 0-initialize the output for a backward weight convolution which uses
/// atomic adds.
/// For f32 type, the output is the filter tensor.
/// for f16 type, the output is the workspace.
LogicalResult zeroInit(Conv2DBwdWeightOp op, PatternRewriter &b) {
  auto loc = op.getLoc();
  auto filterDataType =
      op.filter().getType().cast<MemRefType>().getElementType();
  Value output;
  if (filterDataType == b.getF32Type()) {
    output = op.filter();
  } else if (filterDataType == b.getF16Type()) {
    assert(op.workspace() && "Op has no workspace");
    output = op.workspace();
  } else {
    llvm_unreachable("Incorrect memref type supplied");
  }
  auto outputDataType = output.getType().cast<MemRefType>().getElementType();
  auto collapsedOutput = createCollapseShapeOp(b, loc, output);
  ArrayRef<int64_t> collapsedOutputShape =
      collapsedOutput.getType().cast<MemRefType>().getShape();

  createElementwiseLoop(b, loc, collapsedOutputShape[0], [&](Value iv) {
    auto zeroOp = createZeroConstantOp(b, loc, outputDataType);
    b.create<memref::StoreOp>(loc, zeroOp, collapsedOutput, iv);
  });

  b.eraseOp(op);
  return success();
}

/// Element-wise conversion from the workspace to the output (filter tensor)
/// for a backward weight convolution which uses atomic adds.
LogicalResult elementwiseConversion(Conv2DBwdWeightOp op, PatternRewriter &b) {
  auto loc = op.getLoc();
  assert(op.workspace() && "Op has no workspace");
  auto filter = op.filter();
  auto workspace = op.workspace();
  auto filterDataType = filter.getType().cast<MemRefType>().getElementType();
  auto collapsedFilter = createCollapseShapeOp(b, loc, filter);
  auto collapsedWorkspace = createCollapseShapeOp(b, loc, workspace);
  ArrayRef<int64_t> collapsedFilterShape =
      collapsedFilter.getType().cast<MemRefType>().getShape();
  ArrayRef<int64_t> collapsedWorkspaceShape =
      collapsedWorkspace.getType().cast<MemRefType>().getShape();
  assert((collapsedFilterShape[0] == collapsedWorkspaceShape[0]) &&
         "Filter tensor and workspace size mismatch");

  createElementwiseLoop(b, loc, collapsedWorkspaceShape[0], [&](Value iv) {
    auto loadedValue = b.create<memref::LoadOp>(loc, collapsedWorkspace, iv);
    auto convertedValue =
        createTypeConversionOp(b, loc, loadedValue, filterDataType);
    b.create<memref::StoreOp>(loc, convertedValue, collapsedFilter, iv);
  });

  b.eraseOp(op);
  return success();
}

/// Lowerings for particular convolution algorithms (TODO, new file?)
LogicalResult backwardWeightAtomicAdd(Conv2DBwdWeightOp op,
                                      PatternRewriter &b) {
  auto loc = op.getLoc();
  auto gemmIdAttr = op->template getAttrOfType<IntegerAttr>("gemm_id");
  auto archAttr = op->template getAttrOfType<StringAttr>("arch");
  auto numCuAttr = op->template getAttrOfType<IntegerAttr>("num_cu");
  int64_t numCu = numCuAttr.getInt();

  auto KPackAttr = op->template getAttrOfType<IntegerAttr>("kpack");
  int64_t KPack = KPackAttr.getInt();

  auto MPerBlockAttr = op->template getAttrOfType<IntegerAttr>("m_per_block");
  auto NPerBlockAttr = op->template getAttrOfType<IntegerAttr>("n_per_block");
  auto KPerBlockAttr = op->template getAttrOfType<IntegerAttr>("k_per_block");
  int64_t MPerBlock = MPerBlockAttr.getInt();
  int64_t NPerBlock = NPerBlockAttr.getInt();
  int64_t KPerBlock = KPerBlockAttr.getInt();

  auto filterLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->template getAttrOfType<ArrayAttr>("input_layout");
  auto outputLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("output_layout");

  auto dilationsAttr = op->template getAttrOfType<ArrayAttr>("dilations");
  auto stridesAttr = op->template getAttrOfType<ArrayAttr>("strides");
  auto paddingAttr = op->template getAttrOfType<ArrayAttr>("padding");

  auto xdlopsV2Attr = op->template getAttrOfType<BoolAttr>("xdlopsV2");
  bool isXdlops = (xdlopsV2Attr && xdlopsV2Attr.getValue() == true);

  // Get shape of filter tensor.
  auto filterType = op.filter().getType().template cast<MemRefType>();
  auto filterShape = filterType.getShape();

  // Determine whether to use workspace.
  bool hasWorkspace =
      (filterType.getElementType() == b.getF16Type() && isXdlops);
  if (hasWorkspace) {
    assert(op.workspace() && "Op has no workspace");
  }

  // Emit utility kernels.
  int64_t gemmId = gemmIdAttr.getInt();
  assert((gemmId >= 0) && (gemmId < 3));
  switch (gemmId) {
  case 0:
    // The 0th kernel will 0-init the output (filter tensor).
    return zeroInit(op, b);
  case 2:
    // The 2nd kernel, if used, will conduct element-wise fp32->fp16 conversion
    // from the workspace to the output (filter tensor).
    assert(hasWorkspace);
    return elementwiseConversion(op, b);
  case 1:
  default:
    break;
  }
  // The 1st kernel will conduct the actual backward weight convolution using
  // atomic adds.

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

    if (filterAttr.getValue() == "k") {
      k = filterShape[i];
    } else if (filterAttr.getValue() == "c") {
      c = filterShape[i];
    } else if (filterAttr.getValue() == "y") {
      y = filterShape[i];
    } else if (filterAttr.getValue() == "x") {
      x = filterShape[i];
    } else if (filterAttr.getValue() == "g") {
      g = filterShape[i];
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

  int64_t gemmKBlocks = 1;
  if (failed(calculateKBlockNum(n, ho, wo, g, k, c, y, x, MPerBlock, NPerBlock,
                                KPerBlock, KPack, numCu, &gemmKBlocks))) {
    op->emitOpError("Invalid tuning parameters to compute KBlock.");
    return failure();
  }

  Value gemmFilter, gemmInput, gemmOutput;
  Value gemmInputKPack, gemmOutputKPack;
  // Transform filter tensor.
  {
    SmallVector<StringRef, 5> nonKDims;
    for (StringRef name : filterNames)
      if (name != "g" && name != "k")
        nonKDims.push_back(name);
    // Add a dimension, that'll be ignored when writing the output, for KBlock
    // The existence of this dimension makes the mapping between the C matrix
    // and the filter tensor uninvertable, hence the need for atomic add

    llvm::StringMap<uint32_t> kBlockDims =
        expandNamesInPlace(filterNames, {{{"k", {"kBlock", "k"}}}});
    BottomUpTMBuilder addKBlockTransform(b, filterNames, filterShape, loc);
    BottomUpTMTopDimsWrapper addKBlockWrap(addKBlockTransform,
                                           std::move(kBlockDims));
    addKBlockWrap.passThrough("g");
    addKBlockWrap.addDim("kBlock", gemmKBlocks);
    addKBlockWrap.passThrough({"k", "c", "y", "x"});

    TransformMapAttr addKBlockTransformAttr = addKBlockTransform.get();
    Value filterTensorInUse = (hasWorkspace) ? op.workspace() : op.filter();
    Value withKBlock = b.create<miopen::TransformOp>(loc, filterTensorInUse,
                                                     addKBlockTransformAttr);

    // Create GEMM filter tensor
    // Here, we merge the KBlock dimension into the G dimension
    // keeping the kBlock dimension as the minor index
    // and send K to the M dimension and CYX to the N dimension as usual
    bool isUnfold = (addKBlockTransform.endIndex("c") + 1 ==
                     addKBlockTransform.endIndex("y")) &&
                    (addKBlockTransform.endIndex("y") + 1 ==
                     addKBlockTransform.endIndex("x"));
    auto gemmTransform =
        BottomUpTMBuilder::above(addKBlockTransform, addKBlockTransformAttr);
    gemmTransform.merge("gemmG", 0, {"g", "kBlock"});
    gemmTransform.passThrough({"gemmM"}, {1}, {"k"});
    gemmTransform.merge("gemmN", 2, nonKDims, isUnfold);

    TransformMapAttr gemmTransformAttr = gemmTransform.get();
    gemmFilter = b.create<TransformOp>(loc, withKBlock, gemmTransformAttr);
    // This kernel is only invoked when there's no need for gemm padding
  }

  // Transform input tensor
  {
    // Pad H and W and split N into  n0 and n1 where n0 has size kBlocks and n1
    // is what's left
    llvm::StringMap<uint32_t> firstTransformOutDims = expandNamesInPlace(
        inputNames,
        {{"ni", {"n0", "n1"}}, {"hi", {"hipad"}}, {"wi", {"wipad"}}});

    BottomUpTMBuilder firstTransform(b, inputNames, inputShape, loc);
    BottomUpTMTopDimsWrapper firstWrap(firstTransform,
                                       std::move(firstTransformOutDims));
    firstWrap.passThrough("gi");
    firstWrap.unmerge({"n0", "n1"}, "ni", {gemmKBlocks, n / gemmKBlocks});
    firstWrap.passThrough("ci");
    firstWrap.pad({"hipad", "wipad"}, {"hi", "wi"},
                  {leftPadH, rightPadH, leftPadW, rightPadW});

    TransformMapAttr firstTransformAttr = firstTransform.get();
    Value firstTransformed =
        b.create<TransformOp>(loc, op.input(), firstTransformAttr);

    // The usual mapping of input space to dimensions such that filter elements
    // get multiplied by the right thing
    llvm::StringMap<uint32_t> embedOutDims = expandNamesInPlace(
        firstTransform, {{"hipad", {"y", "ho"}}, {"wipad", {"x", "wo"}}});
    auto embedTransform =
        BottomUpTMBuilder::above(firstTransform, firstTransformAttr);
    BottomUpTMTopDimsWrapper embedWrap(embedTransform, std::move(embedOutDims));
    embedWrap.passThrough({"gi", "n0", "n1", "ci"});
    embedWrap.embed({"y", "ho"}, {y, ho}, "hipad", {dilationH, strideH});
    embedWrap.embed({"x", "wo"}, {x, wo}, "wipad", {dilationW, strideW});

    TransformMapAttr embedTransformAttr = embedTransform.get();
    Value embedded =
        b.create<TransformOp>(loc, firstTransformed, embedTransformAttr);

    // Merge N1HoWO to gemmK and CYX to gemmN
    auto gemmInputTransform =
        BottomUpTMBuilder::above(embedTransform, embedTransformAttr);

    llvm::SmallVector<StringRef, 3> nonNHWDims = {"ci", "y", "x"};
    std::sort(nonNHWDims.begin(), nonNHWDims.end(),
              [&gemmInputTransform](const StringRef &v1,
                                    const StringRef &v2) -> bool {
                return gemmInputTransform.startIndex(v1) <
                       gemmInputTransform.startIndex(v2);
              });
    // In the gemmG dimension, unlike with gemmN, we don't have the same
    // traversal order concerns - a step in the G dimension always first visits
    // kBlock/N0 and then moves on to the next G
    gemmInputTransform.merge("gemmG", 0, {"gi", "n0"});
    gemmInputTransform.merge("gemmK", 1, {"n1", "ho", "wo"});
    gemmInputTransform.merge("gemmN", 2, nonNHWDims);

    TransformMapAttr gemmInputTransformAttr = gemmInputTransform.get();
    gemmInput = b.create<TransformOp>(loc, embedded, gemmInputTransformAttr);

    // KPack for input tensor.
    gemmInputKPack = createKPackLogic(b, loc, gemmInput, gemmInputTransform,
                                      gemmInputTransformAttr, KPack);
  }

  // Transform output tensor
  {
    // First, split the N dimension as in the input
    llvm::StringMap<uint32_t> outDims =
        expandNamesInPlace(outputNames, {{"no", {"n0", "n1"}}});
    BottomUpTMBuilder firstTransform(b, outputNames, outputShape, loc);
    BottomUpTMTopDimsWrapper firstWrap(firstTransform, std::move(outDims));
    firstWrap.passThrough("go");
    firstWrap.unmerge({"n0", "n1"}, "no", {gemmKBlocks, n / gemmKBlocks});
    firstWrap.passThrough({"ko", "ho", "wo"});

    TransformMapAttr firstTransformAttr = firstTransform.get();
    Value transformed =
        b.create<TransformOp>(loc, op.output(), firstTransformAttr);

    // Map G and N0 to gemmG, N1HW to gemmK and K to gemmM
    auto gemmOutputTransform =
        BottomUpTMBuilder::above(firstTransform, firstTransformAttr);
    gemmOutputTransform.merge("gemmG", 0, {"go", "n0"});
    gemmOutputTransform.merge("gemmK", 1, {"n1", "ho", "wo"});
    gemmOutputTransform.passThrough({"gemmM"}, {2}, {"ko"});

    TransformMapAttr gemmOutputTransformAttr = gemmOutputTransform.get();
    gemmOutput =
        b.create<TransformOp>(loc, transformed, gemmOutputTransformAttr);

    // KPack for output tensor.
    gemmOutputKPack = createKPackLogic(b, loc, gemmOutput, gemmOutputTransform,
                                       gemmOutputTransformAttr, KPack);
  }

  // Set attributes for gridwise_gemm op.
  llvm::SmallVector<NamedAttribute, 8> gridwiseGemmAttrs{
      b.getNamedAttr("gemm_id", gemmIdAttr), b.getNamedAttr("arch", archAttr),
      b.getNamedAttr("num_cu", numCuAttr)};

  // Supply KPack information into gridwiseGemmAttrs.
  if (KPack > 1) {
    gridwiseGemmAttrs.push_back(
        b.getNamedAttr("kpack", b.getI32IntegerAttr(KPack)));
  }

  // xdlopsV2.
  if (isXdlops)
    gridwiseGemmAttrs.push_back(
        b.getNamedAttr("xdlopsV2", b.getBoolAttr(true)));

  // This kernel is not run when there is padding on the GEMM
  auto paddingInfo = PaddingInfoAttr::get(b.getContext(), 0, 0, 0);
  auto storeMethod = StoreMethod::AtomicAdd;

  Value gemmA = gemmOutputKPack;
  Value gemmB = gemmInputKPack;
  Value gemmC = gemmFilter;
  if (isXdlops) {
    auto gop = b.create<GridwiseGemmV2Op>(loc, gemmA, gemmB, gemmC, paddingInfo,
                                          storeMethod, gridwiseGemmAttrs);
    affixGridwiseGemmAttributes(op, gop, b);
  } else {
    op->emitOpError("Backward weight atomic add kernel requires xdlops and "
                    "shouldn't have been tried without them");
  }

  // Finally, erase the original Conv2D op.
  b.eraseOp(op);

  return success();
}

LogicalResult backwardData(Conv2DBwdDataOp op, PatternRewriter &b) {
  auto loc = op.getLoc();
  auto gemmIdAttr = op->template getAttrOfType<IntegerAttr>("gemm_id");
  auto archAttr = op->template getAttrOfType<StringAttr>("arch");
  auto numCuAttr = op->template getAttrOfType<IntegerAttr>("num_cu");

  auto KPackAttr = op->template getAttrOfType<IntegerAttr>("kpack");
  int64_t KPack = KPackAttr.getInt();

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

  int64_t gemmId = gemmIdAttr.getInt();
  // In case the actual gemm ID is -1, emit the zero initialization kernel.
  if (gemmId < 0) {
    return zeroInit(op, b);
  }

  int64_t iYTilda = gemmId / xTilda;
  int64_t iXTilda = gemmId % xTilda;
  int64_t yDotSlice = math_util::integer_divide_ceil(y - iYTilda, yTilda);
  int64_t xDotSlice = math_util::integer_divide_ceil(x - iXTilda, xTilda);

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

  bool hasHPadding = (leftPadH != 0 || rightPadH != 0);
  bool hasWPadding = (leftPadW != 0 || rightPadW != 0);
  bool hasPadding = hasHPadding || hasWPadding;
  bool isStride2Pad1 = ((strideH > 1 || strideW > 1) && hasPadding);

  // Both isOriginalKernelSupport and needExtraPad are used.
  bool needExtraPad = false;
  bool isOriginalKernelSupport = true;

  if (!isXdlops) {
    PopulateParams populateParams;
    std::tie(isOriginalKernelSupport, needExtraPad, gemmMExtra, gemmNExtra,
             gemmKExtra) =
        calculatePaddingKernelSize(gemmMSize, gemmNSize, gemmKSize,
                                   obtainConvDirection(op),
                                   obtainConvDataType(op), populateParams);
  } else { // xdlops
    PopulateParamsXDL populateParamsXDL;
    std::tie(isOriginalKernelSupport, needExtraPad, gemmMExtra, gemmNExtra,
             gemmKExtra) =
        calculatePaddingKernelSize(gemmMSize, gemmNSize, gemmKSize,
                                   obtainConvDirection(op),
                                   obtainConvDataType(op), populateParamsXDL);
  }

  LogicalResult supportedPaddingKernel = isSupportedBackwardDataPaddingKernel(
      isXdlops, isStride2Pad1, gemmMExtra, gemmKExtra, gemmNExtra, op);
  // don't do backward data padding kernel if isSupportPaddingKernel=false
  if (failed(supportedPaddingKernel) && !isOriginalKernelSupport)
    return failure();

  Value gemmFilter, gemmInput, gemmOutput;
  Value gemmFilterKPack, gemmOutputKPack;
  // Transform filter tensor.
  {
    // Embed y/x into {y/x}dot and {y/x}tilda (Why the
    // particular embed coefficients is in a presentation somewhere)
    llvm::StringMap<uint32_t> embedDims = expandNamesInPlace(
        filterNames, {{"y", {"ydot", "ytilda"}}, {"x", {"xdot", "xtilda"}}});
    BottomUpTMBuilder embedTransform(b, filterNames, filterShape, loc);
    BottomUpTMTopDimsWrapper embedWrap(embedTransform, std::move(embedDims));

    embedWrap.passThrough({"g", "k", "c"});
    embedWrap.embed({"ydot", "ytilda"}, {yDot, yTilda}, "y",
                    {strideH / gcdStrideDilationH, 1});
    embedWrap.embed({"xdot", "xtilda"}, {xDot, xTilda}, "x",
                    {strideW / gcdStrideDilationW, 1});

    TransformMapAttr embedTransformAttr = embedTransform.get();
    Value embeddedFilter =
        b.create<TransformOp>(loc, op.filter(), embedTransformAttr);

    // Take slices in the ydot, ytilda, xdot, and xtilda dimensions
    // to reflect which kernel we're performing
    auto sliceTransform =
        BottomUpTMBuilder::above(embedTransform, embedTransformAttr);
    sliceTransform.passThrough({"g", "k", "c"});
    sliceTransform.slice({"ydotslice", "xdotslice"}, {"ydot", "xdot"}, {0, 0},
                         {yDotSlice, xDotSlice});
    sliceTransform.slice({"ytildaslice", "xtildaslice"}, {"ytilda", "xtilda"},
                         {iYTilda, iXTilda}, {iYTilda + 1, iXTilda + 1});

    TransformMapAttr sliceTransformAttr = sliceTransform.get();
    Value slicedFilter =
        b.create<TransformOp>(loc, embeddedFilter, sliceTransformAttr);

    // Set up gemm by passing g -> gemmG, merging
    // [k, ydotslice, xdotslice] to gemmK, and [c, ytildaslice, xtildaslice]
    // to gemmM
    auto gemmFilterTransform =
        BottomUpTMBuilder::above(sliceTransform, sliceTransformAttr);
    gemmFilterTransform.passThrough({"gemmG"}, {0}, {"g"});
    gemmFilterTransform.merge("gemmK", 1, {"k", "ydotslice", "xdotslice"});
    gemmFilterTransform.merge("gemmM", 2, {"c", "ytildaslice", "xtildaslice"});

    TransformMapAttr gemmFilterTransformAttr = gemmFilterTransform.get();
    gemmFilter =
        b.create<TransformOp>(loc, slicedFilter, gemmFilterTransformAttr);

    // Filter padding
    bool filterCheckPadGemmM = (gemmMExtra > 0);
    bool filterCheckPadGemmK = (gemmKExtra > 0);
    if (filterCheckPadGemmM || filterCheckPadGemmK) {
      auto padTransform = BottomUpTMBuilder::above(gemmFilterTransform,
                                                   gemmFilterTransformAttr);
      padTransform.passThrough("gemmG");
      if (filterCheckPadGemmK) {
        padTransform.pad("gemmKPad", "gemmK", 0, gemmKExtra);
      } else {
        padTransform.passThrough("gemmK");
      }

      if (filterCheckPadGemmM) {
        padTransform.pad("gemmMPad", "gemmM", 0, gemmMExtra);
      } else {
        padTransform.passThrough("gemmM");
      }

      TransformMapAttr padTransformAttr = padTransform.get();
      // Replace filter gemm with padded version
      gemmFilter = b.create<TransformOp>(loc, gemmFilter, padTransformAttr);
    }

    // KPack for filter tensor.
    gemmFilterKPack = createKPackLogic(b, loc, gemmFilter, gemmFilterTransform,
                                       gemmFilterTransformAttr, KPack);
  }

  // outside its usual scope so we can look up input tensor dim order
  // for the backwards padding kernel info
  BottomUpTMBuilder padInputTransform(b, inputNames, inputShape, loc);
  // Transform input tensor
  {
    padInputTransform.passThrough({"gi", "ni", "ci"});
    padInputTransform.pad({"hipad", "wipad"},
                          {padInputTransform.startIndex("hi"),
                           padInputTransform.startIndex("wi")},
                          {"hi", "wi"},
                          {leftPadH, rightPadH, leftPadW, rightPadW});

    TransformMapAttr padTransformAttr = padInputTransform.get();
    Value paddedInput =
        b.create<TransformOp>(loc, op.input(), padTransformAttr);

    // Split hipad, wipad into ytilda, htilda, xtilda, wtilda
    llvm::StringMap<uint32_t> embedDims = expandNamesInPlace(
        padInputTransform,
        {{"hipad", {"ytilda", "htilda"}}, {"wipad", {"xtilda", "wtilda"}}});
    auto tildaEmbedTransform =
        BottomUpTMBuilder::above(padInputTransform, padTransformAttr);
    BottomUpTMTopDimsWrapper tildaEmbedWrap(tildaEmbedTransform,
                                            std::move(embedDims));
    tildaEmbedWrap.passThrough({"gi", "ni", "ci"});
    tildaEmbedWrap.embed({"ytilda", "htilda"}, {yTilda, hTilda}, "hipad",
                         {dilationH, strideH});
    tildaEmbedWrap.embed({"xtilda", "wtilda"}, {xTilda, wTilda}, "wipad",
                         {dilationW, strideW});

    TransformMapAttr tildaEmbedTransformAttr = tildaEmbedTransform.get();
    Value tildaEmbedded =
        b.create<TransformOp>(loc, paddedInput, tildaEmbedTransformAttr);

    // Slice all the tilda dimensions: ytilda and xtilda get slices of length
    // 1 while htilda and wtilda have slice indices computed above
    auto sliceTransform =
        BottomUpTMBuilder::above(tildaEmbedTransform, tildaEmbedTransformAttr);
    sliceTransform.passThrough({"gi", "ni", "ci"});
    sliceTransform.slice({"yslice", "xslice"}, {"ytilda", "xtilda"},
                         {iYTilda, iXTilda}, {iYTilda + 1, iXTilda + 1});
    sliceTransform.slice({"hslice", "wslice"}, {"htilda", "wtilda"},
                         {iHTildaLeft, iWTildaLeft},
                         {iHTildaRight, iWTildaRight});

    TransformMapAttr sliceTransformAttr = sliceTransform.get();
    Value sliced =
        b.create<TransformOp>(loc, tildaEmbedded, sliceTransformAttr);

    // C plus the length 1 slices (yslice and xslice) become the gemmM
    // dimension G, N, and the h and w slices become gemmN
    auto gemmTransform =
        BottomUpTMBuilder::above(sliceTransform, sliceTransformAttr);
    gemmTransform.passThrough({"gemmG"}, {0}, {"gi"});
    gemmTransform.merge("gemmM", 1, {"ci", "yslice", "xslice"});
    gemmTransform.merge("gemmN", 2, {"ni", "hslice", "wslice"});

    TransformMapAttr gemmTransformAttr = gemmTransform.get();
    gemmInput = b.create<TransformOp>(loc, sliced, gemmTransformAttr);

    bool inputCheckPadGemmM = (gemmMExtra > 0);
    bool inputCheckPadGemmN = (gemmNExtra > 0);
    if (inputCheckPadGemmM || inputCheckPadGemmN) {
      auto padTransform =
          BottomUpTMBuilder::above(gemmTransform, gemmTransformAttr);
      padTransform.passThrough("gemmG");
      if (inputCheckPadGemmM) {
        padTransform.pad("gemmMPad", "gemmM", 0, gemmMExtra);
      } else {
        padTransform.passThrough("gemmM");
      }

      if (inputCheckPadGemmN) {
        padTransform.pad("gemmNPad", "gemmN", 0, gemmNExtra);
      } else {
        padTransform.passThrough("gemmN");
      }

      TransformMapAttr padTransformAttr = padTransform.get();
      // Replace input gemm with padded version
      gemmInput = b.create<TransformOp>(loc, gemmInput, padTransformAttr);
    }
  }

  // Transform output tensor
  {
    // Embed ho to ydot and htilda and wo to xdot and ytilda
    llvm::StringMap<uint32_t> embedDims = expandNamesInPlace(
        outputNames, {{"ho", {"ydot", "htilda"}}, {"wo", {"xdot", "wtilda"}}});
    BottomUpTMBuilder embedTransform(b, outputNames, outputShape, loc);
    BottomUpTMTopDimsWrapper embedWrap(embedTransform, std::move(embedDims));
    embedWrap.passThrough({"go", "no", "ko"});
    embedWrap.embed({"ydot", "htilda"}, {yDot, hTilda}, "ho",
                    {(-dilationH) / gcdStrideDilationH, 1});
    embedWrap.embed({"xdot", "wtilda"}, {xDot, wTilda}, "wo",
                    {(-dilationW) / gcdStrideDilationW, 1});

    TransformMapAttr embedTransformAttr = embedTransform.get();
    Value embedded =
        b.create<TransformOp>(loc, op.output(), embedTransformAttr);

    // Take the same slices in ydot, xdot, htilda, and wtilda as were taken in
    // the filter and input
    auto sliceTransform =
        BottomUpTMBuilder::above(embedTransform, embedTransformAttr);
    sliceTransform.passThrough({"go", "no", "ko"});
    sliceTransform.slice({"yslice", "xslice"}, {"ydot", "xdot"}, {0, 0},
                         {yDotSlice, xDotSlice});
    sliceTransform.slice({"hslice", "wslice"}, {"htilda", "wtilda"},
                         {iHTildaLeft, iWTildaLeft},
                         {iHTildaRight, iWTildaRight});

    TransformMapAttr sliceTransformAttr = sliceTransform.get();
    Value sliced = b.create<TransformOp>(loc, embedded, sliceTransformAttr);

    // Merge k, yslice, and xslice to gemmK and n, hslice, and wslice to gemmN
    auto gemmOutputTransform =
        BottomUpTMBuilder::above(sliceTransform, sliceTransformAttr);
    gemmOutputTransform.passThrough({"gemmG"}, {0}, {"go"});
    gemmOutputTransform.merge("gemmK", 1, {"ko", "yslice", "xslice"});
    gemmOutputTransform.merge("gemmN", 2, {"no", "hslice", "wslice"});

    TransformMapAttr gemmOutputTransformAttr = gemmOutputTransform.get();
    gemmOutput = b.create<TransformOp>(loc, sliced, gemmOutputTransformAttr);

    bool outputCheckPadGemmK = (gemmKExtra > 0);
    bool outputCheckPadGemmN = (gemmNExtra > 0);
    if (outputCheckPadGemmK || outputCheckPadGemmN) {
      auto padTransform = BottomUpTMBuilder::above(gemmOutputTransform,
                                                   gemmOutputTransformAttr);
      padTransform.passThrough("gemmG");
      if (outputCheckPadGemmK) {
        padTransform.pad("gemmKPad", "gemmK", 0, gemmKExtra);
      } else {
        padTransform.passThrough("gemmK");
      }

      if (outputCheckPadGemmN) {
        padTransform.pad("gemmNPad", "gemmN", 0, gemmNExtra);
      } else {
        padTransform.passThrough("gemmN");
      }

      TransformMapAttr padTransformAttr = padTransform.get();
      // Replace output gemm with padded version
      gemmOutput = b.create<TransformOp>(loc, gemmOutput, padTransformAttr);
    }

    // KPack for output tensor.
    gemmOutputKPack = createKPackLogic(b, loc, gemmOutput, gemmOutputTransform,
                                       gemmOutputTransformAttr, KPack);
  }

  // Set attributes for gridwise_gemm op.
  llvm::SmallVector<NamedAttribute, 8> gridwiseGemmAttrs{
      b.getNamedAttr("gemm_id", gemmIdAttr), b.getNamedAttr("arch", archAttr),
      b.getNamedAttr("num_cu", numCuAttr)};
  // xdlopsV2.
  if (isXdlops)
    gridwiseGemmAttrs.push_back(
        b.getNamedAttr("xdlopsV2", b.getBoolAttr(true)));

  auto paddingInfo =
      PaddingInfoAttr::get(b.getContext(), gemmMExtra, gemmKExtra, gemmNExtra);

  // Supply KPack information into gridwiseGemmAttrs.
  if (KPack > 1) {
    gridwiseGemmAttrs.push_back(
        b.getNamedAttr("kpack", b.getI32IntegerAttr(KPack)));
  }

  Value gemmA = gemmFilterKPack;
  Value gemmB = gemmOutputKPack;
  Value gemmC = gemmInput;
  // Emit miopen.gridwise_gemm op.
  // Emit miopen.gridwise_gemm_v2 if using xdlops
  if (isXdlops) {
    auto gop = b.create<GridwiseGemmV2Op>(loc, gemmA, gemmB, gemmC, paddingInfo,
                                          StoreMethod::Set, gridwiseGemmAttrs);
    affixGridwiseGemmAttributes(op, gop, b);
  } else {
    auto gop = b.create<GridwiseGemmOp>(loc, gemmA, gemmB, gemmC, paddingInfo,
                                        gridwiseGemmAttrs);
    affixGridwiseGemmAttributes(op, gop, b);
  }
  // Finally, erase the original Conv2D op.
  b.eraseOp(op);

  return success();
}

template <typename T> struct Conv2DRewritePattern : public OpRewritePattern<T> {
  const static ArgumentFields fields;
  const static ConvOpType convOpType;
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op, PatternRewriter &b) const override {
    bool isXdlops = false;
    auto xdlopsV2Attr = op->template getAttrOfType<BoolAttr>("xdlopsV2");
    if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true)
      isXdlops = true;
    auto dataType =
        op.input().getType().template cast<MemRefType>().getElementType();
    if (ConvOpType::BwdData == convOpType) {
      return backwardData(cast<Conv2DBwdDataOp>(op), b);
    }
    auto loc = op.getLoc();

    auto archAttr = op->template getAttrOfType<StringAttr>("arch");
    auto numCuAttr = op->template getAttrOfType<IntegerAttr>("num_cu");

    auto KPackAttr = op->template getAttrOfType<IntegerAttr>("kpack");
    int64_t KPack = KPackAttr.getInt();

    auto filterLayoutAttr =
        op->template getAttrOfType<ArrayAttr>("filter_layout");
    auto inputLayoutAttr =
        op->template getAttrOfType<ArrayAttr>("input_layout");
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
      auto inputAttr =
          inputLayoutAttr.getValue()[i].template cast<StringAttr>();
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

    int64_t gemmMSize, gemmNSize, gemmKSize;
    int64_t gemmMExtra, gemmNExtra, gemmKExtra;
    gemmMSize = gemmNSize = gemmKSize = 0;
    gemmMExtra = gemmNExtra = gemmKExtra = 0;
    // compute we should use extra padding kernel or not
    // c,k already / g ,so we can skip / g here
    switch (convOpType) {
    case ConvOpType::Fwd:
      gemmMSize = k;
      gemmKSize = c * y * x;
      gemmNSize = n * ho * wo;
      break;
    case ConvOpType::BwdData:
      gemmMSize = c;
      gemmKSize = k * y * x;
      gemmNSize = n * ho * wo;
      break;
    case ConvOpType::BwdWeight:
      gemmMSize = k;
      gemmKSize = n * ho * wo;
      gemmNSize = c * y * x;
      break;
    }

    // isOriginalKernelSupport is not used.
    // Only needExtraPad is used.
    bool isOriginalKernelSupport = true;
    bool needExtraPad = false;

    if (!isXdlops) {
      PopulateParams populateParams;
      std::tie(isOriginalKernelSupport, needExtraPad, gemmMExtra, gemmNExtra,
               gemmKExtra) =
          calculatePaddingKernelSize(gemmMSize, gemmNSize, gemmKSize,
                                     convOpType, dataType, populateParams);
    } else { // xdlops
      PopulateParamsXDL populateParamsXDL;
      std::tie(isOriginalKernelSupport, needExtraPad, gemmMExtra, gemmNExtra,
               gemmKExtra) =
          calculatePaddingKernelSize(gemmMSize, gemmNSize, gemmKSize,
                                     convOpType, dataType, populateParamsXDL);
    }

    if (ConvOpType::BwdWeight == convOpType && isXdlops &&
        (dataType == b.getF32Type() || dataType == b.getF16Type()) &&
        needExtraPad == false) {
      // current backward weight with atomic_add can only run under xdlops +
      // fp32 / fp16.
      return backwardWeightAtomicAdd(cast<Conv2DBwdWeightOp>(op), b);
    }

    // Transform filter tensor.

    // set layout attribute.
    // Weight tensor transformation for Conv2DOp
    // - PassThrough G dimension to dimension 0, name it gemmG.
    // - Merge non-K dimensions to dimension 1, name it as gemmK.
    //   Optimization: If non-K dimensions are consequetive, apply unfold.
    // - PassThrough K dimension to dimension 2, name it as gemmM.
    //
    // Weight tensor transformation for Conv2DBwdWeightOp
    // - PassThrough G dimension to dimension 0, name it gemmG
    // - PassThrough K dimension to dimension 1, name it as gemmM.
    // - Merge non-K dimensions to dimension 2, name it as gemmN.
    SmallVector<StringRef, 5> filterNonKDims;
    for (StringRef name : filterNames)
      if (name != "g" && name != "k")
        filterNonKDims.push_back(name);

    bool noNonKPad = (convOpType == ConvOpType::BwdWeight && gemmNExtra == 0) ||
                     (convOpType == ConvOpType::Fwd && gemmKExtra == 0);

    BottomUpTMBuilder filterTransform(b, filterNames, filterShape, loc);
    filterTransform.passThrough({"gemmG"}, {0}, {"g"});
    bool isUnfold = filterTransform.startIndex("g") == 0 &&
                    (filterTransform.startIndex("k") == 1 ||
                     filterTransform.startIndex("k") == 4) &&
                    noNonKPad;
    switch (convOpType) {
    case ConvOpType::Fwd:
      filterTransform.merge("gemmK", 1, filterNonKDims, isUnfold);
      filterTransform.passThrough({"gemmM"}, {2}, {"k"});
      break;
    case ConvOpType::BwdWeight:
      filterTransform.passThrough({"gemmM"}, {1}, {"k"});
      filterTransform.merge("gemmN", 2, filterNonKDims, isUnfold);
      break;
    case ConvOpType::BwdData:
      llvm_unreachable("Backward data has been sent elsewhere");
      break;
    }

    TransformMapAttr filterTransformAttr = filterTransform.get();
    Value gemmFilter =
        b.create<TransformOp>(loc, op.filter(), filterTransformAttr);

    BottomUpTMBuilder padGemmFilterTransform = filterTransform;
    TransformMapAttr padGemmFilterTransformAttr = filterTransformAttr;
    Value gemmFilterPad = gemmFilter;

    // filter pad start
    // K:output channel, C:input channel,Y:filter height,X:filter width
    // filter dim : K & merge(C,Y,X) , if C*Y*X is under 64 or 32
    // we pad CYX to 32 or 64, then mlir can do gemm
    // we add more one transform to do pad
    bool filterCheckPadGemmM = false;
    bool filterCheckPadGemmK = false;
    bool filterCheckPadGemmN = false;
    filterCheckPadGemmM =
        (convOpType == ConvOpType::Fwd && gemmMExtra > 0) ||
        (convOpType == ConvOpType::BwdWeight && gemmMExtra > 0);
    filterCheckPadGemmK = (convOpType == ConvOpType::Fwd && gemmKExtra > 0);
    filterCheckPadGemmN =
        (convOpType == ConvOpType::BwdWeight && gemmNExtra > 0);
    bool isFilterPad = false;
    if (filterCheckPadGemmM || filterCheckPadGemmK || filterCheckPadGemmN) {
      isFilterPad = true;
      padGemmFilterTransform =
          BottomUpTMBuilder::above(filterTransform, filterTransformAttr);
      padGemmFilterTransform.passThrough("gemmG");

      // Note that, when padding a gemm dimension that came from the non-K
      // tensor dimensions, only the leading dimension is added to the oob check
      // set, as adding all the dimensions historically led to miscompilation
      if (filterCheckPadGemmK) {
        padGemmFilterTransform.pad("gemmKPad", "gemmK", 0, gemmKExtra);
      } else if (convOpType != ConvOpType::BwdWeight) {
        // Backward weight has no GemmK on its filter
        padGemmFilterTransform.passThrough("gemmK");
      }

      if (filterCheckPadGemmM) {
        padGemmFilterTransform.pad("gemmMPad", "gemmM", 0, gemmMExtra);
      } else {
        padGemmFilterTransform.passThrough("gemmM");
      }

      if (filterCheckPadGemmN) {
        padGemmFilterTransform.pad("gemmNPad", "gemmN", 0, gemmNExtra);
      } else if (convOpType != ConvOpType::Fwd) {
        padGemmFilterTransform.passThrough("gemmN");
      }

      padGemmFilterTransformAttr = padGemmFilterTransform.get();
      gemmFilterPad =
          b.create<TransformOp>(loc, gemmFilter, padGemmFilterTransformAttr);
      // filter pad end
    }

    // KPack for filter tensor.
    Value gemmFilterKPack = gemmFilterPad;
    if ((KPack > 1) && (convOpType == ConvOpType::Fwd)) {
      BottomUpTMBuilder sourceTransform =
          (isFilterPad) ? padGemmFilterTransform : filterTransform;
      TransformMapAttr sourceTransformAttr =
          (isFilterPad) ? padGemmFilterTransformAttr : filterTransformAttr;
      Value source = (isFilterPad) ? gemmFilterPad : gemmFilter;
      gemmFilterKPack = createKPackLogic(b, loc, source, sourceTransform,
                                         sourceTransformAttr, KPack);
    }

    // Transform input tensor.
    // Input tensor step 1: padded input.

    // set layout attribute.
    // Padded input tensor transformation:
    // - Pass through ni, gi, and ci, not renaming them
    // - Padd hi and wi as specified in padding attributes, renaming them to
    // hipad and wipad
    BottomUpTMBuilder padInputTransform(b, inputNames, inputShape, loc);
    padInputTransform.passThrough("ni");
    padInputTransform.passThrough("gi");
    padInputTransform.passThrough("ci");

    llvm::SmallVector<int64_t, 4> padArgs = {leftPadH, rightPadH, leftPadW,
                                             rightPadW};
    llvm::SmallVector<uint32_t, 2> padOutDims = {
        padInputTransform.startIndex("hi"), padInputTransform.startIndex("wi")};
    padInputTransform.pad({"hipad", "wipad"}, padOutDims, {"hi", "wi"},
                          padArgs);

    TransformMapAttr padInputTransformAttr = padInputTransform.get();

    Value paddedInput =
        b.create<TransformOp>(loc, op.input(), padInputTransformAttr);

    // Input tensor step 2 : embedded input.
    // Embedded input tensor transformation:
    // - PassThrough gi, ni, and ci
    // - Embed hipad to y and ho with size filter y by output h and
    //   coefficients dilationH and strideH
    // - Embed wipad to x and wo with size filter x by output h and
    //   coefficients dilationW and strideW

    llvm::StringMap<uint32_t> embeddedInputDims = expandNamesInPlace(
        padInputTransform, {{"hipad", {"y", "ho"}}, {"wipad", {"x", "wo"}}});
    BottomUpTMBuilder embedInputTransform =
        BottomUpTMBuilder::above(padInputTransform, padInputTransformAttr);
    BottomUpTMTopDimsWrapper embedInputWrap(embedInputTransform,
                                            std::move(embeddedInputDims));
    embedInputWrap.passThrough({"ni", "gi", "ci"});
    embedInputWrap.embed({"y", "ho"}, {y, ho}, "hipad", {dilationH, strideH});
    embedInputWrap.embed({"x", "wo"}, {x, wo}, "wipad", {dilationW, strideW});

    TransformMapAttr embedInputTransformAttr = embedInputTransform.get();
    Value embeddedInput =
        b.create<TransformOp>(loc, paddedInput, embedInputTransformAttr);

    // Input tensor step 3: GEMM'd input
    //
    // - PassThrough gi to dimension 0 and name it gemmG, then
    // For Conv2DOp:
    // - Merge ci, y, x dimensions to dimension 1, name it as gemmK.
    // - Merge ni, ho, wo dimensions to dimension 2, name it as gemmN.
    //
    // For Conv2DBwdWeightOp:
    // - Part 1: Merge ni, ho, wo dimensions to dimension 1, name it as gemmK.
    // - Part 2: Merge ci, y, x dimensions to dimension 2, name it as gemmN.

    auto gemmInputTransform =
        BottomUpTMBuilder::above(embedInputTransform, embedInputTransformAttr);
    gemmInputTransform.passThrough({"gemmG"}, {0}, {"gi"});

    llvm::SmallVector<StringRef, 3> nonNHWDims = {"ci", "y", "x"};
    std::sort(nonNHWDims.begin(), nonNHWDims.end(),
              [&gemmInputTransform](const StringRef &v1,
                                    const StringRef &v2) -> bool {
                return gemmInputTransform.startIndex(v1) <
                       gemmInputTransform.startIndex(v2);
              });

    llvm::SmallVector<StringRef, 3> mergeToK, mergeToN;
    switch (convOpType) {
    case ConvOpType::Fwd:
      mergeToK = std::move(nonNHWDims);
      mergeToN = {"ni", "ho", "wo"};
      break;
    case ConvOpType::BwdWeight:
      mergeToK = {"ni", "ho", "wo"};
      mergeToN = std::move(nonNHWDims);
      break;
    case ConvOpType::BwdData:
      llvm_unreachable("Backward data is in another function");
    }
    gemmInputTransform.merge("gemmK", 1, mergeToK);
    gemmInputTransform.merge("gemmN", 2, mergeToN);

    TransformMapAttr gemmInputTransformAttr = gemmInputTransform.get();
    Value gemmInput =
        b.create<TransformOp>(loc, embeddedInput, gemmInputTransformAttr);

    BottomUpTMBuilder padGemmInputTransform = gemmInputTransform;
    TransformMapAttr padGemmInputTransformAttr = gemmInputTransformAttr;
    Value gemmInputPad = gemmInput;

    // input padding start
    // input : NHW & CRS , if CRS is under 64 or 32
    // we pad CRS to 32 or 64, then mlir can do gemm
    // we add more one transform to do pad

    // input forward : gemmK,gemmN
    // backward weights: gemmK,gemmN
    // so we don't need to pad gemmK
    bool inputCheckPadGemmK = false;
    bool inputCheckPadGemmN = false;
    inputCheckPadGemmK =
        (convOpType == ConvOpType::Fwd && gemmKExtra > 0) ||
        (convOpType == ConvOpType::BwdWeight && gemmKExtra > 0);
    inputCheckPadGemmN =
        (convOpType == ConvOpType::Fwd && gemmNExtra > 0) ||
        (convOpType == ConvOpType::BwdWeight && gemmNExtra > 0);
    bool isInputPad = false;
    if (inputCheckPadGemmK || inputCheckPadGemmN) {
      isInputPad = true;
      padGemmInputTransform =
          BottomUpTMBuilder::above(gemmInputTransform, gemmInputTransformAttr);
      padGemmInputTransform.passThrough("gemmG");
      if (inputCheckPadGemmK) {
        padGemmInputTransform.pad("gemmKPad", "gemmK", 0, gemmKExtra);
      } else {
        padGemmInputTransform.passThrough("gemmK");
      }

      if (inputCheckPadGemmN) {
        padGemmInputTransform.pad("gemmNPad", "gemmN", 0, gemmNExtra);
      } else {
        padGemmInputTransform.passThrough("gemmN");
      }

      padGemmInputTransformAttr = padGemmInputTransform.get();
      gemmInputPad =
          b.create<TransformOp>(loc, gemmInput, padGemmInputTransformAttr);
      // input padding end
    }

    // KPack for input tensor.
    Value gemmInputKPack = gemmInputPad;
    if ((KPack > 1) && ((convOpType == ConvOpType::Fwd) ||
                        (convOpType == ConvOpType::BwdWeight))) {
      BottomUpTMBuilder sourceTransform =
          (isInputPad) ? padGemmInputTransform : gemmInputTransform;
      TransformMapAttr sourceTransformAttr =
          (isInputPad) ? padGemmInputTransformAttr : gemmInputTransformAttr;
      Value source = (isInputPad) ? gemmInputPad : gemmInput;
      gemmInputKPack = createKPackLogic(b, loc, source, sourceTransform,
                                        sourceTransformAttr, KPack);
    }

    // Transform output tensor.
    // - PassThrough G to dimmension 0, name it gemmG, then
    // Output tensor transformation for Conv2DOp:
    // - PassThrough K dimension to dimension 1, named gemmM
    // - Merge non-K dimensions to dimension2, named gemmN

    // Output tensor transformation for backward weight:
    // - Merge non-K dimensions to dimension 1, named gemmK
    // - PassThrough K dimension to dimension 2, name it gemmM
    SmallVector<StringRef, 5> outputNonKDims;
    for (StringRef name : outputNames)
      if (name != "go" && name != "ko")
        outputNonKDims.push_back(name);

    BottomUpTMBuilder outputTransform(b, outputNames, outputShape, loc);
    outputTransform.passThrough({"gemmG"}, {0}, {"go"});
    switch (convOpType) {
    case ConvOpType::Fwd:
      outputTransform.passThrough({"gemmM"}, {1}, {"ko"});
      outputTransform.merge("gemmN", 2, outputNonKDims);
      break;
    case ConvOpType::BwdWeight:
      outputTransform.merge("gemmK", 1, outputNonKDims);
      outputTransform.passThrough({"gemmM"}, {2}, {"ko"});
      break;
    case ConvOpType::BwdData:
      llvm_unreachable("Backward data has been sent elsewhere");
      break;
    }

    TransformMapAttr outputTransformAttr = outputTransform.get();
    Value gemmOutput =
        b.create<TransformOp>(loc, op.output(), outputTransformAttr);

    BottomUpTMBuilder padGemmOutputTransform = outputTransform;
    TransformMapAttr padGemmOutputTransformAttr = outputTransformAttr;
    Value gemmOutputPad = gemmOutput;

    // output padding start
    // output matrix dim: K & NHW
    // when backward weight , GEMMK = NHW
    // N:batch size, H:output height ,W:output width
    // If size of N*h*w is under 32 or 64 ,we pad it to 32 or 64
    // then mlir can do gemm
    // we just add more one transform to do it

    bool outputCheckPadGemmK = false;
    bool outputCheckPadGemmM = false;
    bool outputCheckPadGemmN = false;
    outputCheckPadGemmK =
        (convOpType == ConvOpType::BwdWeight && gemmKExtra > 0);
    outputCheckPadGemmM =
        (convOpType == ConvOpType::BwdWeight && gemmMExtra > 0) ||
        (convOpType == ConvOpType::Fwd && gemmMExtra > 0);
    outputCheckPadGemmN = (convOpType == ConvOpType::Fwd && gemmNExtra > 0);
    bool isOutputPad = false;
    if (outputCheckPadGemmK || outputCheckPadGemmM || outputCheckPadGemmN) {
      isOutputPad = true;
      padGemmOutputTransform =
          BottomUpTMBuilder::above(outputTransform, outputTransformAttr);
      padGemmOutputTransform.passThrough("gemmG");

      if (outputCheckPadGemmK) {
        padGemmOutputTransform.pad("gemmKPad", "gemmK", 0, gemmKExtra);
      } else if (convOpType != ConvOpType::Fwd) {
        padGemmOutputTransform.passThrough("gemmK");
      }

      if (outputCheckPadGemmM) {
        padGemmOutputTransform.pad("gemmMPad", "gemmM", 0, gemmMExtra);
      } else {
        padGemmOutputTransform.passThrough("gemmM");
      }

      if (outputCheckPadGemmN) {
        padGemmOutputTransform.pad("gemmNPad", "gemmN", 0, gemmNExtra);
      } else if (convOpType != ConvOpType::BwdWeight) {
        padGemmOutputTransform.passThrough("gemmN");
      }

      padGemmOutputTransformAttr = padGemmOutputTransform.get();
      gemmOutputPad =
          b.create<TransformOp>(loc, gemmOutput, padGemmOutputTransformAttr);
      // output padding end
    }

    // KPack for output tensor.
    Value gemmOutputKPack = gemmOutputPad;
    if ((KPack > 1) && (convOpType == ConvOpType::BwdWeight)) {
      BottomUpTMBuilder sourceTransform =
          (isOutputPad) ? padGemmOutputTransform : outputTransform;
      TransformMapAttr sourceTransformAttr =
          (isOutputPad) ? padGemmOutputTransformAttr : outputTransformAttr;
      Value source = (isOutputPad) ? gemmOutputPad : gemmOutput;
      gemmOutputKPack = createKPackLogic(b, loc, source, sourceTransform,
                                         sourceTransformAttr, KPack);
    }

    // Set attributes for gridwise_gemm op.
    llvm::SmallVector<NamedAttribute, 8> gridwiseGemmAttrs{
        b.getNamedAttr("arch", archAttr),
        b.getNamedAttr("num_cu", numCuAttr),
    };

    // xdlopsV2.
    if (isXdlops)
      gridwiseGemmAttrs.push_back(
          b.getNamedAttr("xdlopsV2", b.getBoolAttr(true)));

    SmallVector<Value, 3> arguments = {gemmFilterKPack, gemmInputKPack,
                                       gemmOutputKPack};

    Value gemmA, gemmB, gemmC;
    gemmA = arguments[fields.gridwiseGemmArgumentPosition[0]];
    gemmB = arguments[fields.gridwiseGemmArgumentPosition[1]];
    gemmC = arguments[fields.gridwiseGemmArgumentPosition[2]];

    // Create padding info attr
    auto paddingInfo = PaddingInfoAttr::get(b.getContext(), gemmMExtra,
                                            gemmKExtra, gemmNExtra);

    auto storeMethod = StoreMethod::Set;
    // Emit miopen.gridwise_gemm op.
    // Emit miopen.gridwise_gemm_v2 if xdlopsV2 attribute is true.

    // Supply KPack information into gridwiseGemmAttrs.
    if ((KPack > 1) && ((convOpType == ConvOpType::Fwd) ||
                        (convOpType == ConvOpType::BwdWeight))) {
      gridwiseGemmAttrs.push_back(
          b.getNamedAttr("kpack", b.getI32IntegerAttr(KPack)));
    } else {
      gridwiseGemmAttrs.push_back(
          b.getNamedAttr("kpack", b.getI32IntegerAttr(1)));
    }

    if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true) {
      auto gop =
          b.create<GridwiseGemmV2Op>(loc, gemmA, gemmB, gemmC, paddingInfo,
                                     storeMethod, gridwiseGemmAttrs);
      affixGridwiseGemmAttributes(op, gop, b);
    } else {
      auto gop = b.create<GridwiseGemmOp>(loc, gemmA, gemmB, gemmC, paddingInfo,
                                          gridwiseGemmAttrs);
      affixGridwiseGemmAttributes(op, gop, b);
    }

    // Finally, erase the original Conv2D op.
    b.eraseOp(op);

    return success();
  }
};

template <>
const ArgumentFields Conv2DRewritePattern<Conv2DOp>::fields = {
    {0, 1, 2},
    {"KM", "KN", "MN"},
};
template <>
const ConvOpType Conv2DRewritePattern<Conv2DOp>::convOpType = ConvOpType::Fwd;

template <>
const ArgumentFields Conv2DRewritePattern<Conv2DBwdDataOp>::fields = {
    {0, 2, 1},
    {"KM", "MN", "KN"},
};

template <>
const ConvOpType Conv2DRewritePattern<Conv2DBwdDataOp>::convOpType =
    ConvOpType::BwdData;

template <>
const ArgumentFields Conv2DRewritePattern<Conv2DBwdWeightOp>::fields = {
    {2, 1, 0},
    {"MN", "KN", "KM"},
};

template <>
const ConvOpType Conv2DRewritePattern<Conv2DBwdWeightOp>::convOpType =
    ConvOpType::BwdWeight;

// Explicitly instantiate the template to operation type
template struct Conv2DRewritePattern<Conv2DOp>;
template struct Conv2DRewritePattern<Conv2DBwdDataOp>;
template struct Conv2DRewritePattern<Conv2DBwdWeightOp>;

void LowerMIOpenOpsStep1Pass::runOnOperation() {
  MLIRContext *ctx = &getContext();

  ConversionTarget target(*ctx);
  target.addIllegalOp<miopen::Conv2DOp, miopen::Conv2DBwdDataOp,
                      miopen::Conv2DBwdWeightOp>();
  target.addLegalOp<miopen::TransformOp, miopen::GridwiseGemmOp,
                    miopen::GridwiseGemmV2Op, miopen::WorkgroupIdOp,
                    miopen::WorkitemIdOp>();
  // Below are required legalize for the lowering of Conv2DBwdWeightOp
  target.addLegalDialect<arith::ArithmeticDialect, memref::MemRefDialect,
                         AffineDialect, scf::SCFDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<Conv2DRewritePattern<Conv2DOp>,
               Conv2DRewritePattern<Conv2DBwdDataOp>,
               Conv2DRewritePattern<Conv2DBwdWeightOp>>(ctx);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
} // end anonymous namespace

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep1Pass() {
  return std::make_unique<LowerMIOpenOpsStep1Pass>();
}
