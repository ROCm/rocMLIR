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

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/TransformMapBuilder.h"
#include "mlir/Dialect/MIOpen/Tuning/ConvContext.h"
#include "mlir/Dialect/MIOpen/Tuning/GemmContext.h"
#include "mlir/Dialect/MIOpen/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/MIOpen/Tuning/UtilityParams.h"
#include "mlir/Dialect/MIOpen/utility/builderUtils.h"
#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"
#include "mlir/Dialect/MIOpen/utility/math.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "miopen-conv-to-gemm"

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

struct MIOpenConvToGemmPass
    : public MIOpenConvToGemmPassBase<MIOpenConvToGemmPass> {
  void runOnOperation() override;
};

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
    if (llvm::find(actual, name) == actual.end()) {
      return op.emitOpError("Layout mismatch in ")
             << argName << " tensor: Expected it to have a `" << name
             << "` dimension";
    }
  }
  return success();
}

/// Get the dimension names for the given `op` into `filterNames`, `inputNames`
/// and `outputNames`, returning failure if `op`'s layout doesn't contain all of
/// the expected dimension names.
template <typename T>
LogicalResult getConvDimNames(T op, SmallVectorImpl<StringRef> &filterNames,
                              SmallVectorImpl<StringRef> &inputNames,
                              SmallVectorImpl<StringRef> &outputNames) {
  auto filterLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->template getAttrOfType<ArrayAttr>("input_layout");
  auto outputLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("output_layout");

  unsigned size = filterLayoutAttr.size();
  if (size != inputLayoutAttr.size() || size != outputLayoutAttr.size())
    return op.emitOpError("All convolution layouts must have the same length");

  filterNames.reserve(size);
  inputNames.reserve(size);
  outputNames.reserve(size);

  for (unsigned i = 0; i < size; ++i) {
    auto filterAttr =
        filterLayoutAttr.getValue()[i].template cast<StringAttr>();
    auto inputAttr = inputLayoutAttr.getValue()[i].template cast<StringAttr>();
    auto outputAttr =
        outputLayoutAttr.getValue()[i].template cast<StringAttr>();

    filterNames.push_back(filterAttr.getValue());
    inputNames.push_back(inputAttr.getValue());
    outputNames.push_back(outputAttr.getValue());
  }
  if (failed(
          checkNames(filterNames, {"k", "g", "c", "y", "x"}, "filter", op)) ||
      failed(checkNames(inputNames, {"ni", "gi", "ci", "hi", "wi"}, "input",
                        op)) ||
      failed(checkNames(outputNames, {"no", "go", "ko", "ho", "wo"}, "output",
                        op))) {
    return failure();
  }
  return success();
}

void affixGridwiseGemmAttributes(Operation *convOp, Operation *gop,
                                 OpBuilder &b) {
  gop->setAttr("block_size", convOp->getAttr("block_size"));
  gop->setAttr("m_per_block", convOp->getAttr("m_per_block"));
  gop->setAttr("n_per_block", convOp->getAttr("n_per_block"));
  gop->setAttr("k_per_block", convOp->getAttr("k_per_block"));
  gop->setAttr("matrix_a_source_data_per_read",
               convOp->getAttr("matrix_a_source_data_per_read"));
  gop->setAttr("matrix_a_source_vector_read_dim",
               convOp->getAttr("matrix_a_source_vector_read_dim"));
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

/// Create an elementwise utility kernel.
/// The callback has type (builder, location, collapsedBuffers, coordinate).
/// Note: you are expected to handle out of bounds, such as by using
/// miopen.buffer_store
LogicalResult createElementwiseLoop(
    OpBuilder &b, Location loc, Operation *convOp, ValueRange memrefs,
    int64_t vectorLen,
    function_ref<void(OpBuilder &, Location, ValueRange, Value)> emitBodyFunc) {
  int64_t blockSize = convOp->getAttrOfType<IntegerAttr>("block_size").getInt();
  int64_t elemsPerThread =
      convOp->getAttrOfType<IntegerAttr>("elems_per_thread").getInt();
  if (elemsPerThread % vectorLen != 0)
    return convOp->emitOpError("Unevenly vectorized elementwise kernel");

  Value workgroupId = b.create<WorkgroupIdOp>(loc, b.getIndexType());
  Value workgroupDim = b.create<ConstantIndexOp>(loc, blockSize);
  Value elemsPerThreadOp = b.create<ConstantIndexOp>(loc, elemsPerThread);
  Value workitemId = b.create<WorkitemIdOp>(loc, b.getIndexType());

  SmallVector<Value, 2> collapsedBufs;
  for (Value memref : memrefs) {
    if (auto transform =
            dyn_cast_or_null<TransformOp>(memref.getDefiningOp())) {
      return convOp->emitOpError(
          "Arguments to utility kernels must be pure memrefs");
    }
    Value collapsed = createCollapseShapeOp(b, loc, memref);
    collapsedBufs.push_back(collapsed);
  }
  int64_t collapsedLen =
      collapsedBufs[0].getType().cast<MemRefType>().getShape()[0];
  for (Value c : collapsedBufs)
    if (c.getType().cast<MemRefType>().getNumElements() != collapsedLen)
      return convOp->emitOpError(
          "Utility kernel arguments have different lengths");

  Value offset = b.create<MulIOp>(
      loc,
      b.create<AddIOp>(loc, b.create<MulIOp>(loc, workgroupId, workgroupDim),
                       workitemId),
      elemsPerThreadOp);

  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value vectorLenOp = b.create<arith::ConstantIndexOp>(loc, vectorLen);
  auto loop = b.create<scf::ForOp>(loc, zero, elemsPerThreadOp, vectorLenOp);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    Value index = b.create<arith::AddIOp>(loc, offset, loop.getInductionVar());
    emitBodyFunc(b, loc, collapsedBufs, index);
  }
  return success();
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
  Type outputType = output.getType().cast<MemRefType>().getElementType();
  constexpr int64_t kZeroInitVecLen = 4;
  Type storeType = VectorType::get(kZeroInitVecLen, outputType);
  Value zeroOp = createZeroConstantOp(b, loc, storeType);
  ArrayAttr leftOob = b.getI32ArrayAttr({});
  ArrayAttr rightOob = b.getI32ArrayAttr({0});

  auto loopBody = [&zeroOp, &leftOob, &rightOob](OpBuilder &b, Location loc,
                                                 ValueRange collapsed,
                                                 Value index) {
    b.create<BufferStoreOp>(
        loc, zeroOp, collapsed[0], leftOob, rightOob, index,
        StoreMethodAttr::get(b.getContext(), StoreMethod::Set),
        /*offset=*/IntegerAttr());
  };
  LogicalResult res =
      createElementwiseLoop(b, loc, op, output, kZeroInitVecLen, loopBody);
  if (failed(res))
    return failure();

  b.eraseOp(op);
  return success();
}

/// 0-initialize the output for a backward weight convolution which uses
/// atomic adds.
/// For f32 type, the output is the filter tensor.
/// for f16 type, the output is the workspace.
LogicalResult zeroInit(Conv2DBwdWeightOp op, PatternRewriter &b) {
  Location loc = op.getLoc();
  Type filterDataType =
      op.filter().getType().cast<MemRefType>().getElementType();
  Value output;
  if (filterDataType == b.getF32Type()) {
    output = op.filter();
  } else if (filterDataType == b.getF16Type()) {
    if (!op.workspace())
      return op.emitOpError("op has no workspace");
    output = op.workspace();
  } else {
    return op.emitOpError("Unsupported zeroing data type");
  }

  Type outputType = output.getType().cast<MemRefType>().getElementType();
  constexpr int64_t kZeroInitVecLen = 4;
  Type storeType = VectorType::get(kZeroInitVecLen, outputType);
  Value zeroOp = createZeroConstantOp(b, loc, storeType);
  ArrayAttr leftOob = b.getI32ArrayAttr({});
  ArrayAttr rightOob = b.getI32ArrayAttr({0});

  auto loopBody = [&zeroOp, &leftOob, &rightOob](OpBuilder &b, Location loc,
                                                 ValueRange collapsed,
                                                 Value index) {
    b.create<BufferStoreOp>(
        loc, zeroOp, collapsed[0], leftOob, rightOob, index,
        StoreMethodAttr::get(b.getContext(), StoreMethod::Set),
        /*offset=*/IntegerAttr());
  };

  LogicalResult res =
      createElementwiseLoop(b, loc, op, output, kZeroInitVecLen, loopBody);
  if (failed(res))
    return failure();

  b.eraseOp(op);
  return success();
}

/// Element-wise conversion from the workspace to the output (filter tensor)
/// for a backward weight convolution which uses atomic adds.
LogicalResult elementwiseConversion(Conv2DBwdWeightOp op, PatternRewriter &b) {
  Location loc = op.getLoc();
  if (!op.workspace())
    return op.emitOpError("op has no workspace");
  Value filter = op.filter();
  Value workspace = op.workspace();
  Type filterDataType = filter.getType().cast<MemRefType>().getElementType();
  Type workspaceDataType =
      workspace.getType().cast<MemRefType>().getElementType();

  int64_t kConversionVectorLen = 4;
  Type loadType = VectorType::get(kConversionVectorLen, workspaceDataType);
  Type storeType = VectorType::get(kConversionVectorLen, filterDataType);
  ArrayAttr leftOob = b.getI32ArrayAttr({});
  ArrayAttr rightOob = b.getI32ArrayAttr({0});

  auto loopBody = [&loadType, &storeType, &leftOob,
                   &rightOob](OpBuilder &b, Location loc, ValueRange collapsed,
                              Value index) {
    Value loaded =
        b.create<BufferLoadOp>(loc, loadType, collapsed[0], leftOob, rightOob,
                               index, /*offset=*/IntegerAttr());
    Value converted = createTypeConversionOp(b, loc, loaded, storeType);
    b.create<BufferStoreOp>(
        loc, converted, collapsed[1], leftOob, rightOob, index,
        StoreMethodAttr::get(b.getContext(), StoreMethod::Set),
        /*offset=*/IntegerAttr());
  };
  LogicalResult res = createElementwiseLoop(b, loc, op, {workspace, filter},
                                            kConversionVectorLen, loopBody);
  if (failed(res))
    return failure();

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

  auto KPackAttr = op->template getAttrOfType<IntegerAttr>("kpack");
  int64_t KPack = KPackAttr.getInt();

  auto KBlocksAttr = op->template getAttrOfType<IntegerAttr>("kblocks");
  int64_t gemmKBlocks = KBlocksAttr.getInt();

  ConvolutionContext ctx = populateConvContext(op);

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
  int64_t leftPadH = ctx.getPaddingVal()[0];
  int64_t leftPadW = ctx.getPaddingVal()[2];
  int64_t rightPadH = ctx.getPaddingVal()[1];
  int64_t rightPadW = ctx.getPaddingVal()[3];

  int64_t dilationH = ctx.getDilationVal()[0];
  int64_t dilationW = ctx.getDilationVal()[1];
  int64_t strideH = ctx.getStrideVal()[0];
  int64_t strideW = ctx.getStrideVal()[1];
  ConvolutionDims convDims = ctx.getConvDims();

  llvm::SmallVector<StringRef, 5> filterNames, inputNames, outputNames;
  if (failed(getConvDimNames(op, filterNames, inputNames, outputNames))) {
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
    firstWrap.unmerge({"n0", "n1"}, "ni",
                      {gemmKBlocks, convDims.n / gemmKBlocks});
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
    embedWrap.embed({"y", "ho"}, {convDims.y, convDims.ho}, "hipad",
                    {dilationH, strideH});
    embedWrap.embed({"x", "wo"}, {convDims.x, convDims.wo}, "wipad",
                    {dilationW, strideW});

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
    firstWrap.unmerge({"n0", "n1"}, "no",
                      {gemmKBlocks, convDims.n / gemmKBlocks});
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

  ConvolutionContext ctx = populateConvContext(op);

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
  int64_t leftPadH = ctx.getPaddingVal()[0];
  int64_t leftPadW = ctx.getPaddingVal()[2];
  int64_t rightPadH = ctx.getPaddingVal()[1];
  int64_t rightPadW = ctx.getPaddingVal()[3];

  int64_t dilationH = ctx.getDilationVal()[0];
  int64_t dilationW = ctx.getDilationVal()[1];
  int64_t strideH = ctx.getStrideVal()[0];
  int64_t strideW = ctx.getStrideVal()[1];
  ConvolutionDims convDims = ctx.getConvDims();
  SmallVector<StringRef, 5> filterNames, inputNames, outputNames;
  if (failed(getConvDimNames(op, filterNames, inputNames, outputNames))) {
    return failure();
  }

  int64_t gcdStrideDilationH = math_util::gcd(strideH, dilationH);
  int64_t gcdStrideDilationW = math_util::gcd(strideW, dilationW);

  int64_t yTilda = strideH / gcdStrideDilationH;
  int64_t xTilda = strideW / gcdStrideDilationW;

  int64_t yDot = math_util::integer_divide_ceil(convDims.y, yTilda);
  int64_t xDot = math_util::integer_divide_ceil(convDims.x, xTilda);

  int64_t hTilda = convDims.ho + math_util::integer_divide_ceil(
                                     dilationH * (convDims.y - 1), strideH);
  int64_t wTilda = convDims.wo + math_util::integer_divide_ceil(
                                     dilationW * (convDims.x - 1), strideW);

  int64_t iHTildaLeft = math_util::integer_divide_floor(
      std::max(0l, leftPadH - dilationH * (yTilda - 1)), strideH);
  int64_t iWTildaLeft = math_util::integer_divide_floor(
      std::max(0l, leftPadW - dilationW * (xTilda - 1)), strideW);

  int64_t iHTildaRight = std::min(
      hTilda,
      math_util::integer_divide_ceil(leftPadH + convDims.hi - 1, strideH) + 1);
  int64_t iWTildaRight = std::min(
      wTilda,
      math_util::integer_divide_ceil(leftPadW + convDims.wi - 1, strideW) + 1);

  int64_t hTildaSlice = iHTildaRight - iHTildaLeft;
  int64_t wTildaSlice = iWTildaRight - iWTildaLeft;

  int64_t gemmId = gemmIdAttr.getInt();
  // In case the actual gemm ID is -1, emit the zero initialization kernel.
  if (gemmId < 0) {
    return zeroInit(op, b);
  }

  int64_t iYTilda = gemmId / xTilda;
  int64_t iXTilda = gemmId % xTilda;
  int64_t yDotSlice =
      math_util::integer_divide_ceil(convDims.y - iYTilda, yTilda);
  int64_t xDotSlice =
      math_util::integer_divide_ceil(convDims.x - iXTilda, xTilda);

  // backward data only, it's igemm v4r1 algo
  // c is input chaneels , k is output channels
  // n is batch , yDotSlice,xDotSlice computed in above
  int64_t gemmMSize = convDims.c;
  int64_t gemmKSize = convDims.k * yDotSlice * xDotSlice;
  int64_t gemmNSize = convDims.n * hTildaSlice * wTildaSlice;
  GemmContext gemmSize(gemmMSize, gemmKSize, gemmNSize);
  Optional<GemmContext> maybeGemmExtraPad;

  bool isXdlops = false;
  auto xdlopsV2Attr = op->template getAttrOfType<BoolAttr>("xdlopsV2");
  if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true)
    isXdlops = true;

  if (!isXdlops) {
    PopulateParams populateParams;
    maybeGemmExtraPad =
        calculatePaddingKernelSize(gemmSize, obtainConvDirection(op),
                                   obtainConvDataType(op), populateParams);
  } else { // xdlops
    PopulateParamsXDL populateParamsXDL;
    maybeGemmExtraPad =
        calculatePaddingKernelSize(gemmSize, obtainConvDirection(op),
                                   obtainConvDataType(op), populateParamsXDL);
  }
  auto gemmExtraPad = maybeGemmExtraPad.getValueOr(GemmContext(0, 0, 0));

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
    bool filterCheckPadGemmM = (gemmExtraPad.m > 0);
    bool filterCheckPadGemmK = (gemmExtraPad.k > 0);
    if (filterCheckPadGemmM || filterCheckPadGemmK) {
      auto padTransform = BottomUpTMBuilder::above(gemmFilterTransform,
                                                   gemmFilterTransformAttr);
      padTransform.passThrough("gemmG");
      if (filterCheckPadGemmK) {
        padTransform.pad("gemmKPad", "gemmK", 0, gemmExtraPad.k);
      } else {
        padTransform.passThrough("gemmK");
      }

      if (filterCheckPadGemmM) {
        padTransform.pad("gemmMPad", "gemmM", 0, gemmExtraPad.m);
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

    bool inputCheckPadGemmM = (gemmExtraPad.m > 0);
    bool inputCheckPadGemmN = (gemmExtraPad.n > 0);
    if (inputCheckPadGemmM || inputCheckPadGemmN) {
      auto padTransform =
          BottomUpTMBuilder::above(gemmTransform, gemmTransformAttr);
      padTransform.passThrough("gemmG");
      if (inputCheckPadGemmM) {
        padTransform.pad("gemmMPad", "gemmM", 0, gemmExtraPad.m);
      } else {
        padTransform.passThrough("gemmM");
      }

      if (inputCheckPadGemmN) {
        padTransform.pad("gemmNPad", "gemmN", 0, gemmExtraPad.n);
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

    bool outputCheckPadGemmK = (gemmExtraPad.k > 0);
    bool outputCheckPadGemmN = (gemmExtraPad.n > 0);
    if (outputCheckPadGemmK || outputCheckPadGemmN) {
      auto padTransform = BottomUpTMBuilder::above(gemmOutputTransform,
                                                   gemmOutputTransformAttr);
      padTransform.passThrough("gemmG");
      if (outputCheckPadGemmK) {
        padTransform.pad("gemmKPad", "gemmK", 0, gemmExtraPad.k);
      } else {
        padTransform.passThrough("gemmK");
      }

      if (outputCheckPadGemmN) {
        padTransform.pad("gemmNPad", "gemmN", 0, gemmExtraPad.n);
      } else {
        padTransform.passThrough("gemmN");
      }

      TransformMapAttr padTransformAttr = padTransform.get();
      // Replace output gemm with padded version
      gemmOutputTransformAttr = padTransformAttr;
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

  auto paddingInfo = PaddingInfoAttr::get(b.getContext(), gemmExtraPad.m,
                                          gemmExtraPad.k, gemmExtraPad.n);

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

    ConvolutionContext ctx = populateConvContext(op);

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
    int64_t leftPadH = ctx.getPaddingVal()[0];
    int64_t leftPadW = ctx.getPaddingVal()[2];
    int64_t rightPadH = ctx.getPaddingVal()[1];
    int64_t rightPadW = ctx.getPaddingVal()[3];

    int64_t dilationH = ctx.getDilationVal()[0];
    int64_t dilationW = ctx.getDilationVal()[1];
    int64_t strideH = ctx.getStrideVal()[0];
    int64_t strideW = ctx.getStrideVal()[1];
    ConvolutionDims convDims = ctx.getConvDims();

    llvm::SmallVector<StringRef, 5> filterNames, inputNames, outputNames;
    if (failed(getConvDimNames(op, filterNames, inputNames, outputNames))) {
      return failure();
    }

    // compute we should use extra padding kernel or not
    // c,k already / g ,so we can skip / g here
    GemmContext gemmSize = GemmContext::fromConvolution(convOpType, convDims);
    Optional<GemmContext> maybeGemmExtraPad;

    if (!isXdlops) {
      PopulateParams populateParams;
      maybeGemmExtraPad = calculatePaddingKernelSize(gemmSize, convOpType,
                                                     dataType, populateParams);
    } else { // xdlops
      PopulateParamsXDL populateParamsXDL;
      maybeGemmExtraPad = calculatePaddingKernelSize(
          gemmSize, convOpType, dataType, populateParamsXDL);
    }

    if (ConvOpType::BwdWeight == convOpType && isXdlops &&
        (dataType == b.getF32Type() || dataType == b.getF16Type()) &&
        !maybeGemmExtraPad.hasValue()) {
      // current backward weight with atomic_add can only run under xdlops +
      // fp32 / fp16.
      return backwardWeightAtomicAdd(cast<Conv2DBwdWeightOp>(op), b);
    }
    auto gemmExtraPad = maybeGemmExtraPad.getValueOr(GemmContext(0, 0, 0));

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

    bool noNonKPad =
        (convOpType == ConvOpType::BwdWeight && gemmExtraPad.n == 0) ||
        (convOpType == ConvOpType::Fwd && gemmExtraPad.k == 0);

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
        (convOpType == ConvOpType::Fwd && gemmExtraPad.m > 0) ||
        (convOpType == ConvOpType::BwdWeight && gemmExtraPad.m > 0);
    filterCheckPadGemmK = (convOpType == ConvOpType::Fwd && gemmExtraPad.k > 0);
    filterCheckPadGemmN =
        (convOpType == ConvOpType::BwdWeight && gemmExtraPad.n > 0);
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
        padGemmFilterTransform.pad("gemmKPad", "gemmK", 0, gemmExtraPad.k);
      } else if (convOpType != ConvOpType::BwdWeight) {
        // Backward weight has no GemmK on its filter
        padGemmFilterTransform.passThrough("gemmK");
      }

      if (filterCheckPadGemmM) {
        padGemmFilterTransform.pad("gemmMPad", "gemmM", 0, gemmExtraPad.m);
      } else {
        padGemmFilterTransform.passThrough("gemmM");
      }

      if (filterCheckPadGemmN) {
        padGemmFilterTransform.pad("gemmNPad", "gemmN", 0, gemmExtraPad.n);
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
    embedInputWrap.embed({"y", "ho"}, {convDims.y, convDims.ho}, "hipad",
                         {dilationH, strideH});
    embedInputWrap.embed({"x", "wo"}, {convDims.x, convDims.wo}, "wipad",
                         {dilationW, strideW});

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
        (convOpType == ConvOpType::Fwd && gemmExtraPad.k > 0) ||
        (convOpType == ConvOpType::BwdWeight && gemmExtraPad.k > 0);
    inputCheckPadGemmN =
        (convOpType == ConvOpType::Fwd && gemmExtraPad.n > 0) ||
        (convOpType == ConvOpType::BwdWeight && gemmExtraPad.n > 0);
    bool isInputPad = false;
    if (inputCheckPadGemmK || inputCheckPadGemmN) {
      isInputPad = true;
      padGemmInputTransform =
          BottomUpTMBuilder::above(gemmInputTransform, gemmInputTransformAttr);
      padGemmInputTransform.passThrough("gemmG");
      if (inputCheckPadGemmK) {
        padGemmInputTransform.pad("gemmKPad", "gemmK", 0, gemmExtraPad.k);
      } else {
        padGemmInputTransform.passThrough("gemmK");
      }

      if (inputCheckPadGemmN) {
        padGemmInputTransform.pad("gemmNPad", "gemmN", 0, gemmExtraPad.n);
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
        (convOpType == ConvOpType::BwdWeight && gemmExtraPad.k > 0);
    outputCheckPadGemmM =
        (convOpType == ConvOpType::BwdWeight && gemmExtraPad.m > 0) ||
        (convOpType == ConvOpType::Fwd && gemmExtraPad.m > 0);
    outputCheckPadGemmN = (convOpType == ConvOpType::Fwd && gemmExtraPad.n > 0);
    bool isOutputPad = false;
    if (outputCheckPadGemmK || outputCheckPadGemmM || outputCheckPadGemmN) {
      isOutputPad = true;
      padGemmOutputTransform =
          BottomUpTMBuilder::above(outputTransform, outputTransformAttr);
      padGemmOutputTransform.passThrough("gemmG");

      if (outputCheckPadGemmK) {
        padGemmOutputTransform.pad("gemmKPad", "gemmK", 0, gemmExtraPad.k);
      } else if (convOpType != ConvOpType::Fwd) {
        padGemmOutputTransform.passThrough("gemmK");
      }

      if (outputCheckPadGemmM) {
        padGemmOutputTransform.pad("gemmMPad", "gemmM", 0, gemmExtraPad.m);
      } else {
        padGemmOutputTransform.passThrough("gemmM");
      }

      if (outputCheckPadGemmN) {
        padGemmOutputTransform.pad("gemmNPad", "gemmN", 0, gemmExtraPad.n);
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
    auto paddingInfo = PaddingInfoAttr::get(b.getContext(), gemmExtraPad.m,
                                            gemmExtraPad.k, gemmExtraPad.n);

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

// MITPRewritePattern
// Fold linarg.generic and memref.alloc generated by transpose op into
// miopen.transform
struct RemoveTrivialTransposePattern
    : public OpRewritePattern<linalg::GenericOp> {
  // Explicit constructor to set a higher pattern benefic than the more general
  // pattern below.
  explicit RemoveTrivialTransposePattern(MLIRContext *ctx)
      : OpRewritePattern<linalg::GenericOp>(ctx, /*benefit=*/2) {}

  miopen::TransformOp makeTranspose(PatternRewriter &b, Value inp,
                                    const AffineMapAttr &inMap,
                                    const AffineMapAttr &outMap) const {
    AffineMap inpIdxMap = inMap.getAffineMap();
    AffineMap outpIdxMap = outMap.getAffineMap();
    Location loc = inp.getLoc();
    MemRefType inpType = inp.getType().template cast<MemRefType>();
    ArrayRef<int64_t> inpShape = inpType.getShape();

    SmallVector<uint32_t, 8> endDims;
    SmallVector<uint32_t, 8> startDims;
    for (uint32_t i = 0, e = inpShape.size(); i < e; ++i) {
      startDims.push_back(i);
      uint32_t inMapped = inpIdxMap.getDimPosition(i);
      endDims.push_back(outpIdxMap.getDimPosition(inMapped));
    }
    miopen::BottomUpTMBuilder transform(b, inpShape, loc);
    transform.passThrough(endDims, startDims);
    auto tfOp = b.create<miopen::TransformOp>(loc, inp, transform.get(),
                                              inpType.getMemorySpaceAsInt());
    return tfOp;
  }

  LogicalResult matchAndRewrite(linalg::GenericOp laGeneric,
                                PatternRewriter &b) const override {
    // 0. Test compatibility
    // 0.0. Only fully parallel for now
    for (StringRef itr :
         laGeneric.iterator_types().getAsValueRange<StringAttr>()) {
      if (itr != "parallel") {
        return failure();
      }
    }

    bool bPassing = false;
    laGeneric.getRegion().walk([&](linalg::YieldOp yieldOp) {
      Value laReturn = yieldOp->getOperand(0);
      bPassing = (laReturn == laGeneric.getRegion().getArgument(0));
    });

    // 0.1. Test it only passes through 1:1 and no other calculation
    if (laGeneric.inputs().size() != 1 || laGeneric.outputs().size() != 1 ||
        !bPassing) {
      return failure();
    }

    // 0.2. linalg.generic lowered from tosa.transpose should have memref.alloc
    Value out = *laGeneric.outputs().begin();
    auto allocToDel = out.getDefiningOp<memref::AllocOp>();
    if (!allocToDel) {
      return failure();
    }

    // get maps to construct a transforming map for the transpose
    auto idxMaps =
        laGeneric->template getAttrOfType<ArrayAttr>("indexing_maps");
    AffineMapAttr inIdxMap = idxMaps[0].cast<AffineMapAttr>();
    AffineMapAttr outIdxMap = idxMaps[1].cast<AffineMapAttr>();
    auto tpTransform =
        makeTranspose(b, laGeneric->getOperand(0), inIdxMap, outIdxMap);

    b.replaceOp(allocToDel, {tpTransform});
    b.eraseOp(laGeneric);
    return success();
  }
};

/// If there is a chain of operations that leads from `v` to
/// a miopen.conv2d* op, return that convolution.
static Operation *getConvUser(Value v) {
  for (Operation *user : v.getUsers()) {
    if (isa<Conv2DOp, Conv2DBwdDataOp, Conv2DBwdWeightOp>(user))
      return user;
    if (auto transform = dyn_cast<TransformOp>(user))
      if (Operation *upstream = getConvUser(transform.output()))
        return upstream;
  }
  return nullptr;
}

/// If the input to a linalg.generic is the output of a conv2d and the indexing
/// map for that input is a non-trivial permutation of an identity, convert that
/// indexing map to a transpose. This must happen before gridwise gemm
/// conversion because all the transforms on the convolution output are
/// collected at that time.
struct FoldTransposingConvAccess : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp laGeneric,
                                PatternRewriter &b) const override {
    Location loc = laGeneric->getLoc();
    // We do this out-of-line so as to not invalidate our iterator
    SmallVector<std::tuple<unsigned, Operation *, Value, AffineMap,
                           SmallVector<uint32_t, 4>>>
        toReplace;

    for (OpOperand &operand : laGeneric->getOpOperands()) {
      Value opValue = operand.get();
      Operation *convUser = getConvUser(opValue);
      if (!convUser)
        continue;

      for (Operation *user : opValue.getUsers()) {
        if (isa<linalg::GenericOp>(user) && user != laGeneric) {
          LLVM_DEBUG(llvm::dbgs() << "Multiple generics on same conv output\n");
          return failure();
        }
      }

      if (!isa_and_nonnull<memref::AllocOp>(opValue.getDefiningOp()))
        continue;

      AffineMap idxMap = laGeneric.getTiedIndexingMap(&operand);
      if (idxMap.isMinorIdentityWithBroadcasting())
        continue;
      SmallVector<uint32_t, 4> permutation;
      if (!idxMap.isPermutationOfMinorIdentityWithBroadcasting(permutation))
        continue;

      unsigned opIndex = operand.getOperandNumber();
      toReplace.emplace_back(opIndex, convUser, opValue, idxMap, permutation);
    }

    // Actually do the rewrites, if any
    for (auto &tuple : toReplace) {
      unsigned opIndex = std::get<0>(tuple);
      Operation *convolution = std::get<1>(tuple);
      Value opValue = std::get<2>(tuple);
      AffineMap idxMap = std::get<3>(tuple);
      SmallVector<uint32_t, 4> permutation = std::get<4>(tuple);
      LLVM_DEBUG(llvm::dbgs() << "Replacing index map with permutation ");
      LLVM_DEBUG(llvm::interleaveComma(permutation, llvm::dbgs()));
      LLVM_DEBUG(llvm::dbgs() << "\n");

      auto allocation = cast<memref::AllocOp>(opValue.getDefiningOp());
      // Swap out the allocation for the form it needs to take in order to
      // eliminate the non-trivial map.
      ArrayRef<int64_t> shape = opValue.getType().cast<MemRefType>().getShape();
      SmallVector<int64_t, 4> newShape(shape.size(), -1LL);
      SmallVector<uint32_t, 4> endIdentity;
      for (uint32_t i = 0, e = shape.size(); i < e; ++i) {
        endIdentity.push_back(i);
        newShape[permutation[i]] = shape[i];
      }

      // All this new stuff needs to go where the old memref.alloc was
      PatternRewriter::InsertionGuard guard(b);
      b.setInsertionPointAfterValue(allocation);
      auto newAllocType = allocation.getType()
                              .cast<MemRefType>()
                              .clone(newShape)
                              .cast<MemRefType>();
      Value newAlloc =
          b.replaceOpWithNewOp<memref::AllocOp>(allocation, newAllocType);

      miopen::BottomUpTMBuilder permuteMapBuilder(b, newShape, loc);
      permuteMapBuilder.passThrough(endIdentity, permutation);
      TransformMapAttr permuteMapAttr = permuteMapBuilder.get();
      auto permuted =
          b.create<miopen::TransformOp>(loc, newAlloc, permuteMapAttr);
      llvm::SmallPtrSet<Operation *, 2> skips = {laGeneric, permuted};
      newAlloc.replaceAllUsesExcept(permuted, skips);

      // Correct indexing maps
      AffineMap composed =
          idxMap.compose(permuteMapAttr.getMap().getAffineMap());
      Attribute newMaps =
          laGeneric.indexing_maps().replaceImmediateSubAttribute(
              {{opIndex, AffineMapAttr::get(composed)}});
      laGeneric->setAttr(laGeneric.indexing_mapsAttrName(), newMaps);

      // Correct convolution so it's not vectorized
      // TODO(kdrewnia): Maybe be more intelligent here
      convolution->setAttr("matrix_c_data_per_copy", b.getI32IntegerAttr(1));
    }

    return success(!toReplace.empty());
  }
};

void MIOpenConvToGemmPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);

  RewritePatternSet patternsTP(ctx);
  patternsTP.add<RemoveTrivialTransposePattern, FoldTransposingConvAccess>(ctx);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patternsTP))))
    signalPassFailure();

  target.addIllegalOp<miopen::Conv2DOp, miopen::Conv2DBwdDataOp,
                      miopen::Conv2DBwdWeightOp>();
  target.addLegalOp<miopen::TransformOp, miopen::GridwiseGemmOp,
                    miopen::GridwiseGemmV2Op, miopen::WorkgroupIdOp,
                    miopen::WorkitemIdOp, miopen::BufferLoadOp,
                    miopen::BufferStoreOp>();
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

std::unique_ptr<Pass> mlir::miopen::createMIOpenConvToGemmPass() {
  return std::make_unique<MIOpenConvToGemmPass>();
}
