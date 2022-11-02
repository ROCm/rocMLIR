//===- ConvToGemm.cpp - MLIR Rock ops lowering passes ------------===//
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
// This pass converts rock.conv2d into rock.transform and
// rock.gemm.
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/UtilityParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKCONVTOGEMMPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-conv-to-gemm"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;
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

struct RockConvToGemmPass
    : public rock::impl::RockConvToGemmPassBase<RockConvToGemmPass> {
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

/// Return the type of v if the underlying convolution has a result, otherwise
/// return null, allowing the lowering here to be, in principle, generic over
/// tensors and memrefs.
Type getResultType(Operation *convOp, Value outArg) {
  if (convOp->getNumResults() == 1)
    return outArg.getType();
  return nullptr;
}

/// Create an elementwise utility kernel.
/// The callback has type (builder, location, collapsedBuffers, coordinate).
/// Note: you are expected to handle out of bounds, such as by using
/// rock.buffer_store
LogicalResult createElementwiseLoop(
    OpBuilder &b, Location loc, Operation *convOp, ValueRange memrefs,
    int64_t vectorLen,
    function_ref<void(OpBuilder &, Location, ValueRange, Value)> emitBodyFunc) {
  RockGemmWrapperInterface gemmWrapper =
      convOp->getResult(0).getDefiningOp<RockGemmWrapperInterface>();
  uint32_t blockSize = gemmWrapper.getBlockSize();
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
    if (!memref.getType().isa<MemRefType>()) {
      // TODO: determine if we can relax this if we push bufferization down
      return convOp->emitOpError(
          "Arguments to utility kernels must be memrefs");
    }
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

/// 0-initialize the output for a backward data convolution.
/// The output is the input tensor.
LogicalResult zeroInit(Conv2DBwdDataOp op, PatternRewriter &b) {
  auto loc = op.getLoc();
  TypedValue<ShapedType> output = op.getInput();
  Type outputType = output.getType().getElementType();
  constexpr int64_t kZeroInitVecLen = 4;
  Type storeType = VectorType::get(kZeroInitVecLen, outputType);
  Value zeroOp = createZeroConstantOp(b, loc, storeType);
  ArrayAttr leftOob = b.getI32ArrayAttr({});
  ArrayAttr rightOob = b.getI32ArrayAttr({0});

  auto loopBody = [&zeroOp, &leftOob, &rightOob](OpBuilder &b, Location loc,
                                                 ValueRange collapsed,
                                                 Value index) {
    b.create<BufferStoreOp>(loc, zeroOp, collapsed[0], leftOob, rightOob, index,
                            b.getAttr<StoreMethodAttr>(StoreMethod::Set),
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
  Type filterDataType = op.getFilter().getType().getElementType();
  TypedValue<ShapedType> output(nullptr);
  if (filterDataType == b.getF32Type()) {
    output = op.getFilter();
  } else if (filterDataType == b.getF16Type()) {
    if (!op.getWorkspace())
      return op.emitOpError("op has no workspace");
    output = op.getWorkspace();
  } else {
    return op.emitOpError("Unsupported zeroing data type");
  }

  Type outputType = output.getType().getElementType();
  constexpr int64_t kZeroInitVecLen = 4;
  Type storeType = VectorType::get(kZeroInitVecLen, outputType);
  Value zeroOp = createZeroConstantOp(b, loc, storeType);
  ArrayAttr leftOob = b.getI32ArrayAttr({});
  ArrayAttr rightOob = b.getI32ArrayAttr({0});

  auto loopBody = [&zeroOp, &leftOob, &rightOob](OpBuilder &b, Location loc,
                                                 ValueRange collapsed,
                                                 Value index) {
    b.create<BufferStoreOp>(loc, zeroOp, collapsed[0], leftOob, rightOob, index,
                            b.getAttr<StoreMethodAttr>(StoreMethod::Set),
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
  if (!op.getWorkspace())
    return op.emitOpError("op has no workspace");
  TypedValue<ShapedType> filter = op.getFilter();
  TypedValue<ShapedType> workspace = op.getWorkspace();
  Type filterDataType = filter.getType().getElementType();
  Type workspaceDataType = workspace.getType().getElementType();

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
    b.create<BufferStoreOp>(loc, converted, collapsed[1], leftOob, rightOob,
                            index, b.getAttr<StoreMethodAttr>(StoreMethod::Set),
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

  Attribute tuningParams = op.getParamsAttr();
  if (!tuningParams) {
    return op.emitOpError("can't lower without tuning parameters\n");
  }

  if (!op.getKBlocks().has_value())
    return op.emitOpError("must have kBlocks set at lowering");
  int64_t gemmKBlocks = op.getKBlocks()->getZExtValue();

  ConvolutionContext ctx = populateConvContext(op);

  GemmFeatures features = op.getFeatures();
  bool isXdlops = bitEnumContainsAll(features, GemmFeatures::mfma);

  // Get shape of filter tensor.
  ShapedType filterType = op.getFilter().getType();
  auto filterShape = filterType.getShape();

  // Determine whether to use workspace.
  bool hasWorkspace =
      (filterType.getElementType() == b.getF16Type() && isXdlops);
  if (hasWorkspace && !op.getWorkspace()) {
    return op.emitOpError(
        "workspace needed for f16 atomic add but none provided");
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

  if (!isXdlops)
    return op->emitOpError("atomic add kernel requires xdlops");

  // Get shape of input tensor.
  ShapedType inputType = op.getInput().getType();
  ArrayRef<int64_t> inputShape = inputType.getShape();

  // Get shape of output tensor.
  ShapedType outputType = op.getOutput().getType();
  ArrayRef<int64_t> outputShape = outputType.getShape();

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
    Value filterTensorInUse =
        (hasWorkspace) ? op.getWorkspace() : op.getFilter();
    Value withKBlock = b.create<rock::TransformOp>(loc, filterTensorInUse,
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
        b.create<TransformOp>(loc, op.getInput(), firstTransformAttr);

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
        b.create<TransformOp>(loc, op.getOutput(), firstTransformAttr);

    // Map G and N0 to gemmG, N1HW to gemmK and K to gemmM
    auto gemmOutputTransform =
        BottomUpTMBuilder::above(firstTransform, firstTransformAttr);
    gemmOutputTransform.merge("gemmG", 0, {"go", "n0"});
    gemmOutputTransform.merge("gemmK", 1, {"n1", "ho", "wo"});
    gemmOutputTransform.passThrough({"gemmM"}, {2}, {"ko"});

    TransformMapAttr gemmOutputTransformAttr = gemmOutputTransform.get();
    gemmOutput =
        b.create<TransformOp>(loc, transformed, gemmOutputTransformAttr);
  }

  // This kernel is not run when there is padding on the GEMM
  auto storeMethod = b.getAttr<StoreMethodAttr>(StoreMethod::AtomicAdd);

  auto gemm = b.create<GemmOp>(
      loc, getResultType(op, gemmFilter), gemmOutput, gemmInput, gemmFilter,
      /*aTransposed=*/b.getUnitAttr(), /*bTransposed=*/nullptr,
      /*cTransposed=*/nullptr, op.getArchAttr(), op.getNumCuAttr(),
      op.getFeaturesAttr(), storeMethod, op.getDerivedBlockSizeAttr(),
      op.getGridSizeAttr(), op.getParamsAttr());
  gemm->setAttr("gemm_id", gemmIdAttr);

  // Finally, erase the original Conv2D op.
  b.eraseOp(op);

  return success();
}

LogicalResult backwardData(Conv2DBwdDataOp op, PatternRewriter &b) {
  Location loc = op.getLoc();
  auto gemmIdAttr = op->getAttrOfType<IntegerAttr>("gemm_id");

  ConvolutionContext ctx = populateConvContext(op);

  // Get shape of filter tensor.
  ShapedType filterType = op.getFilter().getType();
  ArrayRef<int64_t> filterShape = filterType.getShape();

  // Get shape of input tensor.
  ShapedType inputType = op.getInput().getType();
  ArrayRef<int64_t> inputShape = inputType.getShape();

  // Get shape of output tensor.
  ShapedType outputType = op.getOutput().getType();
  ArrayRef<int64_t> outputShape = outputType.getShape();

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

  Value gemmFilter, gemmInput, gemmOutput;
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
        b.create<TransformOp>(loc, op.getFilter(), embedTransformAttr);

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
  }

  // Transform input tensor
  {
    BottomUpTMBuilder padInputTransform(b, inputNames, inputShape, loc);
    padInputTransform.passThrough({"gi", "ni", "ci"});
    padInputTransform.pad({"hipad", "wipad"},
                          {padInputTransform.startIndex("hi"),
                           padInputTransform.startIndex("wi")},
                          {"hi", "wi"},
                          {leftPadH, rightPadH, leftPadW, rightPadW});

    TransformMapAttr padTransformAttr = padInputTransform.get();
    Value paddedInput =
        b.create<TransformOp>(loc, op.getInput(), padTransformAttr);

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
        b.create<TransformOp>(loc, op.getOutput(), embedTransformAttr);

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
  }

  // Emit rock.gemm op.
  auto storeMethod = b.getAttr<StoreMethodAttr>(StoreMethod::Set);
  auto gemm = b.create<GemmOp>(
      loc, getResultType(op, gemmInput), gemmFilter, gemmOutput, gemmInput,
      /*aTransposed=*/b.getUnitAttr(), /*bTransposed=*/nullptr,
      /*cTransposed=*/nullptr, op.getArchAttr(), op.getNumCuAttr(),
      op.getFeaturesAttr(), storeMethod, op.getDerivedBlockSizeAttr(),
      op.getGridSizeAttr(), op.getParamsAttr());
  gemm->setAttr("gemm_id", gemmIdAttr);

  // Finally, erase the original Conv2D op.
  b.eraseOp(op);

  return success();
}

template <typename T> struct Conv2DRewritePattern : public OpRewritePattern<T> {
  const static ArgumentFields fields;
  const static ConvOpType convOpType;
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op, PatternRewriter &b) const override {
    GemmFeatures features = op.getFeatures();

    Type dataType = op.getInput().getType().getElementType();
    if (ConvOpType::BwdData == convOpType) {
      return backwardData(cast<Conv2DBwdDataOp>(op), b);
    }
    Location loc = op.getLoc();

    ConvolutionContext ctx = populateConvContext(op);

    // Get shape of filter tensor.
    ShapedType filterType = op.getFilter().getType();
    ArrayRef<int64_t> filterShape = filterType.getShape();

    // Get shape of input tensor.
    ShapedType inputType = op.getInput().getType();
    ArrayRef<int64_t> inputShape = inputType.getShape();

    // Get shape of output tensor.
    ShapedType outputType = op.getOutput().getType();
    ArrayRef<int64_t> outputShape = outputType.getShape();

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

    Attribute tuningParams = op.getParamsAttr();
    GemmSize gemmSize = op.getGemmSize();
    Optional<GemmSize> maybeGemmExtraPad;

    if (tuningParams) {
      maybeGemmExtraPad = requiredPadding(tuningParams, gemmSize);
    } else {
      // We don't know if this'll be a padding kernel, so we can't promise an
      // unfold or rely on atomic add, and so set the extraPad to a nonsense but
      // existing value.
      maybeGemmExtraPad = GemmSize{-1, -1, -1, -1};
    }

    // TODO: don't restrict this to xdlops only once we've validated on a gfx11
    // machine
    if (ConvOpType::BwdWeight == convOpType &&
        isWrWAtomicKernel(features, dataType, maybeGemmExtraPad.has_value())) {
      return backwardWeightAtomicAdd(cast<Conv2DBwdWeightOp>(op), b);
    }
    auto gemmExtraPad = maybeGemmExtraPad.value_or(GemmSize{0, 0, 0, 0});

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
        b.create<TransformOp>(loc, op.getFilter(), filterTransformAttr);

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
        b.create<TransformOp>(loc, op.getInput(), padInputTransformAttr);

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
        b.create<TransformOp>(loc, op.getOutput(), outputTransformAttr);

    SmallVector<Value, 3> arguments = {gemmFilter, gemmInput, gemmOutput};

    Value gemmA, gemmB, gemmC;
    gemmA = arguments[fields.gridwiseGemmArgumentPosition[0]];
    gemmB = arguments[fields.gridwiseGemmArgumentPosition[1]];
    gemmC = arguments[fields.gridwiseGemmArgumentPosition[2]];

    // Emit rock.gemm op.
    auto storeMethod = b.getAttr<StoreMethodAttr>(StoreMethod::Set);
    b.create<GemmOp>(loc, getResultType(op, gemmC), gemmA, gemmB, gemmC,
                     /*aTransposed=*/b.getUnitAttr(), /*bTransposed=*/nullptr,
                     /*cTransposed=*/nullptr, op.getArchAttr(),
                     op.getNumCuAttr(), op.getFeaturesAttr(), storeMethod,
                     op.getDerivedBlockSizeAttr(), op.getGridSizeAttr(),
                     tuningParams);

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

void RockConvToGemmPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);

  target.addIllegalOp<rock::Conv2DOp, rock::Conv2DBwdDataOp,
                      rock::Conv2DBwdWeightOp>();
  target.addLegalOp<rock::TransformOp, rock::GemmOp, rock::WorkgroupIdOp,
                    rock::WorkitemIdOp, rock::BufferLoadOp,
                    rock::BufferStoreOp>();
  // Below are required legalize for the lowering of Conv2DBwdWeightOp
  target.addLegalDialect<arith::ArithmeticDialect, memref::MemRefDialect,
                         scf::SCFDialect>();

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
