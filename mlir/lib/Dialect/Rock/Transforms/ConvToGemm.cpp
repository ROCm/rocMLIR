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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockConvInterface.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/UtilityParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <iterator>

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
// Conv (forward, backward) lowering.
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
  //   printf("checkNames({");
  //   for (auto name : actual) printf("%s,", name);
  //   printf("}, {");
  //   for (auto name : expected) printf("%s,", name);
  //   printf("}\n");

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

// Sort the dimensions in `names` so that they are in the order they appear in
// within `transform`. This allows Merge{} operations to not preform
// transposes that are not needed.
void matchUnderlyingOrder(SmallVectorImpl<StringRef> &names,
                          BottomUpTMBuilder &transform) {
  std::sort(names.begin(), names.end(),
            [&transform](const StringRef &v1, const StringRef &v2) -> bool {
              return transform.startIndex(v1) < transform.startIndex(v2);
            });
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

  auto update_old_name = [](StringAttr name) {
#if 1
    auto ctx = name.getContext();
    if (name == "y")
      return StringAttr::get(ctx, "0");
    if (name == "x")
      return StringAttr::get(ctx, "1");
    if (name.strref().starts_with_insensitive("h")) {
      auto namestr = name.str();
      return StringAttr::get(ctx, std::string("0") +
                                      namestr.substr(1, namestr.length() - 1));
    }
    if (name.strref().starts_with_insensitive("w")) {
      auto namestr = name.str();
      return StringAttr::get(ctx, std::string("1") +
                                      namestr.substr(1, namestr.length() - 1));
    }
#endif /* 1 */
    return name;
  };

  for (unsigned i = 0; i < size; ++i) {
    auto filterAttr = update_old_name(
        filterLayoutAttr.getValue()[i].template cast<StringAttr>());
    auto inputAttr = update_old_name(
        inputLayoutAttr.getValue()[i].template cast<StringAttr>());
    auto outputAttr = update_old_name(
        outputLayoutAttr.getValue()[i].template cast<StringAttr>());

    filterNames.push_back(filterAttr.getValue());
    inputNames.push_back(inputAttr.getValue());
    outputNames.push_back(outputAttr.getValue());
  }
  if (failed(
          checkNames(filterNames, {"k", "g", "c", "0", "1"}, "filter", op)) ||
      failed(checkNames(inputNames, {"ni", "gi", "ci", "0i", "1i"}, "input",
                        op)) ||
      failed(checkNames(outputNames, {"no", "go", "ko", "0o", "1o"}, "output",
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

static int64_t getUtilityVectorizationLen(ShapedType shape,
                                          int64_t elemsPerThread) {
  int64_t numElems = shape.getNumElements();
  constexpr int64_t kMaxVectorOpLen = 4; // words
  int64_t elemsPerWord = (32 / shape.getElementTypeBitWidth());
  return math_util::gcd(math_util::gcd(numElems, elemsPerThread),
                        kMaxVectorOpLen * elemsPerWord);
}

/// Create an elementwise utility kernel.
/// The callback has type (builder, location, collapsedBuffers, coordinate).
/// Note: you are expected to handle out of bounds, such as by using
/// rock.buffer_store
template <typename OpType>
LogicalResult createElementwiseLoop(
    OpBuilder &b, Location loc, OpType kernelOp, ValueRange memrefs,
    int64_t vectorLen,
    function_ref<void(OpBuilder &, Location, ValueRange, Value)> emitBodyFunc) {
  if (!kernelOp.getBlockSize().has_value())
    return kernelOp.emitOpError("block size not defined for utility kernel");
  if (!kernelOp.getGridSize().has_value())
    return kernelOp.emitOpError("grid size not defined for utility kernel");
  if (!kernelOp.getElemsPerThread().has_value())
    return kernelOp.emitOpError(
        "elemsPerThread not defined fer utility kernel");
  uint32_t blockSize = *kernelOp.getBlockSize();
  int64_t elemsPerThread = kernelOp.getElemsPerThread()->getSExtValue();
  if (elemsPerThread % vectorLen != 0)
    return kernelOp.emitOpError("unevenly vectorized elementwise kernel");

  Value workgroupId = b.create<WorkgroupIdOp>(loc, b.getIndexType());
  Value workgroupDim = b.create<ConstantIndexOp>(loc, blockSize);
  Value elemsPerThreadOp = b.create<ConstantIndexOp>(loc, elemsPerThread);
  Value workitemId = b.create<WorkitemIdOp>(loc, b.getIndexType());

  SmallVector<Value, 2> collapsedBufs;
  for (Value memref : memrefs) {
    if (!memref.getType().isa<MemRefType>()) {
      // TODO: determine if we can relax this if we push bufferization down
      return kernelOp.emitOpError(
          "arguments to utility kernels must be memrefs");
    }
    if (auto transform =
            dyn_cast_or_null<TransformOp>(memref.getDefiningOp())) {
      return kernelOp.emitOpError(
          "arguments to utility kernels must be pure memrefs");
    }
    Value collapsed = createCollapseShapeOp(b, loc, memref);
    collapsedBufs.push_back(collapsed);
  }
  int64_t collapsedLen =
      collapsedBufs[0].getType().cast<MemRefType>().getShape()[0];
  for (Value c : collapsedBufs)
    if (c.getType().cast<MemRefType>().getNumElements() != collapsedLen)
      return kernelOp.emitOpError(
          "utility kernel arguments have different lengths");

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

/// Create a private buffer that can hold type `type`.
static Value makePrivateGpuAlloc(OpBuilder &b, Location loc, Type type) {
  Type elemTy = type;
  int64_t numElems = 1;
  if (auto vecTy = dyn_cast<VectorType>(type)) {
    elemTy = vecTy.getElementType();
    numElems = vecTy.getNumElements();
  }
  auto memrefTy =
      MemRefType::get(numElems, elemTy, nullptr,
                      gpu::AddressSpaceAttr::get(type.getContext(),
                                                 gpu::AddressSpace::Private));
  Value memref = b.create<rock::GpuAllocOp>(loc, memrefTy);
  return memref;
}

/// Store `value` into a private memref buffer to make it an acceptable argument
/// to memref.store. Returns the allocated buffer.
static Value makeGpuAllocContaining(OpBuilder &b, Value v) {
  Location loc = v.getLoc();
  Type type = v.getType();
  Value memref = makePrivateGpuAlloc(b, loc, type);
  b.create<rock::InBoundsStoreOp>(
      loc, v, memref, b.createOrFold<arith::ConstantIndexOp>(loc, 0));
  return memref;
}

/// 0-initialize a given buffer.
struct ZeroInitKernelRewritePattern final
    : public OpConversionPattern<InitKernelOp> {
  using OpConversionPattern<InitKernelOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(InitKernelOp op, InitKernelOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    TypedValue<ShapedType> buffer = op.getBuffer();
    Type bufferType = buffer.getType().getElementType();
    if (!op.getElemsPerThread().has_value())
      return op->emitOpError("elems per thread not set");

    int64_t initVectorLen = getUtilityVectorizationLen(
        buffer.getType(), op.getElemsPerThread()->getZExtValue());
    Type storeType = vectorTypeOrSelf(bufferType, initVectorLen);
    bool needs64BitIdx = is4GBMemoryType(buffer.getType());

    Type elementType = mlir::getElementTypeOrSelf(storeType);
    Value initOp;
    auto initValueAttr = op.getInitValueAttr();
    if (initValueAttr) {
      if (auto floatInitValueAttr = initValueAttr.value().cast<FloatAttr>()) {
        auto initValue = floatInitValueAttr.getValue().convertToFloat();
        initOp =
            createConstantFloatOp(b, loc, storeType, elementType, initValue);
      } else if (auto intInitValueAttr =
                     initValueAttr.value().cast<IntegerAttr>()) {
        auto initValue = intInitValueAttr.getValue().getSExtValue();
        initOp = createConstantIntOp(b, loc, storeType, elementType, initValue);
      } else {
        return failure();
      }
    } else {
      initOp = createZeroConstantOp(b, loc, storeType);
    }

    Value zeroIndex = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
    Value memref = makeGpuAllocContaining(b, initOp);
    Value trueOp =
        b.createOrFold<arith::ConstantIntOp>(loc, true, b.getI1Type());
    GemmFeatures features = op.getFeatures();

    auto loopBody = [&memref, &initVectorLen, &trueOp, &features, &zeroIndex,
                     &needs64BitIdx](OpBuilder &b, Location loc,
                                     ValueRange collapsed, Value index) {
      b.create<GlobalStoreOp>(loc, memref, collapsed[0],
                              APInt(64, initVectorLen), features,
                              StoreMethod::Set, /*sourceCoord=*/zeroIndex,
                              /*valid=*/trueOp, index, needs64BitIdx,
                              /*canStoreOffEnd=*/true);
    };
    LogicalResult res =
        createElementwiseLoop(b, loc, op, buffer, initVectorLen, loopBody);
    if (failed(res))
      return failure();

    b.eraseOp(op);
    return success();
  }
};

/// Element-wise conversion from the workspace to the output (filter tensor)
/// for a backward weight convolution which uses atomic adds.
struct ConvertingCopyKernelRewritePattern final
    : public OpConversionPattern<ConvertingCopyKernelOp> {
  using OpConversionPattern<ConvertingCopyKernelOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ConvertingCopyKernelOp op,
                                ConvertingCopyKernelOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    auto input = cast<TypedValue<ShapedType>>(adaptor.getInput());
    auto output = cast<TypedValue<ShapedType>>(adaptor.getOutput());
    Type inputDataType = input.getType().getElementType();
    Type outputDataType = output.getType().getElementType();
    if (!op.getElemsPerThread().has_value())
      return op->emitOpError("elems per thread not set");

    int64_t conversionVectorLen = getUtilityVectorizationLen(
        input.getType(), op.getElemsPerThread()->getZExtValue());

    Type loadType = vectorTypeOrSelf(inputDataType, conversionVectorLen);
    Type storeType = vectorTypeOrSelf(outputDataType, conversionVectorLen);
    Value trueOp = b.create<arith::ConstantIntOp>(loc, true, b.getI1Type());
    GemmFeatures features = op.getFeatures();
    bool needs64BitIdx =
        is4GBMemoryType(input.getType()) || is4GBMemoryType(output.getType());
    Value storeMemref = makePrivateGpuAlloc(b, loc, storeType);
    Value zeroIndex = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
    auto loopBody = [&loadType, &storeType, &conversionVectorLen, &trueOp,
                     &storeMemref, &zeroIndex, &features,
                     &needs64BitIdx](OpBuilder &b, Location loc,
                                     ValueRange collapsed, Value index) {
      Value loaded = b.create<GlobalLoadOp>(
          loc, loadType, collapsed[0], /*valid=*/trueOp, index, needs64BitIdx,
          /*canReadOffEnd=*/true);
      Value converted = createTypeConversionOp(b, loc, loaded, storeType);
      b.create<InBoundsStoreOp>(loc, converted, storeMemref, zeroIndex);
      b.create<GlobalStoreOp>(
          loc, storeMemref, collapsed[1], APInt(64, conversionVectorLen),
          features, StoreMethod::Set, /*sourceCoord=*/zeroIndex,
          /*valid=*/trueOp, index, needs64BitIdx, /*canWriteOffEnd=*/true);
    };
    LogicalResult res = createElementwiseLoop(b, loc, op, {input, output},
                                              conversionVectorLen, loopBody);
    if (failed(res))
      return failure();

    b.eraseOp(op);
    return success();
  }
};

/// Layout normalization.

/// Make the dimensions that are the values in `mapping` and exist within
/// `toLayout` be in the same relative order as the dimensions that the keys of
/// `mapping` have within `fromLayout`, where both layout are given by the
/// names of the attributes containing them.
///
/// To enable usage in rewrite patterns, returns failure() when no change is
/// made.
LogicalResult makeToLayoutLikeFromLayoutAlong(
    PatternRewriter &b, RockConvInterface op, StringRef fromLayoutAttrName,
    TypedValue<ShapedType> toArg, StringRef toLayoutAttrName,
    const llvm::StringMap<StringAttr> &mapping) {
  llvm::SmallVector<StringAttr> expectedOrder;
  auto fromLayout = op->getAttrOfType<ArrayAttr>(fromLayoutAttrName);
  auto toLayout = op->getAttrOfType<ArrayAttr>(toLayoutAttrName);
  for (StringRef fromName : fromLayout.getAsValueRange<StringAttr>()) {
    auto maybeCorresponding = mapping.find(fromName);
    if (maybeCorresponding != mapping.end())
      expectedOrder.push_back(maybeCorresponding->getValue());
  }

  llvm::SmallDenseMap<StringAttr, size_t> toLayoutIdxs;
  for (auto pair : llvm::enumerate(toLayout.getAsRange<StringAttr>()))
    toLayoutIdxs.insert({pair.value(), pair.index()});

  bool inOrder = true;
  size_t prevIndex = 0;
  for (StringAttr expected : expectedOrder) {
    size_t thisIndex = toLayoutIdxs.find(expected)->getSecond();
    if (thisIndex <
        prevIndex) { // the values are not in the relative expected order
      inOrder = false;
      break;
    }
    prevIndex = thisIndex;
  }
  if (inOrder)
    return failure();

  /// And now we have to actually do the thing
  // Is just an attribute to allow array builder
  SmallVector<Attribute> newToLayout;
  llvm::SmallDenseSet<StringAttr> permutedDimsSet;
  for (StringAttr toPermute : expectedOrder)
    permutedDimsSet.insert(toPermute);

  SmallVector<StringAttr>::const_iterator expectedOrderIter =
      expectedOrder.begin();
  for (StringAttr dim : toLayout.getAsRange<StringAttr>()) {
    if (permutedDimsSet.contains(dim)) {
      newToLayout.push_back(*expectedOrderIter);
      ++expectedOrderIter;
    } else {
      newToLayout.push_back(dim);
    }
  }

  SmallVector<StringRef> oldToLayoutRefs;
  llvm::copy(toLayout.getAsValueRange<StringAttr>(),
             std::back_inserter(oldToLayoutRefs));
  ArrayRef<int64_t> toShape = toArg.getType().getShape();

  BottomUpTMBuilder relayout(b, oldToLayoutRefs, toShape, op.getLoc());
  llvm::StringMap<uint32_t> newToLayoutIdxs;
  for (auto pair : llvm::enumerate(newToLayout)) {
    StringRef value = pair.value().cast<StringAttr>().getValue();
    newToLayoutIdxs.insert({value, pair.index()});
  }
  BottomUpTMTopDimsWrapper relayoutWrapped(relayout,
                                           std::move(newToLayoutIdxs));

  relayoutWrapped.passThrough(oldToLayoutRefs);
  TransformMapAttr relayoutAttr = relayout.get();

  Value transformed = b.create<TransformOp>(op.getLoc(), toArg, relayoutAttr);
  for (OpOperand &operand : op->getOpOperands())
    if (operand.get() == toArg)
      operand.set(transformed);

  op->setAttr(toLayoutAttrName, b.getArrayAttr(newToLayout));
  return success();
}

struct MatchLayoutsToInput final
    : public OpInterfaceRewritePattern<RockConvInterface> {
  using OpInterfaceRewritePattern<RockConvInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(RockConvInterface op,
                                PatternRewriter &b) const override {
    TypedValue<ShapedType> filter = op.getFilter(), output = op.getOutput();
    const llvm::StringMap<StringAttr> inputToFilter = {
        {"ci", b.getStringAttr("c")},
        {"0i", b.getStringAttr("0")},
        {"1i", b.getStringAttr("1")},
        {"hi", b.getStringAttr("y")},
        {"wi", b.getStringAttr("x")}};
    const llvm::StringMap<StringAttr> inputToOutput = {
        {"ni", b.getStringAttr("no")},
        {"0i", b.getStringAttr("0o")},
        {"1i", b.getStringAttr("1o")},
        {"hi", b.getStringAttr("ho")},
        {"wi", b.getStringAttr("wo")}};

    LogicalResult didRelayoutFilter = makeToLayoutLikeFromLayoutAlong(
        b, op, "input_layout", filter, "filter_layout", inputToFilter);
    LogicalResult didRelayoutOutput = makeToLayoutLikeFromLayoutAlong(
        b, op, "input_layout", output, "output_layout", inputToOutput);
    return success(didRelayoutFilter.succeeded() ||
                   didRelayoutOutput.succeeded());
  }
};

/// Lowerings for particular convolution algorithms (TODO, new file?)
LogicalResult backwardWeightAtomicAdd(ConvBwdWeightOp op, PatternRewriter &b) {
  Location loc = op.getLoc();

  Attribute tuningParams = op.getParamsAttr();
  if (!tuningParams) {
    return op.emitOpError("can't lower without tuning parameters\n");
  }

  if (!op.getKBlocks().has_value())
    return op.emitOpError("must have kBlocks set at lowering");
  int64_t gemmKBlocks = op.getKBlocks()->getZExtValue();

  ConvolutionContext ctx = populateConvContext(op);

  // Get shape of filter tensor.
  ShapedType filterType = op.getFilter().getType();
  auto filterShape = filterType.getShape();

  GemmFeatures features = op.getFeatures();
  bool isAccel = rock::isAccel(features);

  // Determine whether to use workspace.
  bool hasWorkspace =
      (filterType.getElementType() == b.getF16Type() && isAccel);
  if (hasWorkspace && !op.getWorkspace()) {
    return op.emitOpError(
        "workspace needed for f16 atomic add but none provided");
  }

  // The 1st kernel will conduct the actual backward weight convolution using
  // atomic adds.
  if (!isAccel)
    return op->emitOpError("atomic add kernel requires gemm acceleration");

  // Get shape of input tensor.
  ShapedType inputType = op.getInput().getType();
  ArrayRef<int64_t> inputShape = inputType.getShape();

  // Get shape of output tensor.
  ShapedType outputType = op.getOutput().getType();
  ArrayRef<int64_t> outputShape = outputType.getShape();

  // Obtain convolution parameters: padding / dilation / stride.
  auto pads = ctx.getPaddingVal();
  auto dilations = ctx.getDilationVal();
  auto strides = ctx.getStrideVal();
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
    addKBlockWrap.passThrough({"k", "c", "0", "1"});

    TransformMapAttr addKBlockTransformAttr = addKBlockTransform.get();
    Value filterTensorInUse =
        (hasWorkspace) ? op.getWorkspace() : op.getFilter();
    Value withKBlock = b.create<rock::TransformOp>(loc, filterTensorInUse,
                                                   addKBlockTransformAttr);

    // Create GEMM filter tensor
    // Here, we merge the KBlock dimension into the G dimension
    // keeping the kBlock dimension as the minor index
    // and send K to the M dimension and CYX to the N dimension as usual
    auto gemmTransform =
        BottomUpTMBuilder::above(addKBlockTransform, addKBlockTransformAttr);
    gemmTransform.merge("gemmG", 0, {"g", "kBlock"});
    gemmTransform.passThrough({"gemmM"}, {1}, {"k"});
    gemmTransform.merge("gemmN", 2, nonKDims);

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
        {{"ni", {"n0", "n1"}}, {"0i", {"0ipad"}}, {"1i", {"1ipad"}}});


//     llvm::StringMap<uint32_t> firstTransformOutDimsB = expandNamesInPlace(
//         inputNames,
//         {{"ni", {"n0", "n1"}}});
//     for (auto& x : firstTransformOutDimsB)
//       firstTransformOutDims.insert(&x);



    BottomUpTMBuilder firstTransform(b, inputNames, inputShape, loc);
    BottomUpTMTopDimsWrapper firstWrap(firstTransform,
                                       std::move(firstTransformOutDims));
    firstWrap.passThrough("gi");
    firstWrap.unmerge({"n0", "n1"}, "ni",
                      {gemmKBlocks, convDims.n / gemmKBlocks});
    firstWrap.passThrough("ci");
    firstWrap.pad({"0ipad", "1ipad"}, {"0i", "1i"}, pads);

    TransformMapAttr firstTransformAttr = firstTransform.get();
    Value firstTransformed =
        b.create<TransformOp>(loc, op.getInput(), firstTransformAttr);

    // The usual mapping of input space to dimensions such that filter elements
    // get multiplied by the right thing
    llvm::StringMap<uint32_t> embedOutDims = expandNamesInPlace(
        firstTransform, {{"0ipad", {"0", "0o"}}, {"1ipad", {"1", "1o"}}});
    auto embedTransform =
        BottomUpTMBuilder::above(firstTransform, firstTransformAttr);
    BottomUpTMTopDimsWrapper embedWrap(embedTransform, std::move(embedOutDims));
    embedWrap.passThrough({"gi", "n0", "n1", "ci"});
    embedWrap.embed({"0", "0o"}, {convDims.fil[0], convDims.out[0]}, "0ipad",
                    {dilations[0], strides[0]});
    embedWrap.embed({"1", "1o"}, {convDims.fil[1], convDims.out[1]}, "1ipad",
                    {dilations[1], strides[1]});

    TransformMapAttr embedTransformAttr = embedTransform.get();
    Value embedded =
        b.create<TransformOp>(loc, firstTransformed, embedTransformAttr);

    // Merge N1HoWO to gemmK and CYX to gemmN
    auto gemmInputTransform =
        BottomUpTMBuilder::above(embedTransform, embedTransformAttr);

    llvm::SmallVector<StringRef, 3> nonNHWDims = {"ci", "0", "1"};
    matchUnderlyingOrder(nonNHWDims, gemmInputTransform);
    llvm::SmallVector<StringRef, 3> nhwDims = {"n1", "0o", "1o"};
    matchUnderlyingOrder(nhwDims, gemmInputTransform);

    // In the gemmG dimension, unlike with gemmN, we don't have the same
    // traversal order concerns - a step in the G dimension always first visits
    // kBlock/N0 and then moves on to the next G
    gemmInputTransform.merge("gemmG", 0, {"gi", "n0"});
    gemmInputTransform.merge("gemmK", 1, nhwDims);
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
    firstWrap.passThrough({"ko", "0o", "1o"});

    TransformMapAttr firstTransformAttr = firstTransform.get();
    Value transformed =
        b.create<TransformOp>(loc, op.getOutput(), firstTransformAttr);

    // Map G and N0 to gemmG, N1HW to gemmK and K to gemmM
    auto gemmOutputTransform =
        BottomUpTMBuilder::above(firstTransform, firstTransformAttr);
    llvm::SmallVector<StringRef, 3> nhwDims = {"n1", "0o", "1o"};
    matchUnderlyingOrder(nhwDims, gemmOutputTransform);
    gemmOutputTransform.merge("gemmG", 0, {"go", "n0"});
    gemmOutputTransform.merge("gemmK", 1, nhwDims);
    gemmOutputTransform.passThrough({"gemmM"}, {2}, {"ko"});

    TransformMapAttr gemmOutputTransformAttr = gemmOutputTransform.get();
    gemmOutput =
        b.create<TransformOp>(loc, transformed, gemmOutputTransformAttr);
  }

  // This kernel is not run when there is padding on the GEMM
  auto storeMethod = b.getAttr<StoreMethodAttr>(StoreMethod::AtomicAdd);

  b.create<GemmOp>(
      loc, getResultType(op, gemmFilter), gemmOutput, gemmInput, gemmFilter,
      /*aTransposed=*/b.getUnitAttr(), /*bTransposed=*/nullptr,
      /*cTransposed=*/nullptr, op.getArchAttr(), op.getNumCUAttr(),
      op.getFeaturesAttr(), storeMethod, op.getDerivedBlockSizeAttr(),
      op.getGridSizeAttr(), op.getParamsAttr());

  // Finally, erase the original Conv op.
  b.eraseOp(op);

  return success();
}

LogicalResult backwardData(ConvBwdDataOp op, PatternRewriter &b) {
  Location loc = op.getLoc();
  IntegerAttr kernelIdAttr = op.getKernelIdAttr();

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
  auto pads = ctx.getPaddingVal();
  auto dilations = ctx.getDilationVal();
  auto strides = ctx.getStrideVal();
  ConvolutionDims convDims = ctx.getConvDims();
  SmallVector<StringRef, 5> filterNames, inputNames, outputNames;
  if (failed(getConvDimNames(op, filterNames, inputNames, outputNames))) {
    return failure();
  }

  SmallVector<int64_t, 5> gcdStrideDilations;
  assert(strides.size() == dilations.size());
  for (const auto &[stride, dilation] : zip(strides, dilations)) {
    gcdStrideDilations.push_back(math_util::gcd(stride, dilation));
  }

  SmallVector<int64_t, 5> filTilda;
  for (const auto &[stride, gcdSD] : zip(strides, gcdStrideDilations)) {
    filTilda.push_back(stride / gcdSD);
  }

  SmallVector<int64_t, 5> filDots;
  for (const auto &[fil, tilda] : zip(convDims.fil, filTilda)) {
    filDots.push_back(math_util::integer_divide_ceil(fil, tilda));
  }

  SmallVector<int64_t, 5> outTilda;
  for (const auto &[out, dilation, fil, stride] :
       zip(convDims.out, dilations, convDims.fil, strides)) {
    outTilda.push_back(
        out + math_util::integer_divide_ceil(dilation * (fil - 1), stride));
  }

  SmallVector<int64_t, 5> iTildaLeft;
  SmallVector<int64_t, 5> iTildaRight;
  for (const auto &[padindex, dilation, tilda, stride] :
       enumerate(dilations, filTilda, strides)) {
    iTildaLeft.push_back(math_util::integer_divide_floor(
        std::max((int64_t)0, pads[2 * padindex] - dilation * (tilda - 1)),
        stride));
  }
  for (const auto &[padindex, out, in, stride] :
       enumerate(outTilda, convDims.in, strides)) {
    iTildaRight.push_back(std::min(
        out,
        math_util::integer_divide_ceil(pads[2 * padindex] + in - 1, stride) +
            1));
  }

  int64_t kernelId = kernelIdAttr.getInt();
  int64_t iYTilda = kernelId / filTilda[1];
  int64_t iXTilda = kernelId % filTilda[1];
  int64_t yDotSlice =
      math_util::integer_divide_ceil(convDims.fil[0] - iYTilda, filTilda[0]);
  int64_t xDotSlice =
      math_util::integer_divide_ceil(convDims.fil[1] - iXTilda, filTilda[0]);

  // backward data only, it's igemm v4r1 algo
  // c is input chaneels , k is output channels
  // n is batch , yDotSlice,xDotSlice computed in above

  Value gemmFilter, gemmInput, gemmOutput;
  // Transform filter tensor.
  {
    // Embed y/x into {y/x}dot and {y/x}tilda (Why the
    // particular embed coefficients is in a presentation somewhere)
    llvm::StringMap<uint32_t> embedDims = expandNamesInPlace(
        filterNames, {{"0", {"0dot", "0tilda"}}, {"1", {"1dot", "1tilda"}}});
    BottomUpTMBuilder embedTransform(b, filterNames, filterShape, loc);
    BottomUpTMTopDimsWrapper embedWrap(embedTransform, std::move(embedDims));

    embedWrap.passThrough({"g", "k", "c"});
    embedWrap.embed({"0dot", "0tilda"}, {filDots[0], filTilda[0]}, "0",
                    {strides[0] / gcdStrideDilations[0], 1});
    embedWrap.embed({"1dot", "1tilda"}, {filDots[1], filTilda[1]}, "1",
                    {strides[1] / gcdStrideDilations[1], 1});

    TransformMapAttr embedTransformAttr = embedTransform.get();
    Value embeddedFilter =
        b.create<TransformOp>(loc, op.getFilter(), embedTransformAttr);

    // Take slices in the ydot, ytilda, xdot, and xtilda dimensions
    // to reflect which kernel we're performing
    auto sliceTransform =
        BottomUpTMBuilder::above(embedTransform, embedTransformAttr);
    sliceTransform.passThrough({"g", "k", "c"});
    sliceTransform.slice({"0dotslice", "1dotslice"}, {"0dot", "1dot"}, {0, 0},
                         {yDotSlice, xDotSlice});
    sliceTransform.slice({"0tildaslice", "1tildaslice"}, {"0tilda", "1tilda"},
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
    gemmFilterTransform.merge("gemmK", 1, {"k", "0dotslice", "1dotslice"});
    gemmFilterTransform.merge("gemmM", 2, {"c", "0tildaslice", "1tildaslice"});

    TransformMapAttr gemmFilterTransformAttr = gemmFilterTransform.get();
    gemmFilter =
        b.create<TransformOp>(loc, slicedFilter, gemmFilterTransformAttr);
  }

  // Transform input tensor
  {
    BottomUpTMBuilder padInputTransform(b, inputNames, inputShape, loc);
    padInputTransform.passThrough({"gi", "ni", "ci"});
    padInputTransform.pad({"0ipad", "1ipad"},
                          {padInputTransform.startIndex("0i"),
                           padInputTransform.startIndex("1i")},
                          {"0i", "1i"}, pads);

    TransformMapAttr padTransformAttr = padInputTransform.get();
    Value paddedInput =
        b.create<TransformOp>(loc, op.getInput(), padTransformAttr);

    // Split 0ipad, 1ipad into ytilda, htilda, xtilda, wtilda
    llvm::StringMap<uint32_t> embedDims = expandNamesInPlace(
        padInputTransform,
        {{"0ipad", {"0ftilda", "0itilda"}}, {"1ipad", {"1ftilda", "1itilda"}}});
    auto tildaEmbedTransform =
        BottomUpTMBuilder::above(padInputTransform, padTransformAttr);
    BottomUpTMTopDimsWrapper tildaEmbedWrap(tildaEmbedTransform,
                                            std::move(embedDims));
    tildaEmbedWrap.passThrough({"gi", "ni", "ci"});
    tildaEmbedWrap.embed({"0ftilda", "0itilda"}, {filTilda[0], outTilda[0]},
                         "0ipad", {dilations[0], strides[0]});
    tildaEmbedWrap.embed({"1ftilda", "1itilda"}, {filTilda[1], outTilda[1]},
                         "1ipad", {dilations[1], strides[1]});

    TransformMapAttr tildaEmbedTransformAttr = tildaEmbedTransform.get();
    Value tildaEmbedded =
        b.create<TransformOp>(loc, paddedInput, tildaEmbedTransformAttr);

    // Slice all the tilda dimensions: ytilda and xtilda get slices of length
    // 1 while htilda and wtilda have slice indices computed above
    auto sliceTransform =
        BottomUpTMBuilder::above(tildaEmbedTransform, tildaEmbedTransformAttr);
    sliceTransform.passThrough({"gi", "ni", "ci"});
    sliceTransform.slice({"0slice", "1slice"}, {"0ftilda", "1ftilda"},
                         {iYTilda, iXTilda}, {iYTilda + 1, iXTilda + 1});
    sliceTransform.slice({"0islice", "1islice"}, {"0itilda", "1itilda"}, iTildaLeft,
                         iTildaRight);

    TransformMapAttr sliceTransformAttr = sliceTransform.get();
    Value sliced =
        b.create<TransformOp>(loc, tildaEmbedded, sliceTransformAttr);

    // C plus the length 1 slices (yslice and xslice) become the gemmM
    // dimension G, N, and the h and w slices become gemmN
    auto gemmTransform =
        BottomUpTMBuilder::above(sliceTransform, sliceTransformAttr);
    gemmTransform.passThrough({"gemmG"}, {0}, {"gi"});
    gemmTransform.merge("gemmM", 1, {"ci", "0slice", "1slice"});
    gemmTransform.merge("gemmN", 2, {"ni", "0islice", "1islice"});

    TransformMapAttr gemmTransformAttr = gemmTransform.get();
    gemmInput = b.create<TransformOp>(loc, sliced, gemmTransformAttr);
  }

  // Transform output tensor
  {
    // Embed 0o to 0dot and 0tilda and 1o to 1dot and 1tilda
    llvm::StringMap<uint32_t> embedDims = expandNamesInPlace(
        outputNames, {{"0o", {"0dot", "0tilda"}}, {"1o", {"1dot", "1tilda"}}});
    BottomUpTMBuilder embedTransform(b, outputNames, outputShape, loc);
    BottomUpTMTopDimsWrapper embedWrap(embedTransform, std::move(embedDims));
    embedWrap.passThrough({"go", "no", "ko"});
    embedWrap.embed({"0dot", "0tilda"}, {filDots[0], outTilda[0]}, "0o",
                    {(-dilations[0]) / gcdStrideDilations[0], 1});
    embedWrap.embed({"1dot", "1tilda"}, {filDots[1], outTilda[1]}, "1o",
                    {(-dilations[1]) / gcdStrideDilations[1], 1});

    TransformMapAttr embedTransformAttr = embedTransform.get();
    Value embedded =
        b.create<TransformOp>(loc, op.getOutput(), embedTransformAttr);

    // Take the same slices in ydot, xdot, 0tilda, and 1tilda as were taken in
    // the filter and input
    auto sliceTransform =
        BottomUpTMBuilder::above(embedTransform, embedTransformAttr);
    sliceTransform.passThrough({"go", "no", "ko"});
    sliceTransform.slice({"0slice", "1slice"}, {"0dot", "1dot"}, {0, 0},
                         {yDotSlice, xDotSlice});
    sliceTransform.slice({"0islice", "1islice"}, {"0tilda", "1tilda"}, iTildaLeft,
                         iTildaRight);

    TransformMapAttr sliceTransformAttr = sliceTransform.get();
    Value sliced = b.create<TransformOp>(loc, embedded, sliceTransformAttr);

    // Merge k, yslice, and xslice to gemmK and n, hslice, and wslice to gemmN
    auto gemmOutputTransform =
        BottomUpTMBuilder::above(sliceTransform, sliceTransformAttr);
    gemmOutputTransform.passThrough({"gemmG"}, {0}, {"go"});
    gemmOutputTransform.merge("gemmK", 1, {"ko", "0slice", "1slice"});
    gemmOutputTransform.merge("gemmN", 2, {"no", "0islice", "1islice"});

    TransformMapAttr gemmOutputTransformAttr = gemmOutputTransform.get();
    gemmOutput = b.create<TransformOp>(loc, sliced, gemmOutputTransformAttr);
  }

  // Emit rock.gemm op.
  auto storeMethod = b.getAttr<StoreMethodAttr>(StoreMethod::Set);
  auto gemm = b.create<GemmOp>(
      loc, getResultType(op, gemmInput), gemmFilter, gemmOutput, gemmInput,
      /*aTransposed=*/b.getUnitAttr(), /*bTransposed=*/nullptr,
      /*cTransposed=*/nullptr, op.getArchAttr(), op.getNumCUAttr(),
      op.getFeaturesAttr(), storeMethod, op.getDerivedBlockSizeAttr(),
      op.getGridSizeAttr(), op.getParamsAttr());
  // Bounced along for debugging purposes, not used below
  gemm->setAttr("kernelId", kernelIdAttr);

  // Finally, erase the original Conv op.
  b.eraseOp(op);

  return success();
}

template <typename T>
struct ConvRewritePattern : public OpRewritePattern<T> {
  const static ArgumentFields fields;
  const static ConvOpType convOpType;
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op, PatternRewriter &b) const override {
    GemmFeatures features = op.getFeatures();

    Type dataType = op.getInput().getType().getElementType();
    if (ConvOpType::BwdData == convOpType) {
      return backwardData(cast<ConvBwdDataOp>(op), b);
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

    // Obtain convolution parameters: padding / dilation / stride.
    auto dilations = ctx.getDilationVal();
    auto strides = ctx.getStrideVal();
    ConvolutionDims convDims = ctx.getConvDims();

    llvm::SmallVector<StringRef, 5> filterNames, inputNames, outputNames;
    if (failed(getConvDimNames(op, filterNames, inputNames, outputNames))) {
      return failure();
    }

    auto tuningParams = op.getParamsAttr();
    GemmSize gemmSize = op.getGemmSize();
    std::optional<GemmSize> maybeGemmExtraPad;

    if (tuningParams) {
      maybeGemmExtraPad = requiredPadding(tuningParams, gemmSize);
    } else {
      // We don't know if this'll be a padding kernel, so we can't promise an
      // unfold or rely on atomic add, and so set the extraPad to a nonsense but
      // existing value.
      maybeGemmExtraPad = GemmSize{-1, -1, -1, -1};
    }

    if (ConvOpType::BwdWeight == convOpType &&
        isWrWAtomicKernel(features, dataType, maybeGemmExtraPad.has_value())) {
      return backwardWeightAtomicAdd(cast<ConvBwdWeightOp>(op), b);
    }

    // Transform filter tensor.

    // set layout attribute.
    // Weight tensor transformation for ConvOp
    // - PassThrough G dimension to dimension 0, name it gemmG.
    // - Merge non-K dimensions to dimension 1, name it as gemmK.
    //   Optimization: If non-K dimensions are consequetive, apply unfold.
    // - PassThrough K dimension to dimension 2, name it as gemmM.
    //
    // Weight tensor transformation for ConvBwdWeightOp
    // - PassThrough G dimension to dimension 0, name it gemmG
    // - PassThrough K dimension to dimension 1, name it as gemmM.
    // - Merge non-K dimensions to dimension 2, name it as gemmN.
    SmallVector<StringRef, 5> filterNonKDims;
    for (StringRef name : filterNames)
      if (name != "g" && name != "k")
        filterNonKDims.push_back(name);

    BottomUpTMBuilder filterTransform(b, filterNames, filterShape, loc);
    filterTransform.passThrough({"gemmG"}, {0}, {"g"});
    switch (convOpType) {
    case ConvOpType::Fwd:
      filterTransform.merge("gemmK", 1, filterNonKDims);
      filterTransform.passThrough({"gemmM"}, {2}, {"k"});
      break;
    case ConvOpType::BwdWeight:
      filterTransform.passThrough({"gemmM"}, {1}, {"k"});
      filterTransform.merge("gemmN", 2, filterNonKDims);
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
    // 0ipad and 1ipad
    BottomUpTMBuilder padInputTransform(b, inputNames, inputShape, loc);
    padInputTransform.passThrough("ni");
    padInputTransform.passThrough("gi");
    padInputTransform.passThrough("ci");

    llvm::SmallVector<uint32_t, 2> padOutDims = {
        padInputTransform.startIndex("0i"), padInputTransform.startIndex("1i")};
    padInputTransform.pad({"0ipad", "1ipad"}, padOutDims, {"0i", "1i"},
                          ctx.getPaddingVal());

    TransformMapAttr padInputTransformAttr = padInputTransform.get();

    Value paddedInput =
        b.create<TransformOp>(loc, op.getInput(), padInputTransformAttr);

    // Input tensor step 2 : embedded input.
    // Embedded input tensor transformation:
    // - PassThrough gi, ni, and ci
    // - Embed 0ipad to y and ho with size filter y by output h and
    //   coefficients dilations[0] and strides[0]
    // - Embed 1ipad to x and wo with size filter x by output h and
    //   coefficients dilations[1] and strides[1]

    llvm::StringMap<uint32_t> embeddedInputDims = expandNamesInPlace(
        padInputTransform, {{"0ipad", {"0", "0o"}}, {"1ipad", {"1", "1o"}}});

    BottomUpTMBuilder embedInputTransform =
        BottomUpTMBuilder::above(padInputTransform, padInputTransformAttr);
    BottomUpTMTopDimsWrapper embedInputWrap(embedInputTransform,
                                            std::move(embeddedInputDims));
    embedInputWrap.passThrough({"ni", "gi", "ci"});
    embedInputWrap.embed({"0", "0o"}, {convDims.fil[0], convDims.out[0]},
                         "0ipad", {dilations[0], strides[0]});
    embedInputWrap.embed({"1", "1o"}, {convDims.fil[1], convDims.out[1]},
                         "1ipad", {dilations[1], strides[1]});

    TransformMapAttr embedInputTransformAttr = embedInputTransform.get();
    Value embeddedInput =
        b.create<TransformOp>(loc, paddedInput, embedInputTransformAttr);

    // Input tensor step 3: GEMM'd input
    //
    // - PassThrough gi to dimension 0 and name it gemmG, then
    // For ConvOp:
    // - Merge ci, y, x dimensions to dimension 1, name it as gemmK.
    // - Merge ni, ho, wo dimensions to dimension 2, name it as gemmN.
    //
    // For ConvBwdWeightOp:
    // - Part 1: Merge ni, ho, wo dimensions to dimension 1, name it as gemmK.
    // - Part 2: Merge ci, y, x dimensions to dimension 2, name it as gemmN.

    auto gemmInputTransform =
        BottomUpTMBuilder::above(embedInputTransform, embedInputTransformAttr);
    gemmInputTransform.passThrough({"gemmG"}, {0}, {"gi"});

    llvm::SmallVector<StringRef, 3> nonNHWDims = {"ci", "0", "1"};
    matchUnderlyingOrder(nonNHWDims, gemmInputTransform);
    llvm::SmallVector<StringRef, 3> nhwDims = {"ni", "0o", "1o"};
    matchUnderlyingOrder(nhwDims, gemmInputTransform);

    llvm::SmallVector<StringRef, 3> mergeToK, mergeToN;
    switch (convOpType) {
    case ConvOpType::Fwd:
      mergeToK = std::move(nonNHWDims);
      mergeToN = std::move(nhwDims);
      break;
    case ConvOpType::BwdWeight:
      mergeToK = std::move(nhwDims);
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
    // Output tensor transformation for ConvOp:
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
                     op.getNumCUAttr(), op.getFeaturesAttr(), storeMethod,
                     op.getDerivedBlockSizeAttr(), op.getGridSizeAttr(),
                     tuningParams);

    // Finally, erase the original Conv op.
    b.eraseOp(op);

    return success();
  }
};

template <>
const ArgumentFields ConvRewritePattern<ConvOp>::fields = {
    {0, 1, 2},
    {"KM", "KN", "MN"},
};
template <>
const ConvOpType ConvRewritePattern<ConvOp>::convOpType = ConvOpType::Fwd;

template <>
const ArgumentFields ConvRewritePattern<ConvBwdDataOp>::fields = {
    {0, 2, 1},
    {"KM", "MN", "KN"},
};

template <>
const ConvOpType ConvRewritePattern<ConvBwdDataOp>::convOpType =
    ConvOpType::BwdData;

template <>
const ArgumentFields ConvRewritePattern<ConvBwdWeightOp>::fields = {
    {2, 1, 0},
    {"MN", "KN", "KM"},
};

template <>
const ConvOpType ConvRewritePattern<ConvBwdWeightOp>::convOpType =
    ConvOpType::BwdWeight;

// Explicitly instantiate the template to operation type
template struct ConvRewritePattern<ConvOp>;
template struct ConvRewritePattern<ConvBwdDataOp>;
template struct ConvRewritePattern<ConvBwdWeightOp>;

void RockConvToGemmPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet preConvToGemmPatterns(ctx);
  preConvToGemmPatterns.add<MatchLayoutsToInput>(ctx);

  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(preConvToGemmPatterns)))) {
    signalPassFailure();
    return;
  }

  ConversionTarget target(*ctx);

  target.addIllegalOp<rock::ConvOp, rock::ConvBwdDataOp, rock::ConvBwdWeightOp,
                      rock::InitKernelOp, rock::ConvertingCopyKernelOp>();
  target.addLegalOp<rock::TransformOp, rock::GemmOp, rock::WorkgroupIdOp,
                    rock::WorkitemIdOp, rock::GlobalLoadOp, rock::GlobalStoreOp,
                    rock::GpuAllocOp, rock::InBoundsStoreOp>();
  // Below are required legalize for the lowering of ConvBwdWeightOp
  target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                         scf::SCFDialect>();

  RewritePatternSet patterns(ctx);
  patterns
      .add<ConvRewritePattern<ConvOp>, ConvRewritePattern<ConvBwdDataOp>,
           ConvRewritePattern<ConvBwdWeightOp>, ZeroInitKernelRewritePattern,
           ConvertingCopyKernelRewritePattern>(ctx);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
} // end anonymous namespace
