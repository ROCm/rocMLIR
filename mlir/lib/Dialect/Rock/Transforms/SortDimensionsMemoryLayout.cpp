//===- SortDimensionsMemoryLayout.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters sort dimensions using the memory layout (lower stride first).
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/BufferDependencyAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKSORTDIMENSIONSMEMORYLAYOUTPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-sort-dimensions-memory-layout"

using namespace mlir;

namespace {
struct RockSortDimensionsMemoryLayoutPass
    : public rock::impl::RockSortDimensionsMemoryLayoutPassBase<
          RockSortDimensionsMemoryLayoutPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

template <typename Container>
static FailureOr<Container> reorderArrayAttr(Container inputArray,
                                             ArrayRef<uint32_t> permutation) {
  if (inputArray.size() != permutation.size())
    return failure();

  // Create a vector to store the reordered elements
  Container reorderedElements;
  reorderedElements.reserve(inputArray.size());

  // Perform the reordering
  for (unsigned idx : permutation) {
    if (idx >= inputArray.size())
      return failure();

    reorderedElements.push_back(inputArray[idx]);
  }

  return reorderedElements;
}
//

//  traces input arguments of the GEMM operation back to blockArguments. It
//  records sequence of rock.transforms between gemm argument to blockArgument
//  if there is any. It is possible that single gemm arg is mapped to multiple
//  blockArguments. BlockArguments are recorded in `blockArgs` and series of
//  rock.TransformAttr sequences for each `blockArgs` is recorded in
//  transformAttrsMap.
static LogicalResult traceGemmInputToBlockArgs(
    Value inputArg, PatternRewriter &b,
    llvm::DenseMap<Value, SmallVector<Attribute>> &transformAttrsMap,
    llvm::SmallSetVector<Value, 2> &blockArgs,
    const BufferDependencyAnalysis &deps) {
  Value source;
  ArrayAttr transforms;
  // below call to `rock.untransform` is concatenating existing transform
  // sequence on `inputArg` with rock.transform sequence found by tracing upto
  // source from `inputArg` as staring point.
  // For example,
  // SeqExisting -> inputArgs --> Seq --> source
  // transforms == SeqExisting + Seq
  // transformAttrsMap[inputArg] = SeqExisting
  // transformAttrsMap[Source] = SeqExisting + Seq

  if (!isa<BlockArgument>(inputArg) &&
      isa<memref::AllocOp>(inputArg.getDefiningOp())) {
    source = inputArg;
    transforms = b.getArrayAttr(SmallVector<Attribute>{});
  } else {
    std::tie(source, transforms, std::ignore) =
        rock::untransform(b, inputArg, transformAttrsMap[inputArg]);
  }

  // insert transform sequence on source into the map if it doesn't already
  // exists. if it does then we've found a loop or case where multiple operators
  // are writing to same `memref.alloc`
  if (!transformAttrsMap
           .insert({source, SmallVector<Attribute>{transforms.begin(),
                                                   transforms.end()}})
           .second) {
    return failure();
  }
  if (isa<BlockArgument>(source)) {
    blockArgs.insert(source);
    return success();
  }
  FailureOr<memref::AllocOp> allocOp = mlir::rock::findMemrefAlloc(source);
  if (failed(allocOp)) {
    return failure();
  }
  std::optional<llvm::SmallVector<OpOperand *>> allocOpWriters =
      deps.getWriters(allocOp.value());
  if (!allocOpWriters.has_value()) {
    return failure();
  }
  bool hasSuccess = false;
  for (OpOperand *allocWriteOperand : allocOpWriters.value()) {
    auto writerOp =
        dyn_cast<MemoryEffectOpInterface>(allocWriteOperand->getOwner());
    if (!writerOp)
      continue;
    SmallVector<MemoryEffects::EffectInstance> effects;
    writerOp.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      OpOperand *writerOpOperand = effect.getEffectValue<OpOperand *>();
      // test that same buffer is not being read and written to
      if (writerOpOperand && isa<MemoryEffects::Read>(effect.getEffect()) &&
          writerOpOperand != allocWriteOperand) {
        Value writerOpOperandValue = writerOpOperand->get();
        // Add existing transform sequences on `writerOpOperandValue` to
        // continue concatenating in recursive calls.
        transformAttrsMap[writerOpOperandValue] = transformAttrsMap.at(source);
        if (succeeded(traceGemmInputToBlockArgs(
                writerOpOperandValue, b, transformAttrsMap, blockArgs, deps))) {
          hasSuccess = true;
        }
      }
    }
  }
  // return success if it has found trace to any blockArg
  return success(hasSuccess);
}

template <typename Container>
static FailureOr<std::tuple<Value, Container, SmallVector<uint32_t>>>
sortByMemoryLayout(Value tensor, const Container &layout, PatternRewriter &b) {
  // trace input tensor to blockArgument first and do necessary error checking
  llvm::DenseMap<Value, SmallVector<Attribute>> transformAttrsMap;
  llvm::SmallSetVector<Value, 2> blockArgs;
  BufferDependencyAnalysis deps(tensor.getParentBlock()->getParentOp());
  if (failed(traceGemmInputToBlockArgs(tensor, b, transformAttrsMap, blockArgs,
                                       deps))) {
    return std::make_tuple(tensor, layout, SmallVector<uint32_t>{});
  }
  assert(!blockArgs.empty());
  SmallVector<Attribute> transformsList;
  for (const auto blockArg : blockArgs) {
    // make sure all the blockArgs have been mapped to some transform sequence
    // or empty transform sequence
    if (!transformAttrsMap.contains(blockArg)) {
      return std::make_tuple(tensor, layout, SmallVector<uint32_t>{});
    }
    if (transformsList.empty()) {
      transformsList = transformAttrsMap[blockArg];
    } else if (transformsList != transformAttrsMap[blockArg]) {
      // Currently we do not handle case where some block arg goes through
      // different sequence of transforms. All blockArgs must have same
      // transforms for now.
      return std::make_tuple(tensor, layout, SmallVector<uint32_t>{});
    }
  }
  if (transformsList.empty()) {
    return std::make_tuple(tensor, layout, SmallVector<uint32_t>{});
  }
  ArrayAttr transforms = b.getArrayAttr(transformsList);
  rock::TransformMapAttr firstCoordTransform =
      cast<rock::TransformMapAttr>(transformsList[0]);
  int64_t upperRank = firstCoordTransform.getUpperBounds().size();
  SmallVector<uint32_t> strides(upperRank);
  for (int64_t idx = 0; idx < upperRank; idx++) {
    FailureOr<llvm::SmallDenseMap<int64_t, SmallVector<rock::SubDimInfo>>>
        maybeLowerSubDims = rock::getLowerSubDimensions(b, transforms, idx);
    if (failed(maybeLowerSubDims)) {
      return failure();
    }

    auto lowerSubDims = maybeLowerSubDims.value();
    // if it's empty, it's a unit dimension
    uint32_t minStride =
        lowerSubDims.empty() ? 1 : std::numeric_limits<uint32_t>::max();

    for (auto [dim, subDimInfos] : lowerSubDims) {
      LLVM_DEBUG(llvm::dbgs() << "dim=" << dim << ":");
      LLVM_DEBUG(llvm::interleaveComma(subDimInfos, llvm::dbgs()));
      LLVM_DEBUG(llvm::dbgs() << "\n");
      for (auto subDim : subDimInfos)
        minStride = std::min(minStride, static_cast<uint32_t>(subDim.stride));
    }
    strides[idx] = minStride;
  }

  LLVM_DEBUG(llvm::dbgs() << "strides=");
  LLVM_DEBUG(llvm::interleaveComma(strides, llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  // get sorted indices
  SmallVector<uint32_t> startIndices(upperRank);
  std::iota(startIndices.begin(), startIndices.end(), 0);
  llvm::sort(startIndices.begin(), startIndices.end(),
             [&strides](uint32_t i1, uint32_t i2) {
               return strides[i1] > strides[i2];
             });

  LLVM_DEBUG(llvm::dbgs() << "startIndices=");
  LLVM_DEBUG(llvm::interleaveComma(startIndices, llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  Container layoutVec(layout.begin(), layout.end());
  auto newLayout = reorderArrayAttr(layoutVec, startIndices);
  LLVM_DEBUG(llvm::dbgs() << "layout=");
  LLVM_DEBUG(llvm::interleaveComma(layout, llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  if (failed(newLayout))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "newLayout=");
  LLVM_DEBUG(llvm::interleaveComma(newLayout.value(), llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  SmallVector<uint32_t> endIndices(upperRank);
  std::iota(endIndices.begin(), endIndices.end(), 0);
  LLVM_DEBUG(llvm::dbgs() << "endIndices=");
  LLVM_DEBUG(llvm::interleaveComma(endIndices, llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  // nothing to do, same ordering
  if (endIndices == startIndices)
    return std::make_tuple(tensor, layout, strides);

  rock::BottomUpTMBuilder sortDims(b, firstCoordTransform.getUpperBounds(),
                                   tensor.getLoc());
  sortDims.passThrough(endIndices, startIndices);

  SmallVector<Attribute> transformAttrs{sortDims.get()};
  return std::make_tuple(
      rock::transform(b, tensor, b.getArrayAttr(transformAttrs)),
      newLayout.value(), strides);
}

static std::tuple<Value, UnitAttr, SmallVector<StringRef>>
reorderBatch(Value tensor, const SmallVector<StringRef> &layout,
             StringRef expectedLastNonTransposed, PatternRewriter &b) {
  // if batch is not first, we need to transpose it
  Value newTensor = tensor;
  SmallVector<StringRef> newLayout;
  newLayout.reserve(layout.size());
  if (layout.size() == 3 && layout[0] != "G") {
    ArrayAttr transforms;
    std::tie(std::ignore, transforms, std::ignore) =
        rock::untransform(b, tensor);
    rock::TransformMapAttr firstCoordTransform =
        cast<rock::TransformMapAttr>(transforms[0]);
    uint32_t batchPos = (layout[2] == "G") ? 2 : 1;
    uint32_t nonBatchFastPos = (batchPos == 2) ? 1 : 2;
    SmallVector<uint32_t> startIndices({batchPos, 0, nonBatchFastPos});
    SmallVector<uint32_t> endIndices{0, 1, 2};

    // update layout
    newLayout.push_back("G");
    newLayout.push_back(layout[0]); // slowest
    newLayout.push_back(layout[nonBatchFastPos]);

    rock::BottomUpTMBuilder reorderBatchDim(
        b, firstCoordTransform.getUpperBounds(), tensor.getLoc());
    reorderBatchDim.passThrough(endIndices, startIndices);

    SmallVector<Attribute> transformAttrs{reorderBatchDim.get()};
    newTensor = rock::transform(b, tensor, b.getArrayAttr(transformAttrs));
  } else {
    newLayout.append(layout.begin(), layout.end());
  }

  LLVM_DEBUG(llvm::dbgs() << "finalLayout=");
  LLVM_DEBUG(llvm::interleaveComma(newLayout, llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  // Return if it's transposed
  UnitAttr transposed =
      (newLayout[layout.size() - 1] == expectedLastNonTransposed)
          ? nullptr
          : b.getUnitAttr();

  return std::make_tuple(newTensor, transposed, newLayout);
}

template <typename ContainerTy, typename ElementTy>
static std::optional<size_t> findIndex(const ContainerTy &container,
                                       const ElementTy &element) {
  const auto *it = llvm::find(container, element);
  if (it == container.end())
    return std::nullopt;
  return std::distance(container.begin(), it);
}

template <typename OpT>
static SmallVector<Operation *> getOperations(func::FuncOp &func) {
  SmallVector<Operation *, 4> ops;
  func.walk([&ops](OpT operation) { ops.push_back(operation); });

  return ops;
}

// Result of tensor layout processing
struct TensorLayoutResult {
  Value tensor;
  UnitAttr transposed;
  SmallVector<StringRef> initialLayout;
  SmallVector<StringRef> finalLayout;

  // Convenience method to check if processing succeeded
  explicit operator bool() const { return tensor != nullptr; }
};

// Helper function to process a tensor with sort and batch reorder operations.
static TensorLayoutResult processTensorLayout(PatternRewriter &b, Value tensor,
                                              ArrayRef<StringRef> layoutDims,
                                              bool isTransposed, int tensorRank,
                                              const std::string &debugName) {
  // Setup layout based on transpose attribute
  SmallVector<StringRef> layout(layoutDims.begin(), layoutDims.end());
  if (isTransposed && layout.size() == 3)
    layout = {layout[0], layout[2], layout[1]};
  if (tensorRank == 2 && layout.size() == 3)
    layout = {layout[1], layout[2]};

  SmallVector<StringRef> initialLayout = layout;

  // Sort by memory layout
  auto maybeSorted = sortByMemoryLayout(tensor, layout, b);
  if (failed(maybeSorted))
    return {Value{}, UnitAttr{}, initialLayout, SmallVector<StringRef>{}};

  auto sorted = maybeSorted.value();
  LLVM_DEBUG(llvm::dbgs() << debugName << "=" << std::get<0>(sorted) << "\n");

  // Reorder batch
  auto batchReorder = reorderBatch(std::get<0>(sorted), std::get<1>(sorted),
                                   layoutDims.back(), b);

  Value newTensor = std::get<0>(batchReorder);
  UnitAttr transposed = std::get<1>(batchReorder);
  auto finalLayout = std::get<2>(batchReorder);

  LLVM_DEBUG(llvm::dbgs() << "new" << debugName << "=" << newTensor << "\n");
  LLVM_DEBUG(llvm::dbgs() << "transposed" << debugName << "=" << transposed
                          << "\n");

  return {newTensor, transposed, initialLayout, finalLayout};
}

static FailureOr<std::tuple<Value, Value, Value, UnitAttr, UnitAttr, UnitAttr>>
commonGemmGemm(rock::RockGemmGemmWrapperInterface op, Value q, Value k, Value v,
               PatternRewriter &b) {
  // Process Q tensor
  auto resultQ =
      processTensorLayout(b, q, {"G", "M", "K"}, op.getTransposedA(),
                          cast<ShapedType>(q.getType()).getRank(), "TensorQ");

  if (!resultQ)
    return op.emitOpError("sortByMemoryLayout failed for TensorQ");

  // Process K tensor
  auto resultK =
      processTensorLayout(b, k, {"G", "K", "N"}, op.getTransposedB(),
                          cast<ShapedType>(k.getType()).getRank(), "TensorK");

  if (!resultK)
    return op.emitOpError("sortByMemoryLayout failed for TensorK");

  // Process V tensor
  auto resultV =
      processTensorLayout(b, v, {"G", "K", "N"}, op.getTransposedC(),
                          cast<ShapedType>(v.getType()).getRank(), "TensorV");

  if (!resultV)
    return op.emitOpError("sortByMemoryLayout failed for TensorV");

  // no need to create transforms if it's the same tensor
  if (resultQ.finalLayout == resultQ.initialLayout &&
      resultK.finalLayout == resultK.initialLayout &&
      resultV.finalLayout == resultV.initialLayout)
    return failure();

  return std::make_tuple(resultQ.tensor, resultK.tensor, resultV.tensor,
                         resultQ.transposed, resultK.transposed,
                         resultV.transposed);
}

template <typename OpT>
static FailureOr<std::tuple<Value, Value, ArrayAttr, ArrayAttr, bool>>
commonConv(OpT op, PatternRewriter &b) {

  auto filter = op.getFilter();
  auto input = op.getInput();

  auto filterLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->template getAttrOfType<ArrayAttr>("input_layout");

  SmallVector<Attribute> filterLayout(filterLayoutAttr.begin(),
                                      filterLayoutAttr.end());
  SmallVector<Attribute> inputLayout(inputLayoutAttr.begin(),
                                     inputLayoutAttr.end());

  auto maybeSortedFilter = sortByMemoryLayout(filter, filterLayout, b);
  auto maybeSortedInput = sortByMemoryLayout(input, inputLayout, b);

  if (failed(maybeSortedFilter) || failed(maybeSortedInput))
    return op.emitOpError("sortByMemoryLayout failed");

  auto sortedFilter = maybeSortedFilter.value();
  auto sortedInput = maybeSortedInput.value();

  auto newFilter = std::get<0>(sortedFilter);
  auto newInput = std::get<0>(sortedInput);
  LLVM_DEBUG(llvm::dbgs() << "newFilter=" << newFilter
                          << "\nnewInput=" << newInput << "\n");
  auto newFilterLayout = std::get<1>(sortedFilter);
  auto newInputLayout = std::get<1>(sortedInput);
  auto inputStrides = std::get<2>(sortedInput);

  // This is needed because ConvToGemm merges gemm K using the input layout.
  // However, if the layout is chw, we can't vectorize the loads, so it's
  // better to keep the previous behavior. So that, at least weights loads are
  // vectorized.
  // TODO: improve this
  if (inputStrides.size() > 1) {
    SmallVector<Attribute, 3> spatialDims;
    for (auto attr : inputLayout) {
      if (attr != b.getStringAttr("ni") && attr != b.getStringAttr("gi") &&
          attr != b.getStringAttr("ci"))
        spatialDims.push_back(attr);
    }
    LLVM_DEBUG(llvm::dbgs() << "inputStrides (" << inputStrides.size() << ")=");
    LLVM_DEBUG(llvm::interleaveComma(inputStrides, llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    auto ciPos = findIndex(inputLayout, b.getStringAttr("ci")).value();
    for (auto spatialDim : spatialDims) {
      auto spatialDimPos = findIndex(inputLayout, spatialDim).value();
      if (inputStrides[ciPos] > inputStrides[spatialDimPos]) {
        return failure();
      }
    }
  }
  bool noChange = newFilter == filter && newInput == input;
  return std::make_tuple(newFilter, newInput, b.getArrayAttr(newFilterLayout),
                         b.getArrayAttr(newInputLayout), noChange);
}

template <typename T>
struct ConvRewritePattern : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op, PatternRewriter &b) const final {
    auto maybeConvInfo = commonConv(op, b);
    if (failed(maybeConvInfo))
      return failure();

    Value newFilter, newInput;
    ArrayAttr newFilterLayout, newInputLayout;
    bool noChange;
    std::tie(newFilter, newInput, newFilterLayout, newInputLayout, noChange) =
        maybeConvInfo.value();

    // no need to create transforms if it's the same tensor
    if (noChange)
      return failure();

    auto newOp = b.replaceOpWithNewOp<rock::ConvOp>(
        op, op->getResultTypes(), newFilter, newInput, op.getOutput(),
        op.getFeaturesAttr(), op.getDerivedBlockSizeAttr(),
        op.getGridSizeAttr(), op.getPadding(), op.getStrides(),
        op.getDilations(), op.getParams() ? op.getParams().value() : nullptr);

    if (auto attr = op->template getAttrOfType<StringAttr>("perf_config"))
      newOp->setAttr("perf_config", attr);

    newOp->setAttr("filter_layout", newFilterLayout);
    newOp->setAttr("input_layout", newInputLayout);
    auto outputLayoutAttr =
        op->template getAttrOfType<ArrayAttr>("output_layout");
    if (outputLayoutAttr)
      newOp->setAttr("output_layout", outputLayoutAttr);

    return success();
  }
};

struct GemmRewritePattern : public OpRewritePattern<rock::GemmOp> {
  using OpRewritePattern<rock::GemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::GemmOp op,
                                PatternRewriter &b) const final {
    auto tensorA = op.getA();
    auto tensorB = op.getB();
    auto scaleA = op.getScaleA();
    auto scaleB = op.getScaleB();
    bool isScaledGemm = scaleA != Value{nullptr} && scaleB != Value{nullptr};

    // Process tensor A
    auto resultA =
        processTensorLayout(b, tensorA, {"G", "M", "K"}, op.getATransposed(),
                            tensorA.getType().getRank(), "TensorA");

    if (!resultA)
      return op.emitOpError("sortByMemoryLayout failed on TensorA");

    // Process tensor B
    auto resultB =
        processTensorLayout(b, tensorB, {"G", "K", "N"}, op.getBTransposed(),
                            tensorB.getType().getRank(), "TensorB");

    if (!resultB)
      return op.emitOpError("sortByMemoryLayout failed on TensorB");

    // no need to create transforms if it's the same tensor
    bool changeInLayout = resultA.finalLayout != resultA.initialLayout ||
                          resultB.finalLayout != resultB.initialLayout;

    // Process scale tensors if present
    Value newTensorScaleA = Value{nullptr};
    Value newTensorScaleB = Value{nullptr};
    UnitAttr transposedScaleA = UnitAttr{nullptr};
    UnitAttr transposedScaleB = UnitAttr{nullptr};
    if (isScaledGemm) {
      // Process scale A
      auto resultScaleA = processTensorLayout(
          b, scaleA, {"G", "M", "K"}, op.getAScaleTransposed(),
          scaleA.getType().getRank(), "ScaleA");

      if (!resultScaleA)
        return op.emitOpError("sortByMemoryLayout failed for scale tensors");

      // Process scale B
      auto resultScaleB = processTensorLayout(
          b, scaleB, {"G", "K", "N"}, op.getBScaleTransposed(),
          scaleB.getType().getRank(), "ScaleB");

      if (!resultScaleB)
        return op.emitOpError("sortByMemoryLayout failed for scale tensors");

      newTensorScaleA = resultScaleA.tensor;
      newTensorScaleB = resultScaleB.tensor;
      transposedScaleA = resultScaleA.transposed;
      transposedScaleB = resultScaleB.transposed;

      changeInLayout =
          changeInLayout ||
          (resultScaleA.finalLayout != resultScaleA.initialLayout) ||
          (resultScaleB.finalLayout != resultScaleB.initialLayout);
    }

    // if no change in layout, return failure
    if (!changeInLayout)
      return failure();

    auto newGemm = b.replaceOpWithNewOp<rock::GemmOp>(
        op, op->getResultTypes(), resultA.tensor, resultB.tensor, op.getC(),
        newTensorScaleA, newTensorScaleB, resultA.transposed,
        resultB.transposed, op.getCTransposedAttr(), transposedScaleA,
        transposedScaleB, op.getFeaturesAttr(), op.getStoreMethodAttr(),
        op.getDerivedBlockSizeAttr(), op.getGridSizeAttr(),
        op.getParams() ? op.getParams().value() : nullptr);

    if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
      newGemm->setAttr("perf_config", attr);

    return success();
  }
};

struct AttentionRewritePattern : public OpRewritePattern<rock::AttentionOp> {
  using OpRewritePattern<rock::AttentionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::AttentionOp op,
                                PatternRewriter &b) const final {
    auto q = op.getQueries();
    auto k = op.getKeys();
    auto v = op.getValues();

    auto maybeRewrite = commonGemmGemm(op, q, k, v, b);
    if (failed(maybeRewrite))
      return failure();

    Value newTensorQ, newTensorK, newTensorV;
    UnitAttr transposedQ, transposedK, transposedV;
    std::tie(newTensorQ, newTensorK, newTensorV, transposedQ, transposedK,
             transposedV) = maybeRewrite.value();

    auto newOp = rock::AttentionOp::create(
        b, op->getLoc(), op->getResultTypes(), newTensorQ, newTensorK,
        newTensorV, op.getPreSoftmaxElemWiseInputs(), op.getCurrentSeqLen(),
        op.getOut(), op.getLse(), op.getNumHeadsQAttr(), op.getNumHeadsKVAttr(),
        transposedQ, transposedK, transposedV, op.getOTransposedAttr(),
        op.getCausalAttr(), op.getPrefixCausalAttr(), op.getSplitKVAttr(),
        op.getFeaturesAttr(), op.getStoreMethodAttr(), op.getSoftmaxTypeAttr(),
        op.getParams0Attr(), op.getParams1Attr(), op.getFirstGemmIndicesAttr(),
        op.getPreSoftmaxHasSplitKVTransformsAttr());

    // copy linalg::GenericOp if there's any
    bool linalgOpFound = false;
    op.getPreSoftmaxBody().walk(
        [&linalgOpFound](linalg::GenericOp genOp) { linalgOpFound = true; });
    if (linalgOpFound) {
      b.inlineRegionBefore(op.getPreSoftmaxBody(), newOp.getPreSoftmaxBody(),
                           newOp.getPreSoftmaxBody().begin());
    }
    if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
      newOp->setAttr("perf_config", attr);

    b.replaceOp(op, newOp);

    return success();
  }
};

struct ConvElementwiseGemmRewritePattern
    : public OpRewritePattern<rock::ConvElementwiseGemmOp> {
  using OpRewritePattern<rock::ConvElementwiseGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::ConvElementwiseGemmOp op,
                                PatternRewriter &rw) const final {
    auto maybeConvInfo = commonConv(op, rw);
    if (failed(maybeConvInfo))
      return failure();

    // handle filter and input, conv params
    Value newFilter, newInput;
    ArrayAttr newFilterLayout, newInputLayout;
    bool convNoChange;
    std::tie(newFilter, newInput, newFilterLayout, newInputLayout,
             convNoChange) = maybeConvInfo.value();
    auto c = op.getC();

    // handle "c"
    auto resultC =
        processTensorLayout(rw, c, {"G", "K", "N"}, op.getTransposedC(),
                            cast<ShapedType>(c.getType()).getRank(), "TensorC");

    if (!resultC)
      return op.emitOpError("sortByMemoryLayout failed on TensorC");

    // no need to create transforms if it's the same tensors
    if (convNoChange && resultC.finalLayout == resultC.initialLayout)
      return failure();

    auto newOp = rock::ConvElementwiseGemmOp::create(
        rw, op->getLoc(), op->getResultTypes(), newFilter, newInput,
        resultC.tensor, op.getElemwiseInputs(), op.getOut(), resultC.transposed,
        op.getOTransposedAttr(), op.getFeaturesAttr(), op.getStoreMethodAttr(),
        op.getPaddingAttr(), op.getStridesAttr(), op.getDilationsAttr(),
        op.getParams0Attr(), op.getParams1Attr(), op.getFirstGemmIndicesAttr());

    // set attributes
    newOp->setAttr("filter_layout", newFilterLayout);
    newOp->setAttr("input_layout", newInputLayout);

    // copy linalg::GenericOp if there's any
    bool linalgOpFound = false;
    op.getPreSecondGemmBody().walk(
        [&linalgOpFound](linalg::GenericOp genOp) { linalgOpFound = true; });
    if (linalgOpFound) {
      rw.inlineRegionBefore(op.getPreSecondGemmBody(),
                            newOp.getPreSecondGemmBody(),
                            newOp.getPreSecondGemmBody().begin());
    }
    if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
      newOp->setAttr("perf_config", attr);

    rw.replaceOp(op, newOp);

    return success();
  }
};

struct GemmElementwiseGemmRewritePattern
    : public OpRewritePattern<rock::GemmElementwiseGemmOp> {
  using OpRewritePattern<rock::GemmElementwiseGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::GemmElementwiseGemmOp op,
                                PatternRewriter &rw) const final {
    auto a = op.getA();
    auto b = op.getB();
    auto c = op.getC();

    auto maybeRewrite = commonGemmGemm(op, a, b, c, rw);
    if (failed(maybeRewrite))
      return failure();

    Value newTensorQ, newTensorK, newTensorV;
    UnitAttr transposedQ, transposedK, transposedV;
    std::tie(newTensorQ, newTensorK, newTensorV, transposedQ, transposedK,
             transposedV) = maybeRewrite.value();

    auto newOp = rock::GemmElementwiseGemmOp::create(
        rw, op->getLoc(), op->getResultTypes(), newTensorQ, newTensorK,
        newTensorV, op.getElemwiseInputs(), op.getOut(), transposedQ,
        transposedK, transposedV, op.getOTransposedAttr(), op.getFeaturesAttr(),
        op.getStoreMethodAttr(), op.getParams0Attr(), op.getParams1Attr(),
        op.getFirstGemmIndicesAttr());

    // copy linalg::GenericOp if there's any
    bool linalgOpFound = false;
    op.getPreSecondGemmBody().walk(
        [&linalgOpFound](linalg::GenericOp genOp) { linalgOpFound = true; });
    if (linalgOpFound) {
      rw.inlineRegionBefore(op.getPreSecondGemmBody(),
                            newOp.getPreSecondGemmBody(),
                            newOp.getPreSecondGemmBody().begin());
    }
    if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
      newOp->setAttr("perf_config", attr);

    rw.replaceOp(op, newOp);

    return success();
  }
};

void RockSortDimensionsMemoryLayoutPass::runOnOperation() {
  auto func = getOperation();
  if (!func->hasAttr("kernel")) {
    return;
  }
  auto &ctx = getContext();
  GreedyRewriteConfig config;
  config.setStrictness(GreedyRewriteStrictness::ExistingOps);

  RewritePatternSet patternsConv(&ctx);
  patternsConv.add<ConvRewritePattern<rock::ConvOp>>(&ctx);
  if (failed(applyOpPatternsGreedily(getOperations<rock::ConvOp>(func),
                                     std::move(patternsConv), config)))
    return signalPassFailure();

  RewritePatternSet patternsGemm(&ctx);
  patternsGemm.add<GemmRewritePattern>(&ctx);
  if (failed(applyOpPatternsGreedily(getOperations<rock::GemmOp>(func),
                                     std::move(patternsGemm), config)))
    return signalPassFailure();

  RewritePatternSet patternsAttention(&ctx);
  patternsAttention.add<AttentionRewritePattern>(&ctx);
  if (failed(applyOpPatternsGreedily(getOperations<rock::AttentionOp>(func),
                                     std::move(patternsAttention), config)))
    return signalPassFailure();

  RewritePatternSet patternsGemmElementwiseGemm(&ctx);
  patternsGemmElementwiseGemm.add<GemmElementwiseGemmRewritePattern>(&ctx);
  if (failed(applyOpPatternsGreedily(
          getOperations<rock::GemmElementwiseGemmOp>(func),
          std::move(patternsGemmElementwiseGemm), config)))
    return signalPassFailure();

  RewritePatternSet patternsConvElementwiseGemm(&ctx);
  patternsConvElementwiseGemm.add<ConvElementwiseGemmRewritePattern>(&ctx);
  if (failed(applyOpPatternsGreedily(
          getOperations<rock::ConvElementwiseGemmOp>(func),
          std::move(patternsConvElementwiseGemm), config)))
    return signalPassFailure();
}
