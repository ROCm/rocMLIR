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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <limits>
#include <numeric>
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/Passes.h"

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
    : public rock::impl::RockSortDimensionsMemoryLayoutPassBase<RockSortDimensionsMemoryLayoutPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

template <typename Container>
FailureOr<Container> reorderArrayAttr(Container inputArray,
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

template <typename Container>
static FailureOr<std::tuple<Value, Container, SmallVector<uint32_t>>>
sortByMemoryLayout(Value tensor, const Container &layout, PatternRewriter &b) {
  ArrayAttr transforms;
  Value source;
  std::tie(source, transforms, std::ignore) = rock::untransform(b, tensor);

  if (transforms.empty()) {
    return std::make_tuple(tensor, layout, SmallVector<uint32_t>{});
  }

  rock::TransformMapAttr firstCoordTransform =
      cast<rock::TransformMapAttr>(transforms[0]);
  int64_t upperRank = firstCoordTransform.getUpperBounds().size();

  // no need to do anything if it's not a block argument
  if (!isa<BlockArgument>(source))
    return std::make_tuple(tensor, layout, SmallVector<uint32_t>{});

  SmallVector<uint32_t> strides(upperRank);
  for (int64_t idx = 0; idx < upperRank; idx++) {
    FailureOr<llvm::SmallDenseMap<int64_t, SmallVector<rock::SubDimInfo>>>
        maybeLowerSubDims =
            rock::getLowerSubDimensions(b, transforms, idx, true);
    if (failed(maybeLowerSubDims)) {
      LLVM_DEBUG(llvm::dbgs() << "lowerSubDims creation using "
                                 "getLowerSubDimensions is unsuccesful.\n");
      return failure();
    }
    auto lowerSubDims = maybeLowerSubDims.value();
    uint32_t minStride = std::numeric_limits<uint32_t>::max();
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
    return std::make_tuple(tensor, layout, SmallVector<uint32_t>{});

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
std::optional<size_t> findIndex(const ContainerTy &container,
                                const ElementTy &element) {
  auto it = llvm::find(container, element);
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

template <typename T>
struct ConvRewritePattern : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op, PatternRewriter &b) const final {
    auto filter = op.getFilter();
    auto input = op.getInput();

    auto filterLayoutAttr =
        op->template getAttrOfType<ArrayAttr>("filter_layout");
    auto inputLayoutAttr =
        op->template getAttrOfType<ArrayAttr>("input_layout");
    auto outputLayoutAttr =
        op->template getAttrOfType<ArrayAttr>("output_layout");

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
                            << "\n newInput=" << newInput << "\n");
    auto newFilterLayout = std::get<1>(sortedFilter);
    auto newInputLayout = std::get<1>(sortedInput);
    auto inputStrides = std::get<2>(sortedInput);

    // no need to create transforms if it's the same tensor
    if (newFilter == filter && newInput == input)
      return failure();

    // This is needed because ConvToGemm merges gemm K using the input layout.
    // However, if the layout is chw, we can't vectorize the loads, so it's
    // better to keep the previous behavior. So that, at least weights loads are
    // vectorized.
    // TODO: improve this
    SmallVector<StringAttr, 3> nonSpatialDims;
    for (auto attr : inputLayout) {
      auto name = cast<StringAttr>(attr);
      if (name != "ni" && name != "gi" && name != "ci")
        nonSpatialDims.push_back(name);
    }

    auto ciPos = findIndex(inputLayout, b.getStringAttr("ci")).value();
    for (auto spatialDim : nonSpatialDims) {
      auto spatialDimPos = findIndex(inputLayout, spatialDim).value();
      if (inputStrides[ciPos] > inputStrides[spatialDimPos]) {
        return failure();
      }
    }

    auto newOp = b.replaceOpWithNewOp<rock::ConvOp>(
        op, op->getResultTypes(), newFilter, newInput, op.getOutput(),
        op.getArch(), op.getFeatures(), op.getDerivedBlockSizeAttr(),
        op.getGridSizeAttr(), op.getPadding(), op.getStrides(),
        op.getDilations(), op.getParams() ? op.getParams().value() : nullptr,
        op.getNumCUAttr());

    newOp->setAttr("filter_layout", b.getArrayAttr(newFilterLayout));
    newOp->setAttr("input_layout", b.getArrayAttr(newInputLayout));
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

    SmallVector<StringRef> layoutA{"G", "M", "K"};
    if (op.getATransposedAttr())
      layoutA = {"G", "K", "M"};
    if (tensorA.getType().getRank() == 2)
      layoutA = {layoutA[1], layoutA[2]};

    SmallVector<StringRef> layoutB{"G", "K", "N"};
    if (op.getBTransposedAttr())
      layoutB = {"G", "N", "K"};
    if (tensorB.getType().getRank() == 2)
      layoutB = {layoutB[1], layoutB[2]};

    auto maybeSortedA = sortByMemoryLayout(tensorA, layoutA, b);
    auto maybeSortedB = sortByMemoryLayout(tensorB, layoutB, b);

    if (failed(maybeSortedA) || failed(maybeSortedB))
      return op.emitOpError("sortByMemoryLayout failed");

    auto sortedA = maybeSortedA.value();
    auto sortedB = maybeSortedB.value();

    LLVM_DEBUG(llvm::dbgs() << "sortedA=" << std::get<0>(sortedA)
                            << " sortedB=" << std::get<0>(sortedB) << "\n");

    // the batch size is currently required to be the first one. If that's not
    // the case we need to add an extra transform.
    auto batchReorderA =
        reorderBatch(std::get<0>(sortedA), std::get<1>(sortedA), "K", b);
    auto batchReorderB =
        reorderBatch(std::get<0>(sortedB), std::get<1>(sortedB), "N", b);

    Value newTensorA = std::get<0>(batchReorderA);
    Value newTensorB = std::get<0>(batchReorderB);
    UnitAttr transposedA = std::get<1>(batchReorderA);
    UnitAttr transposedB = std::get<1>(batchReorderB);
    auto finalLayoutA = std::get<2>(batchReorderA);
    auto finalLayoutB = std::get<2>(batchReorderB);

    LLVM_DEBUG(llvm::dbgs() << "newTensorA=" << newTensorA
                            << " newTensorB=" << newTensorB << "\n");
    LLVM_DEBUG(llvm::dbgs() << "transposedA=" << transposedA
                            << "\ntransposedB=" << transposedB << "\n");

    // no need to create transforms if it's the same tensor
    if (finalLayoutA == layoutA && finalLayoutB == layoutB)
      return failure();

    b.replaceOpWithNewOp<rock::GemmOp>(
        op, op->getResultTypes(), newTensorA, newTensorB, op.getC(),
        transposedA, transposedB, op.getCTransposedAttr(), op.getArchAttr(),
        op.getNumCUAttr(), op.getFeaturesAttr(), op.getStoreMethodAttr(),
        op.getDerivedBlockSizeAttr(), op.getGridSizeAttr(),
        op.getParams() ? op.getParams().value() : nullptr);

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

    SmallVector<StringRef> layoutQ{"G", "M", "K"};
    if (op.getQTransposed())
      layoutQ = {"G", "K", "M"};
    if (q.getType().getRank() == 2)
      layoutQ = {layoutQ[1], layoutQ[2]};

    SmallVector<StringRef> layoutK{"G", "K", "N"};
    if (op.getKTransposed())
      layoutK = {"G", "N", "K"};
    if (k.getType().getRank() == 2)
      layoutK = {layoutK[1], layoutK[2]};

    SmallVector<StringRef> layoutV{"G", "K", "N"};
    if (op.getVTransposed())
      layoutV = {"G", "N", "K"};
    if (k.getType().getRank() == 2)
      layoutV = {layoutV[1], layoutV[2]};

    auto maybeSortedQ = sortByMemoryLayout(q, layoutQ, b);
    auto maybeSortedK = sortByMemoryLayout(k, layoutK, b);
    auto maybeSortedV = sortByMemoryLayout(v, layoutV, b);

    if (failed(maybeSortedQ) || failed(maybeSortedK) || failed(maybeSortedV))
      return op.emitOpError("sortByMemoryLayout failed");

    auto sortedQ = maybeSortedQ.value();
    auto sortedK = maybeSortedK.value();
    auto sortedV = maybeSortedV.value();

    LLVM_DEBUG(llvm::dbgs() << "sortedQ=" << std::get<0>(sortedQ)
                            << " sortedK=" << std::get<0>(sortedK)
                            << " sortedV=" << std::get<0>(sortedV) << "\n");

    // the batch size is currently required to be the first one. If that's not
    // the case we need to add an extra transform.
    auto batchReorderQ =
        reorderBatch(std::get<0>(sortedQ), std::get<1>(sortedQ), "K", b);
    auto batchReorderK =
        reorderBatch(std::get<0>(sortedK), std::get<1>(sortedK), "N", b);
    auto batchReorderV =
        reorderBatch(std::get<0>(sortedV), std::get<1>(sortedV), "N", b);

    Value newTensorQ = std::get<0>(batchReorderQ);
    Value newTensorK = std::get<0>(batchReorderK);
    Value newTensorV = std::get<0>(batchReorderV);
    UnitAttr transposedQ = std::get<1>(batchReorderQ);
    UnitAttr transposedK = std::get<1>(batchReorderK);
    UnitAttr transposedV = std::get<1>(batchReorderV);
    auto finalLayoutQ = std::get<2>(batchReorderQ);
    auto finalLayoutK = std::get<2>(batchReorderK);
    auto finalLayoutV = std::get<2>(batchReorderV);

    // no need to create transforms if it's the same tensor
    if (finalLayoutQ == layoutQ && finalLayoutK == layoutK &&
        finalLayoutV == layoutV)
      return failure();

    auto newOp = b.create<rock::AttentionOp>(
        op->getLoc(), op->getResultTypes(), newTensorQ, newTensorK, newTensorV,
        op.getPreSoftmaxElemWiseInputs(), op.getCurrentSeqLen(), op.getOut(),
        transposedQ, transposedK, transposedV, op.getOTransposedAttr(),
        op.getArchAttr(), op.getFeaturesAttr(), op.getNumCUAttr(),
        op.getParams0Attr(), op.getParams1Attr(), op.getFirstGemmIdxAttr());

    // copy linalg::GenericOp if there's any
    bool linalgOpFound = false;
    op.getPreSoftmaxBody().walk(
        [&linalgOpFound](linalg::GenericOp genOp) { linalgOpFound = true; });
    if (linalgOpFound) {
      b.inlineRegionBefore(op.getPreSoftmaxBody(), newOp.getPreSoftmaxBody(),
                           newOp.getPreSoftmaxBody().begin());
    }
    b.replaceOp(op, newOp);

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
  config.strictMode = GreedyRewriteStrictness::ExistingOps;

  RewritePatternSet patternsConv(&ctx);
  patternsConv.add<ConvRewritePattern<rock::ConvOp>>(&ctx);
  if (failed(applyOpPatternsGreedily(getOperations<rock::ConvOp>(func),
                                     std::move(patternsConv), config)))
    return signalPassFailure();

  RewritePatternSet patternsConvBwdData(&ctx);
  patternsConvBwdData.add<ConvRewritePattern<rock::ConvBwdDataOp>>(&ctx);
  if (failed(applyOpPatternsGreedily(getOperations<rock::ConvBwdDataOp>(func),
                                     std::move(patternsConvBwdData), config)))
    return signalPassFailure();

  RewritePatternSet patternsConvBwdWeight(&ctx);
  patternsConvBwdWeight.add<ConvRewritePattern<rock::ConvBwdWeightOp>>(&ctx);
  if (failed(
          applyOpPatternsGreedily(getOperations<rock::ConvBwdWeightOp>(func),
                                  std::move(patternsConvBwdWeight), config)))
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
}
