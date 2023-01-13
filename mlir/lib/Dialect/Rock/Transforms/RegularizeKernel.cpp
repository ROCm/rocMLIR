//===- RegularizeKernel.cpp - rewrites to allow Rock kernel fusion  ------===//
//
// Copyright 2022 Advanced Micro Devices.
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
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREGULARIZEKERNELPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-fold-transpose"

using namespace mlir;
using namespace mlir::rock;

namespace {

// This rewrite will rewrite the linalg IO that has view like-ops surrounding
// them to be consumed by the linalg operation itself adjusting the indexing
// maps to faithfully represent them.
struct CollapseRewritePattern
    : public OpRewritePattern<memref::CollapseShapeOp> {
  using OpRewritePattern<memref::CollapseShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CollapseShapeOp collapse,
                                PatternRewriter &rewriter) const override;
};

struct ExpandRewritePattern : public OpRewritePattern<memref::ExpandShapeOp> {
  using OpRewritePattern<memref::ExpandShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ExpandShapeOp collapse,
                                PatternRewriter &rewriter) const override;
};

static int64_t lookup(int64_t val,
                      SmallVector<std::tuple<int64_t, uint32_t>, 8> &pairs) {
  for (auto ii = pairs.begin(); ii != pairs.end(); ++ii) {
    if (std::get<0>(*ii) == val) {
      auto idx = std::get<1>(*ii);
      pairs.erase(ii);
      return idx;
    }
  }
  return -1;
}

static bool
findCombination(int64_t inpSize,
                SmallVector<std::tuple<int64_t, uint32_t>, 8> &outPairs,
                uint32_t reqLen, uint32_t start, uint32_t curLen, bool check[],
                SmallVector<uint32_t> &mergeDims) {
  if (curLen > reqLen)
    return false;
  else if (curLen == reqLen) {
    int64_t outSize = 1;
    for (size_t i = 0; i < outPairs.size(); i++) {
      if (check[i])
        outSize *= std::get<0>(outPairs[i]);
    }
    if (outSize == inpSize) {
      for (size_t i = 0; i < outPairs.size(); i++) {
        if (check[i])
          mergeDims.push_back(std::get<1>(outPairs[i]));
      }
      return true;
    }
    return false;
  }
  if (start + curLen == mergeDims.size()) {
    // terminate
    return false;
  }
  check[start] = true;
  if (findCombination(inpSize, outPairs, reqLen, start + 1, curLen + 1, check,
                      mergeDims))
    return true;
  check[start] = false;
  if (findCombination(inpSize, outPairs, reqLen, start + 1, curLen, check,
                      mergeDims))
    return true;
  return false;
}

static void collectMatches(ArrayRef<int64_t> inpShape,
                           ArrayRef<int64_t> outShape,
                           SmallVector<SmallVector<uint32_t>> &merges) {
  SmallVector<std::tuple<int64_t, uint32_t>, 8> outPairs;
  for (auto &pair : llvm::enumerate(outShape))
    outPairs.push_back({pair.value(), pair.index()});

  // 0. find all exact matches
  for (const auto &pair : llvm::enumerate(inpShape)) {
    auto inpSize = pair.value();
    int64_t fidx = lookup(inpSize, outPairs);
    if (fidx >= 0) {
      merges[pair.index()] = {fidx};
    }
  }

  // 1. look for adjacent matches
  assert(outPairs.size() <= 8);
  bool check[8] = {
      false,
  };
  for (const auto &pair : llvm::enumerate(inpShape)) {
    auto inpIdx = pair.index();
    if (merges[inpIdx].empty()) {
      auto inpSize = pair.value();
      SmallVector<uint32_t> mergeDims;
      for (uint32_t i = 2; i < outPairs.size(); ++i) {
        if (findCombination(inpSize, outPairs, i, 0, 0, check, mergeDims))
          break;
      }
      assert(!mergeDims.empty());
      merges[inpIdx] = mergeDims;
    }
  }

  // 2. remove all 1s from outPairs
  auto oit = outPairs.begin();
  for (uint32_t i = 0, e = outPairs.size(); i < e; ++i) {
    if (std::get<0>(*oit) == 1) {
      uint32_t outIdx = std::get<1>(*oit);
      uint32_t inpIdx = 0;
      while (merges[inpIdx].empty())
        inpIdx++;
      merges[inpIdx].push_back(outIdx);
      outPairs.erase(oit);
    } else
      ++oit;
  }
  assert(outPairs.empty());
}

static void expandTensor(PatternRewriter &b, memref::ExpandShapeOp rop,
                         ArrayRef<int64_t> inpShape,
                         ArrayRef<int64_t> outShape) {
  // %3 = "tosa.reshape"(%2) {new_shape = [1, 12, 12, 32]} :
  // (tensor<1x12x384xf32>) -> tensor<1x12x12x32xf32>
  //    - inpShape = [1, 12, 384]
  //    - outShape = [1, 12, 12, 32]
  SmallVector<SmallVector<uint32_t>> merges(inpShape.size(), {});
  collectMatches(inpShape, outShape, merges);

  rock::BottomUpTMBuilder transform(b, inpShape, rop.getLoc());
  for (auto idxAndMerge : llvm::enumerate(merges)) {
    uint32_t idx = idxAndMerge.index();
    auto mergeDims = idxAndMerge.value();
    if (mergeDims.size() == 1) {
      transform.passThrough({mergeDims[0]}, {idx});
    } else {
      SmallVector<SmallString<8>> mergeNames;
      SmallVector<int64_t> mergeSizes;
      SmallVector<StringRef> mergeNameRefs;
      for (auto midx : mergeDims) {
        SmallString<8> mname(Twine("exp" + Twine(midx)).str());
        mergeNames.push_back(mname);
        mergeNameRefs.push_back(mergeNames.back());
        mergeSizes.push_back(outShape[midx]);
      }
      transform.unmerge(mergeNameRefs, mergeDims, transform.startName(idx),
                        mergeSizes);
    }
  }

  b.replaceOpWithNewOp<rock::TransformOp>(rop, rop.getOperand(),
                                          transform.get());
}

static void collapseTensor(PatternRewriter &b, memref::CollapseShapeOp rop,
                           ArrayRef<int64_t> inpShape,
                           ArrayRef<int64_t> outShape) {
  // %5 = "tosa.reshape"(%4) {new_shape = [12, 12, 32]} :
  // (tensor<1x12x12x32xf32>) -> tensor<12x12x32xf32>
  //    - inpShape = [1, 12, 12, 32]
  //    - outShape = [12, 12, 32]
  SmallVector<SmallVector<uint32_t>> merges(outShape.size(), {});
  collectMatches(outShape, inpShape, merges);

  rock::TopDownTMBuilder transform(b, outShape, rop.getLoc());
  for (auto idxAndMerge : llvm::enumerate(merges)) {
    uint32_t idx = idxAndMerge.index();
    auto mergeDims = idxAndMerge.value();
    if (mergeDims.size() == 1) {
      transform.passThrough({mergeDims[0]}, {idx});
    } else {
      SmallVector<SmallString<8>> mergeNames;
      SmallVector<int64_t> mergeSizes;
      SmallVector<StringRef> mergeNameRefs;
      for (auto midx : mergeDims) {
        SmallString<8> mname(Twine("m" + Twine(midx)).str());
        mergeNames.push_back(mname);
        mergeNameRefs.push_back(mergeNames.back());
        mergeSizes.push_back(inpShape[midx]);
      }
      transform.merge(mergeNameRefs, mergeDims, transform.startName(idx),
                      mergeSizes);
    }
  }

  b.replaceOpWithNewOp<rock::TransformOp>(rop, rop.getOperand(),
                                          transform.get());
}

LogicalResult
CollapseRewritePattern::matchAndRewrite(memref::CollapseShapeOp collapse,
                                        PatternRewriter &rewriter) const {

  auto inpType = collapse.getOperand().getType().cast<ShapedType>();
  auto outType = collapse.getResultType().cast<ShapedType>();
  collapseTensor(rewriter, collapse, inpType.getShape(), outType.getShape());

  return success();
}

LogicalResult
ExpandRewritePattern::matchAndRewrite(memref::ExpandShapeOp expand,
                                      PatternRewriter &rewriter) const {

  auto inpType = expand.getOperand().getType().cast<ShapedType>();
  auto outType = expand.getResultType().cast<ShapedType>();
  expandTensor(rewriter, expand, inpType.getShape(), outType.getShape());

  return success();
}

////////////////////////////////////////////////////////////////////////
////  Shuffle Transforms To Writers
////////////////////////////////////////////////////////////////////////
struct ShuffleTransformsUpRewritePattern
    : public OpRewritePattern<rock::GridwiseGemmOp> {
  using OpRewritePattern<rock::GridwiseGemmOp>::OpRewritePattern;

  static LogicalResult shuffleTransformsUp(PatternRewriter &b, Operation *start,
                                           Value result);
  static LogicalResult applyTransforms(PatternRewriter &b, Value alloc,
                                       Operation *forOp, Operation *backOp);

  LogicalResult matchAndRewrite(rock::GridwiseGemmOp ggemm,
                                PatternRewriter &rewriter) const override {
    // Input:
    // %x = alloc
    // %y = tx0 (%x)
    // %y = gemm (...)
    // %z = tx1 (%x)
    // %a = lag0 (%z)   -- input
    // Result:
    // %x' = alloc
    // %z' = tx1' (%x')
    // %y = tx0 (%z')
    // %y = gemm (...)
    // %a = lag0 (%x')

    return shuffleTransformsUp(rewriter, ggemm, ggemm.getC());
  }
};

LogicalResult ShuffleTransformsUpRewritePattern::applyTransforms(
    PatternRewriter &b, Value alloc, Operation *forwOp, Operation *backOp) {
  LogicalResult lres = failure();
  Value result;
  SmallVector<TransformMapAttr> transforms;
  while (auto top = dyn_cast<rock::TransformOp>(forwOp)) {
    result = top.getResult();
    if (!result.hasOneUse())
      return failure(); // currently restricted to 1 reader
    auto tmap = rock::invertTransformMap(b, top.getTransform());
    if (!tmap)
      return failure(); // not invertible
    transforms.push_back(tmap);
    forwOp = (*result.getUses().begin()).getOwner();
  }
  if (transforms.size()) {
    // check forwOp is reader
    // apply transformation
    PatternRewriter::InsertionGuard guard(b);
    b.setInsertionPoint(alloc.getDefiningOp());
    Value val = b.create<memref::AllocOp>(forwOp->getLoc(),
                                          result.getType().cast<MemRefType>())
                    .getResult();
    forwOp->replaceUsesOfWith(result, val);
    for (auto tx : llvm::reverse(transforms)) {
      auto top = b.create<rock::TransformOp>(forwOp->getLoc(), val, tx);
      val = top.getResult();
    }
    backOp->replaceUsesOfWith(alloc, val);
    lres = success();
  }
  // recurse to the next level
  if (auto laop = dyn_cast<linalg::GenericOp>(forwOp)) {
    if (laop.getNumOutputs() == 1) {
      if (succeeded(shuffleTransformsUp(b, forwOp, laop.getOutputs()[0])))
        lres = success();
    }
  }
  return lres;
}

LogicalResult ShuffleTransformsUpRewritePattern::shuffleTransformsUp(
    PatternRewriter &b, Operation *start, Value result) {
  Operation *prevOp = start;
  // trace back to alloc
  while (auto op = result.getDefiningOp()) {
    if (isa<memref::AllocOp>(op)) {
      // apply
      Operation *readPath = nullptr;
      for (auto &use : result.getUses()) {
        Operation *useOp = use.getOwner();
        if (useOp != prevOp) {
          if (readPath != nullptr)
            return failure(); // multi readers
          readPath = useOp;
        }
      }
      if (readPath == nullptr)
        return failure();
      return applyTransforms(b, result, readPath, prevOp);
    } else if (auto top = dyn_cast<rock::TransformOp>(op)) {
      prevOp = op;
      result = top.getOperand();
    } else {
      // unknown path
      assert(0);
    }
  }
  return failure();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
struct RockRegularizeKernelPass
    : public rock::impl::RockRegularizeKernelPassBase<
          RockRegularizeKernelPass> {
  void runOnOperation() override;
};
} // end namespace

void RockRegularizeKernelPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto func = getOperation();
  if (!func->hasAttr("kernel")) {
    return;
  }

  {
    RewritePatternSet patterns(ctx);
    patterns.add<CollapseRewritePattern, ExpandRewritePattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }

  {
    RewritePatternSet patterns(ctx);
    patterns.add<ShuffleTransformsUpRewritePattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }

  {
    RewritePatternSet patterns(ctx);
    patterns.add<PushToReadersRewritePattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
}
