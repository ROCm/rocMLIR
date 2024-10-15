//===- SugarToLoops.cpp - Lower Rock sugar and transforming_for  ----===//
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
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKSUGARTOLOOPSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-sugar-to-loops"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockSugarToLoopsPass
    : public rock::impl::RockSugarToLoopsPassBase<RockSugarToLoopsPass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// TransformingFor lowering.
//===----------------------------------------------------------------------===//
struct TransformingForRewritePattern
    : public OpRewritePattern<TransformingForOp> {
  using OpRewritePattern<TransformingForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransformingForOp op,
                                PatternRewriter &b) const override {
    using AffineResults = SmallVector<Value, 8>;
    Location loc = op.getLoc();
    SmallVector<int64_t> bounds;
    bounds.reserve(op.getBounds().size());
    for (llvm::APInt v : op.getBounds().getAsValueRange<IntegerAttr>()) {
      int64_t bound = v.getZExtValue();
      bounds.push_back(bound);
    }

    SmallVector<int64_t> strides;
    strides.reserve(op.getStrides().size());
    for (llvm::APInt v : op.getStrides().getAsValueRange<IntegerAttr>()) {
      int64_t stride = v.getZExtValue();
      strides.push_back(stride);
    }

    bool useDiffs = op.getUseIndexDiffs().value_or(false);
    bool unroll = op.getForceUnroll().value_or(false);

    uint32_t nDomains = op.domains();

    // For each iteration domain, store the initial outputs of each affine map
    // in the transform chain when using index diffs. When there are no index
    // diffs, this value is ignored.
    SmallVector<SmallVector<AffineResults>, 2> lowerInits;
    // For each domain, store the sequence of composed affine maps needed to
    // compute the result coordinate, along with the transform map that
    // triggered each break in the chain. Such a break is created at any point
    // where the validity of map coordinates is impacted.
    SmallVector<SmallVector<std::pair<AffineMap, TransformMapAttr>>, 2>
        allComposedMaps;

    if (useDiffs) {
      for (uint32_t i = 0; i < nDomains; ++i) {
        lowerInits.emplace_back();
        // Needed to handle the empty map case correctly.
        allComposedMaps.emplace_back();
        SmallVectorImpl<AffineResults> &lowerInit = lowerInits.back();
        ArrayAttr transforms = op.getTransforms(i);
        lowerInit.reserve(transforms.size());
        if (transforms.empty()) {
          AffineResults init(op.getUpperInits(i));
          lowerInit.push_back(init);
          continue;
        }
        for (auto t : transforms.getAsRange<TransformMapAttr>()) {
          AffineMap map = t.getMap().getAffineMap();
          std::optional<AffineResults> init;
          if (lowerInit.empty())
            init = affine::expandAffineMap(b, loc, map, op.getUpperInits(i));
          else
            init = affine::expandAffineMap(b, loc, map,
                                           lowerInit[lowerInit.size() - 1]);
          if (!init)
            return failure();
          lowerInit.push_back(*init);
        }
      }
    } else { // !useDiffs
      for (uint32_t i = 0; i < nDomains; ++i) {
        allComposedMaps.emplace_back();
        SmallVectorImpl<std::pair<AffineMap, TransformMapAttr>> &composedMaps =
            allComposedMaps.back();
        ArrayAttr transforms = op.getTransforms(i);
        SmallVector<TransformMapAttr> toCompose;
        for (auto t : transforms.getAsRange<TransformMapAttr>()) {
          toCompose.push_back(t);
          if (mapImpactsValidity(t)) {
            AffineMap composed = composeTransforms(toCompose);
            composedMaps.emplace_back(composed, t);
            toCompose.clear();
          }
        }
        // Account for all maps after the last validity impact.
        AffineMap finalComposed = composeTransforms(toCompose);
        composedMaps.emplace_back(finalComposed, nullptr);
      }
    }

    // Having done pre-computation, create an affine loop nest over the upper
    // rectangle. This'll be unrolled as needed.
    llvm::SmallVector<affine::AffineForOp, 5> loops;
    llvm::SmallVector<Value, 5> ivs;

    // We're about to setInsertionPointTo{Start,End} a bunch to jum around
    // inside loops, so save the original insertion point to not mess up the
    // rewriting infrastructure and to avoid copying builders around.
    OpBuilder::InsertionGuard goingIntoLoopsGuard(b);
    for (const auto &pair : llvm::zip(bounds, strides)) {
      int64_t bound, stride;
      std::tie(bound, stride) = pair;
      // This is a one-iteration loop, so don't actually emit a loop so as to
      // enable constant folding.
      if (bound == stride && bound > 0) {
        ivs.push_back(b.createOrFold<arith::ConstantIndexOp>(loc, 0));
        continue;
      }
      llvm::SmallVector<Value, 3> iterInits;
      if (loops.empty())
        llvm::copy(op.getIterInits(), std::back_inserter(iterInits));
      else
        llvm::copy(loops[loops.size() - 1].getRegionIterArgs(),
                   std::back_inserter(iterInits));
      auto loop =
          b.create<affine::AffineForOp>(loc, 0, bound, stride, iterInits);
      ivs.push_back(loop.getInductionVar());
      // remove default affine.yield for cleaner code later
      if (iterInits.empty())
        b.eraseOp(loop.getBody()->getTerminator());
      b.setInsertionPointToStart(loop.getBody());
      loops.push_back(loop);
    }

    // Create code to actually transform the coordinates
    IRMapping cloneMap;
    Block::BlockArgListType validities = op.getValidities();
    for (uint32_t i = 0; i < nDomains; ++i) {
      Block::BlockArgListType lower = op.getLowerCoords(i);
      ArrayAttr transforms = op.getTransforms(i);
      if (!useDiffs || transforms.empty()) {
        AffineResults computed;
        Value isValid =
            b.create<arith::ConstantIntOp>(loc, true, b.getI1Type());
        // Start by offsetting the upper inputs.
        for (auto p : llvm::zip(op.getUpperInits(i), ivs)) {
          computed.push_back(
              b.create<AddIOp>(loc, std::get<0>(p), std::get<1>(p)));
        }
        for (const auto &[composedMap, transform] : allComposedMaps[i]) {
          if (!composedMap) // empty transformations
            continue;
          std::optional<AffineResults> transformed =
              affine::expandAffineMap(b, loc, composedMap, computed);
          if (!transformed)
            return failure();
          computed.assign(*transformed);
          if (transform) { // Time for bounds checks or other validity updates
            Value validityUpdate =
                updateValidityAfter(b, loc, transform, computed);
            isValid =
                b.createOrFold<arith::AndIOp>(loc, validityUpdate, isValid);
          }
        }
        for (auto p : llvm::zip(lower, computed)) {
          cloneMap.map(std::get<0>(p), std::get<1>(p));
        }
        cloneMap.map(validities[i], isValid);
      } else { // index diff maps
        IndexDiffUpdateOp lastDiff;
        Value isValid =
            b.create<arith::ConstantIntOp>(loc, true, b.getI1Type());
        for (const auto &[t, lowerInit] : llvm::zip(
                 transforms.getAsRange<TransformMapAttr>(), lowerInits[i])) {
          if (!lastDiff)
            lastDiff = b.create<IndexDiffUpdateOp>(loc, t, ivs, lowerInit);
          else
            lastDiff = b.create<IndexDiffUpdateOp>(
                loc, t, lastDiff.getLowerDiff(), lowerInit);
          if (mapImpactsValidity(t)) {
            Value validityUpdate =
                updateValidityAfter(b, loc, t, lastDiff.getLowerIndices());
            isValid =
                b.createOrFold<arith::AndIOp>(loc, validityUpdate, isValid);
          }
        }
        for (auto p : llvm::zip(lower, lastDiff.getLowerIndices())) {
          cloneMap.map(std::get<0>(p), std::get<1>(p));
        }
        cloneMap.map(validities[i], isValid);
      }
    }
    // One-iteration loops should be inlined, do so here.
    if (loops.empty()) {
      auto yieldOp = cast<rock::YieldOp>(op.getBody()->getTerminator());
      b.replaceAllUsesWith(op.getResults(), yieldOp.getOperands());
      SmallVector<Value> blockArgValues;
      for (auto [blockArg, initValue] :
           llvm::zip(op.getIterArgs(), op.getIterInits())) {
        cloneMap.map(blockArg, initValue);
      }
      blockArgValues.reserve(op.getBody()->getNumArguments());
      for (auto blockArg : op.getBody()->getArguments())
        blockArgValues.push_back(cloneMap.lookup(blockArg));
      b.inlineBlockBefore(op.getBody(), op.getOperation()->getBlock(),
                          op.getOperation()->getIterator(), blockArgValues);
      b.eraseOp(yieldOp);
      b.eraseOp(op);
      return success();
    }
    // Map loop arguments, clone operations in body
    affine::AffineForOp il = loops[loops.size() - 1];
    for (auto p : llvm::zip(op.getIterArgs(), il.getRegionIterArgs())) {
      cloneMap.map(std::get<0>(p), std::get<1>(p));
    }
    for (Operation &bodyOp : op.getBody()->getOperations()) {
      if (auto yield = dyn_cast<rock::YieldOp>(bodyOp)) {
        llvm::SmallVector<Value, 3> terminatorArgs;
        for (Value v : op.getBody()->getTerminator()->getOperands()) {
          terminatorArgs.push_back(cloneMap.lookupOrDefault(v));
        }
        b.create<affine::AffineYieldOp>(loc, terminatorArgs);
      } else {
        b.clone(bodyOp, cloneMap);
      }
    }

    if (loops.size() > 1) {
      for (size_t i = 0, e = loops.size() - 1; i < e; ++i) {
        affine::AffineForOp inner = loops[i + 1];
        b.setInsertionPointToEnd(loops[i].getBody());
        b.create<affine::AffineYieldOp>(loc, inner.getResults());
      }
    }

    b.replaceOp(op, loops[0].getResults());
    // Note: the unrolling process doesn't play nice with pattern rewrites
    // Therefore, we just mark loops for unrolling and deal with it in a
    // separate pass
    if (unroll)
      for (affine::AffineForOp loop : loops)
        loop->setAttr("forceUnroll", b.getUnitAttr());

    return success();
  }
};

// Determine if the operation provided is a constant, and return its value if it
// is
std::optional<int64_t> isConstantValue(Value v) {
  auto *op = v.getDefiningOp();
  if (nullptr == op)
    return std::nullopt;
  while (auto cast = dyn_cast<IndexCastOp>(op)) {
    op = cast.getIn().getDefiningOp();
  }
  if (auto intOp = dyn_cast<ConstantIntOp>(op)) {
    return intOp.value();
  }
  if (auto indexOp = dyn_cast<ConstantIndexOp>(op)) {
    return indexOp.value();
  }
  return std::nullopt;
}

struct ExtractMultiBufferRewritePattern
    : public OpRewritePattern<ExtractMultiBufferOp> {
  using OpRewritePattern<ExtractMultiBufferOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ExtractMultiBufferOp op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();
    // This operation lowers to a `switch` statement implemented via
    // `arith.select`
    SmallVector<Value> buffers = llvm::to_vector(op.getBuffers());
    assert(!buffers.empty() && "There should be at least one buffer");
    size_t multiBufferFactor = buffers.size();
    Value mbFactor = b.createOrFold<ConstantIndexOp>(loc, multiBufferFactor);
    Value currentBuffer = buffers.back();
    Value modSelectIndex =
        b.create<arith::RemUIOp>(loc, op.getSelectIndex(), mbFactor);
    for (size_t i = 1; i < multiBufferFactor; i++) {
      auto idx =
          b.createOrFold<ConstantIndexOp>(loc, multiBufferFactor - i - 1);
      auto cmp = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                         modSelectIndex, idx);
      currentBuffer = b.create<arith::SelectOp>(
          loc, cmp, currentBuffer, buffers[multiBufferFactor - i - 1]);
    }
    b.replaceAllUsesWith(op.getResult(), currentBuffer);

    b.eraseOp(op);
    return success();
  }
};

struct IndexDiffUpdateRewritePattern
    : public OpRewritePattern<IndexDiffUpdateOp> {
  using OpRewritePattern<IndexDiffUpdateOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IndexDiffUpdateOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    TransformMapAttr transformMap = op.getMap();

    Operation::operand_range upperIndicesDiff = op.getUpperDiffs();
    Operation::operand_range lowerIndicesOriginal = op.getLowerOrig();

    // Ensure index_diff_update is lowered in def-use order
    bool reevaluateOps = false;
    do {
      reevaluateOps = false;
      for (Value v : op->getOperands()) {
        Operation *defOp = v.getDefiningOp();
        if (auto pred = dyn_cast_or_null<IndexDiffUpdateOp>(defOp)) {
          PatternRewriter::InsertionGuard predGuard(b);
          // Prevent def-after-use errors
          b.setInsertionPoint(pred);
          if (failed(matchAndRewrite(pred, b)))
            return failure();
          reevaluateOps = true;
          break;
        }
        // Handle constant folds from unrolling just in case of a stray
        // affine.apply
        SmallVector<Value> constants;
        if (nullptr != defOp && succeeded(b.tryFold(defOp, constants)) &&
            !constants.empty()) {
          b.replaceOp(defOp, constants);
          reevaluateOps = true;
          break;
        }
      }
    } while (reevaluateOps);

    Value zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);
    // Obtain the shape of lower level memref.
    ArrayRef<int64_t> lowerLayerShape = transformMap.getLowerBounds();

    // Input:
    // - upper_diff
    // - lower_indices_original
    // - lower_layer_bounds
    // - F : a vector of functions mapping upper level dimensions to lower level
    // dimensions with attached metadata about how they're constructed
    //
    // Output:
    // - lower_diff : the computed diffs on the lower layer. such information
    //                would be passed to the next layer below as upper diff.
    // - lower_indices_updated : the updated lower layer indices. clients will
    //                           use the values to issue loads / stores.
    //
    // For each transform f specified in F:
    //   Let P be the upper dimensions used by f.
    //   Let Q be the lower dimensions used by f.
    //   Let T be upper_layer_bounds.
    //
    //   Switch f.type:
    //     Case Pad :
    //       |P| = |Q|
    //       For each i in P, and its counterpart j in Q
    //         lower_diff[j] = upper_diff[i]
    //         lower_indices_updated[j] = lower_indices_origina[j] +
    //         lower_diff[j]
    //
    //     Case PassThrough :
    //       |P| = |Q|
    //       For each i in P, and its counterpart j in Q
    //         lower_diff[j] = upper_diff[i]
    //         lower_indices_updated[j] = lower_indices_origina[j] +
    //         lower_diff[j]
    //
    //     Case Slice :
    //       |P| = |Q|
    //       For each i in P, and its counterpart j in Q
    //         lower_diff[j] = upper_diff[i]
    //         lower_indices_updated[j] = lower_indices_origina[j] +
    //         lower_diff[j]
    //
    //     Case Embed:
    //       |P| = k, currently k will be >= 2.
    //       |Q| shall be 1
    //       Let (p_{0}, ... , p_{k-1}) be elements in P, |P| = k
    //       Let (e_{0}, ... , e_{k-1}) be parameters of P
    //       Let j be the counterpart in q
    //       lower_diff[j] = sum_over_P(e_{i} * upper_diff[p_{i}])
    //       lower_indices_updated[j] = lower_indices_origina[j] + lower_diff[j]
    //
    //     Case UnMerge:
    //       |Q| shall be 1
    //       Let (p_{0}, ... , p_{k-1}) be elements in P, |P| = k
    //       Let (e_{0}, ... , e_{k-1}) be parameters of P
    //       Let (f_{0}, ... , f_{k-1})
    //         The value of f_{i} is defined as:
    //           f_{k-1} = 1
    //           f_{i} = mul_over_{domain: e_[i+1 .. k-1], iterator=l}(T_{l})
    //       Let j be the counterpart in q
    //         lower_diff[j] = sum_over_P(f_{i} * upper_diff[p_{i}])
    //         lower_indices_updated[j] = lower_indices_origina[j] +
    //         lower_diff[j]
    //
    //     Case Unfold:
    //       This transformation is currently only used on filter, when c/y/x
    //       dimensions are together.
    //       |P| shall be 1
    //       Let (q_{0}, ... , q_{k-1}) be elements in Q, |Q| = k
    //       Let (f_{0}, ... , f_{k-1}) be elements in F to compute from P to Q
    //       For each i in Q,
    //         lower_diff_tilda[i] = f_{i}(upper_diff)
    //       For each i in Q,
    //         lower_indices_modified[i] = lower_indices_original[i] +
    //           lower_diff_tilda[i]
    //       lower_diff = lower_diff_tilda
    //       lower_indices_updated = lower_indices_modified
    //
    //     Case Merge:
    //       |P| shall be 1
    //       Let (q_{0}, ... , q_{k-1}) be elements in Q, |Q| = k
    //       Let (f_{0}, ... , f_{k-1}) be elements in F to compute from P to Q
    //       For each i in Q,
    //         lower_diff_tilda[i] = f_{i}(upper_diff)
    //       For each i in Q,
    //         lower_indices_modified[i] = lower_indices_original[i] +
    //           lower_diff_tilda[i]
    //       For each i in Q, starting from i-1 down to 0 in descending order
    //         lower_indices_carrychecked[i] = carry/overflow check for
    //           lower_indices_modified[i]
    //       lower_diff = lower_indices_carrychecked - lower_indices_original
    //       lower_indices_updated = lower_indices_carrychecked
    //

    // Look into layout attribute inside transform metadata.

    // lower level diff map
    // key : lower level dimension value.
    // value : lower level diff on that dimension.
    DenseMap<uint32_t, Value> lowerIndicesDiffMap;

    // lower level updated coordinate map
    // key : lower level dimension value.
    // value : lower level updated coordinate on that dimension.
    DenseMap<uint32_t, Value> lowerIndicesUpdatedMap;

    auto addToOriginal = [&b, loc](Value original, Value diff) -> Value {
      auto mbDiffConst = isConstantValue(diff);
      if (mbDiffConst.has_value()) {
        int64_t diff = mbDiffConst.value();
        if (diff == 0) {
          return original;
        }
        auto mbOriginalConst = isConstantValue(original);
        if (mbOriginalConst.has_value()) {
          return b.create<ConstantIndexOp>(loc, diff + mbOriginalConst.value());
        }
      }
      return b.create<AddIOp>(loc, original, diff);
    };

    LLVM_DEBUG(llvm::dbgs()
               << "Applying index diffs to " << transformMap << "\n");
    // Iterate through all transformations specified in g.
    for (auto mapping : transformMap.getOps()) {
      LLVM_DEBUG(llvm::dbgs() << "transform: " << mapping << "\n");

      // Obtain transformation information from f.
      TransformType transformation = mapping.getType();
      ArrayRef<uint32_t> p = mapping.getUpperDims();
      ArrayRef<uint32_t> q = mapping.getLowerDims();
      ArrayRef<int64_t> e = mapping.getParams();

      if (transformation == TransformType::Embed) {
        assert(e.size() == p.size());
        assert(q.size() == 1);
        Value lowerDiff = zeroConstantOp;
        for (unsigned iter = 0; iter < e.size(); ++iter) {
          int64_t coefficient = e[iter];
          uint32_t upperDim = p[iter];
          auto mbUpperDiff = isConstantValue(upperIndicesDiff[upperDim]);
          auto mbLowerDiff = isConstantValue(lowerDiff);
          if (mbUpperDiff.has_value() && mbLowerDiff.has_value()) {
            lowerDiff = b.create<ConstantIndexOp>(
                loc, mbLowerDiff.value() + coefficient * mbUpperDiff.value());
          } else {
            lowerDiff = b.create<AddIOp>(
                loc, lowerDiff,
                b.create<MulIOp>(loc,
                                 b.create<ConstantIndexOp>(loc, coefficient),
                                 upperIndicesDiff[upperDim]));
          }
        }

        uint32_t lowerDim = q[0];
        lowerIndicesDiffMap[lowerDim] = lowerDiff;
        lowerIndicesUpdatedMap[lowerDim] =
            addToOriginal(lowerIndicesOriginal[lowerDim], lowerDiff);
      } else if (transformation == TransformType::Unmerge) {
        assert(e.size() == p.size());
        assert(q.size() == 1);
        uint32_t upperDim = p[0];
        Value lowerDiff = upperIndicesDiff[upperDim];
        for (unsigned iter = 1; iter < e.size(); ++iter) {
          int64_t coefficient = e[iter];
          uint32_t upperDim = p[iter];
          auto mbUpperDiff = isConstantValue(upperIndicesDiff[upperDim]);
          auto mbLowerDiff = isConstantValue(lowerDiff);
          if (mbUpperDiff.has_value() && mbLowerDiff.has_value()) {
            lowerDiff = b.create<ConstantIndexOp>(
                loc, mbUpperDiff.value() + coefficient * mbLowerDiff.value());
          } else {
            lowerDiff = b.create<AddIOp>(
                loc, upperIndicesDiff[upperDim],
                b.create<MulIOp>(loc,
                                 b.create<ConstantIndexOp>(loc, coefficient),
                                 lowerDiff));
          }
        }
        uint32_t lowerDim = q[0];
        lowerIndicesDiffMap[lowerDim] = lowerDiff;
        lowerIndicesUpdatedMap[lowerDim] =
            addToOriginal(lowerIndicesOriginal[lowerDim], lowerDiff);
      } else if ((transformation == TransformType::PassThrough) ||
                 (transformation == TransformType::Pad) ||
                 (transformation == TransformType::Slice)) {
        assert(p.size() == q.size());
        for (unsigned iter = 0; iter < q.size(); ++iter) {
          uint32_t upperDim = p[iter];
          uint32_t lowerDim = q[iter];
          Value upperDiff = upperIndicesDiff[upperDim];
          Value lowerDiff = upperDiff;
          lowerIndicesDiffMap[lowerDim] = lowerDiff;
          lowerIndicesUpdatedMap[lowerDim] =
              addToOriginal(lowerIndicesOriginal[lowerDim], lowerDiff);
        }
      } else if (transformation == TransformType::Merge) {
        assert(p.size() == 1);
        uint32_t upperDim = p[0];

        // Obtain the affine map underlying the transform.
        AffineMap affineMap = transformMap.getMap().getAffineMap();

        SmallVector<Value, 8> lowerDiffModified;
        auto mbUpperDiffVal = isConstantValue(upperIndicesDiff[upperDim]);
        if (mbUpperDiffVal.has_value()) {
          // In case upper level diff is a constant, use constantFold.
          int64_t upperDiff = mbUpperDiffVal.value();

          // Populate an upper diff vector with all indices 0, other than
          // upperDim dimension set as upperDiff.
          SmallVector<Attribute, 8> upperDiffModified;
          for (unsigned iter = 0; iter < upperIndicesDiff.size(); ++iter) {
            int64_t v = (iter == upperDim) ? upperDiff : 0;
            upperDiffModified.push_back(b.getI32IntegerAttr(v));
          }
          assert(upperDiffModified.size() == upperIndicesDiff.size());

          // Apply map to compute index lower diff, from index upper diff using
          // constantFold.
          SmallVector<Attribute, 8> lowerDiffModifiedAttr;
          (void)affineMap.constantFold(upperDiffModified,
                                       lowerDiffModifiedAttr);
          assert(lowerDiffModifiedAttr.size() == lowerIndicesOriginal.size());

          for (uint32_t iter = 0; iter < lowerDiffModifiedAttr.size(); ++iter) {
            lowerDiffModified.push_back(b.create<ConstantIndexOp>(
                loc,
                mlir::cast<IntegerAttr>(lowerDiffModifiedAttr[iter]).getInt()));
          }
          assert(lowerDiffModified.size() == lowerIndicesOriginal.size());
        } else {
          // In case upper level diff is not constant, use
          // affine::expandAffineMap.

          Value upperDiff = upperIndicesDiff[upperDim];

          // Populate an upper diff vector with all indices 0, other than
          // upperDim dimension set as upperDiff.
          SmallVector<Value, 8> upperDiffModified;
          for (uint32_t iter = 0; iter < upperIndicesDiff.size(); ++iter) {
            Value v = (iter == upperDim) ? upperDiff : zeroConstantOp;
            upperDiffModified.push_back(v);
          }
          assert(upperDiffModified.size() == upperIndicesDiff.size());

          // Apply map to compute index lower diff, from index upper diff using
          // affine::expandAffineMap.
          lowerDiffModified =
              affine::expandAffineMap(b, loc, affineMap, upperDiffModified)
                  .value();
          assert(lowerDiffModified.size() == lowerIndicesOriginal.size());
        }

        // Obtain lower diffs prior to carry check.
        SmallVector<Value, 8> lowerDiffs;
        for (unsigned iter = 0; iter < q.size(); ++iter) {
          uint32_t lowerDim = q[iter];
          Value lowerDiff = lowerDiffModified[lowerDim];
          lowerDiffs.push_back(lowerDiff);
        }
        assert(lowerDiffs.size() == q.size());

        // Compute updated lower indices by adding original lower indices with
        // lower diffs.
        SmallVector<Value, 8> lowerIndicesModified;
        for (uint32_t iter = 0; iter < q.size(); ++iter) {
          uint32_t lowerDim = q[iter];
          lowerIndicesModified.push_back(
              addToOriginal(lowerIndicesOriginal[lowerDim], lowerDiffs[iter]));
        }
        assert(lowerIndicesModified.size() == q.size());

        // Add carry check, excluding leading runs of unit dimensions,
        // whose carry information is irrelevant.

        // Carry checked lower indices.
        DenseMap<uint32_t, Value> lowerDiffsCarryChecked;
        DenseMap<uint32_t, Value> lowerIndicesCarryChecked;
        for (uint32_t iter = 0; iter < q.size(); ++iter) {
          int64_t lowerDim = q[iter];
          lowerDiffsCarryChecked[lowerDim] = lowerDiffs[iter];
          lowerIndicesCarryChecked[lowerDim] = lowerIndicesModified[iter];
        }
        assert(lowerDiffsCarryChecked.size() == lowerIndicesModified.size());
        assert(lowerIndicesCarryChecked.size() == lowerIndicesModified.size());

        // Runs of initial length 1 don't require a carry check. This replaces
        // MIOpen's Unfold. In the affine map, these initial 1s are the constant
        // 0, and so they should stay there.
        ssize_t slowestDimIdx =
            e.take_while([](int64_t len) { return len == 1; }).size();

        // We only implement carry logic. Borrow logic would never happen as
        // upper index diffs would always be positive in the current
        // algorithm.
        Value overflowOp = zeroConstantOp;
        for (ssize_t iter = q.size() - 1; iter >= slowestDimIdx; --iter) {
          uint32_t lowerDim = q[iter];
          int64_t upperBound = e[iter];
          if (upperBound == 1) {
            // The carry will necessarily bounce to the next dimension,
            // so don't bother generating code for it.
            continue;
          }
          // If the overflow is statically 0, nothing gets added
          Value diff =
              addToOriginal(lowerDiffsCarryChecked[lowerDim], overflowOp);
          Value index =
              addToOriginal(lowerIndicesCarryChecked[lowerDim], overflowOp);

          // Don't generate overflow for the uppermost dimension,
          // as this can lead to adresses wrapping back into bounds
          if (iter == slowestDimIdx) {
            lowerDiffsCarryChecked[lowerDim] = diff;
            lowerIndicesCarryChecked[lowerDim] = index;
            continue;
          }
          auto mbConstantDiff = isConstantValue(diff);
          auto mbConstantIndex = isConstantValue(index);

          // If we get lucky, everything is constant and so we have a constant
          // result
          if (mbConstantIndex.has_value() && mbConstantDiff.has_value()) {
            int64_t index = mbConstantIndex.value();
            int64_t diff = mbConstantDiff.value();
            if (index < upperBound) {
              overflowOp = zeroConstantOp;
              lowerIndicesCarryChecked[lowerDim] =
                  b.create<ConstantIndexOp>(loc, index);
              lowerDiffsCarryChecked[lowerDim] =
                  b.create<ConstantIndexOp>(loc, diff);
            } else {
              int64_t carry = index / upperBound;
              int64_t newIndex = index % upperBound;
              int64_t newDiff = diff - (carry * upperBound);
              overflowOp = b.create<ConstantIndexOp>(loc, carry);
              lowerIndicesCarryChecked[lowerDim] =
                  b.create<ConstantIndexOp>(loc, newIndex);
              lowerDiffsCarryChecked[lowerDim] =
                  b.create<ConstantIndexOp>(loc, newDiff);
            }
            continue;
          }
          // No change -> no carry-out
          if (mbConstantDiff.value_or(-1L) == 0) {
            overflowOp = zeroConstantOp;
            lowerDiffsCarryChecked[lowerDim] = diff;
            lowerIndicesCarryChecked[lowerDim] = index;
            continue;
          }

          Value upperBoundOp = b.create<ConstantIndexOp>(loc, upperBound);
          Value carry = b.create<DivUIOp>(loc, index, upperBoundOp);
          Value newIndex = b.create<RemUIOp>(loc, index, upperBoundOp);
          // If the merge is, as is typical, near the end of the
          // transformations this computation should get hit by the dead code
          // eleminator
          Value newDiff = b.create<SubIOp>(
              loc, diff, b.create<MulIOp>(loc, carry, upperBoundOp));

          overflowOp = carry;
          lowerDiffsCarryChecked[lowerDim] = newDiff;
          lowerIndicesCarryChecked[lowerDim] = newIndex;
        }

        assert(lowerDiffsCarryChecked.size() == lowerIndicesModified.size());
        assert(lowerIndicesCarryChecked.size() == lowerIndicesModified.size());
        lowerDiffs.clear();
        lowerIndicesModified.clear();
        for (uint32_t iter = 0; iter < q.size(); ++iter) {
          uint32_t lowerDim = q[iter];
          lowerDiffs.push_back(lowerDiffsCarryChecked[lowerDim]);
          lowerIndicesModified.push_back(lowerIndicesCarryChecked[lowerDim]);
        }
        assert(lowerDiffs.size() == q.size());
        assert(lowerIndicesModified.size() == q.size());

        // Set lowerIndicesDiffMap and lowerIndicesUpdatedMap.
        for (uint32_t iter = 0; iter < q.size(); ++iter) {
          int64_t lowerDim = q[iter];
          lowerIndicesDiffMap[lowerDim] = lowerDiffs[iter];
          lowerIndicesUpdatedMap[lowerDim] = lowerIndicesModified[iter];
        }
      } else if (transformation == TransformType::AddDim) {
        // Do nothing - the dimension will be dropped by the code below
      } else if (transformation == TransformType::Broadcast) {
        // lower broadcast dims, uses map
        for (uint32_t i = 0; i < e.size(); ++i) {
          int64_t lowerLen = e[i];
          Value lowerLenOp = b.create<ConstantIndexOp>(loc, lowerLen);
          auto mbUpperDiff = isConstantValue(upperIndicesDiff[p[i]]);
          Value wrappedDiff;
          if (mbUpperDiff.has_value()) {
            wrappedDiff =
                b.create<ConstantIndexOp>(loc, *mbUpperDiff % lowerLen);
          } else {
            wrappedDiff = b.createOrFold<RemUIOp>(loc, upperIndicesDiff[p[i]],
                                                  lowerLenOp);
          }
          Value newLower =
              addToOriginal(lowerIndicesOriginal[q[i]], wrappedDiff);
          newLower = b.createOrFold<RemUIOp>(loc, newLower, lowerLenOp);
          Value lowerDiff =
              b.createOrFold<SubIOp>(loc, newLower, lowerIndicesOriginal[q[i]]);
          lowerIndicesDiffMap[q[i]] = lowerDiff;
          lowerIndicesUpdatedMap[q[i]] = newLower;
        }
      } else if (transformation == TransformType::ConstDim) {
        for (uint32_t i = 0; i < q.size(); ++i) {
          // A constant dimension has its original value and no difference
          lowerIndicesDiffMap[q[i]] = zeroConstantOp;
          lowerIndicesUpdatedMap[q[i]] = lowerIndicesOriginal[q[i]];
        }
      }
    } // for (auto mapping : transforms.getOps())

    // Populate results: indices, _then_ diffs
    SmallVector<Value, 10> results;
    assert(lowerIndicesUpdatedMap.size() == lowerLayerShape.size());
    for (unsigned iter = 0; iter < lowerLayerShape.size(); ++iter)
      results.push_back(lowerIndicesUpdatedMap[iter]);

    assert(lowerIndicesDiffMap.size() == lowerLayerShape.size());
    for (unsigned iter = 0; iter < lowerLayerShape.size(); ++iter)
      results.push_back(lowerIndicesDiffMap[iter]);
    b.replaceOp(op, results);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ExtractSlice lowering.
//===----------------------------------------------------------------------===//
struct ExtractSliceRewritePattern : public OpRewritePattern<ExtractSliceOp> {
  using OpRewritePattern<ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractSliceOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    Value base = op.getCoord();
    if (auto destType = dyn_cast<VectorType>(op.getResult().getType())) {
      if (destType == cast<VectorType>(op.getVector().getType())) {
        // Extracting something the same size as the vector is a noop since
        // the index must be 0 for the op to be defined. This is here in case
        // the canonicalizer didn't catch this or didn't run.
        b.replaceOp(op, op.getVector());
        return success();
      }
      int64_t size = destType.getNumElements();
      Value ret = createZeroConstantOp(b, loc, destType);
      for (int64_t i = 0; i < size; ++i) {
        Value cDest = b.createOrFold<ConstantIndexOp>(loc, i);
        Value cSrc = b.createOrFold<AddIOp>(loc, base, cDest);
        Value v = b.create<vector::ExtractElementOp>(loc, op.getVector(), cSrc);
        ret = b.create<vector::InsertElementOp>(loc, v, ret, cDest);
      }
      b.replaceOp(op, ret);
    } else {
      b.replaceOpWithNewOp<vector::ExtractElementOp>(op, op.getVector(), base);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// InsertSlice lowering.
//===----------------------------------------------------------------------===//
struct InsertSliceRewritePattern : public OpRewritePattern<InsertSliceOp> {
  using OpRewritePattern<InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertSliceOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    Value base = op.getCoord();
    if (auto srcType = dyn_cast<VectorType>(op.getSource().getType())) {
      if (srcType == cast<VectorType>(op.getDest().getType())) {
        // Inserting a slice of the same size as the destination is a noop
        // since the index must be 0 for the op to be defined. This is here in
        // case the canonicalizer didn't run or didn't catch the problem.
        b.replaceOp(op, op.getSource());
        return success();
      }
      int64_t size = srcType.getNumElements();
      Value ret = op.getDest();
      for (int64_t i = 0; i < size; ++i) {
        Value cSrc = b.createOrFold<ConstantIndexOp>(loc, i);
        Value cDest = b.createOrFold<AddIOp>(loc, base, cSrc);
        Value v = b.create<vector::ExtractElementOp>(loc, op.getSource(), cSrc);
        ret = b.create<vector::InsertElementOp>(loc, v, ret, cDest);
      }
      b.replaceOp(op, ret);
    } else {
      b.replaceOpWithNewOp<vector::InsertElementOp>(op, op.getSource(),
                                                    op.getDest(), base);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GlobalLoad lowering.
//===----------------------------------------------------------------------===//

/// Return the number of elements in `memref`, accounting for the possibility
/// of dynamic shapes.
static Value computeMemRefNumElements(OpBuilder &b, Location loc,
                                      Value memref) {
  auto type = cast<MemRefType>(memref.getType());
  if (type.hasStaticShape())
    return b.createOrFold<ConstantIndexOp>(loc, type.getNumElements());
  Value result = b.createOrFold<arith::ConstantIndexOp>(loc, 1);
  for (int64_t i = 0, e = type.getRank(); i < e; ++i) {
    Value dimConst = b.createOrFold<arith::ConstantIndexOp>(loc, i);
    Value dim = b.createOrFold<memref::DimOp>(loc, memref, dimConst);
    result = b.createOrFold<arith::MulIOp>(loc, b.getIndexType(), dim, result);
  }
  return result;
}

static Value getConstIntOrIndexValue(OpBuilder &b, Location loc, int64_t value,
                                     Type type) {
  if (isa<IndexType>(type)) {
    return b.create<ConstantIndexOp>(loc, value);
  }
  return b.create<ConstantIntOp>(loc, value, type);
}

// Manually flatten a set of coordinates into a single address
static Value flattenCoords(OpBuilder &b, Location loc, ArrayRef<Value> coords,
                           ArrayRef<int64_t> shape) {
  Value flatCoord = coords.back();
  Type coordType = flatCoord.getType();
  int64_t stride = 1;
  for (int i = shape.size() - 2; i >= 0; i--) {
    stride *= shape[i + 1];
    flatCoord = b.create<arith::AddIOp>(
        loc, flatCoord,
        b.create<arith::MulIOp>(
            loc, coords[i],
            getConstIntOrIndexValue(b, loc, stride, coordType)));
  }

  return flatCoord;
}

// Manually unflatten a single address into a set of coordinates
static void unflattenCoords(OpBuilder &b, Location loc, Value flatAddress,
                            ArrayRef<int64_t> shape,
                            SmallVector<Value> &unflattenedAddress) {
  unflattenedAddress.resize(shape.size());
  Type coordType = flatAddress.getType();
  int64_t coeff = 1;
  for (int i = shape.size() - 1; i >= 0; i--) {
    Value thisCoord = b.create<arith::DivUIOp>(
        loc, flatAddress, getConstIntOrIndexValue(b, loc, coeff, coordType));
    thisCoord = b.create<arith::RemUIOp>(
        loc, thisCoord, getConstIntOrIndexValue(b, loc, shape[i], coordType));
    unflattenedAddress[i] = thisCoord;
    coeff *= shape[i];
  }
}

/// Atomic add for a scalar fp16. Using the CAS loop (atomicRMWOp) alternative
/// is significantly slower so we extend the scalar in a vector and use the
/// buffer_atomic_add_fp16 instead. We have to take care of the alignment
/// manually
static void atomicFp16AddAligned(OpBuilder &b, Location loc, Value data,
                                 Value dest, ArrayRef<Value> coords,
                                 bool useBufferOobChecks) {

  assert(isa<ShapedType>(dest.getType()) && "Data needs to have a shape!");
  ArrayRef<int64_t> shape = cast<ShapedType>(dest.getType()).getShape();
  assert(coords.size() == shape.size() &&
         "Shape and coordinates should have the same size!");

  // Always try to pack a scalar fp16 into a vector of 2 elements
  const int packedVectorLen = 2;

  // Compute the last non-unit dim
  int64_t lastNonUnitDim = shape.size() - 1;
  while (shape[lastNonUnitDim] == 1 && lastNonUnitDim > 0)
    lastNonUnitDim--;

  // Get the flattened size
  int64_t flattenedSize = 1;
  for (auto s : shape) {
    flattenedSize *= s;
  }

  // If last non-unit dimension is odd, we need to work on the flattened version
  // of the matrix
  Value address = coords[lastNonUnitDim];
  if (shape[lastNonUnitDim] % 2 != 0)
    address = flattenCoords(b, loc, coords, shape);

  // If all the shapes are odd, we have no choice: we need to add a guard and
  // use unpacked atomic_rmw to compute the atomic addition for the last
  // element: In that case, the last element  will be aligned, but it will be
  // "half" out of boundaries, which means the hardware will simply give up and
  // won't do  anything. However, we cannot step back, because the step back
  // would be unaligned
  if (flattenedSize % 2 != 0) {
    Value lastElem = b.create<arith::ConstantIntOp>(loc, flattenedSize - 1, 32);
    Value isNotLastElem = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, address, lastElem);
    auto guard = b.create<scf::IfOp>(loc, isNotLastElem, true);
    b.setInsertionPointToStart(&guard.getElseRegion().front());
    SmallVector<Value> indexCoords;
    for (auto c : coords)
      indexCoords.push_back(b.create<IndexCastOp>(loc, b.getIndexType(), c));
    b.create<memref::AtomicRMWOp>(loc, AtomicRMWKind::addf, data, dest,
                                  indexCoords);
    b.setInsertionPointToStart(&guard.getThenRegion().front());
  }

  // Useful consts
  Type addressElemType = address.getType();
  Value zero = getConstIntOrIndexValue(b, loc, 0, addressElemType);
  Value one = getConstIntOrIndexValue(b, loc, 1, addressElemType);
  Value two = getConstIntOrIndexValue(b, loc, 2, addressElemType);

  // Extended packed data to use with the intrinsic
  Value dataExt = createZeroConstantOp(
      b, loc, vectorTypeOrSelf(b.getF16Type(), packedVectorLen));
  Value dataExt0 = b.create<vector::InsertElementOp>(loc, data, dataExt, zero);
  Value dataExt1 = b.create<vector::InsertElementOp>(loc, data, dataExt, one);

  // Manual alignment logic : if (addr % 2 != 0) step{AddressData}Back
  Value stepBack = b.create<arith::SubIOp>(loc, address, one);
  Value alignment = b.create<arith::RemUIOp>(loc, address, two);
  Value stepBackCond =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, alignment, zero);

  // Step back data and address
  Value selectAddress =
      b.create<arith::SelectOp>(loc, stepBackCond, stepBack, address);
  Value selectDataExt =
      b.create<arith::SelectOp>(loc, stepBackCond, dataExt1, dataExt0);

  SmallVector<Value> alignedCoords(coords);
  alignedCoords[lastNonUnitDim] = selectAddress;

  // If the last non-unit dim is odd, we need to unflatten the address back and
  // use the structured address to write back to memory
  if (shape[lastNonUnitDim] % 2 != 0)
    unflattenCoords(b, loc, selectAddress, shape, alignedCoords);

  b.create<amdgpu::RawBufferAtomicFaddOp>(loc, selectDataExt, dest,
                                          alignedCoords, useBufferOobChecks,
                                          nullptr, nullptr);
}

/// Call builder(int64_t base, Type thisElem) repeatedly, advancing `base` by
/// the length of `thisElem` each time, where `thisElem` is a scalar or vector
/// less than 128 bits long. This is temporary and needed for the buffer
/// operations - once address space works in the backend, we'll just be able to
/// global_load.
static void perHardwareOp(Type realType,
                          llvm::function_ref<void(int64_t, Type)> builder) {
  Type elemType = getElementTypeOrSelf(realType);
  int64_t numElems = 1;
  if (auto vecTy = dyn_cast<VectorType>(realType))
    numElems = vecTy.getNumElements();
  int64_t maxElemsPerOp = 128 / elemType.getIntOrFloatBitWidth();
  int64_t offset = 0;
  while (offset < numElems) {
    int64_t thisOpNumElems = std::min(numElems - offset, maxElemsPerOp);
    // The ops only work on powers of two so correct this here.
    thisOpNumElems = (int64_t)1 << llvm::Log2_64(thisOpNumElems);
    Type thisOpType = vectorTypeOrSelf(elemType, thisOpNumElems);
    builder(offset, thisOpType);
    offset += thisOpNumElems;
  }
}

static Value asGlobal(PatternRewriter &b, Value memref) {
  auto type = cast<MemRefType>(memref.getType());
  auto globalType = MemRefType::get(
      type.getShape(), type.getElementType(), type.getLayout(),
      b.getAttr<gpu::AddressSpaceAttr>(gpu::AddressSpace::Global));
  return b.createOrFold<memref::MemorySpaceCastOp>(memref.getLoc(), globalType,
                                                   memref);
}

/// The buffer coordinate oob trick requires having at least one coordinate, so
/// this function takes 0D memrefs and adds a single unit-length dimension
/// so all the IR validates.
static Value zeroDMemrefAsOneD(PatternRewriter &b, Value memref) {
  auto type = cast<MemRefType>(memref.getType());
  auto oneDType = MemRefType::get({1}, type.getElementType(), nullptr,
                                  type.getMemorySpace());
  ArrayAttr expansions = b.getArrayAttr({});
  SmallVector<ReassociationIndices, 4> reassociation;
  for (Attribute attr : expansions) {
    ArrayAttr arrayAttrElem = cast<ArrayAttr>(attr);
    ReassociationIndices indices;
    for (Attribute indexAttr : arrayAttrElem) {
      indices.push_back(cast<IntegerAttr>(indexAttr).getInt());
    }
    reassociation.push_back(indices);
  }
  ArrayRef<ReassociationIndices> reassociationRef = reassociation;
  return b.createOrFold<memref::ExpandShapeOp>(memref.getLoc(), oneDType,
                                               memref, reassociationRef);
}

std::tuple<SmallVector<Value>, Type> getCoordsAndType(PatternRewriter &b,
                                                      GlobalLoadOp op) {
  MemRefType srcType = op.getSource().getType();
  Location loc = op.getLoc();
  SmallVector<Value> coords(op.getSourceCoord());
  Type originalLoadedType = op.getResult().getType();
  int64_t originalLoadVecLen = 1;
  if (VectorType originalLoadVecType =
          dyn_cast<VectorType>(originalLoadedType)) {
    originalLoadVecLen = originalLoadVecType.getNumElements();
  }
  if (srcType.getElementType().getIntOrFloatBitWidth() >= 8 ||
      originalLoadVecLen != 1) {
    return {coords, originalLoadedType};
  } else if (srcType.getElementType().getIntOrFloatBitWidth() == 4) {
    ArrayRef<int64_t> shape = srcType.getShape();
    Value flatAddress = flattenCoords(b, loc, coords, shape);
    Type coordType = flatAddress.getType();
    Value one = getConstIntOrIndexValue(b, loc, 1, coordType);
    flatAddress = b.createOrFold<arith::ShRUIOp>(loc, flatAddress, one);
    flatAddress = b.createOrFold<arith::ShLIOp>(loc, flatAddress, one);
    SmallVector<Value> newCoords;
    unflattenCoords(b, loc, flatAddress, shape, newCoords);
    Type loadedType = VectorType::get({2}, srcType.getElementType());
    return {newCoords, loadedType};
  } else {
    llvm_unreachable("less than 4 bit element types are not implemented");
  }
}

Value selectDataIf4b(PatternRewriter &b, GlobalLoadOp op, Value loadedVec) {
  MemRefType srcType = op.getSource().getType();
  Type originalLoadedType = op.getResult().getType();
  if (srcType.getElementType().getIntOrFloatBitWidth() >= 8) {
    return loadedVec;
  }
  int64_t originalLoadVecLen = 1;
  if (VectorType originalLoadVecType =
          dyn_cast_if_present<VectorType>(originalLoadedType)) {
    originalLoadVecLen = originalLoadVecType.getNumElements();
  }
  if (originalLoadVecLen != 1) {
    return loadedVec;
  } else if (srcType.getElementType().getIntOrFloatBitWidth() == 4) {
    assert(isa<VectorType>(loadedVec.getType()));
    Location loc = op.getLoc();
    SmallVector<Value, 5> coords(op.getSourceCoord());
    ArrayRef<int64_t> shape = srcType.getShape();
    Value flatAddress = flattenCoords(b, loc, coords, shape);
    Type coordType = flatAddress.getType();
    Value one = getConstIntOrIndexValue(b, loc, 1, coordType);
    Value lsb = b.createOrFold<arith::AndIOp>(loc, flatAddress, one);
    return b.createOrFold<vector::ExtractElementOp>(loc, loadedVec, lsb);
  } else {
    llvm_unreachable("less than 4 bit element types are not implemented");
  }
}

struct GlobalLoadRewritePattern : public OpRewritePattern<GlobalLoadOp> {
  using OpRewritePattern<GlobalLoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GlobalLoadOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    Value source = op.getSource();
    Value valid = op.getValid();

    Type loadedType;
    SmallVector<Value> coords;
    std::tie(coords, loadedType) = getCoordsAndType(b, op);

    llvm::APInt validConst = APInt::getZero(1);
    bool hasI64Idx = op.getNeeds64BitIdx();
    bool isAlwaysValid =
        matchPattern(valid, m_ConstantInt(&validConst)) && validConst.isOne();
    Value numElems = computeMemRefNumElements(b, loc, source);
    APInt numElemsConst(64, 0);
    bool isStaticSize = matchPattern(numElems, m_ConstantInt(&numElemsConst));
    bool emitOobChecks =
        !isStaticSize || !isAlwaysValid || (hasI64Idx && op.getCanReadOffEnd());

    APInt numBytes =
        numElemsConst *
        (cast<ShapedType>(source.getType()).getElementTypeBitWidth() / 8);
    // In cases where we need more than 2 GB of offset to index but are still
    // using 32-bit indexing, we'll need to use buffer operations. In the
    // dymanic shape case, we'll already be in the i64 case, so we don't set
    // this.
    bool useBufferOps = !hasI64Idx && (numBytes.trunc(32).isNegative() ||
                                       emitOobChecks || op.getCanReadOffEnd());

    if (!useBufferOps) {
      source = asGlobal(b, source);
    } else if (useBufferOps && emitOobChecks && coords.empty()) {
      source = zeroDMemrefAsOneD(b, source);
      coords.push_back(b.createOrFold<ConstantIndexOp>(loc, 0));
    }

    PatternRewriter::InsertionGuard insertGuard(b);
    if (emitOobChecks && !useBufferOps) {
      Value cond = valid;
      if (op.getCanReadOffEnd()) {
        Value fallsOffEnd = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::uge, coords[0], numElems);
        cond = b.create<arith::AndIOp>(loc, fallsOffEnd, cond);
      }
      auto guard = b.create<scf::IfOp>(loc, loadedType, cond, true, true);
      b.replaceOp(op, guard);

      b.setInsertionPointToEnd(guard.getBody(1));
      Value zeroes = createZeroConstantOp(b, loc, loadedType);
      b.create<scf::YieldOp>(loc, zeroes);
      b.setInsertionPointToEnd(guard.getBody(0));
    }

    if (useBufferOps) {
      // Implement bounds checks for buffer ops by sending any out of bounds
      // write off the end of the buffer, causing the hardware to return 0
      // if the write is out of bounds.
      if (emitOobChecks) {
        Value zeroConstantOp = b.createOrFold<ConstantIndexOp>(loc, 0);
        for (Value &c : MutableArrayRef<Value>(coords).drop_back())
          c = b.create<arith::SelectOp>(loc, valid, c, zeroConstantOp);
        Value &lastCoord = coords.back();
        lastCoord = b.create<arith::SelectOp>(loc, valid, lastCoord, numElems);
      }
      for (Value &c : MutableArrayRef<Value>(coords))
        c = b.create<IndexCastOp>(loc, b.getI32Type(), c);
      Value origLastCoord = coords.empty() ? nullptr : coords.back();
      Value loaded = createZeroConstantOp(b, loc, loadedType);
      perHardwareOp(loadedType, [&](int64_t offset, Type thisOpTy) {
        Value offsetConst = b.createOrFold<arith::ConstantIndexOp>(loc, offset);
        if (offset != 0) {
          Value offsetI32Const =
              b.createOrFold<arith::ConstantIntOp>(loc, offset, 32);
          coords.back() =
              b.create<arith::AddIOp>(loc, origLastCoord, offsetI32Const);
        }
        Value thisLoad = b.create<amdgpu::RawBufferLoadOp>(
            loc, thisOpTy, source, coords,
            /*boundsCheck=*/(emitOobChecks || op.getCanReadOffEnd()), nullptr,
            nullptr);
        if (isa<VectorType>(loadedType))
          loaded = b.createOrFold<InsertSliceOp>(loc, loadedType, thisLoad,
                                                 loaded, offsetConst);
        else
          loaded = thisLoad;
      });
      loaded = selectDataIf4b(b, op, loaded);
      b.replaceOp(op, loaded);
    } else {
      Value loaded;
      if (isa<VectorType>(loadedType))
        loaded = b.create<vector::LoadOp>(loc, loadedType, source, coords);
      else
        loaded = b.create<memref::LoadOp>(loc, loadedType, source, coords);
      loaded = selectDataIf4b(b, op, loaded);
      if (emitOobChecks)
        b.create<scf::YieldOp>(loc, loaded);
      else
        b.replaceOp(op, loaded);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GlobalStore lowering.
//===----------------------------------------------------------------------===//

// Implement atomic floating point max.
//
// This _used_ to implement a clever hack for gfx9xx that used a type pun
// to integer and then did an atomic signed_max/unsigned_min for
// positive/negative floats, but, with large tensors, we can't do that
// because upstream doesn't have the ability to do type puns on a memref.
// We don't know if the smax/umin trick is faster than a CAS loop,
// so we're keeping it around in the comments below for performance evaluation
// once we have a reduction-utilizing client.
//
// Returns the operation that the underlying global_store could be replaced
// with.
static Operation *makeAtomicFmax(PatternRewriter &b, Location loc, Value data,
                                 Value dest, ArrayRef<Value> coords,
                                 bool useBufferOps, bool useBufferOobChecks) {
  // if (bitEnumContainsAll(features, GemmFeatures::atomic_fmax_f32)) {
  if (useBufferOps)
    return b.create<amdgpu::RawBufferAtomicFmaxOp>(
        loc, data, dest, coords, useBufferOobChecks, nullptr, nullptr);
  return b.create<memref::AtomicRMWOp>(loc, AtomicRMWKind::maximumf, data, dest,
                                       coords);
// Disabled because we can't make this hack work in general.
#if 0
  }
  Value dataAsInt = b.create<arith::BitcastOp>(loc, b.getI32Type(), data);
  // Note: this doesn't work, and you'd need to add an upstream op which does
  // this.
  Value destAsInt = b.createOrFold<memref::CastOp>(
      loc, cast<MemRefType>(dest.getType()).clone(b.getI32Type()), dest);
  Value zeroConstantOp = b.createOrFold<ConstantIntOp>(loc, 0, 32);
  Value signbitConstantOp = b.createOrFold<ConstantIntOp>(loc, 0x80000000, 32);
  Value sign = b.create<arith::AndIOp>(loc, signbitConstantOp, dataAsInt);
  auto isPos = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                       zeroConstantOp, sign);
  auto ret = b.create<scf::IfOp>(
      loc, isPos,
      [&](OpBuilder &b, Location loc) {
        if (useBuffers)
          b.create<amdgpu::RawBufferAtomicSmaxOp>(loc, dataAsInt, destAsInt,
                                                  coords, useBufferOobChecks,
                                                  nullptr, nullptr);
        else
          b.create<memref::AtomicRMWOp>(loc, AtomicRMWKind::maxs, dataAsInt,
                                        destAsInt, coords);
      },
      [&](OpBuilder &b, Location loc) {
        if (useBuffers)
          b.create<amdgpu::RawBufferAtomicUminOp>(loc, dataAsInt, destAsInt,
                                                  coords, useBufferOobChecks,
                                                  nullptr, nullptr);
        else
          b.create<memref::AtomicRMWOp>(loc, AtomicRMWKind::minu, dataAsInt,
                                        destAsInt, coords);
      });
  return ret;
#endif
}

struct GlobalStoreRewritePattern : public OpRewritePattern<GlobalStoreOp> {
  using OpRewritePattern<GlobalStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GlobalStoreOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    Value dest = op.getDest();
    Value valid = op.getValid();
    Type elemTy = cast<MemRefType>(dest.getType()).getElementType();
    int64_t len = op.getLength().getZExtValue();
    Type storeTy = vectorTypeOrSelf(elemTy, len);

    SmallVector<Value, 5> coords(op.getDestCoord());
    Value sourceStart = op.getSourceCoord();

    llvm::APInt validConst = APInt::getZero(1);
    bool hasI64Idx = op.getNeeds64BitIdx();
    bool isAlwaysValid =
        matchPattern(valid, m_ConstantInt(&validConst)) && validConst.isOne();
    Value numElems = computeMemRefNumElements(b, loc, dest);
    APInt numElemsConst(64, 0);
    matchPattern(numElems, m_ConstantInt(&numElemsConst));
    bool isStaticSize = matchPattern(numElems, m_ConstantInt(&numElemsConst));
    bool emitOobChecks = !isStaticSize || !isAlwaysValid ||
                         (hasI64Idx && op.getCanStoreOffEnd());

    APInt numBytes =
        numElemsConst *
        (cast<ShapedType>(dest.getType()).getElementTypeBitWidth() / 8);
    // In cases where we need more than 2 GB of offset to index but are still
    // using 32-bit indexing, we'll need to use buffer operations. In the
    // dymanic shape case, we'll already be in the i64 case, so we don't set
    // this.
    StoreMethod memoryOp = op.getStoreMethod();
    bool isAtomic = memoryOp != StoreMethod::Set;

    bool isAtomicF16add = memoryOp == StoreMethod::AtomicAdd && elemTy.isF16();
    bool useBufferOps =
        !hasI64Idx && (numBytes.trunc(32).isNegative() || emitOobChecks ||
                       op.getCanStoreOffEnd() || isAtomicF16add);
    bool useBufferOobChecks =
        useBufferOps && (emitOobChecks || op.getCanStoreOffEnd());

    if (!useBufferOps) {
      dest = asGlobal(b, dest);
    } else if (useBufferOps && emitOobChecks && coords.empty()) {
      dest = zeroDMemrefAsOneD(b, dest);
      coords.push_back(b.createOrFold<ConstantIndexOp>(loc, 0));
    }
    PatternRewriter::InsertionGuard insertGuard(b);
    if (emitOobChecks && !useBufferOps) {
      Value cond = valid;
      if (op.getCanStoreOffEnd()) {
        Value fallsOffEnd = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::uge, coords[0], numElems);
        cond = b.create<arith::AndIOp>(loc, fallsOffEnd, cond);
      }
      auto guard = b.create<scf::IfOp>(loc, cond, false);
      // This goes to the start because there's alread a terminator.
      b.setInsertionPointToStart(guard.getBody(0));
    }

    if (useBufferOps) {
      if (emitOobChecks) {
        Value zeroConstantOp = b.createOrFold<ConstantIndexOp>(loc, 0);
        for (Value &c : MutableArrayRef<Value>(coords).drop_back())
          c = b.create<arith::SelectOp>(loc, valid, c, zeroConstantOp);
        Value &lastCoord = coords.back();
        lastCoord = b.create<arith::SelectOp>(loc, valid, lastCoord, numElems);
      }
      for (Value &c : MutableArrayRef<Value>(coords))
        c = b.create<IndexCastOp>(loc, b.getI32Type(), c);
    }
    Value origLastCoord = coords.empty() ? nullptr : coords.back();

    if (isAtomic) {
      bool usePackedFp16 = (elemTy.isF16() && (len % 2 == 0));
      int inc = (usePackedFp16 ? 2 : 1);
      Type loadType = (usePackedFp16 ? vectorTypeOrSelf(elemTy, inc) : elemTy);

      for (int64_t i = 0; i < len; i += inc) {
        Value thisSrc = sourceStart;
        if (i > 0)
          thisSrc = b.createOrFold<arith::AddIOp>(
              loc, thisSrc, b.createOrFold<arith::ConstantIndexOp>(loc, i));
        Value data =
            b.create<InBoundsLoadOp>(loc, loadType, op.getSource(), thisSrc);
        if (i > 0) {
          Value offsetConst;
          if (useBufferOps)
            offsetConst = b.createOrFold<arith::ConstantIntOp>(loc, i, 32);
          else
            offsetConst = b.createOrFold<arith::ConstantIndexOp>(loc, i);
          coords.back() =
              b.createOrFold<arith::AddIOp>(loc, origLastCoord, offsetConst);
        }
        if (memoryOp == StoreMethod::AtomicAdd) {
          if (useBufferOps && (usePackedFp16 || elemTy.isF32()))
            b.create<amdgpu::RawBufferAtomicFaddOp>(
                loc, data, dest, coords, useBufferOobChecks, nullptr, nullptr);
          else if (useBufferOps && elemTy.isF16())
            atomicFp16AddAligned(b, loc, data, dest, coords,
                                 useBufferOobChecks);
          else
            b.create<memref::AtomicRMWOp>(loc, AtomicRMWKind::addf, data, dest,
                                          coords);
        } else if (memoryOp == StoreMethod::AtomicMax) {
          makeAtomicFmax(b, loc, data, dest, coords, useBufferOps,
                         useBufferOobChecks);
        } else {
          llvm_unreachable("We don't support this atomic type");
        }
      }
      b.eraseOp(op);
      return success();
    }
    Value data =
        b.create<InBoundsLoadOp>(loc, storeTy, op.getSource(), sourceStart);
    bool nontemporal = op.getNontemporal();
    if (!useBufferOps) {
      if (isa<VectorType>(storeTy))
        b.create<vector::StoreOp>(loc, data, dest, coords, nontemporal);
      else
        b.create<memref::StoreOp>(loc, data, dest, coords, nontemporal);
    } else {
      perHardwareOp(storeTy, [&](int64_t offset, Type thisStoreTy) {
        Value offsetConst = b.createOrFold<arith::ConstantIndexOp>(loc, offset);
        if (offset != 0) {
          Value offsetI32Const =
              b.createOrFold<arith::ConstantIntOp>(loc, offset, 32);
          coords.back() =
              b.create<arith::AddIOp>(loc, origLastCoord, offsetI32Const);
        }
        Value thisData = data;
        if (isa<VectorType>(data.getType()))
          thisData =
              b.create<ExtractSliceOp>(loc, thisStoreTy, data, offsetConst);
        b.create<amdgpu::RawBufferStoreOp>(
            loc, thisData, dest, coords,
            /*boundsCheck=*/(emitOobChecks || op.getCanStoreOffEnd()), nullptr,
            nullptr);
      });
    }
    b.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// InBoundsLoad lowering.
//===----------------------------------------------------------------------===//
struct InBoundsLoadRewritePattern : public OpRewritePattern<InBoundsLoadOp> {
  using OpRewritePattern<InBoundsLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InBoundsLoadOp op,
                                PatternRewriter &b) const override {
    if (auto destType = dyn_cast<VectorType>(op.getResult().getType())) {
      b.replaceOpWithNewOp<vector::TransferReadOp>(
          op, destType, op.getSource(), op.getCoords(),
          /*inbounds=*/ArrayRef<bool>(true));
    } else {
      b.replaceOpWithNewOp<memref::LoadOp>(op, op.getSource(), op.getCoords());
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// InBoundsStore lowering.
//===----------------------------------------------------------------------===//
struct InBoundsStoreRewritePattern : public OpRewritePattern<InBoundsStoreOp> {
  using OpRewritePattern<InBoundsStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InBoundsStoreOp op,
                                PatternRewriter &b) const override {
    if (auto srcType = dyn_cast<VectorType>(op.getData().getType())) {
      b.replaceOpWithNewOp<vector::TransferWriteOp>(
          op, op.getData(), op.getDest(), op.getCoords(),
          /*inbounds=*/ArrayRef<bool>(true));
    } else {
      b.replaceOpWithNewOp<memref::StoreOp>(op, op.getData(), op.getDest(),
                                            op.getCoords());
    }
    return success();
  }
};

void RockSugarToLoopsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  func::FuncOp op = getOperation();
  // Expand transforming_for loops so that the load/store patterns have as much
  // info about validity as possible.
  RewritePatternSet initialLoopPatterns(ctx);
  initialLoopPatterns
      .add<ExtractMultiBufferRewritePattern, TransformingForRewritePattern>(
          ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(initialLoopPatterns))))
    signalPassFailure();

  RewritePatternSet patterns(ctx);
  patterns.add<ExtractSliceRewritePattern, InsertSliceRewritePattern,
               GlobalLoadRewritePattern, GlobalStoreRewritePattern,
               InBoundsLoadRewritePattern, InBoundsStoreRewritePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();

  IRRewriter b(op.getContext());
  // Apply loop invariant code motion to all loops before unrolling.
  WalkResult loopMinimizationResult = op.walk<WalkOrder::PostOrder>(
      [&b](LoopLikeOpInterface loop) -> WalkResult {
        if (succeeded(loop.promoteIfSingleIteration(b)))
          return WalkResult::advance();
        // Affine loops don't implement the iteration promoter interface and
        // need their own method.
        if (auto affineLoop =
                dyn_cast<affine::AffineForOp>(loop.getOperation())) {
          if (succeeded(affine::promoteIfSingleIteration(affineLoop)))
            return WalkResult::advance();
        }
        moveLoopInvariantCode(loop);
        return WalkResult::advance();
      });
  if (loopMinimizationResult.wasInterrupted())
    return signalPassFailure();

  // Note that the reason unrolling is a separate call here is that
  // 1) You can't use loop unrolling from within a pattern rewriter
  // 2) If we make it a seperate pass, canonicizers might remove the
  // forceUnroll attribute we've used
  WalkResult unrollResult =
      op.walk<WalkOrder::PostOrder>([](affine::AffineForOp loop) -> WalkResult {
        Attribute forceUnrollAttr = loop->getAttr("forceUnroll");
        if (!forceUnrollAttr)
          return WalkResult::advance();
        // Since this is a post-order walk through a perfect loop nest, the
        // first loop we see is innermost and therefore unrollable
        if (failed(mlir::affine::loopUnrollFull(loop)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });

  if (unrollResult.wasInterrupted())
    return signalPassFailure();

  // Expand index_diff_update after unrolling since index diffs depend so
  // heeavily on having constant diffs.

  // TODO(kdrewnia): At each level of the loop nest, create an index_diff_update
  // for each coordinate

  // Note: even if all these patterns are moved before unrolling, a call to
  // applyPatternsAndFoldGreedily() is needed for the Fold part of that
  // function. Specifically, affine loop unrolling generates affine.apply()
  // calls that are then constant-folded away by this rewriter
  RewritePatternSet postUnrollPatterns(ctx);
  postUnrollPatterns.add<IndexDiffUpdateRewritePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(postUnrollPatterns))))
    signalPassFailure();
}
} // end anonymous namespace
