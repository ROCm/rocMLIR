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

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
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
#include "mlir/IR/ValueRange.h"
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
/*struct TransformingForRewritePattern
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
          affine::AffineForOp::create(b, loc, 0, bound, stride, iterInits);
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
            arith::ConstantIntOp::create(b, loc, b.getI1Type(), true);
        // Start by offsetting the upper inputs.
        for (auto p : llvm::zip(op.getUpperInits(i), ivs)) {
          computed.push_back(
              AddIOp::create(b, loc, std::get<0>(p), std::get<1>(p)));
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
            arith::ConstantIntOp::create(b, loc, b.getI1Type(), true);
        for (const auto &[t, lowerInit] : llvm::zip(
                 transforms.getAsRange<TransformMapAttr>(), lowerInits[i])) {
          if (!lastDiff)
            lastDiff = IndexDiffUpdateOp::create(b, loc, t, ivs, lowerInit);
          else
            lastDiff = IndexDiffUpdateOp::create(
                b, loc, t, lastDiff.getLowerDiff(), lowerInit);
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
        affine::AffineYieldOp::create(b, loc, terminatorArgs);
      } else {
        b.clone(bodyOp, cloneMap);
      }
    }

    if (loops.size() > 1) {
      for (size_t i = 0, e = loops.size() - 1; i < e; ++i) {
        affine::AffineForOp inner = loops[i + 1];
        b.setInsertionPointToEnd(loops[i].getBody());
        affine::AffineYieldOp::create(b, loc, inner.getResults());
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
};*/

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

    Value zeroConstantOp = ConstantIndexOp::create(b, loc, 0);
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
          return ConstantIndexOp::create(b, loc,
                                         diff + mbOriginalConst.value());
        }
      }
      return AddIOp::create(b, loc, original, diff);
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
            lowerDiff = ConstantIndexOp::create(
                b, loc,
                mbLowerDiff.value() + coefficient * mbUpperDiff.value());
          } else {
            lowerDiff = AddIOp::create(
                b, loc, lowerDiff,
                MulIOp::create(b, loc,
                               ConstantIndexOp::create(b, loc, coefficient),
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
            lowerDiff = ConstantIndexOp::create(
                b, loc,
                mbUpperDiff.value() + coefficient * mbLowerDiff.value());
          } else {
            lowerDiff = AddIOp::create(
                b, loc, upperIndicesDiff[upperDim],
                MulIOp::create(b, loc,
                               ConstantIndexOp::create(b, loc, coefficient),
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
            lowerDiffModified.push_back(ConstantIndexOp::create(
                b, loc,
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
                  ConstantIndexOp::create(b, loc, index);
              lowerDiffsCarryChecked[lowerDim] =
                  ConstantIndexOp::create(b, loc, diff);
            } else {
              int64_t carry = index / upperBound;
              int64_t newIndex = index % upperBound;
              int64_t newDiff = diff - (carry * upperBound);
              overflowOp = ConstantIndexOp::create(b, loc, carry);
              lowerIndicesCarryChecked[lowerDim] =
                  ConstantIndexOp::create(b, loc, newIndex);
              lowerDiffsCarryChecked[lowerDim] =
                  ConstantIndexOp::create(b, loc, newDiff);
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

          Value upperBoundOp = ConstantIndexOp::create(b, loc, upperBound);
          Value carry = DivUIOp::create(b, loc, index, upperBoundOp);
          Value newIndex = RemUIOp::create(b, loc, index, upperBoundOp);
          // If the merge is, as is typical, near the end of the
          // transformations this computation should get hit by the dead code
          // eleminator
          Value newDiff = SubIOp::create(
              b, loc, diff, MulIOp::create(b, loc, carry, upperBoundOp));

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
          Value lowerLenOp = ConstantIndexOp::create(b, loc, lowerLen);
          auto mbUpperDiff = isConstantValue(upperIndicesDiff[p[i]]);
          Value wrappedDiff;
          if (mbUpperDiff.has_value()) {
            wrappedDiff =
                ConstantIndexOp::create(b, loc, *mbUpperDiff % lowerLen);
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

void RockSugarToLoopsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  func::FuncOp op = getOperation();
  // TODO(roctriton): transforms to arithmetic operations in loads and stores
  // Expand transforming_for loops so that the load/store patterns have as much
  // info about validity as possible.
  // RewritePatternSet initialLoopPatterns(ctx);
  // initialLoopPatterns
  //     .add<TransformingForRewritePattern>(
  //         ctx);
  // if (failed(applyPatternsGreedily(getOperation(),
  //                                  std::move(initialLoopPatterns))))
  //   signalPassFailure();

  // Expand index_diff_update after unrolling since index diffs depend so
  // heeavily on having constant diffs.

  // TODO(kdrewnia): At each level of the loop nest, create an index_diff_update
  // for each coordinate

  // Note: even if all these patterns are moved before unrolling, a call to
  // applyPatternsGreedily() is needed for the Fold part of that
  // function. Specifically, affine loop unrolling generates affine.apply()
  // calls that are then constant-folded away by this rewriter
  RewritePatternSet postUnrollPatterns(ctx);
  postUnrollPatterns.add<IndexDiffUpdateRewritePattern>(ctx);
  if (failed(applyPatternsGreedily(op, std::move(postUnrollPatterns))))
    signalPassFailure();
}
} // end anonymous namespace
