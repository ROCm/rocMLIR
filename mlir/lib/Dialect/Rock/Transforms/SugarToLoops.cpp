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
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/AMDGPU/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
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
    // Compute the initial output values of the lower coordinates.
    // In the case of an index diff map-based loop, compute all intermediate
    // results. When there are no index diff maps, use the composed affine map
    SmallVector<AffineMap, 2> composedMaps;
    SmallVector<SmallVector<SmallVector<Value, 8>, 2>, 2> lowerInits;
    for (uint32_t i = 0; i < nDomains; ++i) {
      SmallVector<SmallVector<Value, 8>, 2> lowerInit;
      ArrayAttr transforms = op.getTransforms(i);
      if (transforms.empty()) {
        SmallVector<Value, 8> init;
        llvm::copy(op.getUpperInits(i), std::back_inserter(init));
        lowerInit.push_back(std::move(init));
        composedMaps.push_back({}); // don't throw off composed maps count
      } else if (useDiffs) {
        for (auto t : transforms.getAsRange<TransformMapAttr>()) {
          AffineMap map = t.getMap().getAffineMap();
          Optional<SmallVector<Value, 8>> init;
          if (lowerInit.size() == 0)
            init = expandAffineMap(b, loc, map, op.getUpperInits(i));
          else
            init =
                expandAffineMap(b, loc, map, lowerInit[lowerInit.size() - 1]);
          if (!init)
            return failure();
          lowerInit.push_back(std::move(*init));
        }
      } else {
        AffineMap composed = composeTransforms(transforms);
        composedMaps.push_back(composed);
        Optional<SmallVector<Value, 8>> init =
            expandAffineMap(b, loc, composed, op.getUpperInits(i));
        if (!init.has_value())
          return failure();
        lowerInit.push_back(std::move(*init));
      }
      lowerInits.push_back(lowerInit);
    }

    // Having done pre-computation, create an affine loop nest over the upper
    // rectangle. This'll be unrolled as needed.
    llvm::SmallVector<AffineForOp, 5> loops;
    llvm::SmallVector<Value, 5> ivs;
    OpBuilder ilb = b;
    for (const auto &pair : llvm::zip(bounds, strides)) {
      int64_t bound, stride;
      std::tie(bound, stride) = pair;
      llvm::SmallVector<Value, 3> iterInits;
      if (loops.empty())
        llvm::copy(op.getIterInits(), std::back_inserter(iterInits));
      else
        llvm::copy(loops[loops.size() - 1].getRegionIterArgs(),
                   std::back_inserter(iterInits));
      auto loop = ilb.create<AffineForOp>(loc, 0, bound, stride, iterInits);
      ivs.push_back(loop.getInductionVar());
      if (iterInits
              .empty()) // remove default affine.yield for cleaner code later
        b.eraseOp(loop.getBody()->getTerminator());
      ilb = OpBuilder::atBlockBegin(loop.getBody(), ilb.getListener());
      loops.push_back(loop);
    }

    // Create code to actually transform the coordinates
    BlockAndValueMapping cloneMap;
    for (uint32_t i = 0; i < nDomains; ++i) {
      Block::BlockArgListType lower = op.getLowerCoords(i);
      ArrayAttr transforms = op.getTransforms(i);
      if (!useDiffs || transforms.empty()) {
        llvm::SmallVector<Value, 5> stepped;
        for (auto p : llvm::zip(op.getUpperInits(i), ivs)) {
          stepped.push_back(
              ilb.create<AddIOp>(loc, std::get<0>(p), std::get<1>(p)));
        }
        if (!transforms.empty()) {
          Optional<SmallVector<Value, 8>> transformed =
              expandAffineMap(ilb, loc, composedMaps[i], stepped);
          if (!transformed)
            return failure();
          stepped.clear();
          stepped.assign(std::move(*transformed));
        }
        for (auto p : llvm::zip(lower, stepped)) {
          cloneMap.map(std::get<0>(p), std::get<1>(p));
        }
      } else { // index diff maps
        IndexDiffUpdateOp lastDiff;
        for (auto p : llvm::zip(transforms.getAsRange<TransformMapAttr>(),
                                lowerInits[i])) {
          TransformMapAttr t = std::get<0>(p);
          SmallVector<Value, 8> &lowerInit = std::get<1>(p);
          if (!lastDiff)
            lastDiff = ilb.create<IndexDiffUpdateOp>(loc, t, ivs, lowerInit);
          else
            lastDiff = ilb.create<IndexDiffUpdateOp>(
                loc, t, lastDiff.getLowerDiff(), lowerInit);
        }
        for (auto p : llvm::zip(lower, lastDiff.getLowerIndices())) {
          cloneMap.map(std::get<0>(p), std::get<1>(p));
        }
      }
    }

    // Map loop arguments, clone operations in body
    AffineForOp il = loops[loops.size() - 1];
    for (auto p : llvm::zip(op.getIterArgs(), il.getRegionIterArgs())) {
      cloneMap.map(std::get<0>(p), std::get<1>(p));
    }
    for (Operation &bodyOp : op.getBody()->getOperations()) {
      if (auto yield = dyn_cast<rock::YieldOp>(bodyOp)) {
        llvm::SmallVector<Value, 3> terminatorArgs;
        for (Value v : op.getBody()->getTerminator()->getOperands()) {
          terminatorArgs.push_back(cloneMap.lookupOrDefault(v));
        }
        ilb.create<AffineYieldOp>(loc, terminatorArgs);
      } else {
        ilb.clone(bodyOp, cloneMap);
      }
    }

    if (loops.size() > 1) {
      for (size_t i = 0, e = loops.size() - 1; i < e; ++i) {
        AffineForOp inner = loops[i + 1];
        OpBuilder lb =
            OpBuilder::atBlockEnd(loops[i].getBody(), b.getListener());
        lb.create<AffineYieldOp>(loc, inner.getResults());
      }
    }

    b.replaceOp(op, loops[0].getResults());
    // Note: the unrolling process doesn't play nice with pattern rewrites
    // Therefore, we just mark loops for unrolling and deal with it in a
    // separate pass
    if (unroll)
      for (AffineForOp loop : loops)
        loop->setAttr("forceUnroll", b.getUnitAttr());

    return success();
  }
};

// Determine if the operation provided is a constant, and return its value if it
// is
Optional<int64_t> isConstantValue(Value v) {
  auto *op = v.getDefiningOp();
  if (nullptr == op)
    return llvm::None;
  while (auto cast = dyn_cast<IndexCastOp>(op)) {
    op = cast.getIn().getDefiningOp();
  }
  if (auto intOp = dyn_cast<ConstantIntOp>(op)) {
    return intOp.value();
  }
  if (auto indexOp = dyn_cast<ConstantIndexOp>(op)) {
    return indexOp.value();
  }
  return llvm::None;
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
      } else if ((transformation == TransformType::Merge) ||
                 (transformation == TransformType::Unfold)) {
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
            lowerDiffModified.push_back(
                b.create<ConstantIndexOp>(loc, lowerDiffModifiedAttr[iter]
                                                   .template cast<IntegerAttr>()
                                                   .getInt()));
          }
          assert(lowerDiffModified.size() == lowerIndicesOriginal.size());
        } else {
          // In case upper level diff is not constant, use expandAffineMap.

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
          // expandAffineMap.
          lowerDiffModified =
              expandAffineMap(b, loc, affineMap, upperDiffModified).value();
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

        // Add carry check for Merge.
        // For Unfold it's not needed.
        if (transformation == TransformType::Merge) {
          // Carry checked lower indices.
          // FIXME: study how to properly lowerDiffsCarryChecked.
          DenseMap<uint32_t, Value> lowerDiffsCarryChecked;
          DenseMap<uint32_t, Value> lowerIndicesCarryChecked;
          for (uint32_t iter = 0; iter < q.size(); ++iter) {
            int64_t lowerDim = q[iter];
            lowerDiffsCarryChecked[lowerDim] = lowerDiffs[iter];
            lowerIndicesCarryChecked[lowerDim] = lowerIndicesModified[iter];
          }
          assert(lowerDiffsCarryChecked.size() == lowerIndicesModified.size());
          assert(lowerIndicesCarryChecked.size() ==
                 lowerIndicesModified.size());

          // We only implement carry logic. Borrow logic would never happen as
          // upper index diffs would always be positive in the current
          // algorithm.
          Value overflowOp = zeroConstantOp;
          for (ssize_t iter = q.size() - 1; iter >= 0; --iter) {
            uint32_t lowerDim = q[iter];
            int64_t upperBound = e[iter];
            // If the overflow is statically 0, nothing gets added
            Value diff =
                addToOriginal(lowerDiffsCarryChecked[lowerDim], overflowOp);
            Value index =
                addToOriginal(lowerIndicesCarryChecked[lowerDim], overflowOp);

            // Don't generate overflow for the uppermost dimension,
            // as this can lead to adresses wrapping back into bounds
            if (iter == 0) {
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
          assert(lowerIndicesCarryChecked.size() ==
                 lowerIndicesModified.size());
          lowerDiffs.clear();
          lowerIndicesModified.clear();
          for (uint32_t iter = 0; iter < q.size(); ++iter) {
            uint32_t lowerDim = q[iter];
            lowerDiffs.push_back(lowerDiffsCarryChecked[lowerDim]);
            lowerIndicesModified.push_back(lowerIndicesCarryChecked[lowerDim]);
          }
          assert(lowerDiffs.size() == q.size());
          assert(lowerIndicesModified.size() == q.size());
        }

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
    if (auto destType = op.getResult().getType().dyn_cast<VectorType>()) {
      if (destType == op.getVector().getType().cast<VectorType>()) {
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
    if (auto srcType = op.getSource().getType().dyn_cast<VectorType>()) {
      if (srcType == op.getDest().getType().cast<VectorType>()) {
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
// BufferLoad lowering.
//===----------------------------------------------------------------------===//
// TODO(kdrewnia): use "OOB reads = 0" from hardware to remove
// hardcoded zero value
struct BufferLoadRewritePattern : public OpRewritePattern<BufferLoadOp> {
  using OpRewritePattern<BufferLoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(BufferLoadOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    Value source = op.getSource();
    auto sourceType = source.getType().cast<MemRefType>();
    ArrayRef<int64_t> sourceShape = sourceType.getShape();
    int64_t sourceNumElems = sourceType.getNumElements();
    SmallVector<int64_t, 5> sourceStrides;
    int64_t sourceOffset;
    if (failed(getStridesAndOffset(sourceType, sourceStrides, sourceOffset))) {
      return op.emitOpError("Somehow we don't have static strides\n");
    }

    Type loadedType = op.getResult().getType();
    SmallVector<Value, 5> coords;
    coords.reserve(op.getCoords().size());
    llvm::copy(op.getCoords(), std::back_inserter(coords));

    Value zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);

    Value falseOp = b.createOrFold<ConstantIntOp>(loc, 0, b.getI1Type());

    llvm::SmallDenseSet<uint32_t> leftOob, rightOob;
    for (llvm::APInt leftOobDim :
         op.getLeftOobDims().getAsValueRange<IntegerAttr>())
      leftOob.insert(leftOobDim.getZExtValue());
    for (llvm::APInt rightOobDim :
         op.getRightOobDims().getAsValueRange<IntegerAttr>())
      rightOob.insert(rightOobDim.getZExtValue());

    // If a coordinate is out of bounds, set that coordinate to the number of
    // elements in the buffer over the stride in that dimension, ensuring
    // we get an out of bounds store
    for (uint32_t i = 0, e = coords.size(); i < e; ++i) {
      // oob checks on the right for dimension 0 are already handled by the
      // buffer intrinsic
      Value isOob = falseOp;
      if (rightOob.contains(i) && i != 0) {
        Value test = b.create<CmpIOp>(
            loc, CmpIPredicate::sge, coords[i],
            b.createOrFold<ConstantIndexOp>(loc, sourceShape[i]));
        isOob = b.createOrFold<OrIOp>(loc, test, isOob);
      }
      if (leftOob.contains(i)) {
        Value test = b.create<CmpIOp>(loc, CmpIPredicate::slt, coords[i],
                                      zeroConstantOp);
        isOob = b.createOrFold<OrIOp>(loc, test, isOob);
      }
      if (isOob != falseOp) {
        Value oobConst =
            b.create<ConstantIndexOp>(loc, sourceNumElems / sourceStrides[i]);
        coords[i] = b.create<SelectOp>(loc, isOob, oobConst, coords[i]);
      }
    }

    // Emit load instruction
    // use buffer load since the source memref is on address space 0
    SmallVector<Value, 5> coordsI32;
    for (auto v : coords)
      coordsI32.push_back(b.create<IndexCastOp>(loc, b.getI32Type(), v));
    IntegerAttr indexOffset =
        op.getOffset()
            .transform([&b](const APInt &offset) -> IntegerAttr {
              return b.getI32IntegerAttr(offset.getZExtValue());
            })
            .value_or(IntegerAttr());
    b.replaceOpWithNewOp<amdgpu::RawBufferLoadOp>(
        op, loadedType, source, coordsI32, /*boundsCheck=*/true, indexOffset,
        /*sgprOffset=*/nullptr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BufferStore lowering.
//===----------------------------------------------------------------------===//
struct BufferStoreRewritePattern : public OpRewritePattern<BufferStoreOp> {
  using OpRewritePattern<BufferStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferStoreOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    Value data = op.getData();
    Value dest = op.getDest();
    auto destType = dest.getType().cast<MemRefType>();
    ArrayRef<int64_t> destShape = destType.getShape();
    int64_t destNumElems = destType.getNumElements();
    SmallVector<int64_t, 5> destStrides;
    int64_t destOffset;
    if (failed(getStridesAndOffset(destType, destStrides, destOffset))) {
      return op.emitOpError("Somehow we don't have static strides\n");
    }

    SmallVector<Value, 5> coords;
    coords.reserve(op.getCoords().size());
    llvm::copy(op.getCoords(), std::back_inserter(coords));

    Value zeroConstantOp = b.createOrFold<ConstantIndexOp>(loc, 0);
    Value falseOp = b.createOrFold<ConstantIntOp>(loc, 0, b.getI1Type());

    llvm::SmallDenseSet<uint32_t> leftOob, rightOob;
    for (llvm::APInt leftOobDim :
         op.getLeftOobDims().getAsValueRange<IntegerAttr>())
      leftOob.insert(leftOobDim.getZExtValue());
    for (llvm::APInt rightOobDim :
         op.getRightOobDims().getAsValueRange<IntegerAttr>())
      rightOob.insert(rightOobDim.getZExtValue());

    // If a coordinate is out of bounds, set that coordinate to the number of
    // elements in the buffer over the stride in that dimension, ensuring
    // we get an out of bounds store
    for (uint32_t i = 0, e = coords.size(); i < e; ++i) {
      // oob checks on the right for dimension 0 are already handled by the
      // buffer intrinsic
      Value isOob = falseOp;
      if (rightOob.contains(i) && i != 0) {
        Value test = b.create<CmpIOp>(
            loc, CmpIPredicate::sge, coords[i],
            b.createOrFold<ConstantIndexOp>(loc, destShape[i]));
        isOob = b.createOrFold<OrIOp>(loc, test, isOob);
      }
      if (leftOob.contains(i)) {
        Value test = b.create<CmpIOp>(loc, CmpIPredicate::slt, coords[i],
                                      zeroConstantOp);
        isOob = b.createOrFold<OrIOp>(loc, test, isOob);
      }
      if (isOob != falseOp) {
        Value oobConst =
            b.create<ConstantIndexOp>(loc, destNumElems / destStrides[i]);
        coords[i] = b.create<SelectOp>(loc, isOob, oobConst, coords[i]);
      }
    }

    StoreMethod memoryOp = op.getStoreMethod();
    SmallVector<Value, 5> coordsI32;
    for (Value v : coords)
      coordsI32.push_back(b.create<IndexCastOp>(loc, b.getI32Type(), v));
    IntegerAttr indexOffset =
        op.getOffset()
            .transform([&b](const APInt &offset) -> IntegerAttr {
              return b.getI32IntegerAttr(offset.getZExtValue());
            })
            .value_or(IntegerAttr());

    if (memoryOp == StoreMethod::AtomicAdd) {
      // TODO: test padding in atomic add kernels now that we can oob with them
      if (auto dataVector = data.getType().dyn_cast<VectorType>()) {
        int32_t nAtomics = dataVector.getNumElements();
        int32_t offset = op.getOffset()
                             .transform([](const APInt &v) -> int32_t {
                               return v.getZExtValue();
                             })
                             .value_or(0);
        for (int32_t i = 0; i < nAtomics; ++i) {
          Value item = b.create<vector::ExtractElementOp>(
              loc, data, b.create<ConstantIndexOp>(loc, i));
          b.create<amdgpu::RawBufferAtomicFaddOp>(
              loc, item, dest, coordsI32, /*boundsCheck=*/true,
              /*indexOffset=*/b.getI32IntegerAttr(i + offset),
              /*sgprOffset=*/nullptr);
        }
        b.eraseOp(op);
      } else {
        b.replaceOpWithNewOp<amdgpu::RawBufferAtomicFaddOp>(
            op, data, dest, coordsI32, /*boundsCheck=*/true, indexOffset,
            /*sgprOffset=*/nullptr);
      }
    } else {
      b.replaceOpWithNewOp<amdgpu::RawBufferStoreOp>(
          op, data, dest, coordsI32, /*boundsCheck=*/true, indexOffset,
          /*sgprOffset=*/nullptr);
    }
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
    if (auto destType = op.getResult().getType().dyn_cast<VectorType>()) {
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
    if (auto srcType = op.getData().getType().dyn_cast<VectorType>()) {
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

//===----------------------------------------------------------------------===//
// InWarpTranspose lowering.
//===----------------------------------------------------------------------===//
constexpr size_t swizzleGroupSize = InWarpTransposeOp::swizzleGroupSize;
struct InWarpTransposeRewritePattern
    : public OpRewritePattern<InWarpTransposeOp> {
  using OpRewritePattern<InWarpTransposeOp>::OpRewritePattern;

  enum RotationDirection {
    // left rotation by 1 is a b c d -> b c d a
    // right rotation by 1 is a b c d -> d a b c
    Left,
    Right
  };

  // Emit the in-regester rotations needed for an in-register transpose
  //
  // This will rotate the values each line holds by `laneId % groupSize`
  // The emitted code uses a barrel rotator to enable performing these
  // `groupSize` different rotations in O(log(groupSize)) operations Arguments:
  // - `vector`: The vector of values to be rotated
  // - `laneId`: The current lane ID (thread ID % warpSize)
  // - `rotationDir`: whether to rotate left or right
  // - `groupSize` and `totalSize`: the size of the transpose
  // and the total vector length, respectively

  // - `lanePerm` : A mapping of physical lanes to logical lanes in each grou
  // That is, lanePerm[i] tells you where the value in lane i would "normally"
  // be, with all indices modulo the swizzle group size. If empty, the identity
  // map is used. For example, with lanePerm = [0, 2, 1, 3], lanes 1 and 3 will
  // rotate their values by 2 places, as opposed to lanes  2 and 3 Returns: The
  // vector of rotated values
  Value emitRotations(Location loc, PatternRewriter &b, Value vector,
                      Value laneId, RotationDirection dir, uint32_t groupSize,
                      uint32_t totalSize,
                      Optional<ArrayRef<uint32_t>> lanePerm) const {
    assert(totalSize % groupSize == 0 &&
           "block size is divisible by group size");

    uint32_t logGroupSize = llvm::Log2_32_Ceil(groupSize);

    int32_t base = 0, offset = 0, target = 0;
    switch (dir) {
    case Left:
      base = 0;
      offset = 1;
      target = logGroupSize;
      break;
    case Right:
      base = logGroupSize - 1;
      offset = -1;
      target = -1;
      break;
    }

    Value zeroConst = b.create<ConstantIndexOp>(loc, 0);

    llvm::SmallVector<Value, swizzleGroupSize> indexConsts;
    for (uint32_t i = 0; i < totalSize; ++i) {
      indexConsts.push_back(b.create<ConstantIndexOp>(loc, i));
    }

    Value laneInSwizzleGroup;
    if (lanePerm.has_value()) {
      Value groupSizeConst = b.create<ConstantIndexOp>(loc, swizzleGroupSize);
      laneInSwizzleGroup = b.create<RemUIOp>(loc, laneId, groupSizeConst);
    }

    Value result = vector;

    for (int32_t logRotation = base; logRotation != target;
         logRotation += offset) {
      uint32_t rotation = 1 << logRotation;
      Value shouldParticipate;
      if (lanePerm.has_value()) {
        // Non-standard arrangement of rows -> lanes, use longer test
        ArrayRef<uint32_t> theLanePerm = lanePerm.value();
        llvm::SmallVector<Value, swizzleGroupSize> comparisons;
        for (uint32_t i = 0; i < theLanePerm.size(); ++i) {
          if ((theLanePerm[i] & rotation) != 0) {
            Value toTest = b.create<ConstantIndexOp>(loc, i);
            comparisons.push_back(b.create<CmpIOp>(loc, CmpIPredicate::eq,
                                                   laneInSwizzleGroup, toTest));
          }
        }
        if (comparisons.empty()) {
          llvm_unreachable("Permutation on [0, 2^k) didn't have any entries "
                           "with some bit set");
        }
        shouldParticipate =
            std::accumulate(comparisons.begin() + 1, comparisons.end(),
                            comparisons[0], [&b, &loc](Value v1, Value v2) {
                              return b.create<OrIOp>(loc, v1, v2);
                            });
      } else { // The usual case
        Value maskConst = b.create<ConstantIndexOp>(loc, rotation);
        Value shouldParticipateVal = b.create<AndIOp>(loc, laneId, maskConst);
        shouldParticipate = b.create<CmpIOp>(loc, CmpIPredicate::ne,
                                             shouldParticipateVal, zeroConst);
      }

// TODO(kdrewnia): xplicitly emit selects until SWDEV-302607 and SWDEV-302609
// are fixed
#if 0
      scf::IfOp ifb = b.create<scf::IfOp>(
          loc, vector.getType(), shouldParticipate, /*withElseRegion=*/true);
      OpBuilder thenb = ifb.getThenBodyBuilder(b.getListener());

      Value thenResult = result;
      SmallVector<Value> extracted;
      for (uint32_t i = 0; i < totalSize; ++i) {
        extracted.push_back(thenb.create<vector::ExtractElementOp>(
            loc, thenResult, indexConsts[i]));
      }
      for (uint32_t group = 0; group < totalSize; group += groupSize) {
        for (uint32_t i = 0; i < groupSize; ++i) {
          uint32_t dest = 0xdeadbeef;
          switch (dir) {
          case Left:
            // We use groupSize - rotation to prevent underflow
            dest = (i + (groupSize - rotation)) % groupSize;
            break;
          case Right:
            dest = (i + rotation) % groupSize;
            break;
          }
          Value toInsert = extracted[group + i];
          thenResult = thenb.create<vector::InsertElementOp>(
              loc, toInsert, thenResult, indexConsts[group + dest]);
        }
      }
      thenb.create<scf::YieldOp>(loc, thenResult);

      OpBuilder elseb = ifb.getElseBodyBuilder(b.getListener());
      elseb.create<scf::YieldOp>(loc, result);

      result = ifb.getResult(0);
#endif

      SmallVector<Value> extracted;
      for (uint32_t i = 0; i < totalSize; ++i) {
        extracted.push_back(
            b.create<vector::ExtractElementOp>(loc, result, indexConsts[i]));
      }
      for (uint32_t group = 0; group < totalSize; group += groupSize) {
        for (uint32_t i = 0; i < groupSize; ++i) {
          uint32_t dest = 0xdeadbeef;
          switch (dir) {
          case Left:
            // We use groupSize - rotation to prevent underflow
            dest = (i + (groupSize - rotation)) % groupSize;
            break;
          case Right:
            dest = (i + rotation) % groupSize;
            break;
          }
          Value whenRotating = extracted[group + i];
          Value stable = extracted[group + dest];
          Value toInsert =
              b.create<SelectOp>(loc, shouldParticipate, whenRotating, stable);
          result = b.create<vector::InsertElementOp>(loc, toInsert, result,
                                                     indexConsts[group + dest]);
        }
      }
    }

    return result;
  }

  // Before calling this function, we will have emitted rotations so that the
  // group
  //  r[]: 0   1   2   3
  //  t0: 0,0 1,0 2,0 3,0
  //  t1: 0,1 1,1 2,1 3,1
  //  t2: 0,2 1,2 2,2 3,2
  //  t3: 0,3 1,3 2,3 3,3
  // will have become
  //  0,0 1,0 2,0 3,0
  //  3,1 0,1 1,1 2,1
  //  2,2 3,2 0,2 1,2
  //  1,3 2,3 3,3 0,3

  // (plus-minus size changes for other operations).
  // These rotations are the first step in the in-register transpose algorithm
  // as they allow the inter-lane shuffles to be permutation.

  // The goal of this function is to emit code that will lead to the result
  // state
  //  0,0 0,1 0,2 0,3
  //  1,3 1,0 1,1 1,2
  //  2,2 2,3 2,0 2,1
  //  3,1 3,2 3,3 3,0

  Value emitSwizzles(Location loc, PatternRewriter &b, Value vector,
                     uint32_t groupSize, uint32_t totalSize,
                     ArrayRef<uint32_t> inGroupPerm) const {

    llvm::SmallVector<ArrayAttr, swizzleGroupSize> swizzlePerms;

    llvm::SmallVector<int32_t, swizzleGroupSize> perm;
    llvm::SmallVector<uint32_t, swizzleGroupSize> have;
    llvm::SmallVector<uint32_t, swizzleGroupSize> want;
    for (uint32_t r = 0; r < groupSize; ++r) {
      perm.clear();
      have.clear();
      want.clear();

      for (uint32_t t = 0; t < swizzleGroupSize; ++t) {
        // Must correct for, say, 2x2 transpose being a 4 thread x 2 register
        // swizzle
        uint32_t smallGroupDup = groupSize * (t / groupSize);
        uint32_t preSwizzleI =
            (r + (groupSize - t)) % groupSize + smallGroupDup;
        uint32_t preSwizzleJ = t;

        uint32_t expectedThread = inGroupPerm[t];
        uint32_t postSwizzleI = expectedThread;
        uint32_t postSwizzleJ = (r + (groupSize - expectedThread)) % groupSize +
                                groupSize * (expectedThread / groupSize);
        uint32_t preSwizzleElem = preSwizzleJ + swizzleGroupSize * preSwizzleI;
        uint32_t postSwizzleElem =
            postSwizzleJ + swizzleGroupSize * postSwizzleI;
        /*         llvm::dbgs() << "//r = " << r << " t = " << t << ": " <<
           "have ("
                  << preSwizzleI << ", " << preSwizzleJ << ") = " <<
           preSwizzleElem
                  << " want (" << postSwizzleI << ", " << postSwizzleJ << ") = "
                  << postSwizzleElem << "\n"; */
        have.push_back(preSwizzleElem);
        want.push_back(postSwizzleElem);
      }

      for (uint32_t t = 0; t < swizzleGroupSize; ++t) {
        auto *srcElemIter = std::find(have.begin(), have.end(), want[t]);
        assert(srcElemIter != have.end() && "swizzle is not a permutation");
        auto readIdx = srcElemIter - have.begin();
        perm.push_back(readIdx);
      }

      if (perm[0] == 0 && perm[1] == 1 && perm[2] == 2 && perm[3] == 3) {
        swizzlePerms.push_back(b.getI32ArrayAttr({}));
      } else {
        swizzlePerms.push_back(b.getI32ArrayAttr(perm));
      }
    }

    Value result = b.create<vector::BitCastOp>(
        loc,
        VectorType::get(vector.getType().cast<VectorType>().getShape(),
                        b.getI32Type()),
        vector);
    // TODO(kdrewnia): Make this operation variadic and not just vector-valued
    SmallVector<Value> accessConsts;
    SmallVector<Value> initialRegisters;
    for (uint32_t i = 0; i < totalSize; ++i) {
      Value accessConst = b.create<ConstantIndexOp>(loc, i);
      initialRegisters.push_back(
          b.create<vector::ExtractElementOp>(loc, result, accessConst));
      accessConsts.push_back(accessConst);
    }

    SmallVector<Value> swizzledRegisters;
    for (uint32_t i = 0; i < totalSize; ++i) {
      ArrayAttr swizzleSelector = swizzlePerms[i % groupSize];
      if (0 == swizzleSelector.size()) {
        swizzledRegisters.push_back(initialRegisters[i]);
        continue;
      }
      Value swizzled = b.create<gpu::WarpSwizzleOp>(
          loc, b.getI32Type(), initialRegisters[i], swizzleSelector);
      swizzledRegisters.push_back(swizzled);
    }

    for (uint32_t i = 0; i < totalSize; ++i) {
      if (swizzledRegisters[i] != initialRegisters[i]) {
        result = b.create<vector::InsertElementOp>(loc, swizzledRegisters[i],
                                                   result, accessConsts[i]);
      }
    }

    result = b.create<vector::BitCastOp>(loc, vector.getType(), result);
    return result;
  }

  LogicalResult matchAndRewrite(InWarpTransposeOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    Value vector = op.getVector();
    uint32_t totalSize = vector.getType().cast<VectorType>().getNumElements();

    Value laneId = op.getLaneId();
    uint32_t groupSize = op.getSize();

    ArrayAttr inGroupPermAttr = op.getInGroupPerm();
    llvm::SmallVector<uint32_t, swizzleGroupSize> inGroupPerm;
    auto inGroupPermArr = inGroupPermAttr.getValue();
    // ::verify() ensures this is a permutation
    for (uint32_t i = 0; i < swizzleGroupSize; ++i) {
      inGroupPerm.push_back(inGroupPermArr[i]
                                .cast<mlir::IntegerAttr>()
                                .getValue()
                                .getZExtValue());
    }

    Optional<ArrayRef<uint32_t>> maybeInGroupPerm = llvm::None;
    if (inGroupPermAttr != b.getI32ArrayAttr({0, 1, 2, 3})) {
      maybeInGroupPerm = inGroupPerm;
    }

    Value rotatedRight = emitRotations(loc, b, vector, laneId, Right, groupSize,
                                       totalSize, llvm::None);
    Value swizzled =
        emitSwizzles(loc, b, rotatedRight, groupSize, totalSize, inGroupPerm);
    Value rotatedLeft = emitRotations(loc, b, swizzled, laneId, Left, groupSize,
                                      totalSize, maybeInGroupPerm);

    op.replaceAllUsesWith(rotatedLeft);
    op.erase();

    return success();
  }
};

void RockSugarToLoopsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  func::FuncOp op = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.add<TransformingForRewritePattern, ExtractSliceRewritePattern,
               InsertSliceRewritePattern, BufferLoadRewritePattern,
               BufferStoreRewritePattern, InBoundsLoadRewritePattern,
               InBoundsStoreRewritePattern, InWarpTransposeRewritePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();

  // Apply loop invariant code motion to all loops before unrolling
  WalkResult licmResult =
      op.walk<WalkOrder::PostOrder>([](LoopLikeOpInterface loop) -> WalkResult {
        moveLoopInvariantCode(loop);
        return WalkResult::advance();
      });
  if (licmResult.wasInterrupted())
    return signalPassFailure();

  // Note that the reason unrolling is a separate call here is that
  // 1) You can't use loop unrolling from within a pattern rewriter
  // 2) If we make it a seperate pass, canonicizers might remove the
  // forceUnroll attribute we've used
  WalkResult unrollResult =
      op.walk<WalkOrder::PostOrder>([](AffineForOp loop) -> WalkResult {
        Attribute forceUnrollAttr = loop->getAttr("forceUnroll");
        if (!forceUnrollAttr)
          return WalkResult::advance();
        // Since this is a post-order walk through a perfect loop nest, the
        // first loop we see is innermost and therefore unrollable
        if (failed(mlir::loopUnrollFull(loop)))
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
