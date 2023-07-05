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

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
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
            lowerDiffModified.push_back(
                b.create<ConstantIndexOp>(loc, lowerDiffModifiedAttr[iter]
                                                   .template cast<IntegerAttr>()
                                                   .getInt()));
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

/// Return the number of elements in `memref`, accounting for the possibility
/// of dynamic shapes.
static Value computeMemRefNumElements(OpBuilder &b, Location loc,
                                      Value memref) {
  auto type = memref.getType().cast<MemRefType>();
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

struct BufferLoadRewritePattern : public OpRewritePattern<BufferLoadOp> {
  using OpRewritePattern<BufferLoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(BufferLoadOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    Value source = op.getSource();
    Value valid = op.getValid();

    Type loadedType = op.getResult().getType();
    SmallVector<Value, 5> coords(op.getCoords());

    llvm::APInt validConst = APInt::getZero(1);
    bool needOobChecks =
        !coords.empty() && (!matchPattern(valid, m_ConstantInt(&validConst)) ||
                            validConst.isZero());
    if (needOobChecks) {
      Value zeroConstantOp = b.createOrFold<ConstantIndexOp>(loc, 0);
      Value numElements = computeMemRefNumElements(b, loc, source);
      for (Value &c : MutableArrayRef<Value>(coords).drop_back()) {
        c = b.create<arith::SelectOp>(loc, valid, c, zeroConstantOp);
      }
      Value &lastCoord = coords.back();
      lastCoord = b.create<arith::SelectOp>(loc, valid, lastCoord, numElements);
    }

    // Emit load instruction
    // use buffer load since the source memref is on address space 0
    SmallVector<Value, 5> coordsI32;
    for (auto v : coords)
      coordsI32.push_back(b.create<IndexCastOp>(loc, b.getI32Type(), v));
    IntegerAttr indexOffset =
        llvm::transformOptional(op.getOffset(),
                                [&b](const APInt &offset) -> IntegerAttr {
                                  return b.getI32IntegerAttr(
                                      offset.getZExtValue());
                                })
            .value_or(IntegerAttr());
    bool needHardwareOob = needOobChecks || op.getOobIsOverflow();
    b.replaceOpWithNewOp<amdgpu::RawBufferLoadOp>(
        op, loadedType, source, coordsI32, /*boundsCheck=*/needHardwareOob,
        indexOffset,
        /*sgprOffset=*/nullptr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BufferStore lowering.
//===----------------------------------------------------------------------===//
struct BufferStoreRewritePattern : public OpRewritePattern<BufferStoreOp> {
  using OpRewritePattern<BufferStoreOp>::OpRewritePattern;

  // This function creates integer buffer atomic max ops that is numerically
  // identical to floating point integer buffer atomic max ops via controlled
  // type punning. This is done mainly because gfx9 GPUs lack buffer.atomic.fmax
  // intrinsic support.
  LogicalResult createAtomicFMaxUsingIntegerAtomics(
      BufferStoreOp op, Value stVal, Value destMemRef,
      SmallVectorImpl<Value> &coordsI32, bool needHWBoundsCheck,
      IntegerAttr indexOffset, PatternRewriter &b) const {
    if (!stVal.getType().isF32()) {
      return op.emitError(
          "for atomic fmax we only currently support f32 types");
    }
    Location loc = op.getLoc();
    auto stValIntCasted = b.create<LLVM::BitcastOp>(loc, b.getI32Type(), stVal);
    Value zeroConstantOp = b.createOrFold<ConstantIntOp>(loc, 0, 32);
    Value signbitConstantOp =
        b.createOrFold<ConstantIntOp>(loc, 0x80000000, 32);
    Value sign =
        b.create<arith::AndIOp>(loc, signbitConstantOp, stValIntCasted);
    auto isPos = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                         zeroConstantOp, sign);

    scf::IfOp ifb = b.create<scf::IfOp>(loc, isPos, /*withElseRegion=*/true);
    {
      OpBuilder thenb = ifb.getThenBodyBuilder(b.getListener());
      // If the current value is positive doing a signed max integer comparison
      // will be equivalent to a floating point max comparison.
      thenb.create<amdgpu::RawBufferAtomicSmaxOp>(
          loc, stValIntCasted, destMemRef, coordsI32,
          /*boundsCheck=*/needHWBoundsCheck,
          /*indexOffset=*/indexOffset,
          /*sgprOffset=*/nullptr);
    }
    {
      OpBuilder elseb = ifb.getElseBodyBuilder(b.getListener());
      // If the current value is negative doing a unsigned min integer
      // comparison will be equivalent to a floating point max comparison
      // because signed bit will always make negative number larger. Moreover,
      // if both are negative, then smaller of those unsigend integers should be
      // the larger floating point value.
      elseb.create<amdgpu::RawBufferAtomicUminOp>(
          loc, stValIntCasted, destMemRef, coordsI32,
          /*boundsCheck=*/needHWBoundsCheck,
          /*indexOffset=*/indexOffset,
          /*sgprOffset=*/nullptr);
    }
    return success();
  }

  LogicalResult
  createAtomicFMax(BufferStoreOp op, Value stVal, Value destMemRef,
                   SmallVectorImpl<Value> &coordsI32, bool needHWBoundsCheck,
                   IntegerAttr indexOffset, PatternRewriter &b) const {
    Location loc = op.getLoc();
    bool hasAtomicFmaxF32 =
        bitEnumContainsAll(op.getFeatures(), GemmFeatures::atomic_fmax_f32);
    if (hasAtomicFmaxF32) {
      b.create<amdgpu::RawBufferAtomicFmaxOp>(loc, stVal, destMemRef, coordsI32,
                                              /*boundsCheck=*/needHWBoundsCheck,
                                              /*indexOffset=*/indexOffset,
                                              /*sgprOffset=*/nullptr);
    } else {
      LogicalResult result = createAtomicFMaxUsingIntegerAtomics(
          op, stVal, destMemRef, coordsI32, needHWBoundsCheck, indexOffset, b);
      if (failed(result)) {
        return result;
      }
    }
    return success();
  }

  LogicalResult matchAndRewrite(BufferStoreOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    Value data = op.getData();
    Value dest = op.getDest();
    Value valid = op.getValid();

    SmallVector<Value, 5> coords(op.getCoords());

    llvm::APInt validConst = APInt::getZero(1);
    bool needOobChecks =
        !coords.empty() && (!matchPattern(valid, m_ConstantInt(&validConst)) ||
                            validConst.isZero());
    if (needOobChecks) {
      Value zeroConstantOp = b.createOrFold<ConstantIndexOp>(loc, 0);
      Value numElements = computeMemRefNumElements(b, loc, dest);
      for (Value &c : MutableArrayRef<Value>(coords).drop_back()) {
        c = b.create<arith::SelectOp>(loc, valid, c, zeroConstantOp);
      }
      Value &lastCoord = coords.back();
      lastCoord = b.create<arith::SelectOp>(loc, valid, lastCoord, numElements);
    }

    StoreMethod memoryOp = op.getStoreMethod();
    SmallVector<Value, 5> coordsI32;
    for (Value v : coords)
      coordsI32.push_back(b.create<IndexCastOp>(loc, b.getI32Type(), v));
    IntegerAttr indexOffset =
        llvm::transformOptional(op.getOffset(),
                                [&b](const APInt &offset) -> IntegerAttr {
                                  return b.getI32IntegerAttr(
                                      offset.getZExtValue());
                                })
            .value_or(IntegerAttr());

    bool needHardwareOob = needOobChecks || op.getOobIsOverflow();
    if (memoryOp == StoreMethod::AtomicAdd) {
      // TODO: test padding in atomic add kernels now that we can oob with them
      if (auto dataVector = data.getType().dyn_cast<VectorType>()) {
        int32_t nAtomics = dataVector.getNumElements();
        int32_t offset = llvm::transformOptional(op.getOffset(),
                                                 [](const APInt &v) -> int32_t {
                                                   return v.getZExtValue();
                                                 })
                             .value_or(0);
        for (int32_t i = 0; i < nAtomics; ++i) {
          Value item = b.create<vector::ExtractElementOp>(
              loc, data, b.create<ConstantIndexOp>(loc, i));
          b.create<amdgpu::RawBufferAtomicFaddOp>(
              loc, item, dest, coordsI32, /*boundsCheck=*/needHardwareOob,
              /*indexOffset=*/b.getI32IntegerAttr(i + offset),
              /*sgprOffset=*/nullptr);
        }
        b.eraseOp(op);
      } else {
        b.replaceOpWithNewOp<amdgpu::RawBufferAtomicFaddOp>(
            op, data, dest, coordsI32, /*boundsCheck=*/needHardwareOob,
            indexOffset,
            /*sgprOffset=*/nullptr);
      }
    } else if (memoryOp == StoreMethod::AtomicMax) {
      if (auto dataVector = data.getType().dyn_cast<VectorType>()) {
        int32_t nAtomics = dataVector.getNumElements();
        int32_t offset = llvm::transformOptional(op.getOffset(),
                                                 [](const APInt &v) -> int32_t {
                                                   return v.getZExtValue();
                                                 })
                             .value_or(0);
        for (int32_t i = 0; i < nAtomics; ++i) {
          Value item = b.create<vector::ExtractElementOp>(
              loc, data, b.create<ConstantIndexOp>(loc, i));
          LogicalResult result =
              createAtomicFMax(op, item, dest, coordsI32, needHardwareOob,
                               b.getI32IntegerAttr(i + offset), b);
          if (failed(result)) {
            return result;
          }
        }
        b.eraseOp(op);
      } else {
        LogicalResult result = createAtomicFMax(
            op, data, dest, coordsI32, needHardwareOob, indexOffset, b);
        if (failed(result)) {
          return result;
        }
        b.eraseOp(op);
      }
    } else {
      b.replaceOpWithNewOp<amdgpu::RawBufferStoreOp>(
          op, data, dest, coordsI32, /*boundsCheck=*/needHardwareOob,
          indexOffset,
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
                      std::optional<ArrayRef<uint32_t>> lanePerm) const {
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

    std::optional<ArrayRef<uint32_t>> maybeInGroupPerm = std::nullopt;
    if (inGroupPermAttr != b.getI32ArrayAttr({0, 1, 2, 3})) {
      maybeInGroupPerm = inGroupPerm;
    }

    Value rotatedRight = emitRotations(loc, b, vector, laneId, Right, groupSize,
                                       totalSize, std::nullopt);
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
  // Expand transforming_for loops so that the load/store patterns have as much
  // info about validity as possible.
  RewritePatternSet initialLoopPatterns(ctx);
  initialLoopPatterns.add<TransformingForRewritePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(initialLoopPatterns))))
    signalPassFailure();

  RewritePatternSet patterns(ctx);
  patterns.add<ExtractSliceRewritePattern, InsertSliceRewritePattern,
               BufferLoadRewritePattern, BufferStoreRewritePattern,
               InBoundsLoadRewritePattern, InBoundsStoreRewritePattern,
               InWarpTransposeRewritePattern>(ctx);
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
