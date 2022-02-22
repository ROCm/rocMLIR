//===- ExpandShorthand.cpp - Expand indexing shorthand ops  -------===//
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

#include "PassDetail.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MIOpen/AffineMapHelper.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/utility/builderUtils.h"
#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <numeric>

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::miopen;

namespace {
struct MIOpenExpandShorthandPass
    : public MIOpenExpandShorthandPassBase<MIOpenExpandShorthandPass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// IndexDiffUpdate lowering.
//===----------------------------------------------------------------------===//

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
    TransformMapAttr transformMap = op.map();

    Operation::operand_range upperIndicesDiff = op.upperDiffs();
    Operation::operand_range lowerIndicesOriginal = op.lowerOrig();

    // Ensure index_diff_update is lowered in def-use order
    bool reevaluateOps = false;
    do {
      reevaluateOps = false;
      for (Value v : op->getOperands()) {
        if (auto pred =
                dyn_cast_or_null<IndexDiffUpdateOp>(v.getDefiningOp())) {
          if (failed(matchAndRewrite(pred, b)))
            return failure();
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

    // llvm::errs() << "Transform metadata:\n";
    // llvm::errs() << transformMetadata << "\n";
    // llvm::errs() << "Upper indices diff size: "
    //              << upperIndicesDiff.size() << "\n";
    // llvm::errs() << "Lower indices original size: "
    //              << lowerIndicesOriginal.size() << "\n\n";

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
      if (mbDiffConst.hasValue()) {
        int64_t diff = mbDiffConst.getValue();
        if (diff == 0) {
          return original;
        }
        auto mbOriginalConst = isConstantValue(original);
        if (mbOriginalConst.hasValue()) {
          return b.create<ConstantIndexOp>(loc,
                                           diff + mbOriginalConst.getValue());
        }
      }
      return b.create<AddIOp>(loc, original, diff);
    };

    // Iterate through all transformations specified in g.
    for (auto mapping : transformMap.getOps()) {
      // llvm::errs() << "f: " << f << "\n";

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
          if (mbUpperDiff.hasValue() && mbLowerDiff.hasValue()) {
            lowerDiff = b.create<ConstantIndexOp>(
                loc,
                mbLowerDiff.getValue() + coefficient * mbUpperDiff.getValue());
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
          if (mbUpperDiff.hasValue() && mbLowerDiff.hasValue()) {
            lowerDiff = b.create<ConstantIndexOp>(
                loc,
                mbUpperDiff.getValue() + coefficient * mbLowerDiff.getValue());
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
        if (mbUpperDiffVal.hasValue()) {
          // In case upper level diff is a constant, use constantFold.
          int64_t upperDiff = mbUpperDiffVal.getValue();

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
              expandAffineMap(b, loc, affineMap, upperDiffModified).getValue();
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
            if (mbConstantIndex.hasValue() && mbConstantDiff.hasValue()) {
              int64_t index = mbConstantIndex.getValue();
              int64_t diff = mbConstantDiff.getValue();
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
            if (mbConstantDiff.getValueOr(-1L) == 0) {
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
    Value base = op.coord();
    if (auto destType = op.result().getType().dyn_cast<VectorType>()) {
      int64_t size = destType.getNumElements();
      Value ret = createZeroConstantOp(b, loc, destType);
      for (int64_t i = 0; i < size; ++i) {
        Value cDest = b.createOrFold<ConstantIndexOp>(loc, i);
        Value cSrc = b.createOrFold<AddIOp>(loc, base, cDest);
        Value v = b.create<vector::ExtractElementOp>(loc, op.vector(), cSrc);
        ret = b.create<vector::InsertElementOp>(loc, v, ret, cDest);
      }
      b.replaceOp(op, ret);
    } else {
      b.replaceOpWithNewOp<vector::ExtractElementOp>(op, op.vector(), base);
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
    Value base = op.coord();
    if (auto srcType = op.source().getType().dyn_cast<VectorType>()) {
      int64_t size = srcType.getNumElements();
      Value ret = op.dest();
      for (int64_t i = 0; i < size; ++i) {
        Value cSrc = b.createOrFold<ConstantIndexOp>(loc, i);
        Value cDest = b.createOrFold<AddIOp>(loc, base, cSrc);
        Value v = b.create<vector::ExtractElementOp>(loc, op.source(), cSrc);
        ret = b.create<vector::InsertElementOp>(loc, v, ret, cDest);
      }
      b.replaceOp(op, ret);
    } else {
      b.replaceOpWithNewOp<vector::InsertElementOp>(op, op.source(), op.dest(),
                                                    base);
    }
    return success();
  }
};

void MIOpenExpandShorthandPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<IndexDiffUpdateRewritePattern, ExtractSliceRewritePattern,
               InsertSliceRewritePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace

std::unique_ptr<Pass> mlir::miopen::createMIOpenExpandShorthandPass() {
  return std::make_unique<MIOpenExpandShorthandPass>();
}
