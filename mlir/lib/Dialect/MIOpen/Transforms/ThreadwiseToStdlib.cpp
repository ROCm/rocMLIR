//===- LowerMIOpenOps.cpp - MLIR MIOpen ops lowering passes ---------------===//
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
// =============================================================================
//
// These passes convert the MIOpen threadwise ops into constructs from the
// rest of MLIR so that they can be lowered to the GPU and LLVM dialects.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MIOpen/AffineMapHelper.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/XdlopsCodeSelection.h"
#include "mlir/Dialect/MIOpen/utility/builderUtils.h"
#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <iterator>
#include <numeric>

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::miopen;

namespace {
struct LowerMIOpenOpsStep4Pass
    : public MIOpenOpsStep4PassBase<LowerMIOpenOpsStep4Pass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// Utility function to emit load instructions for local buffers
//===----------------------------------------------------------------------===//
Value emitLoadLogic(OpBuilder &b, Location loc, Type loadedType,
                    const Value source, ValueRange coords) {
  return b.create<InBoundsLoadOp>(loc, loadedType, source, coords);
}

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
    for (llvm::APInt v : op.bounds().getAsValueRange<IntegerAttr>()) {
      int64_t bound = v.getZExtValue();
      bounds.push_back(bound);
    }

    bool useDiffs = op.useIndexDiffs().getValueOr(false);
    bool unroll = op.forceUnroll().getValueOr(false);

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
        SmallVector<AffineMap, 2> maps;
        for (auto t : transforms.getAsRange<TransformMapAttr>()) {
          maps.push_back(t.getMap().getAffineMap());
        }
        AffineMap composed = composeTransforms(maps);
        composedMaps.push_back(composed);
        Optional<SmallVector<Value, 8>> init =
            expandAffineMap(b, loc, composed, op.getUpperInits(i));
        if (!init.hasValue())
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
    for (int64_t bound : bounds) {
      llvm::SmallVector<Value, 3> iterInits;
      if (loops.empty())
        llvm::copy(op.iterInits(), std::back_inserter(iterInits));
      else
        llvm::copy(loops[loops.size() - 1].getRegionIterArgs(),
                   std::back_inserter(iterInits));
      auto loop = ilb.create<AffineForOp>(loc, 0, bound, 1, iterInits);
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
                loc, t, lastDiff.lowerDiff(), lowerInit);
        }
        for (auto p : llvm::zip(lower, lastDiff.lowerIndices())) {
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
      if (auto yield = dyn_cast<miopen::YieldOp>(bodyOp)) {
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
    Value source = op.source();
    auto sourceType = source.getType().cast<MemRefType>();
    ArrayRef<int64_t> sourceShape = sourceType.getShape();
    int64_t sourceNumElems = sourceType.getNumElements();
    SmallVector<int64_t, 5> sourceStrides;
    int64_t sourceOffset;
    if (failed(getStridesAndOffset(sourceType, sourceStrides, sourceOffset))) {
      return op.emitOpError("Somehow we don't have static strides\n");
    }

    Type loadedType = op.result().getType();
    SmallVector<Value, 5> coords;
    coords.reserve(op.coords().size());
    llvm::copy(op.coords(), std::back_inserter(coords));

    Value zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);

    Value falseOp = b.createOrFold<ConstantIntOp>(loc, 0, b.getI1Type());

    llvm::SmallDenseSet<uint32_t> leftOob, rightOob;
    for (llvm::APInt leftOobDim :
         op.leftOobDims().getAsValueRange<IntegerAttr>())
      leftOob.insert(leftOobDim.getZExtValue());
    for (llvm::APInt rightOobDim :
         op.rightOobDims().getAsValueRange<IntegerAttr>())
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
    b.replaceOpWithNewOp<gpu::GCNRawBufferLoadOp>(
        op, loadedType, source,
        /*targetIsRDNA=*/false, coordsI32, /*boundsCheck=*/true,
        /*indexOffset=*/nullptr, /*sgprOffset=*/nullptr);
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

    Value data = op.data();
    Value dest = op.dest();
    auto destType = dest.getType().cast<MemRefType>();
    ArrayRef<int64_t> destShape = destType.getShape();
    int64_t destNumElems = destType.getNumElements();
    SmallVector<int64_t, 5> destStrides;
    int64_t destOffset;
    if (failed(getStridesAndOffset(destType, destStrides, destOffset))) {
      return op.emitOpError("Somehow we don't have static strides\n");
    }

    SmallVector<Value, 5> coords;
    coords.reserve(op.coords().size());
    llvm::copy(op.coords(), std::back_inserter(coords));

    Value zeroConstantOp = b.createOrFold<ConstantIndexOp>(loc, 0);
    Value falseOp = b.createOrFold<ConstantIntOp>(loc, 0, b.getI1Type());

    llvm::SmallDenseSet<uint32_t> leftOob, rightOob;
    for (llvm::APInt leftOobDim :
         op.leftOobDims().getAsValueRange<IntegerAttr>())
      leftOob.insert(leftOobDim.getZExtValue());
    for (llvm::APInt rightOobDim :
         op.rightOobDims().getAsValueRange<IntegerAttr>())
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

    StoreMethod memoryOp = op.storeMethod().getValueOr(StoreMethod::Set);

    SmallVector<Value, 5> coordsI32;
    for (Value v : coords)
      coordsI32.push_back(b.create<IndexCastOp>(loc, b.getI32Type(), v));

    if (memoryOp == StoreMethod::AtomicAdd) {
      // TODO: test padding in atomic add kernels now that we can oob with them
      if (auto dataVector = data.getType().dyn_cast<VectorType>()) {
        int32_t nAtomics = dataVector.getNumElements();
        for (int32_t i = 0; i < nAtomics; ++i) {
          Value item = b.create<vector::ExtractElementOp>(
              loc, data, b.create<ConstantIndexOp>(loc, i));
          b.create<gpu::GCNRawBufferAtomicFaddOp>(
              loc, item, dest,
              /*targetIsRDNA=*/false, coordsI32, /*boundsCheck=*/true,
              /*indexOffset=*/b.getI32IntegerAttr(i), /*sgprOffset=*/nullptr);
        }
        b.eraseOp(op);
      } else {
        b.replaceOpWithNewOp<gpu::GCNRawBufferAtomicFaddOp>(
            op, data, dest,
            /*targetIsRDNA=*/false, coordsI32, /*boundsCheck=*/true,
            /*indexOffset=*/nullptr, /*sgprOffset=*/nullptr);
      }
    } else {
      b.replaceOpWithNewOp<gpu::GCNRawBufferStoreOp>(
          op, data, dest,
          /*targetIsRDNA=*/false, coordsI32, /*boundsCheck=*/true,
          /*indexOffset=*/nullptr, /*sgprOffset=*/nullptr);
    }
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
    TransformMapAttr transformMap = op.map();

    Operation::operand_range upperIndicesDiff = op.upperDiffs();
    Operation::operand_range lowerIndicesOriginal = op.lowerOrig();

    // Ensure index_diff_update is lowered in def-use order
    bool reevaluateOps = false;
    do {
      reevaluateOps = false;
      for (Value v : op->getOperands()) {
        Operation *defOp = v.getDefiningOp();
        if (auto pred = dyn_cast_or_null<IndexDiffUpdateOp>(defOp)) {
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
// ThreadwiseGemm lowering.
//===----------------------------------------------------------------------===//
struct ThreadwiseGemmRewritePattern
    : public OpRewritePattern<ThreadwiseGemmOp> {
  using OpRewritePattern<ThreadwiseGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ThreadwiseGemmOp op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();

    auto gemmA = op.matrixA();
    auto gemmB = op.matrixB();
    auto gemmC = op.matrixC();
    auto dataType =
        gemmA.getType().template cast<MemRefType>().getElementType();

    ArrayRef<int64_t> gemmAShape =
        gemmA.getType().cast<MemRefType>().getShape();
    ArrayRef<int64_t> gemmBShape =
        gemmB.getType().cast<MemRefType>().getShape();

    assert(gemmAShape.size() == gemmBShape.size());
    assert((gemmAShape.size() == 3) || (gemmAShape.size() == 4));
    if (gemmAShape.size() == 3) {
      // non-KPack path.
      auto loopG = b.create<AffineForOp>(loc, 0, gemmAShape[0]);
      auto lbG = loopG.getBody();
      b.setInsertionPointToStart(lbG);

      auto loopK = b.create<AffineForOp>(loc, 0, gemmAShape[1]);
      auto lbK = loopK.getBody();
      b.setInsertionPointToStart(lbK);

      auto loopM = b.create<AffineForOp>(loopK.getLoc(), 0, gemmAShape[2]);
      auto lbM = loopM.getBody();
      b.setInsertionPointToStart(lbM);

      auto loopN = b.create<AffineForOp>(loc, 0, gemmBShape[2]);
      auto lbN = loopN.getBody();
      b.setInsertionPointToStart(lbN);

      SmallVector<Value, 3> memIndicesKM;
      extractForInductionVars({loopG, loopK, loopM}, &memIndicesKM);
      auto gemmAKM = b.create<AffineLoadOp>(loc, gemmA, memIndicesKM);

      SmallVector<Value, 3> memIndicesKN;
      extractForInductionVars({loopG, loopK, loopN}, &memIndicesKN);
      auto gemmBKN = b.create<AffineLoadOp>(loc, gemmB, memIndicesKN);

      Value mul;
      if (dataType.isa<IntegerType>()) {
        mul = b.create<MulIOp>(loc, dataType, gemmAKM, gemmBKN);
      } else {
        mul = b.create<MulFOp>(loc, dataType, gemmAKM, gemmBKN);
      }
      SmallVector<Value, 3> memIndicesMN;
      extractForInductionVars({loopG, loopM, loopN}, &memIndicesMN);
      auto gemmCMN = b.create<AffineLoadOp>(loc, gemmC, memIndicesMN);

      Value add;
      if (dataType.isa<IntegerType>()) {
        add = b.create<AddIOp>(loc, dataType, mul, gemmCMN);
      } else {
        add = b.create<AddFOp>(loc, dataType, mul, gemmCMN);
      }
      b.create<AffineStoreOp>(loc, add, gemmC, memIndicesMN);
    } else if (gemmAShape.size() == 4) {
      // KPack path.
      auto loopG = b.create<AffineForOp>(loc, 0, gemmAShape[0]);
      auto lbG = loopG.getBody();
      b.setInsertionPointToStart(lbG);

      auto loopK = b.create<AffineForOp>(loc, 0, gemmAShape[1]);
      auto lbK = loopK.getBody();
      b.setInsertionPointToStart(lbK);

      auto loopM = b.create<AffineForOp>(loopK.getLoc(), 0, gemmAShape[2]);
      auto lbM = loopM.getBody();
      b.setInsertionPointToStart(lbM);

      auto loopKPack = b.create<AffineForOp>(loc, 0, gemmAShape[3]);
      auto lbKPack = loopKPack.getBody();
      b.setInsertionPointToStart(lbKPack);

      auto loopN = b.create<AffineForOp>(loc, 0, gemmBShape[2]);
      auto lbN = loopN.getBody();
      b.setInsertionPointToStart(lbN);

      SmallVector<Value, 4> memIndicesKMKPack;
      extractForInductionVars({loopG, loopK, loopM, loopKPack},
                              &memIndicesKMKPack);
      auto gemmAKMKPack = b.create<AffineLoadOp>(loc, gemmA, memIndicesKMKPack);

      SmallVector<Value, 4> memIndicesKNKPack;
      extractForInductionVars({loopG, loopK, loopN, loopKPack},
                              &memIndicesKNKPack);
      auto gemmBKNKPack = b.create<AffineLoadOp>(loc, gemmB, memIndicesKNKPack);

      Value mul;
      if (dataType.isa<IntegerType>()) {
        mul = b.create<MulIOp>(loc, dataType, gemmAKMKPack, gemmBKNKPack);
      } else {
        mul = b.create<MulFOp>(loc, dataType, gemmAKMKPack, gemmBKNKPack);
      }
      SmallVector<Value, 4> memIndicesMN;
      extractForInductionVars({loopG, loopM, loopN}, &memIndicesMN);
      auto gemmCMN = b.create<AffineLoadOp>(loc, gemmC, memIndicesMN);

      Value add;
      if (dataType.isa<IntegerType>()) {
        add = b.create<AddIOp>(loc, dataType, mul, gemmCMN);
      } else {
        add = b.create<AddFOp>(loc, dataType, mul, gemmCMN);
      }
      b.create<AffineStoreOp>(loc, add, gemmC, memIndicesMN);
    }

    op.erase();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// XdlopsGemmV2 lowering.
//===----------------------------------------------------------------------===//
struct XdlopsGemmV2RewritePattern : public OpRewritePattern<XdlopsGemmV2Op> {
  using OpRewritePattern<XdlopsGemmV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(XdlopsGemmV2Op op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();

    // Obtain critical information.
    int64_t KPack =
        op->hasAttr("kpack")
            ? op->getAttr("kpack").template cast<IntegerAttr>().getInt()
            : 1;
    int64_t M = op->getAttr("m").template cast<IntegerAttr>().getInt();
    int64_t N = op->getAttr("n").template cast<IntegerAttr>().getInt();
    int64_t K = op->getAttr("k").template cast<IntegerAttr>().getInt();
    int64_t MPerWave =
        op->getAttr("m_per_wave").template cast<IntegerAttr>().getInt();
    int64_t NPerWave =
        op->getAttr("n_per_wave").template cast<IntegerAttr>().getInt();

    int64_t ldsOffsetA = op.ldsBufferOffsetA().getSExtValue();
    int64_t ldsOffsetB = op.ldsBufferOffsetB().getSExtValue();

    assert(ldsOffsetA % KPack == 0 &&
           "LDS buffer segment for A is kpack-aligned");
    assert(ldsOffsetB % KPack == 0 &&
           "LDS buffer segment for B is kpack-aligned");
    auto dataType =
        op.matrixA().getType().template cast<MemRefType>().getElementType();

    auto MConstantOp = b.create<ConstantIndexOp>(loc, M);
    auto NConstantOp = b.create<ConstantIndexOp>(loc, N);
    auto KConstantOp = b.create<ConstantIndexOp>(loc, K);

    // The address calculations into the LDS buffer assume that the buffer
    // has type vector<KPack x T>. Then, we convert that into an address
    // in a buffer of Ts through a final multiplicaiton by KPack.
    // However, the LDS buffer offset, which was computed when the buffer was
    // allocated, is an offset into a buffer of T. Therefore, to allow it to
    // easily participate in adress calculations (instead of adding it on at the
    // end) we must divide it by KPack here. Fortunately, this offset will be
    // KPack-alligned and so this is safe
    Value aBase =
        b.create<AddIOp>(loc, op.waveOffsetA(),
                         b.create<ConstantIndexOp>(loc, ldsOffsetA / KPack));
    Value bBase =
        b.create<AddIOp>(loc, op.waveOffsetB(),
                         b.create<ConstantIndexOp>(loc, ldsOffsetB / KPack));

    // Logic to do XDLOPS code selection.
    // llvm::errs() << "Invoke XDLOPS code selection logic:\n";
    // llvm::errs() << "dataType: "; dataType.dump(); llvm::errs() << "\n";
    // llvm::errs() << "MPerWave: " << MPerWave << "\n";
    // llvm::errs() << "NPerWave: " << NPerWave << "\n";

    XdlopsCodeSelection xcs =
        XdlopsCodeSelection::get(dataType, MPerWave, NPerWave, b);

    // Extract values from XdlopsCodeSelection.
    StringRef mfmaInstr = xcs.mfmaInstr;
    int64_t MPerXdlops = xcs.MPerXdlops;
    int64_t NPerXdlops = xcs.NPerXdlops;
    int64_t MRepeats = xcs.MRepeats;
    int64_t NRepeats = xcs.NRepeats;
    VectorType vectorType = xcs.vectorType;
    int64_t vectorNumber = xcs.vectorNumber;
    SmallVector<SmallVector<unsigned, 3>, 2> imms = xcs.imms;
    Type argType = xcs.argType;

    int64_t num_threads_blk = xcs.num_threads_blk;
    int64_t wave_size = xcs.wave_size;
    int64_t num_input_blks = xcs.num_input_blks;
    int64_t num_output_blks = xcs.num_output_blks;
    int64_t k_base = xcs.k_base;

    bool IsKReduction = (num_output_blks == 1) && (num_input_blks > 1);

    // Original C++ logic.
    // const index_t laneId = get_thread_local_1d_id() % mfma_type.wave_size;
    // FloatA a[K * MRepeats];
    // FloatB b[K * NRepeats];
    // constexpr index_t KRepeats = sizeof(FloatA) / (sizeof(data_type) *
    // mfma_type.k_base); auto pa = reinterpret_cast<const data_type*>(&a); auto
    // pb = reinterpret_cast<const data_type*>(&b); constexpr index_t AStride =
    // K * KRepeats; constexpr index_t BStride = K * KRepeats;

    auto tid = b.create<WorkitemIdOp>(loc, b.getIndexType());
    auto laneId =
        b.create<RemUIOp>(loc, tid, b.create<ConstantIndexOp>(loc, wave_size));

    int64_t KRepeats = KPack / k_base;
    if (KRepeats == 0)
      KRepeats = 1;
    // llvm::errs() << "argVectorType: " << argType << "\n";
    // llvm::errs() << "k_base: " << k_base << "\n";
    // llvm::errs() << "KRepeats: " << KRepeats << "\n";
    // llvm::errs() << "K: " << K << "\n";
    // llvm::errs() << "bufferA type: " << op.bufferA().getType() << "\n";
    // llvm::errs() << "bufferB type: " << op.bufferB().getType() << "\n";

    auto MPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, MPerXdlops);
    auto NPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, NPerXdlops);
    auto KBaseConstantOp = b.create<ConstantIndexOp>(loc, k_base);
    auto kBaseKRepeatsConstantOp =
        b.create<ConstantIndexOp>(loc, KRepeats * k_base);

    if (!IsKReduction) {
      // store bufferA logic.

      // Original C++ logic.
      // static_if<!IsKReduction>{}([&](auto) {
      //   for(index_t m_i = 0; m_i < MRepeats; ++m_i)
      //     for(index_t k_i      = 0; k_i < K; ++k_i)
      //       a[k_i + m_i * K] = p_a_wave[k_i * M + laneId + MPerXdlops * m_i];
      // p_a_wave need to be offseted by waveOffsetA.

      auto outerLoopM = b.create<AffineForOp>(loc, 0, MRepeats);
      auto olmb = OpBuilder::atBlockTerminator(outerLoopM.getBody());
      auto olmiv = outerLoopM.getInductionVar();
      auto mOffset = olmb.create<AddIOp>(
          loc, aBase, olmb.create<MulIOp>(loc, MPerXdlopsConstantOp, olmiv));
      auto kOffsetA = olmb.create<MulIOp>(loc, olmiv, KConstantOp);

      auto innerLoopMK = olmb.create<AffineForOp>(loc, 0, K);
      auto ilmkb = OpBuilder::atBlockTerminator(innerLoopMK.getBody());
      auto ilmkiv = innerLoopMK.getInductionVar();

      //       a[k_i + m_i * K] = p_a_wave[k_i * M + laneId + MPerXdlops * m_i];
      // p_a_wave need to be offseted by waveOffsetA.
      Value sourceOffsetA = ilmkb.create<AddIOp>(
          loc,
          ilmkb.create<AddIOp>(
              loc, ilmkb.create<MulIOp>(loc, ilmkiv, MConstantOp), laneId),
          mOffset);

      if (KPack > 1)
        sourceOffsetA = ilmkb.create<MulIOp>(
            loc, sourceOffsetA, ilmkb.create<ConstantIndexOp>(loc, KPack));

      auto destOffsetA = ilmkb.create<AddIOp>(loc, ilmkiv, kOffsetA);

      Value valueA;
      if (KPack > 1) {
        valueA = emitLoadLogic(
            ilmkb, loc,
            op.bufferA().getType().template cast<MemRefType>().getElementType(),
            op.matrixA(), sourceOffsetA);
      } else {
        valueA = ilmkb.create<memref::LoadOp>(loc, dataType, op.matrixA(),
                                              sourceOffsetA);
      }
      ilmkb.create<memref::StoreOp>(loc, valueA, op.bufferA(),
                                    ValueRange{destOffsetA});

      // store bufferB logic.

      // Original C++ logic.
      //   for(index_t n_i = 0; n_i < NRepeats; ++n_i)
      //     for(index_t k_i      = 0; k_i < K; ++k_i)
      //       b[k_i + n_i * K] = p_b_wave[k_i * N + laneId + NPerXdlops * n_i];
      // p_b_wave need to be offseted by waveOffsetB.

      auto outerLoopN = b.create<AffineForOp>(loc, 0, NRepeats);
      auto olnb = OpBuilder::atBlockTerminator(outerLoopN.getBody());
      auto olniv = outerLoopN.getInductionVar();
      auto nOffset = olnb.create<AddIOp>(
          loc, bBase, olnb.create<MulIOp>(loc, NPerXdlopsConstantOp, olniv));
      auto kOffsetB = olnb.create<MulIOp>(loc, olniv, KConstantOp);

      auto innerLoopNK = olnb.create<AffineForOp>(loc, 0, K);
      auto ilnkb = OpBuilder::atBlockTerminator(innerLoopNK.getBody());
      auto ilnkiv = innerLoopNK.getInductionVar();

      //       b[k_i + n_i * K] = p_b_wave[k_i * N + laneId + NPerXdlops * n_i];
      // p_b_wave need to be offseted by waveOffsetB.
      Value sourceOffsetB = ilnkb.create<AddIOp>(
          loc,
          ilnkb.create<AddIOp>(
              loc, ilnkb.create<MulIOp>(loc, ilnkiv, NConstantOp), laneId),
          nOffset);

      if (KPack > 1)
        sourceOffsetB = ilnkb.create<MulIOp>(
            loc, sourceOffsetB, ilnkb.create<ConstantIndexOp>(loc, KPack));

      auto destOffsetB = ilnkb.create<AddIOp>(loc, ilnkiv, kOffsetB);

      Value valueB;
      if (KPack > 1) {
        valueB = emitLoadLogic(
            ilnkb, loc,
            op.bufferB().getType().template cast<MemRefType>().getElementType(),
            op.matrixB(), sourceOffsetB);
      } else {
        valueB = ilnkb.create<memref::LoadOp>(loc, dataType, op.matrixB(),
                                              sourceOffsetB);
      }
      ilnkb.create<memref::StoreOp>(loc, valueB, op.bufferB(),
                                    ValueRange{destOffsetB});

      // Original C++ logic.
      //
      // for(index_t k_i = 0; k_i < K * KRepeats; ++k_i)
      // {
      //     p_c_thread = mfma_type.template run<MPerXdlops * MRepeats,
      //                                         NPerXdlops * NRepeats,
      //                                         AStride,
      //                                         BStride>(
      //         &pa[k_i * mfma_type.k_base], &pb[k_i * mfma_type.k_base],
      //         p_c_thread);
      // }

      // Rewrite as:
      //
      // for(index_t k_i = 0; k_i < K; ++k_i) {
      //   bufferAElement = a[k_i];
      //   bufferBElement = b[k_i];
      //   for(index_t ki_i = 0; ki_i < KRepeats; ++ki_i)
      //     argA = &bufferAElement[ki_i * mfma_type.k_base];
      //     argB = &bufferAElement[ki_i * mfma_type.k_base];
      //     p_c_thread = mfma_type.template run<MPerXlops * MRepeats,
      //                                         NPerXdlops * NRepeats,
      //                                         AStride,
      //                                         BStride>(argA, argB,
      //       p_c_thread);
      // }

      int64_t KForOuterLoop;
      if (KPack > 1) {
        KForOuterLoop = K;
      } else {
        KForOuterLoop = K / k_base;
        if (KForOuterLoop == 0) {
          // KForOuterLoop is too small. Reject lowering.
          return failure();
        }
      }
      auto outerLoop =
          b.create<AffineForOp>(loc, 0, KForOuterLoop, 1, op.vectorCs());
      auto outerLoopb = OpBuilder::atBlockBegin(outerLoop.getBody());
      auto outerLoopiv = outerLoop.getInductionVar();

      MemRefType bufferAType = op.bufferA().getType().cast<MemRefType>();
      MemRefType bufferBType = op.bufferA().getType().cast<MemRefType>();
      Type bufferAElementType = bufferAType.getElementType();
      Type bufferBElementType = bufferBType.getElementType();
      Value bufferAElement = outerLoopb.create<memref::LoadOp>(
          loc, bufferAElementType, op.bufferA(), ValueRange{outerLoopiv});
      Value bufferBElement = outerLoopb.create<memref::LoadOp>(
          loc, bufferBElementType, op.bufferB(), ValueRange{outerLoopiv});

      auto innerLoop = outerLoopb.create<AffineForOp>(
          loc, 0, KRepeats, 1, outerLoop.getRegionIterArgs());
      auto innerLoopb = OpBuilder::atBlockBegin(innerLoop.getBody());
      auto innerLoopiv = innerLoop.getInductionVar();

      Value argA;
      Value argB;
      int64_t argTypeVectorLength =
          (argType.isa<VectorType>())
              ? argType.template cast<VectorType>().getShape()[0]
              : 1;
      if (argTypeVectorLength > 1) {
        Value zeroOp = createZeroConstantOp(innerLoopb, loc, dataType);

        Value offset;
        if (KPack > 1) {
          offset = innerLoopb.create<MulIOp>(loc, innerLoopiv, KBaseConstantOp);
        } else {
          offset = innerLoopb.create<AddIOp>(
              loc, innerLoopb.create<MulIOp>(loc, outerLoopiv, KBaseConstantOp),
              innerLoopb.create<MulIOp>(loc, innerLoopiv, KBaseConstantOp));
        }
        if (bufferAElementType.isa<VectorType>()) {
          // bufferA/BElement loaded on LDS are vectors.
          // argA/B to be supplied to MFMA XDLOPS are also vectors.
          assert(bufferAElementType.isa<VectorType>());
          assert(bufferBElementType.isa<VectorType>());
          assert(bufferAElementType.cast<VectorType>().getShape().size() == 1);
          assert(bufferBElementType.cast<VectorType>().getShape().size() == 1);
          assert(bufferAElementType.cast<VectorType>().getShape()[0] %
                     argTypeVectorLength ==
                 0);
          assert(bufferBElementType.cast<VectorType>().getShape()[0] %
                     argTypeVectorLength ==
                 0);

          argA = innerLoopb.create<vector::SplatOp>(loc, zeroOp, argType);
          argB = innerLoopb.create<vector::SplatOp>(loc, zeroOp, argType);
          for (int64_t i = 0; i < argTypeVectorLength; ++i) {
            Value iConstantOp = innerLoopb.create<ConstantIndexOp>(loc, i);
            Value iPlusOffsetConstantOp =
                innerLoopb.create<AddIOp>(loc, iConstantOp, offset);

            Value elementA = innerLoopb.create<vector::ExtractElementOp>(
                loc, dataType, bufferAElement, iPlusOffsetConstantOp);
            argA = innerLoopb.create<vector::InsertElementOp>(
                loc, argType, elementA, argA, iConstantOp);
            Value elementB = innerLoopb.create<vector::ExtractElementOp>(
                loc, dataType, bufferBElement, iPlusOffsetConstantOp);
            argB = innerLoopb.create<vector::InsertElementOp>(
                loc, argType, elementB, argB, iConstantOp);
          }
        } else {
          // bufferA/BElement loaded on LDS are scalars.
          // argA/B to be supplied to MFMA XDLOPS are vectors.
          argA = innerLoopb.create<vector::TransferReadOp>(
              loc, argType.template cast<VectorType>(), op.bufferA(),
              ValueRange{offset});
          argB = innerLoopb.create<vector::TransferReadOp>(
              loc, argType.template cast<VectorType>(), op.bufferB(),
              ValueRange{offset});
        }
      } else {
        if (bufferAElementType.isa<VectorType>()) {
          // bufferA/BElement loaded on LDS are vectors.
          // argA/B to be supplied to MFMA XDLOPS are scalars.
          assert(bufferAElementType.isa<VectorType>());
          assert(bufferBElementType.isa<VectorType>());
          assert(bufferAElementType.cast<VectorType>().getShape().size() == 1);
          assert(bufferBElementType.cast<VectorType>().getShape().size() == 1);

          argA = innerLoopb.create<vector::ExtractElementOp>(
              loc, dataType, bufferAElement, innerLoopiv);
          argB = innerLoopb.create<vector::ExtractElementOp>(
              loc, dataType, bufferBElement, innerLoopiv);
        } else {
          // bufferA/BElement loaded on LDS are scalars.
          // argA/B to be supplied to MFMA XDLOPS are also scalars.
          argA = bufferAElement;
          argB = bufferBElement;
        }
      }

      SmallVector<Value, 4> mfmas;
      for (int64_t i = 0; i < vectorNumber; ++i) {
        auto vectorC = innerLoop.getRegionIterArgs()[i];
        auto mfma =
            innerLoopb.create<MFMAV2Op>(loc, vectorType, argA, argB, vectorC);

        mfma->setAttr("instr", innerLoopb.getStringAttr(mfmaInstr));
        mfma->setAttr("imm", innerLoopb.getArrayAttr(
                                 {innerLoopb.getI32IntegerAttr(imms[i][0]),
                                  innerLoopb.getI32IntegerAttr(imms[i][1]),
                                  innerLoopb.getI32IntegerAttr(imms[i][2])}));
        mfmas.push_back(mfma);
      }
      innerLoopb.create<AffineYieldOp>(loc, mfmas);

      outerLoopb.create<AffineYieldOp>(loc, innerLoop.results());
      op.replaceAllUsesWith(outerLoop.results());
      op.erase();
    } else {
      // Original C++ logic.
      //     const index_t blk_id = laneId / mfma_type.num_threads_blk;
      //     const index_t blk_td = laneId % mfma_type.num_threads_blk;

      auto NumThreadsBlkConstantOp =
          b.create<ConstantIndexOp>(loc, num_threads_blk);
      auto blk_id = b.create<DivUIOp>(loc, laneId, NumThreadsBlkConstantOp);
      auto blk_td = b.create<RemUIOp>(loc, laneId, NumThreadsBlkConstantOp);

      Value kBaseA = b.create<AddIOp>(loc, aBase, blk_td);
      Value kBaseB = b.create<AddIOp>(loc, bBase, blk_td);

      // Original C++ logic.
      //     // load into registers
      //     for(index_t k_i = 0; k_i < K; k_i += mfma_type.num_input_blks) {
      //         a[k_i] = p_a_wave[(k_i + blk_id) * M + blk_td];
      //         b[k_i] = p_b_wave[(k_i + blk_id) * N + blk_td];
      //     }
      // p_a_wave need to be offseted by waveOffsetA.
      // p_b_wave need to be offseted by waveOffsetB.

      // Instead loop to K, change loop bound to K / num_input_blks.
      auto loopKLoadIteration = K / num_input_blks;
      auto loopKLoad = b.create<AffineForOp>(loc, 0, loopKLoadIteration);

      auto NumInputBlksConstantOp =
          b.create<ConstantIndexOp>(loc, num_input_blks);

      auto lklb = OpBuilder::atBlockTerminator(loopKLoad.getBody());
      auto lkliv = loopKLoad.getInductionVar();

      //         a[k_i] = p_a_wave[(k_i + blk_id) * M + blk_td];
      //         b[k_i] = p_b_wave[(k_i + blk_id) * N + blk_td];
      // p_a_wave need to be offseted by waveOffsetA.
      // p_b_wave need to be offseted by waveOffsetB.

      // NOTICE: We times k_i by num_input_blks in MLIR path.
      Value sourceOffsetA = lklb.create<AddIOp>(
          loc,
          lklb.create<MulIOp>(
              loc,
              lklb.create<AddIOp>(
                  loc, lklb.create<MulIOp>(loc, lkliv, NumInputBlksConstantOp),
                  blk_id),
              MConstantOp),
          kBaseA);

      Value valueA;
      if (KPack > 1) {
        valueA = emitLoadLogic(lklb, loc, argType, op.matrixA(), sourceOffsetA);
      } else {
        valueA = lklb.create<memref::LoadOp>(loc, dataType, op.matrixA(),
                                             ValueRange{sourceOffsetA});
      }
      lklb.create<memref::StoreOp>(loc, valueA, op.bufferA(),
                                   ValueRange{lkliv});

      // NOTICE: We times k_i by num_input_blks in MLIR path.
      Value sourceOffsetB = lklb.create<AddIOp>(
          loc,
          lklb.create<MulIOp>(
              loc,
              lklb.create<AddIOp>(
                  loc, lklb.create<MulIOp>(loc, lkliv, NumInputBlksConstantOp),
                  blk_id),
              NConstantOp),
          kBaseB);

      Value valueB;
      if (KPack > 1) {
        valueB = emitLoadLogic(lklb, loc, argType, op.matrixB(), sourceOffsetB);
      } else {
        valueB = lklb.create<memref::LoadOp>(loc, dataType, op.matrixB(),
                                             ValueRange{sourceOffsetB});
      }
      lklb.create<memref::StoreOp>(loc, valueB, op.bufferB(),
                                   ValueRange{lkliv});

      // Original C++ logic.
      // for(index_t k_i = 0; k_i < K; k_i += mfma_type.num_input_blks)
      // {
      //     for(index_t i = 0; i < KRepeats; ++i)
      //         p_c_thread = mfma_type.template run<MPerXdlops, NPerXdlops,
      //         AStride, BStride>(
      //             &pa[(k_i * KRepeats + i) * mfma_type.k_base],
      //             &pb[(k_i * KRepeats + i) * mfma_type.k_base],
      //             p_c_thread);
      // }

      // Change loop bound to the same as loopKLoadIteration.
      // Instead of increasing num_input_blks, increase k_base.

      if (loopKLoadIteration == 0) {
        // K load iteration is too small. Reject lowering.
        return failure();
      }
      auto outerLoop = b.create<AffineForOp>(loc, 0, loopKLoadIteration, k_base,
                                             op.vectorCs());
      auto outerLoopb = OpBuilder::atBlockBegin(outerLoop.getBody());
      auto outerLoopiv = outerLoop.getInductionVar();
      auto outerOffset =
          outerLoopb.create<MulIOp>(loc, outerLoopiv, kBaseKRepeatsConstantOp);

      auto innerLoop = outerLoopb.create<AffineForOp>(
          loc, 0, KRepeats * k_base, k_base, outerLoop.getRegionIterArgs());
      auto innerLoopb = OpBuilder::atBlockBegin(innerLoop.getBody());
      auto innerLoopiv = innerLoop.getInductionVar();

      auto offset = innerLoopb.create<AddIOp>(loc, outerOffset, innerLoopiv);

      Value argA;
      Value argB;
      int64_t argTypeVectorLength =
          (argType.isa<VectorType>())
              ? argType.template cast<VectorType>().getShape()[0]
              : 1;
      if (argTypeVectorLength > 1) {
        argA = innerLoopb.create<vector::TransferReadOp>(
            loc, argType.template cast<VectorType>(), op.bufferA(),
            ValueRange{offset});
        argB = innerLoopb.create<vector::TransferReadOp>(
            loc, argType.template cast<VectorType>(), op.bufferB(),
            ValueRange{offset});
      } else {
        argA = innerLoopb.create<memref::LoadOp>(loc, argType, op.bufferA(),
                                                 ValueRange{offset});
        argB = innerLoopb.create<memref::LoadOp>(loc, argType, op.bufferB(),
                                                 ValueRange{offset});
      }

      SmallVector<Value, 4> mfmas;
      for (int64_t i = 0; i < vectorNumber; ++i) {
        auto vectorC = innerLoop.getRegionIterArgs()[i];
        auto mfma =
            innerLoopb.create<MFMAV2Op>(loc, vectorType, argA, argB, vectorC);

        mfma->setAttr("instr", innerLoopb.getStringAttr(mfmaInstr));
        mfma->setAttr("imm", innerLoopb.getArrayAttr(
                                 {innerLoopb.getI32IntegerAttr(imms[i][0]),
                                  innerLoopb.getI32IntegerAttr(imms[i][1]),
                                  innerLoopb.getI32IntegerAttr(imms[i][2])}));
        mfmas.push_back(mfma);
      }
      innerLoopb.create<AffineYieldOp>(loc, mfmas);

      outerLoopb.create<AffineYieldOp>(loc, innerLoop.results());

      op.replaceAllUsesWith(outerLoop.results());
      op.erase();
    }

    return success();
  }
};

void LowerMIOpenOpsStep4Pass::runOnOperation() {
  ModuleOp op = getOperation();
  MLIRContext *ctx = &getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<TransformingForRewritePattern, ThreadwiseGemmRewritePattern,
               XdlopsGemmV2RewritePattern, BufferLoadRewritePattern,
               BufferStoreRewritePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    return signalPassFailure();

  // Apply loop invariant code motion to all loops before unrolling
  WalkResult licmResult =
      op.walk<WalkOrder::PostOrder>([](LoopLikeOpInterface loop) -> WalkResult {
        if (failed(moveLoopInvariantCode(loop)))
          return WalkResult::interrupt();
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

  // Run canonicalizers and CSE to clean up the code created by loop unrolling
  OpPassManager cleanupPipeline("builtin.module");
  cleanupPipeline.addPass(createCanonicalizerPass());
  cleanupPipeline.addPass(createCSEPass());
  (void)runPipeline(cleanupPipeline, op);
}

} // end anonymous namespace

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep4Pass() {
  return std::make_unique<LowerMIOpenOpsStep4Pass>();
}
