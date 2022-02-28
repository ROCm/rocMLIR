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
      if (destType == op.vector().getType().cast<VectorType>()) {
        // Extracting something the same size as the vector is a noop since
        // the index must be 0 for the op to be defined. This is here in case
        // the canonicalizer didn't catch this or didn't run.
        b.replaceOp(op, op.vector());
        return success();
      }
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
      if (srcType == op.dest().getType().cast<VectorType>()) {
        // Inserting a slice of the same size as the destination is a noop
        // since the index must be 0 for the op to be defined. This is here in
        // case the canonicalizer didn't run or didn't catch the problem.
        b.replaceOp(op, op.source());
        return success();
      }
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
    if (lanePerm.hasValue()) {
      Value groupSizeConst = b.create<ConstantIndexOp>(loc, swizzleGroupSize);
      laneInSwizzleGroup = b.create<RemUIOp>(loc, laneId, groupSizeConst);
    }

    Value result = vector;

    for (int32_t logRotation = base; logRotation != target;
         logRotation += offset) {
      uint32_t rotation = 1 << logRotation;
      Value shouldParticipate;
      if (lanePerm.hasValue()) {
        // Non-standard arrangement of rows -> lanes, use longer test
        ArrayRef<uint32_t> theLanePerm = lanePerm.getValue();
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

    Value vector = op.vector();
    uint32_t totalSize = vector.getType().cast<VectorType>().getNumElements();

    Value laneId = op.laneId();
    uint32_t groupSize = op.size();

    ArrayAttr inGroupPermAttr = op.inGroupPerm();
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

void MIOpenExpandShorthandPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<ExtractSliceRewritePattern, InsertSliceRewritePattern,
               InWarpTransposeRewritePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace

std::unique_ptr<Pass> mlir::miopen::createMIOpenExpandShorthandPass() {
  return std::make_unique<MIOpenExpandShorthandPass>();
}
