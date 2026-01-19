//===- GridwiseGemmToBlockwise - MLIR Rock ops lowering passes -----===//
//
// Copyright 2026 The MLIR Authors.
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
//
// This pass converts rock.gridwise_gemm_accel and rock.gridwise_attention_accel
// into block- and threadwise ops
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GeneralGemmBlockStructure.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "GridLayoutEmitter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include <optional>
#include <tuple>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKTRANSFORMSTOPOINTERARITHPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-transforms-to-pointer-arith"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockTransformsToPointerArithPass
    : public rock::impl::RockTransformsToPointerArithPassBase<
          RockTransformsToPointerArithPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

namespace {

// Helper to create ArithOp with OperationState
static  Value createArithOp(OpBuilder &builder, Location loc, Type resultType, StringRef name, Attribute constantValue,
                      ValueRange operands) {
  OperationState state(loc, rock::ArithOp::getOperationName());
  rock::ArithOp::build(builder, state, TypeRange{resultType},
                      builder.getStringAttr(name), constantValue, operands);
  return cast<rock::ArithOp>(builder.create(state)).getResult();
}

// Helper function to broadcast tensors to compatible shapes
static std::pair<Value, Value> broadcastTensors(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  auto memrefLhsType = cast<MemRefType>(lhs.getType());
  auto memrefRhsType = cast<MemRefType>(rhs.getType());

  auto lhsShape = memrefLhsType.getShape();
  auto rhsShape = memrefRhsType.getShape();
  auto rank = lhsShape.size();

  // Check if we need broadcasting
  bool needsBroadcast = false;
  SmallVector<int64_t> resultShape(rank);
  for (size_t i = 0; i < rank; ++i) {
    if (lhsShape[i] == 1 && rhsShape[i] != 1) {
      resultShape[i] = rhsShape[i];
      needsBroadcast = true;
    } else if (rhsShape[i] == 1 && lhsShape[i] != 1) {
      resultShape[i] = lhsShape[i];
      needsBroadcast = true;
    } else if (lhsShape[i] == rhsShape[i]) {
      resultShape[i] = lhsShape[i];
    } else {
      // Incompatible shapes - for now, assume they're compatible
      resultShape[i] = std::max(lhsShape[i], rhsShape[i]);
    }
  }

  if (!needsBroadcast) {
    // No broadcasting needed
    return {lhs, rhs};
  }

  // Create the broadcast result type
  auto resultType = MemRefType::get(resultShape, memrefLhsType.getElementType(),
                                    AffineMap{}, memrefLhsType.getMemorySpace());

  // Broadcast lhs if needed
  Value broadcastedLhs = lhs;
  if (!llvm::equal(lhsShape, resultShape)) {
    broadcastedLhs = rock::BroadcastOp::create(builder, loc, resultType, lhs);
  }

  // Broadcast rhs if needed
  Value broadcastedRhs = rhs;
  if (!llvm::equal(rhsShape, resultShape)) {
    broadcastedRhs = rock::BroadcastOp::create(builder, loc, resultType, rhs);
  }

  return {broadcastedLhs, broadcastedRhs};
}
// Helper function to broadcast a scalar to match a tensor's shape
static Value broadcastScalarToTensor(OpBuilder &builder, Location loc, Value scalar, Value tensor) {
  auto memrefType = cast<MemRefType>(tensor.getType());

  // Create a splat operation to broadcast the scalar to the memref shape
  return rock::SplatOp::create(builder, loc, memrefType, scalar);
}

// Helper function to ensure operands have compatible shapes
static std::pair<Value, Value> ensureCompatibleShapes(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();

  auto lhsMemrefType = dyn_cast<MemRefType>(lhsType);
  auto rhsMemrefType = dyn_cast<MemRefType>(rhsType);

  if (!lhsMemrefType && !rhsMemrefType) {
    // Both scalars, no broadcasting needed
    return {lhs, rhs};
  } else if (lhsMemrefType && !rhsMemrefType) {
    // LHS is tensor, RHS is scalar - broadcast RHS
    return {lhs, broadcastScalarToTensor(builder, loc, rhs, lhs)};
  } else if (!lhsMemrefType && rhsMemrefType) {
    // LHS is scalar, RHS is tensor - broadcast LHS
    return {broadcastScalarToTensor(builder, loc, lhs, rhs), rhs};
  } else {
    return broadcastTensors(builder, loc, lhs, rhs);
  }
}

/// Visit affine expressions recursively and build the sequence of operations
/// that correspond to it.  Visitation functions return an Value of the
/// expression subtree they visited or `nullptr` on error.
class AffineApplyExpander
    : public AffineExprVisitor<AffineApplyExpander, Value> {
public:
  /// This internal class expects arguments to be non-null, checks must be
  /// performed at the call site.
  AffineApplyExpander(OpBuilder &builder, ValueRange dimValues,
                      ValueRange symbolValues, Location loc)
      : builder(builder), dimValues(dimValues), symbolValues(symbolValues),
        loc(loc) {}

  Value buildBinaryExpr(AffineBinaryOpExpr expr, const std::string& opName,
                        arith::IntegerOverflowFlags overflowFlags =
                            arith::IntegerOverflowFlags::none) {
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    if (!lhs || !rhs)
      return nullptr;

    // Ensure operands have compatible shapes
    auto [broadcastedLhs, broadcastedRhs] = ensureCompatibleShapes(builder, loc, lhs, rhs);

    // Always use the rock.arith_op wrapper
    return createArithOp(builder, loc, broadcastedLhs.getType(), opName, nullptr,
                        ValueRange{broadcastedLhs, broadcastedRhs});
  }

  Value visitAddExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr(expr, "AddIOp");
  }

  Value visitMulExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr(expr, "MulIOp", arith::IntegerOverflowFlags::nsw);
  }

  /// Euclidean modulo operation: negative RHS is not allowed.
  /// Remainder of the euclidean integer division is always non-negative.
  ///
  /// Implemented as
  ///
  ///     a mod b =
  ///         let remainder = srem a, b;
  ///             negative = a < 0 in
  ///         select negative, remainder + b, remainder.
  Value visitModExpr(AffineBinaryOpExpr expr) {
    if (auto rhsConst = dyn_cast<AffineConstantExpr>(expr.getRHS())) {
      if (rhsConst.getValue() <= 0) {
        emitError(loc, "modulo by non-positive value is not supported");
        return nullptr;
      }
    }

    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    // Ensure operands have compatible shapes
    auto [broadcastedLhs, broadcastedRhs] = ensureCompatibleShapes(builder, loc, lhs, rhs);

    Value remainder = createArithOp(builder, loc, broadcastedLhs.getType(), "RemSIOp", nullptr,
                                    ValueRange{broadcastedLhs, broadcastedRhs});
    Value zeroCst = createArithOp(builder, loc, builder.getI32Type(), "ConstantIntOp",
                                 builder.getIntegerAttr(builder.getI32Type(), 0), ValueRange{});
    Value isRemainderNegative = createArithOp(builder, loc, builder.getI1Type(), "CmpIOp_slt", nullptr,
                                              ValueRange{remainder, zeroCst});
    Value correctedRemainder = createArithOp(builder, loc, broadcastedLhs.getType(), "AddIOp", nullptr,
                                             ValueRange{remainder, broadcastedRhs});
    Value result = createArithOp(builder, loc, broadcastedLhs.getType(), "SelectOp", nullptr,
                                 ValueRange{isRemainderNegative, correctedRemainder, remainder});
    return result;
  }

  /// Floor division operation (rounds towards negative infinity).
  ///
  /// For positive divisors, it can be implemented without branching and with a
  /// single division operation as
  ///
  ///        a floordiv b =
  ///            let negative = a < 0 in
  ///            let absolute = negative ? -a - 1 : a in
  ///            let quotient = absolute / b in
  ///                negative ? -quotient - 1 : quotient
  ///
  /// Note: this lowering does not use arith.floordivsi because the lowering of
  /// that to arith.divsi (see populateCeilFloorDivExpandOpsPatterns) generates
  /// not one but two arith.divsi. That could be changed to one divsi, but one
  /// way or another, going through arith.floordivsi will result in more complex
  /// IR because arith.floordivsi is more general than affine floordiv in that
  /// it supports negative RHS.
  Value visitFloorDivExpr(AffineBinaryOpExpr expr) {
    if (auto rhsConst = dyn_cast<AffineConstantExpr>(expr.getRHS())) {
      if (rhsConst.getValue() <= 0) {
        emitError(loc, "division by non-positive value is not supported");
        return nullptr;
      }
    }
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    // Ensure operands have compatible shapes
    auto [broadcastedLhs, broadcastedRhs] = ensureCompatibleShapes(builder, loc, lhs, rhs);

    Value zeroCst = createArithOp(builder, loc, builder.getI32Type(), "ConstantIntOp",
                                 builder.getIntegerAttr(builder.getI32Type(), 0), ValueRange{});
    Value noneCst = createArithOp(builder, loc, builder.getI32Type(), "ConstantIntOp",
                                 builder.getIntegerAttr(builder.getI32Type(), -1), ValueRange{});
    Value negative = createArithOp(builder, loc, builder.getI1Type(), "CmpIOp_slt", nullptr,
                                           ValueRange{broadcastedLhs, zeroCst});
    Value negatedDecremented = createArithOp(builder, loc, broadcastedLhs.getType(), "SubIOp", nullptr,
                                             ValueRange{noneCst, broadcastedLhs});
    Value dividend = createArithOp(builder, loc, broadcastedLhs.getType(), "SelectOp", nullptr,
                                   ValueRange{negative, negatedDecremented, broadcastedLhs});
    Value quotient = createArithOp(builder, loc, broadcastedLhs.getType(), "DivSIOp", nullptr,
                                   ValueRange{dividend, broadcastedRhs});
    Value correctedQuotient = createArithOp(builder, loc, broadcastedLhs.getType(), "SubIOp", nullptr,
                                            ValueRange{noneCst, quotient});
    Value result = createArithOp(builder, loc, broadcastedLhs.getType(), "SelectOp", nullptr,
                                 ValueRange{negative, correctedQuotient, quotient});
    return result;
  }

  /// Ceiling division operation (rounds towards positive infinity).
  ///
  /// For positive divisors, it can be implemented without branching and with a
  /// single division operation as
  ///
  ///     a ceildiv b =
  ///         let negative = a <= 0 in
  ///         let absolute = negative ? -a : a - 1 in
  ///         let quotient = absolute / b in
  ///             negative ? -quotient : quotient + 1
  ///
  /// Note: not using arith.ceildivsi for the same reason as explained in the
  /// visitFloorDivExpr comment.
  Value visitCeilDivExpr(AffineBinaryOpExpr expr) {
    if (auto rhsConst = dyn_cast<AffineConstantExpr>(expr.getRHS())) {
      if (rhsConst.getValue() <= 0) {
        emitError(loc, "division by non-positive value is not supported");
        return nullptr;
      }
    }
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    // Ensure operands have compatible shapes
    auto [broadcastedLhs, broadcastedRhs] = ensureCompatibleShapes(builder, loc, lhs, rhs);

    Value zeroCst = createArithOp(builder, loc, builder.getI32Type(), "ConstantIntOp",
                                          builder.getIntegerAttr(builder.getI32Type(), 0), ValueRange{});
    Value oneCst = createArithOp(builder, loc, builder.getI32Type(), "ConstantIntOp",
                                 builder.getIntegerAttr(builder.getI32Type(), 1), ValueRange{});
    Value nonPositive = createArithOp(builder, loc, builder.getI1Type(), "CmpIOp_sle", nullptr,
                                      ValueRange{broadcastedLhs, zeroCst});
    Value negated = createArithOp(builder, loc, broadcastedLhs.getType(), "SubIOp", nullptr,
                                  ValueRange{zeroCst, broadcastedLhs});
    Value decremented = createArithOp(builder, loc, broadcastedLhs.getType(), "SubIOp", nullptr,
                                      ValueRange{broadcastedLhs, oneCst});
    Value dividend = createArithOp(builder, loc, broadcastedLhs.getType(), "SelectOp", nullptr,
                                   ValueRange{nonPositive, negated, decremented});
    Value quotient = createArithOp(builder, loc, broadcastedLhs.getType(), "DivSIOp", nullptr,
                                   ValueRange{dividend, broadcastedRhs});
    Value negatedQuotient = createArithOp(builder, loc, broadcastedLhs.getType(), "SubIOp", nullptr,
                                          ValueRange{zeroCst, quotient});
    Value incrementedQuotient = createArithOp(builder, loc, broadcastedLhs.getType(), "AddIOp", nullptr,
                                              ValueRange{quotient, oneCst});
    Value result = createArithOp(builder, loc, broadcastedLhs.getType(), "SelectOp", nullptr,
                                 ValueRange{nonPositive, negatedQuotient, incrementedQuotient});
    return result;
  }

  Value visitConstantExpr(AffineConstantExpr expr) {
    return createArithOp(builder, loc, builder.getI32Type(), "ConstantIntOp",
                        builder.getIntegerAttr(builder.getI32Type(), expr.getValue()), ValueRange{});
  }

  Value visitDimExpr(AffineDimExpr expr) {
    assert(expr.getPosition() < dimValues.size() &&
           "affine dim position out of range");
    return dimValues[expr.getPosition()];
  }

  Value visitSymbolExpr(AffineSymbolExpr expr) {
    assert(expr.getPosition() < symbolValues.size() &&
           "symbol dim position out of range");
    return symbolValues[expr.getPosition()];
  }

 private:
   OpBuilder &builder;
   ValueRange dimValues;
   ValueRange symbolValues;

   Location loc;
 };
  
/// Create a sequence of operations that implement the `expr` applied to the
/// given dimension and symbol values.
static mlir::Value expandAffineExpr(OpBuilder &builder, Location loc,
                                           AffineExpr expr,
                                           ValueRange dimValues,
                                           ValueRange symbolValues) {
  return AffineApplyExpander(builder, dimValues, symbolValues, loc).visit(expr);
}

/// Create a sequence of operations that implement the `affineMap` applied to
/// the given `operands` (as it it were an AffineApplyOp).
static std::optional<SmallVector<Value, 8>>
expandAffineMap(OpBuilder &builder, Location loc,
                              AffineMap affineMap, ValueRange operands) {
  auto numDims = affineMap.getNumDims();
  auto expanded = llvm::to_vector<8>(
      llvm::map_range(affineMap.getResults(),
                      [numDims, &builder, loc, operands](AffineExpr expr) {
                        return expandAffineExpr(builder, loc, expr,
                                                operands.take_front(numDims),
                                                operands.drop_front(numDims));
                      }));
  if (llvm::all_of(expanded, [](Value v) { return v; }))
    return expanded;
  return std::nullopt;
}

static Value updateValidityAfter(OpBuilder &b, Location loc,
                                      TransformMapAttr map,
                                      ValueRange outputs) {
    // auto [broadcastedLhs, broadcastedRhs] = ensureCompatibleShapes(builder, loc, lhs, rhs);
    // Value isValid = createArithOp(b, loc, b.getI1Type(), "ConstantIntOp", b.getBoolAttr(true), ValueRange{})
  Value isValid = createArithOp(b, loc, b.getI1Type(), "ConstantIntOp", b.getBoolAttr(true), ValueRange{});
  ArrayRef<int64_t> lowerBounds = map.getLowerBounds();

  // unsigned < catches both negatives (as all negatives are > the bound)
  // and being too large on the right.
  auto addLowerDimUltClamp = [&](uint32_t lowerDim) {
    int64_t bound = lowerBounds[lowerDim];
    Value boundConst = createArithOp(b, loc, b.getI32Type(), "ConstantIntOp", b.getIntegerAttr(b.getI32Type(), bound), ValueRange{});
    Value output = outputs[lowerDim];
    auto [broadcastedOutput, broadcastedBoundConst] = ensureCompatibleShapes(b, loc, output, boundConst);
    auto memrefType = cast<MemRefType>(broadcastedOutput.getType());
    auto inBoundsType = MemRefType::get(memrefType.getShape(), b.getI1Type(), AffineMap{}, memrefType.getMemorySpace());
    Value inBounds = createArithOp(b, loc, inBoundsType, "CmpIOp_ult", nullptr,
                                              ValueRange{broadcastedOutput, broadcastedBoundConst});
    auto [broadcastedInBounds, broadcastedIsValid] = ensureCompatibleShapes(b, loc, inBounds, isValid);
    isValid = createArithOp(b, loc, broadcastedInBounds.getType(), "AndIOp", nullptr, ValueRange{broadcastedInBounds, broadcastedIsValid});
  };

  for (TransformAttr op : map.getOps()) {
    TransformType type = op.getType();
    ArrayRef<uint32_t> lowerDims = op.getLowerDims();
    ArrayRef<int64_t> params = op.getParams();
    if (type == TransformType::Pad) {
      for (const auto &pair : llvm::enumerate(lowerDims)) {
        size_t leftParam = 2 * pair.index();
        size_t rightParam = leftParam + 1;
        uint32_t lowerDim = pair.value();

        if (params[leftParam] == 0 && params[rightParam] == 0)
          continue;
        addLowerDimUltClamp(lowerDim);
      }
    }
    if (type == TransformType::Embed) {
      if (!embedCanBeInvalid(map, op))
        continue;
      addLowerDimUltClamp(op.getLowerDims()[0]);
    }
  }
  return isValid;
}

//===----------------------------------------------------------------------===//
// BlockwiseStoreTileOp lowering.
//===----------------------------------------------------------------------===//
struct TransformsToPtrRewritePattern
    : public OpRewritePattern<TransformsToPtrOp> {
  using OpRewritePattern<TransformsToPtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransformsToPtrOp op,
                                PatternRewriter &b) const override {
                                  llvm::errs() << "debug0\n";
    using AffineResults = SmallVector<Value>;
    Location loc = op.getLoc();
    Value source = op.getSource();
    ValueRange extraIndices = op.getExtraIndices();
    Value pointers = op.getPointers();
    Value mask = op.getMask();

    source = isolateTransforms(b, source);
                                  llvm::errs() << "debug1\n";

    // TODO(roctriton): buffer could be the output of input fusion instead of an input tensor!
    // Fix this when we enable fusions.
    auto [buffer, transforms, needs64BitIdx] = untransform(b, source);
                                  llvm::errs() << "debug2\n";

    size_t bufferIdxCount = cast<MemRefType>(pointers.getType()).getRank();
    ArrayRef<int64_t> shape = cast<MemRefType>(pointers.getType()).getShape();
    llvm::errs() << "shape=";
    llvm::interleaveComma(shape, llvm::errs());
    llvm::errs() << "\n";
    assert(bufferIdxCount != 0);
    size_t extraIdxCount = extraIndices.size();
    llvm::errs() << "bufferIdxCount="<<bufferIdxCount<<"\n";
    llvm::errs() << "extraIdxCount="<<extraIdxCount<<"\n";
    assert(extraIdxCount >= bufferIdxCount);
    SmallVector<Value> initValues(extraIndices);
    llvm::errs() << "initValues=";
    llvm::interleaveComma(initValues, llvm::errs());
    llvm::errs() << "\n";
                                  llvm::errs() << "debug3\n";
    for(size_t dimension = 0; dimension < shape.size(); ++dimension) {
      // Create memref shape with 1s everywhere except the current dimension
      SmallVector<int64_t> memrefShape(shape.size(), 1);
      memrefShape[dimension] = shape[dimension];

      // Create memref type for the range
      auto privateMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
          gpu::GPUDialect::getPrivateAddressSpace());

      auto memrefType =
          MemRefType::get(memrefShape, b.getI32Type(), AffineMap{}, privateMemoryAddressSpace);
      // Allocate the memref
      Value rangeMemref = rock::GpuAllocOp::create(b, loc, memrefType);
      // Create the range values in the memref
      
      rock::MakeRangeOp::create(b, loc, rangeMemref, b.getIntegerAttr(b.getI32Type(), 0), b.getIntegerAttr(b.getI32Type(), shape[dimension]));
      initValues.push_back(rangeMemref);
    }
    llvm::errs() << "initValues=";
    llvm::interleaveComma(initValues, llvm::errs());
    llvm::errs() << "\n";
                                  llvm::errs() << "debug4\n";

    // TODO(roctriton): check rangeIndices match `pointers` and `mask` shapes

    Value baseAddr =
        memref::ExtractAlignedPointerAsIndexOp::create(b, loc, buffer);
                                  llvm::errs() << "debug5\n";
    baseAddr = b.create<arith::IndexCastOp>(loc, b.getI32Type(), baseAddr);

    ////// Init
    
    // For each domain, store the sequence of composed affine maps needed to
    // compute the result coordinate, along with the transform map that
    // triggered each break in the chain. Such a break is created at any point
    // where the validity of map coordinates is impacted.
    SmallVector<std::pair<AffineMap, TransformMapAttr>>
        composedMaps;
        
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
                                  llvm::errs() << "debug6\n";
                              
    //////


    //////
    
    // Create code to actually transform the coordinates
    AffineResults computed(initValues);
    Value isValid = createArithOp(b, loc, b.getI1Type(), "ConstantIntOp", b.getBoolAttr(true), ValueRange{});
    for (const auto &[composedMap, transform] : composedMaps) {
      if (!composedMap) // empty transformations
        continue;
      std::optional<AffineResults> transformed =
          expandAffineMap(b, loc, composedMap, computed);
      if (!transformed)
        return failure();
      computed.assign(*transformed);
      if (transform) { // Time for bounds checks or other validity updates
        Value validityUpdate =
            updateValidityAfter(b, loc, transform, computed);
        auto [broadcastedValidityUpdate, broadcastedIsValid] = ensureCompatibleShapes(b, loc, validityUpdate, isValid);
        isValid = createArithOp(b, loc, broadcastedValidityUpdate.getType(), "AndIOp", nullptr, ValueRange{broadcastedValidityUpdate, broadcastedIsValid});
      }
    }
                                  llvm::errs() << "debug7\n";

    // add `baseAddr`
    Value baseAddrSplat = rock::SplatOp::create(b, loc, computed[0].getType(), baseAddr);
    Value pointerTensor = createArithOp(b, loc, computed[0].getType(), "AddPtrOp", nullptr,
                      {baseAddrSplat, computed[0]});

                                  llvm::errs() << "debug8 op="<<op<<"\n";
                                  llvm::errs() << "computed=";
                                  llvm::interleaveComma(computed, llvm::errs());
                                  llvm::errs() << "\n";
    // Copy the computed pointer values into the pointers memref
    // Since computed[0] is a memref of the same shape as pointers, we need memref.copy
    memref::CopyOp::create(b, loc, pointerTensor, pointers);

    // Store the validity mask into the mask memref
                                  llvm::errs() << "isValid="<<isValid<<"\n";
    // For mask, we need to handle it similarly - isValid should be a memref that we copy
    auto [broadcastedIsValid, _] = ensureCompatibleShapes(b, loc, isValid, mask);
    memref::CopyOp::create(b, loc, broadcastedIsValid, mask);
    //////
    b.eraseOp(op);
                                  llvm::errs() << "debug9\n";

    return success();
  }
};

} // end anonymous namespace

void RockTransformsToPointerArithPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<TransformsToPtrOp>();
  target.addLegalOp<rock::GpuAllocOp, rock::MakeRangeOp, rock::BroadcastOp, rock::ArithOp>();
  target.addLegalDialect<rock::RockDialect, memref::MemRefDialect, arith::ArithDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<TransformsToPtrRewritePattern>(ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
