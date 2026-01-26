//===- RockToTTIR - MLIR Rock ops lowering passes -------------------------===//
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
// =============================================================================
//
// This pass converts Rock dialect operations to Triton IR
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
// #include "triton/Dialect/Triton/IR/TritonEnums.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKTOTTIRPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-to-ttir"

using namespace mlir;
using namespace mlir::rock;
using namespace mlir::triton;
using namespace mlir::arith;

namespace {
struct RockToTTIRPass : public rock::impl::RockToTTIRPassBase<RockToTTIRPass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// TensorSplatOpRewritePattern - Convert tensor.splat to tt.splat
//===----------------------------------------------------------------------===//
struct TensorSplatOpRewritePattern : public OpRewritePattern<tensor::SplatOp> {
  using OpRewritePattern<tensor::SplatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::SplatOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the source scalar value
    Value src = op.getInput();

    // Get the result tensor type
    auto resultType = op.getResult().getType();

    // Create tt.splat operation
    Value result = triton::SplatOp::create(rewriter, loc, resultType, src);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RockFillOpRewritePattern - Convert rock.fill to tt.splat
// This pattern is applied via greedy rewriting before partial conversion
// because it needs to replace uses of the input tensor, which isn't supported
// in conversion pattern rollback mode.
//===----------------------------------------------------------------------===//
struct RockFillOpRewritePattern : public OpRewritePattern<rock::FillOp> {
  using OpRewritePattern<rock::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::FillOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value inputTensor = op.getInput();
    Value fillValue = op.getValue();

    auto tensorType = dyn_cast<RankedTensorType>(inputTensor.getType());
    if (!tensorType)
      return failure();

    // Create tt.splat operation to create a tensor filled with the value
    Value splatTensor =
        triton::SplatOp::create(rewriter, loc, tensorType, fillValue);

    // Replace all uses of the original tensor (except the fill op itself)
    // with the new splat-filled tensor
    rewriter.replaceAllUsesExcept(inputTensor, splatTensor, op);

    // Erase the fill op
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RockMakeRangeOpRewritePattern - Convert rock.make_range to tt.make_range
//===----------------------------------------------------------------------===//
struct RockMakeRangeOpRewritePattern
    : public OpRewritePattern<rock::MakeRangeOp> {
  using OpRewritePattern<rock::MakeRangeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::MakeRangeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    int32_t start = op.getStart();
    int32_t end = op.getEnd();

    auto tensorType = dyn_cast<RankedTensorType>(op.getType());
    if (!tensorType)
      return failure();

    // triton::MakeRangeOp only supports 1D tensors, but rock::MakeRangeOp
    // can output multi-dimensional tensors where only one dimension is
    // non-unit. Find the non-unit dimension and use expand_dims to restore the
    // shape.
    ArrayRef<int64_t> shape = tensorType.getShape();
    int64_t nonUnitDim = -1;
    SmallVector<int64_t> unitDimIndices;

    for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
      if (shape[i] > 1) {
        if (nonUnitDim != -1) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Expected only one non-unit dimension in MakeRangeOp "
                        "output shape\n");
          return failure();
        }
        nonUnitDim = shape[i];
      } else {
        unitDimIndices.push_back(i);
      }
    }

    if (nonUnitDim == -1) {
      LLVM_DEBUG(llvm::dbgs() << "Expected at least one non-unit dimension\n");
      return failure();
    }

    // Create 1D tensor type for tt.make_range
    auto tensorType1D =
        RankedTensorType::get({nonUnitDim}, rewriter.getI32Type());

    // Create tt.make_range operation (1D)
    Value rangeTensor =
        triton::MakeRangeOp::create(rewriter, loc, tensorType1D, start, end);

    // Use tt.expand_dims to restore the original shape
    // We need to insert unit dimensions at the correct positions
    Value expandedTensor = rangeTensor;
    for (int64_t unitDimIdx : unitDimIndices) {
      // Get current tensor type
      auto currentType = cast<RankedTensorType>(expandedTensor.getType());
      SmallVector<int64_t> newShape(currentType.getShape().begin(),
                                    currentType.getShape().end());
      newShape.insert(newShape.begin() + unitDimIdx, 1);
      auto expandedType =
          RankedTensorType::get(newShape, rewriter.getI32Type());

      expandedTensor = triton::ExpandDimsOp::create(rewriter, loc, expandedType,
                                                    expandedTensor, unitDimIdx);
    }

    rewriter.replaceOp(op, expandedTensor);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RockBroadCastOpRewritePattern - Convert rock.broadcast to tt.broadcast
//===----------------------------------------------------------------------===//
struct RockBroadCastOpRewritePattern
    : public OpRewritePattern<rock::BroadcastOp> {
  using OpRewritePattern<rock::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the source tensor
    Value src = op.getSrc();

    // Get the result tensor type
    auto resultTensorType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultTensorType)
      return failure();

    // Get the source tensor type
    auto srcTensorType = dyn_cast<RankedTensorType>(src.getType());
    if (!srcTensorType)
      return failure();

    // Create tt.broadcast operation
    Value result = triton::BroadcastOp::create(rewriter, loc, resultTensorType, src);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RockWorkgroupIdOpRewritePattern - Convert rock.workgroup_id to
// tt.get_program_id
//===----------------------------------------------------------------------===//
struct RockWorkgroupIdOpRewritePattern
    : public OpRewritePattern<rock::WorkgroupIdOp> {
  using OpRewritePattern<rock::WorkgroupIdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::WorkgroupIdOp op,
                                PatternRewriter &rewriter) const override {
    // tt.get_program_id returns i32
    Value programId = triton::GetProgramIdOp::create(
        rewriter, op.getLoc(), triton::ProgramIDDim::X);
    rewriter.replaceOp(op, programId);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RockLoadTilePtrOpRewritePattern - Convert rock.blockwise_load_tile_ptr to
// tt.load
//===----------------------------------------------------------------------===//
struct RockLoadTilePtrOpRewritePattern
    : public OpRewritePattern<rock::BlockwiseLoadTilePtrOp> {
  using OpRewritePattern<rock::BlockwiseLoadTilePtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::BlockwiseLoadTilePtrOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get operands (all tensors now)
    Value pointerTensor = op.getPointerTensor();
    Value maskTensor = op.getMaskTensor();

    // Get the element type and shape from the result type
    auto resultTensorType = cast<RankedTensorType>(op.getResult().getType());
    Type elementType = resultTensorType.getElementType();

    // Verify pointerTensor is a tensor of i32
    auto ptrTensorType = dyn_cast<RankedTensorType>(pointerTensor.getType());
    if (!ptrTensorType || !ptrTensorType.getElementType().isInteger(32)) {
      LLVM_DEBUG(llvm::dbgs() << "Pointer tensor is not a tensor of i32\n");
      return failure();
    }

    // Verify maskTensor is a tensor of i1
    auto maskTensorType = dyn_cast<RankedTensorType>(maskTensor.getType());
    if (!maskTensorType || !maskTensorType.getElementType().isInteger(1)) {
      LLVM_DEBUG(llvm::dbgs() << "Mask tensor is not a tensor of i1\n");
      return failure();
    }

    // Create pointer type: !tt.ptr<elementType>
    // Use address space 1 (global) as default for Triton
    triton::PointerType ptrType = triton::PointerType::get(elementType, 1);

    // Create tensor of pointers: tensor<...x!tt.ptr<elementType>>
    RankedTensorType ptrTensorOfPtrsType =
        RankedTensorType::get(ptrTensorType.getShape(), ptrType,
                              ptrTensorType.getEncoding());

    // Convert tensor of i32 to tensor of pointers
    Value ptrTensorOfPtrs =
        rewriter.create<rock::CastToPtrOp>(loc, ptrTensorOfPtrsType, pointerTensor);

    // Create tt.load operation
    // LoadOp takes: ptr, mask (optional), other (optional), boundaryCheck,
    // padding, cache, evict, isVolatile Create attributes with default values
    auto boundaryCheckAttr = rewriter.getDenseI32ArrayAttr({});
    auto cacheAttr = triton::CacheModifierAttr::get(
        rewriter.getContext(), triton::CacheModifier::NONE);
    auto evictAttr = triton::EvictionPolicyAttr::get(
        rewriter.getContext(), triton::EvictionPolicy::NORMAL);
    auto isVolatileAttr = rewriter.getBoolAttr(false);

    Value result = rewriter.create<triton::LoadOp>(
        loc, resultTensorType, ptrTensorOfPtrs, maskTensor,
        /*other=*/Value(), boundaryCheckAttr,
        /*padding=*/nullptr, cacheAttr, evictAttr, isVolatileAttr);

    // Replace the op with the loaded tensor result
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RockBlockwiseGemmOpRewritePattern - Convert rock.blockwise_gemm_accel to
// tt.dot
//===----------------------------------------------------------------------===//
struct RockBlockwiseGemmOpRewritePattern
    : public OpRewritePattern<rock::BlockwiseGemmOp> {
  using OpRewritePattern<rock::BlockwiseGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::BlockwiseGemmOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get operands (already tensors)
    Value a = op.getMatrixA();
    Value b = op.getMatrixB();
    Value c = op.getMatrixC();

    // Get the tensor types
    auto aTensorType = dyn_cast<RankedTensorType>(a.getType());
    auto bTensorType = dyn_cast<RankedTensorType>(b.getType());
    auto cTensorType = dyn_cast<RankedTensorType>(c.getType());
    if (!aTensorType || !bTensorType || !cTensorType)
      return failure();

    // Create tt.dot operation
    Value result = rewriter.create<triton::DotOp>(
        loc, cTensorType, a, b, c,
        /*inputPrecision=*/triton::InputPrecision::IEEE,
        /*maxNumImpreciseAcc=*/0);

    // We dont use replaceOp because result has one result whereas op has none.
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RockMicroKernelOpRewritePattern
//===----------------------------------------------------------------------===//
struct RockMicroKernelOpRewritePattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Make sure this is a microkernel (contains a tt.dot operation)
    triton::DotOp dotOpInBody = nullptr;
    Block *body = op.getBody();

    for (Operation &bodyOp : *body) {
      if (auto dot = dyn_cast<triton::DotOp>(bodyOp)) {
        dotOpInBody = dot;
        break;
      }
    }

    if (!dotOpInBody) {
      LLVM_DEBUG(llvm::dbgs() << "Loop is not a microkernel\n");
      return failure();
    }

    // Get the output tensor from the tt.dot's C operand
    Value outputTensor = dotOpInBody.getC();
    auto outputTensorType = dyn_cast<RankedTensorType>(outputTensor.getType());
    if (!outputTensorType) {
      LLVM_DEBUG(llvm::dbgs() << "tt.dot C operand is not a tensor\n");
      return failure();
    }

    // Create init args for the new loop - the output tensor
    SmallVector<Value> initArgs;
    initArgs.push_back(outputTensor);

    // Create new ForOp with same bounds but new init args.
    rewriter.setInsertionPointAfter(op);
    auto newForOp = scf::ForOp::create(
        rewriter, loc, op.getLowerBound(), op.getUpperBound(), op.getStep(),
        initArgs, [](OpBuilder &builder, Location loc, Value, ValueRange) {
          // Create an empty yield (will be updated later with proper operands)
          scf::YieldOp::create(builder, loc, ValueRange{});
        });
    newForOp->setAttrs(op->getAttrs());

    // Move operations from old body to new body, except the scf.yield
    // terminator.
    Block *oldBody = op.getBody();
    Block *newBody = newForOp.getBody();

    // Collect ops to move (excluding the terminator)
    SmallVector<Operation *> opsToMove;
    for (Operation &bodyOp : *oldBody) {
      if (!bodyOp.hasTrait<OpTrait::IsTerminator>()) {
        opsToMove.push_back(&bodyOp);
      }
    }

    // Move the ops to the new body (before the yield terminator)
    Operation *newTerminator = newBody->getTerminator();
    for (Operation *opToMove : opsToMove) {
      opToMove->moveBefore(newTerminator);
    }

    // Replace old block arguments with new ones.
    // First argument is the induction variable.
    op.getInductionVar().replaceAllUsesWith(newForOp.getInductionVar());

    // Find the tt.dot operation in the new body and update its C operand
    // to use the iter arg instead of the original tensor
    triton::DotOp dotOp = nullptr;
    for (Operation &bodyOp : *newBody) {
      if (auto dot = dyn_cast<triton::DotOp>(bodyOp)) {
        dotOp = dot;
        break;
      }
    }

    if (!dotOp) {
      LLVM_DEBUG(llvm::dbgs() << "Expected a tt.dot op in the loop body\n");
      return failure();
    }

    // Update the tt.dot's C operand to use the iter arg
    // First iter arg is at index 1 (index 0 is the induction variable)
    dotOp->setOperand(2, newForOp.getBody()->getArgument(1));

    // Update the yield to yield the output of the tt.dot
    auto yieldOp = cast<scf::YieldOp>(newBody->getTerminator());

    rewriter.setInsertionPoint(yieldOp);
    SmallVector<Value> yieldOperands;
    yieldOperands.push_back(dotOp.getResult()); // Output of tt.dot

    // Modify the yield op in place.
    yieldOp->setOperands(yieldOperands);

    // Finally, erase the old ForOp
    rewriter.eraseOp(op);

    return success();
  }
};

struct RockStoreTilePtrOpRewritePattern
    : public OpRewritePattern<rock::BlockwiseStoreTilePtrOp> {
  using OpRewritePattern<rock::BlockwiseStoreTilePtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::BlockwiseStoreTilePtrOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get operands (all tensors now)
    Value pointerTensor = op.getPointerTensor();
    Value maskTensor = op.getMaskTensor();

    // 1. Find the scf.for loop just above and get its last result
    // This is the value to store (accumulated result from the dot product)
    scf::ForOp forOp = nullptr;
    for (Operation &blockOp : llvm::reverse(*op->getBlock())) {
      if (&blockOp == op.getOperation())
        continue;
      if (auto candidateForOp = dyn_cast<scf::ForOp>(&blockOp)) {
        if (candidateForOp.getNumResults() == 1) {
          forOp = candidateForOp;
          break;
        }
      }
    }

    if (!forOp) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot find scf.for with exactly 1 result\n");
      return failure();
    }

    // Get the last result from the loop - this is the value to store
    Value valueToStore = forOp.getResult(forOp.getNumResults() - 1);

    // 2. Verify pointer tensor is a tensor of i32
    auto ptrTensorType = dyn_cast<RankedTensorType>(pointerTensor.getType());
    if (!ptrTensorType || !ptrTensorType.getElementType().isInteger(32)) {
      LLVM_DEBUG(llvm::dbgs() << "Pointer tensor is not a tensor of i32\n");
      return failure();
    }

    // 3. Verify mask tensor is a tensor of i1
    auto maskTensorType = dyn_cast<RankedTensorType>(maskTensor.getType());
    if (!maskTensorType || !maskTensorType.getElementType().isInteger(1)) {
      LLVM_DEBUG(llvm::dbgs() << "Mask tensor is not a tensor of i1\n");
      return failure();
    }

    // 4. Convert the pointer tensor (tensor of i32) to tensor of triton pointers
    // Get element type from the value to store
    auto valueType = cast<RankedTensorType>(valueToStore.getType());
    Type elementType = valueType.getElementType();

    // Create triton pointer type: !tt.ptr<elementType>
    triton::PointerType ptrType = triton::PointerType::get(elementType, 1);

    // Create tensor of pointers type
    RankedTensorType ptrTensorOfPtrsType = RankedTensorType::get(
        ptrTensorType.getShape(), ptrType, ptrTensorType.getEncoding());

    // Cast the i32 tensor to tensor of pointers
    Value ptrTensorOfPtrs = rewriter.create<rock::CastToPtrOp>(
        loc, ptrTensorOfPtrsType, pointerTensor);

    // 5. Create triton::StoreOp or triton::AtomicRMWOp depending on storeMethod
    auto storeMethod = op.getStoreMethod();
    if (storeMethod == rock::StoreMethod::AtomicAdd) {
      // Use FADD for floating point, ADD for integer
      triton::RMWOp rmwOp =
          elementType.isIntOrIndex() ? triton::RMWOp::ADD : triton::RMWOp::FADD;
      // AtomicRMWOp returns the old value, but we don't need it
      rewriter.create<triton::AtomicRMWOp>(
          loc, valueType, rmwOp, ptrTensorOfPtrs, valueToStore, maskTensor,
          triton::MemSemantic::RELAXED, triton::MemSyncScope::GPU);
    } else if (storeMethod == rock::StoreMethod::AtomicMax) {
      // Use MAX for signed int, UMAX for unsigned int
      // For floating point, Triton doesn't have a direct FMAX atomic,
      // so we use MAX (which may need special handling downstream)
      triton::RMWOp rmwOp;
      if (elementType.isUnsignedInteger()) {
        rmwOp = triton::RMWOp::UMAX;
      } else {
        // Signed integer or floating point - use MAX
        rmwOp = triton::RMWOp::MAX;
      }
      rewriter.create<triton::AtomicRMWOp>(
          loc, valueType, rmwOp, ptrTensorOfPtrs, valueToStore, maskTensor,
          triton::MemSemantic::RELAXED, triton::MemSyncScope::GPU);
    } else {
      // Default: StoreMethod::Set - regular store
      // Signature: (ptr, value, mask, boundaryCheck, cache, evict)
      rewriter.create<triton::StoreOp>(
          loc, ptrTensorOfPtrs, valueToStore, maskTensor,
          /*boundaryCheck=*/ArrayRef<int32_t>{},
          /*cache=*/triton::CacheModifier::NONE,
          /*evict=*/triton::EvictionPolicy::NORMAL);
    }

    // Replace the op with the stored value (the result represents the stored tensor)
    rewriter.replaceOp(op, valueToStore);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReturnOpRewritePattern - Update return ops to return nothing and update
// the parent function signature to return void
//===----------------------------------------------------------------------===//
struct ReturnOpRewritePattern : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern<func::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::ReturnOp returnOp,
                                PatternRewriter &rewriter) const override {
    // Only convert return ops that have operands
    if (returnOp.getOperands().empty())
      return failure();

    // Update the parent function's signature to return void
    auto funcOp = returnOp->getParentOfType<func::FuncOp>();
    if (funcOp && funcOp.getFunctionType().getNumResults() > 0) {
      FunctionType newFuncType = FunctionType::get(
          rewriter.getContext(), funcOp.getFunctionType().getInputs(),
          /*results=*/{});
      rewriter.modifyOpInPlace(funcOp,
                               [&]() { funcOp.setFunctionType(newFuncType); });
    }

    rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp);
    return success();
  }
};
} // end anonymous namespace

void RockToTTIRPass::runOnOperation() {
  MLIRContext *ctx = &getContext();

  auto funcOp = getOperation();
  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic())) {
    return;
  }

  // First, apply rock.fill -> tt.splat conversion using greedy rewriting.
  // This must be done before partial conversion because the pattern uses
  // replaceAllUsesExcept which isn't supported in conversion rollback mode.
  {
    RewritePatternSet fillPatterns(ctx);
    fillPatterns.add<RockFillOpRewritePattern>(ctx);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(fillPatterns)))) {
      return signalPassFailure();
    }
  }

  // Apply rock.make_range -> tt.make_range conversion using greedy rewriting.
  {
    RewritePatternSet makeRangePatterns(ctx);
    makeRangePatterns.add<RockMakeRangeOpRewritePattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(makeRangePatterns)))) {
      return signalPassFailure();
    }
  }

  ConversionTarget target(*ctx);

  // Mark Rock ops as illegal - they should be converted
  target.addIllegalOp<tensor::SplatOp>();
  target.addIllegalOp<rock::BroadcastOp>();
  target.addIllegalOp<rock::WorkgroupIdOp>();
  target.addIllegalOp<rock::BlockwiseLoadTilePtrOp>();
  target.addIllegalOp<rock::BlockwiseGemmOp>();
  // Note: rock::MakeRangeOp is already converted in the greedy rewrite phase
  // above

  // Triton and Rock dialects are legal (Rock for now, will be converted later)
  target.addLegalDialect<triton::TritonDialect>();
  target.addLegalDialect<rock::RockDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<arith::ArithDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<TensorSplatOpRewritePattern>(ctx);
  patterns.add<RockBroadCastOpRewritePattern>(ctx);
  patterns.add<RockWorkgroupIdOpRewritePattern>(ctx);
  patterns.add<RockLoadTilePtrOpRewritePattern>(ctx);
  patterns.add<RockBlockwiseGemmOpRewritePattern>(ctx);
  // Note: RockMakeRangeOpRewritePattern is already applied in greedy rewrite
  // above

  // Apply partial conversion - convert tensor.splat and Rock ops to Triton ops
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }

  // Second conversion step: convert the micro kernel loop
  // by converting the scf.for op to a scf.for op with iter_args and
  // yield and rewrite the store tile ptr op to triton::store op.
  ConversionTarget target2(*ctx);
  target2.addLegalDialect<scf::SCFDialect>();
  target2.addLegalDialect<func::FuncDialect>();
  target2.addLegalDialect<arith::ArithDialect>();
  target2.addLegalDialect<rock::RockDialect>();
  target2.addLegalDialect<triton::TritonDialect>();
  target2.addLegalDialect<bufferization::BufferizationDialect>();
  target2.addLegalOp<memref::ExtractAlignedPointerAsIndexOp>();
  target2.addDynamicallyLegalOp<scf::ForOp>(
      [](scf::ForOp op) { return op.getNumResults() > 0; });
  target2.addIllegalOp<rock::BlockwiseStoreTilePtrOp>();
  target2.addDynamicallyLegalOp<func::ReturnOp>([](func::ReturnOp op) {
    return op.getOperands().empty();
  });

  RewritePatternSet patterns2(ctx);
  patterns2.add<RockMicroKernelOpRewritePattern>(ctx);
  patterns2.add<RockStoreTilePtrOpRewritePattern>(ctx);
  patterns2.add<ReturnOpRewritePattern>(ctx);
  if (failed(applyFullConversion(getOperation(), target2,
                                    std::move(patterns2)))) {
    return signalPassFailure();
  }
}