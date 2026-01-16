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
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
// #include "triton/Dialect/Triton/IR/TritonEnums.h"

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
using namespace mlir::bufferization;
using namespace mlir::memref;

namespace {
struct RockToTTIRPass
    : public rock::impl::RockToTTIRPassBase<RockToTTIRPass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// RockArithOpRewritePattern - Convert rock.arith_op to arith operations
//===----------------------------------------------------------------------===//
struct RockArithOpRewritePattern
    : public OpRewritePattern<rock::ArithOp> {
  using OpRewritePattern<rock::ArithOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::ArithOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    StringRef opName = op.getName();
    
    // Get operands and result type
    ValueRange operands = op.getOperands();
    Type resultType = op.getResult().getType();
    
    // Convert memref types to tensor types, converting index to i32
    SmallVector<Value> tensorOperands;
    tensorOperands.reserve(operands.size());
    Type i32Type = rewriter.getI32Type();
    
    for (Value operand : operands) {
      Type operandType = operand.getType();
      if (auto memrefType = dyn_cast<MemRefType>(operandType)) {
        // Convert memref to tensor first
        Type originalTensorType = getTensorTypeFromMemRefType(memrefType);
        Value tensor = ToTensorOp::create(rewriter, loc, originalTensorType, operand,
                                          /*restrict=*/true, /*writable=*/false);
        
        // If memref element type is index, convert tensor<...xindex> to tensor<...xi32>
        if (memrefType.getElementType().isIndex()) {
          auto tensorRankedType = cast<RankedTensorType>(originalTensorType);
          auto i32TensorType = RankedTensorType::get(
              tensorRankedType.getShape(), i32Type, tensorRankedType.getEncoding());
          tensor = rewriter.create<arith::IndexCastOp>(loc, i32TensorType, tensor);
        }
        
        tensorOperands.push_back(tensor);
      } else if (operandType.isIndex()) {
        // Convert scalar index to i32
        Value i32Value = rewriter.create<arith::IndexCastOp>(loc, i32Type, operand);
        tensorOperands.push_back(i32Value);
      } else {
        // Keep non-memref, non-index operands as-is
        tensorOperands.push_back(operand);
      }
    }
    
    // Convert result type from memref to tensor if needed, converting index to i32
    Type tensorResultType = resultType;
    if (auto memrefType = dyn_cast<MemRefType>(resultType)) {
      tensorResultType = getTensorTypeFromMemRefType(memrefType);
      // If memref element type is index, convert to i32 tensor
      if (memrefType.getElementType().isIndex()) {
        auto tensorRankedType = cast<RankedTensorType>(tensorResultType);
        tensorResultType = RankedTensorType::get(
            tensorRankedType.getShape(), i32Type, tensorRankedType.getEncoding());
      }
    } else if (resultType.isIndex()) {
      // If result is scalar index, convert to i32
      tensorResultType = i32Type;
    }
    
    // Create the corresponding arith operation based on the name
    Value result;
    
    if (opName == "AddIOp") {
      if (tensorOperands.size() != 2)
        return failure();
      result = rewriter.create<AddIOp>(loc, tensorOperands[0], tensorOperands[1]);
    } else if (opName == "SubIOp") {
      if (tensorOperands.size() != 2)
        return failure();
      result = rewriter.create<SubIOp>(loc, tensorOperands[0], tensorOperands[1]);
    } else if (opName == "MulIOp") {
      if (tensorOperands.size() != 2)
        return failure();
      result = rewriter.create<MulIOp>(loc, tensorOperands[0], tensorOperands[1]);
    } else if (opName == "DivSIOp") {
      if (tensorOperands.size() != 2)
        return failure();
      result = rewriter.create<DivSIOp>(loc, tensorOperands[0], tensorOperands[1]);
    } else if (opName == "DivUIOp") {
      if (tensorOperands.size() != 2)
        return failure();
      result = rewriter.create<DivUIOp>(loc, tensorOperands[0], tensorOperands[1]);
    } else if (opName == "RemSIOp") {
      if (tensorOperands.size() != 2)
        return failure();
      result = rewriter.create<RemSIOp>(loc, tensorOperands[0], tensorOperands[1]);
    } else if (opName == "RemUIOp") {
      if (tensorOperands.size() != 2)
        return failure();
      result = rewriter.create<RemUIOp>(loc, tensorOperands[0], tensorOperands[1]);
    } else if (opName == "AddFOp") {
      if (tensorOperands.size() != 2)
        return failure();
      result = rewriter.create<AddFOp>(loc, tensorOperands[0], tensorOperands[1]);
    } else if (opName == "SubFOp") {
      if (tensorOperands.size() != 2)
        return failure();
      result = rewriter.create<SubFOp>(loc, tensorOperands[0], tensorOperands[1]);
    } else if (opName == "MulFOp") {
      if (tensorOperands.size() != 2)
        return failure();
      result = rewriter.create<MulFOp>(loc, tensorOperands[0], tensorOperands[1]);
    } else if (opName == "DivFOp") {
      if (tensorOperands.size() != 2)
        return failure();
      result = rewriter.create<DivFOp>(loc, tensorOperands[0], tensorOperands[1]);
    } else if (opName == "RemFOp") {
      if (tensorOperands.size() != 2)
        return failure();
      result = rewriter.create<RemFOp>(loc, tensorOperands[0], tensorOperands[1]);
    } else if (opName == "ConstantIndexOp") {
      // ConstantIndexOp uses the constantValue attribute
      // Convert to i32 constant instead of index
      auto constantValueOpt = op.getConstantValue();
      if (!constantValueOpt.has_value())
        return failure();
      Attribute constantValue = constantValueOpt.value();
      if (auto intAttr = dyn_cast<IntegerAttr>(constantValue)) {
        // Create i32 constant instead of index constant
        Value indexConst = rewriter.create<ConstantIndexOp>(loc, intAttr.getInt());
        result = rewriter.create<arith::IndexCastOp>(loc, i32Type, indexConst);
      } else {
        return failure();
      }
    } else if (opName == "ConstantIntOp") {
      // ConstantOp uses the constantValue attribute
      auto constantValueOpt = op.getConstantValue();
      if (!constantValueOpt.has_value())
        return failure();
      Attribute constantValue = constantValueOpt.value();
      result = rewriter.create<ConstantOp>(loc, cast<TypedAttr>(constantValue));
    } else {
      // Unknown operation name
      return op.emitError() << "Unknown arith operation: " << opName;
    }
    
    // If result is a tensor of index, convert to i32 tensor
    if (auto resultTensorType = dyn_cast<RankedTensorType>(result.getType())) {
      if (resultTensorType.getElementType().isIndex()) {
        auto i32TensorType = RankedTensorType::get(
            resultTensorType.getShape(), i32Type, resultTensorType.getEncoding());
        result = rewriter.create<arith::IndexCastOp>(loc, i32TensorType, result);
      }
    } else if (result.getType().isIndex()) {
      // If result is scalar index, convert to i32
      result = rewriter.create<arith::IndexCastOp>(loc, i32Type, result);
    }
    
    // If the result type is a memref but we got a tensor, convert back
    // (though typically we want to keep tensors for Triton)
    if (isa<MemRefType>(resultType) && isa<TensorType>(result.getType())) {
      // If original result was memref<...xindex>, we need to convert i32 tensor back to index memref
      auto originalMemrefType = cast<MemRefType>(resultType);
      if (originalMemrefType.getElementType().isIndex()) {
        // Convert i32 tensor back to index tensor, then to memref
        auto resultTensorType = cast<RankedTensorType>(result.getType());
        auto indexTensorType = RankedTensorType::get(
            resultTensorType.getShape(), rewriter.getIndexType(), resultTensorType.getEncoding());
        Value indexTensor = rewriter.create<arith::IndexCastOp>(loc, indexTensorType, result);
        Value memrefResult = ToBufferOp::create(rewriter, loc, resultType, indexTensor);
        rewriter.replaceOp(op, memrefResult);
        return success();
      }
      // Convert tensor back to memref if needed
      Value memrefResult = ToBufferOp::create(rewriter, loc, resultType, result);
      rewriter.replaceOp(op, memrefResult);
    } else {
      rewriter.replaceOp(op, result);
    }
    
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RockSplatOpRewritePattern - Convert rock.splat to tt.splat
//===----------------------------------------------------------------------===//
struct RockSplatOpRewritePattern
    : public OpRewritePattern<rock::SplatOp> {
  using OpRewritePattern<rock::SplatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::SplatOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // llvm::errs() << "### RockSplatOpRewritePattern\n";
    
    // Get the source scalar value
    Value src = op.getSrc();
    
    // Get the result memref type and convert it to tensor type
    Type resultType = op.getResult().getType();
    if (auto memrefType = dyn_cast<MemRefType>(resultType)) {
      // Convert memref type to tensor type
      Type tensorType = getTensorTypeFromMemRefType(memrefType);
      
      // Check if we need to convert index to i32
      // If source is index and result memref element type is index, 
      // we need to convert to i32 for tt.splat
      Value srcToSplat = src;
      Type finalTensorType = tensorType;
      
      if (src.getType().isIndex() && memrefType.getElementType().isIndex()) {
        // Convert index to i32
        Type i32Type = rewriter.getI32Type();
        
        // Create tensor type with i32 element type
        auto tensorRankedType = cast<RankedTensorType>(tensorType);
        finalTensorType = RankedTensorType::get(
            tensorRankedType.getShape(), i32Type, tensorRankedType.getEncoding());
        
        // Convert index to i32
        // TODO: Does triton use i32? Or should we use i64?
        srcToSplat = rewriter.create<arith::IndexCastOp>(
            loc, i32Type, src);
      }
      
      // Create tt.splat operation
      Value result = triton::SplatOp::create(rewriter, loc, finalTensorType, srcToSplat);

      // llvm::errs() << "====> result:\n";
      // result.dump();
      
      rewriter.replaceOp(op, result);
      return success();
    }
    
    return failure();
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
    
    // Get the source memref
    Value src = op.getSrc();
    
    // Get the result memref type and convert it to tensor type
    Type resultType = op.getResult().getType();
    auto resultMemrefType = dyn_cast<MemRefType>(resultType);
    if (!resultMemrefType)
      return failure();
    
    // Get the source memref type
    auto srcMemrefType = dyn_cast<MemRefType>(src.getType());
    if (!srcMemrefType)
      return failure();
    
    // Convert source memref to tensor
    Type srcTensorType = getTensorTypeFromMemRefType(srcMemrefType);
    Value srcTensor = ToTensorOp::create(rewriter, loc, srcTensorType, src,
                                        /*restrict=*/true, /*writable=*/false);
    
    // Convert result memref type to tensor type
    Type resultTensorType = getTensorTypeFromMemRefType(resultMemrefType);
    
    // Handle index to i32 conversion for both source and result
    Type i32Type = rewriter.getI32Type();
    Value finalSrcTensor = srcTensor;
    Type finalResultTensorType = resultTensorType;
    
    if (srcMemrefType.getElementType().isIndex()) {
      // Convert source tensor from index to i32
      auto srcTensorRankedType = cast<RankedTensorType>(srcTensorType);
      auto i32SrcTensorType = RankedTensorType::get(
          srcTensorRankedType.getShape(), i32Type, srcTensorRankedType.getEncoding());
      finalSrcTensor = rewriter.create<arith::IndexCastOp>(loc, i32SrcTensorType, srcTensor);
    }
    
    if (resultMemrefType.getElementType().isIndex()) {
      // Convert result tensor type from index to i32
      auto resultTensorRankedType = cast<RankedTensorType>(resultTensorType);
      finalResultTensorType = RankedTensorType::get(
          resultTensorRankedType.getShape(), i32Type, resultTensorRankedType.getEncoding());
    }
    
    // Create tt.broadcast operation
    Value result = triton::BroadcastOp::create(rewriter, loc, finalResultTensorType, finalSrcTensor);
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RockLoadTilePtrOpRewritePattern - Convert rock.blockwise_load_tile_ptr to tt.load
//===----------------------------------------------------------------------===//
struct RockLoadTilePtrOpRewritePattern
    : public OpRewritePattern<rock::BlockwiseLoadTilePtrOp> {
  using OpRewritePattern<rock::BlockwiseLoadTilePtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::BlockwiseLoadTilePtrOp op,
                                PatternRewriter &rewriter) const override {
    llvm::errs() << "### RockLoadTilePtrOpRewritePattern\n";

    Location loc = op.getLoc();
    
    // Get operands
    Value pointerTensor = op.getPointerTensor();
    Value maskTensor = op.getMaskTensor();
    Value destRegisters = op.getDestRegisters();
    llvm::errs() << "destRegisters:\n";
    destRegisters.dump();
    
    // Get the element type from destRegisters
    auto destMemrefType = dyn_cast<MemRefType>(destRegisters.getType());
    if (!destMemrefType)
      return failure();
    
    Type elementType = destMemrefType.getElementType();
    
    // Convert pointerTensor from memref<...xindex> to tensor<...xindex>
    auto ptrMemrefType = dyn_cast<MemRefType>(pointerTensor.getType());
    if (!ptrMemrefType || !ptrMemrefType.getElementType().isIndex())
      return failure();
    
    Type ptrTensorType = getTensorTypeFromMemRefType(ptrMemrefType);
    Value ptrTensor = ToTensorOp::create(rewriter, loc, ptrTensorType, pointerTensor,
                                        /*restrict=*/true, /*writable=*/false);
    
    // Convert maskTensor from memref<...xi1> to tensor<...xi1>
    auto maskMemrefType = dyn_cast<MemRefType>(maskTensor.getType());
    if (!maskMemrefType || !maskMemrefType.getElementType().isInteger(1))
      return failure();
    
    Type maskTensorType = getTensorTypeFromMemRefType(maskMemrefType);
    Value maskTensorValue = ToTensorOp::create(rewriter, loc, maskTensorType, maskTensor,
                                             /*restrict=*/true, /*writable=*/false);
    
    // Create pointer type: !tt.ptr<elementType>
    // Use address space 1 (global) as default for Triton
    PointerType ptrType = PointerType::get(elementType, 1);
    
    // Create tensor of pointers: tensor<...x!tt.ptr<elementType>>
    auto ptrTensorRankedType = cast<RankedTensorType>(ptrTensorType);
    RankedTensorType ptrTensorOfPtrsType = RankedTensorType::get(
        ptrTensorRankedType.getShape(), ptrType, ptrTensorRankedType.getEncoding());
    
    // Convert index tensor to pointer tensor using unrealized_conversion_cast
    // This is a temporary cast that will be resolved by type conversion passes
    Value ptrTensorOfPtrs = rewriter.create<UnrealizedConversionCastOp>(
        loc, ptrTensorOfPtrsType, ptrTensor).getResult(0);
    
    // Convert destRegisters type from memref to tensor
    Type resultTensorType = getTensorTypeFromMemRefType(destMemrefType);
    
    // Create tt.load operation
    // LoadOp takes: ptr, mask (optional), other (optional), boundaryCheck, padding, cache, evict, isVolatile
    // Create attributes with default values
    auto boundaryCheckAttr = rewriter.getDenseI32ArrayAttr({});
    auto cacheAttr = triton::CacheModifierAttr::get(rewriter.getContext(), triton::CacheModifier::NONE);
    auto evictAttr = triton::EvictionPolicyAttr::get(rewriter.getContext(), triton::EvictionPolicy::NORMAL);
    auto isVolatileAttr = rewriter.getBoolAttr(false);
    
    Value result = rewriter.create<triton::LoadOp>(loc, resultTensorType, ptrTensorOfPtrs,
                                          maskTensorValue, /*other=*/Value(),
                                          boundaryCheckAttr,
                                          /*padding=*/nullptr,
                                          cacheAttr,
                                          evictAttr,
                                          isVolatileAttr);

    llvm::errs() << "====> result:\n";
    result.dump();
    
    // Convert result back to memref if needed (destRegisters is a memref)
    Value finalResult = result;
    // if (isa<MemRefType>(destRegisters.getType()) && isa<TensorType>(result.getType())) {
    //   finalResult = ToBufferOp::create(rewriter, loc, destRegisters.getType(), result);
    // }
    
    // Replace all uses of destRegisters with the result from tt.load
    rewriter.replaceAllUsesExcept(destRegisters, finalResult, op);

    // for (OpOperand &use : destRegisters.getUses()) {  
    //   llvm::errs() << "use:\n";
    //   use.getOwner()->dump();
    // }

    llvm::errs() << "replaceAllUsesWith:\n";
    result.getDefiningOp()->getParentOfType<func::FuncOp>().dump();
    rewriter.eraseOp(op);
    
    // Erase the original BlockwiseLoadTilePtrOp

    llvm::errs() << "Im done:\n";
    result.getDefiningOp()->getParentOfType<func::FuncOp>().dump();

    
    return success();
  }
};

} // end anonymous namespace

void RockToTTIRPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  
  // Mark RockArithOp, RockSplatOp, RockBroadcastOp, and RockLoadTilePtrOp as illegal - they should be converted
  target.addIllegalOp<rock::ArithOp>();
  target.addIllegalOp<rock::SplatOp>();
  target.addIllegalOp<rock::BroadcastOp>();
  target.addIllegalOp<rock::BlockwiseLoadTilePtrOp>();
  
  // Triton and Rock dialects are legal (Rock for now, will be converted later)
  target.addLegalDialect<triton::TritonDialect>();
  target.addLegalDialect<rock::RockDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<bufferization::BufferizationDialect>();
  target.addLegalDialect<memref::MemRefDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<RockArithOpRewritePattern>(ctx);
  patterns.add<RockSplatOpRewritePattern>(ctx);
  patterns.add<RockBroadCastOpRewritePattern>(ctx);
  patterns.add<RockLoadTilePtrOpRewritePattern>(ctx);
  
  // Apply partial conversion - convert RockArithOp, RockSplatOp, and RockLoadTilePtrOp, keep rest as-is
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}