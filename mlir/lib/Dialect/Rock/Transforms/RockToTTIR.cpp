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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeUtilities.h"
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
struct RockToTTIRPass : public rock::impl::RockToTTIRPassBase<RockToTTIRPass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// RockArithOpRewritePattern - Convert rock.arith_op to arith operations
//===----------------------------------------------------------------------===//
struct RockArithOpRewritePattern : public OpRewritePattern<rock::ArithOp> {
  using OpRewritePattern<rock::ArithOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::ArithOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    StringRef opName = op.getName();
    Type resultType = op.getResult().getType();
    Value result;

    // Handle ConstantIntOp
    if (opName == "ConstantIntOp") {
      auto constantValueOpt = op.getConstantValue();
      if (!constantValueOpt.has_value())
        return failure();
      Attribute constantValue = constantValueOpt.value();
      result = rewriter.create<arith::ConstantOp>(
          loc, cast<TypedAttr>(constantValue));
    }
    // Handle ConstantIndexOp
    else if (opName == "ConstantIndexOp") {
      auto constantValueOpt = op.getConstantValue();
      if (!constantValueOpt.has_value())
        return failure();
      Attribute constantValue = constantValueOpt.value();
      if (auto intAttr = dyn_cast<IntegerAttr>(constantValue)) {
        result = rewriter.create<arith::ConstantIndexOp>(loc, intAttr.getInt());
      } else {
        return failure();
      }
    }
    // Handle MulIOp
    else if (opName == "MulIOp") {
      ValueRange operands = op.getOperands();
      if (operands.size() != 2)
        return failure();

      // Convert memref operands to tensors if needed, handling index to i32
      // conversion
      SmallVector<Value> tensorOperands;
      tensorOperands.reserve(operands.size());
      for (Value operand : operands) {
        if (auto memrefType = dyn_cast<MemRefType>(operand.getType())) {
          Type tensorType = getTensorTypeFromMemRefType(memrefType);
          Value tensor =
              ToTensorOp::create(rewriter, loc, tensorType, operand,
                                 /*restrict=*/true, /*writable=*/false);
          tensorOperands.push_back(tensor);
        } else {
          tensorOperands.push_back(operand);
        }
      }

      result = rewriter.create<arith::MulIOp>(loc, tensorOperands[0],
                                              tensorOperands[1]);
    }
    // Handle AddIOp
    else if (opName == "AddIOp") {
      ValueRange operands = op.getOperands();
      if (operands.size() != 2)
        return failure();

      // Convert memref operands to tensors if needed, handling index to i32
      // conversion Type i32Type = rewriter.getI32Type();
      SmallVector<Value> tensorOperands;
      tensorOperands.reserve(operands.size());
      for (Value operand : operands) {
        if (auto memrefType = dyn_cast<MemRefType>(operand.getType())) {
          Type tensorType = getTensorTypeFromMemRefType(memrefType);
          Value tensor =
              ToTensorOp::create(rewriter, loc, tensorType, operand,
                                 /*restrict=*/true, /*writable=*/false);
          tensorOperands.push_back(tensor);
        } else {
          tensorOperands.push_back(operand);
        }
      }

      result = rewriter.create<arith::AddIOp>(loc, tensorOperands[0],
                                              tensorOperands[1]);
    } else {
      // Unknown operation name
      return failure();
    }

    SmallVector<Operation *> users;
    for (Operation *user : op->getUsers()) {
      users.push_back(user);
    }
    bool allUsersExpectMemref = true;
    for (Operation *user : users) {
      if (user->getNumOperands() == 0 ||
          !isa<MemRefType>(user->getOperand(0).getType())) {
        allUsersExpectMemref = false;
        break;
      }
    }
    if (allUsersExpectMemref) {
      // We need to insert a bufferization.to_buffer op after the result
      Value resultBufferized =
          ToBufferOp::create(rewriter, loc, resultType, result);
      result = resultBufferized;
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RockSplatOpRewritePattern - Convert rock.splat to tt.splat
//===----------------------------------------------------------------------===//
struct RockSplatOpRewritePattern : public OpRewritePattern<rock::SplatOp> {
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

      // By definition, SplatOp operand must be an integer type, it cannot be
      // index!
      if (src.getType().isIndex() && memrefType.getElementType().isIndex()) {
        // Convert index to i32
        Type i32Type = rewriter.getI32Type();

        // Create tensor type with i32 element type
        auto tensorRankedType = cast<RankedTensorType>(tensorType);
        finalTensorType =
            RankedTensorType::get(tensorRankedType.getShape(), i32Type,
                                  tensorRankedType.getEncoding());

        // Convert index to i32
        // TODO: Does triton use i32? Or should we use i64?
        // srcToSplat = rewriter.create<arith::IndexCastOp>(
        //     loc, i32Type, src);
      }

      // Create tt.splat operation
      Value result =
          triton::SplatOp::create(rewriter, loc, finalTensorType, srcToSplat);

      // llvm::errs() << "====> result:\n";
      // result.dump();

      // Because now splat returns a tensor and user expected a memref, we need
      // to convert it back to memref
      Value resultMemref =
          ToBufferOp::create(rewriter, loc, resultType, result);

      rewriter.replaceOp(op, resultMemref);
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
    // Type i32Type = rewriter.getI32Type();
    Value finalSrcTensor = srcTensor;
    Type finalResultTensorType = resultTensorType;

    // if (srcMemrefType.getElementType().isIndex()) {
    // Convert source tensor from index to i32
    // auto srcTensorRankedType = cast<RankedTensorType>(srcT ensorType);
    // finalSrcTensor = RankedTensorType::get(
    // srcTensorRankedType.getShape(), rewriter.getIndexType(),
    // srcTensorRankedType.getEncoding());
    // finalSrcTensor = rewriter.create<arith::IndexCastOp>(loc,
    // i32SrcTensorType, srcTensor);
    // }

    // if (resultMemrefType.getElementType().isIndex()) {
    //   // Convert result tensor type from index to i32
    //   auto resultTensorRankedType = cast<RankedTensorType>(resultTensorType);
    //   finalResultTensorType = RankedTensorType::get(
    //       resultTensorRankedType.getShape(), rewriter.getIndexType(),
    //       resultTensorRankedType.getEncoding());
    // }

    // Create tt.broadcast operation
    Value result = triton::BroadcastOp::create(
        rewriter, loc, finalResultTensorType, finalSrcTensor);

    // BroadcastOp returns a tensor, we need to convert it to memref
    Value resultMemref = ToBufferOp::create(rewriter, loc, resultType, result);

    rewriter.replaceOp(op, resultMemref);
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
    Location loc = op.getLoc();
    // tt.get_program_id returns i32, but rock.workgroup_id returns index
    Value programId =
        triton::GetProgramIdOp::create(rewriter, loc, triton::ProgramIDDim::X);
    // Cast i32 to index to match the expected result type
    // Value indexResult =
    //     arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
    //     programId);
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

    // Get operands
    Value pointerTensor = op.getPointerTensor();
    Value maskTensor = op.getMaskTensor();
    Value destRegisters = op.getDestRegisters();

    // Get the element type from destRegisters
    auto destMemrefType = dyn_cast<MemRefType>(destRegisters.getType());
    if (!destMemrefType) {
      llvm::errs() << "Dest registers is not a memref\n";
      return failure();
    }

    Type elementType = destMemrefType.getElementType();

    // Convert pointerTensor from memref<...xindex> to tensor<...xindex>
    auto ptrMemrefType = dyn_cast<MemRefType>(pointerTensor.getType());
    if (!ptrMemrefType || !ptrMemrefType.getElementType().isInteger(32)) {
      llvm::errs() << "Pointer tensor is not a memref of i32\n";
      return failure();
    }

    Type ptrTensorType = getTensorTypeFromMemRefType(ptrMemrefType);
    Value ptrTensor =
        ToTensorOp::create(rewriter, loc, ptrTensorType, pointerTensor,
                           /*restrict=*/true, /*writable=*/false);

    // Convert maskTensor from memref<...xi1> to tensor<...xi1>
    auto maskMemrefType = dyn_cast<MemRefType>(maskTensor.getType());
    if (!maskMemrefType || !maskMemrefType.getElementType().isInteger(1)) {
      llvm::errs() << "Mask tensor is not a tensor of i1\n";
      return failure();
    }

    Type maskTensorType = getTensorTypeFromMemRefType(maskMemrefType);
    Value maskTensorValue =
        ToTensorOp::create(rewriter, loc, maskTensorType, maskTensor,
                           /*restrict=*/true, /*writable=*/false);

    // Create pointer type: !tt.ptr<elementType>
    // Use address space 1 (global) as default for Triton
    triton::PointerType ptrType = triton::PointerType::get(elementType, 1);

    // Create tensor of pointers: tensor<...x!tt.ptr<elementType>>
    auto ptrTensorRankedType = cast<RankedTensorType>(ptrTensorType);
    RankedTensorType ptrTensorOfPtrsType =
        RankedTensorType::get(ptrTensorRankedType.getShape(), ptrType,
                              ptrTensorRankedType.getEncoding());

    // Convert tensor of i32 to tensor of pointers
    Value ptrTensorOfPtrs =
        rewriter.create<rock::CastToPtrOp>(loc, ptrTensorOfPtrsType, ptrTensor);

    // Convert destRegisters type from memref to tensor
    Type resultTensorType = getTensorTypeFromMemRefType(destMemrefType);

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
        loc, resultTensorType, ptrTensorOfPtrs, maskTensorValue,
        /*other=*/Value(), boundaryCheckAttr,
        /*padding=*/nullptr, cacheAttr, evictAttr, isVolatileAttr);

    // The output of tt.load is a tensor, but the gemm will require a memref.
    Value memrefResult =
        ToBufferOp::create(rewriter, loc, destMemrefType, result);

    // Here we should be using the rewriter.replaceAllUsesExcept method, but
    // it crashes...
    for (Operation *user : destRegisters.getUsers()) {
      if (user == op) {
        continue;
      }
      if (isa<rock::BlockwiseGemmAccelOp>(user)) {
        // Search for the operand index of destRegisters in the gemm op
        for (OpOperand &operand : user->getOpOperands()) {
          if (operand.get() == destRegisters) {
            operand.set(memrefResult);
            break;
          }
        }
      }
    }

    // Erase the original BlockwiseLoadTilePtrOp
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RockBlockwiseGemmAccelOpRewritePattern - Convert rock.blockwise_gemm_accel to
// tt.blockwise_gemm_accel
//===----------------------------------------------------------------------===//
struct RockBlockwiseGemmAccelOpRewritePattern
    : public OpRewritePattern<rock::BlockwiseGemmAccelOp> {
  using OpRewritePattern<rock::BlockwiseGemmAccelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::BlockwiseGemmAccelOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get operands
    Value a = op.getMatrixA();
    Value b = op.getMatrixB();
    Value c = op.getMatrixC();

    // Get the element type from a, b, and c.
    auto aMemrefType = dyn_cast<MemRefType>(a.getType());
    auto bMemrefType = dyn_cast<MemRefType>(b.getType());
    auto cMemrefType = dyn_cast<MemRefType>(c.getType());
    if (!aMemrefType || !bMemrefType || !cMemrefType)
      return failure();

    Type aTensorType = getTensorTypeFromMemRefType(aMemrefType);
    Type bTensorType = getTensorTypeFromMemRefType(bMemrefType);
    Type cTensorType = getTensorTypeFromMemRefType(cMemrefType);

    Value cTensor = ToTensorOp::create(rewriter, loc, cTensorType, c,
                                       /*restrict=*/true, /*writable=*/false);
    Value aTensor = ToTensorOp::create(rewriter, loc, aTensorType, a,
                                       /*restrict=*/true, /*writable=*/false);
    Value bTensor = ToTensorOp::create(rewriter, loc, bTensorType, b,
                                       /*restrict=*/true, /*writable=*/false);

    // Create tt.blockwise_gemm_accel operation
    Value result = rewriter.create<triton::DotOp>(
        loc, cTensorType, aTensor, bTensor, cTensor,
        /*inputPrecision=*/triton::InputPrecision::IEEE,
        /*maxNumImpreciseAcc=*/0);

    // We dont use replaceOp because result has one result whereas op has none.
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RockMicroKernelOpRewritePattern 
//===----------------------------------------------------------------------===//
struct RockMicroKernelOpRewritePattern
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    
    // Make sure this is a microkernel
    bool isMicroKernel = false;
    Block *body = op.getBody();

    for (Operation &bodyOp : *body) {
      if (isa<triton::DotOp>(bodyOp)) {
        isMicroKernel = true;
        break;
      }
    }

    if (!isMicroKernel) {
      llvm::errs() << "Loop is not a microkernel\n";
      return failure();
    }

    // Get the parent function first.
    func::FuncOp func = op->getParentOfType<func::FuncOp>();
    if (!func) {
      llvm::errs() << "Cannot find the parent function\n";
      return failure();
    }

    // Find the output buffer by looking for rock::GpuAllocOp with f32 element type.
    Value outputBuffer;
    func.walk([&](rock::GpuAllocOp allocOp) {
      auto memrefType = dyn_cast<MemRefType>(allocOp.getResult().getType());
      if (memrefType && memrefType.getElementType().isF32() &&
          allocOp->getParentOp() == func) {
        outputBuffer = allocOp.getResult();
      }
    });
    if (!outputBuffer) {
      llvm::errs() << "Cannot find output buffer (rock::GpuAllocOp with f32 element type)\n";
      return failure();
    }

    SmallVector<Operation *> argPointers;
    rock::CastToPtrOp castToPtrOp = nullptr;
    // The first argument is the output buffer, which we need to convert to a tensor pointer.
    auto outputBufferType = cast<MemRefType>(outputBuffer.getType());
    auto outputTensorType = getTensorTypeFromMemRefType(outputBufferType);
    Value outputTensor = ToTensorOp::create(rewriter, loc, outputTensorType, outputBuffer,
                                             /*restrict=*/true, /*writable=*/false);
    Value outputTensorPointer = CastToPtrOp::create(rewriter, loc, outputTensorType, outputTensor);
    castToPtrOp = cast<CastToPtrOp>(outputTensorPointer.getDefiningOp());

    // Now, we need to find the arg pointers for A and B only (we dont need C as
    // it is not yielded from the loop).
    // This is hacky, but right now we just want to match triton generated IR.
    func.walk([&](memref::ExtractAlignedPointerAsIndexOp extractOp) {
      auto memrefType = dyn_cast<MemRefType>(extractOp.getSource().getType());
      if (!memrefType) {
        return;  // Skip ops without valid memref type
      }
      // C is F32, so do not add it here.
      if (memrefType.getElementType().isF16()) {
        argPointers.push_back(extractOp);      
      }
    });

    // Make sure all arg pointers are in the function body.
    for (Operation *argPointerOp : argPointers) {
      if (argPointerOp->getParentOp() != func) {
        llvm::errs() << "Arg pointer is not in the function body\n";
        return failure();
      }
    }

    // Make sure all arg pointers input operands are actually a function argument.
    for (Operation *argPointerOp : argPointers) {
      for (OpOperand &operand : argPointerOp->getOpOperands()) {
        if (!isa<BlockArgument>(operand.get())) {
          llvm::errs() << "Arg pointer input operand is not a function argument\n";
          return failure();
        }
      }
    }

    // We need to cast the argPointers to triton pointers, since the yield operands inside the loop
    // are triton pointers.
    // Then, we will create a new scf.for with the same body but with the arg pointers as the init args.
    SmallVector<Value> initArgs;
    initArgs.push_back(castToPtrOp->getResult(0));
    for (Operation *argPointerOp : argPointers) {
      Value result = argPointerOp->getResult(0);

      // The result should have one user that is an arith::IndexCastOp
      if (!result.hasOneUse()) {
        llvm::errs() << "Arg pointer result does not have exactly one user\n";
        return failure();
      }

      Operation *user = *result.getUsers().begin();
      auto indexCastOp = dyn_cast<arith::IndexCastOp>(user);
      if (!indexCastOp) {
        llvm::errs() << "Arg pointer user is not an IndexCastOp\n";
        return failure();
      }

      // The IndexCastOp result should have one user that is a triton::SplatOp
      if (!indexCastOp.getResult().hasOneUse()) {
        llvm::errs() << "IndexCastOp result does not have exactly one user\n";
        return failure();
      }

      Operation *splatUser = *indexCastOp.getResult().getUsers().begin();
      auto splatOp = dyn_cast<triton::SplatOp>(splatUser);
      if (!splatOp) {
        llvm::errs() << "IndexCastOp user is not a SplatOp\n";
        return failure();
      }

      // Get the memref type from the extract op to determine element type
      auto extractOp = cast<memref::ExtractAlignedPointerAsIndexOp>(argPointerOp);
      auto memrefType = cast<MemRefType>(extractOp.getSource().getType());
      Type elementType = memrefType.getElementType();

      // Create triton pointer type
      triton::PointerType ptrType = triton::PointerType::get(elementType, 1);

      // Create tensor of pointers type matching the SplatOp result shape
      auto splatResultType = cast<RankedTensorType>(splatOp.getResult().getType());
      RankedTensorType ptrTensorType = RankedTensorType::get(
          splatResultType.getShape(), ptrType, splatResultType.getEncoding());

      // Create CastToPtrOp from the SplatOp result
      rewriter.setInsertionPointAfter(splatOp);
      Value castResult = rewriter.create<rock::CastToPtrOp>(
          loc, ptrTensorType, splatOp.getResult());

      initArgs.push_back(castResult);
    }

    // Create new ForOp with same bounds but new init args.
    // The empty body builder avoids implicit scf.yield construction.
    auto newForOp = scf::ForOp::create(
        rewriter, loc, op.getLowerBound(), op.getUpperBound(), op.getStep(),
        initArgs, [](OpBuilder &, Location, Value, ValueRange) {});
    newForOp->setAttrs(op->getAttrs());

    // Move operations from old body to new body using splice.
    Block *oldBody = op.getBody();
    Block *newBody = newForOp.getBody();
    newBody->getOperations().splice(newBody->getOperations().begin(),
                                    oldBody->getOperations());

    // Replace old block arguments with new ones.
    // First argument is the induction variable.
    op.getInductionVar().replaceAllUsesWith(newForOp.getInductionVar());
    // Replace any existing iter args (old loop might have had some).
    // for (auto [oldArg, newArg] :
    //      llvm::zip(op.getRegionIterArgs(), newForOp.getRegionIterArgs())) {
    //   oldArg.replaceAllUsesWith(newArg);
    // }

    // Find the tt.load and tt.dot operations in the new body.
    SmallVector<triton::LoadOp> loadOps;
    triton::DotOp dotOp = nullptr;
    for (Operation &bodyOp : *newBody) {
      if (auto load = dyn_cast<triton::LoadOp>(bodyOp)) {
        loadOps.push_back(load);
      } else if (auto dot = dyn_cast<triton::DotOp>(bodyOp)) {
        dotOp = dot;
      }
    }

    if (loadOps.size() < 2) {
      llvm::errs() << "Expected at least 2 tt.load ops in the loop body\n";
      return failure();
    }
    if (!dotOp) {
      llvm::errs() << "Expected a tt.dot op in the loop body\n";
      return failure();
    }

    llvm::errs() << "OK\n";

    // Update the yield to yield the correct values:
    // 1. Input of the first tt.load (pointer tensor)
    // 2. Input of the second tt.load (pointer tensor)
    // 3. Output of the tt.dot
    auto yieldOp = cast<scf::YieldOp>(newBody->getTerminator());

    llvm::errs() << "OK\n";
    rewriter.setInsertionPoint(yieldOp);
    llvm::errs() << "OK\n";
    SmallVector<Value> yieldOperands;
    llvm::errs() << "OK\n";
    yieldOperands.push_back(loadOps[0].getPtr());  // Input of first tt.load
    llvm::errs() << "OK\n";
    yieldOperands.push_back(loadOps[1].getPtr());  // Input of second tt.load
    llvm::errs() << "OK\n";
    yieldOperands.push_back(dotOp.getResult());    // Output of tt.dot

    
    // Modify the yield op in place.
    yieldOp->setOperands(yieldOperands);

    llvm::errs() << "All done: " << yieldOperands.size() << "\n";
    op->getParentOfType<func::FuncOp>().dump();

    // Erase the old ForOp now that all uses are replaced.
    rewriter.eraseOp(op);

    return success();
  }
};
} // end anonymous namespace

void RockToTTIRPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);

  // Mark RockArithOp, RockSplatOp, RockBroadcastOp, and RockLoadTilePtrOp as
  // illegal - they should be converted
  target.addIllegalOp<rock::ArithOp>();
  target.addIllegalOp<rock::SplatOp>();
  target.addIllegalOp<rock::BroadcastOp>();
  target.addIllegalOp<rock::WorkgroupIdOp>();
  target.addIllegalOp<rock::BlockwiseLoadTilePtrOp>();
  target.addIllegalOp<rock::BlockwiseGemmAccelOp>();

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
  patterns.add<RockWorkgroupIdOpRewritePattern>(ctx);
  patterns.add<RockLoadTilePtrOpRewritePattern>(ctx);
  patterns.add<RockBlockwiseGemmAccelOpRewritePattern>(ctx);

  // Apply partial conversion - convert RockArithOp, RockSplatOp, and
  // RockLoadTilePtrOp, keep rest as-is
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
                                      return signalPassFailure();
                                    }
  
  ConversionTarget target2(*ctx);
  target2.addDynamicallyLegalOp<scf::ForOp>([](scf::ForOp op) {
    return op.getNumResults() > 0;
  });

  RewritePatternSet patterns2(ctx);
  patterns2.add<RockMicroKernelOpRewritePattern>(ctx);
  if (failed(applyPartialConversion(getOperation(), target2,
                                    std::move(patterns2)))) {
                                      return signalPassFailure();
                                    }
}