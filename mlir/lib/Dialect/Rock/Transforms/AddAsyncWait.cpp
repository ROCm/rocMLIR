//===- AddAsyncWait - MLIR Rock ops lowering passes -----===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//
//
// This pass adds async wait operations for LDS memory
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToROCDL/MathToROCDL.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseSet.h"


#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKADDASYNCWAITPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-add-async-wait"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockAddAsyncWaitPass
    : public rock::impl::RockAddAsyncWaitPassBase<
          RockAddAsyncWaitPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

int getWaitCount(Operation *insertionPoint) {
  return 0;

  // int waitCount = 0;
  // Operation* parent = insertionPoint->getParentOp();

  // parent->walk([&](Operation *op) {
  //   if (op->getBlock() == insertionPoint->getBlock() && 
  //       op->isBeforeInBlock(insertionPoint) && 
  //       isa<ThreadwiseReadIntoOp>(op)) {
  //     waitCount++;
  //   }    
  // });

  // if (waitCount == 0) waitCount = 63;  // best
  // else if (waitCount == 2) waitCount = 1; // best
  // else if (waitCount == 6) waitCount = 2; // best

  // if (waitCount == 0) waitCount = 0;  // best
  // else if (waitCount == 2) waitCount = 2; // best
  // else if (waitCount == 6) waitCount = 0; // best

  // return waitCount;
}

/// Get the buffer index from an ExtractMultiBufferOp.
/// If the select index is a constant, returns that index (modulo buffer count).
/// Otherwise, returns 0 as the default.
size_t getBufferIndex(rock::ExtractMultiBufferOp extractMultiBuffer) {
  auto buffers = extractMultiBuffer.getBuffers();
  if (buffers.empty()) {
    return 0;
  }
  
  // Check if the select index is a constant
  auto selectIndex = dyn_cast_or_null<arith::ConstantIndexOp>(
      extractMultiBuffer.getSelectIndex().getDefiningOp());
  
  if (selectIndex) {
    // Use the constant index value (modulo buffer count for safety)
    int64_t index = selectIndex.value() % buffers.size();
    LLVM_DEBUG(llvm::dbgs() << "MultiBufferOp select index is a constant: " << index << "\n");
    return static_cast<size_t>(index);
  }
  
  // Otherwise, default to index 0
  return 0;
}

/// Recursively traverse through ExtractMultiBufferOp and ViewOp to find
/// the underlying value and check if it's actually reading from LDS.
bool readsFromLDSValue(Value value) {
  LLVM_DEBUG(llvm::dbgs() << "readsFromLDSValue:" << value << "\n");
  // If the value comes from an ExtractMultiBufferOp, extract the selected buffer
  if (auto extractMultiBuffer = value.getDefiningOp<rock::ExtractMultiBufferOp>()) {
    return false;
    
    size_t bufferIndex = getBufferIndex(extractMultiBuffer);
    auto buffers = extractMultiBuffer.getBuffers();
    if (bufferIndex < buffers.size()) {
      Value selectedBuffer = buffers[bufferIndex];
      // Recursively check the selected buffer
      if (selectedBuffer == value)
        return readsFromLDSValue(selectedBuffer);
    }
    return false;
  }
  
  // If the value comes from a ViewOp, extract the source and recurse
  if (auto viewOp = value.getDefiningOp<memref::ViewOp>()) {
    Value source = viewOp.getSource();
    // Recursively check the source
    return readsFromLDSValue(source);
  }
     
  LLVM_DEBUG(llvm::dbgs() << "Ok, this is reading from LDS\n");
  // TODO: Add actual check for LDS read operations here
  // For now, if we reach a non-ExtractMultiBufferOp/ViewOp user, consider it a read
  return true;
}

/// Find the first use following the pattern:
/// ThreadwiseReadIntoOp -> dest (ExtractMultiBufferOp) -> first operand -> ViewOp -> input
/// Then find the first use of either the ViewOp's input or the ViewOp itself.
/// Returns nullptr if the pattern doesn't match or no use is found.
Operation* findFirstUseAfter(ThreadwiseReadIntoOp readOp) {
  Value dest = readOp.getDest();
  
  // Pattern-match: dest must be an ExtractMultiBufferOp
  auto extractMultiBuffer = dest.getDefiningOp<rock::ExtractMultiBufferOp>();
  if (!extractMultiBuffer) {
    LLVM_DEBUG(llvm::dbgs() << "ThreadwiseReadIntoOp dest is not an ExtractMultiBufferOp\n");
    return nullptr;
  }
  
  // Get the buffer based on the select index
  auto buffers = extractMultiBuffer.getBuffers();
  if (buffers.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "ExtractMultiBufferOp has no buffers\n");
    return nullptr;
  }
  
  size_t bufferIndex = getBufferIndex(extractMultiBuffer);
  Value selectedBuffer = buffers[bufferIndex];
  
  // Pattern-match: selected buffer must come from a memref::ViewOp
  auto viewOp = selectedBuffer.getDefiningOp<memref::ViewOp>();
  if (!viewOp) {
    LLVM_DEBUG(llvm::dbgs() << "Selected buffer is not a memref::ViewOp\n");
    return nullptr;
  }
  
  // Get the input of the ViewOp (the actual value we want to track)
  Value viewInput = viewOp.getSource();
  Value viewResult = viewOp.getResult();
  
  // Find the first use of either the ViewOp's input or the ViewOp itself
  Operation* firstUse = nullptr;
  Block* block = readOp->getBlock();
  
  // Check uses of the ViewOp's input
  for (OpOperand &use : viewInput.getUses()) {
    Operation* user = use.getOwner();
    // Skip ExtractMultiBufferOp that will result in a different buffer being used.
    if (auto userExtractMultiBuffer = dyn_cast<rock::ExtractMultiBufferOp>(user)) {
      size_t userBufferIndex = getBufferIndex(userExtractMultiBuffer);
      Value selectedBuffer2 = userExtractMultiBuffer.getBuffers()[userBufferIndex];
      if (selectedBuffer2 != use.get()) {
        LLVM_DEBUG(llvm::dbgs() << "1 - Skipping\n");
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "1 - NOT Skipping\n");
    }
    // Only consider uses in the same block that come after readOp
    if (user->getBlock() == block && readOp->isBeforeInBlock(user)) {
      if (!firstUse || user->isBeforeInBlock(firstUse)) {
        firstUse = user;
      }
    }
  }
  
  // Check uses of the ViewOp itself
  for (OpOperand &use : viewResult.getUses()) {
    Operation* user = use.getOwner();
    if (!readsFromLDSValue(use.get())) {
      continue;
    }

    // Only consider uses in the same block that come after readOp
    if (user->getBlock() == block && readOp->isBeforeInBlock(user)) {
      if (!firstUse || user->isBeforeInBlock(firstUse)) {
        firstUse = user;
      }
    }
  }
  
  return firstUse;
}

static LogicalResult addAsyncWait(func::FuncOp &func) {
  IRRewriter rewriter(func->getContext());

  // Find all ThreadwiseReadIntoOp operations
  SmallVector<rock::ThreadwiseReadIntoOp> readOps;
  func.walk([&](rock::ThreadwiseReadIntoOp op) {
    // Only add reads that write into LDS memory
    auto memSpace = dyn_cast_or_null<gpu::AddressSpaceAttr>(op.getDest().getType().getMemorySpace());
    if (memSpace && memSpace.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace()) {
      readOps.push_back(op);
    }    
  });

  // Track insertion points to avoid inserting multiple AsyncWaitOps at the same location
  llvm::DenseSet<Operation*> insertionPoints;
  int i = 0;

  for (auto readOp : readOps) {
    Operation* firstUse = findFirstUseAfter(readOp);
    
    if (!firstUse) {
      // If pattern doesn't match or no use found, skip this readOp
      LLVM_DEBUG(llvm::dbgs() << "Pattern doesn't match or no use found for ThreadwiseReadIntoOp\n");
      continue;
    }

    // Only insert one AsyncWaitOp per insertion point
    if (insertionPoints.contains(firstUse)) {
      continue;
    }
    insertionPoints.insert(firstUse);

    int waitToken = getWaitCount(firstUse);
    rewriter.setInsertionPoint(firstUse);
    rock::AsyncWaitOp::create(rewriter, firstUse->getLoc(), i);
    i++;
  }

  return success();
}

void RockAddAsyncWaitPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Only run this pass on GPU kernel functions.
  if (!func->hasAttr("kernel")) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping RockAddAsyncWaitPass on func with "
                               "no kernel attribute\n");
    return;
  }

  if (failed(addAsyncWait(func))) {
    return signalPassFailure();
  }
}
