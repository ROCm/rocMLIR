//===- MemrefToTensor - MLIR Rock ops lowering passes
//-------------------------===//
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
// This pass converts from memref to tensors and converts pointer arithmetic
// to use Triton pointer types.
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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKMEMREFTOTENSORPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-memref-to-tensor"

using namespace mlir;
using namespace mlir::rock;
using namespace mlir::triton;
using namespace mlir::arith;
using namespace mlir::bufferization;
using namespace mlir::memref;

namespace {

/// Helper to determine if a type is a tensor with pointer element type
static bool isTensorOfPointers(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return isa<triton::PointerType>(tensorType.getElementType());
  }
  return false;
}

/// Helper to get element type from a memref type
static Type getMemRefElementType(Type memrefType) {
  if (auto mrt = dyn_cast<MemRefType>(memrefType)) {
    return mrt.getElementType();
  }
  return nullptr;
}

struct RockMemrefToTensorPass
    : public rock::impl::RockMemrefToTensorPassBase<RockMemrefToTensorPass> {
  void runOnOperation() override;

private:
  /// Map from original i32 values to converted tt.ptr values
  IRMapping valueMapping;

  /// Process a single function
  void processFunction(func::FuncOp funcOp);
};

} // end anonymous namespace

void RockMemrefToTensorPass::processFunction(func::FuncOp funcOp) {
  valueMapping.clear();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  // Step 1: Find all extract_aligned_pointer_as_index patterns and collect info
  // before we modify any types
  struct ExtractInfo {
    memref::ExtractAlignedPointerAsIndexOp extractOp;
    unsigned argIndex;
    Type elementType;
    SmallVector<arith::IndexCastOp> indexCasts;
  };
  SmallVector<ExtractInfo> extractInfos;

  funcOp.walk([&](memref::ExtractAlignedPointerAsIndexOp extractOp) {
    Value memrefOperand = extractOp.getSource();

    // Check if this is a block argument
    // TODO(roctriton): input fusions
    auto blockArg = dyn_cast<BlockArgument>(memrefOperand);
    if (!blockArg)
      return;

    Type elementType = getMemRefElementType(memrefOperand.getType());
    if (!elementType)
      return;

    ExtractInfo info;
    info.extractOp = extractOp;
    info.argIndex = blockArg.getArgNumber();
    info.elementType = elementType;

    // Find all index_cast ops that use this extract
    for (Operation *user : extractOp.getResult().getUsers()) {
      if (auto indexCastOp = dyn_cast<arith::IndexCastOp>(user)) {
        Type resultType = indexCastOp.getResult().getType();
        if (resultType.isInteger(32)) {
          info.indexCasts.push_back(indexCastOp);
        }
      }
    }

    if (!info.indexCasts.empty()) {
      extractInfos.push_back(info);
    }
  });

  // If no patterns found, nothing to do
  if (extractInfos.empty())
    return;

  // Step 2: Change function signature from memrefs to tt.ptr
  FunctionType funcType = funcOp.getFunctionType();
  SmallVector<Type> newInputTypes;

  // Build a map from arg index to element type for conversion
  DenseMap<unsigned, Type> argElementTypes;
  for (const auto &info : extractInfos) {
    argElementTypes[info.argIndex] = info.elementType;
  }

  for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
    Type inputType = funcType.getInput(i);
    auto it = argElementTypes.find(i);
    if (it != argElementTypes.end()) {
      // Convert memref to tt.ptr
      auto ptrType = triton::PointerType::get(it->second, 1);
      newInputTypes.push_back(ptrType);
    } else {
      newInputTypes.push_back(inputType);
    }
  }

  // Update the function type
  auto newFuncType =
      FunctionType::get(ctx, newInputTypes, funcType.getResults());
  funcOp.setFunctionType(newFuncType);

  // Update block argument types
  Block &entryBlock = funcOp.front();
  for (auto [argIdx, elementType] : argElementTypes) {
    BlockArgument arg = entryBlock.getArgument(argIdx);
    auto ptrType = triton::PointerType::get(elementType, 1);
    arg.setType(ptrType);
  }

  // Step 3: Map the i32 values from index_cast to the new pointer block
  // arguments and schedule ops for removal
  SmallVector<Operation *, 8> opsToErase;
  SmallVector<Operation *, 8> extractOpsToErase; // Erase these last

  for (const auto &info : extractInfos) {
    Value newPtrArg = entryBlock.getArgument(info.argIndex);

    for (auto indexCastOp : info.indexCasts) {
      // Map the i32 result to the pointer argument
      valueMapping.map(indexCastOp.getResult(), newPtrArg);
      opsToErase.push_back(indexCastOp);
    }

    extractOpsToErase.push_back(info.extractOp);
  }

  // Step 4: Convert tt.splat operations that use mapped values
  funcOp.walk([&](triton::SplatOp splatOp) {
    Value src = splatOp.getSrc();
    Value mappedSrc = valueMapping.lookupOrNull(src);

    if (!mappedSrc)
      return;

    // This splat uses a pointer value, convert it
    Location loc = splatOp.getLoc();
    builder.setInsertionPoint(splatOp);

    auto resultType = dyn_cast<RankedTensorType>(splatOp.getResult().getType());
    if (!resultType)
      return;

    // Get the pointer type
    auto ptrType = dyn_cast<triton::PointerType>(mappedSrc.getType());
    if (!ptrType)
      return;

    // Create new tensor type with pointer element type
    auto newResultType = RankedTensorType::get(resultType.getShape(), ptrType,
                                               resultType.getEncoding());

    // Create new splat with pointer type
    Value newSplat =
        triton::SplatOp::create(builder, loc, newResultType, mappedSrc);

    // Map the old result to the new result
    valueMapping.map(splatOp.getResult(), newSplat);
    opsToErase.push_back(splatOp);
  });

  // Step 5: Propagate pointer types through bufferization.to_buffer ops
  bool changed = true;
  while (changed) {
    changed = false;
    funcOp.walk([&](bufferization::ToBufferOp toBufferOp) {
      if (llvm::is_contained(opsToErase, toBufferOp.getOperation()))
        return;

      Value src = toBufferOp.getTensor();
      Value mappedSrc = valueMapping.lookupOrNull(src);

      if (!mappedSrc)
        return;

      // Check if already mapped
      if (valueMapping.contains(toBufferOp.getResult()))
        return;

      // If the source is now a pointer tensor, propagate the mapping
      valueMapping.map(toBufferOp.getResult(), mappedSrc);
      opsToErase.push_back(toBufferOp);
      changed = true;
    });
  }

  // Step 6: Propagate through bufferization.to_tensor ops
  changed = true;
  while (changed) {
    changed = false;
    funcOp.walk([&](bufferization::ToTensorOp toTensorOp) {
      if (llvm::is_contained(opsToErase, toTensorOp.getOperation()))
        return;

      Value src = toTensorOp.getBuffer();
      Value mappedSrc = valueMapping.lookupOrNull(src);

      if (!mappedSrc)
        return;

      // Check if already mapped
      if (valueMapping.contains(toTensorOp.getResult()))
        return;

      // Map the tensor result to the pointer value
      valueMapping.map(toTensorOp.getResult(), mappedSrc);
      opsToErase.push_back(toTensorOp);
      changed = true;
    });
  }

  // Step 7: Convert arith.addi on pointer tensors to tt.addptr
  changed = true;
  while (changed) {
    changed = false;
    funcOp.walk([&](arith::AddIOp addOp) {
      // Skip if already scheduled for erasure
      if (llvm::is_contained(opsToErase, addOp.getOperation()))
        return;

      Value lhs = addOp.getLhs();
      Value rhs = addOp.getRhs();

      Value mappedLhs = valueMapping.lookupOrNull(lhs);
      Value mappedRhs = valueMapping.lookupOrNull(rhs);

      // Check if either operand is a pointer tensor (either mapped or directly)
      Value ptrOperand = nullptr;
      Value offsetOperand = nullptr;

      if (mappedLhs && isTensorOfPointers(mappedLhs.getType())) {
        ptrOperand = mappedLhs;
        offsetOperand = mappedRhs ? mappedRhs : rhs;
      } else if (mappedRhs && isTensorOfPointers(mappedRhs.getType())) {
        ptrOperand = mappedRhs;
        offsetOperand = mappedLhs ? mappedLhs : lhs;
      } else if (isTensorOfPointers(lhs.getType())) {
        // Direct pointer tensor (already converted)
        ptrOperand = lhs;
        offsetOperand = mappedRhs ? mappedRhs : rhs;
      } else if (isTensorOfPointers(rhs.getType())) {
        ptrOperand = rhs;
        offsetOperand = mappedLhs ? mappedLhs : lhs;
      }

      if (!ptrOperand)
        return;

      // Don't convert if offset is also a pointer (shouldn't happen)
      if (isTensorOfPointers(offsetOperand.getType()))
        return;

      Location loc = addOp.getLoc();
      builder.setInsertionPoint(addOp);

      // Create tt.addptr
      Value newAddPtr = triton::AddPtrOp::create(
          builder, loc, ptrOperand.getType(), ptrOperand, offsetOperand);

      valueMapping.map(addOp.getResult(), newAddPtr);
      opsToErase.push_back(addOp);
      changed = true;
    });
  }

  // Step 8: Propagate pointer tensors through remaining bufferization ops
  changed = true;
  while (changed) {
    changed = false;

    // Handle to_buffer of pointer tensors
    funcOp.walk([&](bufferization::ToBufferOp toBufferOp) {
      if (llvm::is_contained(opsToErase, toBufferOp.getOperation()))
        return;

      Value src = toBufferOp.getTensor();
      Value mappedSrc = valueMapping.lookupOrNull(src);
      Value ptrTensor = mappedSrc ? mappedSrc : src;

      if (!isTensorOfPointers(ptrTensor.getType()))
        return;

      // Check if already mapped
      if (valueMapping.contains(toBufferOp.getResult()))
        return;

      // Map the memref result to the pointer tensor
      valueMapping.map(toBufferOp.getResult(), ptrTensor);
      opsToErase.push_back(toBufferOp);
      changed = true;
    });

    // Handle memref.copy where source maps to pointer tensor
    funcOp.walk([&](memref::CopyOp copyOp) {
      if (llvm::is_contained(opsToErase, copyOp.getOperation()))
        return;

      Value src = copyOp.getSource();
      Value mappedSrc = valueMapping.lookupOrNull(src);

      if (!mappedSrc || !isTensorOfPointers(mappedSrc.getType()))
        return;

      // Check if already mapped
      if (valueMapping.contains(copyOp.getTarget()))
        return;

      // Map the destination to the pointer tensor
      valueMapping.map(copyOp.getTarget(), mappedSrc);
      opsToErase.push_back(copyOp);
      changed = true;
    });

    // Handle to_tensor ops whose memref source maps to a pointer tensor
    funcOp.walk([&](bufferization::ToTensorOp toTensorOp) {
      if (llvm::is_contained(opsToErase, toTensorOp.getOperation()))
        return;

      Value src = toTensorOp.getBuffer();
      Value mappedSrc = valueMapping.lookupOrNull(src);

      if (!mappedSrc || !isTensorOfPointers(mappedSrc.getType()))
        return;

      // Check if already mapped
      if (valueMapping.contains(toTensorOp.getResult()))
        return;

      valueMapping.map(toTensorOp.getResult(), mappedSrc);
      opsToErase.push_back(toTensorOp);
      changed = true;
    });

    // Handle rock.cast_to_ptr - if input maps to pointer tensor, replace it
    funcOp.walk([&](rock::CastToPtrOp castOp) {
      if (llvm::is_contained(opsToErase, castOp.getOperation()))
        return;

      Value src = castOp.getSrc();
      Value mappedSrc = valueMapping.lookupOrNull(src);

      if (!mappedSrc || !isTensorOfPointers(mappedSrc.getType()))
        return;

      // Check if already mapped
      if (valueMapping.contains(castOp.getResult()))
        return;

      // The cast_to_ptr produces a pointer tensor, but we already have one
      valueMapping.map(castOp.getResult(), mappedSrc);
      opsToErase.push_back(castOp);
      changed = true;
    });
  }

  // Step 9: Update all remaining uses
  funcOp.walk([&](Operation *op) {
    // Skip ops we're about to erase
    if (llvm::is_contained(opsToErase, op))
      return;

    // Skip bufferization ops - they have strict type requirements
    if (isa<bufferization::ToBufferOp, bufferization::ToTensorOp>(op))
      return;

    // Skip memref ops - they require memref operands
    if (isa<memref::CopyOp>(op))
      return;

    bool needsUpdate = false;
    for (Value operand : op->getOperands()) {
      if (valueMapping.contains(operand)) {
        needsUpdate = true;
        break;
      }
    }

    if (!needsUpdate)
      return;

    // Update operands
    for (OpOperand &operand : op->getOpOperands()) {
      if (Value mapped = valueMapping.lookupOrNull(operand.get())) {
        operand.set(mapped);
      }
    }
  });

  // Erase the converted operations in reverse order
  for (auto it = opsToErase.rbegin(); it != opsToErase.rend(); ++it) {
    (*it)->erase();
  }

  // Erase extract ops last (since other ops depend on them)
  for (auto *op : extractOpsToErase) {
    op->erase();
  }
}

void RockMemrefToTensorPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  processFunction(funcOp);
}
