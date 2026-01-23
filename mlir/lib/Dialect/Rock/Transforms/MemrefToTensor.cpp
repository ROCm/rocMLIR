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
#include "llvm/Support/raw_ostream.h"

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

  /// Process a single kernel function (convert to tt.func)
  void processFunction(func::FuncOp funcOp);

  /// Fix up func.call ops in wrapper functions that reference converted kernels
  void fixupKernelCalls(ModuleOp moduleOp);
};

} // end anonymous namespace

void RockMemrefToTensorPass::processFunction(func::FuncOp funcOp) {
  valueMapping.clear();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  // Step 1: Find all extract_aligned_pointer_as_index patterns and collect info
  // Pattern: block_arg (tensor) -> to_buffer -> extract_ptr -> index_cast -> tt.splat
  struct ArgConversionInfo {
    unsigned argIndex;
    Type elementType;
    SmallVector<Value> valuesToReplace; // index_cast results to replace with block arg
    // Ops in the chain that need to be erased (in order: splats, index_cast, extract, to_buffer)
    SmallVector<triton::SplatOp> oldSplatOps;
    bufferization::ToBufferOp toBufferOp;
    memref::ExtractAlignedPointerAsIndexOp extractOp;
    arith::IndexCastOp indexCastOp;
  };
  SmallVector<ArgConversionInfo> argsToConvert;

  funcOp.walk([&](memref::ExtractAlignedPointerAsIndexOp extractOp) {
    Value memrefOperand = extractOp.getSource();

    // Trace through to_buffer to find the tensor block argument
    auto toBufferOp = memrefOperand.getDefiningOp<bufferization::ToBufferOp>();
    if (!toBufferOp)
      return;

    Value tensorOperand = toBufferOp.getTensor();
    auto blockArg = dyn_cast<BlockArgument>(tensorOperand);
    if (!blockArg)
      return;

    auto tensorType = dyn_cast<RankedTensorType>(tensorOperand.getType());
    if (!tensorType)
      return;

    // Find index_cast ops that use the extract result
    for (Operation *user : extractOp.getResult().getUsers()) {
      if (auto indexCastOp = dyn_cast<arith::IndexCastOp>(user)) {
        if (indexCastOp.getResult().getType().isInteger(32)) {
          // Found the pattern - record it
          ArgConversionInfo info;
          info.argIndex = blockArg.getArgNumber();
          info.elementType = tensorType.getElementType();
          info.valuesToReplace.push_back(indexCastOp.getResult());
          info.toBufferOp = toBufferOp;
          info.extractOp = extractOp;
          info.indexCastOp = indexCastOp;
          argsToConvert.push_back(info);
        }
      }
    }
  });

  if (argsToConvert.empty())
    return;

  // Step 2: Build new function type with tt.ptr arguments
  FunctionType funcType = funcOp.getFunctionType();
  SmallVector<Type> newInputTypes;
  DenseMap<unsigned, Type> argElementTypes;

  for (const auto &info : argsToConvert) {
    argElementTypes[info.argIndex] = info.elementType;
  }

  for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
    auto it = argElementTypes.find(i);
    if (it != argElementTypes.end()) {
      newInputTypes.push_back(triton::PointerType::get(it->second, 1));
    } else {
      newInputTypes.push_back(funcType.getInput(i));
    }
  }

  auto newFuncType = FunctionType::get(ctx, newInputTypes, funcType.getResults());

  // Collect attributes to copy
  SmallVector<NamedAttribute> attrsToKeep;
  for (NamedAttribute attr : funcOp->getAttrs()) {
    StringRef name = attr.getName();
    if (name == "function_type" || name == "sym_name" || name == "sym_visibility")
      continue;
    attrsToKeep.push_back(attr);
  }

  // Step 3: For each index_cast result, replace its users with the block argument
  // We do this BEFORE changing types so the old ops become dead
  Block &entryBlock = funcOp.front();
  for (auto &info : argsToConvert) {
    BlockArgument blockArg = entryBlock.getArgument(info.argIndex);
    auto ptrType = triton::PointerType::get(info.elementType, 1);

    for (Value oldValue : info.valuesToReplace) {
      // For each user of the index_cast result (like tt.splat), create a replacement
      for (OpOperand &use : llvm::make_early_inc_range(oldValue.getUses())) {
        Operation *user = use.getOwner();

        if (auto splatOp = dyn_cast<triton::SplatOp>(user)) {
          // Create new splat with pointer type
          builder.setInsertionPoint(splatOp);
          auto resultType = cast<RankedTensorType>(splatOp.getResult().getType());
          auto newResultType = RankedTensorType::get(
              resultType.getShape(), ptrType, resultType.getEncoding());
          Value newSplat = triton::SplatOp::create(
              builder, splatOp.getLoc(), newResultType, blockArg);

          // Map old splat result to new for downstream propagation
          valueMapping.map(splatOp.getResult(), newSplat);

          // Replace all uses of old splat with new splat
          splatOp.getResult().replaceAllUsesWith(newSplat);

          // Track old splat for later erasure
          info.oldSplatOps.push_back(splatOp);
        }
      }
    }

    // Update the block argument type
    blockArg.setType(ptrType);
  }

  // Erase the ops in the chain (users first, producers last)
  // Order: old splats -> index_cast -> extract_ptr -> to_buffer
  for (auto &info : argsToConvert) {
    // First erase the old splat ops (they use index_cast result)
    for (auto splatOp : info.oldSplatOps) {
      splatOp.erase();
    }
    // Then erase index_cast (uses extract_ptr result)
    if (info.indexCastOp)
      info.indexCastOp.erase();
    // Then erase extract_ptr (uses to_buffer result)
    if (info.extractOp)
      info.extractOp.erase();
    // Finally erase to_buffer (uses block arg)
    if (info.toBufferOp)
      info.toBufferOp.erase();
  }

  // Step 4: Create tt.func and move body
  builder.setInsertionPoint(funcOp);
  auto ttFuncOp = triton::FuncOp::create(
      builder, funcOp.getLoc(), funcOp.getName(), newFuncType, attrsToKeep);
  ttFuncOp->setAttr("noinline", builder.getBoolAttr(true));

  Region &oldRegion = funcOp.getBody();
  Region &newRegion = ttFuncOp.getBody();
  newRegion.takeBody(oldRegion);

  // Convert func.return to tt.return
  ttFuncOp.walk([&](func::ReturnOp returnOp) {
    builder.setInsertionPoint(returnOp);
    triton::ReturnOp::create(builder, returnOp.getLoc(), returnOp.getOperands());
    returnOp.erase();
  });

  funcOp.erase();

  // Continue with remaining transformations
  SmallVector<Operation *, 8> opsToErase;

  // Step 5: Propagate pointer types through bufferization.to_buffer ops
  bool changed = true;
  while (changed) {
    changed = false;
    ttFuncOp.walk([&](bufferization::ToBufferOp toBufferOp) {
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
    ttFuncOp.walk([&](bufferization::ToTensorOp toTensorOp) {
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
    ttFuncOp.walk([&](arith::AddIOp addOp) {
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
    ttFuncOp.walk([&](bufferization::ToBufferOp toBufferOp) {
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
    ttFuncOp.walk([&](memref::CopyOp copyOp) {
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
    ttFuncOp.walk([&](bufferization::ToTensorOp toTensorOp) {
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
    ttFuncOp.walk([&](rock::CastToPtrOp castOp) {
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
  ttFuncOp.walk([&](Operation *op) {
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
}

/// Fix up func.call ops that reference tt.func kernels.
/// Converts them to tt.call with proper pointer type conversions.
void RockMemrefToTensorPass::fixupKernelCalls(ModuleOp moduleOp) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  // Collect all func.call ops that need to be converted
  SmallVector<func::CallOp> callsToFix;
  moduleOp.walk([&](func::CallOp callOp) {
    // Check if the callee is a triton::FuncOp
    auto callee = moduleOp.lookupSymbol<triton::FuncOp>(callOp.getCallee());
    if (callee) {
      callsToFix.push_back(callOp);
    }
  });

  // Convert each call
  for (func::CallOp callOp : callsToFix) {
    auto callee = moduleOp.lookupSymbol<triton::FuncOp>(callOp.getCallee());
    if (!callee)
      continue;

    builder.setInsertionPoint(callOp);
    Location loc = callOp.getLoc();

    // Get the expected argument types from the triton function
    FunctionType calleeType = callee.getFunctionType();
    SmallVector<Value> newOperands;

    for (auto [idx, operand] : llvm::enumerate(callOp.getOperands())) {
      Type expectedType = calleeType.getInput(idx);

      // If the operand is a memref and the expected type is a tt.ptr,
      // we need to extract the pointer
      if (auto memrefType = dyn_cast<MemRefType>(operand.getType())) {
        if (auto ptrType = dyn_cast<triton::PointerType>(expectedType)) {
          // Extract the aligned pointer from the memref
          Value indexPtr = memref::ExtractAlignedPointerAsIndexOp::create(
              builder, loc, operand);
          // Convert index to i64 for pointer conversion
          Value i64Ptr = arith::IndexCastOp::create(
              builder, loc, builder.getI64Type(), indexPtr);
          // Convert i64 to tt.ptr
          Value ttPtr =
              triton::IntToPtrOp::create(builder, loc, ptrType, i64Ptr);
          newOperands.push_back(ttPtr);
          continue;
        }
      }

      // If types match or no conversion needed, use the operand as-is
      newOperands.push_back(operand);
    }

    // Create the tt.call operation
    triton::CallOp::create(builder, loc, callee, newOperands);

    // Erase the old func.call
    callOp.erase();
  }
}

void RockMemrefToTensorPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();

  // Collect functions first to avoid iterator invalidation when we move/erase
  // them
  SmallVector<func::FuncOp> funcsToProcess;
  SmallVector<func::FuncOp> nonKernelFuncs;
  moduleOp.walk([&](func::FuncOp funcOp) {
    // Only process top-level functions (not in nested modules)
    if (funcOp->getParentOfType<ModuleOp>() != moduleOp)
      return;
    if (funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
      funcsToProcess.push_back(funcOp);
    else
      nonKernelFuncs.push_back(funcOp);
  });

  // Store kernel grid/block sizes as module attributes BEFORE converting to
  // tt.func (which erases the func::FuncOp). These will be used later for
  // gpu.launch_func.
  for (func::FuncOp funcOp : funcsToProcess) {
    std::string kernelName = funcOp.getName().str();
    if (auto gridAttr = funcOp->getAttrOfType<IntegerAttr>(
            rock::GridSizeAttr::getMnemonic())) {
      moduleOp->setAttr("rock.grid_size." + kernelName, gridAttr);
    }
  }

  // Process kernel functions (convert to tt.func)
  for (func::FuncOp funcOp : funcsToProcess) {
    processFunction(funcOp);
  }

  // Store non-kernel functions (host code) as serialized MLIR in a module
  // attribute. This isolates them from Triton passes. We use local scope
  // printing to avoid issues with symbol references that will change during
  // Triton compilation. The host code will be restored and lowered separately
  // after Triton compilation.
  if (!nonKernelFuncs.empty()) {
    OpBuilder builder(ctx);
    SmallVector<Attribute> funcStrings;

    for (func::FuncOp funcOp : nonKernelFuncs) {
      std::string funcStr;
      llvm::raw_string_ostream os(funcStr);
      // Use local scope to allow printing without verifying symbol references
      funcOp.print(os, OpPrintingFlags().useLocalScope());
      funcStrings.push_back(StringAttr::get(ctx, funcStr));
    }

    moduleOp->setAttr("rock.host_functions", ArrayAttr::get(ctx, funcStrings));

    // Erase the original host functions from the main module
    for (func::FuncOp funcOp : nonKernelFuncs) {
      funcOp.erase();
    }
  }
}
