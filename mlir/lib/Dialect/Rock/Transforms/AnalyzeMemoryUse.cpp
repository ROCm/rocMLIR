//===- AnalyzeMemoryUse.cpp - add annotations about indexing, aliasing -===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2023 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This pass will add annotations to kernel functions and their memref arguments
// to reflect memory usage within those functions. This pass is required as it
// propagates the need for 64-bit indexing up to the function level.
//
// In addition to the propagation of indexing requirements, this pass sets
// memref arguments to be `noalias` (a property we guarantee) and sets
// `readonly` and `writeonly` attributes if they can be inferred from the
// function body.
//
// Finally, it will set `nonnull`, `noundef` and (if we can) the relevant
// `dereferencable` flags on arguments so as to give LLVM maximal permission
// to optimize memory accesses at the cost of debugability.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "rock-buffer-load-merge"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKANALYZEMEMORYUSEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockAnalyzeMemoryUsePass final
    : public rock::impl::RockAnalyzeMemoryUsePassBase<
          RockAnalyzeMemoryUsePass> {
  void runOnOperation() final;
};
} // end namespace

void RockAnalyzeMemoryUsePass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr("kernel"))
    return;
  Builder b(func.getContext());

  // 1. Check for 64-bit indexing anywhere in the function
  WalkResult kernelNeeds64Bit = func.walk([](Operation *op) {
    if (auto globalLoad = dyn_cast<GlobalLoadOp>(op))
      return globalLoad.getNeeds64BitIdx() ? WalkResult::interrupt()
                                           : WalkResult::advance();
    if (auto globalStore = dyn_cast<GlobalStoreOp>(op))
      return globalStore.getNeeds64BitIdx() ? WalkResult::interrupt()
                                            : WalkResult::advance();
    return WalkResult::advance();
  });
  if (kernelNeeds64Bit.wasInterrupted())
    func->setAttr("rock.64bitindex", b.getUnitAttr());

  // 2. Walk through memref arguments, attaching the relevant attributes
  SmallVector<Operation *, 0> worklist;
  for (auto [idx, val] : llvm::enumerate(func.getArguments())) {
    auto type = dyn_cast<MemRefType>(val.getType());
    if (!type)
      continue;
    bool isReadonly = true;
    bool isWriteonly = true;
    worklist.append(val.getUsers().begin(), val.getUsers().end());
    while (!worklist.empty()) {
      Operation *user = worklist.pop_back_val();
      if (isa<ViewLikeOpInterface>(user)) {
        worklist.append(user->getUsers().begin(), user->getUsers().end());
        continue;
      }
      if (!isa<GlobalLoadOp>(user))
        isReadonly = false;
      auto storeOp = dyn_cast<GlobalStoreOp>(user);
      if (!storeOp || storeOp.getStoreMethod() != StoreMethod::Set)
        isWriteonly = false;
    }
    // Note: we _could_ set `nontemporal` here, but, in most cases, it's bad for
    // performance. If we want to bring that back, we need to capture the
    // `global_store` ops and set them to `nontemporal` if we've concluded
    // that that memref is writeonly. That'd need a vector of store ops.

    Attribute unit = b.getUnitAttr();
    // Note: we'll need to go back in and add alias scopes after LLVM
    // translation because the AMDGPU backend currently chucks out `noalias`
    // on kernel arguments because it's hard to lower.
    if (isReadonly && isWriteonly)
      // Unused pointer argument is readnone.
      func.setArgAttr(idx, LLVM::LLVMDialect::getReadnoneAttrName(), unit);
    else if (isReadonly)
      func.setArgAttr(idx, LLVM::LLVMDialect::getReadonlyAttrName(), unit);
    else if (isWriteonly)
      func.setArgAttr(idx, LLVM::LLVMDialect::getWriteOnlyAttrName(), unit);

    func.setArgAttr(idx, LLVM::LLVMDialect::getNoAliasAttrName(), unit);
    func.setArgAttr(idx, LLVM::LLVMDialect::getNoCaptureAttrName(), unit);
    func.setArgAttr(idx, LLVM::LLVMDialect::getNoFreeAttrName(), unit);
    func.setArgAttr(idx, LLVM::LLVMDialect::getNonNullAttrName(), unit);
    func.setArgAttr(idx, LLVM::LLVMDialect::getNoUndefAttrName(), unit);

    // `inreg` enables SGPR preloading in new calling conventions.
    //func.setArgAttr(idx, LLVM::LLVMDialect::getInRegAttrName(), unit);
    // As near as we can tell, there's no universe in which global pointers
    // aren't aligned to 16 bytes.
    func.setArgAttr(idx, LLVM::LLVMDialect::getAlignAttrName(),
                    b.getI64IntegerAttr(16));
    // Anyone lying about the size of their input deserves exactly what they
    // get.
    if (type.hasStaticShape())
      func.setArgAttr(idx, LLVM::LLVMDialect::getDereferenceableAttrName(),
                      b.getI64IntegerAttr(type.getNumElements() *
                                          type.getElementTypeBitWidth() / 8));
  }
}
