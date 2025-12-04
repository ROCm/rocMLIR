//===- AddAliasInfo.cpp - Add alias information to operations ------------===//
//
// Copyright 2025 Advanced Micro Devices.
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
//
// This pass adds alias scope information to operations that perform
// direct-to-LDS loads or stores and LDS loads or stores.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKADDDIRECTTOLDSALIASINFOPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-add-direct-to-lds-alias-info"

using namespace mlir;
using namespace mlir::rock;

namespace {
LLVM::AliasScopeDomainAttr getScopeDomain(MLIRContext *ctx) {
  Builder b(ctx);
  return b.getAttr<LLVM::AliasScopeDomainAttr>(
      b.getStringAttr("amdgpu.LoadsScope"),
      b.getStringAttr(
          "Domain to hold alias scopes to specify aliasing information for "
          "LDS loads or direct-to-LDS loads"));
}

LLVM::AliasScopeAttr getDirectToLDSLoadScope(MLIRContext *ctx) {
  Builder b(ctx);
  auto name = b.getStringAttr("amdgpu.DirectToLDSLoads");
  auto desc = b.getStringAttr("Scope containing all operations that perform "
                              "direct global-to-LDS loads");
  return b.getAttr<LLVM::AliasScopeAttr>(name, getScopeDomain(ctx), desc);
}

LLVM::AliasScopeAttr getLDSLoadScope(MLIRContext *ctx) {
  Builder b(ctx);
  auto name = b.getStringAttr("amdgpu.LDSLoads");
  auto desc = b.getStringAttr("Scope containing all LDS load and store ops");
  return b.getAttr<LLVM::AliasScopeAttr>(name, getScopeDomain(ctx), desc);
}

// Should be called for all DirectToLDS loads.
void addDirectToLDSLoadAliasScope(LLVM::AliasAnalysisOpInterface op) {
  auto ctx = op->getContext();
  Builder b(ctx);

  // Do not alias with LDS loads.
  op.setNoAliasScopes(b.getArrayAttr(getLDSLoadScope(ctx)));

  op.setAliasScopes(b.getArrayAttr(getDirectToLDSLoadScope(ctx)));
}

// Should be called for all LDS loads.
void addLDSLoadNoAliasScope(LLVM::AliasAnalysisOpInterface op) {
  auto ctx = op->getContext();
  Builder b(ctx);

  // Do not alias with DirectToLDS loads.
  op.setNoAliasScopes(b.getArrayAttr(getDirectToLDSLoadScope(ctx)));

  // Add to different scope as ops without any scope alias with everything
  op.setAliasScopes(b.getArrayAttr(getLDSLoadScope(ctx)));
}

struct RockAddDirectToLDSAliasInfoPass
    : public rock::impl::RockAddDirectToLDSAliasInfoPassBase<
          RockAddDirectToLDSAliasInfoPass> {
  void runOnOperation() override {
    gpu::GPUModuleOp module = getOperation();
    LLVM_DEBUG(llvm::dbgs()
               << "Running RockAddDirectToLDSAliasInfoPass on GPU module\n");

    // Lambda to check if a value is a pointer to LDS (shared memory).
    auto isLDSPointer = [](Value value) -> bool {
      if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(value.getType())) {
        return ptrType.getAddressSpace() ==
               ROCDL::ROCDLDialect::kSharedMemoryAddressSpace;
      }
      return false;
    };

    // Walk through all LLVM functions in the module.
    module.walk([&](LLVM::LLVMFuncOp func) {
      LLVM_DEBUG(llvm::dbgs() << "Processing: " << func.getName() << "\n");

      func.walk([&](LLVM::AliasAnalysisOpInterface aliasIface) {
        Operation *aliasOp = aliasIface.getOperation();

        // Add LDS loads and stores to noAliasScope.
        if (isa<LLVM::LoadOp, LLVM::StoreOp>(aliasOp)) {
          assert(aliasIface.getAccessedOperands().size() == 1 &&
                 "Expected only one accessed operand");
          Value addr = aliasIface.getAccessedOperands()[0];
          if (isLDSPointer(addr)) {
            LLVM_DEBUG(llvm::dbgs() << aliasOp->getName()
                                    << " with LDS address space: Adding to "
                                       "noAliasScope\n");
            addLDSLoadNoAliasScope(aliasIface);
          }
        } else if (auto memEffectInterface =
                       dyn_cast<MemoryEffectOpInterface>(aliasOp)) {
          // Check if this operation has write effects to LDS memory.
          SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>>
              effects;
          memEffectInterface.getEffects(effects);

          for (const auto &effect : effects) {
            if (!isa<MemoryEffects::Write>(effect.getEffect())) {
              continue;
            }
            Value effectValue = effect.getValue();
            assert(effectValue && "Effect value is null");

            if (isLDSPointer(effectValue)) {
              LLVM_DEBUG(llvm::dbgs()
                         << aliasOp->getName() << ": Adding to aliasScope\n");
              addDirectToLDSLoadAliasScope(aliasIface);
              break;
            }
          }
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "Operation not supported  : " << *aliasOp << "\n");
        }
      });
    });
  }
};
} // namespace
