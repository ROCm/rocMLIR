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
// This pass adds alias scope information to operations that perform direct-to-LDS
// loads or stores and local loads or stores.
// 
// This includes:
// - rocdl.load_to_lds operations (direct loads to LDS)
// - llvm.load operations from global memory to LDS  (local loads)
// - llvm.store operations to LDS from global memory (local stores)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Rock/utility/AliasUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKADDALIASINFOPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "add-alias-info"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockAddAliasInfoPass
    : public rock::impl::RockAddAliasInfoPassBase<RockAddAliasInfoPass> {
  void runOnOperation() override {
    gpu::GPUModuleOp module = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Running RockAddAliasInfoPass on GPU module\n");

    // Walk through all LLVM functions in the module and process their operations
    // TODO: Use LDSAddr instead of 3
    // TODO: What about multibuffering?
    module.walk([&](LLVM::LLVMFuncOp func) {
      LLVM_DEBUG(llvm::dbgs() << "Processing function: " << func.getName() << "\n");
      
      func.walk([&](LLVM::AliasAnalysisOpInterface aliasIface) {
        Operation *aliasOp = aliasIface.getOperation();

        if (auto loadOp = dyn_cast<LLVM::LoadOp>(aliasOp)) {
          LLVM_DEBUG(llvm::dbgs() << "LLVM::LoadOp\n");
          Value addr = loadOp.getAddr();
          if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(addr.getType())) {
            if (ptrType.getAddressSpace() == 3) {
              LLVM_DEBUG(llvm::dbgs() << "LLVM::LoadOp with LDS address space!!!\n");
              addLocalLoadNoAliasScope(loadOp);
            }
          }
        }
        else if (auto storeOp = dyn_cast<LLVM::StoreOp>(aliasOp)) {
          LLVM_DEBUG(llvm::dbgs() << "LLVM::StoreOp\n");
          Value addr = storeOp.getAddr();
          if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(addr.getType())) {
            if (ptrType.getAddressSpace() == 3) {
              LLVM_DEBUG(llvm::dbgs() << "LLVM::StoreOp with LDS address space!!!\n");
              addLocalLoadNoAliasScope(storeOp);
            }
          }
        }
        else if (auto loadToLDSOp = dyn_cast<ROCDL::LoadToLDSOp>(aliasOp)) {
          // We generate GatherToLDSOp, which will be lowered to LoadToLDSOp at this point,
          // so here its the right moment to add alias scope information.
          LLVM_DEBUG(llvm::dbgs() << "ROCDL::LoadToLDSOp!!!\n");
          addDirectToLDSLoadAliasScope(loadToLDSOp);
        }
        else if (auto rawPtrBufferLoadLdsOp = dyn_cast<ROCDL::RawPtrBufferLoadLdsOp>(aliasOp)) {
          LLVM_DEBUG(llvm::dbgs() << "ROCDL::RawPtrBufferLoadLdsOp!!!\n");
          addDirectToLDSLoadAliasScope(rawPtrBufferLoadLdsOp);
        }
        else {
          LLVM_DEBUG(llvm::dbgs()
                     << "Operation not supported  : " << *aliasOp << "\n");
        }
      });
    });
  }
};
} // namespace

