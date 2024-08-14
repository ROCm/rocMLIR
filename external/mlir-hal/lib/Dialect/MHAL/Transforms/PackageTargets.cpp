//===- PackageTargets.cpp -------------------------------------------------===//
//
// Copyright 2020 The MLIR Authors.
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
// This pass applies target implementations to host kernel funcs.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MHAL/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace mhal {
#define GEN_PASS_DEF_MHALPACKAGETARGETSPASS
#include "mlir/Dialect/MHAL/Transforms/Passes.h.inc"
} // namespace mhal
} // namespace mlir

#define DEBUG_TYPE "mhal-package-targets"

using namespace mlir;

namespace {
struct MHALPackageTargetsPass
    : public mhal::impl::MHALPackageTargetsPassBase<MHALPackageTargetsPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getContext());

    llvm::SmallVector<ModuleOp, 8> kernelMods;
    llvm::SmallDenseMap<func::FuncOp, llvm::SmallVector<Attribute, 4>>
        kernelImpls;

    mod->walk([&](ModuleOp kernelMod) {
      if (kernelMod->hasAttr("mhal.module")) {
        SmallVector<gpu::BinaryOp, 8> binaries;
        kernelMod->walk([&](gpu::BinaryOp binary) {
          auto object = cast<gpu::ObjectAttr>(binary.getObjects()[0]);
          binaries.push_back(binary);
          gpu::KernelTableAttr metadata = object.getKernels();
          assert(metadata && "expected a valid metadata attribute");
          // apply target spec to original func
          for (auto [name, kernel] : metadata) {
            if (auto attr = kernel.getAttr<SymbolRefAttr>("original_func")) {
              if (auto kernelFunc = mod.lookupSymbol<func::FuncOp>(attr)) {
                auto archName =
                    kernelMod->getAttrOfType<StringAttr>("mhal.arch")
                        .getValue();
                auto funcName = attr.getLeafReference().getValue();
                uint32_t gridSize =
                    kernel.getAttr<IntegerAttr>("grid_size").getInt();
                uint32_t blockSize =
                    kernel.getAttr<IntegerAttr>("block_size").getInt();

                DictionaryAttr objAttrs;

                auto xobj = mhal::TargetObjectAttr::get(
                    b.getContext(), mhal::TargetObjectType::ELF, archName,
                    objAttrs, object);

                DictionaryAttr pkgAttrs;
                // = b.getDictionaryAttr({
                //     b.getNamedAttr("bare_ptr_abi", true)
                // });
                auto xpkg = mhal::KernelPackageAttr::get(
                    b.getContext(), mhal::TargetType::GPU, archName, funcName,
                    {gridSize, blockSize}, pkgAttrs, xobj);

                kernelImpls[kernelFunc].push_back(xpkg);
              }
            }
          }
        });

        // clean processed gpu.modules
        for (auto binary : binaries) {
          binary.erase();
        }

        // remove __kernel_*
        kernelMods.push_back(kernelMod);
      }
    });

    for (auto pair : kernelImpls) {
      pair.first->setAttr("mhal.targets", b.getArrayAttr({pair.second.begin(),
                                                          pair.second.end()}));
    }

    // cleanup
    for (auto kernelMod : kernelMods)
      kernelMod->erase();
  }
};

} // end anonymous namespace
