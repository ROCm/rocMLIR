//===- ApplyImpl.cpp ------------------------------------------------------===//
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

#include "PassDetail.h"

#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct RockApplyImplPass
    : public RockApplyImplPassBase<RockApplyImplPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getContext());

    llvm::SmallVector<ModuleOp, 8> kernelMods;
    llvm::SmallDenseMap<func::FuncOp, llvm::SmallVector<DictionaryAttr, 4>>
        kernelImpls;

    mod->walk([&](ModuleOp kernelMod) {
      if (kernelMod->hasAttr("kernel.module")) {

        SmallVector<gpu::GPUModuleOp, 8> gpuMods;
        kernelMod->walk([&](gpu::GPUModuleOp gpuMod) {
          auto binaryAttr = gpuMod->getAttrOfType<StringAttr>(
              gpu::getDefaultGpuBinaryAnnotation());
          if (!binaryAttr) {
            gpuMod.emitOpError() << "missing gpu.binary attribute";
            return;
          }

          gpuMods.push_back(gpuMod);

          // apply target spec to original func
          gpuMod.walk([&](LLVM::LLVMFuncOp func) {
            if (auto attr =
                    func->getAttrOfType<SymbolRefAttr>("original_func")) {
              if (auto kernelFunc = mod.lookupSymbol<func::FuncOp>(attr)) {
                std::vector<NamedAttribute> attributes{
                    b.getNamedAttr("type", b.getStringAttr("gpu")),
                    b.getNamedAttr("arch", gpuMod->getAttr("arch")),
                    b.getNamedAttr("grid_size", func->getAttr("grid_size")),
                    b.getNamedAttr("block_size", func->getAttr("block_size")),
                    b.getNamedAttr("binary", binaryAttr)};

                kernelImpls[kernelFunc].push_back(
                    b.getDictionaryAttr(attributes));
              }
            }
          });
        });

        // clean processed gpu.modules
        for (auto gpuMod : gpuMods) {
          gpuMod.erase();
        }

        // remove __kernel_*
        kernelMods.push_back(kernelMod);
      }
    });

    for (auto pair : kernelImpls) {
      pair.first->setAttr("async.targets", b.getArrayAttr({pair.second.begin(),
                                                           pair.second.end()}));
    }

    // cleanup
    for (auto kernelMod : kernelMods)
      kernelMod->erase();
  }
};

} // end anonymous namespace

//===- Passes -------------------------------------------------------------===//
//

std::unique_ptr<Pass> mlir::rock::createRockApplyImplPass() {
  return std::make_unique<RockApplyImplPass>();
}
