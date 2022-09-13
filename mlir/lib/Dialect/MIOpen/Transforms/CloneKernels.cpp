//===- CloneKernels.cpp ---------------------------------------------------===//
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
// This pass clones all kernel functions into an MIOpen Module.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct MIOpenCloneKernelsPass
    : public MIOpenCloneKernelsPassBase<MIOpenCloneKernelsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    llvm::StringRef modName = miopen::MIOpenDialect::kKernelModuleName;
    if (mod.getName() == modName)
      return;

    auto *ctx = &getContext();
    auto gpuMod = mod.lookupSymbol<ModuleOp>(modName);
    if (!gpuMod) {
      OpBuilder b(ctx);
      // create a GPUModuleOp in case the GPU module specified does not exist.
      gpuMod = b.create<ModuleOp>(mod.getLoc(), modName);

      // add the GPUModuleOp into the symbol table.
      SymbolTable symbolTable(mod);
      symbolTable.insert(gpuMod);
    }
    SymbolTable symbolTable(gpuMod);

    for (auto func : mod.getOps<func::FuncOp>()) {
      if (func->hasAttr("kernel")) {
        // clone the func
        auto gpuFunc = func.clone();
        gpuFunc->setAttr("original_func", SymbolRefAttr::get(func));

        SmallString<128> nameBuffer(gpuFunc.getName());
        nameBuffer += "_miopen";
        gpuFunc.setName(nameBuffer);

        // add the GPUModuleOp into the symbol table.
        symbolTable.insert(gpuFunc);

        // TODO: also find all calls and import callee
      }
    }
  }
};

} // end anonymous namespace

//===- Passes -------------------------------------------------------------===//
//

std::unique_ptr<Pass> mlir::miopen::createMIOpenCloneKernelsPass() {
  return std::make_unique<MIOpenCloneKernelsPass>();
}
