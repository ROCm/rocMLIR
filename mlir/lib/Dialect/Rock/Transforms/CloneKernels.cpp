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
// This pass clones all kernel functions into an Rock Module.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Rock/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct RockCloneKernelsPass
    : public RockCloneKernelsPassBase<RockCloneKernelsPass> {
  RockCloneKernelsPass(ArrayRef<StringRef> _chips)
      : chips{_chips.begin(), _chips.end()} {}

  void runOnOperation() override {
    const char *kernel = "kernel.module";
    ModuleOp mod = getOperation();
    if (mod->hasAttr(kernel))
      return;

    for (auto chip : chips) {
      SmallString<32> modName(Twine("__kernel_", chip).str());

      auto *ctx = &getContext();
      auto kernelMod = mod.lookupSymbol<ModuleOp>(modName);
      if (!kernelMod) {
        OpBuilder b(ctx);
        // create a KERNEL ModuleOp in case the KERNEL module specified does not
        // exist.
        kernelMod = b.create<ModuleOp>(mod.getLoc(), StringRef(modName));
        kernelMod->setAttr(kernel, b.getUnitAttr());
        kernelMod->setAttr("kernel.chip", b.getStringAttr(chip));

        // add the KERNELModuleOp into the symbol table.
        SymbolTable symbolTable(mod);
        symbolTable.insert(kernelMod);
      }
      SymbolTable symbolTable(kernelMod);

      for (auto func : mod.getOps<func::FuncOp>()) {
        if (func->hasAttr("kernel")) {
          // clone the func
          auto kernelFunc = func.clone();
          kernelFunc->setAttr("original_func", SymbolRefAttr::get(func));

          // add the KERNELModuleOp into the symbol table.
          symbolTable.insert(kernelFunc);

          // TODO: also find all calls and import callee
        }
      }
    }
  }

private:
  SmallVector<SmallString<32>, 4> chips;
};

} // end anonymous namespace

//===- Passes -------------------------------------------------------------===//
//

std::unique_ptr<Pass>
mlir::rock::createRockCloneKernelsPass(ArrayRef<StringRef> _chips) {
  return std::make_unique<RockCloneKernelsPass>(_chips);
}
