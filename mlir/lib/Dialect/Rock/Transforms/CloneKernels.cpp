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

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <iterator>

#define DEBUG_TYPE "rock-clone-kernels"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKCLONEKERNELSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;

namespace {

struct RockCloneKernelsPass
    : public rock::impl::RockCloneKernelsPassBase<RockCloneKernelsPass> {
  using rock::impl::RockCloneKernelsPassBase<
      RockCloneKernelsPass>::RockCloneKernelsPassBase;
  void runOnOperation() override {
    const char *kernel = "kernel.module";
    ModuleOp mod = getOperation();
    if (mod->hasAttr(kernel))
      return;

    SmallVector<func::FuncOp, 8> kernelFuncs;
    for (auto func : mod.getOps<func::FuncOp>()) {
      if (func->hasAttr("kernel"))
        kernelFuncs.push_back(func);
    }

    for (auto &target : targets) {
      SmallString<32> escapedTarget;
      escapedTarget.reserve(target.size());
      llvm::replace_copy_if(
          target, std::back_inserter(escapedTarget),
          [](char c) { return (c == '-' || c == ':' || c == ','); }, '_');
      for (char &c : escapedTarget) {
        if (c == '+')
          c = 'Y';
      }

      SmallString<32> modName(Twine("__kernel_module_", escapedTarget).str());
      LLVM_DEBUG(llvm::dbgs() << "Cloning to module " << modName << "\n");

      auto *ctx = &getContext();
      auto kernelMod = mod.lookupSymbol<ModuleOp>(modName);
      if (!kernelMod) {
        OpBuilder b(ctx);
        // create a KERNEL ModuleOp in case the KERNEL module specified does not
        // exist.
        kernelMod = b.create<ModuleOp>(mod.getLoc(), StringRef(modName));
        kernelMod->setAttr(kernel, b.getUnitAttr());
        kernelMod->setAttr("kernel.arch", b.getStringAttr(target));

        // add the KERNELModuleOp into the symbol table.
        SymbolTable symbolTable(mod);
        symbolTable.insert(kernelMod);
      }
      SymbolTable symbolTable(kernelMod);

      for (auto func : kernelFuncs) {
        // clone the func
        auto kernelFunc = func.clone();
        kernelFunc->setAttr("original_func", SymbolRefAttr::get(func));

        // add the KERNELModuleOp into the symbol table.
        symbolTable.insert(kernelFunc);

        // TODO: also find all calls and import callee
      }
    }

    // remove kernel attribute from kernelFuncs
    for (auto func : kernelFuncs) {
      func->removeAttr("kernel");
    }
  }
};

} // end anonymous namespace
