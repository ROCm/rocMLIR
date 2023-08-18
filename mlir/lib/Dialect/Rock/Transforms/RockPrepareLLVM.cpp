//===- RockPrepareLLVM.cpp - prepares the generated code for LLVM       ---===//
//
// Copyright 2022 AMD
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

#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKPREPARELLVMPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-prepare-llvm"

using namespace mlir;

namespace {
struct RockPrepareLLVMPass
    : public rock::impl::RockPrepareLLVMPassBase<RockPrepareLLVMPass> {
  void runOnOperation() override;
};
} // end namespace

void RockPrepareLLVMPass::runOnOperation() {
  auto func = getOperation();
  func->walk([](LLVM::GEPOp gepOp) { gepOp.setInbounds(true); });
}
