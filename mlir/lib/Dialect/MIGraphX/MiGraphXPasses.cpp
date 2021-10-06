//===- MIGraphXPasses.cpp - MIGraphX MLIR Passes
//-----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
#include "mlir/Dialect/MIGraphX/Passes.h"

#include "PassDetail.h"

#include "mlir/Pass/Pass.h"

#include <memory>

using namespace mlir;

namespace {
struct MIGraphXIRDumpPass : public MIGraphXIRDumpPassBase<MIGraphXIRDumpPass> {
  void runOnOperation() override;
};

} // end namespace

std::unique_ptr<Pass> mlir::migraphx::createMIGraphXIRDumpPass() {
  return std::make_unique<MIGraphXIRDumpPass>();
}

void MIGraphXIRDumpPass::runOnOperation() {
  // FIXME(jungwook): This should do something
  return;
}
