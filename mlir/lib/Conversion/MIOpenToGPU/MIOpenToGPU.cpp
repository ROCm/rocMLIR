//===- MIOpenToGPU.cpp - MLIR MIOpen ops lowering passes ---------------===//
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
// This pass converts miopen ops to std dialect.
//
//===----------------------------------------------------------------------===//


#include "mlir/Conversion/MIOpenToGPU/MIOpenToGPU.h"
#include "../PassDetail.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {
struct LowerMIOpenOpsToGPUPass : public ConvertMIOpenToGPUBase<LowerMIOpenOpsToGPUPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void LowerMIOpenOpsToGPUPass::runOnOperation() {
  auto op = getOperation();
  OpBuilder b(op.getContext());
  auto loc = op.getLoc();

  // create a GPUModuleOp.
  OperationState state(loc, gpu::GPUModuleOp::getOperationName());
  gpu::GPUModuleOp::build(b, state, "miopen_module");
  auto gpuModule = cast<gpu::GPUModuleOp>(Operation::create(state));
  SymbolTable gpuModuleSymbolTable(gpuModule);

  // add the GPUModuleOp into the symbol table.
  SymbolTable symbolTable(op);
  symbolTable.insert(gpuModule);

  for (auto func : op.getOps<FuncOp>()) {
    // create a GPUFuncOp.
    FunctionType gpuFuncType = func.getType();
    auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, func.getName(), gpuFuncType);

    // TBD: set kernel attribute.
    // gpuFunc.setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
    //                b.getUnitAttr());

    // associate arguments for newly created GPUFuncOp.
    BlockAndValueMapping map;
    for (unsigned idx = 0; idx < func.getNumArguments(); ++idx) {
      auto arg = func.getArgument(idx);
      auto gpuFuncArg = gpuFunc.getArgument(idx);

      map.map(arg, gpuFuncArg);
    }

    // clone function body into newly created GPUFuncOp.
    Region &gpuFuncBody = gpuFunc.body();
    Region &funcBody = func.getBody();
    funcBody.cloneInto(&gpuFuncBody, map);

    // add a branch op to the cloned region.
    Block &funcEntry = funcBody.front();
    Block *clonedFuncEntry = map.lookup(&funcEntry);
    Block &gpuFuncEntry = gpuFuncBody.front();
    b.setInsertionPointToEnd(&gpuFuncEntry);
    b.create<BranchOp>(loc, clonedFuncEntry);

    // remove std.return ops.
    gpuFunc.walk([&](ReturnOp op) { op.erase(); });

    // create a GPU ReturnOp inside the GPUFuncOp.
    b.setInsertionPointToEnd(&gpuFuncBody.back());
    b.create<gpu::ReturnOp>(loc);

    // insert the GPUFuncOp into GPUModuleOp.
    gpuModuleSymbolTable.insert(gpuFunc);
  }

  // TBD: erase the old function.

  for (auto module : op.getOps<gpu::GPUModuleOp>()) {
    module.walk([&](gpu::GPUFuncOp func) {

      func.walk([&](miopen::ThreadwiseGemmOp op) { op.erase(); });

      func.walk([&](miopen::ThreadwiseCopyOp op) { op.erase(); });

      func.walk([&](miopen::GpuAllocOp op) {
        auto loc = op.getLoc();
        auto type = op.output().getType().cast<MemRefType>();

        if (type.getMemorySpace() == 3) {
          Value attribution = func.addWorkgroupAttribution(type);
          op.replaceAllUsesWith(attribution);
        } else if (type.getMemorySpace() == 5) {
          Value attribution = func.addPrivateAttribution(type);
          op.replaceAllUsesWith(attribution);
        } else {
          // TBD: return failure.
          llvm::errs() << "unsupported addrspace!\n";
        }
        op.erase();
      });

      func.walk([&](miopen::LdsBarrierOp op) {
        auto loc = op.getLoc();
        OpBuilder b(op.getContext());
        b.setInsertionPoint(op);
        b.create<gpu::BarrierOp>(loc);
        op.erase();
      });
    });
  }
}

std::unique_ptr<Pass> mlir::createLowerMIOpenOpsToGPUPass() {
  return std::make_unique<LowerMIOpenOpsToGPUPass>();
}
