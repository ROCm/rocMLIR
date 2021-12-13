//===- GPUToMIGraphXPass.cpp - Lowering GPU to MIGraphX Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes GPU operations to the MIGraphX dialect.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Conversion/GPUToMIGraphX/GPUToMIGraphX.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"

#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Async/IR/Async.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/ExecutionEngine/ROCm/BackendUitls.h"
#include "mlir/ExecutionEngine/OptUtils.h"


using namespace mlir;

struct GPUToMIGraphX
    : public GPUToMIGraphXBase<GPUToMIGraphX> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<migraphx::MIGraphXDialect, StandardOpsDialect, gpu::GPUDialect, memref::MemRefDialect, LLVM::LLVMDialect>();
  }

  void runOnFunction() override {
    auto &ctx = getContext();
    OwningRewritePatternList patterns(&ctx);
    LLVMTypeConverter converter(&ctx);

    ConversionTarget target(ctx);
    target.addLegalDialect<migraphx::MIGraphXDialect, StandardOpsDialect, gpu::GPUDialect, memref::MemRefDialect, LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<CallOp>([&](Operation *op) {
      auto fnAttr = op->getAttrOfType<FlatSymbolRefAttr>("callee");
      auto fusedFuncOp = op->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(fnAttr.getValue());
      return (fusedFuncOp.getOperation()->getAttr("kernel") == nullptr); 
    });

    FuncOp func = getFunction();
    mlir::migraphx::populateFuncToCOBJPatterns(
        func.getContext(), &patterns);

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> migraphx::createGPUToMIGraphXPass() {
  return std::make_unique<GPUToMIGraphX>();
}
void migraphx::addGPUToMIGraphXPasses(OpPassManager &pm) {
  pm.addNestedPass<FuncOp>(createGPUToMIGraphXPass());
}
