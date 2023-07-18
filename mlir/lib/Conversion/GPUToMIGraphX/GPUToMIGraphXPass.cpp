//===- GPUToMIGraphXPass.cpp - Lowering GPU to MIGraphX Dialect -----------===//
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

#include "mlir/Conversion/GPUToMIGraphX/GPUToMIGraphX.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/Async/IR/Async.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/ExecutionEngine/OptUtils.h"

namespace mlir {
#define GEN_PASS_DEF_GPUTOMIGRAPHXPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

using namespace mlir;

struct GPUToMIGraphX : public impl::GPUToMIGraphXPassBase<GPUToMIGraphX> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<migraphx::MIGraphXDialect, func::FuncDialect, gpu::GPUDialect,
                memref::MemRefDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    LLVMTypeConverter converter(&ctx);

    ConversionTarget target(ctx);
    target.addLegalDialect<migraphx::MIGraphXDialect, func::FuncDialect,
                           gpu::GPUDialect, memref::MemRefDialect,
                           LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<func::CallOp>([&](Operation *op) {
      auto fnAttr = op->getAttrOfType<FlatSymbolRefAttr>("callee");
      auto fusedFuncOp =
          op->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(
              fnAttr.getValue());
      return (fusedFuncOp.getOperation()->getAttr("kernel") == nullptr);
    });

    func::FuncOp func = getOperation();
    mlir::migraphx::populateFuncToCOBJPatterns(func.getContext(), patterns);

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

void migraphx::addGPUToMIGraphXPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createGPUToMIGraphXPass());
}
