//===- MHALToCPU.cpp - Convert MHAL to CPU dialect --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MHALToCPU/MHALToCPU.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "convert-mhal-to-cpu"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMHALTOCPUPASS
#include "mlir/Conversion/MHALPasses.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::mhal;

//===----------------------------------------------------------------------===//
// Convert MHAL dialect types to CPU types.
//===----------------------------------------------------------------------===//

namespace {
// Helper to pull out the called func
static Optional<func::FuncOp> getCalledFunc(mhal::LaunchOp op) {
  CallOpInterface callIf(op);
  if (auto *callable = callIf.resolveCallable()) {
    if (auto func = dyn_cast<func::FuncOp>(callable))
      return func;
  }

  return std::nullopt;
}
}

//===----------------------------------------------------------------------===//
// Convert mhal.launch ops with 'cpu' target to cpu.launch_func ops with
// required memory staging.
//===----------------------------------------------------------------------===//

namespace {
struct LaunchRewritePattern : public OpRewritePattern<mhal::LaunchOp> {
  using OpRewritePattern<mhal::LaunchOp>::OpRewritePattern;


  LogicalResult matchAndRewrite(mhal::LaunchOp op,
                                PatternRewriter &rw) const override {
    Location loc = op.getLoc();
    auto caller = op->getParentOfType<func::FuncOp>();
    auto module = caller->getParentOfType<ModuleOp>();
    auto *ctx = module.getContext();

    assert(op->getNumResults() == 1); // only 1 mhal.token

#if 0
    auto func = *getCalledFunc(op);
    Location floc = func.getLoc();

    // 2. create dummy cpu.module for reference from cpu.launch_func
    //    - with cpu.binary, arch attributes
    //    - and cpu.func (referenced by cpu.launch_func
    //    cpu.module @<func_name>_module attributes {arch = "gfx908", cpu.binary
    //        = "\7FELF\..."} {
    //      cpu.func @<func_name> (...) attributes {block_size = 256 : i32,
    //          grid_size = 900 : i32, cpu.kernel}

    FunctionOpInterface funcIF(func);
    auto funcName = funcIF.getName();
    auto cpuModuleName = funcName + "_module";

    auto cpuModule = module.lookupSymbol<cpu::CPUModuleOp>(cpuModuleName.str());
    if (!cpuModule) {
      OpBuilder b(ctx);
      cpuModule = b.create<cpu::CPUModuleOp>(floc, cpuModuleName.str());
      cpuModule->setAttr("arch", b.getStringAttr(arch));
      cpuModule->setAttr("cpu.binary", b.getStringAttr(binary));

      SymbolTable symbolTable(module);
      symbolTable.insert(cpuModule);
    }

    auto cpuFunc = cpuModule.lookupSymbol<cpu::CPUFuncOp>(funcName);
    if (!cpuFunc) {
      OpBuilder b(cpuModule.getContext());
      cpuFunc =
          b.create<cpu::CPUFuncOp>(floc, funcName, func.getFunctionType());
      cpuFunc->setAttr("block_size", b.getI32IntegerAttr(blockSize));
      cpuFunc->setAttr("grid_size", b.getI32IntegerAttr(gridSize));
      cpuFunc->setAttr(cpu::CPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());

      SymbolTable symbolTable(cpuModule);
      symbolTable.insert(cpuFunc);

      // Must have a return
      auto block = &cpuFunc.front();
      b.setInsertionPoint(block, block->begin());
      b.create<cpu::ReturnOp>(floc, ValueRange{});
    }

    // 3. create substitute cpu.launch_func
    //    %15 = cpu.wait async
    //    %16 = cpu.launch_func async [%15] @test_fusion_module::@test_fusion
    //    blocks in (%c900, %c1, %c1) threads in (%c256, %c1, %c1)
    //    dynamic_shared_memory_size %c0_i32 args(%4 : memref<128x32x32x8xf32>,
    //    %9 : memref<128x3x3x8xf32>, %14 : memref<128x30x30x128xf32>)

    auto tokenType = rw.getType<cpu::AsyncTokenType>();

    Value oneIdx = rw.createOrFold<arith::ConstantIndexOp>(loc, 1);
    Value blockSizeIdx =
        rw.createOrFold<arith::ConstantIndexOp>(loc, blockSize);
    Value gridSizeIdx = rw.createOrFold<arith::ConstantIndexOp>(loc, gridSize);
    Value dynamicSharedMemorySize;

    // async dependencies
    auto operands = op->getOperands();
    llvm::SmallVector<Value, 8> asyncDeps;
    llvm::SmallVector<Value, 8> cpuOperands;
    size_t diff = operands.size() - func.getNumArguments();
    size_t i = 0;
    if (diff > 0) {
      for (; i < diff; ++i)
        asyncDeps.push_back(operands[i]);
    } else
      assert(diff == 0);

    SmallVector<Value> copyBackOprs(func.getNumArguments(), Value());
    for (; i < operands.size(); ++i) {
      auto fidx = i - diff;
      Value opr = operands[i];
      // move input memories to CPU
      if (opr.getType().isa<MemRefType>()) {
        bool readAccess{
            func.getArgAttr(fidx, func::FuncOp::getReadAccessAttrName())};
        bool writeAccess{
            func.getArgAttr(fidx, func::FuncOp::getWriteAccessAttrName())};
        opr = moveMemory(rw, op, opr, fidx, readAccess, writeAccess,
                         copyBackOprs, asyncDeps);
      }
      cpuOperands.push_back(opr);
    }

    // The cpu.launch_func requires 1 and only 1 token
    if (asyncDeps.size() == 0)
      // There must be at least 1 token
      asyncDeps.push_back(makeWait(rw, loc));
    else if (asyncDeps.size() > 1) {
      // Consolidate to 1 token
      auto launchWait = makeWait(rw, loc, asyncDeps);
      asyncDeps = {launchWait};
    }

    // Make cpu.launch_func
    auto cpuLaunchOp = rw.create<cpu::LaunchFuncOp>(
        loc, asyncDeps, cpuFunc, cpu::KernelDim3{gridSizeIdx, oneIdx, oneIdx},
        cpu::KernelDim3{blockSizeIdx, oneIdx, oneIdx}, dynamicSharedMemorySize,
        cpuOperands);
    Value token = cpuLaunchOp->getResult(0);

    // Insert cpu.memcpy for results
    SmallVector<Value, 8> tokens;
    for (auto pair : llvm::enumerate(copyBackOprs)) {
      if (auto cpuMem = pair.value()) {
        auto dst = operands[diff + pair.index()];
        if (cpuMem.getDefiningOp<memref::AllocOp>())
          std::swap(cpuMem, dst);
        auto memcpy = rw.create<cpu::MemcpyOp>(loc, tokenType,
                                               ValueRange{token}, dst, cpuMem);
        tokens.push_back(memcpy.getResult(0));
      }
    }

    // Consolidate tokens for replacement of mhal.launch
    if (tokens.size() > 1) {
      // insert cpu.wait
      token = makeWait(rw, loc, tokens);
    } else if (tokens.size() == 1)
      token = tokens[0];

    rw.replaceOp(op, {token});

    module->setAttr(cpu::CPUDialect::getContainerModuleAttrName(),
                    rw.getUnitAttr());
#endif
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Remove all mhal.await ops
//===----------------------------------------------------------------------===//

namespace {
struct AwaitRewritePattern : public OpRewritePattern<mhal::AwaitOp> {
  using OpRewritePattern<mhal::AwaitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhal::AwaitOp op,
                                PatternRewriter &rw) const override {
    rw.eraseOp(op);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//

namespace {
struct ConvertMHALToCPUPass
    : public impl::ConvertMHALToCPUPassBase<ConvertMHALToCPUPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertMHALToCPUPass::runOnOperation() {
  auto op = getOperation();
  MLIRContext *ctx = op->getContext();

  // Convert mhal.launch to func.call ops, remove all mhal.await ops
  RewritePatternSet patterns(ctx);
  patterns.add<LaunchRewritePattern>(ctx);
  patterns.add<AwaitRewritePattern>(ctx);

  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
}

