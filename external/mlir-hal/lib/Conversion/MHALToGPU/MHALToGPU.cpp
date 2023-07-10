//===- MHALToGPU.cpp - Convert MHAL to GPU dialect --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MHALToGPU/MHALToGPU.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "convert-mhal-to-gpu"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMHALTOGPUPASS
#include "mlir/Conversion/MHALPasses.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::mhal;

//===----------------------------------------------------------------------===//
// Convert MHAL dialect types to GPU types.
//===----------------------------------------------------------------------===//

namespace {
/// MHALGPUTypeConverter only converts types from the MHAL dialect to
/// the corresponding GPU type and does not convert any other types.
class MHALGPUTypeConverter : public TypeConverter {
public:
  MHALGPUTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](TokenType type) {
      return gpu::AsyncTokenType::get(type.getContext());
    });
  }
};
} // namespace

// Helper to pull out the called func
static std::optional<func::FuncOp> getCalledFunc(mhal::LaunchOp op) {
  CallOpInterface callIf(op);
  if (auto *callable = callIf.resolveCallable()) {
    if (auto func = dyn_cast<func::FuncOp>(callable))
      return func;
  }

  return std::nullopt;
}

// Get target{gpu} attribute from called func
static std::optional<mhal::KernelPackageAttr> getGPUTarget(mhal::LaunchOp op) {
  auto func = getCalledFunc(op);
  if (!func.has_value() || func->getNumResults() != 0)
    return std::nullopt;

  auto attr = (*func)->template getAttrOfType<ArrayAttr>("mhal.targets");
  if (!attr)
    return std::nullopt;

  for (auto targetAttr : attr.getValue()) {
    auto kernelPkg = targetAttr.cast<mhal::KernelPackageAttr>();
    if (kernelPkg && kernelPkg.getType() == mhal::TargetType::GPU)
      return kernelPkg;
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Convert mhal.launch ops with 'gpu' target to gpu.launch_func ops with
// required memory staging.
//===----------------------------------------------------------------------===//

namespace {
struct LaunchRewritePattern : public OpRewritePattern<mhal::LaunchOp> {
  using OpRewritePattern<mhal::LaunchOp>::OpRewritePattern;

  Value makeWait(OpBuilder b, Location loc, ArrayRef<Value> deps = {}) const {
    auto tokenType = b.getType<gpu::AsyncTokenType>();
    return b.create<gpu::WaitOp>(loc, tokenType, deps).getAsyncToken();
  }

  template <typename T> bool isOnDevice(const T &oprUsers) const {
    for (auto opUse : oprUsers) {
      auto gpuLaunch = dyn_cast<gpu::LaunchFuncOp>(opUse);
      auto launch = dyn_cast<mhal::LaunchOp>(opUse);
      // assumes the same GPU
      if (!gpuLaunch && !(launch && getGPUTarget(launch).has_value()))
        return false;
    }
    return true;
  }

  Value moveMemory(OpBuilder b, mhal::LaunchOp launchOp, Value opr,
                   uint32_t fidx, bool readAccess, bool writeAccess,
                   llvm::SmallVector<Value> &copyBackOprs,
                   llvm::SmallVector<Value, 8> &asyncDeps) const {
    if (auto gpuAllocOp = opr.getDefiningOp<gpu::AllocOp>()) {
      // TEST: convergence or multi-input??
      assert(isOnDevice(opr.getUsers()));
      asyncDeps.push_back(gpuAllocOp.getAsyncToken());
      return opr;
    }

    Location loc = opr.getLoc();
    auto tokenType = b.getType<gpu::AsyncTokenType>();
    auto oprAllocOp = opr.getDefiningOp<memref::AllocOp>();
    auto bAlloc = b;
    if (oprAllocOp)
      bAlloc.setInsertionPointAfter(oprAllocOp);

    Value allocWait = makeWait(bAlloc, loc);
    Type gpuMemType = opr.getType();
    auto dst = bAlloc.create<gpu::AllocOp>(loc, gpuMemType, tokenType,
                                           ValueRange{allocWait}, ValueRange{},
                                           ValueRange{});
    Value dstMem = dst.getResult(0);
    Value dstToken = dst.getResult(1);

    auto makeCopy = [&]() {
      if (readAccess) {
        // copy to device
        auto memcpyToken = b.create<gpu::MemcpyOp>(
            loc, tokenType, ValueRange{dstToken}, dstMem, opr);
        dstToken = memcpyToken.getResult(0);
      }
      if (writeAccess) {
        // copy from device
        copyBackOprs[fidx] = oprAllocOp ? opr : dstMem;
      }
    };

    if (oprAllocOp) {
      // if alloc, convert to gpu.alloc and gpu.memcpy's
      SmallVector<Operation *, 4> oprUsers(opr.getUsers());
      if (isOnDevice(oprUsers)) {
        opr.replaceAllUsesWith(dstMem);
      } else {
        // substitute
        launchOp->replaceUsesOfWith(opr, dstMem);
        makeCopy();
      }
    } else
      makeCopy();

    asyncDeps.push_back(dstToken);
    return dstMem;
  }

  LogicalResult matchAndRewrite(mhal::LaunchOp op,
                                PatternRewriter &rw) const override {
    Location loc = op.getLoc();
    auto caller = op->getParentOfType<func::FuncOp>();
    auto module = caller->getParentOfType<ModuleOp>();
    auto *ctx = module.getContext();

    assert(op->getNumResults() == 1); // only 1 mhal.token

    // 1. get target{gpu} attribute from func

    auto kernelPkg = getGPUTarget(op);
    if (!kernelPkg.has_value())
      return rw.notifyMatchFailure(op, "no gpu target");

    auto arch = kernelPkg->getTarget();
    auto targetObj = kernelPkg->getObject();
    auto binary = targetObj.getBinary();
    auto launchDims = kernelPkg->getLaunchDims();
    if (launchDims.size() != 2)
      return rw.notifyMatchFailure(op, "bad launch dims");
    auto gridSize = launchDims[0];
    auto blockSize = launchDims[1];

    auto func = *getCalledFunc(op);
    Location floc = func.getLoc();

    // 2. create dummy gpu.module for reference from gpu.launch_func
    //    - with gpu.binary, arch attributes
    //    - and gpu.func (referenced by gpu.launch_func
    //    gpu.module @<func_name>_module attributes {arch = "gfx908", gpu.binary
    //        = "\7FELF\..."} {
    //      gpu.func @<func_name> (...) attributes {block_size = 256 : i32,
    //          grid_size = 900 : i32, gpu.kernel}

    FunctionOpInterface funcIF(func);
    auto funcName = funcIF.getName();
    auto gpuModuleName = funcName + "_module";

    auto gpuModule = module.lookupSymbol<gpu::GPUModuleOp>(gpuModuleName.str());
    if (!gpuModule) {
      OpBuilder b(ctx);
      gpuModule = b.create<gpu::GPUModuleOp>(floc, gpuModuleName.str());
      gpuModule->setAttr("arch", b.getStringAttr(arch));
      gpuModule->setAttr("gpu.binary", b.getStringAttr(binary));

      SymbolTable symbolTable(module);
      symbolTable.insert(gpuModule);
    }

    auto gpuFunc = gpuModule.lookupSymbol<gpu::GPUFuncOp>(funcName);
    if (!gpuFunc) {
      OpBuilder b(gpuModule.getContext());
      gpuFunc =
          b.create<gpu::GPUFuncOp>(floc, funcName, func.getFunctionType());
      gpuFunc->setAttr("block_size", b.getI32IntegerAttr(blockSize));
      gpuFunc->setAttr("grid_size", b.getI32IntegerAttr(gridSize));
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());

      SymbolTable symbolTable(gpuModule);
      symbolTable.insert(gpuFunc);

      // Must have a return
      auto block = &gpuFunc.front();
      b.setInsertionPoint(block, block->begin());
      b.create<gpu::ReturnOp>(floc, ValueRange{});
    }

    // 3. create substitute gpu.launch_func
    //    %15 = gpu.wait async
    //    %16 = gpu.launch_func async [%15] @test_fusion_module::@test_fusion
    //    blocks in (%c900, %c1, %c1) threads in (%c256, %c1, %c1)
    //    dynamic_shared_memory_size %c0_i32 args(%4 : memref<128x32x32x8xf32>,
    //    %9 : memref<128x3x3x8xf32>, %14 : memref<128x30x30x128xf32>)

    auto tokenType = rw.getType<gpu::AsyncTokenType>();

    Value oneIdx = rw.createOrFold<arith::ConstantIndexOp>(loc, 1);
    Value blockSizeIdx =
        rw.createOrFold<arith::ConstantIndexOp>(loc, blockSize);
    Value gridSizeIdx = rw.createOrFold<arith::ConstantIndexOp>(loc, gridSize);
    Value dynamicSharedMemorySize;

    // async dependencies
    auto operands = op->getOperands();
    llvm::SmallVector<Value, 8> asyncDeps;
    llvm::SmallVector<Value, 8> gpuOperands;
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
      // move input memories to GPU
      if (opr.getType().isa<MemRefType>()) {
        bool readAccess{
            func.getArgAttr(fidx, func::FuncOp::getReadAccessAttrName())};
        bool writeAccess{
            func.getArgAttr(fidx, func::FuncOp::getWriteAccessAttrName())};
        opr = moveMemory(rw, op, opr, fidx, readAccess, writeAccess,
                         copyBackOprs, asyncDeps);
      }
      gpuOperands.push_back(opr);
    }

    // The gpu.launch_func requires 1 and only 1 token
    if (asyncDeps.size() == 0)
      // There must be at least 1 token
      asyncDeps.push_back(makeWait(rw, loc));
    else if (asyncDeps.size() > 1) {
      // Consolidate to 1 token
      auto launchWait = makeWait(rw, loc, asyncDeps);
      asyncDeps = {launchWait};
    }

    // Make gpu.launch_func
    auto gpuLaunchOp = rw.create<gpu::LaunchFuncOp>(
        loc, asyncDeps, gpuFunc, gpu::KernelDim3{gridSizeIdx, oneIdx, oneIdx},
        gpu::KernelDim3{blockSizeIdx, oneIdx, oneIdx}, dynamicSharedMemorySize,
        gpuOperands);
    Value token = gpuLaunchOp->getResult(0);

    // Insert gpu.memcpy for results
    SmallVector<Value, 8> tokens;
    for (auto pair : llvm::enumerate(copyBackOprs)) {
      if (auto gpuMem = pair.value()) {
        auto dst = operands[diff + pair.index()];
        if (gpuMem.getDefiningOp<memref::AllocOp>())
          std::swap(gpuMem, dst);
        auto memcpy = rw.create<gpu::MemcpyOp>(loc, tokenType,
                                               ValueRange{token}, dst, gpuMem);
        tokens.push_back(memcpy.getResult(0));
      }
    }

    // Consolidate tokens for replacement of mhal.launch
    if (tokens.size() > 1) {
      // insert gpu.wait
      token = makeWait(rw, loc, tokens);
    } else if (tokens.size() == 1)
      token = tokens[0];

    rw.replaceOp(op, {token});

    module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                    rw.getUnitAttr());

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert mhal.await to the corresponding GPU API call.
//===----------------------------------------------------------------------===//

namespace {
struct AwaitRewritePattern : public OpRewritePattern<mhal::AwaitOp> {
  using OpRewritePattern<mhal::AwaitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhal::AwaitOp op,
                                PatternRewriter &rw) const override {
    auto tokenType = rw.getType<gpu::AsyncTokenType>();
    Value input = op->getOperand(0);
    if (input.getType() == tokenType) {
      // mhal.await with token type should never have a result type
      assert(op.getResultType() == std::nullopt);
      rw.create<gpu::WaitOp>(op.getLoc(), Type(), input);
      rw.eraseOp(op);
      return success();
    }

    return rw.notifyMatchFailure(op, "no gpu token");
  }
};
} // namespace

//===----------------------------------------------------------------------===//

namespace {
struct ConvertMHALToGPUPass
    : public impl::ConvertMHALToGPUPassBase<ConvertMHALToGPUPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertMHALToGPUPass::runOnOperation() {
  auto op = getOperation();
  MLIRContext *ctx = op->getContext();

  {
    // Convert mhal.launch to gpu.launch if mhal.targets[gpu] exists
    RewritePatternSet patterns(ctx);
    patterns.add<LaunchRewritePattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
  }

  {
    // Convert mhal.await to gpu.wait if has gpu.tokens
    RewritePatternSet patterns(ctx);
    patterns.add<AwaitRewritePattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
  }

  op.walk([](func::FuncOp f) { f->removeAttr("mhal.targets"); });
}
