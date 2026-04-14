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
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "convert-mhal-to-gpu"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMHALTOGPUPASS
#include "mlir/Conversion/MHALPasses.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::mhal;

//===----------------------------------------------------------------------===//
// Lower bufferized host calls to GPU kernels (mhal.targets).
//===----------------------------------------------------------------------===//

/// mhal.targets[gpu] lookup for a kernel function symbol (bufferized
/// func.call).
static std::optional<mhal::KernelPackageAttr> getGPUTarget(func::FuncOp func) {
  if (func.getNumResults() != 0)
    return std::nullopt;

  auto attr = func->getAttrOfType<ArrayAttr>("mhal.targets");
  if (!attr)
    return std::nullopt;

  for (auto targetAttr : attr.getValue()) {
    auto kernelPkg = cast<mhal::KernelPackageAttr>(targetAttr);
    if (kernelPkg && kernelPkg.getType() == mhal::TargetType::GPU)
      return kernelPkg;
  }
  return std::nullopt;
}

namespace {
/// Lower a bufferized func.call to a GPU kernel (mhal.targets) to
/// gpu.launch_func with staging; synchronize with gpu.wait and erase the
/// call.
static LogicalResult lowerKernelCallToGpu(PatternRewriter &rw, func::CallOp op,
                                          func::FuncOp func) {
  Location loc = op.getLoc();
  auto module = op->getParentOfType<ModuleOp>();
  MLIRContext *ctx = module.getContext();

  auto kernelPkg = getGPUTarget(func);
  if (!kernelPkg.has_value())
    return rw.notifyMatchFailure(op, "no gpu target");

  auto targetObj = kernelPkg->getObject();
  auto binary = targetObj.getBinary();
  auto launchDims = kernelPkg->getLaunchDims();
  if (launchDims.size() != 2)
    return rw.notifyMatchFailure(op, "bad launch dims");
  auto gridSize = launchDims[0];
  auto blockSize = launchDims[1];

  FunctionOpInterface funcIF(func);
  auto funcName = funcIF.getName();
  std::string binaryName = (funcName + "_module").str();

  auto binaryOp = module.lookupSymbol<gpu::BinaryOp>(binaryName);
  if (!binaryOp) {
    OpBuilder b(ctx);
    binaryOp = gpu::BinaryOp::create(b, loc, binaryName, nullptr,
                                     ArrayRef<Attribute>({binary}));

    SymbolTable symbolTable(module);
    symbolTable.insert(binaryOp);
  }

  auto makeWait = [&](OpBuilder &b, Location l, ArrayRef<Value> deps) {
    auto tt = b.getType<gpu::AsyncTokenType>();
    return b.create<gpu::WaitOp>(l, tt, deps).getAsyncToken();
  };

  auto userOnDevice = [&](Operation *userOp) {
    if (isa<gpu::LaunchFuncOp>(userOp))
      return true;
    if (auto call = dyn_cast<func::CallOp>(userOp)) {
      if (auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee()))
        return getGPUTarget(callee).has_value();
    }
    return false;
  };

  auto moveMemory = [&](Operation *anchor, Value opr, uint32_t fidx,
                        bool writeAccess,
                        llvm::SmallVector<Value> &copyBackOprs,
                        llvm::SmallVector<Value, 8> &asyncDeps) -> Value {
    if (auto gpuAllocOp = opr.getDefiningOp<gpu::AllocOp>()) {
      assert(llvm::all_of(opr.getUsers(), userOnDevice));
      asyncDeps.push_back(gpuAllocOp.getAsyncToken());
      return opr;
    }
    Location oloc = opr.getLoc();
    OpBuilder b = rw;
    auto tokenType = b.getType<gpu::AsyncTokenType>();
    auto oprAllocOp = opr.getDefiningOp<memref::AllocOp>();
    OpBuilder bAlloc = b;
    if (oprAllocOp)
      bAlloc.setInsertionPointAfter(oprAllocOp);
    Value allocWait = makeWait(bAlloc, oloc, {});
    auto dst = bAlloc.create<gpu::AllocOp>(oloc, opr.getType(), tokenType,
                                           ValueRange{allocWait}, ValueRange{},
                                           ValueRange{});
    Value dstMem = dst.getResult(0);
    Value dstToken = dst.getResult(1);
    auto runCopy = [&] {
      dstToken = b.create<gpu::MemcpyOp>(oloc, tokenType, ValueRange{dstToken},
                                         dstMem, opr)
                     .getResult(0);
      if (writeAccess)
        copyBackOprs[fidx] = oprAllocOp ? opr : dstMem;
    };
    if (oprAllocOp) {
      bool allOnDev = true;
      for (Operation *u : opr.getUsers()) {
        if (!userOnDevice(u)) {
          allOnDev = false;
          break;
        }
      }
      if (allOnDev)
        opr.replaceAllUsesWith(dstMem);
      else {
        anchor->replaceUsesOfWith(opr, dstMem);
        runCopy();
      }
    } else
      runCopy();
    asyncDeps.push_back(dstToken);
    return dstMem;
  };

  auto tokenType = rw.getType<gpu::AsyncTokenType>();
  Value oneIdx = rw.createOrFold<arith::ConstantIndexOp>(loc, 1);
  Value blockSizeIdx = rw.createOrFold<arith::ConstantIndexOp>(loc, blockSize);
  Value gridSizeIdx = rw.createOrFold<arith::ConstantIndexOp>(loc, gridSize);
  Value dynamicSharedMemorySize;

  auto operands = op->getOperands();
  llvm::SmallVector<Value, 8> asyncDeps;
  llvm::SmallVector<Value, 8> gpuOperands;

  SmallVector<Value> copyBackOprs(func.getNumArguments(), Value());
  for (size_t i = 0; i < operands.size(); ++i) {
    auto fidx = i;
    Value opr = operands[i];
    if (isa<MemRefType>(opr.getType())) {
      bool wa{
          func.getArgAttr(fidx, mhal::MHALDialect::getWriteAccessAttrName())};
      opr = moveMemory(op, opr, fidx, wa, copyBackOprs, asyncDeps);
    }
    gpuOperands.push_back(opr);
  }

  if (asyncDeps.empty())
    asyncDeps.push_back(makeWait(rw, loc, {}));
  else if (asyncDeps.size() > 1)
    asyncDeps = {makeWait(rw, loc, asyncDeps)};

  auto gpuLaunchOp = gpu::LaunchFuncOp::create(
      rw, loc,
      SymbolRefAttr::get(ctx, binaryName,
                         {FlatSymbolRefAttr::get(ctx, funcName)}),
      gpu::KernelDim3{gridSizeIdx, oneIdx, oneIdx},
      gpu::KernelDim3{blockSizeIdx, oneIdx, oneIdx}, dynamicSharedMemorySize,
      gpuOperands, tokenType, ValueRange(asyncDeps));
  Value token = gpuLaunchOp->getResult(0);

  // Insert gpu.memcpy for results
  SmallVector<Value, 8> tokens;
  for (auto pair : llvm::enumerate(copyBackOprs)) {
    if (auto gpuMem = pair.value()) {
      auto dst = operands[pair.index()];
      if (gpuMem.getDefiningOp<memref::AllocOp>())
        std::swap(gpuMem, dst);
      auto memcpy = rw.create<gpu::MemcpyOp>(loc, tokenType, ValueRange{token},
                                             dst, gpuMem);
      tokens.push_back(memcpy.getResult(0));
    }
  }

  if (tokens.size() > 1)
    token = makeWait(rw, loc, tokens);
  else if (tokens.size() == 1)
    token = tokens[0];

  rw.create<gpu::WaitOp>(loc, Type(), token);
  rw.eraseOp(op);

  module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                  rw.getUnitAttr());
  return success();
}

/// Bufferized func.call to a kernel with mhal.targets (e.g.
/// clone-harness).
struct KernelFuncCallRewritePattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter &rw) const override {
    if (op.getNumResults() != 0)
      return failure();
    auto func = op->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(
        op.getCallee());
    if (!func || !getGPUTarget(func).has_value())
      return failure();
    assert(op->getNumOperands() == static_cast<size_t>(func.getNumArguments()));
    return lowerKernelCallToGpu(rw, op, func);
  }
};
} // namespace

namespace {
struct ConvertMHALToGPUPass
    : public impl::ConvertMHALToGPUPassBase<ConvertMHALToGPUPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertMHALToGPUPass::runOnOperation() {
  auto op = getOperation();
  MLIRContext *ctx = op->getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<KernelFuncCallRewritePattern>(ctx);

  if (failed(applyPatternsGreedily(op, std::move(patterns))))
    signalPassFailure();

  op.walk([](func::FuncOp f) { f->removeAttr("mhal.targets"); });
}
