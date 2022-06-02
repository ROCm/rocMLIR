//===- AsyncToGPU.cpp - Convert Async to GPU dialect ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AsyncToGPU/AsyncToGPU.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "convert-async-to-llvm"

using namespace mlir;
using namespace mlir::async;

//===----------------------------------------------------------------------===//
// Convert Async dialect types to LLVM types.
//===----------------------------------------------------------------------===//

namespace {
/// AsyncGPUTypeConverter only converts types from the Async dialect to
/// the corresponding GPU type and does not convert any other types.
class AsyncGPUTypeConverter : public TypeConverter {
public:
  AsyncGPUTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(convertAsyncTypes);
  }

  static Optional<Type> convertAsyncTypes(Type type) {
    if (type.isa<TokenType>())
      return gpu::AsyncTokenType::get(type.getContext());

    return llvm::None;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.launch ops with 'gpu' target to gpu.launch_func ops with
// required memory staging.
//===----------------------------------------------------------------------===//

namespace {
class LaunchOpConversion : public OpConversionPattern<async::LaunchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  static Optional<FuncOp> getCalledFunc(async::LaunchOp op) {
    CallOpInterface callIf(op);
    auto *callable = callIf.resolveCallable();
    if (!callable)
      return llvm::None;
    return dyn_cast<FuncOp>(callable);
  }

  // Get target{gpu} attribute from called func
  static Optional<DictionaryAttr> getGPUTarget(async::LaunchOp op) {
    auto func = getCalledFunc(op);
    if (!func.hasValue() || func->getNumResults() != 0)
      return llvm::None;

    auto attr = (*func)->template getAttrOfType<ArrayAttr>("targets");
    if (!attr)
      return llvm::None;

    for (auto targetAttr : attr.getValue()) {
      auto dictAttr = targetAttr.cast<DictionaryAttr>();
      auto type = dictAttr.get("type");
      if (type && type.template cast<StringAttr>() == "gpu")
        return dictAttr;
    }
    return llvm::None;
  }

  static Value makeWait(OpBuilder b, Location loc, ArrayRef<Value> deps={}) {
    auto tokenType = b.getType<gpu::AsyncTokenType>();
    return b.create<gpu::WaitOp>(loc, tokenType, deps).asyncToken();
  }

  Value moveMemory(OpBuilder b, Value opr, uint32_t fidx,
                   func::AccessMode accessMode,
                   llvm::SmallVector<Value, 8> &copyBackOprs,
                   llvm::SmallVector<Value, 8> &asyncDeps) const {
    auto loc = opr.getLoc();
    auto tokenType = b.getType<gpu::AsyncTokenType>();
    auto oprAllocOp = opr.getDefiningOp<memref::AllocOp>();
    if (oprAllocOp)
      b.setInsertionPoint(oprAllocOp);
    /// hack: add one global wait for gpu.allocs
    auto allocWait = makeWait(b, loc);
    auto gpuMemType = opr.getType();
    auto dst = b.create<gpu::AllocOp>(loc, gpuMemType, tokenType,
                                      ValueRange{allocWait}, ValueRange{},
                                      ValueRange{});
    auto dstMem = dst.getResult(0);
    auto dstToken = dst.getResult(1);
    // if alloc, convert to gpu.alloc
    if (oprAllocOp) {
      // TODO(sjw): make sure accessors are all on the GPU
      oprAllocOp->replaceAllUsesWith(ValueRange{dstMem});
    } else {
      if (func::isAccessModeRead(accessMode)) {
        // else copy to device
        auto asyncDeps = ValueRange{dstToken};
        auto memcpyToken =
            b.create<gpu::MemcpyOp>(loc, tokenType, asyncDeps, dstMem, opr);
        dstToken = memcpyToken.getResult(0);
      }
      if (func::isAccessModeWrite(accessMode)) {
        copyBackOprs[fidx] = dstMem;
      }
    }
    asyncDeps.push_back(dstToken);
    return dstMem;
  }

  LogicalResult
  matchAndRewrite(async::LaunchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = module.getContext();

    assert(op->getNumResults() == 1); // only 1 async.token

    // 1. get target{gpu} attribute from func

    auto gpuAttr = getGPUTarget(op);
    if (!gpuAttr.hasValue())
      return op.emitOpError("requires a gpu target");

    auto arch = gpuAttr->get("arch");
    auto binary = gpuAttr->get("binary");
    auto blockSize = gpuAttr->get("block_size").template cast<IntegerAttr>();
    auto gridSize = gpuAttr->get("grid_size").template cast<IntegerAttr>();

    auto func = *getCalledFunc(op);
    auto floc = func.getLoc();

    // Also capture the accessMap for the params, default all to read-write
    SmallVector<func::AccessMode, 8> accessVec(func.getNumArguments(),
                                               func::AccessMode::ReadWrite);
    if (auto accessMap =
            func->template getAttrOfType<ArrayAttr>("access_map")) {
      std::transform(
          accessMap.begin(), accessMap.end(), accessVec.begin(),
          [](Attribute a) {
            return a.template cast<func::AccessModeAttr>().getValue();
          });
    }

    // 2. create dummy gpu.module
    //    - with gpu.binary, arch attributes
    //    - and llvm.func (may not be necessary
    //    gpu.module @<func_name>_module attributes {arch = "gfx908", gpu.binary
    //    = "\7FELF\..."} {
    //      llvm.func @<func_name> (...) attributes {block_size = 256 : i32,
    //      gpu.kernel, grid_size = 900 : i32, rocdl.kernel}
    //        // does any of that matter?

    FunctionOpInterface funcIF(func);
    auto funcName = funcIF.getName();
    auto gpuModuleName = funcName + "_module";

    auto gpuModule = module.lookupSymbol<gpu::GPUModuleOp>(gpuModuleName.str());
    if (!gpuModule) {
      OpBuilder b(ctx);
      gpuModule = b.create<gpu::GPUModuleOp>(floc, gpuModuleName.str());
      gpuModule->setAttr("arch", arch);
      gpuModule->setAttr("gpu.binary", binary);

      SymbolTable symbolTable(module);
      symbolTable.insert(gpuModule);
    }

    auto gpuFunc = gpuModule.lookupSymbol<gpu::GPUFuncOp>(funcName);
    if (!gpuFunc) {
      OpBuilder b(gpuModule.getContext());
      gpuFunc =
          b.create<gpu::GPUFuncOp>(floc, funcName, func.getFunctionType());
      gpuFunc->setAttr("block_size", blockSize);
      gpuFunc->setAttr("grid_size", gridSize);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());

      SymbolTable symbolTable(gpuModule);
      symbolTable.insert(gpuFunc);

      // Emit gpu convolution logic.
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

    auto tokenType = rewriter.getType<gpu::AsyncTokenType>();

    auto zeroIdx =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    auto blockSizeIdx = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(blockSize.getValue().getLimitedValue()));
    auto gridSizeIdx = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(gridSize.getValue().getLimitedValue()));
    Value dynamicSharedMemorySize;

    // async dependencies
    auto operands = adaptor.getOperands();
    llvm::SmallVector<Value, 8> asyncDeps;
    llvm::SmallVector<Value, 8> gpuOperands;
    size_t diff = operands.size() - func.getNumArguments();
    size_t i = 0;
    if (diff > 0) {
      for (; i < diff; ++i)
        asyncDeps.push_back(operands[i]);
    } else
      assert(diff == 0);

    SmallVector<Value, 8> copyBackOprs(func.getNumArguments(), Value());
    for (; i < operands.size(); ++i) {
      auto fidx = i - diff;
      Value opr = operands[i];
      // move input memories to GPU
      if (!opr.getDefiningOp<gpu::AllocOp>()) {
        opr = moveMemory(rewriter, opr, fidx, accessVec[fidx], copyBackOprs,
                         asyncDeps);
      }
      gpuOperands.push_back(opr);
    }

    // The gpu.launch_func requires 1 and only 1 token
    if (asyncDeps.size() == 0)
      // There must be at least 1 token
      asyncDeps.push_back(makeWait(rewriter, loc));
    else if (asyncDeps.size() > 1) {
      // Consolidate to 1 token
      auto launchWait = makeWait(rewriter, loc, asyncDeps);
      asyncDeps = {launchWait};
    }

    // Make gpu.launch_func
    auto gpuLaunchOp = rewriter.create<gpu::LaunchFuncOp>(
        loc, asyncDeps, gpuFunc,
        gpu::KernelDim3{gridSizeIdx, zeroIdx, zeroIdx},
        gpu::KernelDim3{blockSizeIdx, zeroIdx, zeroIdx},
        dynamicSharedMemorySize, gpuOperands);
    Value token = gpuLaunchOp->getResult(0);

    // Insert gpu.memcpy for results
    SmallVector<Value, 8> tokens;
    for (auto pair : llvm::enumerate(copyBackOprs)) {
      if (auto gpuMem = pair.value()) {
        auto dst = operands[diff + pair.index()];
        auto memcpy = rewriter.create<gpu::MemcpyOp>(
            loc, tokenType, ValueRange{token}, dst, gpuMem);
        tokens.push_back(memcpy.getResult(0));
      }
    }

    // Consolidate tokens for replacement of async.launch
    if (tokens.size() > 1) {
      // insert gpu.wait
      token = makeWait(rewriter, loc, tokens);
    } else if (tokens.size() == 1)
      token = tokens[0];

    rewriter.replaceOp(op, {token});

    module->setAttr("gpu.container_module", rewriter.getUnitAttr());

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.await to the corresponding GPU API call.
//===----------------------------------------------------------------------===//

namespace {
class AwaitOpConversion : public OpConversionPattern<async::AwaitOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(async::AwaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tokenType = rewriter.getType<gpu::AsyncTokenType>();
    rewriter.create<gpu::WaitOp>(op.getLoc(), tokenType, adaptor.getOperands());
    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//

namespace {
struct ConvertAsyncToGPUPass
    : public ConvertAsyncToGPUBase<ConvertAsyncToGPUPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertAsyncToGPUPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module->getContext();

  // Convert async dialect types and operations to LLVM dialect.
  AsyncGPUTypeConverter converter;
  RewritePatternSet patterns(ctx);

  // Convert return operations inside async.execute regions.
  patterns.add<LaunchOpConversion>(converter, ctx);

  // Convert async.await operations
  patterns.add<AwaitOpConversion>(converter, ctx);

  ConversionTarget target(*ctx);
  target.addLegalOp<arith::ConstantOp, func::ConstantOp,
                    UnrealizedConversionCastOp>();
  target.addLegalDialect<gpu::GPUDialect, LLVM::LLVMDialect>();

  // All operations from Async dialect must be lowered to the runtime API and
  // LLVM intrinsics calls.
  target.addIllegalDialect<async::AsyncDialect>();

  // Add dynamic legality constraints to apply conversions defined above.
  // target.addDynamicallyLegalOp<async::LaunchOp>([&](async::LaunchOp op) {
  //     return !LaunchOpConversion::getGPUTarget(op).hasValue();
  // });
  // target.addDynamicallyLegalOp<async::AwaitOp>([&](async::AwaitOp op) {
  //     return true;
  // });

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertAsyncToGPUPass() {
  return std::make_unique<ConvertAsyncToGPUPass>();
}
