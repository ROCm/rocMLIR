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

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MIOpen/LowerMIOpenOps.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/XdlopsCodeSelection.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {
struct LowerMIOpenOpsToGPUPass : public ConvertMIOpenToGPUBase<LowerMIOpenOpsToGPUPass> {
public:
  LowerMIOpenOpsToGPUPass() = default;
  void runOnOperation() override;
};

struct LowerMIOpenOpsWithinGPUModulePass
    : public ConvertMIOpenWithinGPUModuleBase<
          LowerMIOpenOpsWithinGPUModulePass> {
  void runOnOperation() override;
};
} // end anonymous namespace

namespace {

//===----------------------------------------------------------------------===//
// MIOpen Operation pattern lowering.
//===----------------------------------------------------------------------===//

struct MIGPUAllocRewritePattern : public OpRewritePattern<miopen::GpuAllocOp> {
  using OpRewritePattern<miopen::GpuAllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::GpuAllocOp op,
                                PatternRewriter &b) const override {
    auto type = op.output().getType().cast<MemRefType>();
    auto func = op->getParentOfType<gpu::GPUFuncOp>();

    if (type.getMemorySpaceAsInt() == gpu::GPUDialect::getWorkgroupAddressSpace()) {
      Value attribution = func.addWorkgroupAttribution(type);
      op.replaceAllUsesWith(attribution);
    } else if (type.getMemorySpaceAsInt() ==
               gpu::GPUDialect::getPrivateAddressSpace()) {
      Value attribution = func.addPrivateAttribution(type);
      op.replaceAllUsesWith(attribution);
    } else {
      // TBD: return failure.
      llvm::errs() << "unsupported addrspace!\n";
    }
    op.erase();
    return success();
  }
};

struct MIDataConvertRewritePattern
    : public OpRewritePattern<miopen::DataConvertOp> {
  using OpRewritePattern<miopen::DataConvertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::DataConvertOp op,
                                PatternRewriter &b) const override {
    Value nop =
        b.create<gpu::BFConvertOp>(op.getLoc(), op.out().getType(), op.in());
    op.replaceAllUsesWith(nop);
    op.erase();
    return success();
  }
};

template <typename Tmi, typename Tgpu>
struct MIOpRewritePattern : public OpRewritePattern<Tmi> {
  using OpRewritePattern<Tmi>::OpRewritePattern;

  LogicalResult matchAndRewrite(Tmi op, PatternRewriter &b) const override {
    b.create<Tgpu>(op.getLoc());
    op.erase();
    return success();
  }
};

template <typename Tmi, typename Tgpu>
struct MIIdRewritePattern : public OpRewritePattern<Tmi> {
  using OpRewritePattern<Tmi>::OpRewritePattern;

  LogicalResult matchAndRewrite(Tmi op, PatternRewriter &b) const override {
    Value nop = b.create<Tgpu>(op.getLoc(), b.getIndexType(), "x");
    op.replaceAllUsesWith(nop);
    op.erase();
    return success();
  }
};

struct MIMFMARewritePattern : public OpRewritePattern<miopen::MFMAV2Op> {
  using OpRewritePattern<miopen::MFMAV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::MFMAV2Op op,
                                PatternRewriter &b) const override {
    auto gpuMfmaOp = b.create<gpu::MFMAOp>(
        op.getLoc(), op.getType(), op.sourceA(), op.sourceB(), op.destC());
    gpuMfmaOp->setAttr("instr", op->getAttr("instr"));
    gpuMfmaOp->setAttr("imm", op->getAttr("imm"));
    op.replaceAllUsesWith(Value(gpuMfmaOp));
    op.erase();
    return success();
  }
};

} // namespace

void LowerMIOpenOpsToGPUPass::runOnOperation() {
  auto op = getOperation();
  auto *ctx = op.getContext();
  OpBuilder b(ctx);
  auto loc = op.getLoc();

  // Annotate this module as a container module.
  op->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
              UnitAttr::get(ctx));

  auto makeGpuModule = [&](StringRef name) {
    // create a GPUModuleOp in case the GPU module specified does not exist.
    auto gpuModule = b.create<gpu::GPUModuleOp>(loc, name);

    // add the GPUModuleOp into the symbol table.
    SymbolTable symbolTable(op);
    symbolTable.insert(gpuModule);

    return gpuModule;
  };

  auto processGpuKernelFunc = [&](FuncOp &theFunc,
                                  OpBuilder &b) -> gpu::GPUFuncOp {
    // create a GPUFuncOp.
    FunctionType gpuFuncType = theFunc.getType();
    auto gpuFunc =
        b.create<gpu::GPUFuncOp>(loc, theFunc.getName(), gpuFuncType);

    // Set kernel attribute.
    gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    if (auto attr = theFunc->getAttr("block_size"))
      gpuFunc->setAttr("block_size", attr);
    if (auto attr = theFunc->getAttr("grid_size"))
      gpuFunc->setAttr("grid_size", attr);

    // associate arguments for newly created GPUFuncOp.
    BlockAndValueMapping map;
    for (unsigned idx = 0; idx < theFunc.getNumArguments(); ++idx) {
      auto arg = theFunc.getArgument(idx);
      auto gpuFuncArg = gpuFunc.getArgument(idx);

      map.map(arg, gpuFuncArg);
    }

    // clone function body into newly created GPUFuncOp.
    Region &gpuFuncBody = gpuFunc.body();
    Region &funcBody = theFunc.getBody();
    funcBody.cloneInto(&gpuFuncBody, map);

    // add a branch op to the cloned region.
    Block &funcEntry = funcBody.front();
    Block *clonedFuncEntry = map.lookup(&funcEntry);
    Block &gpuFuncEntry = gpuFuncBody.front();
    b.setInsertionPointToEnd(&gpuFuncEntry);
    b.create<BranchOp>(loc, clonedFuncEntry);

    return gpuFunc;
  };

  SmallVector<FuncOp, 1> processedFuncs;
  // Check parameters and populate default values if necessary.
  for (auto func : op.getOps<FuncOp>()) {
    if (func->hasAttr("kernel")) {
      std::string gfname = func.getName().str();
      gfname += "_module";
      auto gpuMod = makeGpuModule(gfname);
      // Set up the symbol table for the GPU ModuleOp.
      SymbolTable gpuModuleSymbolTable(gpuMod);
      // Reset builder insertion point to the beginning of the GPU module,
      // as it would be modified inside the lambda.
      OpBuilder bmod(gpuMod.getContext());
      auto gpuFunc = processGpuKernelFunc(func, bmod);

      // insert the GPUFuncOp into GPUModuleOp.
      gpuModuleSymbolTable.insert(gpuFunc);

      processedFuncs.push_back(func);
    }
  }

  // Remove all processed FuncOp instances.
  for (auto func : processedFuncs) {
    func.erase();
  }

  // Convert MIOpen ops to GPU Ops
  int gpuModCount = 0;
  op.walk([this, &gpuModCount](gpu::GPUModuleOp gpuMod) {
    gpuModCount++;
    auto *ctx = &getContext();
    OwningRewritePatternList patterns(ctx);

    // miopen-lowering
    patterns.insert<MIGPUAllocRewritePattern>(ctx);
    patterns.insert<MIDataConvertRewritePattern>(ctx);
    patterns.insert<MIOpRewritePattern<miopen::WorkgroupBarrierOp, gpu::BarrierOp>>(ctx);
    patterns.insert<MIOpRewritePattern<miopen::LDSBarrierOp, gpu::LDSBarrierOp>>(ctx);
    patterns.insert<MIIdRewritePattern<miopen::WorkgroupIdOp, gpu::BlockIdOp>>(ctx);
    patterns.insert<MIIdRewritePattern<miopen::WorkitemIdOp, gpu::ThreadIdOp>>(ctx);
    patterns.insert<MIOpRewritePattern<ReturnOp, gpu::ReturnOp>>(ctx);

    patterns.insert<MIMFMARewritePattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(gpuMod, std::move(patterns))))
      signalPassFailure();
  });

  if (gpuModCount == 0) {
    // Must have at least 1 gpu.module for rocm-runner
    makeGpuModule("miopen_gpu_module");
  }
}

void LowerMIOpenOpsWithinGPUModulePass::runOnOperation() {
  auto *ctx = &getContext();
  OwningRewritePatternList patterns(ctx);

  // miopen-lowering
  patterns.insert<Conv2DRewritePattern<miopen::Conv2DOp>>(ctx);
  patterns.insert<Conv2DRewritePattern<miopen::Conv2DBwdDataOp>>(ctx);
  patterns.insert<Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>>(ctx);

  // TBD: miopen-affine-transform
  // TBD: miopen-affix-params

  // miopen-lowering-step2
  patterns.insert<GridwiseGemmRewritePattern>(ctx);

  // miopen-lowering-step3
  patterns.insert<FillRewritePattern>(ctx);
  patterns.insert<SubviewRewritePattern>(ctx);
  patterns.insert<TransformRewritePattern>(ctx);
  patterns.insert<BlockwiseGemmRewritePattern>(ctx);

  // miopen-lowering-step4
  patterns.insert<ThreadwiseGemmRewritePattern>(ctx);
  patterns.insert<ThreadwiseCopyRewritePattern>(ctx);

  // miopen-lowering-step5
  populateAffineToStdConversionPatterns(patterns);
  populateLoopToStdConversionPatterns(patterns);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::createLowerMIOpenOpsToGPUPass() {
  return std::make_unique<LowerMIOpenOpsToGPUPass>();
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createLowerMIOpenOpsWithinGPUModulePass() {
  return std::make_unique<LowerMIOpenOpsWithinGPUModulePass>();
}
