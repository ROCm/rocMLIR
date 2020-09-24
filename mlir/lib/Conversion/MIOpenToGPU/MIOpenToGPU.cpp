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
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/XdlopsCodeSelection.h"
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
public:
  LowerMIOpenOpsToGPUPass() = default;
  LowerMIOpenOpsToGPUPass(StringRef kernelName, StringRef gpuModuleName) {
    this->kernelName = kernelName.str();
    this->gpuModuleName = gpuModuleName.str();
  }
  void runOnOperation() override;
};

struct LowerMIOpenOpsWithinGPUModulePass
    : public ConvertMIOpenWithinGPUModuleBase<
          LowerMIOpenOpsWithinGPUModulePass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void LowerMIOpenOpsToGPUPass::runOnOperation() {
  auto op = getOperation();
  OpBuilder b(op.getContext());
  auto loc = op.getLoc();

  // Annotate this module as a container module.
  op.setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
             UnitAttr::get(op.getContext()));

  // Check parameters and populate default values if necessary.
  if (kernelName.empty())
    kernelName = "miopen_conv2d_kcyx_nchw_nkhw";
  if (gpuModuleName.empty())
    gpuModuleName = "miopen_kernel_module";

  bool theGpuModuleExist = false;
  gpu::GPUModuleOp theGpuModule;
  for (auto gpuModule : op.getOps<gpu::GPUModuleOp>()) {
    if (gpuModule.getName() == gpuModuleName) {
      theGpuModuleExist = true;
      theGpuModule = gpuModule;
      break;
    }
  }

  if (!theGpuModuleExist) {
    // create a GPUModuleOp in case the GPU module specified does not exist.
    OperationState state(loc, gpu::GPUModuleOp::getOperationName());
    gpu::GPUModuleOp::build(b, state, gpuModuleName);
    theGpuModule = cast<gpu::GPUModuleOp>(Operation::create(state));

    // add the GPUModuleOp into the symbol table.
    SymbolTable symbolTable(op);
    symbolTable.insert(theGpuModule);
  }

  bool theGpuFuncExist = false;
  gpu::GPUFuncOp theGpuFunc;
  for (auto gpuFunc : theGpuModule.getOps<gpu::GPUFuncOp>()) {
    if (gpuFunc.getName() == kernelName) {
      theGpuFuncExist = true;
      theGpuFunc = gpuFunc;

      // Set kernel attribute.
      gpuFunc.setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                      b.getUnitAttr());
      break;
    }
  }

  if (!theGpuFuncExist) {
    // Try locate a FuncOp which has kernelName, and convert it to a GPUFuncOp.

    SymbolTable gpuModuleSymbolTable(theGpuModule);

    bool theFuncExist = false;
    FuncOp theFunc;
    for (auto func : op.getOps<FuncOp>()) {
      if (func.getName() == kernelName) {
        theFuncExist = true;
        theFunc = func;
        break;
      }
    }

    // Early exit if the function to be rewritten does not exist.
    if (!theFuncExist)
      return;

    // create a GPUFuncOp.
    FunctionType gpuFuncType = theFunc.getType();
    auto gpuFunc =
        b.create<gpu::GPUFuncOp>(loc, theFunc.getName(), gpuFuncType);

    // Set kernel attribute.
    gpuFunc.setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());

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

    // remove std.return ops.
    gpuFunc.walk([&](ReturnOp op) { op.erase(); });

    // create a GPU ReturnOp inside the GPUFuncOp.
    b.setInsertionPointToEnd(&gpuFuncBody.back());
    b.create<gpu::ReturnOp>(loc);

    // insert the GPUFuncOp into GPUModuleOp.
    gpuModuleSymbolTable.insert(gpuFunc);

    // Erase old FuncOp instance.
    theFunc.erase();
  }

  // Convert GPU-specific ops to GPU dialect.
  for (auto module : op.getOps<gpu::GPUModuleOp>()) {
    module.walk([&](gpu::GPUFuncOp gpuFunc) {
      gpuFunc.walk([&](miopen::GpuAllocOp op) {
        auto type = op.output().getType().cast<MemRefType>();

        if (type.getMemorySpace() ==
            gpu::GPUDialect::getWorkgroupAddressSpace()) {
          Value attribution = gpuFunc.addWorkgroupAttribution(type);
          op.replaceAllUsesWith(attribution);
        } else if (type.getMemorySpace() ==
                   gpu::GPUDialect::getPrivateAddressSpace()) {
          Value attribution = gpuFunc.addPrivateAttribution(type);
          op.replaceAllUsesWith(attribution);
        } else {
          // TBD: return failure.
          llvm::errs() << "unsupported addrspace!\n";
        }
        op.erase();
      });

      // TBD see if these patterns could be re-written using tablgen.
      gpuFunc.walk([&](miopen::WorkgroupBarrierOp op) {
        auto loc = op.getLoc();
        OpBuilder b(op.getContext());
        b.setInsertionPoint(op);
        b.create<gpu::BarrierOp>(loc);
        op.erase();
      });

      gpuFunc.walk([&](miopen::WorkgroupIdOp op) {
        auto loc = op.getLoc();
        OpBuilder b(op.getContext());
        b.setInsertionPoint(op);
        Value bid = b.create<gpu::BlockIdOp>(loc, b.getIndexType(), "x");
        op.replaceAllUsesWith(bid);
        op.erase();
      });

      gpuFunc.walk([&](miopen::WorkitemIdOp op) {
        auto loc = op.getLoc();
        OpBuilder b(op.getContext());
        b.setInsertionPoint(op);
        Value tid = b.create<gpu::ThreadIdOp>(loc, b.getIndexType(), "x");
        op.replaceAllUsesWith(tid);
        op.erase();
      });

      gpuFunc.walk([&](miopen::MFMAV2Op op) {
        auto loc = op.getLoc();
        OpBuilder b(op.getContext());
        b.setInsertionPoint(op);

        auto gpuMfmaOp = b.create<gpu::MFMAOp>(loc, op.getType(), op.sourceA(), op.sourceB(), op.destC());
        gpuMfmaOp.setAttr("instr", op.getAttr("instr"));
        gpuMfmaOp.setAttr("imm", op.getAttr("imm"));

        op.replaceAllUsesWith(gpuMfmaOp.destD());
        op.erase();
      });
   
    });
  }
}

void LowerMIOpenOpsWithinGPUModulePass::runOnOperation() {
  OwningRewritePatternList patterns;

  // miopen-lowering
  patterns.insert<Conv2DRewritePattern<miopen::Conv2DOp>>(&getContext());
  patterns.insert<Conv2DRewritePattern<miopen::Conv2DBwdDataOp>>(&getContext());
  patterns.insert<Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>>(
      &getContext());

  // TBD: miopen-affine-transform
  // TBD: miopen-affix-params

  // miopen-lowering-step2
  patterns.insert<GridwiseGemmRewritePattern>(&getContext());

  // miopen-lowering-step3
  patterns.insert<FillRewritePattern>(&getContext());
  patterns.insert<MovePosRewritePattern>(&getContext());
  patterns.insert<SubviewRewritePattern>(&getContext());
  patterns.insert<TransformRewritePattern>(&getContext());
  patterns.insert<BlockwiseGemmRewritePattern>(&getContext());
  patterns.insert<BlockwiseCopyRewritePattern>(&getContext());

  // miopen-lowering-step4
  patterns.insert<ThreadwiseGemmRewritePattern>(&getContext());
  patterns.insert<ThreadwiseCopyRewritePattern>(&getContext());

  // miopen-lowering-step5
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());

  applyPatternsAndFoldGreedily(getOperation(), patterns);
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::createLowerMIOpenOpsToGPUPass(StringRef kernelName,
                                    StringRef gpuModuleName) {
  return std::make_unique<LowerMIOpenOpsToGPUPass>(kernelName, gpuModuleName);
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createLowerMIOpenOpsWithinGPUModulePass() {
  return std::make_unique<LowerMIOpenOpsWithinGPUModulePass>();
}
