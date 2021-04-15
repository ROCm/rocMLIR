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
  LowerMIOpenOpsToGPUPass(StringRef kernelNameList,
                          StringRef gpuModuleNameList) {
    this->kernelNameList = kernelNameList.str();
    this->gpuModuleNameList = gpuModuleNameList.str();
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
  op->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
              UnitAttr::get(op.getContext()));

  // Populate the list of kernels to be converted.
  SmallVector<StringRef, 1> kernelNameTable;
  if (kernelNameList.empty()) {
    // In case the kernel name list is not specified, use all FuncOp with kernel
    // attribute.
    for (auto func : op.getOps<FuncOp>())
      if (func->getAttr("kernel"))
        kernelNameTable.push_back(func.getName());
  } else {
    // Split kernelNameList into a vector separated with comma.
    StringRef remainingKernelNameList = kernelNameList;
    do {
      std::pair<StringRef, StringRef> p = remainingKernelNameList.split(',');
      kernelNameTable.push_back(p.first);
      remainingKernelNameList = p.second;
    } while (!remainingKernelNameList.empty());
  }

  // Populate the list of GPU modules to host kernels.
  SmallVector<StringRef, 1> gpuModuleNameTable;
  if (gpuModuleNameList.empty()) {
    // In case the GPU module name is not specified, simply use kernel name
    // list.
    gpuModuleNameTable = kernelNameTable;
  } else {
    // Split gpuModuleNameList into a vector separated with comma.
    StringRef remainingGpuModuleNameList = gpuModuleNameList;
    do {
      std::pair<StringRef, StringRef> p = remainingGpuModuleNameList.split(',');
      gpuModuleNameTable.push_back(p.first);
      remainingGpuModuleNameList = p.second;
    } while (!remainingGpuModuleNameList.empty());
  }

  if (gpuModuleNameTable.size() != kernelNameTable.size()) {
    llvm::errs() << "FAIL!\n";
    return;
  }

  // Identify the specified GPU ModuleOp.
  bool gpuModuleInstancesExist = false;
  SmallVector<gpu::GPUModuleOp, 1> gpuModuleTable;
  for (auto gpuModule : op.getOps<gpu::GPUModuleOp>()) {
    for (auto gpuModuleName : gpuModuleNameTable) {
      if (gpuModule.getName() == gpuModuleName) {
        gpuModuleInstancesExist = true;
        gpuModuleTable.push_back(gpuModule);
        break;
      }
    }
  }

  SymbolTable symbolTable(op);
  if (!gpuModuleInstancesExist) {
    // create a GPUModuleOp in case the GPU module specified does not exist.
    for (auto gpuModuleName : gpuModuleNameTable) {
      OperationState state(loc, gpu::GPUModuleOp::getOperationName());
      gpu::GPUModuleOp::build(b, state, gpuModuleName);
      auto gpuModule = cast<gpu::GPUModuleOp>(Operation::create(state));

      gpuModuleTable.push_back(gpuModule);

      // add the GPUModuleOp into the symbol table.
      symbolTable.insert(gpuModule);
    }
  }

  // Identify the specified GPU FuncOp instances.
  bool gpuFuncInstancesExist = false;
  SmallVector<gpu::GPUFuncOp, 1> gpuFuncTable;
  for (auto gpuModule : gpuModuleTable) {
    for (auto gpuFunc : gpuModule.getOps<gpu::GPUFuncOp>()) {
      for (auto kernelName : kernelNameTable) {
        if (gpuFunc.getName() == kernelName) {
          gpuFuncInstancesExist = true;
          gpuFuncTable.push_back(gpuFunc);

          // Set kernel attribute.
          gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                           b.getUnitAttr());
          break;
        }
      }
    }
  }

  if (!gpuFuncInstancesExist) {
    // Lambda to process the identified FuncOp.
    auto processGpuKernelFunc = [&b, &loc](FuncOp &theFunc) -> gpu::GPUFuncOp {
      // create a GPUFuncOp.
      FunctionType gpuFuncType = theFunc.getType();
      auto gpuFunc =
          b.create<gpu::GPUFuncOp>(loc, theFunc.getName(), gpuFuncType);

      // Set kernel attribute.
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
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

      // remove std.return ops.
      gpuFunc.walk([&](ReturnOp op) { op.erase(); });

      // create a GPU ReturnOp inside the GPUFuncOp.
      b.setInsertionPointToEnd(&gpuFuncBody.back());
      b.create<gpu::ReturnOp>(loc);

      return gpuFunc;
    };

    // Try locate a FuncOp which has kernelName, and convert it to a GPUFuncOp.
    // Logic to use the lambda.
    // Walkthrough all the FuncOp, check if it's within kernelNameTable, process
    // it if true.
    SmallVector<StringRef, 1> processedKernelNameTable;
    for (auto func : op.getOps<FuncOp>()) {
      for (unsigned i = 0; i < kernelNameTable.size(); ++i) {
        auto kernelName = kernelNameTable[i];
        if (func.getName() == kernelName) {
          auto gpuModule = gpuModuleTable[i];
          // Set up the symbol table for the GPU ModuleOp.
          SymbolTable gpuModuleSymbolTable(gpuModule);

          // Reset builder insertion point to the beginning of the GPU module,
          // as it would be modified inside the lambda.
          b.setInsertionPointToStart(&(gpuModule.body()).front());
          auto gpuFunc = processGpuKernelFunc(func);

          // insert the GPUFuncOp into GPUModuleOp.
          gpuModuleSymbolTable.insert(gpuFunc);

          processedKernelNameTable.push_back(kernelName);
          break;
        }
      }
    }

    // Remove all processed FuncOp instances.
    while (!processedKernelNameTable.empty()) {
      auto iter = processedKernelNameTable.begin();
      auto kernelName = *iter;
      bool funcRemoved = false;
      for (auto func : op.getOps<FuncOp>()) {
        if (func.getName() == kernelName) {
          funcRemoved = true;
          func.erase();
          break;
        }
      }
      if (funcRemoved)
        processedKernelNameTable.erase(iter);
    }
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

      gpuFunc.walk([&](miopen::DataConvertOp op) {
        auto loc = op.getLoc();
        OpBuilder b(op.getContext());
        b.setInsertionPoint(op);
        Value cast =
            b.create<gpu::BFConvertOp>(loc, op.out().getType(), op.in());
        op.replaceAllUsesWith(cast);
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

      // TBD see if these patterns could be re-written using tablgen.
      gpuFunc.walk([&](miopen::LDSBarrierOp op) {
        auto loc = op.getLoc();
        OpBuilder b(op.getContext());
        b.setInsertionPoint(op);
        b.create<gpu::LDSBarrierOp>(loc);
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
        gpuMfmaOp->setAttr("instr", op->getAttr("instr"));
        gpuMfmaOp->setAttr("imm", op->getAttr("imm"));

        op.replaceAllUsesWith(gpuMfmaOp.destD());
        op.erase();
      });

      gpuFunc.walk([&](miopen::Conv2DDummyOp op) { op.erase(); });
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

  applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::createLowerMIOpenOpsToGPUPass(StringRef kernelNameList,
                                    StringRef gpuModuleNameList) {
  return std::make_unique<LowerMIOpenOpsToGPUPass>(kernelNameList,
                                                   gpuModuleNameList);
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createLowerMIOpenOpsWithinGPUModulePass() {
  return std::make_unique<LowerMIOpenOpsWithinGPUModulePass>();
}
