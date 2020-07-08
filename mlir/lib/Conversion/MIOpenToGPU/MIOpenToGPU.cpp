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

      gpuFunc.walk([&](miopen::MFMAOp op) {
        auto loc = op.getLoc();
        OpBuilder b(op.getContext());
        b.setInsertionPoint(op);

        // Obtain the data type of matrix A/B, shape of matrix C, and attributes.
        auto dataType = op.sourceA().getType();
        auto memRefType = op.destC().getType().cast<MemRefType>();
        auto shape = memRefType.getShape();

        int64_t MPerWave = 64;
        int64_t NPerWave = 64;
        if (op.getAttr("m_per_wave"))
          MPerWave = op.getAttr("m_per_wave").cast<IntegerAttr>().getInt();
        if (op.getAttr("n_per_wave"))
          NPerWave = op.getAttr("n_per_wave").cast<IntegerAttr>().getInt();

        // From the data type and attributes, determine:
        // - Which MFMA instruction be used.
        // - How many MFMA instructions be used.
        // - How the input matrix C would be dissected.
        // - How are immediates of MFMA instructions be specified.
        StringRef mfmaInstr = "mfma_f32_32x32x1f32";
        unsigned vectorLength = 32;
        unsigned mfmaInstrLength = 1;
        SmallVector<SmallVector<unsigned, 3>, 2> imms;

        if (dataType == b.getF32Type()) {
          if (MPerWave == 64 && NPerWave == 64) {
            // Original C++ logic:
            // __device__ void gcnasm_mfma_f32_32x32x1f32<64, 64>(const float& reg_a, const float& reg_b, float32_t* reg_c)
            //  reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[0], 1, 0, 0);
            //  reg_c[1] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[1], 1, 1, 0);

            // Issue 2 mfma_f32_32x32x1f32.
            // Matrix C will be dissected into 2 vector<32xf32>
            mfmaInstr = "mfma_f32_32x32x1f32";
            vectorLength = 32;
            mfmaInstrLength = 2;
            imms.push_back({ 1, 0, 0 });
            imms.push_back({ 1, 1, 0 });
          } else if (MPerWave == 32 && NPerWave == 64) {
            // Original C++ logic:
            // __device__ void gcnasm_mfma_f32_32x32x1f32<32, 64>(const float& reg_a, const float& reg_b, float32_t* reg_c)
            //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[0], 1, 0, 0);
            mfmaInstr = "mfma_f32_32x32x1f32";
            vectorLength = 32;
            mfmaInstrLength = 1;
            imms.push_back({ 1, 0, 0 });
          } else if (MPerWave == 64 && NPerWave == 32) {
            // Original C++ logic:
            // __device__ void gcnasm_mfma_f32_32x32x1f32<64, 32>(const float& reg_a, const float& reg_b, float32_t* reg_c)
            //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[0], 0, 0, 1);
            mfmaInstr = "mfma_f32_32x32x1f32";
            vectorLength = 32;
            mfmaInstrLength = 1;
            imms.push_back({ 0, 0, 1 });
          } else if (MPerWave == 32 && NPerWave == 32) {
            // Original C++ logic:
            // __device__ void gcnasm_mfma_f32_32x32x2f32(const float& reg_a, const float& reg_b, float16_t* reg_c)
            //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2f32(reg_a, reg_b, reg_c[0], 0, 0, 0);
            mfmaInstr = "mfma_f32_32x32x2f32";
            vectorLength = 16;
            mfmaInstrLength = 1;
            imms.push_back({ 0, 0, 0 });
          } else if (MPerWave == 16 && NPerWave == 16) {
            // Original C++ logic:
            // __device__ void gcnasm_mfma_f32_16x16x4f32(const float& reg_a, const float& reg_b, float4_t* reg_c)
            //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x4f32(reg_a, reg_b, reg_c[0], 0, 0, 0);
            mfmaInstr = "mfma_f32_16x16x4f32";
            vectorLength = 4;
            mfmaInstrLength = 1;
            imms.push_back({ 0, 0, 0 });
          } else if (MPerWave == 16 && NPerWave == 64) {
            // Original C++ logic:
            // __device__ void gcnasm_mfma_f32_16x16x1f32<16, 64>(const float& reg_a, const float& reg_b, float16_t* reg_c)
            //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x1f32(reg_a, reg_b, reg_c[0], 2, 0, 0);
            mfmaInstr = "mfma_f32_16x16x1f32";
            vectorLength = 16;
            mfmaInstrLength = 1;
            imms.push_back({ 2, 0, 0 });
          } else if (MPerWave == 64 && NPerWave == 16) {
            // Original C++ logic:
            // __device__ void gcnasm_mfma_f32_16x16x1f32<64, 16>(const float& reg_a, const float& reg_b, float16_t* reg_c)
            //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x1f32(reg_a, reg_b, reg_c[0], 0, 0, 4);
            mfmaInstr = "mfma_f32_16x16x1f32";
            vectorLength = 16;
            mfmaInstrLength = 1;
            imms.push_back({ 0, 0, 4 });
          } else if (MPerWave == 8 && NPerWave == 64) {
            // Original C++ logic:
            // __device__ void gcnasm_mfma_f32_4x4x1f32<4, 64>(const float& reg_a, const float& reg_b, float4_t* reg_c)
            //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(reg_a, reg_b, reg_c[0], 4, 0, 0);
            mfmaInstr = "mfma_f32_4x4x1f32";
            vectorLength = 4;
            mfmaInstrLength = 1;
            imms.push_back({ 4, 0, 0 });
          } else if (MPerWave == 4 && NPerWave == 64) {
            // Original C++ logic:
            // __device__ void gcnasm_mfma_f32_4x4x1f32<8, 64>(const float& reg_a, const float& reg_b, float4_t* reg_c)
            //     reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(reg_a, reg_b, reg_c[0], 4, 0, 0);
            //     reg_c[1] = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(reg_a, reg_b, reg_c[1], 4, 1, 0);
            mfmaInstr = "mfma_f32_4x4x1f32";
            vectorLength = 4;
            mfmaInstrLength = 2;
            imms.push_back({ 4, 0, 0 });
            imms.push_back({ 4, 1, 0 });
          } else {
            // Unhandled cases.
          }
        }

        auto vfloatType = VectorType::get({vectorLength}, dataType);
        auto resultMemRefType = MemRefType::get({shape[0] / vectorLength}, vfloatType);
        auto vectorTypeCast = b.create<vector::TypeCastOp>(loc, resultMemRefType, op.destC());

        for (unsigned iter = 0; iter < mfmaInstrLength; ++iter) {
          // Emit iteration constant.
          auto iterConstant = b.create<ConstantIndexOp>(loc, iter);
          // Emit load.
          auto vectorLoad = b.create<LoadOp>(loc, vfloatType, vectorTypeCast, ValueRange{iterConstant});
          // Emit gpu.mfma operation.
          auto gpuMfmaOp = b.create<gpu::MFMAOp>(loc, vfloatType, op.sourceA(), op.sourceB(), vectorLoad);
          gpuMfmaOp.setAttr("instr", b.getStringAttr(mfmaInstr));
          gpuMfmaOp.setAttr("imm", b.getArrayAttr({
                                                   b.getI32IntegerAttr(imms[iter][0]),
                                                   b.getI32IntegerAttr(imms[iter][1]),
                                                   b.getI32IntegerAttr(imms[iter][2]),
                                                  }));
          // Emit store.
          auto vectorStore = b.create<StoreOp>(loc, gpuMfmaOp, vectorTypeCast, ValueRange{iterConstant});
        }

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
