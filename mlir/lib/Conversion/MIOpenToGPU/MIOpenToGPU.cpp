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
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
    Location loc = op->getLoc();

    if (type.getMemorySpaceAsInt() == gpu::GPUDialect::getWorkgroupAddressSpace()) {
      Value attribution = func.addWorkgroupAttribution(type, loc);
      op.replaceAllUsesWith(attribution);
    } else if (type.getMemorySpaceAsInt() ==
               gpu::GPUDialect::getPrivateAddressSpace()) {
      Value attribution = func.addPrivateAttribution(type, loc);
      op.replaceAllUsesWith(attribution);
    } else {
      // TBD: return failure.
      llvm::errs() << "unsupported addrspace!\n";
    }
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
    Value nop =
        b.create<Tgpu>(op.getLoc(), b.getIndexType(), gpu::Dimension::x);
    op.replaceAllUsesWith(nop);
    op.erase();
    return success();
  }
};

struct MIMFMARewritePattern : public OpRewritePattern<miopen::MFMAV2Op> {
  using OpRewritePattern<miopen::MFMAV2Op>::OpRewritePattern;

  // The below implements the concatenation of vector<4xi8> to an i32:
  // a[0] | (a[1] << 8) | (a[2] << 16) | (a[3] << 24)
  LogicalResult vector4i8Toi32(const Value &vec4i8, Value &i32vals,
                               PatternRewriter &b) const {
    if (!vec4i8.getType().isa<VectorType>())
      return failure();
    auto vecType = vec4i8.getType().cast<VectorType>();

    // Only supports 1d conversion
    if (vecType.getRank() != 1)
      return failure();

    // The dimension must be exactly 4
    auto dim = vecType.getShape().back();
    if (dim != 4)
      return failure();

    auto loc = vec4i8.getLoc();
    for (int i = 0; i < dim; ++i) {
      Value iterOp = b.create<arith::ConstantIndexOp>(loc, i);
      Value extracted = b.create<vector::ExtractElementOp>(
          loc, b.getIntegerType(8), vec4i8, iterOp);
      Value i32A =
          b.create<arith::ExtUIOp>(loc, b.getIntegerType(32), extracted);
      Value shiftWidth =
          b.create<arith::ConstantIntOp>(loc, i * 8, b.getIntegerType(32));
      Value i32AShifted = b.create<arith::ShLIOp>(loc, i32A, shiftWidth);
      i32vals = b.create<arith::OrIOp>(loc, i32vals, i32AShifted);
    }
    return success();
  }

  LogicalResult matchAndRewrite(miopen::MFMAV2Op op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();
    bool isVeci8 = false;
    if (op.sourceA().getType().isa<VectorType>()) {
      auto vectorType = op.sourceA().getType().cast<VectorType>();
      isVeci8 = vectorType.getElementType() == b.getIntegerType(8);
    }

    gpu::MFMAOp gpuMfmaOp;
    Value sourceA = op.sourceA();
    Value sourceB = op.sourceB();
    // Note: For i8 type, gpu.mfma op requires both arguments to be i32 instead
    // of vector<4xi8>, thus the explicit conversion below
    if (isVeci8) {
      sourceA = b.create<arith::ConstantIntOp>(loc, 0, b.getIntegerType(32));
      sourceB = b.create<arith::ConstantIntOp>(loc, 0, b.getIntegerType(32));
      LogicalResult res = vector4i8Toi32(op.sourceA(), sourceA, b);
      if (res.failed()) {
        return res;
      }

      res = vector4i8Toi32(op.sourceB(), sourceB, b);
      if (res.failed()) {
        return res;
      }
    }

    gpuMfmaOp = b.create<gpu::MFMAOp>(op.getLoc(), op.getType(), sourceA,
                                      sourceB, op.destC());
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

  auto processGpuKernelFunc = [&](gpu::GPUModuleOp &gpuMod,
                                  FuncOp &theFunc) -> gpu::GPUFuncOp {
    // Set up the symbol table for the GPU ModuleOp.
    SymbolTable gpuModuleSymbolTable(gpuMod);
    // Reset builder insertion point to the beginning of the GPU module,
    // as it would be modified inside the lambda.
    OpBuilder b(gpuMod.getContext());

    // create a GPUFuncOp.
    FunctionType gpuFuncType = theFunc.getFunctionType();
    auto gpuFunc =
        b.create<gpu::GPUFuncOp>(loc, theFunc.getName(), gpuFuncType);

    // insert the GPUFuncOp into GPUModuleOp.
    gpuModuleSymbolTable.insert(gpuFunc);

    // Set kernel attribute.
    int64_t gridSize = 0;
    int64_t blockSize = 0;
    gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    if (auto attr = theFunc->getAttr("block_size")) {
      gpuFunc->setAttr("block_size", attr);
      blockSize = attr.template cast<IntegerAttr>().getInt();
    }
    if (auto attr = theFunc->getAttr("grid_size")) {
      gpuFunc->setAttr("grid_size", attr);
      gridSize = attr.template cast<IntegerAttr>().getInt();
    }

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
    b.create<cf::BranchOp>(loc, clonedFuncEntry);

    // copy original_func attribute
    const char *attrName = "original_func";
    if (auto attr = theFunc->getAttrOfType<SymbolRefAttr>(attrName)) {
      gpuFunc->setAttr(attrName, attr);
    }

    // convert all calls to gpu.launch_func
    SmallVector<func::CallOp, 4> calls;
    op.walk([&](func::CallOp call) {
      if (auto callable = call.getCallableForCallee()) {
        if (FlatSymbolRefAttr symRef = callable.dyn_cast<SymbolRefAttr>()
                                           .dyn_cast<FlatSymbolRefAttr>()) {
          if (symRef.getValue() == theFunc.getName()) {
            OpBuilder b(call);
            auto gridVal = b.create<arith::ConstantIndexOp>(loc, gridSize);
            auto blockVal = b.create<arith::ConstantIndexOp>(loc, blockSize);
            auto cst1 = b.create<arith::ConstantIndexOp>(loc, 1);
            auto dynamicSharedMemSize =
                b.create<arith::ConstantIntOp>(loc, 0, b.getI32Type());
            gpu::KernelDim3 gridDims{gridVal, cst1, cst1};
            gpu::KernelDim3 blockDims{blockVal, cst1, cst1};
            b.create<gpu::LaunchFuncOp>(loc, gpuFunc, gridDims, blockDims,
                                        dynamicSharedMemSize,
                                        call.getCallOperands());
            calls.push_back(call);
          }
        }
      }
    });

    for (auto &call : calls) {
      call.erase();
    }

    return gpuFunc;
  };

  SmallVector<FuncOp, 1> processedFuncs;
  // Check parameters and populate default values if necessary.
  for (auto func : op.getOps<FuncOp>()) {
    if (func->hasAttr("kernel")) {
      std::string gfname = func.getName().str();
      gfname += "_module";
      auto gpuMod = makeGpuModule(gfname);
      processGpuKernelFunc(gpuMod, func);

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
    RewritePatternSet patterns(ctx);

    // miopen-lowering
    patterns.add<MIGPUAllocRewritePattern,
                 MIOpRewritePattern<miopen::WorkgroupBarrierOp, gpu::BarrierOp>,
                 MIOpRewritePattern<miopen::LDSBarrierOp, gpu::LDSBarrierOp>,
                 MIIdRewritePattern<miopen::WorkgroupIdOp, gpu::BlockIdOp>,
                 MIIdRewritePattern<miopen::WorkitemIdOp, gpu::ThreadIdOp>,
                 MIOpRewritePattern<func::ReturnOp, gpu::ReturnOp>,
                 MIMFMARewritePattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(gpuMod, std::move(patterns))))
      signalPassFailure();
  });

  if (gpuModCount == 0) {
    // Must have at least 1 gpu.module for rocm-runner
    makeGpuModule("miopen_gpu_module");
  }
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::createLowerMIOpenOpsToGPUPass() {
  return std::make_unique<LowerMIOpenOpsToGPUPass>();
}
