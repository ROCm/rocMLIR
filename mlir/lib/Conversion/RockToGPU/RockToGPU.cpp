//===- RockToGPU.cpp - MLIR Rock ops lowering passes ---------------===//
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
// This pass converts rock ops to std dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/RockToGPU/RockToGPU.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTROCKTOGPUPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct LowerRockOpsToGPUPass
    : public impl::ConvertRockToGPUPassBase<LowerRockOpsToGPUPass> {
public:
  using impl::ConvertRockToGPUPassBase<
      LowerRockOpsToGPUPass>::ConvertRockToGPUPassBase;
  void runOnOperation() override;
};
} // end anonymous namespace

namespace {

//===----------------------------------------------------------------------===//
// Rock Operation pattern lowering.
//===----------------------------------------------------------------------===//

struct MIGPUAllocRewritePattern : public OpRewritePattern<rock::GpuAllocOp> {
  using OpRewritePattern<rock::GpuAllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::GpuAllocOp op,
                                PatternRewriter &b) const override {
    constexpr int64_t widestLoadOpBitwidth = 512;

    auto type = op.getOutput().getType();
    auto func = op->getParentOfType<gpu::GPUFuncOp>();
    Location loc = op->getLoc();

    auto memSpaceValue = type.getMemorySpace()
                             .dyn_cast_or_null<gpu::AddressSpaceAttr>()
                             .getValue();
    if (memSpaceValue == gpu::GPUDialect::getWorkgroupAddressSpace()) {
      BlockArgument attribution = func.addWorkgroupAttribution(type, loc);
      func.setWorkgroupAttributionAttr(
          attribution.getArgNumber() - func.getFirstWorkgroupAttributionIndex(),
          LLVM::LLVMDialect::getAlignAttrName(),
          b.getI64IntegerAttr(widestLoadOpBitwidth / 8));
      b.replaceOp(op, attribution);
    } else if (memSpaceValue == gpu::GPUDialect::getPrivateAddressSpace()) {
      Value attribution = func.addPrivateAttribution(type, loc);
      b.replaceOp(op, attribution);
    } else {
      return b.notifyMatchFailure(loc, "unsupported addrspace!\n");
    }
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
    b.replaceOpWithNewOp<Tgpu>(op, b.getIndexType(), gpu::Dimension::x);
    return success();
  }
};
} // namespace

void LowerRockOpsToGPUPass::runOnOperation() {
  ModuleOp op = getOperation();
  MLIRContext *ctx = op.getContext();
  OpBuilder b(ctx);
  Location loc = op.getLoc();

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
                                  func::FuncOp &theFunc) -> LogicalResult {
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
    int32_t gridSize = 0;
    int32_t blockSize = 0;
    gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    if (auto attr = theFunc->getAttr("block_size")) {
      gpuFunc->setAttr("block_size", attr);
      blockSize = attr.template cast<IntegerAttr>().getInt();
      gpuFunc->setAttr(gpu::GPUFuncOp::getKnownBlockSizeAttrName(),
                       b.getDenseI32ArrayAttr({blockSize, 1, 1}));
    }
    if (auto attr = theFunc->getAttr("grid_size")) {
      gpuFunc->setAttr("grid_size", attr);
      gridSize = attr.template cast<IntegerAttr>().getInt();
      gpuFunc->setAttr(gpu::GPUFuncOp::getKnownGridSizeAttrName(),
                       b.getDenseI32ArrayAttr({gridSize, 1, 1}));
    }

    if (auto attr = theFunc->getAttr("wave_size")) {
      int32_t waveSize = attr.template cast<IntegerAttr>().getInt();
      if (blockSize / waveSize >= 2) {
        gpuFunc->setAttr("rocdl.waves_per_eu", b.getI32IntegerAttr(2));
      }
    }

    // associate arguments for newly created GPUFuncOp.
    IRMapping map;
    for (auto pair : llvm::zip(theFunc.getArguments(), gpuFunc.getArguments()))
      map.map(std::get<0>(pair), std::get<1>(pair));

    // clone function body into newly created GPUFuncOp.
    Region &gpuFuncBody = gpuFunc.getBody();
    Region &funcBody = theFunc.getBody();
    funcBody.cloneInto(&gpuFuncBody, map);

    // add a branch op to the cloned region.
    Block &funcEntry = funcBody.front();
    Block *clonedFuncEntry = map.lookup(&funcEntry);
    Block &gpuFuncEntry = gpuFuncBody.front();
    b.setInsertionPointToEnd(&gpuFuncEntry);
    b.create<cf::BranchOp>(loc, clonedFuncEntry);

    // Clone in global constants
    llvm::SmallDenseMap<SymbolRefAttr, FlatSymbolRefAttr> clonedConsts;
    WalkResult result = funcBody.walk([&](memref::GetGlobalOp op)
                                          -> WalkResult {
      SymbolRefAttr globalSym = op.getNameAttr();
      auto toClone = dyn_cast_or_null<memref::GlobalOp>(
          SymbolTable::lookupNearestSymbolFrom(op, globalSym));
      if (!toClone)
        return WalkResult::interrupt();
      if (toClone->getParentOfType<gpu::GPUModuleOp>() == gpuMod)
        // Already cloned, continue
        return WalkResult::advance();
      auto maybeMapped = clonedConsts.find(globalSym);
      if (maybeMapped == clonedConsts.end()) {
        OpBuilder::InsertionGuard guard(b);
        Operation *cloned = toClone.clone();
        // There probably shouldn't be any renames, but let's be careful.
        StringAttr newNameAttr = gpuModuleSymbolTable.insert(cloned);
        clonedConsts.insert({globalSym, FlatSymbolRefAttr::get(newNameAttr)});
      }
      op.setNameAttr(clonedConsts.find(globalSym)->second);
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return theFunc.emitOpError("failed to clone referenced global constants");
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
                                        call.getArgOperands());
            calls.push_back(call);
          }
        }
      }
    });

    for (auto &call : calls) {
      call.erase();
    }

    return success();
  };

  SmallVector<func::FuncOp, 1> processedFuncs;
  // Check parameters and populate default values if necessary.
  for (auto func : op.getOps<func::FuncOp>()) {
    if (func->hasAttr("kernel")) {
      std::string gfname = func.getName().str();
      gfname += "_module";
      auto gpuMod = makeGpuModule(gfname);
      if (failed(processGpuKernelFunc(gpuMod, func)))
        signalPassFailure();

      processedFuncs.push_back(func);
    }
  }

  // Remove all processed FuncOp instances.
  for (auto func : processedFuncs) {
    func.erase();
  }

  // Convert Rock ops to GPU Ops
  int gpuModCount = 0;
  op.walk([this, &gpuModCount](gpu::GPUModuleOp gpuMod) {
    gpuModCount++;
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    // rock-lowering
    patterns.add<MIGPUAllocRewritePattern,
                 MIOpRewritePattern<rock::WorkgroupBarrierOp, gpu::BarrierOp>,
                 MIOpRewritePattern<rock::LDSBarrierOp, amdgpu::LDSBarrierOp>,
                 MIIdRewritePattern<rock::WorkgroupIdOp, gpu::BlockIdOp>,
                 MIIdRewritePattern<rock::WorkitemIdOp, gpu::ThreadIdOp>,
                 MIOpRewritePattern<func::ReturnOp, gpu::ReturnOp>>(ctx);

    if (failed(applyPatternsAndFoldGreedily(gpuMod, std::move(patterns))))
      signalPassFailure();
  });

  if (gpuModCount == 0) {
    // Must have at least 1 gpu.module for rocm-runner
    makeGpuModule("rock_gpu_module");
  }
}
