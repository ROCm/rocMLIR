//===- RestoreHostCode.cpp - Restore host functions after Triton compilation
//-------------------------===//
//
// Copyright 2026 The MLIR Authors.
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
// This pass restores host functions that were stored during
// RockMemrefToTensorPass and converts them to use gpu.launch_func with a
// gpu.binary containing the HSACO.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/compileUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKRESTOREHOSTCODEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-restore-host-code"

using namespace mlir;
using namespace mlir::rock;

static FailureOr<std::pair<gpu::ObjectAttr, DenseMap<StringRef, size_t>>>
createGpuBinary(OpBuilder builder, ModuleOp moduleOp,
                RockRestoreHostCodePassOptions &options,
                SmallVectorImpl<KernelInfo> &kernels) {
  // Get the HSACO binary from the triton.hsaco attribute
  auto hsacoAttr = moduleOp->getAttrOfType<StringAttr>("triton.hsaco");
  if (!hsacoAttr) {
    return failure();
  }

  // Build a map from kernel names to their info
  DenseMap<StringRef, size_t> kernelMap;
  for (size_t i = 0; i < kernels.size(); ++i) {
    kernelMap[kernels[i].name] = i;
  }

  // Create kernel metadata for the gpu.binary
  MLIRContext *ctx = builder.getContext();
  SmallVector<gpu::KernelMetadataAttr> kernelMetadata;
  auto ptrType = LLVM::LLVMPointerType::get(ctx);

  for (const KernelInfo &kernel : kernels) {
    // Create a function type with 5 pointer arguments (matching HSACO metadata)
    // GEMM kernels typically expect: A, B, C, workspace1, workspace2
    SmallVector<Type> argTypes(5, ptrType);
    auto kernelFuncType = FunctionType::get(ctx, argTypes, {});

    // Create metadata for this kernel
    // KernelMetadataAttr::get(StringAttr name, Type functionType, ...)
    auto metadata =
        gpu::KernelMetadataAttr::get(builder.getStringAttr(kernel.name),
                                     /*functionType=*/kernelFuncType,
                                     /*argAttrs=*/nullptr,
                                     /*metadata=*/nullptr);
    kernelMetadata.push_back(metadata);
  }

  // Create the kernel table
  auto kernelTable = gpu::KernelTableAttr::get(ctx, kernelMetadata);

  // Create the ROCDL target attribute
  // ROCDLTargetAttr::get(ctx, optLevel, triple, chip, features, abiVersion,
  // ...)
  auto rocdlTarget = ROCDL::ROCDLTargetAttr::get(ctx,
                                                 /*optLevel=*/options.optLevel,
                                                 /*triple=*/options.triple,
                                                 /*chip=*/options.arch,
                                                 /*features=*/options.features,
                                                 /*abiVersion=*/"500");

  // Create the object attribute with the HSACO
  // ObjectAttr::get(Attribute target, CompilationTarget format, StringAttr
  // object, ...)
  auto objectAttr = gpu::ObjectAttr::get(
      rocdlTarget,
      gpu::CompilationTarget::Binary, // format enum directly
      hsacoAttr,
      /*properties=*/nullptr, kernelTable);
  return std::make_pair(objectAttr, kernelMap);
}

namespace {

struct RockRestoreHostCodePass
    : public rock::impl::RockRestoreHostCodePassBase<RockRestoreHostCodePass> {
  using RockRestoreHostCodePassBase::RockRestoreHostCodePassBase;

  void runOnOperation() override;

private:
  /// Parse and restore host functions from the serialized attribute
  bool restoreHostFunctions(ModuleOp moduleOp);

  /// Collect kernel information from LLVM functions
  LogicalResult collectKernelInfo(ModuleOp moduleOp,
                                  SmallVector<KernelInfo> &kernels);

  /// Create gpu.binary from HSACO and convert calls to gpu.launch_func
  LogicalResult
  createGpuBinaryAndLaunchFuncs(ModuleOp moduleOp,
                                RockRestoreHostCodePassOptions &options,
                                SmallVector<KernelInfo> &kernels);

  /// Remove kernel LLVM functions (they're now in the binary)
  void removeKernelFunctions(SmallVector<KernelInfo> &kernels);
};

} // end anonymous namespace

/// Parse and restore host functions from the serialized attribute
bool RockRestoreHostCodePass::restoreHostFunctions(ModuleOp moduleOp) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  auto hostFuncsAttr =
      moduleOp->getAttrOfType<ArrayAttr>("rock.host_functions");
  if (!hostFuncsAttr || hostFuncsAttr.empty())
    return false;

  builder.setInsertionPointToEnd(moduleOp.getBody());

  for (Attribute attr : hostFuncsAttr) {
    auto strAttr = dyn_cast<StringAttr>(attr);
    if (!strAttr)
      continue;

    // Parse the function from the stored string
    // Wrap it in a module for parsing
    std::string moduleStr = "module {\n" + strAttr.getValue().str() + "\n}";

    // Use parseSourceString with verification disabled for symbols
    ParserConfig config(ctx, /*verifyAfterParse=*/false);
    auto parsedModule = parseSourceString<ModuleOp>(moduleStr, config);
    if (!parsedModule) {
      emitWarning(moduleOp.getLoc())
          << "Failed to parse stored host function, skipping";
      continue;
    }

    // Move each operation from the parsed module to our module
    for (Operation &op :
         llvm::make_early_inc_range(parsedModule->getBody()->getOperations())) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;
      op.moveBefore(&moduleOp.getBody()->back());
    }
  }

  // Remove the attribute
  moduleOp->removeAttr("rock.host_functions");
  return true;
}

LogicalResult
RockRestoreHostCodePass::collectKernelInfo(ModuleOp moduleOp,
                                           SmallVector<KernelInfo> &kernels) {
  // Get Triton metadata from module attributes for block size
  // The HSACO is compiled with these settings, so we must use them for launch
  int64_t numWarps = -1;
  int64_t warpSize = -1;
  int64_t sharedMemory = 0;

  if (auto numWarpsAttr =
          moduleOp->getAttrOfType<IntegerAttr>("ttg.num-warps"))
    numWarps = numWarpsAttr.getInt();
  if (auto warpSizeAttr =
          moduleOp->getAttrOfType<IntegerAttr>("ttg.threads-per-warp"))
    warpSize = warpSizeAttr.getInt();
  if (auto sharedAttr = moduleOp->getAttrOfType<IntegerAttr>("ttg.shared"))
    sharedMemory = sharedAttr.getInt();

  if (numWarps == -1) {
    LLVM_DEBUG(llvm::dbgs() << "ttg.num-warps not found\n");
    return failure();
  }
  if (warpSize == -1) {
    LLVM_DEBUG(llvm::dbgs() << "ttg.threads-per-warp not found\n");
    return failure();
  }

  int64_t tritonBlockSize = numWarps * warpSize;
  moduleOp.walk([&](LLVM::LLVMFuncOp funcOp) {
    if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
      return;

    KernelInfo info;
    info.name = funcOp.getName();
    info.llvmFunc = funcOp;
    info.blockSize = tritonBlockSize; // Use Triton's block size (matches HSACO)
    info.sharedMemorySize = sharedMemory;

    // Get the saved grid_size from module attribute (set by MemrefToTensor)
    // This is the problem-specific value from the original rocMLIR kernel
    std::string gridAttrName = "rock.grid_size." + info.name;
    if (auto gridAttr = moduleOp->getAttrOfType<IntegerAttr>(gridAttrName))
      info.gridSize = gridAttr.getInt();

    // Store the argument types from the LLVM function
    auto llvmFuncType = funcOp.getFunctionType();
    for (unsigned i = 0; i < llvmFuncType.getNumParams(); ++i) {
      info.argTypes.push_back(llvmFuncType.getParamType(i));
    }

    kernels.push_back(info);
  });
  return success();
}

LogicalResult RockRestoreHostCodePass::createGpuBinaryAndLaunchFuncs(
    ModuleOp moduleOp, RockRestoreHostCodePassOptions &options,
    SmallVector<KernelInfo> &kernels) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);
  Location loc = moduleOp.getLoc();

  FailureOr<std::pair<gpu::ObjectAttr, DenseMap<StringRef, size_t>>>
      maybeBinary = createGpuBinary(builder, moduleOp, options, kernels);
  if (failed(maybeBinary)) {
    LLVM_DEBUG(llvm::dbgs() << "Could not find binary\n");
    return failure();
  }
  gpu::ObjectAttr objectAttr = maybeBinary.value().first;
  DenseMap<StringRef, size_t> kernelMap = maybeBinary.value().second;

  // Create the gpu.binary operation at module level
  // BinaryOp::create(builder, loc, name, offloadingHandler, objects)
  builder.setInsertionPointToStart(moduleOp.getBody());
  auto binaryOp = gpu::BinaryOp::create(builder, loc, "rock_kernels",
                                        /*offloadingHandler=*/nullptr,
                                        builder.getArrayAttr({objectAttr}));

  // Collect all func.call ops that call kernels
  SmallVector<func::CallOp> callsToConvert;
  moduleOp.walk([&](func::CallOp callOp) {
    if (kernelMap.count(callOp.getCallee())) {
      callsToConvert.push_back(callOp);
    }
  });

  // Convert each call to gpu.launch_func
  for (func::CallOp callOp : callsToConvert) {
    auto it = kernelMap.find(callOp.getCallee());
    if (it == kernelMap.end())
      continue;
    KernelInfo &kernel = kernels[it->second];

    builder.setInsertionPoint(callOp);
    Location callLoc = callOp.getLoc();

    // Create grid and block dimensions
    Value one = arith::ConstantIndexOp::create(builder, callLoc, 1);
    Value gridX =
        arith::ConstantIndexOp::create(builder, callLoc, kernel.gridSize);
    Value blockX =
        arith::ConstantIndexOp::create(builder, callLoc, kernel.blockSize);

    // Convert memref arguments to LLVM pointers for the kernel
    SmallVector<Value> launchArgs;
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    for (Value operand : callOp.getOperands()) {
      Value memrefVal = operand;

      // If it's a tensor, first convert to memref
      if (auto tensorType = dyn_cast<TensorType>(operand.getType())) {
        auto memrefType =
            MemRefType::get(tensorType.getShape(), tensorType.getElementType());
        memrefVal =
            bufferization::ToBufferOp::create(builder, callLoc, memrefType, operand);
      }

      if (isa<MemRefType>(memrefVal.getType())) {
        // Extract aligned pointer from memref and convert to LLVM pointer
        Value indexPtr = memref::ExtractAlignedPointerAsIndexOp::create(
            builder, callLoc, memrefVal);
        // Convert index to i64 then to pointer
        Value i64Val = arith::IndexCastOp::create(
            builder, callLoc, builder.getI64Type(), indexPtr);
        Value llvmPtr =
            LLVM::IntToPtrOp::create(builder, callLoc, ptrType, i64Val);
        launchArgs.push_back(llvmPtr);
      } else {
        launchArgs.push_back(operand);
      }
    }

    // Triton kernels may have additional workspace arguments (typically 2 more)
    // Add null pointers for these
    // The HSACO typically expects 5 pointer arguments for GEMM: A, B, C,
    // workspace1, workspace2
    while (launchArgs.size() < 5) {
      Value nullPtr = LLVM::ZeroOp::create(builder, callLoc, ptrType);
      launchArgs.push_back(nullPtr);
    }

    // Create dynamic shared memory size if needed
    Value dynSharedMem = nullptr;
    if (kernel.sharedMemorySize > 0) {
      dynSharedMem = arith::ConstantOp::create(
          builder, callLoc, builder.getI32Type(),
          builder.getI32IntegerAttr(kernel.sharedMemorySize));
    }

    // Create gpu.launch_func
    // Note: gpu.launch_func expects kernel operands to have proper types
    gpu::LaunchFuncOp::create(
        builder, callLoc,
        SymbolRefAttr::get(ctx, binaryOp.getName(),
                           {SymbolRefAttr::get(ctx, kernel.name)}),
        gpu::KernelDim3{gridX, one, one},  // grid dimensions
        gpu::KernelDim3{blockX, one, one}, // block dimensions
        dynSharedMem, launchArgs,
        /*asyncTokenType=*/nullptr,
        /*asyncDependencies=*/ValueRange{},
        /*clusterSize=*/std::nullopt);

    // gpu.launch_func doesn't return values - it modifies buffers in-place.
    // Replace uses of the func.call result with the output operand.
    // For GEMM/Conv, the output (C matrix) is the last tensor argument.
    if (callOp.getNumResults() > 0) {
      // Find the last tensor operand - this is the output that was modified
      Value outputOperand;
      for (Value operand : llvm::reverse(callOp.getOperands())) {
        if (isa<TensorType, MemRefType>(operand.getType())) {
          outputOperand = operand;
          break;
        }
      }
      if (outputOperand) {
        // Replace all uses of the call result with the output operand
        callOp.getResult(0).replaceAllUsesWith(outputOperand);
      }
    }

    // Erase the old func.call
    callOp.erase();
  }
  return success();
}

void RockRestoreHostCodePass::removeKernelFunctions(
    SmallVector<KernelInfo> &kernels) {
  // Remove the LLVM kernel functions since they're now in the binary
  for (KernelInfo &kernel : kernels) {
    if (kernel.llvmFunc)
      kernel.llvmFunc.erase();
  }
}

void RockRestoreHostCodePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  // Build options from pass parameters
  RockRestoreHostCodePassOptions options;
  options.triple = triple.getValue();
  options.arch = arch.getValue();
  options.features = features.getValue();
  options.optLevel = optLevel.getValue();

  // Mark the module as containing GPU code
  moduleOp->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                    builder.getUnitAttr());

  // Restore host functions from the serialized attribute
  if (!restoreHostFunctions(moduleOp)) {
    // No host functions to restore
    return;
  }

  // Collect kernel information from LLVM functions
  SmallVector<KernelInfo> kernels;
  if (failed(collectKernelInfo(moduleOp, kernels)))
    signalPassFailure();

  // If we have kernels, create gpu.binary and convert calls to gpu.launch_func
  if (!kernels.empty()) {
    if (failed(createGpuBinaryAndLaunchFuncs(moduleOp, options, kernels)))
      signalPassFailure();
    removeKernelFunctions(kernels);
  }
}
