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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKRESTOREHOSTCODEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-restore-host-code"

using namespace mlir;
using namespace mlir::rock;

namespace {

/// Information about a compiled kernel
struct KernelInfo {
  StringRef name;
  LLVM::LLVMFuncOp llvmFunc;
  int64_t gridSize = 1;
  int64_t blockSize = 256;
  SmallVector<Type> argTypes; // Original func argument types
};

struct RockRestoreHostCodePass
    : public rock::impl::RockRestoreHostCodePassBase<RockRestoreHostCodePass> {
  void runOnOperation() override;

private:
  /// Parse and restore host functions from the serialized attribute
  bool restoreHostFunctions(ModuleOp moduleOp);

  /// Collect kernel information from LLVM functions
  void collectKernelInfo(ModuleOp moduleOp,
                         SmallVectorImpl<KernelInfo> &kernels);

  /// Create gpu.binary from HSACO and convert calls to gpu.launch_func
  void createGpuBinaryAndLaunchFuncs(ModuleOp moduleOp,
                                     SmallVectorImpl<KernelInfo> &kernels);

  /// Remove kernel LLVM functions (they're now in the binary)
  void removeKernelFunctions(SmallVectorImpl<KernelInfo> &kernels);
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

void RockRestoreHostCodePass::collectKernelInfo(
    ModuleOp moduleOp, SmallVectorImpl<KernelInfo> &kernels) {
  moduleOp.walk([&](LLVM::LLVMFuncOp funcOp) {
    if (!funcOp->hasAttr("kernel"))
      return;

    KernelInfo info;
    info.name = funcOp.getName();
    info.llvmFunc = funcOp;

    // Get grid and block sizes from attributes
    if (auto gridAttr = funcOp->getAttrOfType<IntegerAttr>("grid_size"))
      info.gridSize = gridAttr.getInt();
    if (auto blockAttr = funcOp->getAttrOfType<IntegerAttr>("block_size"))
      info.blockSize = blockAttr.getInt();

    // Store the argument types from the LLVM function
    auto llvmFuncType = funcOp.getFunctionType();
    for (unsigned i = 0; i < llvmFuncType.getNumParams(); ++i) {
      info.argTypes.push_back(llvmFuncType.getParamType(i));
    }

    kernels.push_back(info);
  });
}

void RockRestoreHostCodePass::createGpuBinaryAndLaunchFuncs(
    ModuleOp moduleOp, SmallVectorImpl<KernelInfo> &kernels) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);
  Location loc = moduleOp.getLoc();

  // Get the HSACO binary from the triton.hsaco attribute
  auto hsacoAttr = moduleOp->getAttrOfType<StringAttr>("triton.hsaco");
  if (!hsacoAttr) {
    emitWarning(loc)
        << "No triton.hsaco attribute found, skipping GPU binary creation";
    return;
  }

  // Build a map from kernel names to their info
  DenseMap<StringRef, size_t> kernelMap;
  for (size_t i = 0; i < kernels.size(); ++i) {
    kernelMap[kernels[i].name] = i;
  }

  // Create kernel metadata for the gpu.binary
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
  auto rocdlTarget = ROCDL::ROCDLTargetAttr::get(
      ctx,
      /*optLevel=*/2,
      /*triple=*/"amdgcn-amd-amdhsa",
      /*chip=*/"gfx1100", // TODO: get from module attributes
      /*features=*/"",
      /*abiVersion=*/"400");

  // Create the object attribute with the HSACO
  // ObjectAttr::get(Attribute target, CompilationTarget format, StringAttr
  // object, ...)
  auto objectAttr = gpu::ObjectAttr::get(
      rocdlTarget,
      gpu::CompilationTarget::Binary, // format enum directly
      hsacoAttr,
      /*properties=*/nullptr, kernelTable);

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
      if (isa<MemRefType>(operand.getType())) {
        // Extract aligned pointer from memref and convert to LLVM pointer
        Value indexPtr = memref::ExtractAlignedPointerAsIndexOp::create(
            builder, callLoc, operand);
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

    // Create gpu.launch_func
    // Note: gpu.launch_func expects kernel operands to have proper types
    gpu::LaunchFuncOp::create(
        builder, callLoc,
        SymbolRefAttr::get(ctx, binaryOp.getName(),
                           {SymbolRefAttr::get(ctx, kernel.name)}),
        gpu::KernelDim3{gridX, one, one},  // grid dimensions
        gpu::KernelDim3{blockX, one, one}, // block dimensions
        /*dynamicSharedMemorySize=*/nullptr, launchArgs,
        /*asyncToken=*/nullptr,
        /*asyncDependencies=*/ValueRange{},
        /*clusterSize=*/std::nullopt);

    // Erase the old func.call
    callOp.erase();
  }
}

void RockRestoreHostCodePass::removeKernelFunctions(
    SmallVectorImpl<KernelInfo> &kernels) {
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
  collectKernelInfo(moduleOp, kernels);

  // If we have kernels, create gpu.binary and convert calls to gpu.launch_func
  if (!kernels.empty()) {
    createGpuBinaryAndLaunchFuncs(moduleOp, kernels);
    removeKernelFunctions(kernels);
  }
}
