//===- ROCDLToLLVMIRTranslation.cpp - Translate ROCDL to LLVM IR ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR ROCDL dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"

#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::createIntrinsicCall;

// Create a call to ROCm-Device-Library function
// Currently this routine will work only for calling ROCDL functions that
// take a single int32 argument. It is likely that the interface of this
// function will change to make it more generic.
static llvm::Value *createDeviceFunctionCall(llvm::IRBuilderBase &builder,
                                             StringRef fn_name, int parameter) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::FunctionType *function_type = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(module->getContext()), // return type.
      llvm::Type::getInt32Ty(module->getContext()), // parameter type.
      false);                                       // no variadic arguments.
  llvm::Function *fn = dyn_cast<llvm::Function>(
      module->getOrInsertFunction(fn_name, function_type).getCallee());
  llvm::Value *fn_op0 = llvm::ConstantInt::get(
      llvm::Type::getInt32Ty(module->getContext()), parameter);
  return builder.CreateCall(fn, ArrayRef<llvm::Value *>(fn_op0));
}

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the ROCDL dialect to LLVM IR.
class ROCDLDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "mlir/Dialect/LLVMIR/ROCDLConversions.inc"

    return failure();
  }

  /// Attaches module-level metadata for functions marked as kernels.
  LogicalResult
  amendOperation(Operation *op, NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    if (attribute.first == ROCDL::ROCDLDialect::getKernelFuncAttrName()) {
      auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!func)
        return failure();

      // For GPU kernels,
      // 1. Insert AMDGPU_KERNEL calling convention.
      // 2. Insert amdgpu-flat-workgroup-size(1, 1024) attribute.
      llvm::Function *llvmFunc =
          moduleTranslation.lookupFunction(func.getName());
      llvmFunc->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
      llvmFunc->addFnAttr("amdgpu-flat-work-group-size", "1, 256");
    }
    return success();
  }
};
} // end namespace

void mlir::registerROCDLDialectTranslation(DialectRegistry &registry) {
  registry.insert<ROCDL::ROCDLDialect>();
  registry.insert<gpu::GPUDialect>();
  registry.addDialectInterface<ROCDL::ROCDLDialect,
                               ROCDLDialectLLVMIRTranslationInterface>();
  registerLLVMDialectTranslation(registry);
}

void mlir::registerROCDLDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerROCDLDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}

void mlir::registerToROCDLIRTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-rocdlir",
      [](ModuleOp module, raw_ostream &output) {
        // Locate a GPU module within a Module. Use it if we find one.
        Operation *m = nullptr;
        auto *block = module.getBody();
        for (auto op = block->begin(); op != block->end(); ++op)
          if (auto gpuModule = dyn_cast<gpu::GPUModuleOp>(op)) {
            m = gpuModule;
            break;
          }

        llvm::LLVMContext llvmContext;
        auto llvmModule = translateModuleToLLVMIR(m, llvmContext);
        if (!llvmModule)
          return failure();

        StringRef amdgcnTriple = "amdgcn-amd-amdhsa";
        StringRef amdgcnDataLayout =
            "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-"
            "v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:"
            "1024-v2048:2048-n32:64-S32-A5-ni:7";
        llvmModule->setTargetTriple(amdgcnTriple);
        llvmModule->setDataLayout(amdgcnDataLayout);

        llvmModule->print(output, nullptr);
        return success();
      },
      [](DialectRegistry &registry) {
        mlir::registerROCDLDialectTranslation(registry);
      });
}
