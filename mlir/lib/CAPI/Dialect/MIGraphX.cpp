//===- MIGraphX.cpp - C Interface for MIGraphX dialect
//------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/MIGraphX.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
#include "mlir/Dialect/MIGraphX/Pipeline.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Pipelines.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/Support/TargetSelect.h"
#include <vector>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(MIGraphX, migraphx,
                                      mlir::migraphx::MIGraphXDialect)

// Returns the required buffer size if called with null buffer
// and fill information in the passed ptr when provided.
MLIR_CAPI_EXPORTED
void mlirGetKernelInfo(MlirModule module, int *size, void *data) {
  auto mod = unwrap(module);
  int argNum = 0;
  int argIdx = 0;
  llvm::StringRef kernelName;

  // Either of pointers should be provided.
  assert((size != nullptr || data != nullptr) &&
         "Either size or data pointer should be provided");
  std::vector<int> info;
  mod.walk([&](mlir::FuncOp f) {
    auto args = f.getArguments();
    for (auto arg : args) {
      argNum++;
      auto sType = arg.getType().template cast<mlir::ShapedType>();
      auto rank = sType.getRank();
      info.push_back(rank);
      for (int i = 0; i < rank; i++)
        info.push_back(sType.getDimSize(i));
      argIdx += rank;
    }
    kernelName = f.getName();
  });
  if (data == nullptr && size != nullptr) {
    *size = (1 + argNum + argIdx) * sizeof(int) + kernelName.size();
  } else if (data != nullptr) {
    int argSize = argNum + argIdx;
    int *argData = (int *)data;
    argData[0] = argNum;
    for (int i = 0; i < argSize; i++)
      argData[i + 1] = info[i];
    char *nameData = (char *)(argData + argSize + 1);
    for (size_t i = 0, e = kernelName.size(); i < e; ++i) {
      nameData[i] = kernelName[i];
    }
  }
}

// Returns block_size and grid_size as uint32_t[2]
MLIR_CAPI_EXPORTED void mlirGetKernelAttrs(MlirModule module, uint32_t *attrs) {
  auto mod = unwrap(module);
  mod.walk([&](mlir::LLVM::LLVMFuncOp llvmFunc) {
    attrs[0] =
        llvmFunc->getAttrOfType<mlir::IntegerAttr>("block_size").getInt();
    attrs[1] = llvmFunc->getAttrOfType<mlir::IntegerAttr>("grid_size").getInt();
  });
}

// Returns the size of compiled binary if called with null ptr
// and return the compiled binary when buffer is provided
MLIR_CAPI_EXPORTED bool mlirGetBinary(MlirModule module, int *size, char *bin) {
  bool success = false;
  auto mod = unwrap(module);
  if (bin == nullptr && size == nullptr)
    return success;
  mod.walk([&](mlir::gpu::GPUModuleOp gpuModule) {
    auto hsacoAttr = gpuModule->getAttrOfType<mlir::StringAttr>(
        mlir::gpu::getDefaultGpuBinaryAnnotation());
    if (hsacoAttr) {
      if (bin != nullptr) { // return binary regardless the presence of *size
        std::string hsaco = hsacoAttr.getValue().str();
        std::copy(hsaco.begin(), hsaco.end(), bin);
        success = true;
      } else {
        *size = hsacoAttr.getValue().size();
      }
    }
  });
  return success;
}

// pipelines

MLIR_CAPI_EXPORTED
void mlirMIGraphXAddHighLevelPipeline(MlirPassManager pm) {
  auto passMan = unwrap(pm);
  // FIXME : WA for the multithreading issue, potentially fixed in upstream.
  passMan->getContext()->disableMultithreading();
  passMan->setNesting(mlir::PassManager::Nesting::Implicit);
  mlir::migraphx::addHighLevelPipeline(*passMan);
  mlir::miopen::buildBufferizePipeline(*passMan);
}

MLIR_CAPI_EXPORTED void mlirMIGraphXAddBackendPipeline(MlirPassManager pm,
                                                       const char *chip,
                                                       const char *triple,
                                                       const char *features) {
  mlir::registerGpuSerializeToHsacoPass();
  auto passMan = unwrap(pm);
  passMan->setNesting(mlir::PassManager::Nesting::Implicit);
  mlir::miopen::KernelOptions kOpts;
  kOpts.tuningFallback = true;
  mlir::miopen::buildKernelPipeline(*passMan, kOpts);
  mlir::miopen::BackendOptions opts;
  opts.triple = triple;
  opts.chip = chip;
  opts.features = features;
  opts.optLevel = 3;
  opts.indexBitwidth = 64;
  mlir::miopen::buildBackendPipeline(*passMan, opts);
}
