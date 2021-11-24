//===- MIGraphX.cpp - C Interface for MIGraphX dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/MIGraphX.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
#include "mlir/Dialect/MIGraphX/Pipeline.h"
#include "mlir/Dialect/MIOpen/Pipeline.h"
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/Support/TargetSelect.h"
#include <vector>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(MIGraphX, migraphx,
                                      mlir::migraphx::MIGraphXDialect)

// Returns the number of operands in the FuncOp and fill information in the passed ptr.
MLIR_CAPI_EXPORTED
void mlirGetKernelInfo(MlirModule module, void *data){
  auto mod = unwrap(module);
  std::vector<int> info;
  int argNum = 0;
  int argIdx = 0;
  llvm::StringRef kernelName;
  mod.walk([&](mlir::FuncOp f) {
    auto args = f.getArguments();
    for(auto arg: args){
      argNum++;
      auto sType = arg.getType().template cast<mlir::ShapedType>();
      auto rank = sType.getRank();
      info.push_back(rank);
      for (int i=0; i<rank; i++)
        info.push_back(sType.getDimSize(i));
      argIdx += rank;
    }
    kernelName = f.getName();
  });
  int size = argNum + argIdx;
  int* argData = (int*)data;
  argData[0] = argNum;
  for (int i=0; i<size; i++)
    argData[i+1] = info[i];
  char* nameData = (char*)(argData + size + 1);
  for (int i=0; i<kernelName.size();i++){
    nameData[i] = kernelName[i];
  }
}

MLIR_CAPI_EXPORTED
int mlirGetKernelInfoSize(MlirModule module){
  auto mod = unwrap(module);
  int argNum = 0;
  int argIdx = 0;
  llvm::StringRef kernelName;
  mod.walk([&](mlir::FuncOp f) {
    auto args = f.getArguments();
    for(auto arg: args){
      argNum++;
      auto sType = arg.getType().template cast<mlir::ShapedType>();
      auto rank = sType.getRank();
      argIdx += rank;
      kernelName = f.getName();
    }
  });
  int size = (1 + argNum + argIdx)*sizeof(int) + kernelName.size();
  return size;
}


// Returns block_size and grid_size as int[2]
MLIR_CAPI_EXPORTED void
mlirGetKernelAttrs(MlirModule module, int *attrs) {
  auto mod = unwrap(module);
  mod.walk([&](mlir::LLVM::LLVMFuncOp llvmFunc) {
    attrs[0] = llvmFunc->getAttrOfType<mlir::IntegerAttr>("block_size").getInt();
    attrs[1] = llvmFunc->getAttrOfType<mlir::IntegerAttr>("grid_size").getInt();
  });
}

// Returns the size of compiled binary
MLIR_CAPI_EXPORTED int
mlirGetBinarySize(MlirModule module){
  auto mod = unwrap(module);
  size_t size;
  mod.walk([&](mlir::gpu::GPUModuleOp gpuModule) {
    auto hsacoAttr = gpuModule->getAttrOfType<mlir::StringAttr>(mlir::gpu::getDefaultGpuBinaryAnnotation());
    if (hsacoAttr) {
      size = hsacoAttr.getValue().size();
    }
  });
  return size;
}

// Returns the compiled binary
MLIR_CAPI_EXPORTED bool
mlirGetBinary(MlirModule module, char *bin) {
  bool success = false;
  auto mod = unwrap(module);
  mod.walk([&](mlir::gpu::GPUModuleOp gpuModule) {
    auto hsacoAttr = gpuModule->getAttrOfType<mlir::StringAttr>(mlir::gpu::getDefaultGpuBinaryAnnotation());
    if (hsacoAttr) {
      std::string hsaco = hsacoAttr.getValue().str();
      std::copy(hsaco.begin(), hsaco.end(), bin);
      success = true;
    }
  });
  return success;
}

// pipelines

MLIR_CAPI_EXPORTED
void mlirMIGraphXAddHighLevelPipeline(MlirPassManager pm) {
  auto passMan = unwrap(pm);
  passMan->setNesting(mlir::PassManager::Nesting::Implicit);
  mlir::migraphx::addHighLevelPipeline(*passMan);
  mlir::miopen::addHighLevelPipeline(*passMan);
}

MLIR_CAPI_EXPORTED void
mlirMIGraphXAddBackendPipeline(MlirPassManager pm, const char* chip) {
  mlir::registerGpuSerializeToHsacoPass();

  auto passMan = unwrap(pm);
  passMan->setNesting(mlir::PassManager::Nesting::Implicit);

  const char *triple = "amdgcn-amd-amdhsa";
  const char *features = "";
  const char *perfConfig = "";
  mlir::miopen::addPipeline(*passMan, perfConfig, false, true);
  mlir::miopen::addBackendPipeline(*passMan, triple, chip, features);
}
