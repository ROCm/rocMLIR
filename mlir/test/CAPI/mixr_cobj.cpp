//===- tosa_miir.cpp - Simple test of C and MIIR APIs ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: mlir-mixr-full-test 2>&1 | FileCheck %s
 *  */
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/GPU.h"
#include "mlir-c/Dialect/MIGraphX.h"
#include "mlir-c/Dialect/Tosa.h"
#include "mlir-c/IR.h"
#include "mlir-c/IntegerSet.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir-c/RegisterRocMLIR.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/MIGraphX/Pipeline.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitRocMLIRDialects.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <string>
#include <vector>

void printToString(MlirStringRef str, void *userData) {
  std::string *strref = static_cast<std::string *>(userData);
  strref->append(str.data, str.length);
}

MlirModule makeAndDumpMIXR(MlirContext ctx, MlirLocation location) {
  MlirModule moduleOp = mlirModuleCreateEmpty(location);
  MlirBlock moduleBody = mlirModuleGetBody(moduleOp);

  // Set func arguments
  int64_t inDims[] = {1, 64, 56, 56};
  int64_t filter0Dims[] = {64, 64, 1, 1};
  int64_t bias0Dims[] = {64};

  MlirType inType = mlirRankedTensorTypeGet(4, inDims, mlirF32TypeGet(ctx),
                                            mlirAttributeGetNull());
  MlirType filter0Type = mlirRankedTensorTypeGet(
      4, filter0Dims, mlirF32TypeGet(ctx), mlirAttributeGetNull());
  MlirType bias0Type = mlirRankedTensorTypeGet(
      1, bias0Dims, mlirF32TypeGet(ctx), mlirAttributeGetNull());
  MlirType funcBodyArgTypes[] = {inType, filter0Type, bias0Type};
  MlirLocation funcBodyLocations[] = {location, location, location};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirBlock funcBody =
      mlirBlockCreate(sizeof(funcBodyArgTypes) / sizeof(MlirType),
                      funcBodyArgTypes, funcBodyLocations);
  mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  //-------------- func op

  // Set func attributes
  MlirAttribute funcTypeAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString(
               "(tensor<1x64x56x56xf32>, tensor<64x64x1x1xf32>, "
               "tensor<64xf32>) -> (tensor<1x64x56x56xf32>)"));
  MlirAttribute funcNameAttr =
      mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString("\"main\""));
  MlirNamedAttribute funcAttrs[] = {
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("func_type")),
          funcTypeAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_name")),
          funcNameAttr)};

  // Set func op
  MlirOperationState funcState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("func.func"), location);
  mlirOperationStateAddAttributes(&funcState, 2, funcAttrs);
  mlirOperationStateAddOwnedRegions(&funcState, 1, &funcBodyRegion);
  MlirOperation func = mlirOperationCreate(&funcState);
  mlirBlockInsertOwnedOperation(moduleBody, 0, func);

  //-------------- conv0 = migraphx.convolution

  // Set conv0 arguments : arg0 from the func and constant filter0
  MlirValue funcArg0 = mlirBlockGetArgument(funcBody, 0);
  MlirValue funcArg1 = mlirBlockGetArgument(funcBody, 1);
  MlirValue conv0Operands[] = {funcArg0, funcArg1};

  // Set convolution attributes
  // padding, stride, dilation, group, padding_mode
  MlirAttribute conv0PaddingAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString("[0:i64, 0:i64, 0:i64, 0:i64]"));
  MlirAttribute conv0StrideAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString("[1:i64, 1:i64]"));
  MlirAttribute conv0DilationAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString("[1:i64, 1:i64]"));
  MlirAttribute conv0GroupAttr =
      mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString("1:i64"));
  MlirAttribute conv0PaddingModeAttr =
      mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString("0:i64"));
  MlirNamedAttribute conv0Attrs[] = {
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("padding")),
          conv0PaddingAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("stride")),
          conv0StrideAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("dilation")),
          conv0DilationAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("group")),
          conv0GroupAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx,
                            mlirStringRefCreateFromCString("padding_mode")),
          conv0PaddingModeAttr)};

  // Set output shape
  int64_t conv0Dims[] = {1, 64, 56, 56};
  MlirType conv0Type = mlirRankedTensorTypeGet(
      4, conv0Dims, mlirF32TypeGet(ctx), mlirAttributeGetNull());

  // Set convolution op
  MlirOperationState conv0OpState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("migraphx.convolution"), location);
  mlirOperationStateAddResults(&conv0OpState, 1, &conv0Type);
  mlirOperationStateAddOperands(&conv0OpState, 2, conv0Operands);
  mlirOperationStateAddAttributes(&conv0OpState, 5, conv0Attrs);
  MlirOperation conv0Op = mlirOperationCreate(&conv0OpState);
  mlirBlockAppendOwnedOperation(funcBody, conv0Op);
  MlirValue conv0Value = mlirOperationGetResult(conv0Op, 0);
  /*
    //-------------- migraphx.relu op
    // Set relu0 arguments
    MlirValue relu0Operands[] = {conv0Value};

    // Set relu op
    int64_t relu0Dims[] = {1, 64, 56, 56};
    MlirType relu0Type = mlirRankedTensorTypeGet(4, relu0Dims,
    mlirF32TypeGet(ctx), mlirAttributeGetNull()); MlirOperationState relu0State
    = mlirOperationStateGet( mlirStringRefCreateFromCString("migraphx.relu"),
    location); mlirOperationStateAddResults(&relu0State, 1, &relu0Type);
    mlirOperationStateAddOperands(&relu0State, 1, relu0Operands);

    MlirOperation relu0Op = mlirOperationCreate(&relu0State);
    mlirBlockAppendOwnedOperation(funcBody, relu0Op);
    MlirValue relu0Value = mlirOperationGetResult(relu0Op, 0);
  */
  //-------------- std.return op

  MlirValue retOperands[] = {conv0Value};
  MlirOperationState retState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("func.return"), location);
  mlirOperationStateAddOperands(&retState, 1, retOperands);
  MlirOperation ret = mlirOperationCreate(&retState);
  mlirBlockAppendOwnedOperation(funcBody, ret);

  return moduleOp;
}

static bool constructAndTraverseIr(MlirContext ctx) {
  MlirLocation location1 = mlirLocationUnknownGet(ctx);
  MlirModule moduleOp1 = makeAndDumpMIXR(ctx, location1);

  auto module = unwrap(moduleOp1);

  const char *triple = "amdgcn-amd-amdhsa";
  const char *chip = "gfx908";
  const char *features = "";

  MlirOperation moduleMO = mlirModuleGetOperation(moduleOp1);

  mlir::PassManager pm0(module->getName(),
                        mlir::PassManager::Nesting::Implicit);
  mlir::PassManager pm(module->getName(), mlir::PassManager::Nesting::Implicit);

  mlir::migraphx::addHighLevelPipeline(pm0);
  (void)pm0.run(module);
  mlirOperationDump(moduleMO);

  mlir::rock::buildBufferizePipeline(pm);
  (void)pm.run(module);
  mlirOperationDump(moduleMO);

  size_t argIdx = 0;
  module.walk([&](mlir::func::FuncOp f) {
    auto args = f.getArguments();
    for (auto arg : args) {
      argIdx += 3; // 3 per memref : allocated ptr, aligned ptr, offset
      auto sType = arg.getType().template cast<mlir::ShapedType>();
      long long rank = sType.getRank();
      printf("rank:%lld, dim:", rank);
      for (int64_t i = 0; i < rank; ++i)
        printf("<%lld>", static_cast<long long>(sType.getDimSize(i)));
      printf("\n");
      argIdx += rank * 2; // 2 per each dimension : size, stride
    }
    std::string kernelName = f.getName().str();
    printf("Kernel name : %s\n", kernelName.c_str());
  });
  // CHECK: rank:4, dim:<1><64><56><56>
  // CHECK: rank:4, dim:<64><64><1><1>
  // CHECK: rank:1, dim:<64>
  // CHECK: rank:4, dim:<1><64><56><56>
  // CHECK: Kernel name : main

  // 4 memref in this example : input, filter, bias and result
  // example : memref<1x64x56x56xf32>
  // uses 11 params : ptr, ptr, 0 /*offset */, 1, 64, 56, 56, 1, 64, 56, 56
  // printf("Estimated #kernel params : %d\n", argIdx);

  mlir::rock::buildKernelPipeline(pm);

  mlir::rock::BackendOptions opts;
  opts.triple = triple;
  opts.chip = chip;
  opts.features = features;
  mlir::rock::buildBackendPipeline(pm, opts);

  auto status = pm.run(module);

  module.walk([&](mlir::LLVM::LLVMFuncOp llvmFunc) {
    size_t block_size =
        llvmFunc->getAttrOfType<mlir::IntegerAttr>("block_size").getInt();
    size_t grid_size =
        llvmFunc->getAttrOfType<mlir::IntegerAttr>("grid_size").getInt();
    auto funcType =
        llvmFunc.getFunctionType().dyn_cast<mlir::LLVM::LLVMFunctionType>();
    uint32_t numOperands = funcType.getNumParams();
    printf("kernel params : %u\n", numOperands);
    printf("block_size : %zu\n", block_size);
    printf("grid_size : %zu\n", grid_size);
  });
  // CHECK: kernel params : 38
  // CHECK: block_size : 64
  // CHECK: grid_size : 56

  size_t size;
  module.walk([&](mlir::gpu::GPUModuleOp gpuModule) {
    auto hsacoAttr = gpuModule->getAttrOfType<mlir::StringAttr>(
        mlir::gpu::getDefaultGpuBinaryAnnotation());
    if (hsacoAttr) {
      size = hsacoAttr.getValue().size();
      // printf("Binary size : %d\n", size);
    }
  });

  std::vector<char> buffer(size);
  module.walk([&](mlir::gpu::GPUModuleOp gpuModule) {
    auto hsacoAttr = gpuModule->getAttrOfType<mlir::StringAttr>(
        mlir::gpu::getDefaultGpuBinaryAnnotation());
    if (hsacoAttr) {
      std::string hsaco = hsacoAttr.getValue().str();
      std::copy(hsaco.begin(), hsaco.end(), buffer.data());
      /*std::cout << "hsaco = ";
      for(auto o: buffer)
        std::cout << o;
      std::cout << std::endl;*/
    }
  });

  mlirModuleDestroy(moduleOp1);
  if (status.succeeded()) {
    // CHECK: PASSED!
    printf("PASSED!\n");
    return true;
  }
  return false;
}

int main() {
  MlirContext ctx = mlirContextCreate();
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirRegisterRocMLIRDialects(registry);
  mlirRegisterRocMLIRPasses();
  mlirContextAppendDialectRegistry(ctx, registry);
  // TODO: this is a emulation of an old behavior, we should load only the
  // dialects we use
  mlirContextLoadAllAvailableDialects(ctx);
  mlirDialectRegistryDestroy(registry);

  if (!constructAndTraverseIr(ctx)) {
    printf("FAILED!\n");
    return 1;
  }

  mlirContextDestroy(ctx);

  return 0;
}
