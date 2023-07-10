//===- tosa_miir.cpp - Simple test of C and MIIR APIs ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: mlir-tosa-miir-test 2>&1 | FileCheck %s
 *  */
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/MIGraphX.h"
#include "mlir-c/Dialect/Tosa.h"
#include "mlir-c/IR.h"
#include "mlir-c/IntegerSet.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir-c/RegisterRocMLIR.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/Support/TargetSelect.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
  MlirType inType = mlirRankedTensorTypeGet(4, inDims, mlirF32TypeGet(ctx),
                                            mlirAttributeGetNull());
  int64_t filtDims[] = {64, 64, 1, 1};
  MlirType filtType = mlirRankedTensorTypeGet(4, filtDims, mlirF32TypeGet(ctx),
                                              mlirAttributeGetNull());
  int64_t biasDims[] = {1, 64, 1, 1};
  MlirType biasType = mlirRankedTensorTypeGet(4, biasDims, mlirF32TypeGet(ctx),
                                              mlirAttributeGetNull());
  MlirType funcBodyArgTypes[] = {inType, filtType, biasType};
  MlirLocation funcBodyLocations[] = {location, location, location};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirType resultType = mlirRankedTensorTypeGet(4, inDims, mlirF32TypeGet(ctx),
                                                mlirAttributeGetNull());
  MlirType funcResultTypes[] = {resultType};
  MlirType funcType =
      mlirFunctionTypeGet(ctx, 3, funcBodyArgTypes, 1, funcResultTypes);
  MlirBlock funcBody =
      mlirBlockCreate(sizeof(funcBodyArgTypes) / sizeof(MlirType),
                      funcBodyArgTypes, funcBodyLocations);
  mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  //-------------- func op

  // Set func attributes
  std::string funcTypeStr;
  mlirTypePrint(funcType, printToString, &funcTypeStr);

  MlirAttribute funcTypeAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreate(funcTypeStr.data(), funcTypeStr.size()));
  MlirAttribute funcNameAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString("\"tosa_kernel\""));
  MlirNamedAttribute funcAttrs[] = {
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("type")),
          funcTypeAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_name")),
          funcNameAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("kernel")),
          mlirUnitAttrGet(ctx))};

  // Set func op
  MlirOperationState funcState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("builtin.func"), location);
  mlirOperationStateAddAttributes(&funcState, 3, funcAttrs);
  mlirOperationStateAddOwnedRegions(&funcState, 1, &funcBodyRegion);
  MlirOperation func = mlirOperationCreate(&funcState);
  mlirBlockInsertOwnedOperation(moduleBody, 0, func);

  //-------------- conv0 = tosa.conv2d

  // Set conv0 arguments : arg0 from the func and constant filter0
  MlirValue funcArg0 = mlirBlockGetArgument(funcBody, 0);
  MlirValue funcArg1 = mlirBlockGetArgument(funcBody, 1);
  MlirValue funcArg2 = mlirBlockGetArgument(funcBody, 2);
  MlirValue conv0Operands[] = {funcArg0, funcArg1, funcArg2};

  // Set convolution attributes
  // padding, stride, dilation, group, padding_mode
  MlirAttribute conv0PaddingAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString("[0:i64, 0:i64, 0:i64, 0:i64]"));
  MlirAttribute conv0StrideAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString("[1:i64, 1:i64]"));
  MlirAttribute conv0DilationAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString("[1:i64, 1:i64]"));
  MlirNamedAttribute conv0Attrs[] = {
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("pad")),
          conv0PaddingAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("stride")),
          conv0StrideAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("dilation")),
          conv0DilationAttr)};

  // Set output shape
  int64_t conv0Dims[] = {1, 64, 56, 56};
  MlirType conv0Type = mlirRankedTensorTypeGet(
      4, conv0Dims, mlirF32TypeGet(ctx), mlirAttributeGetNull());

  // Set convolution op
  MlirOperationState conv0OpState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("tosa.conv2d"), location);
  mlirOperationStateAddResults(&conv0OpState, 1, &conv0Type);
  mlirOperationStateAddOperands(&conv0OpState, 3, conv0Operands);
  mlirOperationStateAddAttributes(&conv0OpState, 3, conv0Attrs);
  MlirOperation conv0Op = mlirOperationCreate(&conv0OpState);
  mlirBlockAppendOwnedOperation(funcBody, conv0Op);
  MlirValue conv0Value = mlirOperationGetResult(conv0Op, 0);

  //-------------- tosa.relu op

  MlirAttribute relu0MinFPAttr =
      mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString("-1.0:f32"));
  MlirAttribute relu0MaxFPAttr =
      mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString("1.0:f32"));
  MlirAttribute relu0MinIntAttr =
      mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString("-1:i64"));
  MlirAttribute relu0MaxIntAttr =
      mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString("1:i64"));
  MlirNamedAttribute relu0Attrs[] = {
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("min_fp")),
          relu0MinFPAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("max_fp")),
          relu0MaxFPAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("min_int")),
          relu0MinIntAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("max_int")),
          relu0MaxIntAttr)};

  // Set relu0 arguments
  MlirValue relu0Operands[] = {conv0Value};

  // Set relu op
  int64_t relu0Dims[] = {1, 64, 56, 56};
  MlirType relu0Type = mlirRankedTensorTypeGet(
      4, relu0Dims, mlirF32TypeGet(ctx), mlirAttributeGetNull());
  MlirOperationState relu0State = mlirOperationStateGet(
      mlirStringRefCreateFromCString("tosa.clamp"), location);
  mlirOperationStateAddResults(&relu0State, 1, &relu0Type);
  mlirOperationStateAddOperands(&relu0State, 1, relu0Operands);
  mlirOperationStateAddAttributes(&relu0State, 4, relu0Attrs);

  MlirOperation relu0Op = mlirOperationCreate(&relu0State);
  mlirBlockAppendOwnedOperation(funcBody, relu0Op);
  MlirValue relu0Value = mlirOperationGetResult(relu0Op, 0);

  //-------------- std.return op

  MlirValue retOperands[] = {relu0Value};
  MlirOperationState retState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("func.return"), location);
  mlirOperationStateAddOperands(&retState, 1, retOperands);
  MlirOperation ret = mlirOperationCreate(&retState);
  mlirBlockAppendOwnedOperation(funcBody, ret);

  MlirOperation module = mlirModuleGetOperation(moduleOp);
  mlirOperationDump(module);
  // CHECK-LABEL: func @tosa_kernel

  // module  {
  //   func @tosa_kernel(%arg0: tensor<1x64x56x56xf32>, %arg1:
  //   tensor<64x64x1x1xf32>, %arg2: tensor<1x64x1x1xf32>) ->
  //   tensor<1x64x56x56xf32> attributes {kernel, arch =
  //   "amdgcn-amd-amdhsa:gfx908"} {
  //     %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0,
  //     0, 0, 0], stride = [1, 1]} : (tensor<1x64x56x56xf32>,
  //     tensor<64x64x1x1xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
  //     %1 = "tosa.clamp"(%0) {max_fp = 1.000000e+00 : f32, max_int = 1 : i64,
  //     min_fp = -1.000000e+00 : f32, min_int = -1 : i64} :
  //     (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32> return %1 :
  //     tensor<1x64x56x56xf32>
  //   }
  // }

  return moduleOp;
}

static bool constructAndTraverseIr(MlirContext ctx) {
  MlirLocation location1 = mlirLocationUnknownGet(ctx);
  MlirModule moduleOp1 = makeAndDumpMIXR(ctx, location1);

  auto module = unwrap(moduleOp1);

  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  const char *triple = "amdgcn-amd-amdhsa";
  const char *chip = "gfx908";
  const char *features = "";

  mlir::PassManager pm(module->getName(), mlir::PassManager::Nesting::Implicit);

  mlir::rock::buildBufferizePipeline(pm);

  mlir::rock::buildKernelPipeline(pm);

  mlir::rock::BackendOptions opts;
  opts.triple = triple;
  opts.chip = chip;
  opts.features = features;
  mlir::rock::buildBackendPipeline(pm, opts);

  auto status = pm.run(module);

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
