//===- tosa_miir.cpp - Simple test of C and MIIR APIs ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: mlir-mixr-fullc-test 2>&1 | FileCheck %s
 *  */
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Dialect/MIGraphX.h"
#include "mlir-c/Dialect/Rock.h"
#include "mlir-c/IR.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir-c/RegisterRocMLIR.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

MlirModule makeAndDumpMIXR(MlirContext ctx, MlirLocation location) {
  MlirModule module = mlirModuleCreateEmpty(location);
  MlirBlock moduleBody = mlirModuleGetBody(module);

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
  MlirLocation funcBodyArglocs[] = {location, location, location};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirBlock funcBody =
      mlirBlockCreate(sizeof(funcBodyArgTypes) / sizeof(MlirType),
                      funcBodyArgTypes, funcBodyArglocs);
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
          mlirIdentifierGet(ctx,
                            mlirStringRefCreateFromCString("function_type")),
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
  mlirOperationSetAttributeByName(
      func, mlirStringRefCreateFromCString("kernel"), mlirUnitAttrGet(ctx));
  mlirOperationSetAttributeByName(
      func, mlirStringRefCreateFromCString("arch"),
      mlirStringAttrGet(
          ctx, mlirStringRefCreateFromCString("gfx908:sramecc+:xnack-")));
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

  //-------------- migraphx.relu op

  // Set relu0 arguments
  MlirValue relu0Operands[] = {conv0Value};

  // Set relu op
  int64_t relu0Dims[] = {1, 64, 56, 56};
  MlirType relu0Type = mlirRankedTensorTypeGet(
      4, relu0Dims, mlirF32TypeGet(ctx), mlirAttributeGetNull());
  MlirOperationState relu0State = mlirOperationStateGet(
      mlirStringRefCreateFromCString("migraphx.relu"), location);
  mlirOperationStateAddResults(&relu0State, 1, &relu0Type);
  mlirOperationStateAddOperands(&relu0State, 1, relu0Operands);

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

  return module;
}

static bool constructAndTraverseIr(MlirContext ctx) {
  MlirLocation location1 = mlirLocationUnknownGet(ctx);
  MlirModule module = makeAndDumpMIXR(ctx, location1);

  MlirPassManager pm = mlirPassManagerCreate(ctx);
  MlirPassManager pm1 = mlirPassManagerCreate(ctx);
  // 1st pipeline to call
  mlirMIGraphXAddHighLevelPipeline(pm);
  MlirOperation moduleOp = mlirModuleGetOperation(module);
  mlirPassManagerRunOnOp(pm, moduleOp);

  MlirRockTuningSpace tuningSpace = mlirRockTuningSpaceCreate(module);
  printf("Got tuning space,\n");
  int qNum = mlirRockTuningGetNumParamsQuick(tuningSpace);
  int fNum = mlirRockTuningGetNumParamsFull(tuningSpace);
  printf("quick set = %d, full set = %d\n", qNum, fNum);
  MlirRockTuningParam tuningParam = mlirRockTuningParamCreate();
  MlirRockTuningTable tuningTable = mlirRockTuningTableCreate();

  for (int i = 0; i < 2; i++) {
    if (!mlirRockTuningParamGet(tuningSpace, i, tuningParam)) {
      printf("fails to obtain param\n");
      return false;
    }
    float fakeTime = (float)(i + 1);
    char *paramStr = strdup(mlirRockTuningGetParamStr(tuningParam));
    char *problemKey = strdup(mlirRockTuningGetKey(tuningTable, module));
    printf(
        "Update perfconfig for the problem string(%s): \"%s\" with time %f\n",
        problemKey, paramStr, fakeTime);
    if (!mlirRockTuningUpdateTable(tuningTable,
                                   mlirRockTuningGetKey(tuningTable, module),
                                   paramStr, fakeTime)) {
      printf("fails to update table, maybe existing config is faster\n");
    }
    free(paramStr);
    free(problemKey);
  }

  if (!mlirRockTuningSetFromTable(tuningTable, module)) {
    printf("fails to set param\n");
    return false;
  }

  mlirRockTuningTableDestroy(tuningTable);
  mlirRockTuningParamDestroy(tuningParam);
  mlirRockTuningSpaceDestroy(tuningSpace);

  mlirOperationDump(moduleOp);
  // CHECK-LABEL: func @main

  // returns the required buffer size to hold information including
  // ranks, dimensions of each arguments and kernel name.
  int argSize = 0;
  mlirGetKernelInfo(module, &argSize, NULL);
  void *argInfo = malloc(argSize + 1);
  // get the data
  mlirGetKernelInfo(module, NULL, (void *)argInfo);
  int *argData = (int *)argInfo;

  int idx = 1;
  for (int i = 0; i < argData[0]; i++) {
    // iterate per each memref argument
    int rank = argData[idx++];
    printf("arg#%d (rank %d): ", i, rank);
    for (int j = 0; j < rank; j++) {
      printf("<dim %d : %d> ", j, argData[idx++]);
    }
    printf("\n");
  }

  // The last part of the retrieved data contains the kernel name.
  char *nameData = (char *)(argData + idx);
  ((char *)argInfo)[argSize] = '\0';
  printf("kernel name : %s\n", nameData);

  // 2nd pipeline to call
  const char *deviceName = "gfx908:sramecc+:xnack-";
  if (!mlirMIGraphXAddBackendPipeline(pm1, deviceName)) {
    printf("Errors in building backend pipeline\n");
    return false;
  }
  mlirPassManagerRunOnOp(pm1, moduleOp);

  uint32_t attrs[2];
  // returns block and grid sizes
  mlirGetKernelAttrs(module, attrs);
  printf("block size : %d, grid size : %d\n", attrs[0], attrs[1]);

  // returns binary size
  int binSize = 0;
  mlirGetBinary(module, &binSize, NULL);
  printf("bin size : %d\n", binSize);

  char *compiledBin = malloc(binSize);
  // Initialize the memory to hold binary, just for verification, not necessary.
  for (int i = 0; i < binSize; i++)
    compiledBin[i] = '0';

  // get binary
  if (mlirGetBinary(module, NULL, compiledBin)) {
    // printf("dump : %s \n", compiledBin);
    // CHECK: PASSED!
    printf("PASSED!\n");
  }

  mlirPassManagerDestroy(pm);
  mlirPassManagerDestroy(pm1);
  mlirModuleDestroy(module);
  return true;
}

int main(void) {
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
