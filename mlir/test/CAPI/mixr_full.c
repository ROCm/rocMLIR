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
#include "mlir-c/Pass.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir-c/RegisterRocMLIR.h"
#include "mlir-c/Support.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

MlirModule makeAndDumpMIXR(MlirContext ctx, MlirLocation loc) {
  MlirModule module = mlirModuleCreateEmpty(loc);
  MlirBlock moduleBody = mlirModuleGetBody(module);

  // Set func arguments
  int64_t inDims[] = {1, 64, 56, 56};
  int64_t inStrides[] = {200704, 3136, 56, 1};
  int64_t filter0Dims[] = {64, 64, 1, 1};
  int64_t filter0Strides[] = {64, 1, 1, 1};
  int64_t bias0Dims[] = {64};
  int64_t bias0Strides[] = {1};

  MlirType inType =
      rocmlirMIXRShapedTypeGet(4, inDims, inStrides, mlirF32TypeGet(ctx));
  MlirType filter0Type = rocmlirMIXRShapedTypeGet(
      4, filter0Dims, filter0Strides, mlirF32TypeGet(ctx));
  MlirType bias0Type =
      rocmlirMIXRShapedTypeGet(1, bias0Dims, bias0Strides, mlirF32TypeGet(ctx));
  MlirType funcBodyArgTypes[] = {inType, filter0Type, bias0Type};
  MlirLocation funcBodyArglocs[] = {loc, loc, loc};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirBlock funcBody =
      mlirBlockCreate(sizeof(funcBodyArgTypes) / sizeof(MlirType),
                      funcBodyArgTypes, funcBodyArglocs);
  mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  //-------------- func op

  // Set func attributes
  MlirAttribute funcTypeAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString(
               "(!migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1>, "
               "!migraphx.shaped<64x64x1x1xf32, 64x1x1x1>, "
               "!migraphx.shaped<64xf32, 1>) -> "
               "(!migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1>)"));
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
  MlirOperationState funcState =
      mlirOperationStateGet(mlirStringRefCreateFromCString("func.func"), loc);
  mlirOperationStateAddAttributes(&funcState, 2, funcAttrs);
  mlirOperationStateAddOwnedRegions(&funcState, 1, &funcBodyRegion);
  MlirOperation func = mlirOperationCreate(&funcState);
  mlirOperationSetAttributeByName(func,
                                  mlirStringRefCreateFromCString("rock.kernel"),
                                  mlirUnitAttrGet(ctx));
  mlirOperationSetAttributeByName(
      func, mlirStringRefCreateFromCString("rock.arch"),
      mlirStringAttrGet(
          ctx, mlirStringRefCreateFromCString("gfx908:sramecc+:xnack-")));
  mlirBlockInsertOwnedOperation(moduleBody, 0, func);

  //-------------- conv0 = migraphx.convolution

  // Set conv0 arguments : arg0 from the func and constant filter0
  MlirValue funcArg0 = mlirBlockGetArgument(funcBody, 0);
  MlirValue funcArg1 = mlirBlockGetArgument(funcBody, 1);
  MlirValue conv0Operands[] = {funcArg0, funcArg1};

  // Set convolution attributes
  // padding, stride, dilation, group, padding_mode, acc_type
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
  int64_t conv0Strides[] = {200704, 3136, 56, 1};
  MlirType conv0Type =
      rocmlirMIXRShapedTypeGet(4, conv0Dims, conv0Strides, mlirF32TypeGet(ctx));

  // Set convolution op
  MlirOperationState conv0OpState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("migraphx.convolution"), loc);
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
  int64_t relu0Strides[] = {200704, 3136, 56, 1};
  MlirType relu0Type =
      rocmlirMIXRShapedTypeGet(4, relu0Dims, relu0Strides, mlirF32TypeGet(ctx));
  MlirOperationState relu0State = mlirOperationStateGet(
      mlirStringRefCreateFromCString("migraphx.relu"), loc);
  mlirOperationStateAddResults(&relu0State, 1, &relu0Type);
  mlirOperationStateAddOperands(&relu0State, 1, relu0Operands);

  MlirOperation relu0Op = mlirOperationCreate(&relu0State);
  mlirBlockAppendOwnedOperation(funcBody, relu0Op);
  MlirValue relu0Value = mlirOperationGetResult(relu0Op, 0);

  //-------------- std.return op

  MlirValue retOperands[] = {relu0Value};
  MlirOperationState retState =
      mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
  mlirOperationStateAddOperands(&retState, 1, retOperands);
  MlirOperation ret = mlirOperationCreate(&retState);
  mlirBlockAppendOwnedOperation(funcBody, ret);

  return module;
}

static MlirModule cloneModule(MlirModule module) {
  return mlirModuleFromOperation(
      mlirOperationClone(mlirModuleGetOperation(module)));
}

static bool constructAndTraverseIr(MlirContext ctx) {
  MlirLocation loc = mlirLocationUnknownGet(ctx);
  MlirModule module = makeAndDumpMIXR(ctx, loc);
  MlirOperation moduleOp = mlirModuleGetOperation(module);

  MlirPassManager highLevelPm = mlirPassManagerCreate(ctx);
  MlirPassManager backendPm = mlirPassManagerCreate(ctx);
  // Call high level pipeline on root module.
  mlirMIGraphXAddHighLevelPipeline(highLevelPm);
  if (mlirLogicalResultIsFailure(
          mlirPassManagerRunOnOp(highLevelPm, moduleOp))) {
    printf("Running high-level pipeline failed\n");
    return false;
  }

  MlirRockTuningSpace tuningSpace =
      mlirRockTuningSpaceCreate(module, RocmlirTuningParamSetKindFull);
  printf("Got tuning space,\n");
  unsigned fNum = mlirRockTuningGetNumParams(tuningSpace);
  // CHECK: full set = 932
  printf("full set = %u\n", fNum);
  MlirRockTuningParam tuningParam = mlirRockTuningParamCreate();
  MlirRockTuningTable tuningTable = mlirRockTuningTableCreate();

  MlirMIGraphXBackendOptions opts = {"gfx908:sramecc+:xnack-", NULL, 3};
  unsigned numSuccesses = 0;
  char problemKey[ROCMLIR_TUNING_KEY_BUFSZ];
  size_t problemBytes =
      mlirRockTuningGetKey(module, problemKey, ROCMLIR_TUNING_KEY_BUFSZ);
  if (problemBytes >= ROCMLIR_TUNING_KEY_BUFSZ) {
    printf("Tuning key string too long - %lu bytes", problemBytes);
    return false;
  }

  for (unsigned i = 0; i < fNum && numSuccesses < 2; ++i) {
    if (!mlirRockTuningParamGet(tuningSpace, i, tuningParam)) {
      printf("fails to obtain param\n");
      return false;
    }

    float fakeTime = (float)(i + 1);
    char paramStr[ROCMLIR_TUNING_PARAM_STRING_BUFSZ];
    size_t paramBytes = mlirRockTuningParamToString(
        tuningParam, paramStr, ROCMLIR_TUNING_PARAM_STRING_BUFSZ);
    if (paramBytes >= ROCMLIR_TUNING_PARAM_STRING_BUFSZ) {
      printf("Parameter string too long - %lu bytes", paramBytes);
      return false;
    }

    MlirModule tuningClone = cloneModule(module);
    MlirOperation tuningCloneOp = mlirModuleGetOperation(tuningClone);
    mlirRockTuningSetFromStr(tuningClone,
                             mlirStringRefCreateFromCString(paramStr));
    MlirPassManager checkPm = mlirPassManagerCreate(ctx);
    if (!mlirMIGraphXAddBackendPipeline(checkPm, &opts)) {
      mlirPassManagerDestroy(checkPm);
      mlirModuleDestroy(tuningClone);
      continue;
    }
    if (mlirLogicalResultIsFailure(
            mlirPassManagerRunOnOp(checkPm, tuningCloneOp))) {
      // CHECK-NOT: is not applicable
      printf("Perfconfig \"%s\" is not applicable to the problem string(%s)\n",
             paramStr, problemKey);
      mlirPassManagerDestroy(checkPm);
      mlirModuleDestroy(tuningClone);
      continue;
    }
    mlirPassManagerDestroy(checkPm);
    // CHECK-2: Update perfconfig for the problem
    printf(
        "Update perfconfig for the problem string(%s): \"%s\" with time %f\n",
        problemKey, paramStr, fakeTime);
    // CHECK: fails to update table, existing config is faster
    if (!mlirRockTuningUpdateTable(
            tuningTable, mlirStringRefCreateFromCString(problemKey),
            mlirStringRefCreateFromCString(paramStr), fakeTime)) {
      printf("fails to update table, existing config is faster\n");
    }
    ++numSuccesses;
    mlirModuleDestroy(tuningClone);
  }

  if (!mlirRockTuningSetFromTable(tuningTable, module)) {
    printf("fails to set param\n");
    return false;
  }

  mlirRockTuningTableDestroy(tuningTable);
  mlirRockTuningParamDestroy(tuningParam);
  mlirRockTuningSpaceDestroy(tuningSpace);

  // Run compilation pipeline on tuned config.
  if (!mlirMIGraphXAddBackendPipeline(backendPm, &opts)) {
    printf("Errors in building backend pipeline\n");
    return false;
  }
  if (mlirLogicalResultIsFailure(mlirPassManagerRunOnOp(backendPm, moduleOp))) {
    printf("Errors in running backend pipeline on known-good config\n");
    return false;
  }

  uint32_t attrs[3];
  // returns block size, grid size, and cluster size
  mlirGetKernelAttrs(module, attrs);
  printf("block size : %d, grid size : %d, cluster size : %d\n", attrs[0],
         attrs[1], attrs[2]);

  // returns binary size
  size_t binSize = 0;
  if (!mlirGetBinary(module, &binSize, NULL)) {
    return false;
  }
  printf("bin size : %lu\n", binSize);

  char *compiledBin = malloc(binSize);
  // Initialize the memory to hold binary, just for verification, not necessary.
  memset(compiledBin, '\0', binSize);

  // get binary
  if (mlirGetBinary(module, NULL, compiledBin)) {
    // printf("dump : %s \n", compiledBin);
    // CHECK: PASSED!
    printf("PASSED!\n");
  }

  mlirPassManagerDestroy(highLevelPm);
  mlirPassManagerDestroy(backendPm);
  mlirModuleDestroy(module);
  return true;
}

// Check estimated LDS usage against the arch capacity from problem sizes
// alone (no module/compilation).
static void checkLdsUsageFits(MlirContext ctx) {
  MlirType f16 = mlirF16TypeGet(ctx);
  MlirModule noModule = {NULL};
  int gemmGemmFits = mlirMIGraphXLDSUsageFitsArch(64, "gfx942", f16, noModule);
  // CHECK: gemm-gemm LDS fits : 1
  printf("gemm-gemm LDS fits : %d\n", gemmGemmFits);
  // A problem far larger than the arch's shared memory does not fit.
  int hugeFits = mlirMIGraphXLDSUsageFitsArch(8192, "gfx942", f16, noModule);
  // CHECK: huge LDS fits : 0
  printf("huge LDS fits : %d\n", hugeFits);

  // The estimator is width-based rather than limited to compiler-supported
  // attention input types, so f64 can be estimated as well.
  MlirType f64 = mlirF64TypeGet(ctx);
  int f64Fits = mlirMIGraphXLDSUsageFitsArch(64, "gfx942", f64, noModule);
  // CHECK: f64 LDS fits : 1
  printf("f64 LDS fits : %d\n", f64Fits);

  // A type without an integer or floating-point width cannot be estimated.
  MlirType none = mlirNoneTypeGet(ctx);
  int unsupportedTypeFits =
      mlirMIGraphXLDSUsageFitsArch(64, "gfx942", none, noModule);
  // CHECK: unsupported type LDS fits : 0
  printf("unsupported type LDS fits : %d\n", unsupportedTypeFits);

  // A non-positive gemmO is an invalid problem and cannot be estimated.
  int invalidGemmOFits =
      mlirMIGraphXLDSUsageFitsArch(0, "gfx942", f16, noModule);
  // CHECK: invalid gemmO LDS fits : 0
  printf("invalid gemmO LDS fits : %d\n", invalidGemmOFits);

  // A null arch is rejected.
  int nullArchFits = mlirMIGraphXLDSUsageFitsArch(64, NULL, f16, noModule);
  // CHECK: null arch LDS fits : 0
  printf("null arch LDS fits : %d\n", nullArchFits);

  // A malformed arch string is rejected by arch validation, so it reports
  // "does not fit" rather than aborting inside the estimator.
  int badArchFits = mlirMIGraphXLDSUsageFitsArch(64, "badarch", f16, noModule);
  // CHECK: invalid arch LDS fits : 0
  printf("invalid arch LDS fits : %d\n", badArchFits);

  // When a module is supplied, LDS fit is decided by lowering it and running
  // the kernel/backend pipeline internally; the per-problem (gemmO, arch,
  // elementType) args are ignored. The module is passed in the MIGraphX dialect
  // (the function clones and lowers it). This convolution is applicable on the
  // module's own arch (gfx908) with the smallest config.
  MlirLocation loc = mlirLocationUnknownGet(ctx);
  MlirModule module = makeAndDumpMIXR(ctx, loc);
  int moduleFits = mlirMIGraphXLDSUsageFitsArch(0, NULL, f16, module);
  // CHECK: module LDS fits : 1
  printf("module LDS fits : %d\n", moduleFits);
  mlirModuleDestroy(module);

  // Conversely, a fused GEMM+GEMM (dot -> dot) whose second GEMM carries an
  // explicit perf config requesting block/K tiles far larger than the arch's
  // shared memory cannot fit into LDS, so the lowering pipeline's LDS gate
  // rejects it and the module path reports "does not fit".
  const char *bigGemmGemmSrc =
      "module {\n"
      "  func.func @main(%arg0: !migraphx.shaped<1x64x64xf16, 4096x64x1>, "
      "%arg1: !migraphx.shaped<1x64x64xf16, 4096x64x1>, "
      "%arg2: !migraphx.shaped<1x64x64xf16, 4096x64x1>) -> "
      "!migraphx.shaped<1x64x64xf16, 4096x64x1> "
      "attributes {rock.kernel = \"mixr\", "
      "rock.arch = \"gfx942:sramecc+:xnack-\"} {\n"
      "    %0 = migraphx.dot %arg0, %arg1 : "
      "!migraphx.shaped<1x64x64xf16, 4096x64x1>, "
      "!migraphx.shaped<1x64x64xf16, 4096x64x1> -> "
      "!migraphx.shaped<1x64x64xf16, 4096x64x1>\n"
      "    %1 = migraphx.dot %0, %arg2 "
      "{perf_config = \"attn:v3:256,256,256,256,32,32,16,8,1,1,2,0,1\"} : "
      "!migraphx.shaped<1x64x64xf16, 4096x64x1>, "
      "!migraphx.shaped<1x64x64xf16, 4096x64x1> -> "
      "!migraphx.shaped<1x64x64xf16, 4096x64x1>\n"
      "    return %1 : !migraphx.shaped<1x64x64xf16, 4096x64x1>\n"
      "  }\n"
      "}\n";
  MlirModule bigModule = mlirModuleCreateParse(
      ctx, mlirStringRefCreateFromCString(bigGemmGemmSrc));
  if (mlirModuleIsNull(bigModule)) {
    printf("failed to parse oversized gemm-gemm module\n");
  } else {
    int bigModuleFits = mlirMIGraphXLDSUsageFitsArch(0, NULL, f16, bigModule);
    // CHECK: oversized module LDS fits : 0
    printf("oversized module LDS fits : %d\n", bigModuleFits);
    mlirModuleDestroy(bigModule);
  }
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirRegisterRocMLIRDialects(registry);
  mlirRegisterRocMLIRPasses();
  mlirRegisterRocMLIRLibCLOptions();
  mlirContextAppendDialectRegistry(ctx, registry);
  // TODO: this is a emulation of an old behavior, we should load only the
  // dialects we use
  mlirContextLoadAllAvailableDialects(ctx);
  mlirDialectRegistryDestroy(registry);

  if (!constructAndTraverseIr(ctx)) {
    printf("FAILED!\n");
    return 1;
  }

  checkLdsUsageFits(ctx);

  mlirContextDestroy(ctx);
  return 0;
}
