//===- mixr_attention.c - Test C API for migraphx.attention --------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2026 Advanced Micro Devices
//
//===----------------------------------------------------------------------===//

/* RUN: mlir-mixr-attention-test 2>&1 | FileCheck %s
 *  */
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Dialect/MIGraphX.h"
#include "mlir-c/IR.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir-c/RegisterRocMLIR.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Dump the module after asserting it verifies. Catches future regressions
// where a builder change produces IR that prints fine but fails the
// verifier (which mlirOperationDump does not check on its own).
static void verifyAndDump(MlirModule mod, const char *testName) {
  if (!mlirOperationVerify(mlirModuleGetOperation(mod))) {
    fprintf(stderr, "FAIL: %s produced invalid IR\n", testName);
    mlirOperationDump(mlirModuleGetOperation(mod));
    exit(1);
  }
  mlirOperationDump(mlirModuleGetOperation(mod));
}

static MlirOperation createFuncOp(MlirContext ctx, MlirLocation loc,
                                  const char *name, const char *funcTypeSig,
                                  MlirRegion bodyRegion) {
  MlirAttribute funcTypeAttr =
      mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString(funcTypeSig));
  MlirAttribute funcNameAttr =
      mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString(name));
  MlirNamedAttribute funcAttrs[] = {
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx,
                            mlirStringRefCreateFromCString("function_type")),
          funcTypeAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_name")),
          funcNameAttr)};

  MlirOperationState funcState =
      mlirOperationStateGet(mlirStringRefCreateFromCString("func.func"), loc);
  mlirOperationStateAddAttributes(&funcState, 2, funcAttrs);
  mlirOperationStateAddOwnedRegions(&funcState, 1, &bodyRegion);
  return mlirOperationCreate(&funcState);
}

static MlirOperation createReturnOp(MlirLocation loc, intptr_t numOperands,
                                    MlirValue *operands) {
  MlirOperationState retState =
      mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
  mlirOperationStateAddOperands(&retState, numOperands, operands);
  return mlirOperationCreate(&retState);
}

// CHECK-LABEL: === Test: basic attention ===
static void testBasicAttention(MlirContext ctx, MlirLocation loc) {
  fprintf(stderr, "=== Test: basic attention ===\n");
  MlirModule moduleOp = mlirModuleCreateEmpty(loc);
  MlirBlock moduleBody = mlirModuleGetBody(moduleOp);

  int64_t qDims[] = {2, 64, 128};
  int64_t qStrides[] = {8192, 128, 1};
  int64_t kDims[] = {2, 128, 256};
  int64_t kStrides[] = {32768, 256, 1};
  int64_t vDims[] = {2, 256, 64};
  int64_t vStrides[] = {16384, 64, 1};

  MlirType qType =
      rocmlirMIXRShapedTypeGet(3, qDims, qStrides, mlirF16TypeGet(ctx));
  MlirType kType =
      rocmlirMIXRShapedTypeGet(3, kDims, kStrides, mlirF16TypeGet(ctx));
  MlirType vType =
      rocmlirMIXRShapedTypeGet(3, vDims, vStrides, mlirF16TypeGet(ctx));

  MlirType funcBodyArgTypes[] = {qType, kType, vType};
  MlirLocation funcBodyLocs[] = {loc, loc, loc};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirBlock funcBody = mlirBlockCreate(3, funcBodyArgTypes, funcBodyLocs);
  mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  int64_t resultDims[] = {2, 64, 64};
  int64_t resultStrides[] = {4096, 64, 1};
  MlirType resultType = rocmlirMIXRShapedTypeGet(3, resultDims, resultStrides,
                                                 mlirF16TypeGet(ctx));

  MlirRegion emptyRegion = mlirRegionCreate();

  MlirOperation attnOp = rocmlirMIGraphXAttentionCreate(
      loc, mlirBlockGetArgument(funcBody, 0), mlirBlockGetArgument(funcBody, 1),
      mlirBlockGetArgument(funcBody, 2), 0, NULL, resultType, (MlirType){NULL},
      (MlirType){NULL}, emptyRegion, MLIR_MIGRAPHX_ATTENTION_NONE,
      (MlirValue){NULL}, (MlirValue){NULL}, 0, 0);
  mlirBlockAppendOwnedOperation(funcBody, attnOp);

  MlirValue attnResult = mlirOperationGetResult(attnOp, 0);
  MlirOperation retOp = createReturnOp(loc, 1, &attnResult);
  mlirBlockAppendOwnedOperation(funcBody, retOp);

  MlirOperation func =
      createFuncOp(ctx, loc, "\"test_basic_attention\"",
                   "(!migraphx.shaped<2x64x128xf16, 8192x128x1>, "
                   "!migraphx.shaped<2x128x256xf16, 32768x256x1>, "
                   "!migraphx.shaped<2x256x64xf16, 16384x64x1>) -> "
                   "!migraphx.shaped<2x64x64xf16, 4096x64x1>",
                   funcBodyRegion);
  mlirBlockInsertOwnedOperation(moduleBody, 0, func);

  // CHECK: migraphx.attention %arg0, %arg1, %arg2
  // CHECK: !migraphx.shaped<2x64x64xf16, 4096x64x1>
  verifyAndDump(moduleOp, "basic attention");
  mlirModuleDestroy(moduleOp);
}

// CHECK-LABEL: === Test: attention with LSE ===
static void testAttentionWithLse(MlirContext ctx, MlirLocation loc) {
  fprintf(stderr, "=== Test: attention with LSE ===\n");
  MlirModule moduleOp = mlirModuleCreateEmpty(loc);
  MlirBlock moduleBody = mlirModuleGetBody(moduleOp);

  int64_t qDims[] = {2, 64, 128};
  int64_t qStrides[] = {8192, 128, 1};
  int64_t kDims[] = {2, 128, 256};
  int64_t kStrides[] = {32768, 256, 1};
  int64_t vDims[] = {2, 256, 64};
  int64_t vStrides[] = {16384, 64, 1};

  MlirType qType =
      rocmlirMIXRShapedTypeGet(3, qDims, qStrides, mlirF16TypeGet(ctx));
  MlirType kType =
      rocmlirMIXRShapedTypeGet(3, kDims, kStrides, mlirF16TypeGet(ctx));
  MlirType vType =
      rocmlirMIXRShapedTypeGet(3, vDims, vStrides, mlirF16TypeGet(ctx));

  MlirType funcBodyArgTypes[] = {qType, kType, vType};
  MlirLocation funcBodyLocs[] = {loc, loc, loc};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirBlock funcBody = mlirBlockCreate(3, funcBodyArgTypes, funcBodyLocs);
  mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  int64_t resultDims[] = {2, 64, 64};
  int64_t resultStrides[] = {4096, 64, 1};
  MlirType resultType = rocmlirMIXRShapedTypeGet(3, resultDims, resultStrides,
                                                 mlirF16TypeGet(ctx));

  int64_t lseDims[] = {2, 64};
  int64_t lseStrides[] = {64, 1};
  MlirType lseType =
      rocmlirMIXRShapedTypeGet(2, lseDims, lseStrides, mlirF32TypeGet(ctx));

  MlirRegion emptyRegion = mlirRegionCreate();

  // softmax_type = f32 is required because V is f16 and LSE is f32; the
  // verifier requires lse element type to match the effective softmax
  // type.
  MlirOperation attnOp = rocmlirMIGraphXAttentionCreate(
      loc, mlirBlockGetArgument(funcBody, 0), mlirBlockGetArgument(funcBody, 1),
      mlirBlockGetArgument(funcBody, 2), 0, NULL, resultType, lseType,
      mlirF32TypeGet(ctx), emptyRegion, MLIR_MIGRAPHX_ATTENTION_NONE,
      (MlirValue){NULL}, (MlirValue){NULL}, 0, 0);
  mlirBlockAppendOwnedOperation(funcBody, attnOp);

  MlirValue results[] = {mlirOperationGetResult(attnOp, 0),
                         mlirOperationGetResult(attnOp, 1)};
  MlirOperation retOp = createReturnOp(loc, 2, results);
  mlirBlockAppendOwnedOperation(funcBody, retOp);

  MlirOperation func =
      createFuncOp(ctx, loc, "\"test_attention_with_lse\"",
                   "(!migraphx.shaped<2x64x128xf16, 8192x128x1>, "
                   "!migraphx.shaped<2x128x256xf16, 32768x256x1>, "
                   "!migraphx.shaped<2x256x64xf16, 16384x64x1>) -> "
                   "(!migraphx.shaped<2x64x64xf16, 4096x64x1>, "
                   "!migraphx.shaped<2x64xf32, 64x1>)",
                   funcBodyRegion);
  mlirBlockInsertOwnedOperation(moduleBody, 0, func);

  // CHECK: migraphx.attention %arg0, %arg1, %arg2
  // CHECK: softmax_type = f32
  // CHECK: !migraphx.shaped<2x64x64xf16, 4096x64x1>
  // CHECK-SAME: !migraphx.shaped<2x64xf32, 64x1>
  verifyAndDump(moduleOp, "attention with LSE");
  mlirModuleDestroy(moduleOp);
}

// CHECK-LABEL: === Test: attention with softmaxType ===
static void testAttentionWithSoftmaxType(MlirContext ctx, MlirLocation loc) {
  fprintf(stderr, "=== Test: attention with softmaxType ===\n");
  MlirModule moduleOp = mlirModuleCreateEmpty(loc);
  MlirBlock moduleBody = mlirModuleGetBody(moduleOp);

  int64_t qDims[] = {2, 64, 128};
  int64_t qStrides[] = {8192, 128, 1};
  int64_t kDims[] = {2, 128, 256};
  int64_t kStrides[] = {32768, 256, 1};
  int64_t vDims[] = {2, 256, 64};
  int64_t vStrides[] = {16384, 64, 1};

  MlirType qType =
      rocmlirMIXRShapedTypeGet(3, qDims, qStrides, mlirF16TypeGet(ctx));
  MlirType kType =
      rocmlirMIXRShapedTypeGet(3, kDims, kStrides, mlirF16TypeGet(ctx));
  MlirType vType =
      rocmlirMIXRShapedTypeGet(3, vDims, vStrides, mlirF16TypeGet(ctx));

  MlirType funcBodyArgTypes[] = {qType, kType, vType};
  MlirLocation funcBodyLocs[] = {loc, loc, loc};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirBlock funcBody = mlirBlockCreate(3, funcBodyArgTypes, funcBodyLocs);
  mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  int64_t resultDims[] = {2, 64, 64};
  int64_t resultStrides[] = {4096, 64, 1};
  MlirType resultType = rocmlirMIXRShapedTypeGet(3, resultDims, resultStrides,
                                                 mlirF16TypeGet(ctx));

  MlirRegion emptyRegion = mlirRegionCreate();

  MlirOperation attnOp = rocmlirMIGraphXAttentionCreate(
      loc, mlirBlockGetArgument(funcBody, 0), mlirBlockGetArgument(funcBody, 1),
      mlirBlockGetArgument(funcBody, 2), 0, NULL, resultType, (MlirType){NULL},
      mlirF32TypeGet(ctx), emptyRegion, MLIR_MIGRAPHX_ATTENTION_NONE,
      (MlirValue){NULL}, (MlirValue){NULL}, 0, 0);
  mlirBlockAppendOwnedOperation(funcBody, attnOp);

  MlirValue attnResult = mlirOperationGetResult(attnOp, 0);
  MlirOperation retOp = createReturnOp(loc, 1, &attnResult);
  mlirBlockAppendOwnedOperation(funcBody, retOp);

  MlirOperation func =
      createFuncOp(ctx, loc, "\"test_attention_softmax_type\"",
                   "(!migraphx.shaped<2x64x128xf16, 8192x128x1>, "
                   "!migraphx.shaped<2x128x256xf16, 32768x256x1>, "
                   "!migraphx.shaped<2x256x64xf16, 16384x64x1>) -> "
                   "!migraphx.shaped<2x64x64xf16, 4096x64x1>",
                   funcBodyRegion);
  mlirBlockInsertOwnedOperation(moduleBody, 0, func);

  // CHECK: migraphx.attention %arg0, %arg1, %arg2
  // CHECK: softmax_type = f32
  verifyAndDump(moduleOp, "attention with softmaxType");
  mlirModuleDestroy(moduleOp);
}

// Helper to create a migraphx.add op
static MlirOperation createMIXRAddOp(MlirLocation loc, MlirValue lhs,
                                     MlirValue rhs, MlirType resultType) {
  MlirOperationState state = mlirOperationStateGet(
      mlirStringRefCreateFromCString("migraphx.add"), loc);
  MlirValue operands[] = {lhs, rhs};
  mlirOperationStateAddOperands(&state, 2, operands);
  mlirOperationStateAddResults(&state, 1, &resultType);
  return mlirOperationCreate(&state);
}

// Helper to create a migraphx.yield terminator with an optional value
static MlirOperation createMIXRYieldOp(MlirLocation loc, MlirValue value) {
  MlirOperationState state = mlirOperationStateGet(
      mlirStringRefCreateFromCString("migraphx.yield"), loc);
  if (value.ptr)
    mlirOperationStateAddOperands(&state, 1, &value);
  return mlirOperationCreate(&state);
}

// CHECK-LABEL: === Test: attention with preSoftmaxInputs ===
static void testAttentionWithPreSoftmaxInputs(MlirContext ctx,
                                              MlirLocation loc) {
  fprintf(stderr, "=== Test: attention with preSoftmaxInputs ===\n");
  MlirModule moduleOp = mlirModuleCreateEmpty(loc);
  MlirBlock moduleBody = mlirModuleGetBody(moduleOp);

  int64_t qDims[] = {2, 64, 128};
  int64_t qStrides[] = {8192, 128, 1};
  int64_t kDims[] = {2, 128, 256};
  int64_t kStrides[] = {32768, 256, 1};
  int64_t vDims[] = {2, 256, 64};
  int64_t vStrides[] = {16384, 64, 1};
  int64_t biasDims[] = {2, 64, 256};
  int64_t biasStrides[] = {16384, 256, 1};

  MlirType qType =
      rocmlirMIXRShapedTypeGet(3, qDims, qStrides, mlirF16TypeGet(ctx));
  MlirType kType =
      rocmlirMIXRShapedTypeGet(3, kDims, kStrides, mlirF16TypeGet(ctx));
  MlirType vType =
      rocmlirMIXRShapedTypeGet(3, vDims, vStrides, mlirF16TypeGet(ctx));
  MlirType biasType =
      rocmlirMIXRShapedTypeGet(3, biasDims, biasStrides, mlirF16TypeGet(ctx));

  MlirType funcBodyArgTypes[] = {qType, kType, vType, biasType};
  MlirLocation funcBodyLocs[] = {loc, loc, loc, loc};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirBlock funcBody = mlirBlockCreate(4, funcBodyArgTypes, funcBodyLocs);
  mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  int64_t qkDims[] = {2, 64, 256};
  int64_t qkStrides[] = {16384, 256, 1};
  MlirType qkType =
      rocmlirMIXRShapedTypeGet(3, qkDims, qkStrides, mlirF16TypeGet(ctx));

  int64_t resultDims[] = {2, 64, 64};
  int64_t resultStrides[] = {4096, 64, 1};
  MlirType resultType = rocmlirMIXRShapedTypeGet(3, resultDims, resultStrides,
                                                 mlirF16TypeGet(ctx));

  // Build a preSoftmaxBody: migraphx.add(%qk, %bias)
  MlirRegion bodyRegion = mlirRegionCreate();
  MlirType bodyArgTypes[] = {qkType, biasType};
  MlirLocation bodyArgLocs[] = {loc, loc};
  MlirBlock bodyBlock = mlirBlockCreate(2, bodyArgTypes, bodyArgLocs);
  mlirRegionAppendOwnedBlock(bodyRegion, bodyBlock);

  MlirValue bbArg0 = mlirBlockGetArgument(bodyBlock, 0);
  MlirValue bbArg1 = mlirBlockGetArgument(bodyBlock, 1);
  MlirOperation addOp = createMIXRAddOp(loc, bbArg0, bbArg1, qkType);
  mlirBlockAppendOwnedOperation(bodyBlock, addOp);
  MlirValue addResult = mlirOperationGetResult(addOp, 0);
  MlirOperation yieldOp = createMIXRYieldOp(loc, addResult);
  mlirBlockAppendOwnedOperation(bodyBlock, yieldOp);

  MlirValue biasValue = mlirBlockGetArgument(funcBody, 3);
  MlirOperation attnOp = rocmlirMIGraphXAttentionCreate(
      loc, mlirBlockGetArgument(funcBody, 0), mlirBlockGetArgument(funcBody, 1),
      mlirBlockGetArgument(funcBody, 2), 1, &biasValue, resultType,
      (MlirType){NULL}, (MlirType){NULL}, bodyRegion,
      MLIR_MIGRAPHX_ATTENTION_NONE, (MlirValue){NULL}, (MlirValue){NULL}, 0, 0);
  mlirBlockAppendOwnedOperation(funcBody, attnOp);

  MlirValue attnResult = mlirOperationGetResult(attnOp, 0);
  MlirOperation retOp = createReturnOp(loc, 1, &attnResult);
  mlirBlockAppendOwnedOperation(funcBody, retOp);

  MlirOperation func =
      createFuncOp(ctx, loc, "\"test_attention_pre_softmax\"",
                   "(!migraphx.shaped<2x64x128xf16, 8192x128x1>, "
                   "!migraphx.shaped<2x128x256xf16, 32768x256x1>, "
                   "!migraphx.shaped<2x256x64xf16, 16384x64x1>, "
                   "!migraphx.shaped<2x64x256xf16, 16384x256x1>) -> "
                   "!migraphx.shaped<2x64x64xf16, 4096x64x1>",
                   funcBodyRegion);
  mlirBlockInsertOwnedOperation(moduleBody, 0, func);

  // CHECK: migraphx.attention
  // CHECK: pre_softmax_inputs
  // CHECK: migraphx.add
  verifyAndDump(moduleOp, "attention with preSoftmaxInputs");
  mlirModuleDestroy(moduleOp);
}

// CHECK-LABEL: === Test: attention with preSoftmaxBody and LSE ===
static void testAttentionWithBodyAndLse(MlirContext ctx, MlirLocation loc) {
  fprintf(stderr, "=== Test: attention with preSoftmaxBody and LSE ===\n");
  MlirModule moduleOp = mlirModuleCreateEmpty(loc);
  MlirBlock moduleBody = mlirModuleGetBody(moduleOp);

  int64_t qDims[] = {2, 64, 128};
  int64_t qStrides[] = {8192, 128, 1};
  int64_t kDims[] = {2, 128, 256};
  int64_t kStrides[] = {32768, 256, 1};
  int64_t vDims[] = {2, 256, 64};
  int64_t vStrides[] = {16384, 64, 1};
  int64_t biasDims[] = {2, 64, 256};
  int64_t biasStrides[] = {16384, 256, 1};

  MlirType f16 = mlirF16TypeGet(ctx);
  MlirType f32 = mlirF32TypeGet(ctx);

  MlirType qType = rocmlirMIXRShapedTypeGet(3, qDims, qStrides, f16);
  MlirType kType = rocmlirMIXRShapedTypeGet(3, kDims, kStrides, f16);
  MlirType vType = rocmlirMIXRShapedTypeGet(3, vDims, vStrides, f16);
  MlirType biasType = rocmlirMIXRShapedTypeGet(3, biasDims, biasStrides, f16);

  MlirType funcBodyArgTypes[] = {qType, kType, vType, biasType};
  MlirLocation funcBodyLocs[] = {loc, loc, loc, loc};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirBlock funcBody = mlirBlockCreate(4, funcBodyArgTypes, funcBodyLocs);
  mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  int64_t resultDims[] = {2, 64, 64};
  int64_t resultStrides2[] = {4096, 64, 1};
  MlirType resultType =
      rocmlirMIXRShapedTypeGet(3, resultDims, resultStrides2, f16);

  int64_t lseDims[] = {2, 64};
  int64_t lseStrides2[] = {64, 1};
  MlirType lseType = rocmlirMIXRShapedTypeGet(2, lseDims, lseStrides2, f32);

  int64_t qkDims[] = {2, 64, 256};
  int64_t qkStrides[] = {16384, 256, 1};
  MlirType qkType = rocmlirMIXRShapedTypeGet(3, qkDims, qkStrides, f16);

  // Build preSoftmaxBody: migraphx.add(%qk, %bias)
  MlirRegion bodyRegion = mlirRegionCreate();
  MlirType bodyArgTypes[] = {qkType, biasType};
  MlirLocation bodyArgLocs[] = {loc, loc};
  MlirBlock bodyBlock = mlirBlockCreate(2, bodyArgTypes, bodyArgLocs);
  mlirRegionAppendOwnedBlock(bodyRegion, bodyBlock);

  MlirValue bbArg0 = mlirBlockGetArgument(bodyBlock, 0);
  MlirValue bbArg1 = mlirBlockGetArgument(bodyBlock, 1);
  MlirOperation addOp = createMIXRAddOp(loc, bbArg0, bbArg1, qkType);
  mlirBlockAppendOwnedOperation(bodyBlock, addOp);
  MlirValue addResult = mlirOperationGetResult(addOp, 0);

  MlirOperation yieldOp = createMIXRYieldOp(loc, addResult);
  mlirBlockAppendOwnedOperation(bodyBlock, yieldOp);

  MlirValue biasValue = mlirBlockGetArgument(funcBody, 3);
  MlirOperation attnOp = rocmlirMIGraphXAttentionCreate(
      loc, mlirBlockGetArgument(funcBody, 0), mlirBlockGetArgument(funcBody, 1),
      mlirBlockGetArgument(funcBody, 2), 1, &biasValue, resultType, lseType,
      mlirF32TypeGet(ctx), bodyRegion, MLIR_MIGRAPHX_ATTENTION_NONE,
      (MlirValue){NULL}, (MlirValue){NULL}, 0, 0);
  mlirBlockAppendOwnedOperation(funcBody, attnOp);

  MlirValue attnResults[] = {mlirOperationGetResult(attnOp, 0),
                             mlirOperationGetResult(attnOp, 1)};
  MlirOperation retOp = createReturnOp(loc, 2, attnResults);
  mlirBlockAppendOwnedOperation(funcBody, retOp);

  MlirOperation func =
      createFuncOp(ctx, loc, "\"test_attention_body_and_lse\"",
                   "(!migraphx.shaped<2x64x128xf16, 8192x128x1>, "
                   "!migraphx.shaped<2x128x256xf16, 32768x256x1>, "
                   "!migraphx.shaped<2x256x64xf16, 16384x64x1>, "
                   "!migraphx.shaped<2x64x256xf16, 16384x256x1>) -> "
                   "(!migraphx.shaped<2x64x64xf16, 4096x64x1>, "
                   "!migraphx.shaped<2x64xf32, 64x1>)",
                   funcBodyRegion);
  mlirBlockInsertOwnedOperation(moduleBody, 0, func);

  // CHECK: migraphx.attention
  // CHECK: pre_softmax_inputs
  // CHECK: migraphx.add
  // CHECK: softmax_type = f32
  // CHECK: !migraphx.shaped<2x64x64xf16, 4096x64x1>
  // CHECK-SAME: !migraphx.shaped<2x64xf32, 64x1>
  verifyAndDump(moduleOp, "attention with preSoftmaxBody and LSE");
  mlirModuleDestroy(moduleOp);
}

// CHECK-LABEL: === Test: attention causal ===
static void testAttentionCausal(MlirContext ctx, MlirLocation loc) {
  fprintf(stderr, "=== Test: attention causal ===\n");
  MlirModule moduleOp = mlirModuleCreateEmpty(loc);
  MlirBlock moduleBody = mlirModuleGetBody(moduleOp);

  int64_t qDims[] = {2, 64, 128};
  int64_t qStrides[] = {8192, 128, 1};
  int64_t kDims[] = {2, 128, 256};
  int64_t kStrides[] = {32768, 256, 1};
  int64_t vDims[] = {2, 256, 64};
  int64_t vStrides[] = {16384, 64, 1};

  MlirType qType =
      rocmlirMIXRShapedTypeGet(3, qDims, qStrides, mlirF16TypeGet(ctx));
  MlirType kType =
      rocmlirMIXRShapedTypeGet(3, kDims, kStrides, mlirF16TypeGet(ctx));
  MlirType vType =
      rocmlirMIXRShapedTypeGet(3, vDims, vStrides, mlirF16TypeGet(ctx));

  MlirType funcBodyArgTypes[] = {qType, kType, vType};
  MlirLocation funcBodyLocs[] = {loc, loc, loc};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirBlock funcBody = mlirBlockCreate(3, funcBodyArgTypes, funcBodyLocs);
  mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  int64_t resultDims[] = {2, 64, 64};
  int64_t resultStrides[] = {4096, 64, 1};
  MlirType resultType = rocmlirMIXRShapedTypeGet(3, resultDims, resultStrides,
                                                 mlirF16TypeGet(ctx));

  MlirRegion emptyRegion = mlirRegionCreate();

  MlirOperation attnOp = rocmlirMIGraphXAttentionCreate(
      loc, mlirBlockGetArgument(funcBody, 0), mlirBlockGetArgument(funcBody, 1),
      mlirBlockGetArgument(funcBody, 2), 0, NULL, resultType, (MlirType){NULL},
      (MlirType){NULL}, emptyRegion, MLIR_MIGRAPHX_ATTENTION_CAUSAL,
      (MlirValue){NULL}, (MlirValue){NULL}, 0, 0);
  mlirBlockAppendOwnedOperation(funcBody, attnOp);

  MlirValue attnResult = mlirOperationGetResult(attnOp, 0);
  MlirOperation retOp = createReturnOp(loc, 1, &attnResult);
  mlirBlockAppendOwnedOperation(funcBody, retOp);

  MlirOperation func =
      createFuncOp(ctx, loc, "\"test_attention_causal\"",
                   "(!migraphx.shaped<2x64x128xf16, 8192x128x1>, "
                   "!migraphx.shaped<2x128x256xf16, 32768x256x1>, "
                   "!migraphx.shaped<2x256x64xf16, 16384x64x1>) -> "
                   "!migraphx.shaped<2x64x64xf16, 4096x64x1>",
                   funcBodyRegion);
  mlirBlockInsertOwnedOperation(moduleBody, 0, func);

  // CHECK: migraphx.attention %arg0, %arg1, %arg2
  // CHECK: features = causal
  verifyAndDump(moduleOp, "attention causal");
  mlirModuleDestroy(moduleOp);
}

// CHECK-LABEL: === Test: attention kvcache ===
static void testAttentionKVCache(MlirContext ctx, MlirLocation loc) {
  fprintf(stderr, "=== Test: attention kvcache ===\n");
  MlirModule moduleOp = mlirModuleCreateEmpty(loc);
  MlirBlock moduleBody = mlirModuleGetBody(moduleOp);

  int64_t qDims[] = {1, 32, 1, 128};
  int64_t qStrides[] = {4096, 128, 128, 1};
  int64_t kDims[] = {1, 32, 128, 1024};
  int64_t kStrides[] = {4194304, 131072, 1024, 1};
  int64_t vDims[] = {1, 32, 1024, 128};
  int64_t vStrides[] = {4194304, 131072, 128, 1};
  int64_t seqLenDims[] = {1, 32};
  int64_t seqLenStrides[] = {32, 1};

  MlirType f16 = mlirF16TypeGet(ctx);
  MlirType i32 = mlirIntegerTypeGet(ctx, 32);

  MlirType qType = rocmlirMIXRShapedTypeGet(4, qDims, qStrides, f16);
  MlirType kType = rocmlirMIXRShapedTypeGet(4, kDims, kStrides, f16);
  MlirType vType = rocmlirMIXRShapedTypeGet(4, vDims, vStrides, f16);
  MlirType seqLenType =
      rocmlirMIXRShapedTypeGet(2, seqLenDims, seqLenStrides, i32);

  MlirType funcBodyArgTypes[] = {qType, kType, vType, seqLenType};
  MlirLocation funcBodyLocs[] = {loc, loc, loc, loc};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirBlock funcBody = mlirBlockCreate(4, funcBodyArgTypes, funcBodyLocs);
  mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  int64_t resultDims[] = {1, 32, 1, 128};
  int64_t resultStrides[] = {4096, 128, 128, 1};
  MlirType resultType =
      rocmlirMIXRShapedTypeGet(4, resultDims, resultStrides, f16);

  MlirRegion emptyRegion = mlirRegionCreate();

  MlirOperation attnOp = rocmlirMIGraphXAttentionCreate(
      loc, mlirBlockGetArgument(funcBody, 0), mlirBlockGetArgument(funcBody, 1),
      mlirBlockGetArgument(funcBody, 2), 0, NULL, resultType, (MlirType){NULL},
      (MlirType){NULL}, emptyRegion, MLIR_MIGRAPHX_ATTENTION_KVCACHE,
      mlirBlockGetArgument(funcBody, 3), (MlirValue){NULL}, 0, 0);
  mlirBlockAppendOwnedOperation(funcBody, attnOp);

  MlirValue attnResult = mlirOperationGetResult(attnOp, 0);
  MlirOperation retOp = createReturnOp(loc, 1, &attnResult);
  mlirBlockAppendOwnedOperation(funcBody, retOp);

  MlirOperation func = createFuncOp(
      ctx, loc, "\"test_attention_kvcache\"",
      "(!migraphx.shaped<1x32x1x128xf16, 4096x128x128x1>, "
      "!migraphx.shaped<1x32x128x1024xf16, 4194304x131072x1024x1>, "
      "!migraphx.shaped<1x32x1024x128xf16, 4194304x131072x128x1>, "
      "!migraphx.shaped<1x32xi32, 32x1>) -> "
      "!migraphx.shaped<1x32x1x128xf16, 4096x128x128x1>",
      funcBodyRegion);
  mlirBlockInsertOwnedOperation(moduleBody, 0, func);

  // CHECK: migraphx.attention %arg0, %arg1, %arg2 current_seq_len
  // CHECK: features = kvcache
  verifyAndDump(moduleOp, "attention KV-cache");
  mlirModuleDestroy(moduleOp);
}

// CHECK-LABEL: === Test: attention splitKV ===
static void testAttentionSplitKV(MlirContext ctx, MlirLocation loc) {
  fprintf(stderr, "=== Test: attention splitKV ===\n");
  MlirModule moduleOp = mlirModuleCreateEmpty(loc);
  MlirBlock moduleBody = mlirModuleGetBody(moduleOp);

  int64_t qDims[] = {1, 32, 1, 128};
  int64_t qStrides[] = {4096, 128, 128, 1};
  int64_t kDims[] = {1, 32, 128, 1024};
  int64_t kStrides[] = {4194304, 131072, 1024, 1};
  int64_t vDims[] = {1, 32, 1024, 128};
  int64_t vStrides[] = {4194304, 131072, 128, 1};

  MlirType f16 = mlirF16TypeGet(ctx);
  MlirType f32 = mlirF32TypeGet(ctx);

  MlirType qType = rocmlirMIXRShapedTypeGet(4, qDims, qStrides, f16);
  MlirType kType = rocmlirMIXRShapedTypeGet(4, kDims, kStrides, f16);
  MlirType vType = rocmlirMIXRShapedTypeGet(4, vDims, vStrides, f16);

  MlirType funcBodyArgTypes[] = {qType, kType, vType};
  MlirLocation funcBodyLocs[] = {loc, loc, loc};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirBlock funcBody = mlirBlockCreate(3, funcBodyArgTypes, funcBodyLocs);
  mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  int64_t resultDims[] = {1, 32, 2, 1, 128};
  int64_t resultStrides[] = {8192, 256, 128, 128, 1};
  MlirType resultType =
      rocmlirMIXRShapedTypeGet(5, resultDims, resultStrides, f16);

  int64_t lseDims[] = {1, 32, 2, 1};
  int64_t lseStrides[] = {64, 2, 1, 1};
  MlirType lseType = rocmlirMIXRShapedTypeGet(4, lseDims, lseStrides, f32);

  MlirRegion emptyRegion = mlirRegionCreate();

  // softmax_type = f32 is required because V is f16 and LSE is f32; the
  // verifier requires lse element type to match the effective softmax
  // type.
  MlirOperation attnOp = rocmlirMIGraphXAttentionCreate(
      loc, mlirBlockGetArgument(funcBody, 0), mlirBlockGetArgument(funcBody, 1),
      mlirBlockGetArgument(funcBody, 2), 0, NULL, resultType, lseType,
      mlirF32TypeGet(ctx), emptyRegion, MLIR_MIGRAPHX_ATTENTION_SPLITKV,
      (MlirValue){NULL}, (MlirValue){NULL},
      /*splitKV=*/2, 0);
  mlirBlockAppendOwnedOperation(funcBody, attnOp);

  MlirValue results[] = {mlirOperationGetResult(attnOp, 0),
                         mlirOperationGetResult(attnOp, 1)};
  MlirOperation retOp = createReturnOp(loc, 2, results);
  mlirBlockAppendOwnedOperation(funcBody, retOp);

  MlirOperation func = createFuncOp(
      ctx, loc, "\"test_attention_splitkv\"",
      "(!migraphx.shaped<1x32x1x128xf16, 4096x128x128x1>, "
      "!migraphx.shaped<1x32x128x1024xf16, 4194304x131072x1024x1>, "
      "!migraphx.shaped<1x32x1024x128xf16, 4194304x131072x128x1>) -> "
      "(!migraphx.shaped<1x32x2x1x128xf16, 8192x256x128x128x1>, "
      "!migraphx.shaped<1x32x2x1xf32, 64x2x1x1>)",
      funcBodyRegion);
  mlirBlockInsertOwnedOperation(moduleBody, 0, func);

  // CHECK: migraphx.attention %arg0, %arg1, %arg2
  // CHECK: softmax_type = f32
  // CHECK-SAME: features = splitkv
  // CHECK-SAME: splitKV = 2
  verifyAndDump(moduleOp, "attention splitKV");
  mlirModuleDestroy(moduleOp);
}

// CHECK-LABEL: === Test: attention prefix offset ===
static void testAttentionPrefixOffset(MlirContext ctx, MlirLocation loc) {
  fprintf(stderr, "=== Test: attention prefix offset ===\n");
  MlirModule moduleOp = mlirModuleCreateEmpty(loc);
  MlirBlock moduleBody = mlirModuleGetBody(moduleOp);

  int64_t qDims[] = {1, 32, 1, 128};
  int64_t qStrides[] = {4096, 128, 128, 1};
  int64_t kDims[] = {1, 32, 128, 1024};
  int64_t kStrides[] = {4194304, 131072, 1024, 1};
  int64_t vDims[] = {1, 32, 1024, 128};
  int64_t vStrides[] = {4194304, 131072, 128, 1};
  int64_t prefixDims[] = {1, 32};
  int64_t prefixStrides[] = {32, 1};

  MlirType f16 = mlirF16TypeGet(ctx);
  MlirType i32 = mlirIntegerTypeGet(ctx, 32);

  MlirType qType = rocmlirMIXRShapedTypeGet(4, qDims, qStrides, f16);
  MlirType kType = rocmlirMIXRShapedTypeGet(4, kDims, kStrides, f16);
  MlirType vType = rocmlirMIXRShapedTypeGet(4, vDims, vStrides, f16);
  MlirType prefixType =
      rocmlirMIXRShapedTypeGet(2, prefixDims, prefixStrides, i32);

  MlirType funcBodyArgTypes[] = {qType, kType, vType, prefixType};
  MlirLocation funcBodyLocs[] = {loc, loc, loc, loc};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirBlock funcBody = mlirBlockCreate(4, funcBodyArgTypes, funcBodyLocs);
  mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  int64_t resultDims[] = {1, 32, 1, 128};
  int64_t resultStrides[] = {4096, 128, 128, 1};
  MlirType resultType =
      rocmlirMIXRShapedTypeGet(4, resultDims, resultStrides, f16);

  MlirRegion emptyRegion = mlirRegionCreate();

  MlirOperation attnOp = rocmlirMIGraphXAttentionCreate(
      loc, mlirBlockGetArgument(funcBody, 0), mlirBlockGetArgument(funcBody, 1),
      mlirBlockGetArgument(funcBody, 2), 0, NULL, resultType, (MlirType){NULL},
      (MlirType){NULL}, emptyRegion,
      MLIR_MIGRAPHX_ATTENTION_CAUSAL | MLIR_MIGRAPHX_ATTENTION_PREFIX_OFFSET,
      (MlirValue){NULL}, mlirBlockGetArgument(funcBody, 3), 0, 0);
  mlirBlockAppendOwnedOperation(funcBody, attnOp);

  MlirValue attnResult = mlirOperationGetResult(attnOp, 0);
  MlirOperation retOp = createReturnOp(loc, 1, &attnResult);
  mlirBlockAppendOwnedOperation(funcBody, retOp);

  MlirOperation func = createFuncOp(
      ctx, loc, "\"test_attention_prefix_offset\"",
      "(!migraphx.shaped<1x32x1x128xf16, 4096x128x128x1>, "
      "!migraphx.shaped<1x32x128x1024xf16, 4194304x131072x1024x1>, "
      "!migraphx.shaped<1x32x1024x128xf16, 4194304x131072x128x1>, "
      "!migraphx.shaped<1x32xi32, 32x1>) -> "
      "!migraphx.shaped<1x32x1x128xf16, 4096x128x128x1>",
      funcBodyRegion);
  mlirBlockInsertOwnedOperation(moduleBody, 0, func);

  // CHECK: migraphx.attention %arg0, %arg1, %arg2 prefix_offset
  // CHECK: features = "causal|prefix_offset"
  verifyAndDump(moduleOp, "attention prefix-offset");
  mlirModuleDestroy(moduleOp);
}

// CHECK-LABEL: === Test: attention sliding window ===
static void testAttentionSlidingWindow(MlirContext ctx, MlirLocation loc) {
  fprintf(stderr, "=== Test: attention sliding window ===\n");
  MlirModule moduleOp = mlirModuleCreateEmpty(loc);
  MlirBlock moduleBody = mlirModuleGetBody(moduleOp);

  int64_t qDims[] = {2, 64, 128}, qStr[] = {8192, 128, 1};
  int64_t kDims[] = {2, 128, 256}, kStr[] = {32768, 256, 1};
  int64_t vDims[] = {2, 256, 64}, vStr[] = {16384, 64, 1};
  int64_t slDims[] = {2}, slStr[] = {1};
  MlirType f16 = mlirF16TypeGet(ctx);
  MlirType i32 = mlirIntegerTypeGet(ctx, 32);
  MlirType qT = rocmlirMIXRShapedTypeGet(3, qDims, qStr, f16);
  MlirType kT = rocmlirMIXRShapedTypeGet(3, kDims, kStr, f16);
  MlirType vT = rocmlirMIXRShapedTypeGet(3, vDims, vStr, f16);
  MlirType slT = rocmlirMIXRShapedTypeGet(1, slDims, slStr, i32);
  MlirType funcArgs[] = {qT, kT, vT, slT};
  MlirLocation locs[] = {loc, loc, loc, loc};
  MlirRegion funcReg = mlirRegionCreate();
  MlirBlock funcBody = mlirBlockCreate(4, funcArgs, locs);
  mlirRegionAppendOwnedBlock(funcReg, funcBody);

  int64_t rDims[] = {2, 64, 64}, rStr[] = {4096, 64, 1};
  MlirType rT = rocmlirMIXRShapedTypeGet(3, rDims, rStr, f16);
  MlirRegion emptyReg = mlirRegionCreate();

  MlirOperation attnOp = rocmlirMIGraphXAttentionCreate(
      loc, mlirBlockGetArgument(funcBody, 0), mlirBlockGetArgument(funcBody, 1),
      mlirBlockGetArgument(funcBody, 2), 0, NULL, rT, (MlirType){NULL},
      (MlirType){NULL}, emptyReg,
      MLIR_MIGRAPHX_ATTENTION_KVCACHE | MLIR_MIGRAPHX_ATTENTION_SLIDING_WINDOW,
      mlirBlockGetArgument(funcBody, 3), (MlirValue){NULL}, 0, 64);
  mlirBlockAppendOwnedOperation(funcBody, attnOp);
  MlirValue res = mlirOperationGetResult(attnOp, 0);
  MlirOperation retOp = createReturnOp(loc, 1, &res);
  mlirBlockAppendOwnedOperation(funcBody, retOp);

  MlirOperation func =
      createFuncOp(ctx, loc, "\"test_sliding_window\"",
                   "(!migraphx.shaped<2x64x128xf16, 8192x128x1>, "
                   "!migraphx.shaped<2x128x256xf16, 32768x256x1>, "
                   "!migraphx.shaped<2x256x64xf16, 16384x64x1>, "
                   "!migraphx.shaped<2xi32, 1>) -> "
                   "!migraphx.shaped<2x64x64xf16, 4096x64x1>",
                   funcReg);
  mlirBlockInsertOwnedOperation(moduleBody, 0, func);

  // CHECK: migraphx.attention
  // CHECK: slidingWindowSize = 64
  verifyAndDump(moduleOp, "attention sliding-window");
  mlirModuleDestroy(moduleOp);
}

// CHECK-LABEL: === Test: attention rejects invalid inputs ===
//
// Pin the contract that rocmlirMIGraphXAttentionCreate enforces in both
// debug and release builds: it must return a null MlirOperation (not
// crash on a NULL deref, not silently produce a half-built op) when the
// caller violates one of the documented input requirements. Each
// sub-case here keeps every other argument valid so the failure isolates
// to the one being exercised, and the FileCheck lines below the function
// pin the exact stderr diagnostic the C API now emits via llvm::errs().
// If a future refactor changes the diagnostic wording or compiles a
// check out, this test fails immediately rather than waiting for a
// downstream caller to crash.
static void testAttentionRejectsInvalidInputs(MlirContext ctx,
                                              MlirLocation loc) {
  fprintf(stderr, "=== Test: attention rejects invalid inputs ===\n");

  // Build valid Q/K/V block args once and reuse them in every sub-case.
  // Two extra i32 batch args (slLen, prefix) act as stand-in
  // currentSeqLen / prefixOffset values for the orphan-operand cases.
  // No module / func is needed because every call below returns null and
  // never constructs an op to append; the scratch region just owns the
  // block whose arguments we hand to the builder as the "valid" Q/K/V.
  int64_t qDims[] = {2, 64, 128}, qStrides[] = {8192, 128, 1};
  int64_t kDims[] = {2, 128, 256}, kStrides[] = {32768, 256, 1};
  int64_t vDims[] = {2, 256, 64}, vStrides[] = {16384, 64, 1};
  int64_t rDims[] = {2, 64, 64}, rStrides[] = {4096, 64, 1};
  int64_t slDims[] = {2}, slStrides[] = {1};
  MlirType f16 = mlirF16TypeGet(ctx);
  MlirType i32 = mlirIntegerTypeGet(ctx, 32);
  MlirType qType = rocmlirMIXRShapedTypeGet(3, qDims, qStrides, f16);
  MlirType kType = rocmlirMIXRShapedTypeGet(3, kDims, kStrides, f16);
  MlirType vType = rocmlirMIXRShapedTypeGet(3, vDims, vStrides, f16);
  MlirType rType = rocmlirMIXRShapedTypeGet(3, rDims, rStrides, f16);
  MlirType slType = rocmlirMIXRShapedTypeGet(1, slDims, slStrides, i32);

  MlirType argTypes[] = {qType, kType, vType, slType, slType};
  MlirLocation argLocs[] = {loc, loc, loc, loc, loc};
  MlirRegion scratchRegion = mlirRegionCreate();
  MlirBlock scratchBlock = mlirBlockCreate(5, argTypes, argLocs);
  mlirRegionAppendOwnedBlock(scratchRegion, scratchBlock);
  MlirValue q = mlirBlockGetArgument(scratchBlock, 0);
  MlirValue k = mlirBlockGetArgument(scratchBlock, 1);
  MlirValue v = mlirBlockGetArgument(scratchBlock, 2);
  MlirValue slLen = mlirBlockGetArgument(scratchBlock, 3);
  MlirValue prefix = mlirBlockGetArgument(scratchBlock, 4);

  // Case 1: null queries operand.
  // CHECK: rocmlirMIGraphXAttentionCreate: queries operand is required
  MlirRegion body1 = mlirRegionCreate();
  MlirOperation op1 = rocmlirMIGraphXAttentionCreate(
      loc, (MlirValue){NULL}, k, v, 0, NULL, rType, (MlirType){NULL},
      (MlirType){NULL}, body1, MLIR_MIGRAPHX_ATTENTION_NONE, (MlirValue){NULL},
      (MlirValue){NULL}, 0, 0);
  if (!mlirOperationIsNull(op1)) {
    fprintf(stderr, "FAIL: null queries should return null op\n");
    exit(1);
  }
  // The builder bailed before mlirOperationStateAddOwnedRegions, so the
  // region is still ours to free.
  mlirRegionDestroy(body1);

  // Case 2: null preSoftmaxBody. The diagnostic includes the
  // "use mlirRegionCreate()" hint so callers don't have to chase the
  // header doc to find the right idiom.
  // CHECK: rocmlirMIGraphXAttentionCreate: preSoftmaxBody region is required
  // CHECK-SAME: use mlirRegionCreate() for an empty body
  MlirOperation op2 = rocmlirMIGraphXAttentionCreate(
      loc, q, k, v, 0, NULL, rType, (MlirType){NULL}, (MlirType){NULL},
      (MlirRegion){NULL}, MLIR_MIGRAPHX_ATTENTION_NONE, (MlirValue){NULL},
      (MlirValue){NULL}, 0, 0);
  if (!mlirOperationIsNull(op2)) {
    fprintf(stderr, "FAIL: null preSoftmaxBody should return null op\n");
    exit(1);
  }

  // Case 3: negative splitKV (representative invalid scalar). Pre-fix
  // this used to silently get dropped because the writer guarded it
  // with `splitKV > 1`; now it's an explicit reject.
  // CHECK: rocmlirMIGraphXAttentionCreate: splitKV must be non-negative
  MlirRegion body3 = mlirRegionCreate();
  MlirOperation op3 = rocmlirMIGraphXAttentionCreate(
      loc, q, k, v, 0, NULL, rType, (MlirType){NULL}, (MlirType){NULL}, body3,
      MLIR_MIGRAPHX_ATTENTION_NONE, (MlirValue){NULL}, (MlirValue){NULL}, -1,
      0);
  if (!mlirOperationIsNull(op3)) {
    fprintf(stderr, "FAIL: negative splitKV should return null op\n");
    exit(1);
  }
  mlirRegionDestroy(body3);

  // Case 4: null location. The builder dereferences location via
  // mlirLocationGetContext / mlirOperationStateGet / the YieldOp builder
  // unconditionally below the input-validation block, so a default-
  // initialised MlirLocation would crash in release builds the same way
  // a default-initialised MlirRegion used to before Case 2 was added.
  // CHECK: rocmlirMIGraphXAttentionCreate: location is required
  MlirRegion body4 = mlirRegionCreate();
  MlirOperation op4 = rocmlirMIGraphXAttentionCreate(
      (MlirLocation){NULL}, q, k, v, 0, NULL, rType, (MlirType){NULL},
      (MlirType){NULL}, body4, MLIR_MIGRAPHX_ATTENTION_NONE, (MlirValue){NULL},
      (MlirValue){NULL}, 0, 0);
  if (!mlirOperationIsNull(op4)) {
    fprintf(stderr, "FAIL: null location should return null op\n");
    exit(1);
  }
  mlirRegionDestroy(body4);

  // Cases 5-8 mirror the verifier's orphan-attr / orphan-operand checks
  // (verifyOrphanAttr / verifyOrphanOperand in MIGraphX.cpp) at the API
  // boundary so the diagnostic happens before the op is built. Each
  // sub-case sets exactly one attr or operand without the corresponding
  // feature flag, so the expected error mentions only that pairing.

  // Case 5: splitKV without splitkv feature.
  // clang-format off
  // CHECK: rocmlirMIGraphXAttentionCreate: 'splitKV' attribute requires feature 'splitkv'
  // clang-format on
  MlirRegion body5 = mlirRegionCreate();
  MlirOperation op5 = rocmlirMIGraphXAttentionCreate(
      loc, q, k, v, 0, NULL, rType, (MlirType){NULL}, (MlirType){NULL}, body5,
      MLIR_MIGRAPHX_ATTENTION_NONE, (MlirValue){NULL}, (MlirValue){NULL}, 4, 0);
  if (!mlirOperationIsNull(op5)) {
    fprintf(stderr, "FAIL: orphan splitKV should return null op\n");
    exit(1);
  }
  mlirRegionDestroy(body5);

  // Case 6: slidingWindowSize without sliding_window feature.
  // clang-format off
  // CHECK: rocmlirMIGraphXAttentionCreate: 'slidingWindowSize' attribute requires feature 'sliding_window'
  // clang-format on
  MlirRegion body6 = mlirRegionCreate();
  MlirOperation op6 = rocmlirMIGraphXAttentionCreate(
      loc, q, k, v, 0, NULL, rType, (MlirType){NULL}, (MlirType){NULL}, body6,
      MLIR_MIGRAPHX_ATTENTION_NONE, (MlirValue){NULL}, (MlirValue){NULL}, 0,
      32);
  if (!mlirOperationIsNull(op6)) {
    fprintf(stderr, "FAIL: orphan slidingWindowSize should return null op\n");
    exit(1);
  }
  mlirRegionDestroy(body6);

  // Case 7: currentSeqLen without kvcache feature.
  // clang-format off
  // CHECK: rocmlirMIGraphXAttentionCreate: 'currentSeqLen' operand requires feature 'kvcache'
  // clang-format on
  MlirRegion body7 = mlirRegionCreate();
  MlirOperation op7 = rocmlirMIGraphXAttentionCreate(
      loc, q, k, v, 0, NULL, rType, (MlirType){NULL}, (MlirType){NULL}, body7,
      MLIR_MIGRAPHX_ATTENTION_NONE, slLen, (MlirValue){NULL}, 0, 0);
  if (!mlirOperationIsNull(op7)) {
    fprintf(stderr, "FAIL: orphan currentSeqLen should return null op\n");
    exit(1);
  }
  mlirRegionDestroy(body7);

  // Case 8: prefixOffset without prefix_offset feature.
  // clang-format off
  // CHECK: rocmlirMIGraphXAttentionCreate: 'prefixOffset' operand requires feature 'prefix_offset'
  // clang-format on
  MlirRegion body8 = mlirRegionCreate();
  MlirOperation op8 = rocmlirMIGraphXAttentionCreate(
      loc, q, k, v, 0, NULL, rType, (MlirType){NULL}, (MlirType){NULL}, body8,
      MLIR_MIGRAPHX_ATTENTION_NONE, (MlirValue){NULL}, prefix, 0, 0);
  if (!mlirOperationIsNull(op8)) {
    fprintf(stderr, "FAIL: orphan prefixOffset should return null op\n");
    exit(1);
  }
  mlirRegionDestroy(body8);

  // Cases 9-11 cover the remaining contract violations documented in
  // mlir-c/Dialect/MIGraphX.h's Doxygen for rocmlirMIGraphXAttentionCreate
  // but not yet pinned by a test: negative count, NULL inputs array when
  // count > 0, and null resultType. These are the last documented reject
  // paths still relying on inspection alone.

  // Case 9: negative numPreSoftmaxInputs.
  // clang-format off
  // CHECK: rocmlirMIGraphXAttentionCreate: numPreSoftmaxInputs must be non-negative
  // clang-format on
  MlirRegion body9 = mlirRegionCreate();
  MlirOperation op9 = rocmlirMIGraphXAttentionCreate(
      loc, q, k, v, -1, NULL, rType, (MlirType){NULL}, (MlirType){NULL}, body9,
      MLIR_MIGRAPHX_ATTENTION_NONE, (MlirValue){NULL}, (MlirValue){NULL}, 0, 0);
  if (!mlirOperationIsNull(op9)) {
    fprintf(stderr,
            "FAIL: negative numPreSoftmaxInputs should return null op\n");
    exit(1);
  }
  mlirRegionDestroy(body9);

  // Case 10: NULL preSoftmaxElemWiseInputs array with positive count.
  // clang-format off
  // CHECK: rocmlirMIGraphXAttentionCreate: preSoftmaxElemWiseInputs array must be non-NULL when count > 0
  // clang-format on
  MlirRegion body10 = mlirRegionCreate();
  MlirOperation op10 = rocmlirMIGraphXAttentionCreate(
      loc, q, k, v, 1, NULL, rType, (MlirType){NULL}, (MlirType){NULL}, body10,
      MLIR_MIGRAPHX_ATTENTION_NONE, (MlirValue){NULL}, (MlirValue){NULL}, 0, 0);
  if (!mlirOperationIsNull(op10)) {
    fprintf(stderr,
            "FAIL: NULL inputs array with count > 0 should return null op\n");
    exit(1);
  }
  mlirRegionDestroy(body10);

  // Case 11: null resultType.
  // CHECK: rocmlirMIGraphXAttentionCreate: resultType is required
  MlirRegion body11 = mlirRegionCreate();
  MlirOperation op11 = rocmlirMIGraphXAttentionCreate(
      loc, q, k, v, 0, NULL, (MlirType){NULL}, (MlirType){NULL},
      (MlirType){NULL}, body11, MLIR_MIGRAPHX_ATTENTION_NONE, (MlirValue){NULL},
      (MlirValue){NULL}, 0, 0);
  if (!mlirOperationIsNull(op11)) {
    fprintf(stderr, "FAIL: null resultType should return null op\n");
    exit(1);
  }
  mlirRegionDestroy(body11);

  // CHECK: PASS: invalid-input cases all returned null
  fprintf(stderr, "PASS: invalid-input cases all returned null\n");
  mlirRegionDestroy(scratchRegion);
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirRegisterRocMLIRDialects(registry);
  mlirContextAppendDialectRegistry(ctx, registry);
  mlirContextLoadAllAvailableDialects(ctx);
  mlirDialectRegistryDestroy(registry);

  MlirLocation loc = mlirLocationUnknownGet(ctx);

  testBasicAttention(ctx, loc);
  testAttentionWithLse(ctx, loc);
  testAttentionWithSoftmaxType(ctx, loc);
  testAttentionWithPreSoftmaxInputs(ctx, loc);
  testAttentionWithBodyAndLse(ctx, loc);
  testAttentionCausal(ctx, loc);
  testAttentionKVCache(ctx, loc);
  testAttentionSplitKV(ctx, loc);
  testAttentionPrefixOffset(ctx, loc);
  testAttentionSlidingWindow(ctx, loc);
  testAttentionRejectsInvalidInputs(ctx, loc);

  mlirContextDestroy(ctx);
  return 0;
}
