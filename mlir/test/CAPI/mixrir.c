//===- mixrir.c - Simple test of C APIs --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: mlir-mixr-capi-test 2>&1 | FileCheck %s
 *  */
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/MIGraphX.h"
#include "mlir-c/IR.h"
#include "mlir-c/IntegerSet.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir-c/RegisterRocMLIR.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

MlirModule makeAndDumpMIXR(MlirContext ctx, MlirLocation location) {
  MlirModule moduleOp = mlirModuleCreateEmpty(location);
  MlirBlock moduleBody = mlirModuleGetBody(moduleOp);

  // Set func arguments
  int64_t inDims[] = {1, 64, 56, 56};
  int64_t inStrides[] = {200704, 3136, 56, 1};
  MlirType inType =
      rocmlirMIXRShapedTypeGet(4, inDims, inStrides, mlirF32TypeGet(ctx));
  MlirType funcBodyArgTypes[] = {inType};
  MlirLocation funcBodyLocations[] = {location};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirBlock funcBody =
      mlirBlockCreate(sizeof(funcBodyArgTypes) / sizeof(MlirType),
                      funcBodyArgTypes, funcBodyLocations);
  mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  //-------------- func op

  // Set func attributes
  MlirAttribute funcTypeAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString(
               "(!migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1>) -> "
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
  MlirOperationState funcState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("func.func"), location);
  mlirOperationStateAddAttributes(&funcState, 2, funcAttrs);
  mlirOperationStateAddOwnedRegions(&funcState, 1, &funcBodyRegion);
  MlirOperation func = mlirOperationCreate(&funcState);
  mlirBlockInsertOwnedOperation(moduleBody, 0, func);

  //-------------- filter0 = migraphx.literal

  // Set constant attributes
  int64_t filter0Dims[] = {64, 64, 1, 1};
  int64_t filter0Strides[] = {64, 1, 1, 1};
  float f32Filter0[4096];
  for (int i = 0; i < 4096; i++) {
    f32Filter0[i] = 1.0f;
  }

  MlirAttribute filter0ValueAttr = mlirDenseElementsAttrFloatGet(
      mlirRankedTensorTypeGet(4, filter0Dims, mlirF32TypeGet(ctx),
                              mlirAttributeGetNull()),
      4096, f32Filter0);
  MlirNamedAttribute filter0Attrs[] = {mlirNamedAttributeGet(
      mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("value")),
      filter0ValueAttr)};

  // Set constant op
  MlirType filter0Type = rocmlirMIXRShapedTypeGet(
      4, filter0Dims, filter0Strides, mlirF32TypeGet(ctx));
  MlirOperationState filter0State = mlirOperationStateGet(
      mlirStringRefCreateFromCString("migraphx.literal"), location);
  mlirOperationStateAddResults(&filter0State, 1, &filter0Type);
  mlirOperationStateAddAttributes(&filter0State, 1, filter0Attrs);

  MlirOperation filter0Op = mlirOperationCreate(&filter0State);
  mlirBlockAppendOwnedOperation(funcBody, filter0Op);
  MlirValue filter0Value = mlirOperationGetResult(filter0Op, 0);

  //-------------- conv0 = migraphx.convolution

  // Set conv0 arguments : arg0 from the func and constant filter0
  MlirValue funcArg0 = mlirBlockGetArgument(funcBody, 0);
  MlirValue conv0Operands[] = {funcArg0, filter0Value};

  // Set convolution attributes
  // padding, stride, dilation, group, padding_mode
  MlirAttribute conv0PaddingAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString("[0:i64, 0:i64]"));
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
      mlirStringRefCreateFromCString("migraphx.convolution"), location);
  mlirOperationStateAddResults(&conv0OpState, 1, &conv0Type);
  mlirOperationStateAddOperands(&conv0OpState, 2, conv0Operands);
  mlirOperationStateAddAttributes(&conv0OpState, 5, conv0Attrs);
  MlirOperation conv0Op = mlirOperationCreate(&conv0OpState);
  mlirBlockAppendOwnedOperation(funcBody, conv0Op);
  MlirValue conv0Value = mlirOperationGetResult(conv0Op, 0);

  //-------------- bias0 = migraphx.constant

  // Set constant attributes
  int64_t bias0Dims[] = {1, 64, 1, 1};
  int64_t bias0Strides[] = {64, 1, 1, 1};
  float f32Bias[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  MlirAttribute bias0ValueAttr = mlirDenseElementsAttrFloatGet(
      mlirRankedTensorTypeGet(4, bias0Dims, mlirF32TypeGet(ctx),
                              mlirAttributeGetNull()),
      64, f32Bias);
  MlirNamedAttribute bias0Attrs[] = {mlirNamedAttributeGet(
      mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("value")),
      bias0ValueAttr)};

  // Set constant op
  MlirType bias0Type =
      rocmlirMIXRShapedTypeGet(4, bias0Dims, bias0Strides, mlirF32TypeGet(ctx));
  MlirOperationState bias0State = mlirOperationStateGet(
      mlirStringRefCreateFromCString("migraphx.literal"), location);
  mlirOperationStateAddResults(&bias0State, 1, &bias0Type);
  mlirOperationStateAddAttributes(&bias0State, 1, bias0Attrs);

  MlirOperation bias0Op = mlirOperationCreate(&bias0State);
  mlirBlockAppendOwnedOperation(funcBody, bias0Op);
  MlirValue bias0Value = mlirOperationGetResult(bias0Op, 0);

  //-------------- migraphx.add op

  // Set add0 arguments
  MlirValue add0Operands[] = {conv0Value, bias0Value};

  // Set add op
  int64_t add0Dims[] = {1, 64, 56, 56};
  int64_t add0Strides[] = {200704, 3136, 56, 1};
  MlirType add0Type =
      rocmlirMIXRShapedTypeGet(4, add0Dims, add0Strides, mlirF32TypeGet(ctx));

  MlirOperationState add0State = mlirOperationStateGet(
      mlirStringRefCreateFromCString("migraphx.add"), location);
  mlirOperationStateAddResults(&add0State, 1, &add0Type);
  mlirOperationStateAddOperands(&add0State, 2, add0Operands);

  MlirOperation add0Op = mlirOperationCreate(&add0State);
  mlirBlockAppendOwnedOperation(funcBody, add0Op);
  MlirValue add0Value = mlirOperationGetResult(add0Op, 0);

  //-------------- migraphx.relu op

  // Set relu0 arguments
  MlirValue relu0Operands[] = {add0Value};

  // Set relu op
  int64_t relu0Dims[] = {1, 64, 56, 56};
  int64_t relu0Strides[] = {200704, 3136, 56, 1};
  MlirType relu0Type =
      rocmlirMIXRShapedTypeGet(4, relu0Dims, relu0Strides, mlirF32TypeGet(ctx));
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

  MlirOperation module = mlirModuleGetOperation(moduleOp);
  mlirOperationDump(module);
  // CHECK-LABEL: func @main

  //  module {
  //    func.func @main(%arg0: !migraphx.shaped<1x64x56x56xf32,
  //    200704x3136x56x1>) -> !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1>
  //    {
  //      %0 = migraphx.literal(dense<1.000000e+00> : tensor<64x64x1x1xf32>) :
  //      <64x64x1x1xf32, 64x1x1x1> %1 = migraphx.convolution %arg0, %0
  //      {dilation = [1, 1], group = 1 : i64, padding = [0, 0], padding_mode =
  //      0 : i64, stride = [1, 1]} : !migraphx.shaped<1x64x56x56xf32,
  //      200704x3136x56x1>, !migraphx.shaped<64x64x1x1xf32, 64x1x1x1> ->
  //      !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1> %2 =
  //      migraphx.literal(dense<1.000000e+00> : tensor<1x64x1x1xf32>) :
  //      <1x64x1x1xf32, 64x1x1x1> %3 = migraphx.add %1, %2 :
  //      !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1>,
  //      !migraphx.shaped<1x64x1x1xf32, 64x1x1x1> ->
  //      !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1> %4 = migraphx.relu
  //      %3 : !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1> ->
  //      !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1> return %4 :
  //      !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1>
  //    }
  //  }

  return moduleOp;
}

static int constructAndTraverseIr(MlirContext ctx) {
  MlirLocation location1 = mlirLocationUnknownGet(ctx);
  MlirModule moduleOp1 = makeAndDumpMIXR(ctx, location1);

  mlirModuleDestroy(moduleOp1);

  return 0;
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

  mlirContextSetAllowUnregisteredDialects(ctx, true /*allow*/);
  if (constructAndTraverseIr(ctx))
    return 1;

  mlirContextDestroy(ctx);

  return 0;
}
