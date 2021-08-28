//===- mixrir.c - Simple test of C APIs ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: mlir-mixr-capi-test 2>&1 | FileCheck %s
 *  */
#include "mlir-c/IR.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/Dialect/Standard.h"
#include "mlir-c/Dialect/MIGraphX.h"
#include "mlir-c/IntegerSet.h"
#include "mlir-c/Registration.h"

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
  MlirType inType = mlirRankedTensorTypeGet(4, inDims, mlirF32TypeGet(ctx));
  MlirType funcBodyArgTypes[] = {inType};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirBlock funcBody = mlirBlockCreate(
      sizeof(funcBodyArgTypes) / sizeof(MlirType), funcBodyArgTypes);
  mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  //-------------- func op

  // Set func attributes
  MlirAttribute funcTypeAttr = mlirAttributeParseGet(
      ctx, 
      mlirStringRefCreateFromCString("(tensor<1x64x56x56xf32>) -> (tensor<1x64x56x56xf32>)"));
  MlirAttribute funcNameAttr = mlirAttributeParseGet(
      ctx,
      mlirStringRefCreateFromCString("\"main\""));
  MlirNamedAttribute funcAttrs[] = {
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("type")),
          funcTypeAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_name")),
          funcNameAttr)};

  // Set func op
  MlirOperationState funcState =
      mlirOperationStateGet(mlirStringRefCreateFromCString("func"), location);
  mlirOperationStateAddAttributes(&funcState, 2, funcAttrs);
  mlirOperationStateAddOwnedRegions(&funcState, 1, &funcBodyRegion);
  MlirOperation func = mlirOperationCreate(&funcState);
  mlirBlockInsertOwnedOperation(moduleBody, 0, func);

  //-------------- filter0 = migraphx.constant

  // Set constant attributes
  // migraphx.constant can have literals in 'value' attribute. Alternatively, 'shape' and 'type'
  // can be set to fill random numbers
  MlirAttribute filter0ShapeAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString("[64:i64, 64:i64, 1:i64, 1:i64]"));
  MlirAttribute filter0TypeAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString("f32"));
  MlirNamedAttribute filter0Attrs[] = {
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("shape")),
          filter0ShapeAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("type")),
          filter0TypeAttr)};

  // Set constant op
  int64_t filter0Dims[] = {64, 64, 1, 1};
  MlirType filter0Type = mlirRankedTensorTypeGet(4, filter0Dims, mlirF32TypeGet(ctx));
  MlirOperationState filter0State = mlirOperationStateGet(
      mlirStringRefCreateFromCString("migraphx.constant"), location);
  mlirOperationStateAddResults(&filter0State, 1, &filter0Type);
  mlirOperationStateAddAttributes(&filter0State, 2, filter0Attrs);
  
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
  MlirAttribute conv0GroupAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString("1:i64"));
  MlirAttribute conv0PaddingModeAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString("0:i64"));
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
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("padding_mode")),
          conv0PaddingModeAttr)};

  // Set output shape
  int64_t conv0Dims[] = {1, 64, 56, 56};
  MlirType conv0Type = mlirRankedTensorTypeGet(4, conv0Dims, mlirF32TypeGet(ctx));

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
  // This example simply gives the same sized tensor.
  MlirAttribute bias0ShapeAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString("[1:i64, 64:i64, 56:i64, 56:i64]"));
  MlirAttribute bias0TypeAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString("f32"));
  MlirNamedAttribute bias0Attrs[] = {
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("shape")),
          bias0ShapeAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("type")),
          bias0TypeAttr)};

  // Set constant op
  int64_t bias0Dims[] = {1, 64, 56, 56};
  MlirType bias0Type = mlirRankedTensorTypeGet(4, bias0Dims, mlirF32TypeGet(ctx));
  MlirOperationState bias0State = mlirOperationStateGet(
      mlirStringRefCreateFromCString("migraphx.constant"), location);
  mlirOperationStateAddResults(&bias0State, 1, &bias0Type);
  mlirOperationStateAddAttributes(&bias0State, 2, bias0Attrs);
  
  MlirOperation bias0Op = mlirOperationCreate(&bias0State);
  mlirBlockAppendOwnedOperation(funcBody, bias0Op);
  MlirValue bias0Value = mlirOperationGetResult(bias0Op, 0);

  //-------------- migraphx.add op

  // Set add0 arguments
  MlirValue add0Operands[] = {conv0Value, bias0Value};

  // Set add op
  int64_t add0Dims[] = {1, 64, 56, 56};
  MlirType add0Type = mlirRankedTensorTypeGet(4, add0Dims, mlirF32TypeGet(ctx));
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
  MlirType relu0Type = mlirRankedTensorTypeGet(4, relu0Dims, mlirF32TypeGet(ctx));
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
      mlirStringRefCreateFromCString("std.return"), location);
  mlirOperationStateAddOperands(&retState, 1, retOperands);
  MlirOperation ret = mlirOperationCreate(&retState);
  mlirBlockAppendOwnedOperation(funcBody, ret);

  MlirOperation module = mlirModuleGetOperation(moduleOp);
  mlirOperationDump(module);
// CHECK-LABEL: func @main

//module  {
//  func @main(%arg0: tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32> {
//    %0 = "migraphx.constant"() {shape = [64, 64, 1, 1], type = f32} : () -> tensor<64x64x1x1xf32>
//    %1 = "migraphx.convolution"(%arg0, %0) {dilation = [1, 1], group = 1 : i64, padding = [0, 0], padding_mode = 0 : i64, stride = [1, 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tensor<1x64x56x56xf32>
//    %2 = "migraphx.constant"() {shape = [1, 64, 56, 56], type = f32} : () -> tensor<1x64x56x56xf32>
//    %3 = "migraphx.add"(%1, %2) : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
//    %4 = "migraphx.relu"(%3) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
//    return %4 : tensor<1x64x56x56xf32>
//  }
//}

  return moduleOp;
}

static int constructAndTraverseIr(MlirContext ctx) {
  MlirLocation location1 = mlirLocationUnknownGet(ctx);

  MlirModule moduleOp1 = makeAndDumpMIXR(ctx, location1);
  MlirOperation module1 = mlirModuleGetOperation(moduleOp1);

  mlirModuleDestroy(moduleOp1);

  return 0;
}

int main() {
  MlirContext ctx = mlirContextCreate();
  mlirRegisterAllDialects(ctx);
  if (constructAndTraverseIr(ctx))
    return 1;
  
  mlirContextDestroy(ctx);

  return 0;
}
