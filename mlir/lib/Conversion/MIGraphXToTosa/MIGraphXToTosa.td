//===-- MIGraphXToTosa.td - MIGraphX to TOSA conversion pattern definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MIGRAPHX_TO_TOSA
#define MLIR_CONVERSION_MIGRAPHX_TO_TOSA
include "mlir/Dialect/Tosa/IR/TosaOps.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "mlir/Dialect/MIGraphX/MIGraphXOps.td"

def : Pat<(MIGraphX_AddOp $input1, $input2), (Tosa_AddOp $input1, $input2)>;
def : Pat<(MIGraphX_SubOp $input1, $input2), (Tosa_SubOp $input1, $input2)>;
def : Pat<(MIGraphX_MulOp $input1, $input2), (Tosa_MulOp $input1, $input2, ConstantAttr<I32Attr, "0">)>;
def : Pat<(MIGraphX_ConstantOp ElementsAttr:$valueAttr, $shapeAttr, $typeAttr), (ConstantOp $valueAttr)>;

def Convert2DTo4DSymmPad : NativeCodeCall<
    "$_builder.getArrayAttr({($0.getValue()[0]), ($0.getValue()[0]), ($0.getValue()[1]), ($0.getValue()[1])})">;
def GetNullAttr : NativeCodeCall<"Attribute()">;

def : Pat<(MIGraphX_ConvolutionOp:$result $input1, $filter, $pad, $stride, $dilation,
      $group, $_ // ignore padding mode
      ),
    (Tosa_Conv2DOp $input1, $filter, (ConstantOp ConstantAttr<RankedF32ElementsAttr<[1]>, "{0.0f}">:$f32attr), (Convert2DTo4DSymmPad $pad), $stride, $dilation, (GetNullAttr))>;

/*
def : Pat<(MIGraphX_AddOp
    (MIGraphX_ConvolutionOp $input1, $filter, $pad, $stride, $dilation,
      $group, $_ // ignore padding mode
      ), $bias),
      (Tosa_Conv2DOp $input1, $filter, $bias, (Convert2DTo4DSymmPad $pad), $stride, $dilation, (GetNullAttr))>;
*/

def : Pat<(MIGraphX_RsqrtOp $input1), (Tosa_RsqrtOp $input1)>;
// Clamp to Relu if it clamps to max/min number, otherwise convert to ceil + floor
// {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} 
def : Pat<(MIGraphX_ReluOp $input1), 
  (Tosa_ClampOp $input1, ConstantAttr<I64Attr, "0">:$mini, ConstantAttr<I64Attr, "2147483647">:$maxi,  
    ConstantAttr<F32Attr, "0">:$minf, ConstantAttr<F32Attr, "3.40282347E+38">:$maxf)>;
def : Pat<(MIGraphX_TransposeOp $input1, $permAttr), 
  (Tosa_TransposeOp $input1, (ConstantOp $permAttr))>;
def : Pat<(MIGraphX_ReshapeOp $input1, $newshape), (Tosa_ReshapeOp $input1, $newshape)>;

#endif // MLIR_CONVERSION_MIGRAPHX_TO_TOSA
