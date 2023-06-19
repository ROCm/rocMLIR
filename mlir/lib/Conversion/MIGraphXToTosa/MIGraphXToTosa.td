//===-- MIGraphXToTosa.td - MIGraphX to TOSA conversion pattern definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MIGRAPHX_TO_TOSA
#define MLIR_CONVERSION_MIGRAPHX_TO_TOSA
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Tosa/IR/TosaOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/Dialect/MIGraphX/MIGraphXOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"

def GetNullAttr : NativeCodeCall<"Attribute()">;

def : Pat<(MIGraphX_AddOp $input1, $input2), (Tosa_AddOp $input1, $input2)>;
def : Pat<(MIGraphX_SubOp $input1, $input2), (Tosa_SubOp $input1, $input2)>;
def : Pat<(MIGraphX_MulOp $input1, $input2), (Tosa_MulOp $input1, $input2, ConstantAttr<I32Attr, "0">)>;
def : Pat<(MIGraphX_PowOp $input1, $input2), (Tosa_PowOp $input1, $input2)>;
def : Pat<(MIGraphX_ClipOp $x, $mins, $maxs),
  (Tosa_MinimumOp (Tosa_MaximumOp $x, $mins, (returnType $x)), $maxs)>;
def : Pat<(MIGraphX_WhereOp $cond, $input1, $input2),
  (Tosa_SelectOp
    (Tosa_CastOp $cond,
      (returnType (NativeCodeCall<"cast<TensorType>($0.getType()).clone($_builder.getIntegerType(1))"> $cond))),
    $input1, $input2)>;

// Unary operators.
def : Pat<(MIGraphX_AbsOp $input1), (Tosa_AbsOp $input1)>;
def : Pat<(MIGraphX_CeilOp $input1), (Tosa_CeilOp $input1)>;
def : Pat<(MIGraphX_ConvertOp $input1), (Tosa_CastOp $input1)>;
def : Pat<(MIGraphX_ExpOp $input1), (Tosa_ExpOp $input1)>;
def : Pat<(MIGraphX_ErfOp $input1), (Tosa_ErfOp $input1)>;
def : Pat<(MIGraphX_FloorOp $input1), (Tosa_FloorOp $input1)>;
def : Pat<(MIGraphX_LogOp $input1), (Tosa_LogOp $input1)>;
def : Pat<(MIGraphX_NegOp $input1), (Tosa_NegateOp $input1, (GetNullAttr))>;
def : Pat<(MIGraphX_RecipOp $input1), (Tosa_ReciprocalOp $input1)>;
def : Pat<(MIGraphX_RsqrtOp $input1), (Tosa_RsqrtOp $input1)>;
def : Pat<(MIGraphX_SigmoidOp $input1), (Tosa_SigmoidOp $input1)>;
def : Pat<(MIGraphX_TanhOp $input1), (Tosa_TanhOp $input1)>;

def : Pat<(MIGraphX_ConstantOp ElementsAttr:$valueAttr, $shapeAttr, $valueTypeAttr), (Arith_ConstantOp $valueAttr)>;
def Convert2DTo4DSymmPad : NativeCodeCall<
    "$_builder.getArrayAttr({($0.getValue()[0]), ($0.getValue()[0]), ($0.getValue()[1]), ($0.getValue()[1])})">;
/* convolution is converted in the cpp file to handle more options.
def : Pat<(MIGraphX_ConvolutionOp:$result $input1, $filter, $pad, $stride, $dilation,
      $group, $_ // ignore padding mode
      ),
    (Tosa_Conv2DOp $input1, $filter, (ConstantOp ConstantAttr<RankedF32ElementsAttr<[1]>, "{0.0f}">:$f32attr), (Convert2DTo4DSymmPad $pad), $stride, $dilation, (GetNullAttr))>;
*/

// Clamp to Relu if it clamps to max/min number, otherwise convert to ceil + floor
// {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}
def : Pat<(MIGraphX_ReluOp $input1),
  (Tosa_ClampOp $input1, ConstantAttr<I64Attr, "0">:$mini, ConstantAttr<I64Attr, "2147483647">:$maxi,
    ConstantAttr<F32Attr, "0">:$minf, ConstantAttr<F32Attr, "3.40282347E+38">:$maxf)>;

def ConvertI64ArrayAttrtoDenseElementsAttr : NativeCodeCall<
    "DenseIntElementsAttr::get(RankedTensorType::get({static_cast<int64_t>($0.size())}, rewriter.getI64Type()), $0.getValue())">;
def : Pat<(MIGraphX_TransposeOp $input1, $permAttr),
  (Tosa_TransposeOp $input1, (Arith_ConstantOp (ConvertI64ArrayAttrtoDenseElementsAttr $permAttr)))>;

#endif // MLIR_CONVERSION_MIGRAPHX_TO_TOSA
