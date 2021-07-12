//===-- conversion.td - Tosa to MIGraphX conversion pattern definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_TOSA_TO_MIGRAPHX
#define MLIR_CONVERSION_TOSA_TO_MIGRAPHX
include "mlir/Dialect/Tosa/IR/TosaOps.td"
include "mlir/Dialect/MIGraphX/MIGraphXOps.td"

def : Pat<(Tosa_AddOp $input1, $input2), (MIGraphX_AddOp $input1, $input2)>;
def : Pat<(Tosa_SubOp $input1, $input2), (MIGraphX_SubOp $input1, $input2)>;

// tosa.mul has attr shift which right_shift the result.
// FIXME: NativeCodeCall to convert attr:rshift to operand:rshift, create const tensor.
// def : Pat<(Tosa_MulOp $input1, $input2, $rshift), 
//   (Tosa_ArithmeticRightShiftOp (MIGraphX_MulOp $input1, $input2), $rshift, ConstBoolAttrFalse)>;
def : Pat<(Tosa_MulOp $input1, $input2, ConstantAttr<I32Attr, "0">), (MIGraphX_MulOp $input1, $input2)>;

def : Pat<(Tosa_RsqrtOp $input1), (MIGraphX_RsqrtOp $input1)>;

//def ExtractAttr : NativeCodeCall<
//    "$0.getValue()[($1)]">;
def Convert4DTo2DSymmPad : NativeCodeCall<
    "$_builder.getArrayAttr({($0.getValue()[0]), ($0.getValue()[2])})">;

def : Pat<(Tosa_Conv2DOp $input1, $filter, $bias, $pad, $stride, $dilation, $_),
    (MIGraphX_AddOp $bias,
    (MIGraphX_ConvolutionOp $input1, $filter, (Convert4DTo2DSymmPad $pad), $stride, $dilation, $_, $_))>;

// Clamp to Relu if it clamps to max/min number, otherwise convert to ceil + floor

// To extract elements in ElementsAttr and construct ArrayAttr from it. e.g., tosa const to attributes
def ConvertElemToArrayAttr: NativeCodeCall<
    "$_builder.getI64ArrayAttr(ArrayRef<int64_t>(llvm::to_vector<8>($0.getValues<int64_t>())))">;
def : Pat<(Tosa_TransposeOp $input1, (Tosa_ConstOp $permAttr)), 
    (MIGraphX_TransposeOp $input1, (ConvertElemToArrayAttr $permAttr))>;
def : Pat<(Tosa_ReshapeOp $input1, $newshape), (MIGraphX_ReshapeOp $input1, $newshape)>;
//FIXME: ignore quantization for now.
def : Pat<(Tosa_PadOp $input1, (Tosa_ConstOp $padDims), $_),
    (MIGraphX_PadOp $input1, (ConvertElemToArrayAttr $padDims), ConstantAttr<F32Attr, "0">:$value, 
    ConstantAttr<I32Attr, "0">:$padMode)>;
// Fixme: fix using enum type PadOpModeAttr<"migraphx::constant_pad">:$padMode

#endif // MLIR_CONVERSION_TOSA_TO_MIGRAPHX
