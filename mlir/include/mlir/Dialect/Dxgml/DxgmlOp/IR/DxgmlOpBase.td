//===- DxgmlOpBase.td - DxgmlOp dialect definitions -------*- tablegen -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines base MLIR DxgmlOp dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DXGMLOP_BASE
#define DXGMLOP_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"

def DxgmlOp_Dialect : Dialect {
  let name = "dxgml_op";
  let cppNamespace = "::mlir::dxgml_op";
  
  let summary = "The DxGML operations dialect";
  let description = [{
    The DxGML operations dialect provides machine learning operations
    used by the DXGML compiler.
  }];

  let useDefaultAttributePrinterParser = 1;
}

// Base class for DxgmlOp dialect ops
class DxgmlOp_Op<string mnemonic, list<Trait> traits = []> :
    Op<DxgmlOp_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

def DxgmlOp_ConvolutionModeEnum : I32EnumAttr<"ConvolutionMode",
    "Convolution mode", [
  I32EnumAttrCase<"convolution", 0, "convolution_mode_convolution">,
  I32EnumAttrCase<"cross_correlation", 1, "convolution_mode_cross_correlation">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_ConvolutionDirectionEnum : I32EnumAttr<"ConvolutionDirection",
    "Convolution direction", [
  I32EnumAttrCase<"forward", 0, "convolution_direction_forward">,
  I32EnumAttrCase<"backward", 1, "convolution_direction_backward">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_DepthSpaceOrderEnum : I32EnumAttr<"DepthSpaceOrder",
    "Depth to space order", [
  I32EnumAttrCase<"depth_column_row", 0, "depth_space_order_depth_column_row">,
  I32EnumAttrCase<"column_row_depth", 1, "depth_space_order_column_row_depth">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_AxisDirectionEnum : I32EnumAttr<"AxisDirection",
    "Axis direction", [
  I32EnumAttrCase<"axis_direction_increasing", 0, "axis_direction_increasing">,
  I32EnumAttrCase<"axis_direction_decreasing", 1, "axis_direction_decreasing">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_CoordinateTransformationModeEnum : I32EnumAttr<"CoordinateTransformationMode",
    "Coordinate transformation mode", [
  I32EnumAttrCase<"half_pixel", 0, "coordinate_transform_mode_half_pixel">,
  I32EnumAttrCase<"half_pixel_symmetric", 1, "coordinate_transform_mode_half_pixel_symmetric">,
  I32EnumAttrCase<"pytorch_half_pixel", 2, "coordinate_transform_mode_pytorch_half_pixel">,
  I32EnumAttrCase<"align_corners", 3, "coordinate_transform_mode_align_corners">,
  I32EnumAttrCase<"asymmetric", 4, "coordinate_transform_mode_asymmetric">,
  I32EnumAttrCase<"tf_crop_and_resize", 5, "coordinate_transform_mode_tf_crop_and_resize">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_InterpolationModeEnum : I32EnumAttr<"InterpolationMode",
    "Interpolation mode", [
  I32EnumAttrCase<"nearest_neighbor", 0, "interpolation_mode_nearest_neighbor">,
  I32EnumAttrCase<"linear", 1, "interpolation_mode_linear">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_IsInfinityModeEnum : I32EnumAttr<"IsInfinityMode",
    "IsInfinity detection mode", [
  I32EnumAttrCase<"either", 0, "is_infinity_mode_either">,
  I32EnumAttrCase<"positive", 1, "is_infinity_mode_positive">,
  I32EnumAttrCase<"negative", 2, "is_infinity_mode_negative">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_KeepAspectRatioPolicyEnum : I32EnumAttr<"KeepAspectRatioPolicy",
    "Keep aspect ratio policy", [
  I32EnumAttrCase<"stretch", 0, "keep_aspect_ratio_policy_stretch">,
  I32EnumAttrCase<"not_larger", 1, "keep_aspect_ratio_policy_not_larger">,
  I32EnumAttrCase<"not_smaller", 2, "keep_aspect_ratio_policy_not_smaller">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_MatrixTransformEnum : I32EnumAttr<"MatrixTransform",
    "Matrix transform", [
  I32EnumAttrCase<"none", 0, "matrix_transform_none">,
  I32EnumAttrCase<"transpose", 1, "matrix_transform_transpose">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_MultiheadAttentionMaskTypeEnum : I32EnumAttr<"MultiheadAttentionMaskType",
    "Multihead attention mask type", [
  I32EnumAttrCase<"none", 0, "multihead_attention_mask_type_none">,
  I32EnumAttrCase<"key_sequence_length", 1, "multihead_attention_mask_type_key_sequence_length">,
  I32EnumAttrCase<"key_sequence_end_start", 2, "multihead_attention_mask_type_key_sequence_end_start">,
  I32EnumAttrCase<"key_query_sequence_length_start_end", 3, "multihead_attention_mask_type_key_query_sequence_length_start_end">,
  I32EnumAttrCase<"boolean", 4, "multihead_attention_mask_type_boolean">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_NearestModeEnum : I32EnumAttr<"NearestMode",
    "Nearest rounding mode", [
  I32EnumAttrCase<"round_prefer_floor", 0, "nearest_mode_round_prefer_floor">,
  I32EnumAttrCase<"round_prefer_ceil", 1, "nearest_mode_round_prefer_ceil">,
  I32EnumAttrCase<"floor", 2, "nearest_mode_floor">,
  I32EnumAttrCase<"ceil", 3, "nearest_mode_ceil">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_PaddingModeEnum : I32EnumAttr<"PaddingMode",
    "Padding mode", [
  I32EnumAttrCase<"constant", 0, "padding_mode_constant">,
  I32EnumAttrCase<"edge", 1, "padding_mode_edge">,
  I32EnumAttrCase<"reflection", 2, "padding_mode_reflection">,
  I32EnumAttrCase<"symmetric", 3, "padding_mode_symmetric">,
  I32EnumAttrCase<"wrap", 4, "padding_mode_wrap">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_QuantizationTypeEnum : I32EnumAttr<"QuantizationType",
    "Quantization type", [
  I32EnumAttrCase<"none", 0, "quantization_type_none">,
  I32EnumAttrCase<"scale", 1, "quantization_type_scale">,
  I32EnumAttrCase<"scale_zero_point", 2, "quantization_type_scale_zero_point">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_RandomGeneratorTypeEnum : I32EnumAttr<"RandomGeneratorType",
    "Random generator type", [
  I32EnumAttrCase<"philox4_x32_10", 0, "dx_gml_random_generator_type_philox4_x32_10">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_RecurrentNetworkDirectionEnum : I32EnumAttr<"RecurrentNetworkDirection",
    "Recurrent network direction", [
  I32EnumAttrCase<"forward", 0, "recurrent_network_direction_forward">,
  I32EnumAttrCase<"backward", 1, "recurrent_network_direction_backward">,
  I32EnumAttrCase<"bidirectional", 2, "recurrent_network_direction_bidirectional">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_ReduceFunctionEnum : I32EnumAttr<"ReduceFunction",
    "Reduce function", [
  I32EnumAttrCase<"argmax", 0, "reduce_function_argmax">,
  I32EnumAttrCase<"argmin", 1, "reduce_function_argmin">,
  I32EnumAttrCase<"average", 2, "reduce_function_average">,
  I32EnumAttrCase<"l1", 3, "reduce_function_l1">,
  I32EnumAttrCase<"l2", 4, "reduce_function_l2">,
  I32EnumAttrCase<"log_sum", 5, "reduce_function_log_sum">,
  I32EnumAttrCase<"log_sum_exp", 6, "reduce_function_log_sum_exp">,
  I32EnumAttrCase<"max", 7, "reduce_function_max">,
  I32EnumAttrCase<"min", 8, "reduce_function_min">,
  I32EnumAttrCase<"multiply", 9, "reduce_function_multiply">,
  I32EnumAttrCase<"sum", 10, "reduce_function_sum">,
  I32EnumAttrCase<"sum_square", 11, "reduce_function_sum_square">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_ResizeInterpolationModeEnum : I32EnumAttr<"ResizeInterpolationMode",
    "Resize interpolation mode", [
  I32EnumAttrCase<"nearest", 0, "resize_interpolation_mode_nearest">,
  I32EnumAttrCase<"linear", 1, "resize_interpolation_mode_linear">,
  I32EnumAttrCase<"cubic", 2, "resize_interpolation_mode_cubic">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_RoundingModeEnum : I32EnumAttr<"RoundingMode",
    "Rounding mode", [
  I32EnumAttrCase<"halves_to_nearest_even", 0, "rounding_mode_halves_to_nearest_even">,
  I32EnumAttrCase<"toward_zero", 1, "rounding_mode_toward_zero">,
  I32EnumAttrCase<"toward_infinity", 2, "rounding_mode_toward_infinity">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

//===----------------------------------------------------------------------===//
// AttrDef base class
//===----------------------------------------------------------------------===//

class DxgmlOp_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<DxgmlOp_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Enum Attribute Wrappers
// These wrap I32EnumAttr values into dialect attribute types, enabling
// the #dxgml_op.<name>_enum_attr<value> syntax used in MLIR files.
//===----------------------------------------------------------------------===//

def DxgmlOp_DepthSpaceOrderAttr : DxgmlOp_Attr<"DepthSpaceOrderEnum",
                                                "depth_space_order_enum_attr"> {
  let summary = "DepthToSpace order attribute";
  let parameters = (ins
    EnumParameter<DxgmlOp_DepthSpaceOrderEnum>:$value
  );
  let assemblyFormat = "`<` $value `>`";
}

def DxgmlOp_ReduceFunctionAttr : DxgmlOp_Attr<"ReduceFunctionEnum",
                                               "reduce_function_enum_attr"> {
  let summary = "Reduce function attribute";
  let parameters = (ins
    EnumParameter<DxgmlOp_ReduceFunctionEnum>:$value
  );
  let assemblyFormat = "`<` $value `>`";
}

def DxgmlOp_KVQuantTypeEnum : I32EnumAttr<"KVQuantType",
    "KV cache quantization type", [
  I32EnumAttrCase<"kv_quant_none", 0, "kv_quant_none">,
  I32EnumAttrCase<"kv_quant_int8", 1, "kv_quant_int8">,
  I32EnumAttrCase<"kv_quant_int4", 2, "kv_quant_int4">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_KVQuantTypeAttr : DxgmlOp_Attr<"KVQuantTypeEnum",
                                             "kv_quant_type_enum_attr"> {
  let summary = "KV cache quantization type attribute";
  let parameters = (ins
    EnumParameter<DxgmlOp_KVQuantTypeEnum>:$value
  );
  let assemblyFormat = "`<` $value `>`";
}

#endif // DXGMLOP_BASE
