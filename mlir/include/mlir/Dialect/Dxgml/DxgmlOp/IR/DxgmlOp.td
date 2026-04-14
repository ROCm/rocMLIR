//===- DxgmlOp.td - DxgmlOp operation definitions ---------*- tablegen -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR DxgmlOp dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef DXGMLOP
#define DXGMLOP

include "mlir/Dialect/Dxgml/DxgmlOp/IR/DxgmlOpBase.td"
include "mlir/Dialect/Dxgml/IR/DxgmlTypes.td"
include "mlir/Dialect/Dxgml/IR/DxgmlAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Constant Operation
//===----------------------------------------------------------------------===//

def DxgmlOp_ConstantOp : DxgmlOp_Op<"constant", [Pure, ConstantLike]> {
  let summary = "Constant value operation";
  let description = [{
    Creates a constant tensor or scalar value from an attribute.

    Example:
    ```mlir
    %w = dxgml_op.constant(#dxgml.constant_resource<_conv1.weight : !dxgml.tensor<32x4x3x3x!dxgml.float16>>)
    %c = dxgml_op.constant(#dxgml.integer<1 : !dxgml.int64>)
    ```
  }];

  let arguments = (ins
    AnyAttr:$value
  );

  let results = (outs AnyType:$result);

  let hasCustomAssemblyFormat = 1;

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Convolution Operation
//===----------------------------------------------------------------------===//

def DxgmlOp_ConvolutionOp : DxgmlOp_Op<"convolution", [Pure]> {
  let summary = "Convolution operation";
  let description = [{
    Applies convolution over an input tensor.

    Example:
    ```mlir
    %0 = dxgml_op.convolution(%input, %filter, %bias) {
      group_count = #dxgml.integer<1 : !dxgml.int64>,
      dilations = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
      start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
      end_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
      strides = #dxgml.dense_integer_elements<[2, 2]> : !dxgml.tensor<2x!dxgml.int64>
    } : (!dxgml.tensor<1x4x224x224x!dxgml.float16>,
         !dxgml.tensor<32x4x3x3x!dxgml.float16>,
         !dxgml.tensor<32x!dxgml.float16>)
      -> !dxgml.tensor<1x32x112x112x!dxgml.float16>
    ```
  }];

  let arguments = (ins
    AnyType:$input,
    AnyType:$filter,
    Optional<AnyType>:$bias,
    OptionalAttr<DxgmlOp_ConvolutionModeEnum>:$mode,
    OptionalAttr<DxgmlOp_ConvolutionDirectionEnum>:$direction,
    OptionalAttr<Dxgml_DenseIntegerElementsAttr>:$strides,
    OptionalAttr<Dxgml_DenseIntegerElementsAttr>:$dilations,
    OptionalAttr<Dxgml_DenseIntegerElementsAttr>:$start_padding,
    OptionalAttr<Dxgml_DenseIntegerElementsAttr>:$end_padding,
    OptionalAttr<Dxgml_IntegerAttr>:$group_count,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );

  let results = (outs AnyType:$output);

  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $filter (`,` $bias^)? `)` attr-dict `:`
    `(` type($input) `,` type($filter) (`,` type($bias)^)? `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// ConvolutionInteger Operation
//===----------------------------------------------------------------------===//

def DxgmlOp_ConvolutionIntegerOp : DxgmlOp_Op<"convolution_integer", [Pure, AttrSizedOperandSegments]> {
  let summary = "Integer convolution operation";
  let description = [{
    Applies integer convolution over input tensors with optional zero-point offsets.
  }];

  let arguments = (ins
    AnyType:$input,
    AnyType:$filter,
    Optional<AnyType>:$input_zero_point,
    Optional<AnyType>:$filter_zero_point,
    Dxgml_DenseIntegerElementsAttr:$strides,
    Dxgml_DenseIntegerElementsAttr:$dilations,
    Dxgml_DenseIntegerElementsAttr:$start_padding,
    Dxgml_DenseIntegerElementsAttr:$end_padding,
    Dxgml_IntegerAttr:$group_count,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );

  let results = (outs AnyType:$output);

  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $filter
    (`,` $input_zero_point^)? (`,` $filter_zero_point^)? `)` attr-dict `:`
    `(` type($input) `,` type($filter)
    (`,` type($input_zero_point)^)? (`,` type($filter_zero_point)^)? `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Elementwise Unary Operations
//===----------------------------------------------------------------------===//

class DxgmlOp_UnaryOp<string mnemonic, string summary_str> :
    DxgmlOp_Op<mnemonic, [Pure]> {
  let summary = summary_str;
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_AbsOp        : DxgmlOp_UnaryOp<"abs",        "Absolute value">;
def DxgmlOp_ACosOp        : DxgmlOp_UnaryOp<"acos",       "Arccosine">;
def DxgmlOp_ACoshOp       : DxgmlOp_UnaryOp<"acosh",      "Inverse hyperbolic cosine">;
def DxgmlOp_ASinOp        : DxgmlOp_UnaryOp<"asin",       "Arcsine">;
def DxgmlOp_ASinhOp       : DxgmlOp_UnaryOp<"asinh",      "Inverse hyperbolic sine">;
def DxgmlOp_ATanOp        : DxgmlOp_UnaryOp<"atan",       "Arctangent">;
def DxgmlOp_ATanhOp       : DxgmlOp_UnaryOp<"atanh",      "Inverse hyperbolic tangent">;
def DxgmlOp_BitCountOp    : DxgmlOp_UnaryOp<"bit_count",  "Bit population count">;
def DxgmlOp_BitNotOp      : DxgmlOp_UnaryOp<"bit_not",    "Bitwise NOT">;
def DxgmlOp_CeilOp        : DxgmlOp_UnaryOp<"ceil",       "Ceiling">;
def DxgmlOp_CosOp         : DxgmlOp_UnaryOp<"cos",        "Cosine">;
def DxgmlOp_CoshOp        : DxgmlOp_UnaryOp<"cosh",       "Hyperbolic cosine">;
def DxgmlOp_ErfOp         : DxgmlOp_UnaryOp<"erf",        "Error function">;
def DxgmlOp_ExpOp         : DxgmlOp_UnaryOp<"exp",        "Exponential">;
def DxgmlOp_FloorOp       : DxgmlOp_UnaryOp<"floor",      "Floor">;
def DxgmlOp_IdentityOp    : DxgmlOp_UnaryOp<"identity",   "Identity">;
def DxgmlOp_IsNanOp       : DxgmlOp_UnaryOp<"is_nan",     "IsNaN check">;
def DxgmlOp_LogOp         : DxgmlOp_UnaryOp<"log",        "Natural logarithm">;
def DxgmlOp_LogicalNotOp  : DxgmlOp_UnaryOp<"logical_not","Logical NOT">;
def DxgmlOp_NegateOp      : DxgmlOp_UnaryOp<"negate",     "Negation">;
def DxgmlOp_ReciprocalOp  : DxgmlOp_UnaryOp<"recip",      "Reciprocal">;
def DxgmlOp_ReluOp        : DxgmlOp_UnaryOp<"relu",       "ReLU activation">;
def DxgmlOp_RoundOp       : DxgmlOp_UnaryOp<"round",      "Round to nearest even">;
def DxgmlOp_SignOp        : DxgmlOp_UnaryOp<"sign",       "Sign function">;
def DxgmlOp_SigmoidOp     : DxgmlOp_UnaryOp<"sigmoid",    "Sigmoid activation">;
def DxgmlOp_SinOp         : DxgmlOp_UnaryOp<"sin",        "Sine">;
def DxgmlOp_SinhOp        : DxgmlOp_UnaryOp<"sinh",       "Hyperbolic sine">;
def DxgmlOp_SoftplusOp    : DxgmlOp_UnaryOp<"softplus",   "Softplus activation">;
def DxgmlOp_SoftsignOp    : DxgmlOp_UnaryOp<"softsign",   "Softsign activation">;
def DxgmlOp_SqrtOp        : DxgmlOp_UnaryOp<"sqrt",       "Square root">;
def DxgmlOp_TanOp         : DxgmlOp_UnaryOp<"tan",        "Tangent">;
def DxgmlOp_TanhOp        : DxgmlOp_UnaryOp<"tanh",       "Hyperbolic tangent">;

//===----------------------------------------------------------------------===//
// Elementwise Binary Operations
//===----------------------------------------------------------------------===//

class DxgmlOp_BinaryOp<string mnemonic, string summary_str> :
    DxgmlOp_Op<mnemonic, [Pure]> {
  let summary = summary_str;
  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $lhs `,` $rhs `)` attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type($output)
  }];
}

def DxgmlOp_AddOp              : DxgmlOp_BinaryOp<"add",               "Element-wise addition">;
def DxgmlOp_ATanYxOp           : DxgmlOp_BinaryOp<"atan_yx",           "Arctangent of y/x">;
def DxgmlOp_BitAndOp           : DxgmlOp_BinaryOp<"bit_and",           "Bitwise AND">;
def DxgmlOp_BitOrOp            : DxgmlOp_BinaryOp<"bit_or",            "Bitwise OR">;
def DxgmlOp_BitShiftLeftOp     : DxgmlOp_BinaryOp<"bit_shift_left",    "Bitwise shift left">;
def DxgmlOp_BitShiftRightOp    : DxgmlOp_BinaryOp<"bit_shift_right",   "Bitwise shift right">;
def DxgmlOp_BitXorOp           : DxgmlOp_BinaryOp<"bit_xor",           "Bitwise XOR">;
def DxgmlOp_DifferenceSquareOp : DxgmlOp_BinaryOp<"difference_square", "Squared difference">;
def DxgmlOp_DivideOp           : DxgmlOp_BinaryOp<"divide",            "Element-wise division">;
def DxgmlOp_LogicalAndOp       : DxgmlOp_BinaryOp<"logical_and",       "Logical AND">;
def DxgmlOp_LogicalEqualsOp    : DxgmlOp_BinaryOp<"logical_equals",    "Logical equals">;
def DxgmlOp_LogicalGreaterOp   : DxgmlOp_BinaryOp<"logical_greater_than",          "Logical greater than">;
def DxgmlOp_LogicalGreaterEqOp : DxgmlOp_BinaryOp<"logical_greater_than_or_equal", "Logical greater than or equal">;
def DxgmlOp_LogicalLessOp      : DxgmlOp_BinaryOp<"logical_less_than",             "Logical less than">;
def DxgmlOp_LogicalLessEqOp    : DxgmlOp_BinaryOp<"logical_less_than_or_equal",    "Logical less than or equal">;
def DxgmlOp_LogicalOrOp        : DxgmlOp_BinaryOp<"logical_or",        "Logical OR">;
def DxgmlOp_LogicalXorOp       : DxgmlOp_BinaryOp<"logical_xor",       "Logical XOR">;
def DxgmlOp_MaxOp              : DxgmlOp_BinaryOp<"max",               "Element-wise maximum">;
def DxgmlOp_MeanOp             : DxgmlOp_BinaryOp<"mean",              "Element-wise mean">;
def DxgmlOp_MinOp              : DxgmlOp_BinaryOp<"min",               "Element-wise minimum">;
def DxgmlOp_ModulusFloorOp     : DxgmlOp_BinaryOp<"modulus_floor",     "Modulus with floor rounding">;
def DxgmlOp_ModulusTruncateOp  : DxgmlOp_BinaryOp<"modulus_truncate",  "Modulus with truncate rounding">;
def DxgmlOp_MultiplyOp         : DxgmlOp_BinaryOp<"multiply",          "Element-wise multiplication">;
def DxgmlOp_PowOp              : DxgmlOp_BinaryOp<"pow",               "Element-wise power">;
def DxgmlOp_SubtractOp         : DxgmlOp_BinaryOp<"subtract",          "Element-wise subtraction">;

//===----------------------------------------------------------------------===//
// Matrix Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_GemmOp : DxgmlOp_Op<"gemm", [Pure]> {
  let summary = "General Matrix Multiplication";
  let description = [{
    Performs GEMM: alpha * TransA(A) * TransB(B) + beta * C
  }];

  let arguments = (ins
    AnyType:$a_tensor,
    AnyType:$b_tensor,
    Optional<AnyType>:$c_tensor,
    OptionalAttr<Dxgml_FloatAttr>:$alpha,
    OptionalAttr<Dxgml_FloatAttr>:$beta,
    OptionalAttr<DxgmlOp_MatrixTransformEnum>:$trans_a,
    OptionalAttr<DxgmlOp_MatrixTransformEnum>:$trans_b,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );

  let results = (outs AnyType:$output);

  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $a_tensor `,` $b_tensor (`,` $c_tensor^)? `)` attr-dict `:`
    `(` type($a_tensor) `,` type($b_tensor) (`,` type($c_tensor)^)? `)` `->` type($output)
  }];
}

def DxgmlOp_MatrixMultiplyIntegerOp : DxgmlOp_Op<"matrix_multiply_integer", [Pure, AttrSizedOperandSegments]> {
  let summary = "Integer matrix multiplication";
  let description = [{
    Performs integer matrix multiplication with optional zero-point offsets.
  }];

  let arguments = (ins
    AnyType:$a_tensor,
    AnyType:$b_tensor,
    Optional<AnyType>:$a_zero_point,
    Optional<AnyType>:$b_zero_point,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );

  let results = (outs AnyType:$output);

  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $a_tensor `,` $b_tensor
    (`,` $a_zero_point^)? (`,` $b_zero_point^)? `)` attr-dict `:`
    `(` type($a_tensor) `,` type($b_tensor)
    (`,` type($a_zero_point)^)? (`,` type($b_zero_point)^)? `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Pooling Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_AveragePoolingOp : DxgmlOp_Op<"average_pooling", [Pure]> {
  let summary = "Average pooling operation";
  let arguments = (ins
    AnyType:$input,
    Dxgml_DenseIntegerElementsAttr:$strides,
    Dxgml_DenseIntegerElementsAttr:$window_size,
    Dxgml_DenseIntegerElementsAttr:$start_padding,
    Dxgml_DenseIntegerElementsAttr:$end_padding,
    OptionalAttr<Dxgml_DenseIntegerElementsAttr>:$dilations,
    OptionalAttr<Dxgml_BoolAttr>:$include_padding,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_MaxPoolingOp : DxgmlOp_Op<"max_pooling", [Pure]> {
  let summary = "Max pooling operation";
  let arguments = (ins
    AnyType:$input,
    Dxgml_DenseIntegerElementsAttr:$strides,
    Dxgml_DenseIntegerElementsAttr:$window_size,
    Dxgml_DenseIntegerElementsAttr:$start_padding,
    Dxgml_DenseIntegerElementsAttr:$end_padding,
    OptionalAttr<Dxgml_DenseIntegerElementsAttr>:$dilations,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output, Optional<AnyType>:$output_indices);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` `(` type($output) (`,` type($output_indices)^)? `)`
  }];
}

def DxgmlOp_MaxUnpoolingOp : DxgmlOp_Op<"max_unpooling", [Pure]> {
  let summary = "Max unpooling operation";
  let arguments = (ins
    AnyType:$input,
    AnyType:$indices,
    Dxgml_DenseIntegerElementsAttr:$strides,
    Dxgml_DenseIntegerElementsAttr:$window_size,
    Dxgml_DenseIntegerElementsAttr:$start_padding,
    Dxgml_DenseIntegerElementsAttr:$end_padding,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $indices `)` attr-dict `:`
    `(` type($input) `,` type($indices) `)` `->` type($output)
  }];
}

def DxgmlOp_LpPoolingOp : DxgmlOp_Op<"lp_pooling", [Pure]> {
  let summary = "Lp pooling operation";
  let arguments = (ins
    AnyType:$input,
    Dxgml_DenseIntegerElementsAttr:$strides,
    Dxgml_DenseIntegerElementsAttr:$window_size,
    Dxgml_DenseIntegerElementsAttr:$start_padding,
    Dxgml_DenseIntegerElementsAttr:$end_padding,
    OptionalAttr<Dxgml_IntegerAttr>:$p,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_GlobalAvgPoolOp : DxgmlOp_Op<"global_avg_pool", [Pure]> {
  let summary = "Global average pooling";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Normalization Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_BatchNormalizationOp : DxgmlOp_Op<"batch_normalization", [Pure]> {
  let summary = "Batch normalization";
  let arguments = (ins
    AnyType:$input,
    AnyType:$scale,
    AnyType:$bias,
    AnyType:$mean,
    AnyType:$variance,
    OptionalAttr<Dxgml_FloatAttr>:$epsilon,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $scale `,` $bias `,` $mean `,` $variance `)` attr-dict `:`
    `(` type($input) `,` type($scale) `,` type($bias) `,` type($mean) `,` type($variance) `)` `->` type($output)
  }];
}

def DxgmlOp_LpNormalizationOp : DxgmlOp_Op<"lp_normalization", [Pure]> {
  let summary = "LP normalization operation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_IntegerAttr>:$p,
    OptionalAttr<Dxgml_IntegerAttr>:$axis,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_LocalResponseNormalizationOp : DxgmlOp_Op<"local_response_normalization", [Pure]> {
  let summary = "Local response normalization";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_IntegerAttr>:$local_size,
    OptionalAttr<Dxgml_FloatAttr>:$alpha,
    OptionalAttr<Dxgml_FloatAttr>:$beta,
    OptionalAttr<Dxgml_FloatAttr>:$bias,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_MeanVarianceNormalizationOp : DxgmlOp_Op<"mean_variance_normalization", [Pure, AttrSizedOperandSegments]> {
  let summary = "Mean variance normalization";
  let arguments = (ins
    AnyType:$input,
    Optional<AnyType>:$scale,
    Optional<AnyType>:$bias,
    OptionalAttr<Dxgml_DenseIntegerElementsAttr>:$axes,
    OptionalAttr<Dxgml_FloatAttr>:$epsilon,
    OptionalAttr<Dxgml_BoolAttr>:$use_mean,
    OptionalAttr<Dxgml_BoolAttr>:$use_variance,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input (`,` $scale^)? (`,` $bias^)? `)` attr-dict `:`
    `(` type($input) (`,` type($scale)^)? (`,` type($bias)^)? `)` `->` type($output)
  }];
}

def DxgmlOp_RmsNormalizationOp : DxgmlOp_Op<"rms_normalization", [Pure]> {
  let summary = "RMS normalization";
  let arguments = (ins
    AnyType:$input,
    AnyType:$scale,
    Optional<AnyType>:$bias,
    OptionalAttr<Dxgml_FloatAttr>:$epsilon,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $scale (`,` $bias^)? `)` attr-dict `:`
    `(` type($input) `,` type($scale) (`,` type($bias)^)? `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Shape Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_ReshapeOp : DxgmlOp_Op<"reshape", [Pure]> {
  let summary = "Reshape tensor";
  let arguments = (ins
    AnyType:$inputTensor,
    Optional<AnyType>:$shape,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $inputTensor (`,` $shape^)? `)` attr-dict `:`
    `(` type($inputTensor) (`,` type($shape)^)? `)` `->` type($result)
  }];
}

def DxgmlOp_TransposeOp : DxgmlOp_Op<"transpose", [Pure]> {
  let summary = "Transpose tensor";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_DenseIntegerElementsAttr>:$permutation,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_SliceOp : DxgmlOp_Op<"slice", [Pure]> {
  let summary = "Slice tensor";
  let arguments = (ins
    AnyType:$input,
    Dxgml_DenseIntegerElementsAttr:$starts,
    Dxgml_DenseIntegerElementsAttr:$ends,
    OptionalAttr<Dxgml_DenseIntegerElementsAttr>:$axes,
    OptionalAttr<Dxgml_DenseIntegerElementsAttr>:$steps,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_ConcatOp : DxgmlOp_Op<"concat", [Pure]> {
  let summary = "Concatenate tensors";
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Dxgml_IntegerAttr:$axis,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $inputs `)` attr-dict `:`
    `(` type($inputs) `)` `->` type($output)
  }];
}

def DxgmlOp_BroadcastOp : DxgmlOp_Op<"broadcast", [Pure]> {
  let summary = "Broadcast tensor";
  let arguments = (ins
    AnyType:$input,
    Optional<AnyType>:$shape,
    OptionalAttr<Dxgml_DenseIntegerElementsAttr>:$target_shape,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input (`,` $shape^)? `)` attr-dict `:`
    `(` type($input) (`,` type($shape)^)? `)` `->` type($output)
  }];
}

def DxgmlOp_BroadcastDimOp : DxgmlOp_Op<"broadcast_dim", [Pure]> {
  let summary = "Broadcast along a dimension";
  let arguments = (ins
    AnyType:$input,
    Optional<AnyType>:$shape,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input (`,` $shape^)? `)` attr-dict `:`
    `(` type($input) (`,` type($shape)^)? `)` `->` type($output)
  }];
}

def DxgmlOp_SplitOp : DxgmlOp_Op<"split", [Pure]> {
  let summary = "Split tensor along an axis";
  let arguments = (ins
    AnyType:$input,
    Optional<AnyType>:$split_sizes,
    OptionalAttr<Dxgml_IntegerAttr>:$axis,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input (`,` $split_sizes^)? `)` attr-dict `:`
    `(` type($input) (`,` type($split_sizes)^)? `)` `->` type($outputs)
  }];
}

def DxgmlOp_SqueezeOp : DxgmlOp_Op<"squeeze", [Pure]> {
  let summary = "Remove dimensions of size 1";
  let arguments = (ins
    AnyType:$input,
    Optional<AnyType>:$axes,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input (`,` $axes^)? `)` attr-dict `:`
    `(` type($input) (`,` type($axes)^)? `)` `->` type($output)
  }];
}

def DxgmlOp_UnsqueezeOp : DxgmlOp_Op<"unsqueeze", [Pure]> {
  let summary = "Insert dimensions of size 1";
  let arguments = (ins
    AnyType:$input,
    AnyType:$axes,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $axes `)` attr-dict `:`
    `(` type($input) `,` type($axes) `)` `->` type($output)
  }];
}

def DxgmlOp_FlattenOp : DxgmlOp_Op<"fold", [Pure]> {
  let summary = "Flatten/fold tensor";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_IntegerAttr>:$axis,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_TileOp : DxgmlOp_Op<"tile", [Pure]> {
  let summary = "Tile/repeat tensor";
  let arguments = (ins
    AnyType:$input,
    AnyType:$repeats,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $repeats `)` attr-dict `:`
    `(` type($input) `,` type($repeats) `)` `->` type($output)
  }];
}

def DxgmlOp_ShapeOp : DxgmlOp_Op<"shape", [Pure]> {
  let summary = "Get shape of tensor";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_IntegerAttr>:$start,
    OptionalAttr<Dxgml_IntegerAttr>:$end,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_DimOp : DxgmlOp_Op<"dim", [Pure]> {
  let summary = "Get a dimension size";
  let arguments = (ins
    AnyType:$input,
    Dxgml_IntegerAttr:$axis,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_ReverseSubsequencesOp : DxgmlOp_Op<"reverse_subsequences", [Pure]> {
  let summary = "Reverse subsequences of a tensor";
  let arguments = (ins
    AnyType:$input,
    AnyType:$sequence_lengths,
    OptionalAttr<Dxgml_IntegerAttr>:$batch_axis,
    OptionalAttr<Dxgml_IntegerAttr>:$time_axis,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $sequence_lengths `)` attr-dict `:`
    `(` type($input) `,` type($sequence_lengths) `)` `->` type($output)
  }];
}

def DxgmlOp_DiagonalMatrixOp : DxgmlOp_Op<"diagonal_matrix", [Pure]> {
  let summary = "Create diagonal matrix";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_IntegerAttr>:$diagonal_fill_begin,
    OptionalAttr<Dxgml_IntegerAttr>:$diagonal_fill_end,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Depth/Space Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_DepthToSpaceOp : DxgmlOp_Op<"depth_to_space", [Pure]> {
  let summary = "Depth to space transformation";
  let arguments = (ins
    AnyType:$input,
    Dxgml_IntegerAttr:$block_size,
    DxgmlOp_DepthSpaceOrderAttr:$depth_space_order,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_SpaceToDepthOp : DxgmlOp_Op<"space_to_depth", [Pure]> {
  let summary = "Space to depth transformation";
  let arguments = (ins
    AnyType:$input,
    Dxgml_IntegerAttr:$block_size,
    DxgmlOp_DepthSpaceOrderAttr:$depth_space_order,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Quantization Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_QuantizeLinearOp : DxgmlOp_Op<"quantize_linear", [Pure]> {
  let summary = "Quantize linear operation";
  let arguments = (ins
    AnyType:$input,
    AnyType:$scale,
    Optional<AnyType>:$zero_point,
    OptionalAttr<DxgmlOp_QuantizationTypeEnum>:$quantization_type,
    OptionalAttr<Dxgml_IntegerAttr>:$axis,
    OptionalAttr<DxgmlOp_RoundingModeEnum>:$rounding_mode,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $scale (`,` $zero_point^)? `)` attr-dict `:`
    `(` type($input) `,` type($scale) (`,` type($zero_point)^)? `)` `->` type($output)
  }];
}

def DxgmlOp_DequantizeLinearOp : DxgmlOp_Op<"dequantize_linear", [Pure]> {
  let summary = "Dequantize linear operation";
  let arguments = (ins
    AnyType:$input,
    AnyType:$scale,
    Optional<AnyType>:$zero_point,
    OptionalAttr<DxgmlOp_QuantizationTypeEnum>:$quantization_type,
    OptionalAttr<Dxgml_IntegerAttr>:$axis,
    OptionalAttr<Dxgml_IntegerAttr>:$block_size,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $scale (`,` $zero_point^)? `)` attr-dict `:`
    `(` type($input) `,` type($scale) (`,` type($zero_point)^)? `)` `->` type($output)
  }];
}

def DxgmlOp_DynamicQuantizeLinearOp : DxgmlOp_Op<"dynamic_quantize_linear", [Pure]> {
  let summary = "Dynamic quantize linear operation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output, AnyType:$scale, AnyType:$zero_point);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` `(` type($output) `,` type($scale) `,` type($zero_point) `)`
  }];
}

//===----------------------------------------------------------------------===//
// Cast / Type Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_CastOp : DxgmlOp_Op<"cast", [Pure]> {
  let summary = "Type cast operation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_IntegerAttr>:$to,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Activation Functions (with parameters)
//===----------------------------------------------------------------------===//

def DxgmlOp_ClipOp : DxgmlOp_Op<"clip", [Pure, AttrSizedOperandSegments]> {
  let summary = "Clip values to [min, max]";
  let arguments = (ins
    AnyType:$input,
    Optional<AnyType>:$min_val,
    Optional<AnyType>:$max_val,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input (`,` $min_val^)? (`,` $max_val^)? `)` attr-dict `:`
    `(` type($input) (`,` type($min_val)^)? (`,` type($max_val)^)? `)` `->` type($output)
  }];
}

def DxgmlOp_CeluOp : DxgmlOp_Op<"celu", [Pure]> {
  let summary = "CELU activation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_FloatAttr>:$alpha,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_EluOp : DxgmlOp_Op<"elu", [Pure]> {
  let summary = "ELU activation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_FloatAttr>:$alpha,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_GeluOp : DxgmlOp_Op<"gelu", [Pure]> {
  let summary = "GELU activation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_HardSigmoidOp : DxgmlOp_Op<"hard_sigmoid", [Pure]> {
  let summary = "Hard sigmoid activation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_FloatAttr>:$alpha,
    OptionalAttr<Dxgml_FloatAttr>:$beta,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_HardSwishOp : DxgmlOp_Op<"hard_swish", [Pure]> {
  let summary = "Hard swish activation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_HardmaxOp : DxgmlOp_Op<"hardmax", [Pure]> {
  let summary = "Hardmax activation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_IntegerAttr>:$axis,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_LeakyReluOp : DxgmlOp_Op<"leaky_relu", [Pure]> {
  let summary = "Leaky ReLU activation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_FloatAttr>:$alpha,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_LinearOp : DxgmlOp_Op<"linear", [Pure]> {
  let summary = "Linear/affine transformation";
  let arguments = (ins
    AnyType:$input,
    AnyType:$weight,
    Optional<AnyType>:$bias,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $weight (`,` $bias^)? `)` attr-dict `:`
    `(` type($input) `,` type($weight) (`,` type($bias)^)? `)` `->` type($output)
  }];
}

def DxgmlOp_ParameterizedReluOp : DxgmlOp_Op<"parameterized_relu", [Pure]> {
  let summary = "Parameterized ReLU activation";
  let arguments = (ins
    AnyType:$input,
    AnyType:$slope,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $slope `)` attr-dict `:`
    `(` type($input) `,` type($slope) `)` `->` type($output)
  }];
}

def DxgmlOp_ParametricSoftplusOp : DxgmlOp_Op<"parametric_softplus", [Pure]> {
  let summary = "Parametric softplus activation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_FloatAttr>:$alpha,
    OptionalAttr<Dxgml_FloatAttr>:$beta,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_ScaledEluOp : DxgmlOp_Op<"scaled_elu", [Pure]> {
  let summary = "Scaled ELU activation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_FloatAttr>:$alpha,
    OptionalAttr<Dxgml_FloatAttr>:$gamma,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_ScaledTanhOp : DxgmlOp_Op<"scaled_tanh", [Pure]> {
  let summary = "Scaled tanh activation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_FloatAttr>:$alpha,
    OptionalAttr<Dxgml_FloatAttr>:$beta,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_ShrinkOp : DxgmlOp_Op<"shrink", [Pure]> {
  let summary = "Shrink activation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_FloatAttr>:$bias,
    OptionalAttr<Dxgml_FloatAttr>:$lambd,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_SoftmaxOp : DxgmlOp_Op<"softmax", [Pure]> {
  let summary = "Softmax activation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<AnyAttr>:$axis,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_LogSoftmaxOp : DxgmlOp_Op<"log_softmax", [Pure]> {
  let summary = "Log softmax activation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<AnyAttr>:$axis,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_SwishOp : DxgmlOp_Op<"swish", [Pure]> {
  let summary = "Swish activation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_ThresholdOp : DxgmlOp_Op<"threshold", [Pure]> {
  let summary = "Threshold activation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_FloatAttr>:$threshold,
    OptionalAttr<Dxgml_FloatAttr>:$value,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_ThresholdedReluOp : DxgmlOp_Op<"thresholded_relu", [Pure]> {
  let summary = "Thresholded ReLU activation";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_FloatAttr>:$alpha,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_ConstantPowOp : DxgmlOp_Op<"constant_pow", [Pure]> {
  let summary = "Element-wise power with constant exponent";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_FloatAttr>:$exponent,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_IsInfinityOp : DxgmlOp_Op<"is_infinity", [Pure]> {
  let summary = "Check for infinity";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<DxgmlOp_IsInfinityModeEnum>:$infinity_mode,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Gather / Scatter Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_GatherOp : DxgmlOp_Op<"gather", [Pure]> {
  let summary = "Gather elements from tensor";
  let arguments = (ins
    AnyType:$input,
    AnyType:$indices,
    OptionalAttr<Dxgml_IntegerAttr>:$axis,
    OptionalAttr<Dxgml_IntegerAttr>:$index_dims,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $indices `)` attr-dict `:`
    `(` type($input) `,` type($indices) `)` `->` type($output)
  }];
}

def DxgmlOp_GatherElementsOp : DxgmlOp_Op<"gather_elements", [Pure]> {
  let summary = "Gather elements along an axis";
  let arguments = (ins
    AnyType:$input,
    AnyType:$indices,
    OptionalAttr<Dxgml_IntegerAttr>:$axis,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $indices `)` attr-dict `:`
    `(` type($input) `,` type($indices) `)` `->` type($output)
  }];
}

def DxgmlOp_GatherNdOp : DxgmlOp_Op<"gather_nd", [Pure]> {
  let summary = "Gather elements with N-dimensional indices";
  let arguments = (ins
    AnyType:$input,
    AnyType:$indices,
    OptionalAttr<Dxgml_IntegerAttr>:$batch_dims,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $indices `)` attr-dict `:`
    `(` type($input) `,` type($indices) `)` `->` type($output)
  }];
}

def DxgmlOp_ScatterOp : DxgmlOp_Op<"scatter", [Pure]> {
  let summary = "Scatter updates into tensor";
  let arguments = (ins
    AnyType:$input,
    AnyType:$indices,
    AnyType:$updates,
    OptionalAttr<Dxgml_IntegerAttr>:$axis,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $indices `,` $updates `)` attr-dict `:`
    `(` type($input) `,` type($indices) `,` type($updates) `)` `->` type($output)
  }];
}

def DxgmlOp_ScatterNdOp : DxgmlOp_Op<"scatter_nd", [Pure]> {
  let summary = "Scatter updates with N-dimensional indices";
  let arguments = (ins
    AnyType:$input,
    AnyType:$indices,
    AnyType:$updates,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $indices `,` $updates `)` attr-dict `:`
    `(` type($input) `,` type($indices) `,` type($updates) `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Reduce Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_ArgmaxOp : DxgmlOp_Op<"argmax", [Pure]> {
  let summary = "Argmax along axes";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<DxgmlOp_AxisDirectionEnum>:$axis_direction,
    OptionalAttr<Dxgml_DenseIntegerElementsAttr>:$axes,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_ArgminOp : DxgmlOp_Op<"argmin", [Pure]> {
  let summary = "Argmin along axes";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<DxgmlOp_AxisDirectionEnum>:$axis_direction,
    OptionalAttr<Dxgml_DenseIntegerElementsAttr>:$axes,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_ReduceOp : DxgmlOp_Op<"reduce", [Pure]> {
  let summary = "Reduce tensor along axes";
  let arguments = (ins
    AnyType:$input,
    DxgmlOp_ReduceFunctionAttr:$reduction_function,
    OptionalAttr<Dxgml_DenseIntegerElementsAttr>:$axes,
    OptionalAttr<Dxgml_BoolAttr>:$keep_dims,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_CumulativeSummationOp : DxgmlOp_Op<"cumulative_summation", [Pure]> {
  let summary = "Cumulative sum along an axis";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_IntegerAttr>:$axis,
    OptionalAttr<Dxgml_BoolAttr>:$reverse,
    OptionalAttr<Dxgml_BoolAttr>:$exclusive,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_CumulativeProductOp : DxgmlOp_Op<"cumulative_product", [Pure]> {
  let summary = "Cumulative product along an axis";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_IntegerAttr>:$axis,
    OptionalAttr<Dxgml_BoolAttr>:$reverse,
    OptionalAttr<Dxgml_BoolAttr>:$exclusive,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_NonzeroCoordinatesOp : DxgmlOp_Op<"nonzero_coordinates", [Pure]> {
  let summary = "Return coordinates of non-zero elements";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_TopKOp : DxgmlOp_Op<"top_k", [Pure]> {
  let summary = "Top-K elements";
  let arguments = (ins
    AnyType:$input,
    Optional<AnyType>:$k,
    OptionalAttr<Dxgml_IntegerAttr>:$axis,
    OptionalAttr<DxgmlOp_AxisDirectionEnum>:$axis_direction,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output_values, AnyType:$output_indices);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input (`,` $k^)? `)` attr-dict `:`
    `(` type($input) (`,` type($k)^)? `)` `->` `(` type($output_values) `,` type($output_indices) `)`
  }];
}

//===----------------------------------------------------------------------===//
// Spatial / Image Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_ResampleOp : DxgmlOp_Op<"resample", [Pure]> {
  let summary = "Resample tensor using interpolation";
  let arguments = (ins
    AnyType:$input,
    Optional<AnyType>:$roi,
    OptionalAttr<DxgmlOp_ResizeInterpolationModeEnum>:$interpolation_mode,
    OptionalAttr<DxgmlOp_NearestModeEnum>:$nearest_mode,
    OptionalAttr<DxgmlOp_CoordinateTransformationModeEnum>:$coordinate_transform_mode,
    OptionalAttr<Dxgml_FloatAttr>:$extrapolation_value,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input (`,` $roi^)? `)` attr-dict `:`
    `(` type($input) (`,` type($roi)^)? `)` `->` type($output)
  }];
}

def DxgmlOp_ResizeOp : DxgmlOp_Op<"resize", [Pure, AttrSizedOperandSegments]> {
  let summary = "Resize tensor to target size";
  let arguments = (ins
    AnyType:$input,
    Optional<AnyType>:$roi,
    Optional<AnyType>:$scales,
    Optional<AnyType>:$sizes,
    OptionalAttr<DxgmlOp_ResizeInterpolationModeEnum>:$interpolation_mode,
    OptionalAttr<DxgmlOp_NearestModeEnum>:$nearest_mode,
    OptionalAttr<DxgmlOp_CoordinateTransformationModeEnum>:$coordinate_transform_mode,
    OptionalAttr<DxgmlOp_KeepAspectRatioPolicyEnum>:$keep_aspect_ratio_policy,
    OptionalAttr<Dxgml_FloatAttr>:$extrapolation_value,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input
    (`,` $roi^)? (`,` $scales^)? (`,` $sizes^)? `)` attr-dict `:`
    `(` type($input)
    (`,` type($roi)^)? (`,` type($scales)^)? (`,` type($sizes)^)? `)` `->` type($output)
  }];
}

def DxgmlOp_Upsample2dOp : DxgmlOp_Op<"upsample2d", [Pure]> {
  let summary = "2D upsampling";
  let arguments = (ins
    AnyType:$input,
    Optional<AnyType>:$scales,
    OptionalAttr<DxgmlOp_InterpolationModeEnum>:$interpolation_mode,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input (`,` $scales^)? `)` attr-dict `:`
    `(` type($input) (`,` type($scales)^)? `)` `->` type($output)
  }];
}

def DxgmlOp_ValueScale2dOp : DxgmlOp_Op<"value_scale2d", [Pure]> {
  let summary = "2D value scaling";
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_DenseFloatElementsAttr>:$scale,
    OptionalAttr<Dxgml_DenseFloatElementsAttr>:$bias,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_GridSampleOp : DxgmlOp_Op<"grid_sample", [Pure]> {
  let summary = "Grid sample operation";
  let arguments = (ins
    AnyType:$input,
    AnyType:$grid,
    OptionalAttr<DxgmlOp_InterpolationModeEnum>:$interpolation_mode,
    OptionalAttr<DxgmlOp_PaddingModeEnum>:$padding_mode,
    OptionalAttr<Dxgml_BoolAttr>:$align_corners,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $grid `)` attr-dict `:`
    `(` type($input) `,` type($grid) `)` `->` type($output)
  }];
}

def DxgmlOp_RoiAlignOp : DxgmlOp_Op<"roi_align", [Pure]> {
  let summary = "ROI align operation";
  let arguments = (ins
    AnyType:$input,
    AnyType:$rois,
    AnyType:$batch_indices,
    OptionalAttr<DxgmlOp_ResizeInterpolationModeEnum>:$interpolation_mode,
    OptionalAttr<Dxgml_IntegerAttr>:$output_height,
    OptionalAttr<Dxgml_IntegerAttr>:$output_width,
    OptionalAttr<Dxgml_IntegerAttr>:$sampling_ratio,
    OptionalAttr<Dxgml_FloatAttr>:$spatial_scale,
    OptionalAttr<Dxgml_BoolAttr>:$coordinate_transform_mode,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $rois `,` $batch_indices `)` attr-dict `:`
    `(` type($input) `,` type($rois) `,` type($batch_indices) `)` `->` type($output)
  }];
}

def DxgmlOp_RoiPoolingOp : DxgmlOp_Op<"roi_pooling", [Pure]> {
  let summary = "ROI pooling operation";
  let arguments = (ins
    AnyType:$input,
    AnyType:$rois,
    OptionalAttr<Dxgml_IntegerAttr>:$output_height,
    OptionalAttr<Dxgml_IntegerAttr>:$output_width,
    OptionalAttr<Dxgml_FloatAttr>:$spatial_scale,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $rois `)` attr-dict `:`
    `(` type($input) `,` type($rois) `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Padding / Fill Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_PaddingOp : DxgmlOp_Op<"padding", [Pure]> {
  let summary = "Pad tensor";
  let arguments = (ins
    AnyType:$input,
    Optional<AnyType>:$padding_value,
    Dxgml_DenseIntegerElementsAttr:$start_padding,
    Dxgml_DenseIntegerElementsAttr:$end_padding,
    OptionalAttr<DxgmlOp_PaddingModeEnum>:$padding_mode,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input (`,` $padding_value^)? `)` attr-dict `:`
    `(` type($input) (`,` type($padding_value)^)? `)` `->` type($output)
  }];
}

def DxgmlOp_FillOp : DxgmlOp_Op<"fill", [Pure]> {
  let summary = "Fill tensor with a value";
  let arguments = (ins
    AnyType:$value,
    AnyType:$output_shape,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $value `,` $output_shape `)` attr-dict `:`
    `(` type($value) `,` type($output_shape) `)` `->` type($output)
  }];
}

def DxgmlOp_FillValueConstantOp : DxgmlOp_Op<"fill_value_constant", [Pure]> {
  let summary = "Fill with constant value";
  let arguments = (ins
    AnyType:$output_shape,
    AnyAttr:$value,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $output_shape `)` attr-dict `:`
    `(` type($output_shape) `)` `->` type($output)
  }];
}

def DxgmlOp_FillValueSequenceOp : DxgmlOp_Op<"fill_value_sequence", [Pure, AttrSizedOperandSegments]> {
  let summary = "Fill with arithmetic sequence";
  let arguments = (ins
    Optional<AnyType>:$start,
    Optional<AnyType>:$delta,
    AnyType:$output_shape,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` (`,` $start^)? (`,` $delta^)? $output_shape `)` attr-dict `:`
    `(` (`,` type($start)^)? (`,` type($delta)^)? type($output_shape) `)` `->` type($output)
  }];
}

def DxgmlOp_OneHotOp : DxgmlOp_Op<"one_hot", [Pure]> {
  let summary = "One-hot encoding";
  let arguments = (ins
    AnyType:$indices,
    AnyType:$depth,
    AnyType:$values,
    OptionalAttr<Dxgml_IntegerAttr>:$axis,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $indices `,` $depth `,` $values `)` attr-dict `:`
    `(` type($indices) `,` type($depth) `,` type($values) `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Recurrent Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_LstmOp : DxgmlOp_Op<"lstm", [Pure, AttrSizedOperandSegments]> {
  let summary = "Long Short-Term Memory";
  let arguments = (ins
    AnyType:$input,
    AnyType:$weight,
    AnyType:$recurrence,
    Optional<AnyType>:$bias,
    Optional<AnyType>:$hidden_init,
    Optional<AnyType>:$cell_mem_init,
    Optional<AnyType>:$sequence_lengths,
    Optional<AnyType>:$peephole,
    OptionalAttr<Dxgml_IntegerAttr>:$hidden_size,
    OptionalAttr<DxgmlOp_RecurrentNetworkDirectionEnum>:$direction,
    OptionalAttr<Dxgml_FloatAttr>:$clip,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs
    AnyType:$output_sequence,
    AnyType:$output_single,
    AnyType:$output_cell_single
  );
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $weight `,` $recurrence
    (`,` $bias^)? (`,` $hidden_init^)? (`,` $cell_mem_init^)?
    (`,` $sequence_lengths^)? (`,` $peephole^)? `)` attr-dict `:`
    `(` type($input) `,` type($weight) `,` type($recurrence)
    (`,` type($bias)^)? (`,` type($hidden_init)^)? (`,` type($cell_mem_init)^)?
    (`,` type($sequence_lengths)^)? (`,` type($peephole)^)? `)` `->`
    `(` type($output_sequence) `,` type($output_single) `,` type($output_cell_single) `)`
  }];
}

def DxgmlOp_GruOp : DxgmlOp_Op<"gru", [Pure, AttrSizedOperandSegments]> {
  let summary = "Gated Recurrent Unit";
  let arguments = (ins
    AnyType:$input,
    AnyType:$weight,
    AnyType:$recurrence,
    Optional<AnyType>:$bias,
    Optional<AnyType>:$hidden_init,
    Optional<AnyType>:$sequence_lengths,
    OptionalAttr<Dxgml_IntegerAttr>:$hidden_size,
    OptionalAttr<DxgmlOp_RecurrentNetworkDirectionEnum>:$direction,
    OptionalAttr<Dxgml_FloatAttr>:$clip,
    OptionalAttr<Dxgml_BoolAttr>:$linear_before_reset,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs
    AnyType:$output_sequence,
    AnyType:$output_single
  );
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $weight `,` $recurrence
    (`,` $bias^)? (`,` $hidden_init^)? (`,` $sequence_lengths^)? `)` attr-dict `:`
    `(` type($input) `,` type($weight) `,` type($recurrence)
    (`,` type($bias)^)? (`,` type($hidden_init)^)? (`,` type($sequence_lengths)^)? `)` `->`
    `(` type($output_sequence) `,` type($output_single) `)`
  }];
}

def DxgmlOp_RnnOp : DxgmlOp_Op<"rnn", [Pure, AttrSizedOperandSegments]> {
  let summary = "Recurrent Neural Network";
  let arguments = (ins
    AnyType:$input,
    AnyType:$weight,
    AnyType:$recurrence,
    Optional<AnyType>:$bias,
    Optional<AnyType>:$hidden_init,
    Optional<AnyType>:$sequence_lengths,
    OptionalAttr<Dxgml_IntegerAttr>:$hidden_size,
    OptionalAttr<DxgmlOp_RecurrentNetworkDirectionEnum>:$direction,
    OptionalAttr<Dxgml_FloatAttr>:$clip,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs
    AnyType:$output_sequence,
    AnyType:$output_single
  );
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $weight `,` $recurrence
    (`,` $bias^)? (`,` $hidden_init^)? (`,` $sequence_lengths^)? `)` attr-dict `:`
    `(` type($input) `,` type($weight) `,` type($recurrence)
    (`,` type($bias)^)? (`,` type($hidden_init)^)? (`,` type($sequence_lengths)^)? `)` `->`
    `(` type($output_sequence) `,` type($output_single) `)`
  }];
}

//===----------------------------------------------------------------------===//
// Attention Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_MultiheadAttentionOp : DxgmlOp_Op<"multihead_attention", [Pure, AttrSizedOperandSegments]> {
  let summary = "Multi-head attention";
  let arguments = (ins
    AnyType:$query,
    AnyType:$key,
    AnyType:$value,
    Optional<AnyType>:$mask_index,
    Optional<AnyType>:$past,
    Optional<AnyType>:$attention_bias,
    Optional<AnyType>:$past_sequence_length,
    OptionalAttr<Dxgml_IntegerAttr>:$num_heads,
    OptionalAttr<DxgmlOp_MultiheadAttentionMaskTypeEnum>:$mask_type,
    OptionalAttr<Dxgml_FloatAttr>:$scale,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output, Optional<AnyType>:$present);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $query `,` $key `,` $value
    (`,` $mask_index^)? (`,` $past^)? (`,` $attention_bias^)?
    (`,` $past_sequence_length^)? `)` attr-dict `:`
    `(` type($query) `,` type($key) `,` type($value)
    (`,` type($mask_index)^)? (`,` type($past)^)? (`,` type($attention_bias)^)?
    (`,` type($past_sequence_length)^)? `)` `->` `(` type($output) (`,` type($present)^)? `)`
  }];
}

def DxgmlOp_GroupQueryAttentionOp : DxgmlOp_Op<"group_query_attention", [Pure]> {
  let summary = "Group query attention";
  let arguments = (ins
    AnyType:$query,
    AnyType:$key,
    AnyType:$value,
    AnyType:$past_key,
    AnyType:$past_value,
    AnyType:$seqlens_k,
    AnyType:$total_sequence_length,
    Variadic<AnyType>:$extra_inputs,
    OptionalAttr<Dxgml_IntegerAttr>:$num_heads,
    OptionalAttr<Dxgml_IntegerAttr>:$kv_num_heads,
    OptionalAttr<Dxgml_FloatAttr>:$scale,
    OptionalAttr<Dxgml_FloatAttr>:$softcap,
    OptionalAttr<Dxgml_IntegerAttr>:$do_rotary,
    OptionalAttr<AnyAttr>:$k_quant_type,
    OptionalAttr<AnyAttr>:$v_quant_type,
    OptionalAttr<StrAttr>:$op_name
  );
  let results = (outs AnyType:$output, AnyType:$present_key, AnyType:$present_value,
                      AnyType:$output_qk_matrix);
  let hasCustomAssemblyFormat = 1;
}

def DxgmlOp_RotaryEmbeddingOp : DxgmlOp_Op<"rotary_embedding", [Pure, AttrSizedOperandSegments]> {
  let summary = "Rotary position embedding";
  let arguments = (ins
    AnyType:$input,
    AnyType:$cos_cache,
    AnyType:$sin_cache,
    Optional<AnyType>:$position_ids,
    Optional<AnyType>:$past_sequence_length,
    OptionalAttr<Dxgml_IntegerAttr>:$num_heads,
    OptionalAttr<Dxgml_IntegerAttr>:$rotary_embedding_dim,
    OptionalAttr<Dxgml_IntegerAttr>:$interleaved,
    OptionalAttr<Dxgml_FloatAttr>:$scale,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `,` $cos_cache `,` $sin_cache
    (`,` $position_ids^)? (`,` $past_sequence_length^)? `)` attr-dict `:`
    `(` type($input) `,` type($cos_cache) `,` type($sin_cache)
    (`,` type($position_ids)^)? (`,` type($past_sequence_length)^)? `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Utility Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_NullPtrOp : DxgmlOp_Op<"null_ptr", [Pure]> {
  let summary = "Null pointer constant";
  let arguments = (ins);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
}

def DxgmlOp_IfOp : DxgmlOp_Op<"if", [Pure]> {
  let summary = "Conditional select";
  let arguments = (ins
    AnyType:$condition,
    AnyType:$true_val,
    AnyType:$false_val,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $condition `,` $true_val `,` $false_val `)` attr-dict `:`
    `(` type($condition) `,` type($true_val) `,` type($false_val) `)` `->` type($output)
  }];
}

def DxgmlOp_UnfoldOp : DxgmlOp_Op<"unfold", [Pure]> {
  let summary = "Unfold/extract sliding windows";
  let arguments = (ins
    AnyType:$input,
    Dxgml_DenseIntegerElementsAttr:$window_strides,
    Dxgml_DenseIntegerElementsAttr:$window_size,
    Dxgml_DenseIntegerElementsAttr:$start_padding,
    Dxgml_DenseIntegerElementsAttr:$end_padding,
    OptionalAttr<Dxgml_DenseIntegerElementsAttr>:$dilations,
    OptionalAttr<Dxgml_StrAttr>:$op_name
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    (`[` $op_name^ `]`)? `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

#endif // DXGMLOP
