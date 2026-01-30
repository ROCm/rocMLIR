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
  
  let assemblyFormat = "`(` $value `)` attr-dict `:` type($result)";
  
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
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$dilations,
    DenseI64ArrayAttr:$start_padding,
    DenseI64ArrayAttr:$end_padding,
    I64Attr:$group_count
  );
  
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $input `,` $filter (`,` $bias^)? `)` attr-dict `:`
    `(` type($input) `,` type($filter) (`,` type($bias)^)? `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Elementwise Unary Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_ReluOp : DxgmlOp_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU activation";
  let description = [{
    Applies ReLU activation: max(x, 0)
    
    Example:
    ```mlir
    %0 = dxgml_op.relu(%input) : (!dxgml.tensor<1x32x224x224x!dxgml.float16>) 
                                -> !dxgml.tensor<1x32x224x224x!dxgml.float16>
    ```
  }];
  
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_SigmoidOp : DxgmlOp_Op<"sigmoid", [Pure, SameOperandsAndResultType]> {
  let summary = "Sigmoid activation";
  let description = [{
    Applies sigmoid activation: 1 / (1 + exp(-x))
  }];
  
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_TanhOp : DxgmlOp_Op<"tanh", [Pure, SameOperandsAndResultType]> {
  let summary = "Tanh activation";
  let description = [{
    Applies hyperbolic tangent activation
  }];
  
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_AbsOp : DxgmlOp_Op<"abs", [Pure, SameOperandsAndResultType]> {
  let summary = "Absolute value";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_ExpOp : DxgmlOp_Op<"exp", [Pure, SameOperandsAndResultType]> {
  let summary = "Exponential";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_LogOp : DxgmlOp_Op<"log", [Pure, SameOperandsAndResultType]> {
  let summary = "Natural logarithm";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_SqrtOp : DxgmlOp_Op<"sqrt", [Pure, SameOperandsAndResultType]> {
  let summary = "Square root";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_NegateOp : DxgmlOp_Op<"negate", [Pure, SameOperandsAndResultType]> {
  let summary = "Negation";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` `(` type($input) `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Elementwise Binary Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_AddOp : DxgmlOp_Op<"add", [Pure]> {
  let summary = "Element-wise addition";
  let description = [{
    Adds two tensors element-wise with broadcasting.
    
    Example:
    ```mlir
    %0 = dxgml_op.add(%a, %b) : (!dxgml.tensor<1x32x224x224x!dxgml.float16>,
                                 !dxgml.tensor<1x32x224x224x!dxgml.float16>)
                              -> !dxgml.tensor<1x32x224x224x!dxgml.float16>
    ```
  }];
  
  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs
  );
  
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type($output)
  }];
}

def DxgmlOp_SubtractOp : DxgmlOp_Op<"subtract", [Pure]> {
  let summary = "Element-wise subtraction";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type($output)
  }];
}

def DxgmlOp_MultiplyOp : DxgmlOp_Op<"multiply", [Pure]> {
  let summary = "Element-wise multiplication";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type($output)
  }];
}

def DxgmlOp_DivideOp : DxgmlOp_Op<"divide", [Pure]> {
  let summary = "Element-wise division";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type($output)
  }];
}

def DxgmlOp_MaxOp : DxgmlOp_Op<"max", [Pure]> {
  let summary = "Element-wise maximum";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type($output)
  }];
}

def DxgmlOp_MinOp : DxgmlOp_Op<"min", [Pure]> {
  let summary = "Element-wise minimum";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Matrix Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_GemmOp : DxgmlOp_Op<"gemm", [Pure]> {
  let summary = "General Matrix Multiplication";
  let description = [{
    Performs GEMM operation: alpha * A * B + beta * C
  }];
  
  let arguments = (ins
    AnyType:$a_tensor,
    AnyType:$b_tensor,
    Optional<AnyType>:$c_tensor,
    OptionalAttr<F64Attr>:$alpha,
    OptionalAttr<F64Attr>:$beta
  );
  
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $a_tensor `,` $b_tensor (`,` $c_tensor^)? `)` attr-dict `:`
    `(` type($a_tensor) `,` type($b_tensor) (`,` type($c_tensor)^)? `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Shape Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_ReshapeOp : DxgmlOp_Op<"reshape", [Pure]> {
  let summary = "Reshape tensor";
  let description = [{
    Reshapes a tensor to a new shape.
  }];
  
  let arguments = (ins
    AnyType:$inputTensor,
    Optional<AnyType>:$shape
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    `(` $inputTensor (`,` $shape^)? `)` attr-dict `:`
    `(` type($inputTensor) (`,` type($shape)^)? `)` `->` type($result)
  }];
}

def DxgmlOp_TransposeOp : DxgmlOp_Op<"transpose", [Pure]> {
  let summary = "Transpose tensor";
  let description = [{
    Transposes a tensor according to permutation.
  }];
  
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<DenseI64ArrayAttr>:$permutation
  );
  
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_SliceOp : DxgmlOp_Op<"slice", [Pure]> {
  let summary = "Slice tensor";
  let description = [{
    Extracts a slice from a tensor.
  }];
  
  let arguments = (ins
    AnyType:$input,
    DenseI64ArrayAttr:$starts,
    DenseI64ArrayAttr:$ends,
    OptionalAttr<DenseI64ArrayAttr>:$axes,
    OptionalAttr<DenseI64ArrayAttr>:$step
  );
  
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_ConcatOp : DxgmlOp_Op<"concat", [Pure]> {
  let summary = "Concatenate tensors";
  let description = [{
    Concatenates tensors along a specified axis.
  }];
  
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    I64Attr:$axis
  );
  
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $inputs `)` attr-dict `:`
    `(` type($inputs) `)` `->` type($output)
  }];
}

def DxgmlOp_BroadcastOp : DxgmlOp_Op<"broadcast", [Pure]> {
  let summary = "Broadcast tensor";
  let description = [{
    Broadcasts a tensor to a target shape.
  }];
  
  let arguments = (ins
    AnyType:$input,
    Optional<AnyType>:$shape,
    OptionalAttr<DenseI64ArrayAttr>:$target_shape
  );
  
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $input (`,` $shape^)? `)` attr-dict `:`
    `(` type($input) (`,` type($shape)^)? `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Depth/Space Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_DepthToSpaceOp : DxgmlOp_Op<"depth_to_space", [Pure]> {
  let summary = "Depth to space transformation";
  let description = [{
    Rearranges data from depth into spatial dimensions.
    
    Example:
    ```mlir
    %0 = dxgml_op.depth_to_space(%input) {
      block_size = #dxgml.integer<2 : !dxgml.int64>,
      depth_space_order = #dxgml_op.depth_space_order_enum_attr<depth_space_order_column_row_depth>
    } : (!dxgml.tensor<1x16x1080x1920x!dxgml.float16>) 
      -> !dxgml.tensor<1x4x2160x3840x!dxgml.float16>
    ```
  }];
  
  let arguments = (ins
    AnyType:$input,
    I64Attr:$block_size,
    DxgmlOp_DepthSpaceOrderEnum:$depth_space_order
  );
  
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_SpaceToDepthOp : DxgmlOp_Op<"space_to_depth", [Pure]> {
  let summary = "Space to depth transformation";
  let description = [{
    Rearranges data from spatial dimensions into depth.
  }];
  
  let arguments = (ins
    AnyType:$input,
    I64Attr:$block_size,
    DxgmlOp_DepthSpaceOrderEnum:$depth_space_order
  );
  
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Utility Operations
//===----------------------------------------------------------------------===//

def DxgmlOp_CastOp : DxgmlOp_Op<"cast", [Pure]> {
  let summary = "Type cast operation";
  let description = [{
    Casts tensor elements to a different type.
  }];
  
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_IdentityOp : DxgmlOp_Op<"identity", [Pure, SameOperandsAndResultType]> {
  let summary = "Identity operation";
  let description = [{
    Returns input unchanged.
  }];
  
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

#endif // DXGMLOP
