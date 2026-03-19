//===- DxgmlOp.td - Dxgml operation definitions -----------*- tablegen -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR Dxgml dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef DXGMLOP
#define DXGMLOP

include "mlir/Dialect/Dxgml/IR/DxgmlBase.td"
include "mlir/Dialect/Dxgml/IR/DxgmlAttrs.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Constant Operation
//===----------------------------------------------------------------------===//

def DxgmlOp_ConstantOp : Dxgml_Op<"Constant", [Pure, ConstantLike]> {
  let summary = "Constant value operation";
  let description = [{
    Creates a constant tensor or scalar value from an attribute.
    
    Example:
    ```mlir
    %w = DxGML.Constant(#DxGML.ConstantResource<weight.1>) : !DxGML.Tensor<32x4x3x3x!DxGML.Float16>
    %c = DxGML.Constant(#DxGML.ConstantValue<[1]>) : !DxGML.Tensor<1x!DxGML.Int64>
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

def DxgmlOp_ConvolutionOp : Dxgml_Op<"Convolution", [Pure]> {
  let summary = "Convolution operation";
  let description = [{
    Applies convolution over an input tensor.
    
    Example:
    ```mlir
    %0 = DxGML.Convolution(%input, %filter, %bias) {
      group_count = #DxGML.ConstantValue<[1]> : !DxGML.Tensor<1x!DxGML.Int64>,
      dilations = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      start_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      end_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      strides = #DxGML.ConstantValue<[2, 2]> : !DxGML.Tensor<2x!DxGML.Int64>
    } : (!DxGML.Tensor<1x4x224x224x!DxGML.Float16>, 
         !DxGML.Tensor<32x4x3x3x!DxGML.Float16>,
         !DxGML.Tensor<32x!DxGML.Float16>) 
      -> !DxGML.Tensor<1x32x112x112x!DxGML.Float16>
    ```
  }];
  
  let arguments = (ins
    AnyType:$input,
    AnyType:$filter,
    Optional<AnyType>:$bias,
    OptionalAttr<Dxgml_ConvolutionModeEnumAttr>:$mode,
    OptionalAttr<Dxgml_ConvolutionDirectionEnumAttr>:$direction,
    Dxgml_ConstantAttr:$strides,
    Dxgml_ConstantAttr:$dilations,
    Dxgml_ConstantAttr:$start_padding,
    Dxgml_ConstantAttr:$end_padding,
    Dxgml_ConstantAttr:$group_count
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

def DxgmlOp_ReluOp : Dxgml_Op<"Relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU activation";
  let description = [{
    Applies ReLU activation: max(x, 0)
    
    Example:
    ```mlir
    %0 = DxGML.Relu(%input) : (!DxGML.Tensor<1x32x224x224x!DxGML.Float16>) 
                                -> !DxGML.Tensor<1x32x224x224x!DxGML.Float16>
    ```
  }];
  
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_SigmoidOp : Dxgml_Op<"Sigmoid", [Pure, SameOperandsAndResultType]> {
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

def DxgmlOp_TanhOp : Dxgml_Op<"Tanh", [Pure, SameOperandsAndResultType]> {
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

def DxgmlOp_AbsOp : Dxgml_Op<"Abs", [Pure, SameOperandsAndResultType]> {
  let summary = "Absolute value";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_ExpOp : Dxgml_Op<"Exp", [Pure, SameOperandsAndResultType]> {
  let summary = "Exponential";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_LogOp : Dxgml_Op<"Log", [Pure, SameOperandsAndResultType]> {
  let summary = "Natural logarithm";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_SqrtOp : Dxgml_Op<"Sqrt", [Pure, SameOperandsAndResultType]> {
  let summary = "Square root";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_NegateOp : Dxgml_Op<"Negate", [Pure, SameOperandsAndResultType]> {
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

def DxgmlOp_AddOp : Dxgml_Op<"Add", [Pure]> {
  let summary = "Element-wise addition";
  let description = [{
    Adds two tensors element-wise with broadcasting.
    
    Example:
    ```mlir
    %0 = DxGML.Add(%a, %b) : (!DxGML.Tensor<1x32x224x224x!DxGML.Float16>,
                              !DxGML.Tensor<1x32x224x224x!DxGML.Float16>)
                           -> !DxGML.Tensor<1x32x224x224x!DxGML.Float16>
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

def DxgmlOp_SubtractOp : Dxgml_Op<"Subtract", [Pure]> {
  let summary = "Element-wise subtraction";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type($output)
  }];
}

def DxgmlOp_MultiplyOp : Dxgml_Op<"Multiply", [Pure]> {
  let summary = "Element-wise multiplication";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type($output)
  }];
}

def DxgmlOp_DivideOp : Dxgml_Op<"Divide", [Pure]> {
  let summary = "Element-wise division";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type($output)
  }];
}

def DxgmlOp_MaxOp : Dxgml_Op<"Max", [Pure]> {
  let summary = "Element-wise maximum";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type($output)
  }];
}

def DxgmlOp_MinOp : Dxgml_Op<"Min", [Pure]> {
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

def DxgmlOp_GemmOp : Dxgml_Op<"Gemm", [Pure]> {
  let summary = "General Matrix Multiplication";
  let description = [{
    Performs GEMM operation: alpha * A * B + beta * C
  }];
  
  let arguments = (ins
    AnyType:$a_tensor,
    AnyType:$b_tensor,
    Optional<AnyType>:$c_tensor,
    OptionalAttr<Dxgml_ConstantAttr>:$alpha,
    OptionalAttr<Dxgml_ConstantAttr>:$beta
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

def DxgmlOp_ReshapeOp : Dxgml_Op<"Reshape", [Pure]> {
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

def DxgmlOp_TransposeOp : Dxgml_Op<"Transpose", [Pure]> {
  let summary = "Transpose tensor";
  let description = [{
    Transposes a tensor according to permutation.
  }];
  
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Dxgml_ConstantAttr>:$permutation
  );
  
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_SliceOp : Dxgml_Op<"Slice", [Pure]> {
  let summary = "Slice tensor";
  let description = [{
    Extracts a slice from a tensor.
  }];
  
  let arguments = (ins
    AnyType:$input,
    Dxgml_ConstantAttr:$starts,
    Dxgml_ConstantAttr:$ends,
    OptionalAttr<Dxgml_ConstantAttr>:$axes,
    OptionalAttr<Dxgml_ConstantAttr>:$step
  );
  
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_ConcatOp : Dxgml_Op<"Concat", [Pure]> {
  let summary = "Concatenate tensors";
  let description = [{
    Concatenates tensors along a specified axis.
  }];
  
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Dxgml_ConstantAttr:$axis
  );
  
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $inputs `)` attr-dict `:`
    `(` type($inputs) `)` `->` type($output)
  }];
}

def DxgmlOp_BroadcastOp : Dxgml_Op<"Broadcast", [Pure]> {
  let summary = "Broadcast tensor";
  let description = [{
    Broadcasts a tensor to a target shape.
  }];
  
  let arguments = (ins
    AnyType:$input,
    Optional<AnyType>:$shape,
    OptionalAttr<Dxgml_ConstantAttr>:$target_shape
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

def DxgmlOp_DepthToSpaceOp : Dxgml_Op<"DepthToSpace", [Pure]> {
  let summary = "Depth to space transformation";
  let description = [{
    Rearranges data from depth into spatial dimensions.
    
    Example:
    ```mlir
    %0 = DxGML.DepthToSpace(%input) {
      block_size = #DxGML.ConstantValue<[2]> : !DxGML.Tensor<1x!DxGML.Int64>,
      depth_space_order = #DxGML.DepthSpaceOrderEnumAttr<DEPTH_SPACE_ORDER_COLUMN_ROW_DEPTH>
    } : (!DxGML.Tensor<1x16x1080x1920x!DxGML.Float16>) 
      -> !DxGML.Tensor<1x4x2160x3840x!DxGML.Float16>
    ```
  }];
  
  let arguments = (ins
    AnyType:$input,
    Dxgml_ConstantAttr:$block_size,
    Dxgml_DepthSpaceOrderEnumAttr:$depth_space_order
  );
  
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:`
    `(` type($input) `)` `->` type($output)
  }];
}

def DxgmlOp_SpaceToDepthOp : Dxgml_Op<"SpaceToDepth", [Pure]> {
  let summary = "Space to depth transformation";
  let description = [{
    Rearranges data from spatial dimensions into depth.
  }];
  
  let arguments = (ins
    AnyType:$input,
    Dxgml_ConstantAttr:$block_size,
    Dxgml_DepthSpaceOrderEnumAttr:$depth_space_order
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

def DxgmlOp_CastOp : Dxgml_Op<"Cast", [Pure]> {
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

def DxgmlOp_IdentityOp : Dxgml_Op<"Identity", [Pure, SameOperandsAndResultType]> {
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
