//===- DxgmlTypes.td - Dxgml type definitions -------------*- tablegen -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR Dxgml dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef DXGML_TYPES
#define DXGML_TYPES

include "mlir/Dialect/Dxgml/IR/DxgmlBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Dxgml Tensor Type
//===----------------------------------------------------------------------===//

def Dxgml_DxgmlTensorType : Dxgml_Type<"DxgmlTensor", "tensor"> {
  let summary = "Multi-dimensional tensor type";
  let description = [{
    Represents a multi-dimensional array with fixed shape and element type.
    
    Syntax: `!dxgml.tensor<DxDx...xElementType>`
    
    Example:
    ```mlir
    !dxgml.tensor<1x32x1080x1920xf16>
    !dxgml.tensor<32xi64>
    ```
  }];
  
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );
  
  let assemblyFormat = "`<` custom<DxgmlTensorType>($shape, $elementType) `>`";
  
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType), [{
      return $_get(elementType.getContext(), shape, elementType);
    }]>
  ];
  
  let extraClassDeclaration = [{
    /// Get the number of dimensions
    unsigned getRank() const { return getShape().size(); }
    
    /// Get the number of elements
    int64_t getNumElements() const {
      int64_t num = 1;
      for (int64_t dim : getShape())
        num *= dim;
      return num;
    }
    
    /// Check if all dimensions are static
    bool hasStaticShape() const {
      return llvm::none_of(getShape(), [](int64_t dim) { return dim < 0; });
    }
  }];
}

#endif // DXGML_TYPES
