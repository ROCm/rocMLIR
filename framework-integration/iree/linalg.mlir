module {
  func.func @testing(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<100xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x10x10xf32>
    %expanded = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [1, 10, 10] : tensor<100xf32> into tensor<1x10x10xf32>
    %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [1, 10, 10] : tensor<100xf32> into tensor<1x10x10xf32>
    %0 = linalg.batch_matmul ins(%expanded, %expanded_0 : tensor<1x10x10xf32>, tensor<1x10x10xf32>) outs(%cst : tensor<1x10x10xf32>) -> tensor<1x10x10xf32>
    %collapsed = tensor.collapse_shape %0 [[0, 1, 2]] : tensor<1x10x10xf32> into tensor<100xf32>
    return %collapsed : tensor<100xf32>
  }
}

