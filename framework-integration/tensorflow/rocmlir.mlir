module {
  func.func @forward(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>, %arg2: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %0 = tensor.empty() : tensor<10x10xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<10x10xf32>) -> tensor<10x10xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<10x10xf32>, tensor<10x10xf32>) outs(%1 : tensor<10x10xf32>) -> tensor<10x10xf32>
    %3 = tensor.empty() : tensor<10x10xf32>
    %mapped = linalg.map { arith.addf } ins(%2, %arg2 : tensor<10x10xf32>, tensor<10x10xf32>) outs(%3 : tensor<10x10xf32>)
    return %mapped : tensor<10x10xf32>
  }
}

