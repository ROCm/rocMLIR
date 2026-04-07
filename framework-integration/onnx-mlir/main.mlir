module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "gemm_add"} {
  func.func @main_graph(%arg0: tensor<10x10xf32> , %arg1: tensor<10x10xf32> ) -> (tensor<10x10xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<10x10xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<10x10xf32>) -> tensor<10x10xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<10x10xf32>, tensor<10x10xf32>) outs(%1 : tensor<10x10xf32>) -> tensor<10x10xf32>
    return %2 : tensor<10x10xf32>
  }
}
