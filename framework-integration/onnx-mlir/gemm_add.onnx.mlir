module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "gemm_add"} {
  func.func @main_graph(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>) -> (tensor<10x10xf32>) {
    %0 = "onnx.MatMul"(%arg0, %arg1) {onnx_node_name = "onnx.MatMul_0"} : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    return %0 : tensor<10x10xf32>
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
