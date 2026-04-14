module attributes {gpu.container_module} {
  func.func @simple_gemm_relu(%arg0: !migraphx.shaped<2x4xf16, 4x1>, %arg1: !migraphx.shaped<4x3xf16, 3x1>, %arg2: !migraphx.shaped<2x3xf16, 3x1>) -> !migraphx.shaped<2x3xf16, 3x1> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.0.0"} {
    %dot_3 = migraphx.dot %arg0, %arg1 : <2x4xf16, 4x1>, <4x3xf16, 3x1> -> <2x3xf16, 3x1>
    %add_4 = migraphx.add %dot_3, %arg2 : <2x3xf16, 3x1>, <2x3xf16, 3x1> -> <2x3xf16, 3x1>
    %relu_5 = migraphx.relu %add_4 : <2x3xf16, 3x1> -> <2x3xf16, 3x1>
    return %relu_5 : !migraphx.shaped<2x3xf16, 3x1>
  }
  gpu.module @rock_gpu_module {
  }
}
