module attributes {gpu.container_module} {
  func.func @conv_bn_relu_pool(%arg0: !migraphx.shaped<1x3x32x32xf32, 3072x1024x32x1>) -> !migraphx.shaped<1x64x7x7xf32, 3136x49x7x1> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.0.0"} {
    %literal_3 = migraphx.literal(dense<0.000000e+00> : tensor<64x3x3x3xf32>) : <64x3x3x3xf32, 27x9x3x1>
    %literal_4 = migraphx.literal(dense<0.000000e+00> : tensor<64xf32>) : <64xf32, 1>
    %literal_5 = migraphx.literal(dense<0.000000e+00> : tensor<64xf32>) : <64xf32, 1>
    %literal_6 = migraphx.literal(dense<0.000000e+00> : tensor<64xf32>) : <64xf32, 1>
    %literal_7 = migraphx.literal(dense<0.000000e+00> : tensor<64xf32>) : <64xf32, 1>
    %literal_8 = migraphx.literal(dense<0.000000e+00> : tensor<64xf32>) : <64xf32, 1>
    %convolution_9 = migraphx.convolution %arg0, %literal_3 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], stride = [2, 2]} : <1x3x32x32xf32, 3072x1024x32x1>, <64x3x3x3xf32, 27x9x3x1> -> <1x64x16x16xf32, 16384x256x16x1>
    %multibroadcast_10 = migraphx.multibroadcast %literal_4 {out_lens = [1, 64, 16, 16]} : <64xf32, 1> -> <1x64x16x16xf32, 0x1x0x0>
    %add_11 = migraphx.add %convolution_9, %multibroadcast_10 : <1x64x16x16xf32, 16384x256x16x1>, <1x64x16x16xf32, 0x1x0x0> -> <1x64x16x16xf32, 16384x256x16x1>
    %batch_norm_inference_12 = migraphx.batch_norm_inference %add_11, %literal_5, %literal_6, %literal_7, %literal_8 {bn_mode = 1 : i64, epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : !migraphx.shaped<1x64x16x16xf32, 16384x256x16x1>, !migraphx.shaped<64xf32, 1>, !migraphx.shaped<64xf32, 1>, !migraphx.shaped<64xf32, 1>, !migraphx.shaped<64xf32, 1> -> <1x64x16x16xf32, 16384x256x16x1>
    %relu_13 = migraphx.relu %batch_norm_inference_12 : <1x64x16x16xf32, 16384x256x16x1> -> <1x64x16x16xf32, 16384x256x16x1>
    %pooling_14 = migraphx.pooling %relu_13 {ceil_mode = 0 : i64, length = [3, 3], mode = "max", padding = [0, 0, 0, 0], stride = [2, 2]} : <1x64x16x16xf32, 16384x256x16x1> -> <1x64x7x7xf32, 3136x49x7x1>
    return %pooling_14 : !migraphx.shaped<1x64x7x7xf32, 3136x49x7x1>
  }
  gpu.module @rock_gpu_module {
  }
}
