module {
  func.func @torch_jit(%arg0: !migraphx.shaped<1x4x2160x3840xf16, 33177600x8294400x3840x1>) -> !migraphx.shaped<1x4x2160x3840xf16, 33177600x8294400x3840x1> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {aimet_torch = 1 : si64}, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "1.13.1"} {
    %_conv1_weight = migraphx.literal(dense<0.000000e+00> : tensor<32x4x3x3xf16>) : <32x4x3x3xf16, 36x9x3x1>
    %_conv1_bias = migraphx.literal(dense<0.000000e+00> : tensor<32xf16>) : <32xf16, 1>
    %_RDB1_conv1_weight = migraphx.literal(dense<0.000000e+00> : tensor<32x32x3x3xf16>) : <32x32x3x3xf16, 288x9x3x1>
    %_RDB1_conv1_bias = migraphx.literal(dense<0.000000e+00> : tensor<32xf16>) : <32xf16, 1>
    %_RDB1_conv2_weight = migraphx.literal(dense<0.000000e+00> : tensor<32x32x3x3xf16>) : <32x32x3x3xf16, 288x9x3x1>
    %_RDB1_conv2_bias = migraphx.literal(dense<0.000000e+00> : tensor<32xf16>) : <32xf16, 1>
    %_RDB1_conv3_weight = migraphx.literal(dense<0.000000e+00> : tensor<32x32x3x3xf16>) : <32x32x3x3xf16, 288x9x3x1>
    %_RDB1_conv3_bias = migraphx.literal(dense<0.000000e+00> : tensor<32xf16>) : <32xf16, 1>
    %_RDB2_conv1_weight = migraphx.literal(dense<0.000000e+00> : tensor<32x32x3x3xf16>) : <32x32x3x3xf16, 288x9x3x1>
    %_RDB2_conv1_bias = migraphx.literal(dense<0.000000e+00> : tensor<32xf16>) : <32xf16, 1>
    %_RDB2_conv2_weight = migraphx.literal(dense<0.000000e+00> : tensor<32x32x3x3xf16>) : <32x32x3x3xf16, 288x9x3x1>
    %_RDB2_conv2_bias = migraphx.literal(dense<0.000000e+00> : tensor<32xf16>) : <32xf16, 1>
    %_RDB2_conv3_weight = migraphx.literal(dense<0.000000e+00> : tensor<32x32x3x3xf16>) : <32x32x3x3xf16, 288x9x3x1>
    %_RDB2_conv3_bias = migraphx.literal(dense<0.000000e+00> : tensor<32xf16>) : <32xf16, 1>
    %_RDB3_conv1_weight = migraphx.literal(dense<0.000000e+00> : tensor<32x32x3x3xf16>) : <32x32x3x3xf16, 288x9x3x1>
    %_RDB3_conv1_bias = migraphx.literal(dense<0.000000e+00> : tensor<32xf16>) : <32xf16, 1>
    %_RDB3_conv2_weight = migraphx.literal(dense<0.000000e+00> : tensor<32x32x3x3xf16>) : <32x32x3x3xf16, 288x9x3x1>
    %_RDB3_conv2_bias = migraphx.literal(dense<0.000000e+00> : tensor<32xf16>) : <32xf16, 1>
    %_RDB3_conv3_weight = migraphx.literal(dense<0.000000e+00> : tensor<32x32x3x3xf16>) : <32x32x3x3xf16, 288x9x3x1>
    %_RDB3_conv3_bias = migraphx.literal(dense<0.000000e+00> : tensor<32xf16>) : <32xf16, 1>
    %_conv_post_weight = migraphx.literal(dense<0.000000e+00> : tensor<96x32x3x3xf16>) : <96x32x3x3xf16, 288x9x3x1>
    %_conv_post_bias = migraphx.literal(dense<0.000000e+00> : tensor<96xf16>) : <96xf16, 1>
    %_conv_final_weight = migraphx.literal(dense<0.000000e+00> : tensor<16x96x1x1xf16>) : <16x96x1x1xf16, 96x1x1x1>
    %_conv_final_bias = migraphx.literal(dense<0.000000e+00> : tensor<16xf16>) : <16xf16, 1>
    %conv1 = migraphx.convolution %arg0, %_conv1_weight {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], stride = [2, 2]} : <1x4x2160x3840xf16, 33177600x8294400x3840x1>, <32x4x3x3xf16, 36x9x3x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %conv1_bias_bcast = migraphx.multibroadcast %_conv1_bias {out_lens = [1, 32, 1080, 1920]} : <32xf16, 1> -> <1x32x1080x1920xf16, 0x1x0x0>
    %conv1_out = migraphx.add %conv1, %conv1_bias_bcast : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 0x1x0x0> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb1_conv1 = migraphx.convolution %conv1_out, %_RDB1_conv1_weight {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <32x32x3x3xf16, 288x9x3x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb1_conv1_bias_bcast = migraphx.multibroadcast %_RDB1_conv1_bias {out_lens = [1, 32, 1080, 1920]} : <32xf16, 1> -> <1x32x1080x1920xf16, 0x1x0x0>
    %rdb1_conv1_out = migraphx.add %rdb1_conv1, %rdb1_conv1_bias_bcast : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 0x1x0x0> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb1_relu1 = migraphx.relu %rdb1_conv1_out : <1x32x1080x1920xf16, 66355200x2073600x1920x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb1_add1 = migraphx.add %rdb1_relu1, %conv1_out : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 66355200x2073600x1920x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb1_conv2 = migraphx.convolution %rdb1_add1, %_RDB1_conv2_weight {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <32x32x3x3xf16, 288x9x3x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb1_conv2_bias_bcast = migraphx.multibroadcast %_RDB1_conv2_bias {out_lens = [1, 32, 1080, 1920]} : <32xf16, 1> -> <1x32x1080x1920xf16, 0x1x0x0>
    %rdb1_conv2_out = migraphx.add %rdb1_conv2, %rdb1_conv2_bias_bcast : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 0x1x0x0> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb1_relu2 = migraphx.relu %rdb1_conv2_out : <1x32x1080x1920xf16, 66355200x2073600x1920x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb1_add2 = migraphx.add %rdb1_add1, %rdb1_relu2 : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 66355200x2073600x1920x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb1_conv3 = migraphx.convolution %rdb1_add2, %_RDB1_conv3_weight {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <32x32x3x3xf16, 288x9x3x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb1_conv3_bias_bcast = migraphx.multibroadcast %_RDB1_conv3_bias {out_lens = [1, 32, 1080, 1920]} : <32xf16, 1> -> <1x32x1080x1920xf16, 0x1x0x0>
    %rdb1_conv3_out = migraphx.add %rdb1_conv3, %rdb1_conv3_bias_bcast : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 0x1x0x0> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb1_block_out = migraphx.add %rdb1_conv3_out, %conv1_out : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 66355200x2073600x1920x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb2_conv1 = migraphx.convolution %rdb1_block_out, %_RDB2_conv1_weight {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <32x32x3x3xf16, 288x9x3x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb2_conv1_bias_bcast = migraphx.multibroadcast %_RDB2_conv1_bias {out_lens = [1, 32, 1080, 1920]} : <32xf16, 1> -> <1x32x1080x1920xf16, 0x1x0x0>
    %rdb2_conv1_out = migraphx.add %rdb2_conv1, %rdb2_conv1_bias_bcast : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 0x1x0x0> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb2_relu1 = migraphx.relu %rdb2_conv1_out : <1x32x1080x1920xf16, 66355200x2073600x1920x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb2_add1 = migraphx.add %rdb2_relu1, %rdb1_block_out : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 66355200x2073600x1920x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb2_conv2 = migraphx.convolution %rdb2_add1, %_RDB2_conv2_weight {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <32x32x3x3xf16, 288x9x3x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb2_conv2_bias_bcast = migraphx.multibroadcast %_RDB2_conv2_bias {out_lens = [1, 32, 1080, 1920]} : <32xf16, 1> -> <1x32x1080x1920xf16, 0x1x0x0>
    %rdb2_conv2_out = migraphx.add %rdb2_conv2, %rdb2_conv2_bias_bcast : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 0x1x0x0> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb2_relu2 = migraphx.relu %rdb2_conv2_out : <1x32x1080x1920xf16, 66355200x2073600x1920x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb2_add2 = migraphx.add %rdb2_add1, %rdb2_relu2 : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 66355200x2073600x1920x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb2_conv3 = migraphx.convolution %rdb2_add2, %_RDB2_conv3_weight {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <32x32x3x3xf16, 288x9x3x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb2_conv3_bias_bcast = migraphx.multibroadcast %_RDB2_conv3_bias {out_lens = [1, 32, 1080, 1920]} : <32xf16, 1> -> <1x32x1080x1920xf16, 0x1x0x0>
    %rdb2_conv3_out = migraphx.add %rdb2_conv3, %rdb2_conv3_bias_bcast : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 0x1x0x0> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb2_block_out = migraphx.add %rdb2_conv3_out, %rdb1_block_out : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 66355200x2073600x1920x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb3_conv1 = migraphx.convolution %rdb2_block_out, %_RDB3_conv1_weight {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <32x32x3x3xf16, 288x9x3x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb3_conv1_bias_bcast = migraphx.multibroadcast %_RDB3_conv1_bias {out_lens = [1, 32, 1080, 1920]} : <32xf16, 1> -> <1x32x1080x1920xf16, 0x1x0x0>
    %rdb3_conv1_out = migraphx.add %rdb3_conv1, %rdb3_conv1_bias_bcast : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 0x1x0x0> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb3_relu1 = migraphx.relu %rdb3_conv1_out : <1x32x1080x1920xf16, 66355200x2073600x1920x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb3_add1 = migraphx.add %rdb3_relu1, %rdb2_block_out : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 66355200x2073600x1920x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb3_conv2 = migraphx.convolution %rdb3_add1, %_RDB3_conv2_weight {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <32x32x3x3xf16, 288x9x3x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb3_conv2_bias_bcast = migraphx.multibroadcast %_RDB3_conv2_bias {out_lens = [1, 32, 1080, 1920]} : <32xf16, 1> -> <1x32x1080x1920xf16, 0x1x0x0>
    %rdb3_conv2_out = migraphx.add %rdb3_conv2, %rdb3_conv2_bias_bcast : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 0x1x0x0> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb3_relu2 = migraphx.relu %rdb3_conv2_out : <1x32x1080x1920xf16, 66355200x2073600x1920x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb3_add2 = migraphx.add %rdb3_add1, %rdb3_relu2 : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 66355200x2073600x1920x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb3_conv3 = migraphx.convolution %rdb3_add2, %_RDB3_conv3_weight {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <32x32x3x3xf16, 288x9x3x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb3_conv3_bias_bcast = migraphx.multibroadcast %_RDB3_conv3_bias {out_lens = [1, 32, 1080, 1920]} : <32xf16, 1> -> <1x32x1080x1920xf16, 0x1x0x0>
    %rdb3_conv3_out = migraphx.add %rdb3_conv3, %rdb3_conv3_bias_bcast : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 0x1x0x0> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %rdb3_block_out = migraphx.add %rdb3_conv3_out, %rdb2_block_out : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <1x32x1080x1920xf16, 66355200x2073600x1920x1> -> <1x32x1080x1920xf16, 66355200x2073600x1920x1>
    %conv_post = migraphx.convolution %rdb3_block_out, %_conv_post_weight {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x32x1080x1920xf16, 66355200x2073600x1920x1>, <96x32x3x3xf16, 288x9x3x1> -> <1x96x1080x1920xf16, 199065600x2073600x1920x1>
    %conv_post_bias_bcast = migraphx.multibroadcast %_conv_post_bias {out_lens = [1, 96, 1080, 1920]} : <96xf16, 1> -> <1x96x1080x1920xf16, 0x1x0x0>
    %conv_post_out = migraphx.add %conv_post, %conv_post_bias_bcast : <1x96x1080x1920xf16, 199065600x2073600x1920x1>, <1x96x1080x1920xf16, 0x1x0x0> -> <1x96x1080x1920xf16, 199065600x2073600x1920x1>
    %conv_post_relu = migraphx.relu %conv_post_out : <1x96x1080x1920xf16, 199065600x2073600x1920x1> -> <1x96x1080x1920xf16, 199065600x2073600x1920x1>
    %conv_final = migraphx.convolution %conv_post_relu, %_conv_final_weight {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x96x1080x1920xf16, 199065600x2073600x1920x1>, <16x96x1x1xf16, 96x1x1x1> -> <1x16x1080x1920xf16, 33177600x2073600x1920x1>
    %conv_final_bias_bcast = migraphx.multibroadcast %_conv_final_bias {out_lens = [1, 16, 1080, 1920]} : <16xf16, 1> -> <1x16x1080x1920xf16, 0x1x0x0>
    %conv_final_out = migraphx.add %conv_final, %conv_final_bias_bcast : <1x16x1080x1920xf16, 33177600x2073600x1920x1>, <1x16x1080x1920xf16, 0x1x0x0> -> <1x16x1080x1920xf16, 33177600x2073600x1920x1>
    %d2s_reshape = migraphx.reshape %conv_final_out {dims = [1, 4, 2, 2, 1080, 1920]} : <1x16x1080x1920xf16, 33177600x2073600x1920x1> -> <1x4x2x2x1080x1920xf16, 33177600x8294400x4147200x2073600x1920x1>
    %d2s_transpose = migraphx.transpose %d2s_reshape {permutation = [0, 1, 4, 2, 5, 3]} : <1x4x2x2x1080x1920xf16, 33177600x8294400x4147200x2073600x1920x1> -> <1x4x1080x2x1920x2xf16, 33177600x8294400x7680x3840x2x1>
    %output = migraphx.reshape %d2s_transpose {dims = [1, 4, 2160, 3840]} : <1x4x1080x2x1920x2xf16, 33177600x8294400x7680x3840x2x1> -> <1x4x2160x3840xf16, 33177600x8294400x3840x1>
    return %output : !migraphx.shaped<1x4x2160x3840xf16, 33177600x8294400x3840x1>
  }
}
