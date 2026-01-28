// Convolution Example with MIGraphX Dialect
// This example demonstrates:
// - Convolution operation (migraphx.convolution)
// - Batch normalization (migraphx.batch_norm_inference)
// - Broadcast and element-wise operations
// - Pooling operation (migraphx.pooling)
//
// To run this example:
// 1. From the build directory:
//    ./bin/rocmlir-driver ../examples/migraphx-dialect/migraphx_convolution_example.mlir --kernel-pipeline=gpu --arch=gfx1150
//
// 2. View intermediate IR after GPU pipeline:
//    ./bin/rocmlir-driver ../examples/migraphx-dialect/migraphx_convolution_example.mlir --kernel-pipeline=gpu --arch=gfx1150 | ./bin/rocmlir-opt

module {
  // Convolution + Batch Norm + ReLU pattern (common in CNNs)
  func.func @conv_bn_relu(
    %input: !migraphx.shaped<1x3x32x32xf32, 3072x1024x32x1>,    // Input: NCHW format, batch=1, channels=3, 32x32
    %filter: !migraphx.shaped<64x3x3x3xf32, 27x9x3x1>,          // Filter: 64 output channels, 3x3 kernel
    %bn_scale: !migraphx.shaped<64xf32, 1>,                      // BN scale parameter
    %bn_bias: !migraphx.shaped<64xf32, 1>,                       // BN bias parameter
    %bn_mean: !migraphx.shaped<64xf32, 1>,                       // BN running mean
    %bn_variance: !migraphx.shaped<64xf32, 1>                    // BN running variance
  ) -> !migraphx.shaped<1x64x15x15xf32, 14400x225x15x1>
    attributes {kernel = "mixr", arch = "gfx1150"} {
    
    // Step 1: 2D Convolution
    // stride=2, padding=1, dilation=1, group=1
    %conv = migraphx.convolution %input, %filter {
      padding = [1, 1, 1, 1],      // [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end]
      stride = [2, 2],              // [stride_h, stride_w]
      dilation = [1, 1],            // [dilation_h, dilation_w]
      group = 1 : i64,              // Number of groups
      padding_mode = 0 : i64        // 0 = constant padding
    } : <1x3x32x32xf32, 3072x1024x32x1>, <64x3x3x3xf32, 27x9x3x1> 
      -> <1x64x16x16xf32, 16384x256x16x1>
    
    // Step 2: Batch Normalization (inference mode)
    // Formula: y = scale * (x - mean) / sqrt(variance + epsilon) + bias
    %bn = migraphx.batch_norm_inference %conv, %bn_scale, %bn_bias, %bn_mean, %bn_variance {
      epsilon = 1.0e-05 : f32,
      momentum = 0.9 : f32,
      bn_mode = 1 : i64              // 1 = spatial mode (per-channel)
    } : !migraphx.shaped<1x64x16x16xf32, 16384x256x16x1>, !migraphx.shaped<64xf32, 1>, !migraphx.shaped<64xf32, 1>, !migraphx.shaped<64xf32, 1>, !migraphx.shaped<64xf32, 1> -> !migraphx.shaped<1x64x16x16xf32, 16384x256x16x1>
    
    // Step 3: ReLU activation
    %relu = migraphx.relu %bn : 
      <1x64x16x16xf32, 16384x256x16x1> -> <1x64x16x16xf32, 16384x256x16x1>
    
    // Step 4: Max Pooling (2x2 kernel, stride=1)
    %pool = migraphx.pooling %relu {
      mode = "max",                  // Max pooling
      padding = [0, 0, 0, 0],       // No padding
      stride = [1, 1],              // Stride
      length = [2, 2],              // Pooling window size
      ceil_mode = 0 : i64           // 0 = floor mode
    } : <1x64x16x16xf32, 16384x256x16x1> -> <1x64x15x15xf32, 14400x225x15x1>
    
    return %pool : !migraphx.shaped<1x64x15x15xf32, 14400x225x15x1>
  }
}
